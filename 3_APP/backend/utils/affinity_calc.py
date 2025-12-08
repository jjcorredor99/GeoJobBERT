import os
from typing import Any, List, Optional

import numpy as np
import torch
from transformers import AutoTokenizer

from model.siames import SiameseTwoTower
from backend.utils.chunker import chunker
from backend.utils.encode_latlon import GeoFourierEncoder
from models_db import VacancyDB, CandidateDB

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "nn.pt")
GEO_B_PATH = os.path.join(BASE_DIR, "geo_B.npy")  # same folder as encode_latlon/geo_B.npy

MAX_LENGTH = 384
FOURIER_D = 8
FOURIER_SCALES = (1200, 400, 150, 50)
FOURIER_SEED = 0

_model: Optional[torch.nn.Module] = None
_tokenizer: Optional[AutoTokenizer] = None
_geo_encoder: Optional[GeoFourierEncoder] = None


# ---------- small helpers ----------

def _safe_get(obj: Any, attr: str, default=None):
    if hasattr(obj, attr):
        return getattr(obj, attr)
    if isinstance(obj, dict):
        return obj.get(attr, default)
    try:
        return obj[attr]
    except Exception:
        return default


def coalesce_list(val: Any) -> List[str]:
    def _is_nan(x: Any) -> bool:
        return isinstance(x, float) and np.isnan(x)

    if isinstance(val, (list, tuple, set)):
        out = []
        for x in val:
            if x is None or _is_nan(x):
                continue
            s = str(x).strip()
            if s:
                out.append(s)
        return out

    if val is None or _is_nan(val):
        return []

    s = str(val).strip()
    return [s] if s else []


def format_section(label: str, values: Any) -> str:
    v = coalesce_list(values)
    return f"{label}: " + ", ".join(v) if v else ""


# ---------- text builders ----------

def build_text_candidate(candidate: CandidateDB) -> str:
    parts: List[str] = []

    desc = _safe_get(candidate, "candidate_description", "")
    if desc:
        parts.append(str(desc))

    parts.append(format_section("experiencia", _safe_get(candidate, "experience_descriptions", [])))
    parts.append(format_section("habilidades", _safe_get(candidate, "skill_names", [])))
    parts.append(format_section("sectores", _safe_get(candidate, "sector_names", [])))

    salary = _safe_get(candidate, "candidate_salary", None)
    if salary is not None:
        parts.append(f"salario: {salary}")

    return " ".join(p for p in parts if p)


def build_text_vacancy(vacancy: VacancyDB) -> str:
    parts: List[str] = []

    base_desc = (
        _safe_get(vacancy, "vacancy_description", None)
        or _safe_get(vacancy, "description", None)
        or ""
    )
    if base_desc:
        parts.append(str(base_desc))

    parts.append(format_section("habilidades", _safe_get(vacancy, "skill_names", [])))
    parts.append(format_section("sectores", _safe_get(vacancy, "sector_names", [])))

    min_salary = _safe_get(vacancy, "min_salary", None)
    if min_salary is not None:
        parts.append(f"salario: {min_salary}")

    return " ".join(p for p in parts if p)


# ---------- geo encoder ----------

def _get_geo_encoder() -> GeoFourierEncoder:
    global _geo_encoder
    if _geo_encoder is None:
        enc = GeoFourierEncoder(
            D=FOURIER_D,
            scales_km=FOURIER_SCALES,
            seed=FOURIER_SEED,
        )
        if os.path.exists(GEO_B_PATH):
            enc.B = np.load(GEO_B_PATH)
        _geo_encoder = enc
    return _geo_encoder


def build_fourier(obj: VacancyDB | CandidateDB) -> torch.Tensor:
    lat = _safe_get(obj, "lat", None)
    lon = _safe_get(obj, "lon", None)
    if lat is None or lon is None:
        raise ValueError("lat/lon must be present in CandidateDB/VacancyDB")

    enc = _get_geo_encoder()
    feats = enc.transform([[lat, lon]])  # (1, D)
    return torch.from_numpy(feats).float()  # [1, D]


# ---------- tokenizer + model ----------

def _get_tokenizer() -> AutoTokenizer:
    global _tokenizer
    if _tokenizer is None:
        model_name = os.getenv("model_name") or os.getenv("MODEL_NAME")
        if not model_name:
            raise RuntimeError("Env var 'model_name' (or 'MODEL_NAME') must be set")
        _tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    return _tokenizer


def load_model() -> SiameseTwoTower:
    global _model
    if _model is not None:
        return _model

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Modelo nn.pt no encontrado en {MODEL_PATH}")

    m = torch.load(MODEL_PATH, map_location=DEVICE)
    if not isinstance(m, torch.nn.Module):
        raise RuntimeError("nn.pt debe contener el modelo completo (nn.Module).")

    m.to(DEVICE)
    m.eval()
    _model = m
    return _model


# ---------- chunks -> [B, C, T] tensors ----------

def _chunks_to_tensor_from_chunker(
    chunks_dict: dict,
    name: str,
    tokenizer: AutoTokenizer,
    max_length: int = MAX_LENGTH,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    chunker(...) returns lists of ids per chunk under keys:
        f"{name}_chunks_input_ids", f"{name}_chunks_attention_mask"
    We turn them into [1, C, T] tensors (B=1).
    """
    ids_list = chunks_dict.get(f"{name}_chunks_input_ids", [])
    mask_list = chunks_dict.get(f"{name}_chunks_attention_mask", [])

    # no text -> single empty chunk
    if not ids_list:
        enc = tokenizer(
            "",
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return enc["input_ids"].unsqueeze(1), enc["attention_mask"].unsqueeze(1)

    pad_id = tokenizer.pad_token_id or 0

    C = len(ids_list)
    T = min(
        max(len(ch) for ch in ids_list),
        max_length,
    )

    input_ids = torch.full((1, C, T), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((1, C, T), dtype=torch.long)

    for i, ids in enumerate(ids_list):
        l = min(len(ids), T)
        input_ids[0, i, :l] = torch.tensor(ids[:l], dtype=torch.long)
        attention_mask[0, i, :l] = 1

    return input_ids, attention_mask


# ---------- main API ----------

def compute_affinity(candidate: CandidateDB, vacancy: VacancyDB) -> float:
    """
    Compute P(match | candidate, vacancy) in [0,1]
    using the trained SiameseTwoTower + TextTower + geo features.
    """
    model = load_model()
    tokenizer = _get_tokenizer()

    # 1) build texts
    job_text = build_text_vacancy(vacancy)
    cand_text = build_text_candidate(candidate)

    # 2) chunk them
    chunks = chunker(
        {"job": job_text, "cand": cand_text},
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
    )

    job_input_ids, job_attention_mask = _chunks_to_tensor_from_chunker(
        chunks, "job", tokenizer, MAX_LENGTH
    )
    cand_input_ids, cand_attention_mask = _chunks_to_tensor_from_chunker(
        chunks, "cand", tokenizer, MAX_LENGTH
    )

    # 3) geo features + remote flag
    vac_fou = build_fourier(vacancy)      # [1, D]
    cand_fou = build_fourier(candidate)   # [1, D]

    remote_val = _safe_get(vacancy, "remote", 0)
    vac_remote = torch.tensor(
        [1.0 if remote_val else 0.0], dtype=torch.float32
    )  # [1]

    # 4) batch for SiameseTwoTower
    batch = {
        "job_input_ids": job_input_ids.to(DEVICE),            # [1, C_job, T]
        "job_attention_mask": job_attention_mask.to(DEVICE),  # [1, C_job, T]
        "cand_input_ids": cand_input_ids.to(DEVICE),          # [1, C_cand, T]
        "cand_attention_mask": cand_attention_mask.to(DEVICE),
        "vac_loc_fourier": vac_fou.to(DEVICE),                # [1, D]
        "cand_loc_fourier": cand_fou.to(DEVICE),              # [1, D]
        "vacant_remote": vac_remote.to(DEVICE),               # [1]
    }

    # 5) forward + sigmoid
    with torch.no_grad():
        z_job, z_cand, logit_scale = model(batch)
        # optional safety clamp in case InfoNCE pushed scale very high
        if isinstance(logit_scale, torch.Tensor):
            logit_scale = logit_scale.clamp(max=100.0)
        logits = (z_job * z_cand).sum(dim=-1) * logit_scale
        prob = torch.sigmoid(logits)[0].item()

    return float(prob)
