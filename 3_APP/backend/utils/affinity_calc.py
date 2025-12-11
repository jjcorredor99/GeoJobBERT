import os
from typing import Any, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from utils.encode_latlon import GeoFourierEncoder
from models_db import VacancyDB, CandidateDB
from model.Model import HybridEmbedder  # <- your hybrid model

from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # backend
MODEL_NAME = "hybrid_lochead_final.pt"
MODEL_PATH = os.path.join(BASE_DIR, "model", MODEL_NAME)
# Cargar variables de entorno desde 3_APP/.env.prod (si existe)

load_dotenv(os.path.join(BASE_DIR, ".env.prod"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Matriz B de Fourier (si la usas)
GEO_B_PATH = os.path.join(BASE_DIR, "geo_B.npy")  # mismo folder que encode_latlon

# Defaults; FOURIER_D will be overwritten from checkpoint if present
FOURIER_D = 8
FOURIER_SCALES = (1200, 400, 150, 50)
FOURIER_SEED = 0

_hybrid_model: Optional[HybridEmbedder] = None
_base_model: Optional[SentenceTransformer] = None
_geo_encoder: Optional[GeoFourierEncoder] = None
_fourier_dim: Optional[int] = None


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


def build_text_vacancy(vacancy: VacancyDB | Any) -> str:
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

    min_salary = _safe_get(vacancy, "min_salary", None) or _safe_get(vacancy, "salary", None)
    if min_salary is not None:
        parts.append(f"salario: {min_salary}")

    return " ".join(p for p in parts if p)


# ---------- geo encoder ----------

def _get_geo_encoder() -> GeoFourierEncoder:
    """
    Lazily create the GeoFourierEncoder with the same dimensionality
    as the model was trained on (from checkpoint config).
    """
    global _geo_encoder, _fourier_dim

    if _geo_encoder is None:
        # ensure _fourier_dim is set from checkpoint
        if _fourier_dim is None:
            _ = load_model()  # sets _fourier_dim

        D = int(_fourier_dim or FOURIER_D)
        enc = GeoFourierEncoder(
            D=D,
            scales_km=FOURIER_SCALES,
            seed=FOURIER_SEED,
        )
        if os.path.exists(GEO_B_PATH):
            enc.B = np.load(GEO_B_PATH)
        _geo_encoder = enc
    return _geo_encoder


def build_fourier(obj: VacancyDB | CandidateDB | Any) -> torch.Tensor:
    lat = _safe_get(obj, "lat", None)
    lon = _safe_get(obj, "lon", None)
    if lat is None or lon is None:
        raise ValueError("lat/lon must be present in CandidateDB/VacancyDB")

    enc = _get_geo_encoder()
    feats = enc.transform([[lat, lon]])  # (1, D)
    return torch.from_numpy(feats).float()  # [1, D]


def build_fourier_from_latlon(lat: float, lon: float) -> torch.Tensor:
    """
    Convenience helper used when we only have raw coordinates instead of
    a full VacancyDB/CandidateDB object.
    """
    enc = _get_geo_encoder()
    feats = enc.transform([[lat, lon]])  # (1, D)
    return torch.from_numpy(feats).float()  # [1, D]


# ---------- Hybrid model loading ----------

def load_model() -> HybridEmbedder:
    """
    Load the HybridEmbedder + base SentenceTransformer from the
    hybrid_lochead_final.pt checkpoint.
    """
    global _hybrid_model, _base_model, _fourier_dim

    if _hybrid_model is not None:
        return _hybrid_model

    if not os.path.exists(MODEL_PATH):
        print(MODEL_PATH, "MODELOOOOO")
        raise RuntimeError(f"Modelo hybrid_lochead_final.pt no encontrado en {MODEL_PATH}")

    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)

    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        print(MODEL_PATH, "ACÁAAA DICE")
        raise RuntimeError(
            "El archivo de modelo debe ser un checkpoint dict con 'model_state_dict'. "
            "Asegúrate de guardar con torch.save({...}) como en el notebook."
        )

    cfg = ckpt.get("model_config", {}) or {}

    # fourier_dim saved during training
    _fourier_dim = int(cfg.get("fourier_dim", 0)) or FOURIER_D

    # --- IMPORTANT PART ---
    # Base model used in training (e.g. "/content/.../base-best_3/checkpoint-76104")
    ckpt_base = cfg.get("base_model_path")

    # Optional override if you moved that directory in this environment
    env_override = os.getenv("HYBRID_BASE_MODEL_PATH", "C:/Users/JuanJoseCorredor/OneDrive - PSYCONOMETRICS SAS/Documentos/uniandes/Tesis 2/3_APP/backend/model/checkpoint-76104")
    print(env_override)
    if env_override:
        base_model_path = env_override
    elif ckpt_base and os.path.exists(ckpt_base):
        base_model_path = ckpt_base
    else:
        raise RuntimeError(
            "No se pudo determinar la ruta del modelo base para HybridEmbedder.\n"
            " - En el checkpoint, 'base_model_path' es: "
            f"{ckpt_base}\n"
            " - Define la variable de entorno HYBRID_BASE_MODEL_PATH apuntando al "
            "directorio local de ese SentenceTransformer (el mismo que usaste al entrenar)."
        )

    _base_model = SentenceTransformer(base_model_path, device=str(DEVICE))

    proj_dim = int(cfg.get("proj_dim", 256))
    loc_out_dim = int(cfg.get("loc_out_dim", 32))

    model = HybridEmbedder(
        base_model=_base_model,
        fourier_dim=_fourier_dim,
        proj_dim=proj_dim,
        loc_out_dim=loc_out_dim,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    # freeze encoder, just in case
    for p in model.base_model.parameters():
        p.requires_grad = False

    _hybrid_model = model
    return _hybrid_model


def _encode_with_fourier(text: str, loc_fourier: torch.Tensor) -> torch.Tensor:
    """
    Encode a single (text, loc_fourier) pair into a L2-normalized embedding [D_emb].
    text: str
    loc_fourier: [1, F]
    """
    if not text or not text.strip():
        raise ValueError("El texto para el encoder no puede estar vacío")

    model = load_model()
    assert _base_model is not None

    # Tokenize with the SentenceTransformer base model
    features = _base_model.tokenize([text])
    features = {k: v.to(DEVICE) for k, v in features.items()}

    loc_fourier = loc_fourier.to(DEVICE).float()
    if loc_fourier.dim() == 1:
        loc_fourier = loc_fourier.unsqueeze(0)  # [1, F]

    with torch.no_grad():
        embs = model(features, loc_fourier)  # [1, D]
        embs = F.normalize(embs, p=2, dim=-1)

    return embs[0]  # [D]


# ---------- main affinity function (used by /apply & test_model_only) ----------

def compute_affinity(
    vacancy: Any,
    cv_text: str,
    candidate_lat: Optional[float] = None,
    candidate_lon: Optional[float] = None,
) -> float:
    """
    Compute an affinity in [0,1] between a vacancy and a candidate CV text,
    using the HybridEmbedder (SentenceTransformer + Fourier geo features).
    """
    cv_text = (cv_text or "").strip()
    if not cv_text:
        return 0.0

    # 1) Build textual representations
    job_text = build_text_vacancy(vacancy)

    # 2) Geo Fourier features
    vac_four = build_fourier(vacancy)  # [1, F]

    if candidate_lat is not None and candidate_lon is not None:
        cand_four = build_fourier_from_latlon(candidate_lat, candidate_lon)  # [1, F]
    else:
        # If we don't know candidate location, fall back to zeros so that
        # location does not contribute to the similarity.
        cand_four = torch.zeros_like(vac_four)

    # 3) Encode both sides with the hybrid model
    job_emb = _encode_with_fourier(job_text, vac_four)   # [D]
    cand_emb = _encode_with_fourier(cv_text, cand_four)  # [D]

    # 4) Cosine similarity in [-1, 1] -> affinity in [0, 1]
    sim = F.cosine_similarity(job_emb, cand_emb, dim=0).item()
    affinity = (sim + 1.0) / 2.0

    return float(affinity)
