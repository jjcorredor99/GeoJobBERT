import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from torch.utils.data import  DataLoader
from sklearn.metrics import roc_auc_score, f1_score

def pairwise_bce_loss(z_job: torch.Tensor, z_cand: torch.Tensor,
                      y: torch.Tensor, logit_scale: torch.Tensor,
                      pos_weight: Optional[float] = None) -> torch.Tensor:
    """
    z_job, z_cand: [B, D] (L2-normalizados en TwoTower)
    y: [B] float {0,1}
    logit_scale: escalar > 0 (learnable)
    pos_weight: opcional para desbalance (N_neg/N_pos)
    """
    logits = (z_job * z_cand).sum(dim=-1) * logit_scale
    pw = torch.tensor(pos_weight, device=logits.device, dtype=logits.dtype) if pos_weight else None
    return F.binary_cross_entropy_with_logits(logits, y, pos_weight=pw)

    
import contextlib
from typing import Optional, Dict, Any
import torch
import torch.nn as nn


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    pos_weight: Optional[float] = None,
    grad_clip: Optional[float] = 1.0,
    grad_accum_steps: int = 1,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,  # AMP
    loss_type: str = "bce",   # "bce" o "info_nce"
) -> Dict[str, Any]:
    model.train()
    total_loss, n = 0.0, 0

    num_steps = len(loader)
    print(f"Total steps this epoch: {num_steps}")

    log_every = 100
    optimizer.zero_grad(set_to_none=True)

    use_amp = (scaler is not None) and (device.type == "cuda")
    if not use_amp:
        raise Exception("you need a GPU")
    autocast_ctx = torch.cuda.amp.autocast

    for step, batch in enumerate(loader, start=1):
        # mover al device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking=True)

        with autocast_ctx():
            z_job, z_cand, scale = model(batch)

            if loss_type == "bce":
                loss = pairwise_bce_loss(
                    z_job,
                    z_cand,
                    batch["label"],
                    scale,
                    pos_weight=pos_weight,
                )
            elif loss_type == "info_nce":
                loss = inbatch_multi_entity_info_nce(
                    z_job,
                    z_cand,
                    scale,
                    job_ids=batch["vacant_id"],    # ajusta nombres si hace falta
                    cand_ids=batch["candidate_id"],
                    labels=batch["label"],
                )
            else:
                raise ValueError(f"Unknown loss_type: {loss_type}")

        # grad accumulation
        loss_to_backprop = loss / max(1, grad_accum_steps)

        if use_amp:
            scaler.scale(loss_to_backprop).backward()
        else:
            loss_to_backprop.backward()

        is_update_step = (step % grad_accum_steps == 0) or (step == num_steps)

        if is_update_step:
            if grad_clip is not None:
                if use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

        # logging
        bs = z_job.size(0)
        total_loss += loss.item() * bs
        n += bs

        if step % log_every == 0 or step == num_steps:
            pct = 100.0 * step / num_steps
            print(f"Step {step}/{num_steps} ({pct:.1f}%), batch_loss={loss.item():.4f}")

    return {"loss": total_loss / max(n, 1)}

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from typing import Dict


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> Dict[str, float]:
    model.eval()
    total, correct = 0, 0
    total_loss = 0.0

    collect_y, collect_p = [], []

    use_amp = (device.type == "cuda")

    for batch in loader:
        # mover batch al device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking=True)

        # forward en mixed precision si hay GPU
        with autocast(enabled=use_amp):
            z_job, z_cand, scale = model(batch)
            logits = (z_job * z_cand).sum(dim=-1) * scale
            loss = F.binary_cross_entropy_with_logits(
                logits,
                batch["label"].float(),   # asegúrate de que sea float
            )
            probs = torch.sigmoid(logits)

        preds = (probs >= threshold).long()
        y = batch["label"].long()

        correct += (preds == y).sum().item()
        n_batch = y.numel()
        total += n_batch
        total_loss += loss.item() * n_batch

        collect_y.append(batch["label"].detach().cpu())
        collect_p.append(probs.detach().cpu())

    metrics = {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
    }

    y_all = torch.cat(collect_y).numpy()
    p_all = torch.cat(collect_p).numpy()
    metrics["roc_auc"] = float(roc_auc_score(y_all, p_all))

    pred_all = (p_all >= threshold).astype("int32")
    metrics["f1"] = float(f1_score(y_all, pred_all))

    return metrics




import torch
import torch.nn.functional as F

def multi_pos_infonce_from_mask(sim: torch.Tensor,
                                pos_mask: torch.Tensor) -> torch.Tensor:
    """
    sim: [N_anchor, N_other]
    pos_mask: [N_anchor, N_other] bool, True si ese par es positivo.

    Para cada anchor i, maximizamos la probabilidad total de todos sus positivos:
        L_i = -log( sum_p exp(sim_{i,p}) / sum_k exp(sim_{i,k}) )
    """
    device = sim.device
    # anchors que tienen al menos un positivo
    valid = pos_mask.any(dim=1)
    if not valid.any():
        return torch.tensor(0.0, device=device, dtype=sim.dtype)

    sim_valid = sim[valid]            # [Nv, M]
    pos_mask_valid = pos_mask[valid]  # [Nv, M]

    # denominador: log sum_k exp(sim)
    logsumexp_all = torch.logsumexp(sim_valid, dim=1)  # [Nv]

    # numerador: log sum_p exp(sim) sólo en positivos
    sim_pos = sim_valid.masked_fill(~pos_mask_valid, float("-inf"))
    logsumexp_pos = torch.logsumexp(sim_pos, dim=1)    # [Nv]

    loss = -(logsumexp_pos - logsumexp_all).mean()
    return loss


def inbatch_multi_entity_info_nce(
    z_job: torch.Tensor,
    z_cand: torch.Tensor,
    logit_scale: torch.Tensor,
    job_ids: torch.Tensor,
    cand_ids: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    InfoNCE con múltiples positivos por vacante/candidato.

    z_job, z_cand: [B, D] (ya normalizados por SiameseTwoTower)
    job_ids: [B]  (id de vacante)
    cand_ids: [B] (id de candidato)
    labels: [B] {0,1}
    """
    device = z_job.device
    labels = labels.bool()

    # Mapear a índices únicos de entidad
    unique_jobs, job_idx = torch.unique(job_ids, return_inverse=True)    # [J], [B]
    unique_cands, cand_idx = torch.unique(cand_ids, return_inverse=True) # [C], [B]
    J = unique_jobs.size(0)
    C = unique_cands.size(0)
    D = z_job.size(1)

    # Agregar embeddings por entidad (media de las filas que la contienen)
    z_job_agg  = torch.zeros(J, D, device=device)
    z_cand_agg = torch.zeros(C, D, device=device)

    z_job_agg.index_add_(0, job_idx, z_job)
    z_cand_agg.index_add_(0, cand_idx, z_cand)

    job_counts  = torch.bincount(job_idx, minlength=J).unsqueeze(-1).clamp_min(1)
    cand_counts = torch.bincount(cand_idx, minlength=C).unsqueeze(-1).clamp_min(1)

    z_job_agg  = z_job_agg  / job_counts
    z_cand_agg = z_cand_agg / cand_counts

    # renormalizar
    z_job_agg  = F.normalize(z_job_agg,  p=2, dim=-1)
    z_cand_agg = F.normalize(z_cand_agg, p=2, dim=-1)

    # Máscara de positivos candidato->vacante
    pos_mask_cj = torch.zeros(C, J, dtype=torch.bool, device=device)
    pos_mask_cj[cand_idx[labels], job_idx[labels]] = True  # sólo filas con label==1

    # Similitudes candidato->vacante
    sim_cj = logit_scale * (z_cand_agg @ z_job_agg.t())  # [C, J]
    loss_cand = multi_pos_infonce_from_mask(sim_cj, pos_mask_cj)

    # Simetría: vacante->candidato
    pos_mask_jc = pos_mask_cj.t()  # [J, C]
    sim_jc = sim_cj.t()            # [J, C]
    loss_job = multi_pos_infonce_from_mask(sim_jc, pos_mask_jc)

    return 0.5 * (loss_cand + loss_job)
