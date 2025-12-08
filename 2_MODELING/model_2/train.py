import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from torch.utils.data import  DataLoader
from sklearn.metrics import roc_auc_score

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
    scaler: Optional[torch.cuda.amp.GradScaler] = None,  # <-- para AMP opcional
) -> Dict[str, Any]:
    model.train()
    total_loss, n = 0.0, 0

    num_steps = len(loader)
    print(f"Total steps this epoch: {num_steps}")

    # logear aprox 10 veces por epoch
    log_every = 100

    optimizer.zero_grad(set_to_none=True)

    # contexto de autocast si hay GPU + scaler
    use_amp = (scaler is not None) and (device.type == "cuda")
    if not use_amp:
        raise Exception("you need a GPU")
    autocast_ctx = torch.cuda.amp.autocast 

    for step, batch in enumerate(loader, start=1):
        # mover batch al device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking=True)

        with autocast_ctx():
            z_job, z_cand, scale = model(batch)
            loss = pairwise_bce_loss(
                z_job,
                z_cand,
                batch["label"],
                scale,
                pos_weight=pos_weight,
            )

        # grad accumulation
        loss_to_backprop = loss / max(1, grad_accum_steps)

        if use_amp:
            scaler.scale(loss_to_backprop).backward()
        else:
            loss_to_backprop.backward()

        # ¿toca hacer step?
        is_update_step = (step % grad_accum_steps == 0) or (step == num_steps)

        if is_update_step:
            if grad_clip is not None:
                if use_amp:
                    # para que el clip vea los gradientes reales
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

        bs = batch["label"].size(0)
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
from sklearn.metrics import roc_auc_score


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

    return metrics


