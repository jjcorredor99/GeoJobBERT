import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.locHead import LocHead
from model.textTower import TextTower
from typing import Dict, Any, List


class SiameseTwoTower(nn.Module):
    def __init__(
        self,
        job_loc_in: List[int],
        cand_loc_in: List[int],
        encoder_name: str = os.getenv("model_name"),
        proj_dim: int = 256,
        max_length: int = 384,
    ):
        super().__init__()

        # un solo encoder
        self.text = TextTower(encoder_name, proj_dim, max_length)

        self.job_loc = LocHead(job_loc_in + 1, 32)
        self.cand_loc = LocHead(cand_loc_in, 32)

        in_dim = proj_dim + 32  # text + loc

        self.out = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.GELU(),
            nn.LayerNorm(proj_dim),
            nn.Dropout(0.1),
        )

        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def forward(self, batch: Dict[str, Any]):
        device = next(self.parameters()).device

        t_job = self.text(
            batch["job_input_ids"].to(device),
            batch["job_attention_mask"].to(device),
        )  # [B, D]

        t_cand = self.text(
            batch["cand_input_ids"].to(device),
            batch["cand_attention_mask"].to(device),
        )  # [B, D]


        job_loc  = batch["vac_loc_fourier"].to(device)      # [B, L_job]
        cand_loc = batch["cand_loc_fourier"].to(device)     # [B, L_cand]

        vac_remote = batch["vacant_remote"].to(device).float()  # [B] o [B,1]
        if vac_remote.dim() == 1:
            vac_remote = vac_remote.unsqueeze(-1)           # [B,1]

        # Concatenamos el flag en la vacante
        job_loc_in  = torch.cat([job_loc, vac_remote], dim=-1)    # [B, L_job+1]
        cand_loc_in = cand_loc                                    # [B, L_cand]

        job_loc_emb  = self.job_loc(job_loc_in)          # [B, 32]
        cand_loc_emb = self.cand_loc(cand_loc_in)        # [B, 32]

        # ----- fusión texto + loc -----
        h_job  = torch.cat([t_job,  job_loc_emb],  dim=-1)
        h_cand = torch.cat([t_cand, cand_loc_emb], dim=-1)
        # proyección en el mismo espacio
        z_job = F.normalize(self.out(h_job), p=2, dim=-1)
        z_cand = F.normalize(self.out(h_cand), p=2, dim=-1)

        return z_job, z_cand, self.logit_scale.exp()
