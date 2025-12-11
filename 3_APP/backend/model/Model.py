"""Esta fue la arquitectura utilizada para M5 y M6"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer



class LocHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, x):
        return self.net(x)
class HybridEmbedder(nn.Module):
    """
    SentenceTransformer + LocHead (Fourier features) + projection.
    """
    def __init__(
        self,
        base_model: SentenceTransformer,
        fourier_dim: int,
        proj_dim: int = 256,
        loc_out_dim: int = 32,
    ):
        super().__init__()
        self.base_model = base_model
        self.loc_head = LocHead(in_dim=fourier_dim, out_dim=loc_out_dim)

        emb_dim = self.base_model.get_sentence_embedding_dimension()
        in_dim = emb_dim + loc_out_dim

        self.out = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.GELU(),
            nn.LayerNorm(proj_dim),
            nn.Dropout(0.1),
        )

    def forward(self, features, fourier):
        # features: dict from base_model.tokenize
        # fourier: [B, fourier_dim]
        outputs = self.base_model(features)
        sent_emb = outputs["sentence_embedding"]          # [B, emb_dim]

        loc = self.loc_head(fourier)                      # [B, loc_out_dim]
        combined = torch.cat([sent_emb, loc], dim=-1)     # [B, emb_dim + loc_out_dim]

        proj = self.out(combined)                         # [B, proj_dim]
        proj = F.normalize(proj, p=2, dim=-1)             # L2 normalization
        return proj