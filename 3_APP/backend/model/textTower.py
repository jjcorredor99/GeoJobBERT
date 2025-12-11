import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class TextTower(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        proj_dim: int = 256,
        max_length: int = 384,
        chunks_subbatch: int = 20
    ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(encoder_name)
        # You actually don't use the tokenizer in forward; you can drop this if not needed
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name, use_fast=True)
        self.max_length = max_length


        self.chunks_subbatch = chunks_subbatch

        H = self.encoder.config.hidden_size
        self.proj = nn.Sequential(
            nn.LayerNorm(H),
            nn.Dropout(0.1),
            nn.Linear(H, proj_dim),
            nn.GELU(),
        )

    # ---- encoding ----
    def encode_chunks(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        B, C, T = input_ids.shape
        flat_ids  = input_ids.view(B * C, T)
        flat_mask = attention_mask.view(B * C, T).to(dtype=torch.bool)

        pooled_list = []
        N = flat_ids.size(0)


        out = self.encoder(
            input_ids=flat_ids,
            attention_mask=flat_mask,
            return_dict=True,
        )
        token_embs = out.last_hidden_state        # [n, T, H]
        pooled_sub = mean_pool_tokens(token_embs, flat_mask)  # [n, H]
        pooled_list.append(pooled_sub)

        pooled = torch.cat(pooled_list, dim=0).view(B, C, -1)  # [B, C, H]
        pooled = mean_pool_chunks(pooled)                      # [B, H]
        return pooled

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        pooled = self.encode_chunks(input_ids, attention_mask)  # [B, H]
        return self.proj(pooled)                                # [B, D]


# utils

def mean_pool_tokens(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()      # [B,T] -> [B,T,1]
    summed = (last_hidden_state * mask).sum(dim=1)   # [B,H]
    counts = mask.sum(dim=1).clamp(min=1e-9)         # [B,1]
    return summed / counts

def mean_pool_chunks(chunk_embs: torch.Tensor) -> torch.Tensor:
    return chunk_embs.mean(dim=1)                    # [B,C,H] -> [B,H]
