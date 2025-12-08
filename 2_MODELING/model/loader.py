import torch
from torch.utils.data import  Dataset
from typing import Dict, Any, List, Optional

# create text tensor
def _pad_3d(list_of_2d: List[List[List[int]]], pad_val: int = 0):
    B = len(list_of_2d)
    C_max = max(len(x) for x in list_of_2d) if B > 0 else 0
    T_max = max((len(t) for x in list_of_2d for t in x), default=0)
    out = torch.full((B, C_max, T_max), pad_val, dtype=torch.long)
    mask = torch.zeros((B, C_max, T_max), dtype=torch.long)
    for i, mat in enumerate(list_of_2d):
        for c, seq in enumerate(mat):
            L = len(seq)
            out[i, c, :L] = torch.tensor(seq, dtype=torch.long)
            mask[i, c, :L] = 1
    return out, mask



#crea tensores 


def make_collate_fn_chunks(pad_id: int = 0):
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # gather ragged lists
        job_ids   = [b["job_chunks_input_ids"]      for b in batch]
        job_mask  = [b["job_chunks_attention_mask"] for b in batch]
        cand_ids  = [b["cand_chunks_input_ids"]     for b in batch]
        cand_mask = [b["cand_chunks_attention_mask"]for b in batch]

        job_input_ids,  job_attention_mask  = _pad_3d(job_ids,   pad_val=pad_id)
        cand_input_ids, cand_attention_mask = _pad_3d(cand_ids,  pad_val=pad_id)


        MAX_JOB_CHUNKS = 4   
        MAX_CAND_CHUNKS = 4 
        job_input_ids      = job_input_ids[:, :MAX_JOB_CHUNKS]
        job_attention_mask = job_attention_mask[:, :MAX_JOB_CHUNKS]

        cand_input_ids      = cand_input_ids[:, :MAX_CAND_CHUNKS]
        cand_attention_mask = cand_attention_mask[:, :MAX_CAND_CHUNKS]


        out = {
            "job_input_ids":  job_input_ids,          # [B, Cmax, Tmax] long
            "job_attention_mask": job_attention_mask, # [B, Cmax, Tmax] long
            "cand_input_ids": cand_input_ids,         # [B, Cmax, Tmax] long
            "cand_attention_mask": cand_attention_mask,
            "label": torch.tensor([b["label"] for b in batch], dtype=torch.float32),
            "vacant_remote": torch.tensor([b["vacant_remote"] for b in batch], dtype=torch.float32),

        }

        out["vac_loc_fourier"] = torch.tensor([b["vac_loc_fourier"] for b in batch], dtype=torch.float32)
        out["cand_loc_fourier"] = torch.tensor([b["cand_loc_fourier"] for b in batch], dtype=torch.float32)
        out["vacant_id"] = torch.tensor(
            [b["vacant_id"] for b in batch], dtype=torch.long
        )
        out["candidate_id"] = torch.tensor(
            [b["candidate_id"] for b in batch], dtype=torch.long
        )


        return out
    return collate_fn



#ok

class PairDatasetPNChunked(Dataset):
    """
    Expects each row to have:
      - 'job_chunks_input_ids'           : List[List[int]]
      - 'job_chunks_attention_mask'      : List[List[int]]
      - 'cand_chunks_input_ids'          : List[List[int]]
      - 'cand_chunks_attention_mask'     : List[List[int]]
      - 'label'                          : 0/1
      - optional 'vacant_fourier_feature', 'candidate_fourier_features'
      - optional 'vacant_remote'         : bool or 0/1
      - (optional) ids: 'vacant_id', 'candidate_id'
    """
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self): 
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        item = {
            # precomputed chunks
            "job_chunks_input_ids":       row["job_chunks_input_ids"],
            "job_chunks_attention_mask":  row["job_chunks_attention_mask"],
            "cand_chunks_input_ids":      row["cand_chunks_input_ids"],
            "cand_chunks_attention_mask": row["cand_chunks_attention_mask"],
            # label
            "label": float(row["label"]),
            "vacant_remote": float(row["vacant_remote"]),

        }

        # location features
        item["vac_loc_fourier"] = row["vacant_fourier_feature"]
        item["cand_loc_fourier"] = row["candidate_fourier_features"]

        # ids (opcionales)
        item["vacant_id"]    = row["vacant_id"]
        item["candidate_id"] = row["candidate_id"]

        item["vacant_remote"] = row["vacant_remote"]

        return item
