import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Tuple


def _pad_3d(list_of_2d: List[List[List[int]]], pad_val: int = 0):
    B = len(list_of_2d)
    if B == 0:
        return torch.empty(0), torch.empty(0)
        
    C_max = max(len(x) for x in list_of_2d) 
    T_max = max((len(t) for x in list_of_2d for t in x), default=0)
    
    out = torch.full((B, C_max, T_max), pad_val, dtype=torch.long)
    mask = torch.zeros((B, C_max, T_max), dtype=torch.long)
    
    for i, mat in enumerate(list_of_2d):
        for c, seq in enumerate(mat):
            L = len(seq)
            out[i, c, :L] = torch.tensor(seq, dtype=torch.long)
            mask[i, c, :L] = 1
            
    return out, mask

def make_triplet_collate_fn(pad_id: int = 0, max_chunks: int = 4):
    
    def collate_fn(batch: List[Dict[str, Any]]):
        
        # 1. Collect Lists
        anchor_ids_list = [b["anchor_ids"] for b in batch]
        pos_ids_list    = [b["pos_ids"] for b in batch]
        neg_ids_list    = [b["neg_ids"] for b in batch]

        # 2. Pad & Tensorize (3D)
        # Note: If job masks are missing in DF, _pad_3d generates basic masks automatically
        anchor_input, anchor_mask = _pad_3d(anchor_ids_list, pad_val=pad_id)
        pos_input,    pos_mask    = _pad_3d(pos_ids_list,    pad_val=pad_id)
        neg_input,    neg_mask    = _pad_3d(neg_ids_list,    pad_val=pad_id)

        # 3. Truncate Chunks (Optimization)
        anchor_input = anchor_input[:, :max_chunks]
        anchor_mask  = anchor_mask[:, :max_chunks]
        
        pos_input = pos_input[:, :max_chunks]
        pos_mask  = pos_mask[:, :max_chunks]
        
        neg_input = neg_input[:, :max_chunks]
        neg_mask  = neg_mask[:, :max_chunks]

        # 4. Build SBERT Feature Dicts
        features_anchor = {
            "input_ids": anchor_input,
            "attention_mask": anchor_mask,
            # Add metadata if your model needs it for the forward pass
            "vacant_id": torch.tensor([b["vacant_id"] for b in batch], dtype=torch.long)
        }

        features_pos = {
            "input_ids": pos_input,
            "attention_mask": pos_mask,
            "candidate_id": torch.tensor([b["pos_candidate_id"] for b in batch], dtype=torch.long)
        }

        features_neg = {
            "input_ids": neg_input,
            "attention_mask": neg_mask,
            "candidate_id": torch.tensor([b["neg_candidate_id"] for b in batch], dtype=torch.long)
        }

        # 5. Return
        # SBERT expects: [Anchor, Positive, Negative], Label(optional)
        return [features_anchor, features_pos, features_neg], None

    return collate_fn



class TripletDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        return {
            # --- ANCHOR (The Job) ---
            # We use the 'pos_job' columns as the Anchor
            "anchor_ids":    row['pos_job_chunks_input_ids'],
            # Assuming you have masks, if not we can assume all 1s or infer, 
            # but usually they are paired with input_ids.
            # If your DF doesn't have 'pos_job_chunks_attention_mask', 
            # we might need to generate it or use the one from neg if available.
            # I will assume standard naming or auto-generation if missing.
            "anchor_mask":   row.get('pos_job_chunks_attention_mask', None), 
            
            # --- POSITIVE (The Candidate) ---
            "pos_ids":       row['pos_cand_chunks_input_ids'],
            "pos_mask":      row['pos_cand_chunks_attention_mask'],
            
            # --- NEGATIVE (The Hard Negative Candidate) ---
            "neg_ids":       row['neg_cand_chunks_input_ids'],
            "neg_mask":      row['neg_cand_chunks_attention_mask'],

            # --- METADATA (Optional) ---
            "vacant_id":        row['vacant_id'],
            "pos_candidate_id": row['pos_candidate_id'],
            "neg_candidate_id": row['neg_candidate_id']
        }