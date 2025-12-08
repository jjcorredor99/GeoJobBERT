from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch


class TripletWithFourierDataset(Dataset):
    def __init__(self, df):
        # text columns in YOUR df
        self.anchor_texts = df["anchor_full_text"].tolist()
        self.pos_texts    = df["pos_candidate_full_text"].tolist()
        self.neg_texts    = df["neg_candidate_full_text"].tolist()

        # Fourier features (arrays)
        self.anchor_four = df["anchor_fourier_feature"].tolist()
        self.pos_four    = df["pos_candidate_fourier_features"].tolist()
        self.neg_four    = df["neg_candidate_fourier_features"].tolist()

    def __len__(self):
        return len(self.anchor_texts)

    def __getitem__(self, idx):
        a_text = self.anchor_texts[idx]
        p_text = self.pos_texts[idx]
        n_text = self.neg_texts[idx]

        a_four = np.asarray(self.anchor_four[idx], dtype="float32")
        p_four = np.asarray(self.pos_four[idx], dtype="float32")
        n_four = np.asarray(self.neg_four[idx], dtype="float32")

        return a_text, p_text, n_text, a_four, p_four, n_four



def make_collate_fn(base_model):
    def collate_fn(batch):
        """
        batch: list of (a_text, p_text, n_text, a_four, p_four, n_four)
        """
        a_texts, p_texts, n_texts, a_four, p_four, n_four = zip(*batch)

        anchor_features = base_model.tokenize(list(a_texts))
        pos_features    = base_model.tokenize(list(p_texts))
        neg_features    = base_model.tokenize(list(n_texts))

        a_four = torch.tensor(np.stack(a_four), dtype=torch.float32)
        p_four = torch.tensor(np.stack(p_four), dtype=torch.float32)
        n_four = torch.tensor(np.stack(n_four), dtype=torch.float32)

        return anchor_features, pos_features, neg_features, a_four, p_four, n_four

    return collate_fn


