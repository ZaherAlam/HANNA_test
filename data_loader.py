import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from utils.embeddings import smiles_to_embedding

class HANNA_Dataset(Dataset):
    def __init__(self, targets_df, features_df):
        self.targets = targets_df.reset_index(drop=True)
        self.features = features_df.reset_index(drop=True)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        row_f = self.features.loc[idx]
        row_t = self.targets.loc[idx]

        x1 = row_f['x1']
        x2 = row_f['x2']
        T = row_f['T(K)']

        emb1 = smiles_to_embedding(row_t['SMILE 1'])
        emb2 = smiles_to_embedding(row_t['SMILE 2'])
        features = torch.tensor(list(emb1) + list(emb2) + [x1, x2, T], dtype=torch.float32)

        ln_gamma1 = torch.tensor(row_t['ln_gamma_1'], dtype=torch.float32)
        ln_gamma2 = torch.tensor(row_t['ln_gamma_2'], dtype=torch.float32)

        return features, torch.tensor(x1), torch.tensor(x2), ln_gamma1, ln_gamma2


def load_dataset(targets_df, features_df, batch_size=64, shuffle=True):
    dataset = HANNA_Dataset(targets_df, features_df)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)