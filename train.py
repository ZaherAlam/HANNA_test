import argparse
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from utils.embeddings import get_smiles_embedding
from models.HANNA import HANNA
from utils.utils import preprocess_input, split_and_reshape_input
from utils.scalers import CustomScaler
from utils.thermodynamics import compute_log10P, thermodynamic_loss_log10P
from utils.utils import initiliaze_ChemBERTA, create_embedding_matrix


# === Argument Parsing ===
parser = argparse.ArgumentParser()
parser.add_argument('--train_targets', type=str, required=True)
parser.add_argument('--train_features', type=str, required=True)
parser.add_argument('--val_targets', type=str, required=False)
parser.add_argument('--val_features', type=str, required=False)
parser.add_argument('--save_dir', type=str, required=True)
parser.add_argument('--config', type=str, required=True)
args = parser.parse_args()

# === Load Config ===
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

nodes = config['model']['nodes']
embedding_dim = config['model']['embedding_dim']
epochs = config['training']['epochs']
batch_size = config['training']['batch_size']
lr = config['training']['lr']

# === Set device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Initialize ChemBERTA ===
ChemBERTA, tokenizer = initiliaze_ChemBERTA(device=device)

# === Create Output Directory ===
os.makedirs(args.save_dir, exist_ok=True)

# === Load Data ===
def prepare_data(targets_path, features_path):
    targets_df = pd.read_csv(targets_path)
    features_df = pd.read_csv(features_path)

    # Ensure ln_gamma columns are floats
    for col in ['ln_gamma_1', 'ln_gamma_2']:
        if col in targets_df.columns:
            targets_df[col] = pd.to_numeric(targets_df[col], errors='coerce')


    X_embeds = []
    for idx, row in targets_df.iterrows():
        T = features_df.loc[idx, 'T(K)']  
        emb_row = create_embedding_matrix(
            row['SMILE 1'], row['SMILE 2'],
            T, device, ChemBERTA, tokenizer,
            x1_values=[row['y1']]
        )
        reshaped = preprocess_input(emb_row)
        X_embeds.append(reshaped[0])  # Use first component

    X_embeds = np.stack(X_embeds)
    return X_embeds, targets_df, features_df


X_train_raw, train_targets, train_features_df = prepare_data(args.train_targets, args.train_features)
# === Preprocess Data ===
scaler = CustomScaler(Embedding_BERT=config['model']['Embedding_ChemBERT'])
print("X_train_raw shape:", X_train_raw.shape)
X_train = scaler.fit_transform(X_train_raw)

if X_train_raw.ndim == 2:
    # Already flattened; reshape before scaling
    X_train_raw = preprocess_input(X_train_raw, Embedding_BERT=embedding_dim)


if args.val_targets and args.val_features:
    X_val_raw, val_targets, val_features_df = prepare_data(args.val_targets, args.val_features)
    X_val = scaler.transform(X_val_raw)
else:
    X_val, val_targets = None, None


# === Convert Data to Tensors ===
T_train, x1_train, emb_train = split_and_reshape_input(X_train)

T_train = torch.tensor(T_train, dtype=torch.float32).to(device)
x1_train = torch.tensor(x1_train, dtype=torch.float32).to(device)
emb_train = torch.tensor(emb_train, dtype=torch.float32).to(device)

if X_val is not None:
    T_val, x1_val, emb_val = split_and_reshape_input(X_val)
    T_val = torch.tensor(T_val, dtype=torch.float32).to(device)
    x1_val = torch.tensor(x1_val, dtype=torch.float32).to(device)
    emb_val = torch.tensor(emb_val, dtype=torch.float32).to(device)

# === Model and Optimizer ===
model = HANNA(Embedding_ChemBERT=embedding_dim, nodes=nodes).to(device)
optimizer = optim.AdamW(model.parameters(), lr=lr)

# === Training Loop ===
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    for i in range(0, len(x1_train), batch_size):
        T_batch = T_train[i:i+batch_size]
        x1_batch = x1_train[i:i+batch_size]
        emb_batch = emb_train[i:i+batch_size]

        pred_ln_gammas, gE = model(T_batch, x1_batch, emb_batch)

        pred_ln_gammas, gE = model(T_batch, x1_batch, emb_batch)
        pred_ln_gamma1 = pred_ln_gammas[:, 0]
        pred_ln_gamma2 = pred_ln_gammas[:, 1]

        # Convert predicted ln_gamma to gamma
        pred_gamma1 = torch.exp(pred_ln_gamma1)
        pred_gamma2 = torch.exp(pred_ln_gamma2)

        train_targets['ln_gamma_1'] = pd.to_numeric(train_targets['ln_gamma_1'], errors='coerce')
        train_targets['ln_gamma_2'] = pd.to_numeric(train_targets['ln_gamma_2'], errors='coerce')


        # Get ground truth ln_gamma values
        ln_gamma1_batch = torch.tensor(train_targets['ln_gamma_1'].values[i:i+batch_size], dtype=torch.float32).to(device)
        ln_gamma2_batch = torch.tensor(train_targets['ln_gamma_2'].values[i:i+batch_size], dtype=torch.float32).to(device)

        # Compute MSE loss directly on ln_gamma predictions
        loss = nn.functional.mse_loss(pred_ln_gamma1, ln_gamma1_batch) + nn.functional.mse_loss(pred_ln_gamma2, ln_gamma2_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        model.eval()
        with torch.no_grad():
            val_pred, _ = model(T_val, x1_val, emb_val)
            val_loss = ...
        print(f"Val Loss: {val_loss:.4f}")


    print(f"Epoch {epoch + 1}/{epochs} | Loss: {epoch_loss:.4f}")

# === Save Model and Scaler ===
torch.save(model.state_dict(), os.path.join(args.save_dir, 'model.pt'))
import joblib
joblib.dump(scaler, os.path.join(args.save_dir, 'scaler.pkl'))
