import argparse
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import joblib
import copy
from utils.HANNA import HANNA
from utils.Utils import preprocess_input, split_and_reshape_input
from utils.Own_Scaler import CustomScaler
from utils.Utils import initiliaze_ChemBERTA, create_embedding_matrix, get_smiles_embedding, preprocess_input, canonicalize_smiles

# === Argument Parsing ===
parser = argparse.ArgumentParser()
parser.add_argument('--train_targets', type=str, required=True)
parser.add_argument('--val_targets', type=str, required=False)
parser.add_argument('--save_dir', type=str, required=True)
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
args = parser.parse_args()

# === Set Seed for Reproducibility ===
seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# === Load Config ===
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

nodes = config['model']['nodes']
embedding_dim = config['model']['embedding_dim']
max_epochs = config['training']['max_epochs']
patience = config['training']['patience']
lr_decay_factor = config['training']['lr_decay_factor']
batch_size = config['training']['batch_size']
lr = config['training']['lr']
early_stop_epochs = config['training']['early_stop_epochs']
l1_beta = config['training']['l1_beta']

# === Set device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Initialize ChemBERTA ===
print("Initializing ChemBERTA...")
ChemBERTA, tokenizer = initiliaze_ChemBERTA(device=device)

# === Create Output Directory ===
os.makedirs(args.save_dir, exist_ok=True)

# === Load Data ===
def prepare_data(targets_path):
    targets_df = pd.read_csv(targets_path)

    # Ensure ln_gamma columns are floats
    for col in ['ln_gamma_1', 'ln_gamma_2']:
        if col in targets_df.columns:
            targets_df[col] = pd.to_numeric(targets_df[col], errors='coerce')

    embedding_cache = {}
    unique_smiles = set(targets_df['SMILE 1']).union(set(targets_df['SMILE 2']))
    for smile in unique_smiles:
        c_smile = canonicalize_smiles(smile)
        embedding_cache[smile] = get_smiles_embedding(c_smile,custom_tokenizer=tokenizer,ChemBERTA=ChemBERTA,device=device).flatten()

    X_embeds = []
    for idx, row in targets_df.iterrows():
        T = targets_df.loc[idx, 'T(K)']  
        x1 = targets_df.loc[idx, 'x1']
        ## Instead of using the utils create_embedding_matrix function, we will build it from cached values.
        # emb_row = create_embedding_matrix(
        #     row['SMILE 1'], row['SMILE 2'],
        #     T, device, ChemBERTA, tokenizer,
        #     x1_values=[x1]
        # )
        emb_row = np.concatenate([[T, x1], embedding_cache[row['SMILE 1']], embedding_cache[row['SMILE 2']]])
        X_embeds.append(emb_row)  # Use first component

    X_embeds = np.stack(X_embeds)
    X_embeds = preprocess_input(X_embeds, Embedding_BERT=embedding_dim)
    return X_embeds, targets_df


X_train_raw, train_targets = prepare_data(args.train_targets)
# === Preprocess Data ===
print("Preprocessing input data...")
scaler = CustomScaler(Embedding_BERT=config['model']['Embedding_ChemBERT'])
X_train = scaler.fit_transform(X_train_raw)

if args.val_targets:
    X_val_raw, val_targets = prepare_data(args.val_targets)
    X_val = scaler.transform(X_val_raw)
else:
    X_val, val_targets = None, None


# === Convert Data to Tensors ===
T_train, x1_train, emb_train = split_and_reshape_input(X_train)

T_train = torch.tensor(T_train, dtype=torch.float32).to(device)
x1_train = torch.tensor(x1_train, dtype=torch.float32).to(device)
emb_train = torch.tensor(emb_train, dtype=torch.float32).to(device)
targets_train = torch.tensor(train_targets[['ln_gamma_1', 'ln_gamma_2']].values, dtype=torch.float32).to(device)

train_dataset = TensorDataset(T_train, x1_train, emb_train, targets_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

if X_val is not None:
    T_val, x1_val, emb_val = split_and_reshape_input(X_val)
    T_val = torch.tensor(T_val, dtype=torch.float32).to(device)
    x1_val = torch.tensor(x1_val, dtype=torch.float32).to(device)
    emb_val = torch.tensor(emb_val, dtype=torch.float32).to(device)
    targets_val = torch.tensor(val_targets[['ln_gamma_1', 'ln_gamma_2']].values, dtype=torch.float32).to(device)

# === Model, optimizer, lr scheduler, loss function ===
model = HANNA(Embedding_ChemBERT=embedding_dim, nodes=nodes).to(device)
optimizer = optim.AdamW(model.parameters(), lr=lr)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=lr_decay_factor, patience=patience)
loss_fn = nn.SmoothL1Loss(beta=l1_beta)
torch.set_printoptions(threshold=float('inf'))

# === Training Loop ===
best_val_epoch = 0
best_val_loss = float('inf')

for epoch in range(max_epochs):
    epoch_loss = 0.0
    for T_batch, x1_batch, emb_batch, targets_batch in train_loader:

        pred_ln_gammas, gE = model(T_batch, x1_batch, emb_batch)

        # Compute MSE loss directly on ln_gamma predictions
        loss = loss_fn(pred_ln_gammas, targets_batch).sum()

        # Step the optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check val loss and step the scheduler
        epoch_loss += loss.item()

        if torch.isnan(loss):
            print(f"Epoch {epoch} - Loss is NaN, stopping training.")
            print("pred", pred_ln_gammas)
            print("targets", targets_batch)
            print(f"Batch T: {T_batch}")
            print(f"x1: {x1_batch}")
            print(f"Batch Loss: {loss.item()}")
            raise ValueError("Loss is NaN, stopping training.")

    val_pred, _ = model(T_val, x1_val, emb_val)
    val_loss = loss_fn(val_pred, targets_val).sum().item()
    lr_scheduler.step(val_loss)
    optimizer.zero_grad()

    print(f"Epoch {epoch} - Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_val_epoch = epoch
        best_model = copy.deepcopy(model)

    if epoch - best_val_epoch >= early_stop_epochs:
        print(f"Early stopping at epoch {epoch}. Best validation loss: {best_val_loss:.4f} at epoch {best_val_epoch}.")
        break

# === Save Model and Scaler ===
torch.save(best_model.state_dict(), os.path.join(args.save_dir, 'model.pt'))
joblib.dump(scaler, os.path.join(args.save_dir, 'scaler.pkl'))
