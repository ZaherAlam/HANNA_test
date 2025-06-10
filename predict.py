import argparse
import os
import torch
import pandas as pd
import numpy as np
import yaml
from utils.HANNA import HANNA
from utils.Utils import preprocess_input, split_and_reshape_input, create_embedding_matrix, initiliaze_ChemBERTA, canonicalize_smiles, get_smiles_embedding
from utils.thermodynamics import compute_log10P, thermodynamic_loss_log10P
from utils.Own_Scaler import CustomScaler
from sklearn.metrics import mean_squared_error

# === Argument Parsing ===
parser = argparse.ArgumentParser()
parser.add_argument('--test_targets', type=str, required=True)
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--out_dir', type=str, required=True)
parser.add_argument('--out_name', type=str, default="predictions.csv")
args = parser.parse_args()

# === Load Configuration ===
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

EMBED_DIM = config['model']['Embedding_ChemBERT']
HIDDEN_NODES = config['model']['nodes']

# === Load Data ===
targets_df = pd.read_csv(args.test_targets)

# === Load Model ===
print("Loading model...")
model = HANNA(Embedding_ChemBERT=EMBED_DIM, nodes=HIDDEN_NODES)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(args.model_path, map_location=device))
model.to(device)
model.eval()

# === Initialize ChemBERTa and Tokenizer ===
print("Initializing ChemBERTA...")
ChemBERTA, tokenizer = initiliaze_ChemBERTA(device=device)

# === Generate Embedding Matrix ===
print("Creating embedding matrix...")
embedding_cache = {}
unique_smiles = set(targets_df['SMILE 1']).union(set(targets_df['SMILE 2']))
for smile in unique_smiles:
    c_smile = canonicalize_smiles(smile)
    embedding_cache[smile] = get_smiles_embedding(c_smile,custom_tokenizer=tokenizer,ChemBERTA=ChemBERTA,device=device).flatten()

all_embeddings = []
for i, row in targets_df.iterrows():
    sm1, sm2 = row['SMILE 1'], row['SMILE 2']
    T = targets_df.loc[i, 'T(K)']
    x1 = targets_df.loc[i, 'x1']
    emb_row = np.concatenate([[T, x1], embedding_cache[row['SMILE 1']], embedding_cache[row['SMILE 2']]])
    all_embeddings.append(emb_row)

embedding_matrix = np.array(all_embeddings)  # shape [N, 2+2*384]

# === Preprocess Input ===
print("Preprocessing input...")
X_test_processed = preprocess_input(embedding_matrix, Embedding_BERT=EMBED_DIM)

# === Scale ===
scaler = CustomScaler(Embedding_BERT=EMBED_DIM)
X_test_scaled = scaler.fit_transform(X_test_processed)  # Note: using fit_transform for demo only

# === Split ===
T_tensor, x_tensor, FPs_tensor = split_and_reshape_input(X_test_scaled)
T_tensor = torch.tensor(T_tensor, dtype=torch.float32, device=device)
x_tensor = torch.tensor(x_tensor, dtype=torch.float32, device=device)
FPs_tensor = torch.tensor(FPs_tensor, dtype=torch.float32, device=device)

# === Predict (do NOT disable gradients since autograd is used in model)
print("Making predictions...")
x_tensor.requires_grad_(True)
T_tensor.requires_grad_(True)
ln_gammas_pred, gE_pred = model(T_tensor, x_tensor, FPs_tensor)

ln_gamma_1 = ln_gammas_pred[:, 0].detach().cpu().numpy()
ln_gamma_2 = ln_gammas_pred[:, 1].detach().cpu().numpy()

# === Final DataFrame ===
output_df = pd.DataFrame({
    'SMILE 1': targets_df['SMILE 1'],
    'SMILE 2': targets_df['SMILE 2'],
    'ln_gamma_1': ln_gamma_1,
    'ln_gamma_2': ln_gamma_2,
})

# === Save Prediction File ===
os.makedirs(args.out_dir, exist_ok=True)
pred_path = os.path.join(args.out_dir, args.out_name)
output_df.to_csv(pred_path, index=False)

# === Evaluation Metrics ===
gamma_true = targets_df[['ln_gamma_1', 'ln_gamma_2']].values
gamma_pred = output_df[['ln_gamma_1', 'ln_gamma_2']].values
gamma_rmse = np.sqrt(((gamma_true - gamma_pred) ** 2).mean())

print(f"RMSE for ln_gamma: {gamma_rmse:.4f}")

score_df = pd.DataFrame({
    'Task': ['ln_gamma_1', 'ln_gamma_2'],
    'Mean rmse': gamma_rmse,
})

score_df.to_csv(os.path.join(args.out_dir, 'test_scores.csv'), index=False)
print(f"✅ Predictions saved to: {pred_path}\n✅ Scores saved to: test_scores.csv")
