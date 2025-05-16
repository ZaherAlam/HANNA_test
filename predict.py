import argparse
import os
import torch
import pandas as pd
import numpy as np
import yaml
from utils.embeddings import get_smiles_embedding
from models.HANNA import HANNA
from utils.utils import preprocess_input, split_and_reshape_input, create_embedding_matrix, initiliaze_ChemBERTA
from utils.thermodynamics import compute_log10P, thermodynamic_loss_log10P
from utils.scalers import CustomScaler
from sklearn.metrics import mean_squared_error

# === Argument Parsing ===
parser = argparse.ArgumentParser()
parser.add_argument('--test_targets', type=str, required=True)
parser.add_argument('--test_features', type=str, required=True)
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
features_df = pd.read_csv(args.test_features)
targets_df = pd.read_csv(args.test_targets)

# === Load Model ===
model = HANNA(Embedding_ChemBERT=EMBED_DIM, nodes=HIDDEN_NODES)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(args.model_path, map_location=device))
model.to(device)
model.eval()

# === Initialize ChemBERTa and Tokenizer ===
ChemBERTA, custom_tokenizer = initiliaze_ChemBERTA(device=device)

# === Generate Embedding Matrix ===
all_embeddings = []
for i, row in targets_df.iterrows():
    sm1, sm2 = row['SMILE 1'], row['SMILE 2']
    T = features_df.loc[i, 'T(K)']
    x1 = features_df.loc[i, 'x1']
    emb_row = create_embedding_matrix(sm1, sm2, T, device, ChemBERTA, custom_tokenizer, x1_values=[x1])[0]
    all_embeddings.append(emb_row)

embedding_matrix = np.array(all_embeddings)  # shape [N, 2+2*384]

# === Preprocess Input ===
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
x_tensor.requires_grad_(True)
T_tensor.requires_grad_(True)
ln_gammas_pred, gE_pred = model(T_tensor, x_tensor, FPs_tensor)


y1_pred = x_tensor.detach().cpu().numpy().flatten()
y2_pred = 1 - y1_pred
ln_gamma_1 = ln_gammas_pred[:, 0].detach().cpu().numpy()
ln_gamma_2 = ln_gammas_pred[:, 1].detach().cpu().numpy()
gE_vals = gE_pred.detach().cpu().numpy()
gamma_1 = np.exp(ln_gamma_1)
gamma_2 = np.exp(ln_gamma_2)
x2_pred = 1 - y1_pred
log10P_pred = np.log10(y1_pred * gamma_1 * 10**features_df['log10P1sat'].values +
                       x2_pred * gamma_2 * 10**features_df['log10P2sat'].values)

# === Final DataFrame ===
output_df = pd.DataFrame({
    'SMILE 1': targets_df['SMILE 1'],
    'SMILE 2': targets_df['SMILE 2'],
    'y1': y1_pred,
    'y2': y2_pred,
    'log10P': log10P_pred,
    'ln_gamma_1': ln_gamma_1,
    'ln_gamma_2': ln_gamma_2,
    'log10P1sat': features_df['log10P1sat'],
    'log10P2sat': features_df['log10P2sat']
})

# === Save Prediction File ===
os.makedirs(args.out_dir, exist_ok=True)
pred_path = os.path.join(args.out_dir, args.out_name)
output_df.to_csv(pred_path, index=False)

# === Evaluation Metrics ===
y_true = targets_df[['y1', 'y2', 'log10P']].values
y_pred = output_df[['y1', 'y2', 'log10P']].values
rmse = np.sqrt(((y_true - y_pred) ** 2).mean(axis=0))

score_df = pd.DataFrame({
    'Task': ['y1', 'y2', 'log10P'],
    'Mean rmse': rmse,
    'Standard deviation rmse': [0]*3,
    'Fold 0 rmse': rmse
})
score_df.to_csv(os.path.join(args.out_dir, 'test_scores.csv'), index=False)
print(f"✅ Predictions saved to: {pred_path}\n✅ Scores saved to: test_scores.csv")
