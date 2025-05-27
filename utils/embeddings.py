from transformers import AutoModel, AutoTokenizer
import torch

def get_smiles_embedding(smiles, tokenizer, model, device, max_length=512):
    tokens = tokenizer(smiles, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length)
    tokens = {k: v.to(device) for k, v in tokens.items()}
    with torch.no_grad():
        outputs = model(**tokens)
    embedding = outputs.last_hidden_state[:, 0, :]  # CLS token
    return embedding.squeeze().cpu().numpy()
