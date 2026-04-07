# app.py

import gradio as gr
import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import os

# =========================
# CONFIG
# =========================
TOP_K_DEFAULT = 5

# =========================
# LOAD DATA + MODELS
# =========================

print("Loading models...")

# Load SBERT model
sbert = SentenceTransformer("all-MiniLM-L6-v2")

# Load trained MLP model
MODEL_PATH = "model.pt"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("model.pt not found. Please save your trained model.")

mlp_model = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
mlp_model.eval()

# Load product data (preprocessed)
DATA_PATH = "products.csv"
EMB_PATH = "product_embeddings.npy"

if not os.path.exists(DATA_PATH) or not os.path.exists(EMB_PATH):
    raise FileNotFoundError("products.csv or product_embeddings.npy missing.")

products_df = pd.read_csv(DATA_PATH)
product_embeddings = np.load(EMB_PATH)

print("Loaded successfully.")


# =========================
# SEARCH FUNCTION
# =========================
def semantic_search(query, top_k):
    if not query.strip():
        return "Please enter a valid query."

    # Encode query
    q_emb = sbert.encode(query)

    scores = []

    # Compute scores
    for i in range(len(product_embeddings)):
        p_emb = product_embeddings[i]

        # Feature engineering
        features = np.concatenate([
            q_emb,
            p_emb,
            np.abs(q_emb - p_emb),
            q_emb * p_emb
        ])

        features_tensor = torch.tensor(features, dtype=torch.float32)

        with torch.no_grad():
            score = mlp_model(features_tensor).item()

        scores.append((i, score))

    # Sort by score
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]

    # Format results
    results = []
    for rank, (idx, score) in enumerate(scores, start=1):
        row = products_df.iloc[idx]
        results.append(
            f"{rank}. {row['product_title']} (Score: {score:.4f})"
        )

    return "\n\n".join(results)


# =========================
# GRADIO UI
# =========================
interface = gr.Interface(
    fn=semantic_search,
    inputs=[
        gr.Textbox(
            label="Enter product query",
            placeholder="e.g., wireless earbuds"
        ),
        gr.Slider(1, 20, value=TOP_K_DEFAULT, label="Top K Results")
    ],
    outputs=gr.Textbox(label="Ranked Products"),
    title="Semantic Product Search (SBERT + MLP)",
    description="Deep learning-based semantic product search using SBERT embeddings and MLP ranking model."
)


# =========================
# RUN
# =========================
if __name__ == "__main__":
    interface.launch()