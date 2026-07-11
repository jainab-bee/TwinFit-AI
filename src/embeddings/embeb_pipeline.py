import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

DATA_PATH = "data/processed/cleaned_train.csv"
SAVE_DIR = "data/processed/embeddings"

os.makedirs(SAVE_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

model = SentenceTransformer("all-MiniLM-L6-v2")

resume_embeddings = model.encode(
    df["resume_clean"].tolist(),
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True
)

jd_embeddings = model.encode(
    df["jd_clean"].tolist(),
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True
)

np.save(f"{SAVE_DIR}/resume_embeddings.npy", resume_embeddings)
np.save(f"{SAVE_DIR}/jd_embeddings.npy", jd_embeddings)

df.to_csv(f"{SAVE_DIR}/labels.csv", index=False)

print("Resume Embeddings:", resume_embeddings.shape)
print("JD Embeddings:", jd_embeddings.shape)
print("Embeddings saved successfully.")