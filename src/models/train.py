import os
import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

sys.path.append(os.getcwd())

from src.models.simennes import build_siamese_model

RESUME_EMB_PATH = "data/processed/embeddings/resume_embeddings.npy"
JD_EMB_PATH = "data/processed/embeddings/jd_embeddings.npy"
LABEL_PATH = "data/processed/embeddings/labels.csv"

MODEL_SAVE_PATH = "saved_models/keras_siamese_binary_model.keras"
BEST_MODEL_PATH = "saved_models/best_keras_siamese_binary_model.keras"

LABEL_COL = "label"

os.makedirs("saved_models", exist_ok=True)

resume_embeddings = np.load(RESUME_EMB_PATH)
jd_embeddings = np.load(JD_EMB_PATH)
df = pd.read_csv(LABEL_PATH)

df = df.reset_index()
df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip()

df = df[df[LABEL_COL].isin(["No Fit", "Good Fit"])].copy()

selected_indices = df["index"].to_numpy()

resume_embeddings = resume_embeddings[selected_indices]
jd_embeddings = jd_embeddings[selected_indices]

label_mapping = {
    "No Fit": 0,
    "Good Fit": 1
}

labels = df[LABEL_COL].map(label_mapping).astype(np.int32).to_numpy()

print("Filtered embeddings:", resume_embeddings.shape, jd_embeddings.shape)
print("Label counts:")
print(df[LABEL_COL].value_counts())

X_train_r, X_val_r, X_train_j, X_val_j, y_train, y_val = train_test_split(
    resume_embeddings,
    jd_embeddings,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

model = build_siamese_model(embedding_dim=resume_embeddings.shape[1])

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        BEST_MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max"
    )
]

history = model.fit(
    [X_train_r, X_train_j],
    y_train,
    validation_data=([X_val_r, X_val_j], y_val),
    epochs=30,
    batch_size=32,
    callbacks=callbacks
)

model.save(MODEL_SAVE_PATH)

print("Last model saved at:", MODEL_SAVE_PATH)
print("Best model saved at:", BEST_MODEL_PATH)

probs = model.predict([X_val_r, X_val_j]).flatten()
preds = (probs >= 0.5).astype(int)

report = classification_report(
    y_val,
    preds,
    target_names=["No Fit", "Good Fit"],
    zero_division=0
)
print(report)

report_path = "outputs/reports/classification_report.txt"
os.makedirs(os.path.dirname(report_path), exist_ok=True)
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report)
print("Classification report saved to:", report_path)