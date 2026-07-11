import os
import sys
import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,classification_report,confusion_matrix)

sys.path.append(os.getcwd())

from src.models.custom_layers import AbsoluteDifference

RESUME_EMB_PATH = "data/processed/embeddings/resume_embeddings.npy"
JD_EMB_PATH = "data/processed/embeddings/jd_embeddings.npy"
LABEL_PATH = "data/processed/embeddings/labels.csv"

MODEL_PATH = r"C:\Users\LENOVO\Desktop\YES@\saved_models\best_keras_siamese_binary_model.keras"

REPORT_PATH = "outputs/predictions/model_report_on_best_keras.txt"
os.makedirs("outputs/predictions", exist_ok=True)

resume_embeddings = np.load(RESUME_EMB_PATH).astype("float32")
jd_embeddings = np.load(JD_EMB_PATH).astype("float32")

df = pd.read_csv(LABEL_PATH)

binary_mask = df["label"].isin(["No Fit", "Good Fit"])

df = df[binary_mask].copy().reset_index(drop=True)

resume_embeddings = resume_embeddings[binary_mask.to_numpy()]
jd_embeddings = jd_embeddings[binary_mask.to_numpy()]

label_mapping = {  "No Fit": 0,  "Good Fit": 1}

labels = df["label"].map(label_mapping).to_numpy(dtype=int)

model = load_model( MODEL_PATH,custom_objects={ "AbsoluteDifference": AbsoluteDifference },compile=False)

probs = model.predict([resume_embeddings, jd_embeddings],batch_size=32,verbose=0).flatten()

preds = (probs >= 0.5).astype(int)

accuracy = accuracy_score(labels, preds)
precision = precision_score(labels, preds, zero_division=0)
recall = recall_score(labels, preds, zero_division=0)
f1 = f1_score(labels, preds, zero_division=0)

conf_matrix = confusion_matrix(labels, preds)

class_report = classification_report(labels,preds,labels=[0, 1],target_names=["No Fit", "Good Fit"],zero_division=0)

report = f"""
SIAMESE BINARY MODEL EVALUATION REPORT
=======================================

Total Binary Samples: {len(labels)}
Threshold: 0.5

Accuracy:  {accuracy:.4f}
Precision: {precision:.4f}
Recall:    {recall:.4f}
F1 Score:  {f1:.4f}

Confusion Matrix:
{conf_matrix}

Classification Report:
{class_report}
"""

print(report)

with open(REPORT_PATH, "w", encoding="utf-8") as file:
    file.write(report)


print("Report saved at:", REPORT_PATH)
