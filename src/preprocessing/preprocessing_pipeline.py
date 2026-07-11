import os
import pandas as pd
from clean_txt import clean_text

RAW_PATH = "data/raw/train.csv"
SAVE_PATH = "data/processed/cleaned_train.csv"

os.makedirs("data/processed", exist_ok=True)

df = pd.read_csv(RAW_PATH)

print("Columns:", df.columns.tolist())

RESUME_COL = "resume_text"
JD_COL = "job_description_text"
LABEL_COL = "label"

df["resume_clean"] = df[RESUME_COL].apply(clean_text)
df["jd_clean"] = df[JD_COL].apply(clean_text)

final_df = df[[RESUME_COL, JD_COL, "resume_clean", "jd_clean", LABEL_COL]]

final_df.to_csv(SAVE_PATH, index=False)

print("Saved:", SAVE_PATH)
print(final_df.head())