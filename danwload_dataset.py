from datasets import load_dataset
import pandas as pd
import os

os.makedirs("data/raw", exist_ok=True)

ds = load_dataset("cnamuangtoun/resume-job-description-fit")

for split in ds.keys():
    df = pd.DataFrame(ds[split])
    df.to_csv(f"data/raw/{split}.csv", index=False)

print("Dataset downloaded successfully!")