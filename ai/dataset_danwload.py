from datasets import load_dataset
import os
from PIL import Image

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_DIR = os.path.join(ROOT_DIR, "datasets")

TRAIN_REAL = os.path.join(DATASET_DIR, "train", "real")
TRAIN_FAKE = os.path.join(DATASET_DIR, "train", "fake")
TEST_REAL = os.path.join(DATASET_DIR, "test", "real")
TEST_FAKE = os.path.join(DATASET_DIR, "test", "fake")

os.makedirs(TRAIN_REAL, exist_ok=True)
os.makedirs(TRAIN_FAKE, exist_ok=True)
os.makedirs(TEST_REAL, exist_ok=True)
os.makedirs(TEST_FAKE, exist_ok=True)

print("Downloading dataset from Hugging Face...")

ds = load_dataset("dragonintelligence/CIFAKE-image-dataset")

print("Dataset downloaded successfully!")


print("Saving training images...")

for i, sample in enumerate(ds["train"]):

    image = sample["image"]

    label = sample["label"]

    if label == 1:
        save_path = os.path.join(TRAIN_REAL, f"real_{i}.png")
    else:
        save_path = os.path.join(TRAIN_FAKE, f"fake_{i}.png")

    image.save(save_path)


print("Saving test images...")

for i, sample in enumerate(ds["test"]):

    image = sample["image"]

    label = sample["label"]

    if label == 1:
        save_path = os.path.join(TEST_REAL, f"real_{i}.png")
    else:
        save_path = os.path.join(TEST_FAKE, f"fake_{i}.png")

    image.save(save_path)

print("\nDataset saved successfully!")

print(DATASET_DIR)