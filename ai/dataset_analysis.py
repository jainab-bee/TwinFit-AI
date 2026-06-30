from datasets import load_dataset
import matplotlib.pyplot as plt
from collections import Counter

print("Loading Dataset...")

dataset = load_dataset("dragonintelligence/CIFAKE-image-dataset")

print("\nDataset Loaded Successfully!\n")

print("Dataset Structure:")
print(dataset)

print("\nTraining Samples :", len(dataset["train"]))
print("Testing Samples  :", len(dataset["test"]))


sample = dataset["train"][0]

print("\nFirst Sample Information")
print("------------------------")
print("Label :", sample["label"])
print("Image Size :", sample["image"].size)

labels = [item["label"] for item in dataset["train"]]

counter = Counter(labels)

print("\nClass Distribution")
print(counter)


plt.figure(figsize=(12,5))

for i in range(5):

    plt.subplot(1,5,i+1)

    plt.imshow(dataset["train"][i]["image"])

    plt.title(f"Label : {dataset['train'][i]['label']}")

    plt.axis("off")

plt.tight_layout()

plt.show()