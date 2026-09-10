"""
Step 6: Evaluate the trained model on the test set + plot training curves.
Run after train.py has produced best_model.pth and training_history.csv
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ---- CONFIG ----
CODE_DIR = Path(r"D:\B.Tech\CE-AI\Sem 7\MP\Code work")
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class BreedDataset(Dataset):
    def __init__(self, csv_path, class_to_idx, transform=None):
        self.df = pd.read_csv(csv_path)
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["filepath"]).convert("RGB")
        label = self.class_to_idx[row["label"]]
        if self.transform:
            image = self.transform(image)
        return image, label


if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # =========================================================
    # PART 1: Plot training curves
    # =========================================================
    history = pd.read_csv(CODE_DIR / "training_history.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history["epoch"], history["train_acc"], label="Train Acc", marker="o", markersize=3)
    axes[0].plot(history["epoch"], history["val_acc"], label="Val Acc", marker="o", markersize=3)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Train vs Validation Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history["epoch"], history["train_loss"], label="Train Loss", marker="o", markersize=3)
    axes[1].plot(history["epoch"], history["val_loss"], label="Val Loss", marker="o", markersize=3)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Train vs Validation Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    curves_path = CODE_DIR / "training_curves.png"
    plt.savefig(curves_path, dpi=150)
    print(f"Saved training curves to: {curves_path}")
    plt.show()

    # =========================================================
    # PART 2: Evaluate on test set
    # =========================================================
    checkpoint = torch.load(CODE_DIR / "best_model.pth", map_location=DEVICE)
    classes = checkpoint["classes"]
    NUM_CLASSES = len(classes)
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}

    print(f"\nLoaded best model from epoch {checkpoint['epoch']}, "
          f"val_acc={checkpoint['val_acc']:.4f}")

    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, NUM_CLASSES)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    test_dataset = BreedDataset(CODE_DIR / "test.csv", class_to_idx, transform=eval_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    test_acc = (all_preds == all_labels).mean()
    print(f"\nTest Accuracy: {test_acc:.4f}")

    # ---- Classification report ----
    target_names = [idx_to_class[i] for i in range(NUM_CLASSES)]
    report = classification_report(all_labels, all_preds, target_names=target_names,
                                     digits=3, zero_division=0)
    print("\nPer-class classification report:\n")
    print(report)

    report_path = CODE_DIR / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Test Accuracy: {test_acc:.4f}\n\n")
        f.write(report)
    print(f"Saved classification report to: {report_path}")

    # ---- Confusion matrix (saved as image, 41 classes so make it large) ----
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(20, 18))
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=target_names, yticklabels=target_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix (Test Acc: {test_acc:.4f})")
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()
    cm_path = CODE_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    print(f"Saved confusion matrix to: {cm_path}")
    plt.show()