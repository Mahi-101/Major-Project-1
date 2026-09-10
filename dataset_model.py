"""
Step 4: Custom Dataset, DataLoaders, and EfficientNet-B0 model setup.
Run this after step3_prepare_data.py has generated train.csv, val.csv, test.csv, classes.txt
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import pandas as pd
from pathlib import Path

# ---- CONFIG ----
CODE_DIR = Path(r"D:\B.Tech\CE-AI\Sem 7\MP\Code work")
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4  # set to 0 if you still get multiprocessing errors on Windows

# ---- Transforms ----
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# ---- Custom Dataset ----
class BreedDataset(Dataset):
    def __init__(self, csv_path, class_to_idx, transform=None):
        self.df = pd.read_csv(csv_path)
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["filepath"]
        label = self.class_to_idx[row["label"]]

        image = Image.open(img_path).convert("RGB")  # handles PNG/GIF alpha channels too
        if self.transform:
            image = self.transform(image)

        return image, label


# ---- Everything below spawns worker processes, so it must be guarded on Windows ----
if __name__ == "__main__":

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # ---- Load class list ----
    with open(CODE_DIR / "classes.txt", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    NUM_CLASSES = len(classes)
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    print(f"Number of classes: {NUM_CLASSES}")

    # ---- Build datasets ----
    train_dataset = BreedDataset(CODE_DIR / "train.csv", class_to_idx, transform=train_transform)
    val_dataset = BreedDataset(CODE_DIR / "val.csv", class_to_idx, transform=eval_transform)
    test_dataset = BreedDataset(CODE_DIR / "test.csv", class_to_idx, transform=eval_transform)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # ---- Build DataLoaders ----
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    # ---- Sanity check: fetch one batch ----
    images, labels = next(iter(train_loader))
    print(f"\nSample batch -> images: {images.shape}, labels: {labels.shape}")

    # ---- Build EfficientNet-B0 model (transfer learning) ----
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    # Replace the final classifier layer for our 41 classes
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, NUM_CLASSES)

    model = model.to(DEVICE)
    print(f"\nModel loaded: EfficientNet-B0, final layer -> {NUM_CLASSES} classes")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,} | Trainable params: {trainable_params:,}")