"""
Step 3: Build a labeled dataframe from the folder structure, drop rare classes,
and create a stratified train/val/test split.

Folder structure expected:
    <DATA_ROOT>/buffalo/<breed_name>/*.jpg|png|jpeg|gif
    <DATA_ROOT>/cattle/<breed_name>/*.jpg|png|jpeg|gif
"""

import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# ---- CONFIG: update this path if needed ----
DATA_ROOT = Path(r"D:\B.Tech\CE-AI\Sem 7\MP\Datasets\archive (1)\clean_images")
OUTPUT_DIR = Path(r"D:\B.Tech\CE-AI\Sem 7\MP\Code work")
MIN_IMAGES_PER_CLASS = 20
VALID_EXTS = {".jpg", ".jpeg", ".png", ".gif"}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Step 1: Walk the folder structure ----
records = []
for species_dir in DATA_ROOT.iterdir():
    if not species_dir.is_dir():
        continue
    species = species_dir.name.lower()  # 'buffalo' or 'cattle'

    for breed_dir in species_dir.iterdir():
        if not breed_dir.is_dir():
            continue
        breed = breed_dir.name.lower()
        label = f"{species}_{breed}"

        for img_path in breed_dir.iterdir():
            if img_path.suffix.lower() in VALID_EXTS:
                records.append({
                    "filepath": str(img_path),
                    "species": species,
                    "breed": breed,
                    "label": label
                })

df = pd.DataFrame(records)
print(f"Total images found: {len(df)}")
print(f"Total classes found: {df['label'].nunique()}")

# ---- Step 2: Drop rare classes ----
class_counts = df["label"].value_counts()
keep_classes = class_counts[class_counts >= MIN_IMAGES_PER_CLASS].index
dropped_classes = class_counts[class_counts < MIN_IMAGES_PER_CLASS]

df_filtered = df[df["label"].isin(keep_classes)].reset_index(drop=True)

print(f"\nDropped {len(dropped_classes)} classes with < {MIN_IMAGES_PER_CLASS} images:")
print(dropped_classes.to_string())
print(f"\nRemaining classes: {df_filtered['label'].nunique()}")
print(f"Remaining images: {len(df_filtered)}")

# ---- Step 3: Stratified train/val/test split (70/15/15) ----
train_df, temp_df = train_test_split(
    df_filtered,
    test_size=0.30,
    stratify=df_filtered["label"],
    random_state=42
)
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=42
)

print(f"\nTrain: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# ---- Step 4: Save CSVs ----
train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
val_df.to_csv(OUTPUT_DIR / "val.csv", index=False)
test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)

# Save the final class list (needed later for label encoding)
class_list = sorted(df_filtered["label"].unique())
with open(OUTPUT_DIR / "classes.txt", "w") as f:
    f.write("\n".join(class_list))

print(f"\nSaved train.csv, val.csv, test.csv, classes.txt to: {OUTPUT_DIR}")
print(f"Final number of classes: {len(class_list)}")