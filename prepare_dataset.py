"""
STEP 2: Dataset Preparation
- Organizes raw Kaggle dataset into per-fruit train/val/test splits (70/15/15)
- Creates YOLO detection dataset with auto-generated bounding box annotations
- Creates per-fruit freshness classification datasets

Expected raw dataset at: ../archive/Fruit Freshness Dataset/Fruit Freshness Dataset/
Download from: https://www.kaggle.com/datasets/abdulrafeyyashir/fresh-vs-rotten-fruit-images
"""

import os
import shutil
import random
import cv2
from pathlib import Path

# ─── CONFIG ──────────────────────────────────────────────────────────────────
RAW_DATASET = os.path.join("..", "archive", "Fruit Freshness Dataset", "Fruit Freshness Dataset")
OUTPUT_DIR = "data"
YOLO_DIR = "data_yolo"
SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

FRUITS = ["Apple", "Banana", "Strawberry"]
LABELS = ["Fresh", "Rotten"]
FRUIT_TO_ID = {"Apple": 0, "Banana": 1, "Strawberry": 2}


def collect_images(raw_dir):
    """Collect all image paths grouped by fruit and freshness."""
    data = {}
    for fruit in FRUITS:
        for label in LABELS:
            folder = os.path.join(raw_dir, fruit, label)
            if not os.path.isdir(folder):
                print(f"  WARNING: {folder} not found, skipping")
                continue
            images = [
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp"))
            ]
            key = (fruit.lower(), label.lower())
            data[key] = sorted(images)
            print(f"  {fruit}/{label}: {len(images)} images")
    return data


def split_list(items, train_r, val_r):
    """Split a list into train/val/test."""
    random.shuffle(items)
    n = len(items)
    n_train = int(n * train_r)
    n_val = int(n * val_r)
    return items[:n_train], items[n_train:n_train + n_val], items[n_train + n_val:]


def prepare_freshness_dataset(data):
    """
    Create per-fruit freshness datasets:
      data/<fruit>/<split>/fresh/
      data/<fruit>/<split>/rotten/
    """
    print("\n=== Preparing Freshness Classification Dataset ===")

    for fruit in FRUITS:
        fruit_l = fruit.lower()
        for label in LABELS:
            label_l = label.lower()
            key = (fruit_l, label_l)
            if key not in data:
                continue

            images = data[key].copy()
            train, val, test = split_list(images, TRAIN_RATIO, VAL_RATIO)

            for split_name, split_imgs in [("train", train), ("val", val), ("test", test)]:
                dest_dir = os.path.join(OUTPUT_DIR, fruit_l, split_name, label_l)
                os.makedirs(dest_dir, exist_ok=True)
                for i, src in enumerate(split_imgs):
                    ext = os.path.splitext(src)[1]
                    dst = os.path.join(dest_dir, f"{fruit_l}_{label_l}_{split_name}_{i:04d}{ext}")
                    shutil.copy2(src, dst)

            print(f"  {fruit}/{label}: train={len(train)}, val={len(val)}, test={len(test)}")


def prepare_yolo_dataset(data):
    """
    Create YOLO detection dataset with auto-generated bbox annotations.
    Each image is treated as containing one fruit filling ~85% of the frame.
    """
    print("\n=== Preparing YOLO Detection Dataset ===")

    for split_name in ["train", "val", "test"]:
        os.makedirs(os.path.join(YOLO_DIR, "images", split_name), exist_ok=True)
        os.makedirs(os.path.join(YOLO_DIR, "labels", split_name), exist_ok=True)

    # Group images by fruit (ignore freshness label for detection)
    fruit_images = {}
    for fruit in FRUITS:
        fruit_l = fruit.lower()
        imgs = []
        for label in LABELS:
            key = (fruit_l, label.lower())
            if key in data:
                imgs.extend(data[key])
        fruit_images[fruit_l] = imgs

    for fruit_l, images in fruit_images.items():
        random.shuffle(images)
        train, val, test = split_list(images.copy(), TRAIN_RATIO, VAL_RATIO)
        class_id = FRUIT_TO_ID[fruit_l.capitalize()]

        for split_name, split_imgs in [("train", train), ("val", val), ("test", test)]:
            for i, src in enumerate(split_imgs):
                img = cv2.imread(src)
                if img is None:
                    continue

                fname = f"{fruit_l}_{split_name}_{i:04d}.jpg"
                dst_img = os.path.join(YOLO_DIR, "images", split_name, fname)
                cv2.imwrite(dst_img, img)

                # YOLO annotation: class_id cx cy w h (normalized, 85% bbox)
                label_path = os.path.join(
                    YOLO_DIR, "labels", split_name,
                    fname.replace(".jpg", ".txt")
                )
                with open(label_path, "w") as f:
                    f.write(f"{class_id} 0.5 0.5 0.85 0.85\n")

        print(f"  {fruit_l}: YOLO annotations generated")

    # Create fruits.yaml config for YOLO training
    yaml_content = f"""path: {os.path.abspath(YOLO_DIR)}
train: images/train
val: images/val
test: images/test

names:
  0: apple
  1: banana
  2: strawberry
"""
    yaml_path = os.path.join(YOLO_DIR, "fruits.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"  YOLO config written to {yaml_path}")


def main():
    random.seed(SEED)
    print(f"Raw dataset path: {os.path.abspath(RAW_DATASET)}")

    if not os.path.isdir(RAW_DATASET):
        print(f"ERROR: Dataset not found at {RAW_DATASET}")
        print("Download from: https://www.kaggle.com/datasets/abdulrafeyyashir/fresh-vs-rotten-fruit-images")
        return

    print("\nCollecting images...")
    data = collect_images(RAW_DATASET)

    prepare_freshness_dataset(data)
    prepare_yolo_dataset(data)

    print("\n=== Dataset preparation complete! ===")
    print(f"  Freshness data: {os.path.abspath(OUTPUT_DIR)}/")
    print(f"  YOLO data:      {os.path.abspath(YOLO_DIR)}/")


if __name__ == "__main__":
    main()
