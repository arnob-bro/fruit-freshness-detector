"""
STEP 3: Train YOLOv8 Fruit Detection Model
- Uses YOLOv8n (nano) pretrained on COCO
- Fine-tunes to detect: apple, banana, strawberry
- Saves best model to detection/yolov8_model.pt
"""

import os
import shutil
from ultralytics import YOLO


def train():
    yaml_path = os.path.join("data_yolo", "fruits.yaml")

    if not os.path.exists(yaml_path):
        print("ERROR: Run prepare_dataset.py first to create YOLO dataset")
        return

    print("Loading YOLOv8n pretrained model...")
    model = YOLO("yolov8n.pt")

    print("Starting training...")
    model.train(
        data=os.path.abspath(yaml_path),
        epochs=50,
        imgsz=640,
        batch=16,
        patience=10,
        save=True,
        project="runs",
        name="fruit_detection",
        exist_ok=True,
        verbose=True,
    )

    # Copy best model to detection/
    os.makedirs("detection", exist_ok=True)
    best_path = os.path.join("runs", "fruit_detection", "weights", "best.pt")
    if os.path.exists(best_path):
        shutil.copy2(best_path, os.path.join("detection", "yolov8_model.pt"))
        print(f"\nBest model saved to detection/yolov8_model.pt")
    else:
        print("WARNING: best.pt not found in training output")

    # Validate on test set
    print("\n=== Test Set Validation ===")
    best_model = YOLO(os.path.join("detection", "yolov8_model.pt"))
    metrics = best_model.val(data=os.path.abspath(yaml_path), split="test")
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    train()
