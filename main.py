"""
Full 2-Stage Inference Pipeline
  Stage 1: YOLOv8 detects fruit type (apple / banana / strawberry)
  Stage 2: Fruit-specific best CNN classifies freshness (Fresh / Rotten)

Usage:
  python main.py <image_path>         Process a single image
  python main.py <folder_path>        Process all images in a folder
  python main.py --webcam             Run real-time webcam detection
"""

import os
import sys
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from ultralytics import YOLO
from models import get_model_by_name

# ─── CONFIG ──────────────────────────────────────────────────────────────────
DETECTION_MODEL = os.path.join("detection", "yolov8_model.pt")
SAVED_DIR = os.path.join("models", "saved")
IMG_SIZE = (224, 224)
DETECTION_CONF = 0.4

FRUIT_CLASSES = {0: "apple", 1: "banana", 2: "strawberry"}

TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class FruitFreshnessDetector:
    """2-stage pipeline: YOLO detection → CNN freshness classification."""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Stage 1: YOLO
        if not os.path.exists(DETECTION_MODEL):
            print(f"ERROR: Detection model not found at {DETECTION_MODEL}")
            print("Run train_yolo.py first.")
            sys.exit(1)
        self.yolo = YOLO(DETECTION_MODEL)
        print(f"  YOLO model loaded from {DETECTION_MODEL}")

        # Stage 2: Per-fruit freshness CNNs
        self.freshness_models = {}
        self.class_names = {}
        for fruit in ["apple", "banana", "strawberry"]:
            model_path = os.path.join(SAVED_DIR, fruit, f"{fruit}_best.pth")
            if not os.path.exists(model_path):
                print(f"  WARNING: No best model for {fruit} at {model_path}")
                continue

            checkpoint = torch.load(model_path, map_location=self.device)
            arch = checkpoint.get('architecture', 'resnet')
            model = get_model_by_name(arch, num_classes=1, pretrained=False, freeze_backbone=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(self.device)
            model.eval()

            self.freshness_models[fruit] = model
            self.class_names[fruit] = checkpoint.get('class_names', ['fresh', 'rotten'])
            print(f"  Loaded {fruit} freshness model ({arch})")

    def detect_fruits(self, image):
        """Run YOLO detection, return list of {bbox, fruit, detection_conf}."""
        results = self.yolo(image, conf=DETECTION_CONF, verbose=False)
        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                fruit = FRUIT_CLASSES.get(class_id, "unknown")
                detections.append({"bbox": (x1, y1, x2, y2), "fruit": fruit, "detection_conf": conf})
        return detections

    def classify_freshness(self, crop_bgr, fruit):
        """Classify a BGR crop as fresh/rotten."""
        if fruit not in self.freshness_models:
            return "unknown", 0.0

        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor = TRANSFORM(crop_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.freshness_models[fruit](tensor)
            prob = output.item() if output.dim() <= 1 else output.squeeze().item()

        # Class mapping depends on ImageFolder order (alphabetical): fresh=0, rotten=1
        # But our models use sigmoid where >0.5 = class 1
        classes = self.class_names.get(fruit, ['fresh', 'rotten'])
        if prob > 0.5:
            label = classes[1] if len(classes) > 1 else "rotten"
            confidence = prob
        else:
            label = classes[0]
            confidence = 1.0 - prob

        # Normalize label
        label = "Fresh" if "fresh" in label.lower() else "Rotten"
        return label, float(confidence)

    def predict(self, image):
        """Full pipeline: detect → crop → classify."""
        detections = self.detect_fruits(image)
        results = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            freshness, conf = self.classify_freshness(crop, det["fruit"])
            results.append({
                "fruit": det["fruit"],
                "freshness": freshness,
                "confidence": conf,
                "detection_conf": det["detection_conf"],
                "bbox": det["bbox"],
            })
        return results

    def visualize(self, image, results):
        """Draw boxes and labels on image."""
        output = image.copy()
        colors = {"Fresh": (0, 200, 0), "Rotten": (0, 0, 200), "unknown": (200, 200, 0)}

        for r in results:
            x1, y1, x2, y2 = r["bbox"]
            color = colors.get(r["freshness"], (200, 200, 0))
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

            label = f"{r['fruit']}: {r['freshness']} ({r['confidence']:.0%})"
            sz = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(output, (x1, y1 - sz[1] - 10), (x1 + sz[0], y1), color, -1)
            cv2.putText(output, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return output


def process_image(detector, path):
    image = cv2.imread(path)
    if image is None:
        print(f"ERROR: Cannot read {path}")
        return
    results = detector.predict(image)
    print(f"\n{path}:")
    if not results:
        print("  No fruits detected.")
    for r in results:
        print(f"  {r['fruit'].capitalize()}: {r['freshness']} "
              f"(conf={r['confidence']:.0%}, det={r['detection_conf']:.0%})")
    output = detector.visualize(image, results)
    out_path = path.rsplit(".", 1)[0] + "_result.jpg"
    cv2.imwrite(out_path, output)
    print(f"  Saved: {out_path}")


def process_folder(detector, folder):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    images = [os.path.join(folder, f) for f in sorted(os.listdir(folder))
              if f.lower().endswith(exts)]
    print(f"Processing {len(images)} images from {folder}")
    for p in images:
        process_image(detector, p)


def run_webcam(detector):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam")
        return
    print("Webcam started. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = detector.predict(frame)
        output = detector.visualize(frame, results)
        cv2.imshow("Fruit Freshness Detector", output)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    print("=" * 50)
    print("  Fruit Freshness Detection System")
    print("=" * 50)

    detector = FruitFreshnessDetector()

    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python main.py <image_path>")
        print("  python main.py <folder_path>")
        print("  python main.py --webcam")
        return

    target = sys.argv[1]
    if target == "--webcam":
        run_webcam(detector)
    elif os.path.isdir(target):
        process_folder(detector, target)
    elif os.path.isfile(target):
        process_image(detector, target)
    else:
        print(f"ERROR: {target} not found")


if __name__ == "__main__":
    main()
