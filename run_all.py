"""
Master script - runs the entire pipeline end-to-end:
  1. Prepare dataset (splits + YOLO annotations)
  2. Train YOLOv8 fruit detection model
  3. Train 15 freshness CNN models (5 architectures x 3 fruits)
  4. Evaluate all models and select best per fruit

Usage: python run_all.py
"""

import subprocess
import sys
import time


def run_step(name, script):
    print(f"\n{'='*60}")
    print(f"  STEP: {name}")
    print(f"{'='*60}\n")
    start = time.time()
    result = subprocess.run([sys.executable, script])
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"\n  ERROR: {name} failed (exit code {result.returncode})")
        print(f"  Fix the issue and re-run: python {script}")
        sys.exit(1)
    print(f"\n  Completed in {elapsed:.1f}s")


def main():
    print("=" * 60)
    print("  FRUIT FRESHNESS DETECTION - FULL PIPELINE")
    print("=" * 60)

    total_start = time.time()

    run_step("1. Prepare Dataset", "prepare_dataset.py")
    run_step("2. Train YOLOv8 Detection Model", "train_yolo.py")
    run_step("3. Train Freshness CNN Models (15 models)", "training/train.py")
    run_step("4. Evaluate All Models", "evaluation/evaluate.py")

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  ALL DONE! Total time: {total_elapsed/60:.1f} minutes")
    print(f"{'='*60}")
    print(f"\nNext steps:")
    print(f"  python main.py <image_path>        # Inference on image")
    print(f"  python main.py --webcam             # Webcam mode")
    print(f"  streamlit run deployment/app.py     # Web UI")
    print(f"  uvicorn deployment.api:app           # REST API")


if __name__ == "__main__":
    main()
