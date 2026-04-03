"""
Streamlit Web Application for 2-Stage Fruit Freshness Detection
Stage 1: YOLO detects fruit type
Stage 2: Per-fruit CNN classifies freshness

Run with: streamlit run deployment/app.py
"""

import streamlit as st
import cv2
import numpy as np
import sys
from pathlib import Path
from PIL import Image

sys.path.append(str(Path(__file__).parent.parent))

from main import FruitFreshnessDetector


@st.cache_resource
def load_detector():
    return FruitFreshnessDetector()


def main():
    st.set_page_config(page_title="Fruit Freshness Detector", page_icon="🍎", layout="wide")
    st.title("🍎 Fruit Freshness Detection System")
    st.markdown(
        "Upload a fruit image → **YOLOv8** detects the fruit type → "
        "**Fruit-specific CNN** classifies freshness (Fresh / Rotten)"
    )
    st.markdown("---")

    # Sidebar
    st.sidebar.header("Configuration")
    conf_threshold = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.4, 0.05)
    show_details = st.sidebar.checkbox("Show detection details", value=True)

    # Load detector
    try:
        detector = load_detector()
        st.sidebar.success("Models loaded!")
    except Exception as e:
        st.sidebar.error(f"Error loading models: {e}")
        st.stop()

    # Upload
    uploaded = st.file_uploader("Upload a fruit image", type=["jpg", "jpeg", "png", "webp", "bmp"])

    if uploaded is not None:
        file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Override detection confidence
        detector.yolo.overrides['conf'] = conf_threshold

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Input")
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)

        with st.spinner("Detecting..."):
            results = detector.predict(image)
            output = detector.visualize(image, results)

        with col2:
            st.subheader("Result")
            st.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB), use_container_width=True)

        if results:
            st.subheader("Detections")
            for i, r in enumerate(results):
                emoji = "✅" if r["freshness"] == "Fresh" else "❌"
                st.markdown(
                    f"**{i+1}. {r['fruit'].capitalize()}** — "
                    f"{emoji} **{r['freshness']}** "
                    f"(Freshness confidence: {r['confidence']:.1%})"
                )
                if show_details:
                    st.caption(f"Detection confidence: {r['detection_conf']:.1%} | "
                               f"Bounding box: {r['bbox']}")
        else:
            st.warning("No fruits detected. Try a different image or lower the confidence threshold.")

    st.markdown("---")
    st.caption(
        "Pipeline: YOLOv8 (fruit detection) → Transfer Learning CNNs "
        "(MobileNetV2, ResNet50, EfficientNetB0, VGG16, DenseNet121)"
    )


if __name__ == '__main__':
    main()
