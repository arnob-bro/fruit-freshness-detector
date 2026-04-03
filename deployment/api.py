"""
REST API for 2-Stage Fruit Freshness Detection
FastAPI-based API: YOLO detection → per-fruit CNN freshness classification

Run with: uvicorn deployment.api:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from main import FruitFreshnessDetector

app = FastAPI(
    title="Fruit Freshness Detection API",
    description="2-Stage pipeline: YOLOv8 detection + per-fruit CNN freshness classification",
    version="2.0.0"
)

detector = None


@app.on_event("startup")
async def startup_event():
    global detector
    try:
        detector = FruitFreshnessDetector()
        print("Detector loaded successfully")
    except Exception as e:
        print(f"WARNING: Could not load detector: {e}")


@app.get("/")
async def root():
    return {
        "message": "Fruit Freshness Detection API",
        "version": "2.0.0",
        "pipeline": "YOLOv8 → per-fruit CNN",
        "model_loaded": detector is not None,
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": detector is not None,
        "fruits_available": list(detector.freshness_models.keys()) if detector else [],
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Upload a fruit image. Returns detected fruits with freshness classification.
    """
    if detector is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")

        results = detector.predict(image)

        return JSONResponse(content={
            "detections": [
                {
                    "fruit": r["fruit"],
                    "freshness": r["freshness"],
                    "freshness_confidence": round(r["confidence"], 4),
                    "detection_confidence": round(r["detection_conf"], 4),
                    "bbox": list(r["bbox"]),
                }
                for r in results
            ],
            "count": len(results),
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/model_info")
async def model_info():
    if detector is None:
        return {"model_loaded": False}

    return {
        "model_loaded": True,
        "detection_model": "YOLOv8n",
        "fruits": list(detector.freshness_models.keys()),
        "pipeline": "YOLO detection → per-fruit best CNN → Fresh/Rotten",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
