# Phase 7: Backend API Development

## Objectives
- Serve the trained model through a production-ready REST API.

## Tech Stack
- **FastAPI** (async, auto-generated OpenAPI docs)
- **Uvicorn/Gunicorn** as ASGI server
- **Redis + Celery** (optional) for async video processing queue

## API Structure
```
src/api/
├── main.py
├── routes/
│   ├── detect.py
│   └── history.py
├── services/
│   ├── preprocessing_service.py
│   └── inference_service.py
├── models/
│   └── schemas.py
└── utils/
```

## Sample FastAPI Endpoint
```python
from fastapi import FastAPI, UploadFile, File
from src.api.services.inference_service import run_detection

app = FastAPI(title="Deepfake Detection API")

@app.post("/api/v1/detect/video")
async def detect_video(file: UploadFile = File(...)):
    contents = await file.read()
    result = run_detection(contents, media_type="video")
    return {
        "filename": file.filename,
        "result": result["label"],
        "confidence": result["confidence"],
        "frames_analyzed": result["frames_analyzed"]
    }

@app.get("/api/v1/health")
async def health_check():
    return {"status": "ok"}
```

## Inference Service Wrapper
```python
def run_detection(file_bytes, media_type):
    frames = extract_frames_from_bytes(file_bytes) if media_type == "video" else [decode_image(file_bytes)]
    faces = [extract_face(f) for f in frames if extract_face(f) is not None]
    preds = [predict(preprocess(f)) for f in faces]
    fake_ratio = sum(1 for label, _ in preds if label == "FAKE") / len(preds)
    final_label = "FAKE" if fake_ratio > 0.5 else "REAL"
    avg_conf = sum(c for _, c in preds) / len(preds)
    return {"label": final_label, "confidence": avg_conf, "frames_analyzed": len(preds)}
```

## Performance Considerations
- Load model once at startup (not per-request) using FastAPI's `lifespan` events.
- Use ONNX Runtime or TensorRT for faster inference in production.
- Add request queueing (Celery + Redis) for long videos to avoid blocking.

## Security
- Validate file type/size (max 100MB, only mp4/jpg/png)
- Rate limiting (e.g., `slowapi`)
- API key/auth for external clients

## Deliverables
- Fully working FastAPI service in `src/api/`
- OpenAPI docs auto-available at `/docs`
- Dockerfile for the API service
