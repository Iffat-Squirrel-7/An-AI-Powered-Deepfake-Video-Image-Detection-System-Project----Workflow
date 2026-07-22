# Phase 10: Deployment (Cloud & Edge / Raspberry Pi Integration)

## Objectives
- Package and deploy the system for real-world use, including optional lightweight edge deployment.

## A. Standard Cloud Deployment

### Dockerize Services
```dockerfile
# Dockerfile (API service)
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose (API + DB + Dashboard)
```yaml
version: '3.9'
services:
  api:
    build: ./api
    ports:
      - "8000:8000"
    depends_on:
      - db
  db:
    image: postgres:15
    environment:
      POSTGRES_DB: deepfake_db
      POSTGRES_PASSWORD: secret
    volumes:
      - pgdata:/var/lib/postgresql/data
  dashboard:
    build: ./dashboard
    ports:
      - "8501:8501"
volumes:
  pgdata:
```

### Cloud Options
| Provider | Suggested Service |
|---|---|
| AWS | ECS/Fargate or EC2 + GPU (g4dn instance) for inference |
| GCP | Cloud Run (CPU-only lightweight model) or GKE with GPU node |
| Azure | Azure Container Apps / AKS |

### CI/CD (GitHub Actions example)
```yaml
name: Deploy
on:
  push:
    branches: [main]
jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build Docker image
        run: docker build -t deepfake-api .
      - name: Push to registry
        run: |
          docker tag deepfake-api ghcr.io/yourorg/deepfake-api:latest
          docker push ghcr.io/yourorg/deepfake-api:latest
```

## B. Edge Deployment (Raspberry Pi / Jetson Nano) — Optional

Use case: a lightweight, offline field-deployable detector (e.g., for on-device screening).

### Steps
1. **Convert model to TFLite** for efficient edge inference:
```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open('model_quantized.tflite', 'wb') as f:
    f.write(tflite_model)
```
2. Use a smaller backbone (EfficientNet-B0 or MobileNetV3) for edge — full Xception is too heavy for Raspberry Pi CPU.
3. Install lightweight runtime on the Pi:
```bash
pip install tflite-runtime opencv-python-headless
```
4. Run inference locally:
```python
import tflite_runtime.interpreter as tflite

interpreter = tflite.Interpreter(model_path="model_quantized.tflite")
interpreter.allocate_tensors()
# ... set input tensor, invoke(), get output tensor
```
5. Optional: use Raspberry Pi Camera Module for real-time frame capture and on-device screening, sending only flagged results to the cloud API for double-checking with the full model.

### Performance Expectations on Raspberry Pi 4
- MobileNetV3/EfficientNet-B0 (quantized): ~150–300ms per frame on CPU — acceptable for near-real-time screening, not full 30fps video.

## Deliverables
- `Dockerfile`, `docker-compose.yml`
- CI/CD pipeline config (`.github/workflows/deploy.yml`)
- TFLite quantized model + Raspberry Pi inference script (`edge/pi_inference.py`)
- Deployment runbook document
