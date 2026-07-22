# Complete Data Flow

## 1. Training-Time Data Flow

```mermaid
flowchart TD
    A[Raw Datasets<br/>FaceForensics++, Celeb-DF, DFDC] --> B[Data Ingestion<br/>Download & verify checksums]
    B --> C[Frame Extraction<br/>per video → N frames]
    C --> D[Face Detection & Cropping<br/>MTCNN / MediaPipe]
    D --> E[Labeling<br/>Real=0 / Fake=1]
    E --> F[Train / Val / Test Split<br/>70 / 15 / 15]
    F --> G[Data Augmentation<br/>flip, rotation, compression, jitter]
    G --> H[TFRecord / PyTorch Dataset<br/>& DataLoader creation]
    H --> I[Model Training Loop<br/>TensorFlow or PyTorch]
    I --> J[Checkpointing & Tracking<br/>MLflow / W&B]
    J --> K[Best Model Selection<br/>based on validation AUC]
    K --> L[Export Model<br/>.h5 / .pt / ONNX]
```

## 2. Inference-Time Data Flow

```mermaid
flowchart TD
    A[User Upload<br/>Video / Image] --> B[FastAPI<br/>receives & validates file]
    B --> C[Preprocessing Service<br/>frame extraction, face detection]
    C --> D[Inference Service<br/>loads exported model, runs prediction]
    D --> E[Post-processing<br/>aggregation, Grad-CAM heatmap]
    E --> F[(Database<br/>PostgreSQL / MongoDB)]
    E --> G[JSON Response<br/>returned to Client]
    F --> H[Dashboard<br/>reads DB, updates analytics]
```

## 3. Data Versioning

- Use **DVC (Data Version Control)** to track dataset versions and preprocessing pipeline outputs alongside Git.
- Tag each trained model with the dataset version + hyperparameters used, stored in `configs/model_registry.yaml`.

## 4. Data Governance Notes

- No personally identifiable raw footage should be stored beyond what's needed for evaluation.
- All datasets must be used respecting their original licenses (e.g., FaceForensics++ requires signed agreement).
