# Phase 4: Model Architecture Design

## Objectives
- Select and design the deep learning architecture(s) for deepfake classification.

## Candidate Architectures

| Model | Pros | Notes |
|---|---|---|
| **XceptionNet** | Strong FaceForensics++ baseline, good accuracy/speed trade-off | Most widely used baseline in papers |
| **EfficientNet-B0/B4** | Excellent accuracy per parameter, scalable | Good for edge deployment (B0) |
| **ResNet50** | Well-understood, transfer-learning friendly | Solid baseline |
| **Vision Transformer (ViT)** | Captures global patterns, strong on large datasets | Needs more data/compute |
| **Two-stream network** (RGB + frequency domain via DCT/FFT) | Captures GAN fingerprint artifacts in frequency space | More complex, higher accuracy on unseen manipulations |

## Recommended Approach
Start with **transfer learning** using a pretrained XceptionNet or EfficientNet (ImageNet weights), fine-tuned on deepfake datasets — this is the fastest path to a strong baseline.

## TensorFlow/Keras Example (XceptionNet)
```python
from tensorflow.keras.applications import Xception
from tensorflow.keras import layers, models

base_model = Xception(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
base_model.trainable = False  # freeze initially

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # binary: real vs fake
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC()])
```

## PyTorch Example (EfficientNet via timm)
```python
import timm
import torch.nn as nn

model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=1)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
```

## Model Design Decisions to Document
- Input resolution (299x299 vs 224x224)
- Freeze vs fine-tune backbone layers
- Single-frame vs sequence-based (LSTM/3D-CNN on frame sequences) architecture
- Loss function: Binary Cross-Entropy (simple) vs Focal Loss (handles class imbalance better)

## Deliverables
- `src/models/xception_model.py`
- `src/models/efficientnet_model.py`
- Model architecture diagram in `docs/model_architecture.png`
- Config file specifying chosen architecture: `configs/model_config.yaml`
