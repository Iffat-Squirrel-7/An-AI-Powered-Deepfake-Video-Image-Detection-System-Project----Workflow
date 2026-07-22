# Phase 3: Data Preprocessing Pipeline

## Objectives
- Convert raw videos/images into clean, aligned face crops ready for model training.

## Key Tasks

### 1. Frame Sampling
- Extract N frames per video (e.g., every 5th frame or fixed FPS=5) to balance dataset size vs. training time.

### 2. Face Detection & Cropping
- Use MTCNN, MediaPipe Face Mesh, or dlib HOG detector.
- Crop with margin (~20%) around detected bounding box to preserve context.

### 3. Face Alignment
- Align using eye/nose landmarks so all faces share consistent orientation.

### 4. Resizing & Normalization
- Resize to model input size: 299x299 (Xception) or 224x224 (EfficientNet/ResNet).
- Normalize pixel values to [0,1] or use ImageNet mean/std normalization.

### 5. Data Augmentation
- Random horizontal flip
- Random brightness/contrast jitter
- JPEG compression artifact simulation (deepfakes often show compression differences)
- Random cropping / slight rotation

```python
import albumentations as A

train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.ImageCompression(quality_lower=60, quality_upper=100, p=0.3),
    A.Rotate(limit=10, p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])
```

### 6. TFRecord / PyTorch Dataset Creation

**TensorFlow (tf.data)**
```python
def load_and_preprocess(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [299, 299]) / 255.0
    return img, label

dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))
dataset = dataset.map(load_and_preprocess).batch(32).prefetch(tf.data.AUTOTUNE)
```

**PyTorch (Dataset class)**
```python
from torch.utils.data import Dataset
from PIL import Image

class DeepfakeDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['filepath']).convert('RGB')
        if self.transform:
            img = self.transform(image=np.array(img))['image']
        return img, row['label']
```

## Deliverables
- Preprocessed face-crop dataset saved to `data/processed/`
- Reusable preprocessing module in `src/preprocessing/`
- Augmentation config file `configs/augmentation.yaml`
