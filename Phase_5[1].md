# Phase 5: Model Training Pipeline

## Objectives
- Build a robust, reproducible training pipeline in TensorFlow and/or PyTorch.

## Key Components

### 1. Training Configuration (YAML-driven)
```yaml
# configs/xception_config.yaml
model: xception
input_size: 299
batch_size: 32
epochs: 30
learning_rate: 0.0001
optimizer: adam
loss: binary_crossentropy
early_stopping_patience: 5
```

### 2. TensorFlow Training Loop
```python
callbacks = [
    tf.keras.callbacks.ModelCheckpoint('models/best_model.h5', save_best_only=True, monitor='val_auc', mode='max'),
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5),
    tf.keras.callbacks.TensorBoard(log_dir='logs/')
]

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=30,
    callbacks=callbacks
)
```

### 3. PyTorch Training Loop
```python
for epoch in range(epochs):
    model.train()
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.float().to(device)
        optimizer.zero_grad()
        outputs = model(imgs).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    model.eval()
    val_loss, val_acc = evaluate(model, val_loader)
    print(f"Epoch {epoch+1}: val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
    scheduler.step(val_loss)
```

### 4. Two-Phase Fine-Tuning Strategy
1. **Phase A:** Freeze backbone, train only classification head (5–10 epochs).
2. **Phase B:** Unfreeze last N layers of backbone, fine-tune with lower learning rate (1e-5).

### 5. Handling Class Imbalance
- Use `class_weight` in Keras `fit()` or weighted `BCEWithLogitsLoss` (pos_weight) in PyTorch.
- Alternative: Focal Loss to focus on hard examples.

### 6. Experiment Tracking
- Log every run to **MLflow** or **Weights & Biases**: hyperparameters, metrics, model artifacts.
```python
import mlflow
mlflow.start_run()
mlflow.log_params({"model": "xception", "lr": 0.0001})
mlflow.log_metric("val_auc", 0.94)
mlflow.end_run()
```

## Deliverables
- `src/training/train_tf.py` and/or `src/training/train_torch.py`
- Trained model checkpoints in `models/`
- Training logs and MLflow/W&B experiment records
