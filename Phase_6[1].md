# Phase 6: Evaluation & Metrics

## Objectives
- Rigorously evaluate model performance and generalization.

## Key Metrics
| Metric | Why It Matters |
|---|---|
| **Accuracy** | Overall correctness (can be misleading with imbalance) |
| **AUC-ROC** | Standard metric for deepfake detection papers; threshold-independent |
| **Precision** | Of predicted "fake", how many are truly fake (avoid false accusations) |
| **Recall** | Of all true fakes, how many were caught |
| **F1-score** | Balance of precision & recall |
| **Confusion Matrix** | Visualize error types |
| **EER (Equal Error Rate)** | Common in biometric/deepfake literature |

## Evaluation Script (TensorFlow)
```python
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

y_pred_prob = model.predict(test_dataset)
y_pred = (y_pred_prob > 0.5).astype(int)

print(classification_report(y_true, y_pred))
print("AUC:", roc_auc_score(y_true, y_pred_prob))
print(confusion_matrix(y_true, y_pred))
```

## Cross-Dataset Generalization Testing
- Train on FaceForensics++, **test on Celeb-DF** (and vice versa) to measure real-world generalization — a model that only performs well on its training dataset is not production-ready.

## Robustness Testing
- Test against compressed videos (low bitrate re-encoding)
- Test against different resolutions
- Test against adversarially perturbed inputs (basic robustness check)

## Explainability Validation
- Generate Grad-CAM heatmaps and manually verify they highlight plausible manipulation regions (mouth, eyes, blending boundary) rather than irrelevant background.

## Deliverables
- `src/evaluation/evaluate.py`
- Evaluation report with metrics table, confusion matrix, ROC curve plots
- Cross-dataset generalization report
