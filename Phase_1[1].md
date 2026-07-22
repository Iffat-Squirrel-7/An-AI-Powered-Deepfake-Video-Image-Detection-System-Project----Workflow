# Phase 1: Requirement Analysis & Research

## Objectives
- Define exact scope: which types of deepfakes to detect (face-swap, face-reenactment, GAN-generated full faces, lip-sync/audio-visual mismatch).
- Study existing literature and SOTA approaches (XceptionNet baseline from FaceForensics++, EfficientNet-based detectors, ViT-based detectors, frequency-domain detectors).
- Identify target users (journalists, social media platforms, general public, researchers).
- Define success metrics (accuracy, AUC, F1, false-positive rate — critical to keep FP low to avoid falsely flagging real content).

## Key Tasks
1. Literature review of deepfake detection papers (2019–2026 SOTA).
2. Competitive analysis of existing tools (Microsoft Video Authenticator, Deepware, Sensity AI).
3. Define functional requirements:
   - Upload image/video
   - Return Real/Fake label + confidence
   - Explainability (heatmap)
   - History/dashboard
4. Define non-functional requirements: latency, scalability, accuracy threshold, privacy.
5. Choose primary frameworks: **TensorFlow** for production-grade serving, **PyTorch** for research/rapid prototyping (or pick one as primary — document decision).

## Deliverables
- Project charter document
- Literature review summary
- Functional & non-functional requirements list
- Feasibility report (compute budget, GPU availability, dataset access)

## Tools
- Python 3.10+, Jupyter/Colab for research notebooks
- Reference papers stored in `/docs/literature/`
