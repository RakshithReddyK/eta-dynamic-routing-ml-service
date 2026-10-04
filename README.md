# ETA Prediction Service — unfinished scaffold

This repository contains data-loading and feature-engineering code plus a FastAPI interface for an ETA prediction experiment. **The checked-in repository is not currently runnable end to end:** the model training and prediction modules imported by the service/tests are missing.

The previous README described a complete trained service and a validation MAE around 2.8 minutes. The current tree does not include the training implementation or evaluation artifact needed to reproduce that number. It should not be presented as a measured result.

## What exists

| Component | Location |
|---|---|
| Synthetic sample trips | `data/sample/synthetic_eta_data.csv` |
| Dataset loader | `src/eta_routing/data/dataset_loader.py` |
| Feature engineering | `src/eta_routing/features/feature_engineering.py` |
| API interface | `src/eta_routing/serving/app.py` |
| Test definitions | `tests/` |
| Dockerfile | `infra/docker/Dockerfile` |
| Inactive CI template | `infra/github/workflows/ci.yml` |

The CI template is outside `.github/workflows/`, so it is not an active GitHub Actions workflow. Virtual-environment files also appear in the tracked tree and should be removed in a dedicated cleanup after verifying a clean dependency install.

## Work required before using this as a system demo

1. Implement the missing training/prediction modules and save a reproducible model artifact.
2. Validate data and split it before fitting any learned preprocessing.
3. Compare regression error against a simple baseline; commit the evaluation report and dataset provenance.
4. Load the model through the API and test missing-model, invalid-input, and inference-failure behavior.
5. Activate CI, reproduce a clean install, and verify the Docker image.
6. Add measured HTTP latency and a model card before claiming deployment readiness.

Dynamic routing, on-time delivery improvements, SLA reductions, and production latency are not demonstrated by this checked-in implementation. Synthetic-data outcomes would not establish business impact without a separate real-world evaluation.

The October 2026 portfolio review was a source audit only for this repository. For the currently exercised serving and retrieval projects, see [Fraud Detection ML](https://github.com/RakshithReddyK/fraud-detection-ML) and [RAG Knowledge Assistant](https://github.com/RakshithReddyK/rag-knowledge-assistant).
