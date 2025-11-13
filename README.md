# ETA Prediction & Dynamic Routing

This repository contains a minimal, production‑ready project for building an AI‑powered service that predicts estimated times of arrival (ETAs) for delivery routes and generates dynamic, optimized routes.  The goal is to reduce empty miles and driver mileage while improving on‑time delivery rates, following the project specification approved in Phase 2.

## Problem

Logistics companies often struggle with inaccurate travel‑time estimates and static route plans that cannot adapt to changing conditions.  Recent research shows that AI‑based rerouting tools can reduce empty miles by **64 %** and driver mileage by **23 %**【115101397648578†L330-L341】.  This project trains machine‑learning models on trip and contextual data (time of day, distance, weather, traffic) to predict trip durations in real time and uses the predictions to suggest better routes.

## Repository structure

```
eta_routing/
├── README.md                 # this file
├── LICENSE                   # MIT licence (modify if you prefer another)
├── CONTRIBUTING.md           # guidelines for contributing
├── CODE_OF_CONDUCT.md        # standard code of conduct
├── requirements.txt          # Python dependencies
├── Makefile                  # convenience commands for setup/test
├── pyproject.toml            # optional project metadata for Poetry users
├── .pre-commit-config.yaml   # formatting and linting hooks
├── .gitignore                # files to ignore in git
├── data/
│   └── sample/
│       └── synthetic_eta_data.csv  # small synthetic dataset for quick experiments
├── src/
│   └── eta_routing/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   └── dataset_loader.py   # load real or synthetic data
│       ├── features/
│       │   ├── __init__.py
│       │   └── feature_engineering.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── train_model.py
│       │   └── predict_model.py
│       └── serving/
│           ├── __init__.py
│           └── app.py             # FastAPI service
├── tests/
│   ├── test_data_valid.py
│   ├── test_features.py
│   ├── test_models.py
│   └── test_api.py
└── infra/
    ├── docker/
    │   └── Dockerfile
    └── github/
        └── workflows/
            └── ci.yml
```

## Quickstart

First, create and activate a virtual environment (Python 3.11+).  Install dependencies using pip:

```bash
pip install -r requirements.txt
```

To experiment with the provided synthetic dataset and train a baseline model:

```bash
python -m src.eta_routing.models.train_model --data-path data/sample/synthetic_eta_data.csv --output models/model.pkl
```

To start the API locally after training:

```bash
uvicorn src.eta_routing.serving.app:app --reload
```

Open `http://localhost:8000/docs` to view interactive API documentation.

## Project highlights

- **Real‑time ETA prediction:** trains a regression model on trip data and contextual features.
- **Dynamic routing:** uses predicted travel times to evaluate candidate routes (basic heuristic implementation provided; can be replaced with OR‑Tools or reinforcement learning later).
- **MLOps ready:** includes unit tests, data validation checks, CI workflow and Dockerfile for reproducible deployment.
- **Extensible:** you can switch from the synthetic dataset to a real one (e.g., NYC taxi trip data) by dropping the file into `data/` and updating `dataset_loader.py` accordingly.

Please see `CONTRIBUTING.md` for details on how to propose enhancements or report issues.