# Real-Time ETA Prediction & Dynamic Routing Service

End-to-end **ETA prediction service** for last-mile logistics.  
This project trains a regression model on trip data (synthetic + real-ready), exposes it via a **FastAPI** service, and is structured like a production ML system: clear metrics, modular code, tests, Docker, and CI hooks.

---

## Why this project exists

Modern logistics platforms (food delivery, ride-hailing, courier services) live or die on **ETA accuracy**:

- Overly optimistic ETAs → angry customers, SLA breaches, support load.
- Overly pessimistic ETAs → lost orders, lower conversion, wasted capacity.

This project demonstrates how to:

- Ingest trip data (CSV/Parquet),
- Engineer features and train a gradient-boosted ETA model,
- Serve low-latency ETAs via an HTTP API,
- Track **business metrics** and **model metrics** in a way that hiring managers actually care about.

---

## Business framing

**User:**  
- Operations / logistics team at a last-mile delivery platform.

**Decision locus:**
- Routing engine / assignment service deciding which driver/courier should take which order and what ETA to show to the customer.

**Example KPIs (business):**

- Reduce **ETA absolute error** from ~5–7 min → **≤3 min** (MAE).
- Increase **on-time delivery rate** by **+3–5%**.
- Reduce **SLA breach rate** (e.g., deliveries >10 min later than promised) by **30–40%**.

**Model & system metrics used here:**

- **MAE (Mean Absolute Error)** in minutes – primary metric.
- **RMSE** (optional) for tail sensitivity.
- **p95 latency** for `/predict` endpoint.
- **Data freshness** – how recent the training/eval data is.

On the sample synthetic dataset, the baseline model achieves:

- **Validation MAE ≈ 2.8 minutes** (synthetic data, reproducible).

---

## Architecture overview

High-level components:

- **Data layer**
  - `data/sample/synthetic_eta_data.csv` – synthetic trip data emulating city rides.
  - Pluggable loader to swap in real datasets (e.g., NYC Taxi, internal trip logs).

- **Feature & model layer**
  - Feature engineering: time features, geospatial distance, trip distance.
  - Model: **XGBoost Regressor** for ETA in minutes.
  - Train/test split + evaluation script.

- **Serving layer**
  - **FastAPI** app with `/predict` endpoint.
  - Returns predicted ETA duration and predicted dropoff timestamp.
  - Ready for containerization via Docker.

- **MLOps / DevEx**
  - **Tests** (Pytest): data loader, feature engineering, model training, API.
  - **CI Ready**: GitHub Actions workflow for lint/tests (just move it to `.github/workflows/ci.yml`).
  - **Dockerfile** for running the API as a container.

---

## Repository structure

```
eta_routing/
├── README.md                 # this file
├── LICENSE                   # MIT licence 
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
