PYTHON := python

install:
	$(PYTHON) -m pip install -r requirements.txt

test:
	pytest -q

format:
	pre-commit run --all-files

run:
	uvicorn src.eta_routing.serving.app:app --reload

train:
	$(PYTHON) -m src.eta_routing.models.train_model --data-path data/sample/synthetic_eta_data.csv --output models/model.pkl

.PHONY: install test format run train