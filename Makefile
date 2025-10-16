    .PHONY: venv setup lint test train predict clean help

    VENV := .venv
    PY := $(VENV)/bin/python
    PIP := $(VENV)/bin/pip

    venv:
	@test -d $(VENV) || python3 -m venv $(VENV)
	@$(PY) -m pip install -U pip setuptools wheel

    setup: venv
	$(PIP) install -e .

    lint:
	$(PIP) install ruff mypy >/dev/null 2>&1 || true
	$(VENV)/bin/ruff check .
	$(VENV)/bin/mypy src
	$(VENV)/bin/ruff format --check .
	$(VENV)/bin/ruff format .
	$(VENV)/bin/ruff check --fix .

    train:
	$(PY) scripts/train.py --config configs/imbalance.yaml

    predict:
	$(PY) scripts/predict.py --config configs/imbalance.yaml --input artifacts/test.csv --out artifacts/preds.csv
	@echo 'Predictions saved to artifacts/preds.csv'

    test:
	$(PIP) install pytest >/dev/null 2>&1 || true
	$(VENV)/bin/pytest -q

    clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache .mypy_cache artifacts

    help:
	@echo 'Targets:'
	@echo '  venv      - create/update .venv'
	@echo '  setup     - install package in editable mode into .venv'
	@echo '  lint      - run ruff & mypy (using .venv)'
	@echo '  train     - train with configs/imbalance.yaml'
	@echo '  predict   - run prediction on artifacts/test.csv -> artifacts/preds.csv'
	@echo '  test      - run pytest'
	@echo '  clean     - remove caches and artifacts'
