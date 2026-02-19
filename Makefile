SHELL := /bin/bash

PYTHON ?= python3
ML_DIR := ml
VENV_DIR := .venv
PIP := $(VENV_DIR)/bin/pip
PY := $(VENV_DIR)/bin/python
BIN := $(VENV_DIR)/bin

.DEFAULT_GOAL := help

help:
	@echo "Targets:"
	@echo "  venv            -> cria venv e instala deps do ML (inclui dev: pytest/ruff)"
	@echo "  ml-test         -> roda testes"
	@echo "  ml-augment      -> roda augment (ARGS=...)"
	@echo "  ml-train        -> roda treino (ARGS=...)"
	@echo "  ml-eval         -> roda avaliação (ARGS=...)"
	@echo "  ml-serve        -> sobe API local (ARGS=...)"
	@echo "  docker-up       -> sobe API com docker-compose"
	@echo "  docker-down     -> derruba docker-compose"
	@echo "  docker-logs     -> logs do serviço"
	@echo "  format          -> ruff format"
	@echo "  lint            -> ruff check"
	@echo ""
	@echo "Exemplos:"
	@echo "  make ml-augment ARGS='--input_dir ../raw/dataset --output_dir ../raw/dataset_aug --img_size 128 --aug_per_image 5 --seed 42'"
	@echo "  make ml-train ARGS='--dataset_dir ../raw/dataset_aug --output_dir artifacts/model_registry/v0001 --img_size 128 --seed 42'"
	@echo "  make ml-serve ARGS='--artifacts_dir artifacts/model_registry/v0001 --host 0.0.0.0 --port 5000'"

venv:
	@test -d $(VENV_DIR) || $(PYTHON) -m venv $(VENV_DIR)
	@$(PIP) install -U pip
	@$(PIP) install -e "./$(ML_DIR)[dev]"

ml-test: venv
	@cd $(ML_DIR) && $(BIN)/pytest -q

ml-augment: venv
	@cd $(ML_DIR) && $(BIN)/garden-ml-augment $(ARGS)

ml-train: venv
	@cd $(ML_DIR) && $(BIN)/garden-ml-train $(ARGS)

ml-eval: venv
	@cd $(ML_DIR) && $(BIN)/garden-ml-eval $(ARGS)

ml-serve: venv
	@cd $(ML_DIR) && $(BIN)/garden-ml-serve $(ARGS)

format: venv
	@cd $(ML_DIR) && $(BIN)/ruff format .

lint: venv
	@cd $(ML_DIR) && $(BIN)/ruff check .

docker-up:
	@docker compose up -d --build

docker-down:
	@docker compose down

docker-logs:
	@docker compose logs -f --tail=200
