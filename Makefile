ML_DIR := ml
VENV_DIR := .venv

ifeq ($(OS),Windows_NT)
	SHELL := cmd.exe
	.SHELLFLAGS := /C
	PYTHON ?= py -3
	VENV_PY := $(VENV_DIR)\Scripts\python.exe
	BIN := $(VENV_DIR)\Scripts
else
	SHELL := /bin/bash
	PYTHON ?= python3
	VENV_PY := $(VENV_DIR)/bin/python
	BIN := $(VENV_DIR)/bin
endif

.DEFAULT_GOAL := help

help:
	@echo Targets:
	@echo   venv            - cria venv e instala deps do ML (inclui dev: pytest/ruff)
	@echo   ml-test         - roda testes
	@echo   ml-augment      - roda augment (ARGS=...)
	@echo   ml-train        - roda treino (ARGS=...)
	@echo   ml-eval         - roda avaliacao (ARGS=...)
	@echo   ml-serve        - sobe API local (ARGS=...)
	@echo   docker-up       - sobe API com docker-compose
	@echo   docker-down     - derruba docker-compose
	@echo   docker-logs     - logs do servico
	@echo   format          - ruff format
	@echo   lint            - ruff check

venv:
ifeq ($(OS),Windows_NT)
	@if not exist "$(VENV_PY)" $(PYTHON) -m venv $(VENV_DIR)
else
	@test -d $(VENV_DIR) || $(PYTHON) -m venv $(VENV_DIR)
endif
	@"$(VENV_PY)" -m pip install -U pip
	@"$(VENV_PY)" -m pip install -e "./$(ML_DIR)[dev]"

ml-test: venv
	@"$(VENV_PY)" -m pytest -q $(ML_DIR)

ml-augment: venv
	@"$(BIN)\garden-ml-augment" $(ARGS)

ml-train: venv
	@"$(BIN)\garden-ml-train" $(ARGS)

ml-eval: venv
	@"$(BIN)\garden-ml-eval" $(ARGS)

ml-serve: venv
	@"$(BIN)\garden-ml-serve" $(ARGS)

format: venv
	@"$(VENV_PY)" -m ruff format $(ML_DIR)

lint: venv
	@"$(VENV_PY)" -m ruff check $(ML_DIR)

docker-up:
	@docker compose up -d --build

docker-down:
	@docker compose down

docker-logs:
	@docker compose logs -f --tail=200
