# Smart Tomato Garden — ML pipeline, Docker API, qualidade.
# Uso: make help | make venv | make ml-pipeline | make docker-run-all

ML_DIR := ml
VENV_DIR := .venv
COMPOSE ?= docker compose
DEV_SERVICE ?= garden-ml-dev
API_SERVICE ?= garden-ml-api
COMPARE_SERVICE ?= garden-ml-compare

# Local (host) — paths relativos à raiz do repo
LOCAL_DATASET_RAW ?= ml/dataset/plantvillage_tomato/raw
LOCAL_DATASET_AUG ?= ml/dataset/plantvillage_tomato/aug
LOCAL_ARTIFACTS_DIR ?= ml/artifacts/model_registry/v0004

# Docker (dentro do container, working_dir=/app/ml)
DATASET_RAW ?= dataset/plantvillage_tomato/raw
DATASET_AUG ?= dataset/plantvillage_tomato/aug
ARTIFACTS_DIR ?= artifacts/model_registry/v0004
ARTIFACTS_DIR_DL ?= artifacts/model_registry/v0004_dl

# Local (host) — paths para DL
LOCAL_ARTIFACTS_DIR_DL ?= ml/artifacts/model_registry/v0004_dl

# Pipeline
IMG_SIZE ?= 128
AUG_PER_IMAGE ?= 5
SEED ?= 42
TEST_SIZE ?= 0.30
CV_FOLDS ?= 5
MANIFEST ?= augmentation_manifest.csv

ifeq ($(OS),Windows_NT)
ifneq ($(strip $(WSL_DISTRO_NAME)$(WSL_INTEROP)),)
	SHELL := /bin/bash
	PYTHON ?= python3
	VENV_PY := $(VENV_DIR)/bin/python
	BIN := $(VENV_DIR)/bin
	SEP := /
else
	SHELL := cmd.exe
	.SHELLFLAGS := /C
	PYTHON ?= py -3
	VENV_PY := $(VENV_DIR)\Scripts\python.exe
	BIN := $(VENV_DIR)\Scripts
	SEP := \\
endif
else
	SHELL := /bin/bash
	PYTHON ?= python3
	VENV_PY := $(VENV_DIR)/bin/python
	BIN := $(VENV_DIR)/bin
	SEP := /
endif

.DEFAULT_GOAL := help
.PHONY := help venv install ml-test ml-augment ml-train ml-eval ml-serve \
	ml-augment-default ml-train-default ml-eval-default ml-pipeline ml-run-all \
	format lint clean \
	docker-build docker-build-api docker-build-dev docker-build-compare docker-build-all \
	docker-shell docker-sanity-raw docker-test \
	docker-clean-aug docker-augment docker-train docker-eval docker-deep docker-pipeline \
	docker-up docker-up-compare docker-down docker-logs docker-logs-compare \
	docker-run-all docker-run-compare \
	up down logs pipeline server stop vars

help:
	@echo "Smart Tomato Garden — Makefile"
	@echo ""
	@echo "  Fluxo completo (Docker):"
	@echo "    run-all, pipeline, server, stop   — pipeline + API em background"
	@echo "    up, down, logs                   — alias: docker-up, docker-down, docker-logs"
	@echo ""
	@echo "  ML local (venv):"
	@echo "    venv, install                    — cria .venv e instala deps (install = venv)"
	@echo "    ml-test                          — pytest em ml/"
	@echo "    ml-augment, ml-train, ml-eval    — com ARGS=\"...\" para customizar"
	@echo "    ml-augment-default, ml-train-default, ml-eval-default  — com defaults"
	@echo "    ml-pipeline                      — ml-test + augment + train + eval (defaults)"
	@echo "    ml-serve                         — sobe API local (foreground)"
	@echo "    ml-run-all                       — ml-pipeline + ml-serve"
	@echo ""
	@echo "  Docker:"
	@echo "    docker-build                     — build api + dev (sem compare)"
	@echo "    docker-build-compare             — build só imagem compare (com DL)"
	@echo "    docker-build-all                 — build api + dev + compare"
	@echo "    docker-build-api, docker-build-dev  — build só uma imagem"
	@echo "    docker-shell                     — shell no container dev (make docker-shell)"
	@echo "    docker-sanity-raw                — conta imagens no dataset RAW no container"
	@echo "    docker-test                      — pytest no container dev"
	@echo "    docker-clean-aug                 — remove pasta aug no container (antes de reroda)"
	@echo "    docker-augment, docker-train, docker-eval  — um passo no container"
	@echo "    docker-deep                      — treino CNN (MobileNetV3) no container dev"
	@echo "    docker-pipeline                  — sanity + test + augment + train + eval"
	@echo "    docker-up                        — sobe API clássica em background (porta 5000)"
	@echo "    docker-up-compare                — sobe API de comparação em background (porta 5001)"
	@echo "    docker-down                      — derruba stack (todos os serviços)"
	@echo "    docker-logs                      — logs da API clássica"
	@echo "    docker-logs-compare              — logs da API de comparação"
	@echo "    docker-run-all                   — docker-build + docker-pipeline + docker-up"
	@echo "    docker-run-compare               — docker-build-compare + docker-deep + docker-up-compare"
	@echo ""
	@echo "  Qualidade:"
	@echo "    format                           — ruff format ml/"
	@echo "    lint                             — ruff check ml/"
	@echo "    clean                            — remove __pycache__, .pytest_cache, .ruff_cache em ml/"
	@echo ""
	@echo "  Outros:"
	@echo "    vars                             — imprime variáveis para .env / override"
	@echo ""
	@echo "  Variáveis (override: make ml-train LOCAL_ARTIFACTS_DIR=ml/artifacts/model_registry/v0005):"
	@echo "    LOCAL_DATASET_RAW=$(LOCAL_DATASET_RAW)"
	@echo "    LOCAL_DATASET_AUG=$(LOCAL_DATASET_AUG)"
	@echo "    LOCAL_ARTIFACTS_DIR=$(LOCAL_ARTIFACTS_DIR)"
	@echo "    IMG_SIZE=$(IMG_SIZE) AUG_PER_IMAGE=$(AUG_PER_IMAGE) SEED=$(SEED) TEST_SIZE=$(TEST_SIZE) CV_FOLDS=$(CV_FOLDS)"

vars:
	@echo "# Colar em .env ou exportar para override"
	@echo "LOCAL_DATASET_RAW=$(LOCAL_DATASET_RAW)"
	@echo "LOCAL_DATASET_AUG=$(LOCAL_DATASET_AUG)"
	@echo "LOCAL_ARTIFACTS_DIR=$(LOCAL_ARTIFACTS_DIR)"
	@echo "GML_ARTIFACTS_DIR=/app/ml/artifacts/model_registry/v0004"
	@echo "GML_PORT=5000"

# --- Venv e ML local ---
venv:
ifeq ($(OS),Windows_NT)
	@if not exist "$(VENV_PY)" uv venv $(VENV_DIR)
else
	@test -d $(VENV_DIR) || uv venv $(VENV_DIR)
endif
	@uv pip install --python "$(VENV_PY)" -e "./$(ML_DIR)[dev,reports,tracking]"

install: venv

ml-test: venv
	@cd $(ML_DIR) && "$(VENV_PY)" -m pytest -q

ml-augment: venv
	@"$(BIN)$(SEP)garden-ml-augment" $(ARGS)

ml-train: venv
	@"$(BIN)$(SEP)garden-ml-train" $(ARGS)

ml-eval: venv
	@"$(BIN)$(SEP)garden-ml-eval" $(ARGS)

ml-serve: venv
	@"$(BIN)$(SEP)garden-ml-serve" --artifacts_dir "$(LOCAL_ARTIFACTS_DIR)" --host 0.0.0.0 --port 5000 $(ARGS)

ml-augment-default: venv
	@"$(BIN)$(SEP)garden-ml-augment" --input_dir "$(LOCAL_DATASET_RAW)" --output_dir "$(LOCAL_DATASET_AUG)" --img_size $(IMG_SIZE) --aug_per_image $(AUG_PER_IMAGE) --seed $(SEED) --segment_before_aug

ml-train-default: venv
	@"$(BIN)$(SEP)garden-ml-train" --dataset_dir "$(LOCAL_DATASET_AUG)" --output_dir "$(LOCAL_ARTIFACTS_DIR)" --img_size $(IMG_SIZE) --test_size $(TEST_SIZE) --seed $(SEED) --cv_folds $(CV_FOLDS) --manifest $(MANIFEST)

ml-eval-default: venv
	@"$(BIN)$(SEP)garden-ml-eval" --dataset_dir "$(LOCAL_DATASET_AUG)" --artifacts_dir "$(LOCAL_ARTIFACTS_DIR)" --img_size $(IMG_SIZE) --manifest $(MANIFEST)

ml-pipeline: ml-test ml-augment-default ml-train-default ml-eval-default

ml-run-all: ml-pipeline
	@"$(BIN)$(SEP)garden-ml-serve" --artifacts_dir "$(LOCAL_ARTIFACTS_DIR)" --host 0.0.0.0 --port 5000

# --- Qualidade ---
format: venv
	@"$(VENV_PY)" -m ruff format $(ML_DIR)

lint: venv
	@"$(VENV_PY)" -m ruff check $(ML_DIR)

clean:
	@$(PYTHON) -c "import shutil, pathlib; base=pathlib.Path('$(ML_DIR)'); [shutil.rmtree(p, ignore_errors=True) for p in list(base.rglob('__pycache__')) + [base/'.pytest_cache', base/'.ruff_cache'] if p.exists()]"
	@echo "clean done"

# --- Docker ---
docker-build:
	@$(COMPOSE) build $(API_SERVICE) $(DEV_SERVICE)

docker-build-api:
	@$(COMPOSE) build $(API_SERVICE)

docker-build-dev:
	@$(COMPOSE) build $(DEV_SERVICE)

docker-build-compare:
	@$(COMPOSE) --profile compare build $(COMPARE_SERVICE)

docker-build-all: docker-build docker-build-compare

docker-shell:
	@$(COMPOSE) run --rm $(DEV_SERVICE) bash

docker-sanity-raw:
	@$(COMPOSE) run --rm -T $(DEV_SERVICE) bash -lc "find '$(DATASET_RAW)' -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) 2>/dev/null | wc -l"

docker-test:
	@$(COMPOSE) run --rm -T $(DEV_SERVICE) bash -lc "pytest -vv -s tests --maxfail=1 --durations=20"

docker-augment:
	@$(COMPOSE) run --rm -T $(DEV_SERVICE) bash -lc "garden-ml-augment --input_dir '$(DATASET_RAW)' --output_dir '$(DATASET_AUG)' --img_size $(IMG_SIZE) --aug_per_image $(AUG_PER_IMAGE) --seed $(SEED) --segment_before_aug"

docker-train:
	@$(COMPOSE) run --rm -T $(DEV_SERVICE) bash -lc "garden-ml-train --dataset_dir '$(DATASET_AUG)' --output_dir '$(ARTIFACTS_DIR)' --img_size $(IMG_SIZE) --test_size $(TEST_SIZE) --seed $(SEED) --cv_folds $(CV_FOLDS) --manifest $(MANIFEST)"

docker-eval:
	@$(COMPOSE) run --rm -T $(DEV_SERVICE) bash -lc "garden-ml-eval --dataset_dir '$(DATASET_AUG)' --artifacts_dir '$(ARTIFACTS_DIR)' --img_size $(IMG_SIZE) --manifest $(MANIFEST)"

docker-deep:
	@$(COMPOSE) run --rm -T $(DEV_SERVICE) bash -lc "garden-ml-deep --dataset_dir '$(DATASET_AUG)' --output_dir '$(ARTIFACTS_DIR_DL)' --seed $(SEED)"

docker-clean-aug:
	@$(COMPOSE) run --rm -T $(DEV_SERVICE) bash -lc "rm -rf '$(DATASET_AUG)' && echo 'aug cleaned'"

docker-pipeline: docker-sanity-raw docker-test docker-augment docker-train docker-eval

docker-up:
	@$(COMPOSE) up -d $(API_SERVICE)

docker-up-compare:
	@$(COMPOSE) --profile compare up -d $(COMPARE_SERVICE)

docker-down:
	@$(COMPOSE) --profile compare down

docker-logs:
	@$(COMPOSE) logs -f --tail=200 $(API_SERVICE)

docker-logs-compare:
	@$(COMPOSE) logs -f --tail=200 $(COMPARE_SERVICE)

docker-run-all: docker-build docker-pipeline docker-up

docker-run-compare: docker-build-compare docker-deep docker-up-compare

# Aliases
run-all: docker-run-all
pipeline: docker-pipeline
server: docker-up
stop: docker-down
up: docker-up
down: docker-down
logs: docker-logs
