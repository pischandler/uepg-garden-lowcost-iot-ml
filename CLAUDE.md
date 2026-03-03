# Contexto do projeto — Smart Tomato Garden (UEPG)

Documento de contexto principal para assistentes e desenvolvedores. Use junto com as regras em `.cursor/rules/` e com `AGENTS.md` / `SKILLS.md`.

## O que é o projeto

- **Objetivo**: Horta inteligente low-cost com **ESP32-S3** (sensores, câmera OV2640, atuadores) e **detecção de doenças em folhas de tomate** via pipeline de ML e API Flask.
- **Fluxo**: Captura no ESP → envio para servidor de inferência → extração de features (mesma do treino) → predição → resposta JSON (classe, score, top-k, latência).

## Estrutura do repositório

```
uepg-garden-lowcost-iot-ml/
├── .cursor/rules/       # Regras por contexto (project, ml, firmware)
├── docs/                # architecture.md, api.md, firmware.md, ml.md
├── firmware/
│   └── smart-tomato-garden/   # C++, PlatformIO, web UI, NVS, /capture, /api/inference/config
├── ml/                  # Pacote garden_ml (Python)
│   ├── src/garden_ml/   # features, training, inference, config, data, image, viz
│   ├── tests/
│   ├── pyproject.toml
│   └── README.md
├── Makefile             # venv, ml-test, ml-augment, ml-train, ml-eval, ml-serve, Docker
├── CLAUDE.md            # Este arquivo
├── AGENTS.md            # Guia para agentes
└── SKILLS.md            # Comandos e caminhos de referência
```

## Decisões técnicas importantes

- **ML**: Pré-processamento **sem** normalização fotométrica. Treino e inferência usam apenas segmentação (HSV) e redimensionamento (letterbox). Contrato documentado em `training_metadata.json` e no README do ml.
- **Features**: 188 dimensões; extração única em `garden_ml/features/extract.py` para treino, avaliação e predictor.
- **Split e vazamento**: Split por grupo (StratifiedGroupKFold); originais duplicados por hash removidos do conjunto de teste.
- **Firmware**: Configuração do servidor de inferência pela interface web (NVS) ou pelo script `tools/set_inference.py`; Wi-Fi em tempo de compilação via `secrets.h`.

## Onde encontrar mais

- **Arquitetura e fluxos**: `docs/architecture.md`
- **API de inferência**: `docs/api.md`, `ml/README.md`
- **Firmware e endpoints**: `firmware/README.md`
- **Comandos e variáveis Make**: `SKILLS.md`, `Makefile`
