# Low-Cost Smart Garden — IoT + ML (ESP32-S3 + Flask)

Horta inteligente de baixo custo que integra **ESP32-S3** (sensores, câmera OV2640, atuadores) com uma **API Flask de inferência** para **detecção de doenças em folhas de tomate**.

Fluxo ponta a ponta: **captura no ESP → envio para o servidor → extração de features → predição → JSON (classe, score, top-k, latência)**.

---

## Estrutura do repositório

```
uepg-garden-lowcost-iot-ml/
├── firmware/
│   └── smart-tomato-garden/   # Firmware ESP32-S3 (C++, PlatformIO), câmera, web UI, NVS
├── ml/                        # Pipeline ML (Python): augment, train, eval, API
│   ├── src/garden_ml/
│   ├── tests/
│   └── pyproject.toml
├── docs/                      # architecture.md, api.md, etc.
├── tools/                     # Scripts auxiliares (ex.: firmware)
├── Makefile                   # venv, ml-test, ml-augment, ml-train, ml-eval, ml-serve, Docker
├── CLAUDE.md                  # Contexto para assistentes
├── AGENTS.md                  # Guia de regras por área
└── SKILLS.md                  # Comandos e caminhos de referência
```

---

## Módulo de hardware (ESP32-S3)

O firmware em `firmware/smart-tomato-garden/` é responsável por:

- Ler sensores (umidade do solo, luz, temperatura/umidade DHT22)
- Controlar atuadores (bomba de irrigação, ventilador)
- Oferecer captura de imagem (`/capture`) e streaming MJPEG (`/stream`) para o servidor de inferência
- Expor saúde, métricas e configuração (incluindo **servidor de inferência** via web ou API)

**GPIO principais** (conferir `firmware/smart-tomato-garden/include/config.h`):

| Componente        | GPIO | Função                |
|-------------------|------|------------------------|
| Sensor umidade   | 1    | Leitura analógica      |
| LDR              | 14   | Intensidade de luz     |
| DHT22            | 21   | Temperatura/umidade    |
| Relé (bomba)     | 47   | Controle da irrigação |

O mapeamento da câmera OV2640 depende da placa utilizada (definido no firmware).

**Documentação detalhada:** [firmware/README.md](firmware/README.md).

---

## Módulo ML (servidor de inferência)

O pacote em `ml/` (Python ≥3.10) inclui:

- **Augmentação** com Albumentations (manifest e config exportados)
- **Extração de 188 features** (Haralick, Zernike, histogramas HSV, LBP, Hu, forma, LAB, chroma) — mesma fonte para treino e inferência
- **Treino** com Random Forest, SVM e XGBoost; CV por grupo (StratifiedGroupKFold) para evitar vazamento
- **Avaliação** com macro F1, balanced accuracy, ECE, matriz de confusão e sensibilidade à iluminação
- **API Flask** com `/health`, `/metrics` (Prometheus), upload de imagem e busca por URL (ex.: `http://esp/capture`)

**Pré-processamento:** não é usada normalização fotométrica; apenas **segmentação HSV** e **redimensionamento** (letterbox). Treino e inferência devem usar o mesmo `img_size` (ex.: 128).

**Documentação detalhada:** [ml/README.md](ml/README.md).

---

## Início rápido

### 1. ML (treino e API)

```bash
# Na raiz do repositório
make venv
make ml-test                    # rodar testes
make ml-augment-default         # augment (usa LOCAL_DATASET_RAW → LOCAL_DATASET_AUG)
make ml-train-default           # treino (usa LOCAL_DATASET_AUG → LOCAL_ARTIFACTS_DIR)
make ml-eval-default            # avaliação
make ml-serve                   # sobe a API em http://0.0.0.0:5000
```

Variáveis do Make (dataset, artefatos, seed, etc.): ver `Makefile` ou `make help`. Referência de comandos: [SKILLS.md](SKILLS.md).

### 2. Testar a API (após artefatos treinados)

```bash
# Upload de imagem
curl -X POST http://127.0.0.1:5000/analisar -F "image=@/caminho/para/folha.jpg"

# Ou servidor busca no ESP (configurar infer_host no ESP antes)
curl -X POST http://127.0.0.1:5000/analisar_url \
  -H "Content-Type: application/json" \
  -d '{"url":"http://192.168.100.12/capture","device_id":"stg-01"}'
```

Resposta típica: `classe_predita`, `score`, `topk`, `timings_ms`, `meta`.

### 3. Firmware (ESP32-S3)

```bash
cd firmware/smart-tomato-garden
# Copiar e preencher secrets
cp include/secrets.example.h include/secrets.h
# Ajustar config.h se necessário
pio run
pio run -t upload
```

Configurar o servidor de inferência no ESP: pela **interface web** (card "Configuração do servidor de inferência") ou com o script `python tools/set_inference.py --esp <IP_ESP> --ml <IP_ML>`.

---

## Documentação adicional

- **Arquitetura e fluxos:** [docs/architecture.md](docs/architecture.md)
- **Endpoints da API:** [docs/api.md](docs/api.md)
- **Contexto para assistentes:** [CLAUDE.md](CLAUDE.md) · [AGENTS.md](AGENTS.md) · [SKILLS.md](SKILLS.md)

---

## Licença e créditos

Recomendado: **MIT**. Incluir arquivo `LICENSE` na raiz.

Desenvolvido na **Universidade Estadual de Ponta Grossa (UEPG)** — Departamento de Informática.  
Orientador: Prof. Luciano J. Senger · Coorientadora: Prof. Gabrielly de Queiroz Pereira.
