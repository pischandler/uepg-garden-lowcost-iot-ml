# Guia para agentes (Cursor / assistentes de código)

Este documento orienta como aplicar contexto e regras ao trabalhar em diferentes partes do repositório.

## Escopo por área

| Área | Regras Cursor | O que priorizar |
|------|----------------|-----------------|
| **Raiz / docs / Makefile** | `.cursor/rules/project.mdc` | Estrutura geral, fluxos, convenções de repo |
| **ML (ml/**)** | `project.mdc` + `ml.mdc` | Pré-processamento único, 188 features, sem normalização fotométrica, artefatos e comandos garden-ml-* |
| **Firmware (firmware/**)** | `project.mdc` + `firmware.mdc` | PlatformIO, NVS, endpoints, secrets, web UI |

## Comportamento recomendado

1. **Antes de editar**
   - Identificar se a mudança é em **firmware** (C++/config/UI) ou **ML** (Python/garden_ml). Carregar as regras correspondentes (glob do Cursor aplica por caminho).
   - Para mudanças que cruzam (ex.: contrato da API de inferência), considerar `project.mdc` + `ml.mdc` + `firmware.mdc`.

2. **ML**
   - Manter uma única fonte de extração de features; não duplicar lógica entre treino e inferência.
   - Não reintroduzir normalização fotométrica sem alinhar em treino e deploy.
   - Testes em `ml/tests/`; rodar com `make ml-test`.

3. **Firmware**
   - Não commitar `secrets.h`; usar `secrets.example.h` como template.
   - Configuração do servidor de inferência: preferir documentar uso da interface web ou do script `set_inference.py`.

4. **Documentação**
   - Atualizar `README.md`, `docs/architecture.md`, `ml/README.md` ou `firmware/README.md` quando alterar fluxos ou contratos.

## Referências rápidas

- **Contexto geral do projeto**: `CLAUDE.md` (ou leitura inicial).
- **Comandos e caminhos**: `SKILLS.md`.
- **Detalhes de arquitetura**: `docs/architecture.md`, `docs/api.md`.
