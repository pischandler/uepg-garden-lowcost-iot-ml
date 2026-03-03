# tools/firmware

Ferramentas auxiliares para o firmware (opcional). Mantidas fora do core do projeto para não poluir o repositório principal.

## Uso típico

- Scripts de build/flash ou validação de ambiente
- Testes de endpoints (ex.: `curl` em `/health`, `/capture`)
- Coleta rápida de amostras de `/capture` para dataset ou testes
- Automação de configuração (ex.: chamadas à API do ESP)

## Observação

As ferramentas **dentro** do projeto do firmware (ex.: `set_inference.py`, `pack_webui.py`) ficam em `firmware/smart-tomato-garden/tools/`. Esta pasta `tools/firmware/` na raiz do repositório é para scripts que são compartilhados ou usados a partir da raiz (ex.: por CI ou por desenvolvedores que não querem entrar no diretório do firmware).

## Documentação

- Firmware: [../../firmware/README.md](../../firmware/README.md)
- README principal: [../README.md](../README.md)
