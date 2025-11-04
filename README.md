# 🌱 Horta Inteligente de Baixo Custo — IoT + IA (ESP32 + Flask)

**Low-cost smart garden** integrating **ESP32-S3** (sensors, actuators and OV2640 camera) with a **Flask API** for **tomato leaf disease detection**.  
The system captures images, extracts **102 features** (HSV, Haralick, LBP, Hu, Zernike, morphology) and classifies them using **XGBoost**.  
End-to-end flow: **capture → transmit → inference → JSON → logs**.

---

## 📁 Estrutura do Projeto

.
├── ia/ # Inteligência Artificial e API Flask
│ ├── requirements.txt
│ ├── app.py # API principal /predict
│ ├── treino/ # Scripts de treinamento
│ │ ├── extrair_features.py
│ │ └── treinar_modelos.py
│ ├── modelos/ # Modelos treinados (.pkl)
│ ├── utils/ # Funções auxiliares
│ └── data/ # Exemplos de dados
└── hardware/ # Código do ESP32-S3
├── src/
│ └── main.cpp
├── include/
│ └── camera_pins.h
└── README.md

yaml
Copiar código

---

## 🧠 Visão Geral

- **Automação**: sensores de solo, luz e temperatura, com controle de bomba e ventoinha via relé (histerese e temporização).  
- **Visão Computacional**: segmentação do limbo foliar e extração de 102 descritores complementares.  
- **Classificação**: modelo **XGBoost** comparado a RF e SVM.  
- **API Flask**: endpoint `/predict` recebe imagens e devolve a classe, confiança e tempo de inferência.  
- **Logs**: registro de leituras, acionamentos e tempos de resposta.

---

## ⚙️ Módulo de IA (`/ia`)

### 🔹 Instalação

```bash
cd ia
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
🔹 Executar a API
bash
Copiar código
export FLASK_APP=app.py      # Windows: $env:FLASK_APP="app.py"
flask run --host 0.0.0.0 --port 5000
Endpoint: POST http://<IP>:5000/predict
Body (multipart): campo image com arquivo JPEG/PNG

Exemplo de resposta (JSON):

json
Copiar código
{
  "classe_predita": "Tomato_Late_blight",
  "score": 0.88,
  "tempo_inferencia_s": 0.21,
  "top3": [
    {"classe": "Tomato_Late_blight", "score": 0.88},
    {"classe": "Tomato_Early_blight", "score": 0.07},
    {"classe": "Tomato_Bacterial_spot", "score": 0.03}
  ]
}
🔹 Treinamento (opcional)
Dataset de referência: PlantVillage (Tomato).

Scripts:

treino/extrair_features.py

treino/treinar_modelos.py

Técnicas:

GridSearchCV (5-fold) para otimizar hiperparâmetros.

Comparação entre Random Forest, SVM e XGBoost.

Modelos salvos em ia/modelos/.

🔌 Módulo de Hardware (/hardware)
🔹 Requisitos
ESP32-S3 Dev Module (com câmera OV2640)

Arduino IDE (ou PlatformIO) com core da Espressif

Drivers USB instalados (CP210x, CH34x, etc.)

🔹 Conexões Principais
Componente	Pino (GPIO)	Função
Sensor de Solo	1	Leitura analógica (umidade)
Sensor de Luz LDR	14	Leitura analógica (luminosidade)
Sensor DHT22	21	Temperatura e umidade do ar
Relé Bomba	47	Acionamento da bomba d’água
Relé Ventoinha	46	Acionamento da ventoinha
Câmera OV2640	conforme camera_pins.h	Captura de imagem

🔹 Upload do Firmware
Abra hardware/src/main.cpp no Arduino IDE.

Configure:

Placa: ESP32S3 Dev Module

USB CDC On Boot: Enabled

Upload Speed: 921600 (ou 460800/115200)

SSID e Senha Wi-Fi no código.

Se necessário:

Mantenha BOOT pressionado e toque EN/RESET durante o upload.

Após o upload, o IP do servidor será exibido no Serial Monitor.

📊 Interpretação dos Logs
Leitura	Unidade	Significado
Luminosidade	%	0 = escuro / 100 = muito claro
Umidade do Solo	%	0 = úmido / 100 = seco
Temperatura	°C	Valor em graus Celsius
Umidade do Ar	%	Umidade relativa do ambiente
Mensagens	texto	Status legíveis: bomba ligada, ventoinha ativa…

Exemplo:

vbnet
Copiar código
======= MONITORAMENTO =======
Luminosidade: 72%
Umidade do Solo: 65%
Temperatura: 27.8°C
Umidade do Ar: 58%
Solo seco! Ligando bomba por 1 segundo...
Irrigação concluída. Bomba desligada.
Temperatura normalizada. Ventoinha desligada.
🧩 Boas Práticas
Usar o mesmo pipeline na inferência e no treinamento.

Garantir iluminação constante na captura das folhas.

Monitorar logs de latência e histórico de acionamentos.

Para uso embarcado de IA: testar CNN quantizadas e redução de modelo.

📜 Licença
Licença sugerida: MIT
Adicione o arquivo LICENSE na raiz do projeto.

🙌 Agradecimentos
Projeto desenvolvido na Universidade Estadual de Ponta Grossa (UEPG) — Departamento de Informática.
Orientador: Prof. Luciano Senger
Coorientadora: Profa. Gabrielly de Queiroz Pereira
