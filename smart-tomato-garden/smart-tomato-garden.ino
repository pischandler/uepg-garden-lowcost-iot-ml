#include "esp_camera.h"
#include <WiFi.h>
#include <DHT.h>

// ================= CONFIGURAÇÃO DO MODELO DA CÂMERA =================
#define CAMERA_MODEL_CUSTOM_ESP32S3_CAM
#include "camera_pins.h"

// ================= CREDENCIAIS WI-FI =================
const char *ssid = "CLARO_2G07ECA7";
const char *password = "Victor3894";

// ================= PINOS E DEFINIÇÕES =================
#define SOIL_SENSOR_PIN 1    // GPIO1 (ADC2_CH0)
#define DHT_PIN 21           // GPIO21
#define DHT_TYPE DHT22
#define LDR_PIN 14
#define RELAY_BOMBA_PIN 47      // GPIO47 - bomba
#define RELAY_VENTOINHA_PIN 46  // GPIO46 - ventoinha (canal 2 do relé)

// ---------- ADC / Calibração ----------
#define ADC_MAX 4095

// Ajuste estes dois valores após medir seu sensor no seu solo:
// SOLO BEM MOLHADO (ADC baixo) e SOLO SECO (ADC alto).
#define SOIL_WET_ADC 900
#define SOIL_DRY_ADC 3000

// Se seu LDR cresce quando escurece, deixe true para inverter e mostrar "luz %" intuitivo
#define LDR_INVERT true

// ---------- Lógicas de controle ----------
#define SOLO_SECO_LIMITE 2500        // limiar em ADC para acionar irrigação (mantido para compatibilidade)
#define TEMPO_IRRIGACAO_MS 1000       // 1 segundo

#define TEMP_MAXIMA_VENTOINHA 28.0
#define TEMP_DESLIGAR_VENTOINHA 26.0

DHT dht(DHT_PIN, DHT_TYPE);
bool bombaLigada = false;
bool ventoinhaLigada = false;
unsigned long tempoInicioIrrigacao = 0;

void startCameraServer();
void setupLedFlash(int pin);

// ==================== Funções utilitárias ====================

static float clampf(float v, float lo, float hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}

// Mapeia e normaliza para 0..100 (%)
static float percentFromRange(int x, int in_min, int in_max, bool invert=false) {
  // evita divisão por zero
  if (in_max == in_min) return 0.0f;
  float t = (float)(x - in_min) / (float)(in_max - in_min);
  t = clampf(t, 0.0f, 1.0f);
  if (invert) t = 1.0f - t;
  return t * 100.0f;
}

// Umidade do solo em % (0 = seco, 100 = bem molhado)
// Como o sensor aumenta ADC quando seca, usamos invert=true no intervalo [SOIL_WET_ADC..SOIL_DRY_ADC]
static float soilPercent(int soilAdc) {
  return percentFromRange(soilAdc, SOIL_WET_ADC, SOIL_DRY_ADC, true);
}

// Luminosidade em % (0 = escuro, 100 = claro) — inverter se necessário
static float ldrPercent(int ldrAdc) {
  return percentFromRange(ldrAdc, 0, ADC_MAX, LDR_INVERT);
}

// Converte o limiar de ADC do solo para % (para mostrar na mensagem)
static float soilThresholdPercent() {
  return soilPercent(SOLO_SECO_LIMITE);
}

// Formata ON/OFF bonitinho
static const char* onoff(bool v) {
  return v ? "LIGADO" : "DESLIGADO";
}

// ==================== Setup ====================
void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(true);
  Serial.println();

  // ========== CONFIGURANDO A CÂMERA ==========
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.grab_mode = CAMERA_GRAB_LATEST;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.jpeg_quality = 12;
  config.fb_count = 2;

  if (!psramFound()) {
    Serial.println("PSRAM não detectada. Ajustando parâmetros de câmera...");
    config.frame_size = FRAMESIZE_SVGA;
    config.fb_location = CAMERA_FB_IN_DRAM;
    config.fb_count = 1;
  } else {
    config.frame_size = FRAMESIZE_QVGA;
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Falha ao iniciar câmera. Código de erro: 0x%x\n", err);
    return;
  }

  sensor_t *s = esp_camera_sensor_get();
  s->set_vflip(s, 1);  // Corrige orientação (se precisar, ajuste para 0)

  // ========== INICIALIZAÇÃO DOS SENSORES ==========
  dht.begin();
  pinMode(RELAY_BOMBA_PIN, OUTPUT);
  pinMode(RELAY_VENTOINHA_PIN, OUTPUT);
  pinMode(LDR_PIN, INPUT);
  digitalWrite(RELAY_BOMBA_PIN, LOW);      // bomba desligada
  digitalWrite(RELAY_VENTOINHA_PIN, LOW);  // ventoinha desligada

  // ========== CONEXÃO WI-FI ==========
  WiFi.begin(ssid, password);
  WiFi.setSleep(false);

  Serial.print("Conectando ao Wi-Fi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWi-Fi conectado!");
  Serial.print("Acesse: http://");
  Serial.println(WiFi.localIP());

  startCameraServer();
}

// ==================== Loop ====================
void loop() {
  // ====== LEITURA DOS SENSORES ======
  int soilAdc = analogRead(SOIL_SENSOR_PIN);
  int ldrAdc  = analogRead(LDR_PIN);
  float temp  = dht.readTemperature();
  float humi  = dht.readHumidity();

  // Tratamento de leituras inválidas do DHT
  bool dhtOk = !(isnan(temp) || isnan(humi));

  // Converte para porcentagens amigáveis
  float soloPct = soilPercent(soilAdc);
  float soloPctLimiar = soilThresholdPercent();
  float luzPct  = ldrPercent(ldrAdc);
  float arPct   = dhtOk ? humi : -1;

  // ====== BLOCO DE STATUS AMIGÁVEL ======
  Serial.println();
  Serial.println(F("────────────────────────────────────────"));
  Serial.printf("⏱  Uptime: %lus\n", millis() / 1000UL);
  Serial.println(F("📊 Status do Sistema"));
  Serial.println(F("────────────────────────────────────────"));
  Serial.printf("🌱 Solo: %.1f%% umidade (ADC=%d) | Limiar p/ irrigar: %.1f%% (ADC>%d)\n",
                soloPct, soilAdc, soloPctLimiar, SOLO_SECO_LIMITE);
  Serial.printf("💡 Luz:  %.1f%% (ADC=%d) %s\n",
                luzPct, ldrAdc, LDR_INVERT ? "(invertido p/ % claro)" : "");
  if (dhtOk) {
    Serial.printf("🌡  Ar:   %.1f°C | 💧 Umidade: %.1f%%\n", temp, arPct);
  } else {
    Serial.println("🌡  Ar:   (falha de leitura do DHT) — tentando novamente no próximo ciclo");
  }
  Serial.printf("⚙️  Bomba: %s | Ventoinha: %s\n", onoff(bombaLigada), onoff(ventoinhaLigada));
  Serial.println(F("────────────────────────────────────────"));

  // ====== CONTROLE DA BOMBA (baseado no limiar ADC existente) ======
  if (!bombaLigada && soilAdc > SOLO_SECO_LIMITE) {
    Serial.printf("🚿 Solo seco (%.1f%% <= %.1f%%). Bomba LIGADA por 1 s...\n",
                  soloPct, soloPctLimiar);
    digitalWrite(RELAY_BOMBA_PIN, HIGH);
    bombaLigada = true;
    tempoInicioIrrigacao = millis();
  }

  if (bombaLigada && millis() - tempoInicioIrrigacao >= TEMPO_IRRIGACAO_MS) {
    digitalWrite(RELAY_BOMBA_PIN, LOW);
    bombaLigada = false;
    Serial.println("✅ Irrigação concluída. Bomba DESLIGADA.");
  }

  // ====== CONTROLE DA VENTOINHA (histerese simples) ======
  if (dhtOk) {
    if (!ventoinhaLigada && temp >= TEMP_MAXIMA_VENTOINHA) {
      digitalWrite(RELAY_VENTOINHA_PIN, HIGH);
      ventoinhaLigada = true;
      Serial.printf("🌀 Temperatura alta (%.1f°C ≥ %.1f°C). Ventoinha LIGADA.\n",
                    temp, TEMP_MAXIMA_VENTOINHA);
    }

    if (ventoinhaLigada && temp <= TEMP_DESLIGAR_VENTOINHA) {
      digitalWrite(RELAY_VENTOINHA_PIN, LOW);
      ventoinhaLigada = false;
      Serial.printf("🧊 Temperatura normalizada (%.1f°C ≤ %.1f°C). Ventoinha DESLIGADA.\n",
                    temp, TEMP_DESLIGAR_VENTOINHA);
    }
  }

  // Intervalo entre ciclos de leitura/controle
  delay(5000);
}