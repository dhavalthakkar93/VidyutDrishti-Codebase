// VidyutDrishti — Solar Node Firmware (WiFi + MQTT)
// Sensors: BH1750 + BME280 + GY-906 + ACS712 + Laser/LDR + OLED
// Publishes JSON to: vidyutdrishti/solar

#include <WiFi.h>
#include <PubSubClient.h>
#include <Wire.h>
#include <BH1750.h>
#include <Adafruit_BME280.h>
#include <Adafruit_MLX90614.h>
#include <Adafruit_SH110X.h>
#include <Adafruit_GFX.h>

// --- WiFi ---
const char* WIFI_SSID = "Dhaval 4G";
const char* WIFI_PASS = "Dhaval@4995";

// --- MQTT ---
const char* MQTT_BROKER = "192.168.29.61";
const int   MQTT_PORT = 1883;
const char* MQTT_TOPIC = "vidyutdrishti/solar";
const char* MQTT_CLIENT_ID = "solar_node_01";

// --- Pins ---
#define CURRENT_PIN     34
#define LDR_PIN         36
#define LASER_PIN       27
#define SCREEN_WIDTH    128
#define SCREEN_HEIGHT   64
#define OLED_ADDR       0x3C
#define OLED_RESET      -1

// --- ACS712 ---
float ACS_ZERO_V = 2.33;
const float ACS_SENSITIVITY = 0.185;
const float ADC_REF = 3.3;
const int ADC_MAX = 4095;
const float PANEL_VOLTAGE = 8.0;
const float LUX_TO_GHI = 1.0 / 120.0;

// --- Publish interval ---
const unsigned long PUBLISH_INTERVAL = 2000;
unsigned long lastPublish = 0;

// --- Objects ---
WiFiClient wifiClient;
PubSubClient mqtt(wifiClient);
BH1750 lightMeter;
Adafruit_BME280 bme;
Adafruit_MLX90614 mlx;
Adafruit_SH1106G display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

bool bh1750OK = false;
bool bme280OK = false;
bool mlxOK = false;
bool oledOK = false;

void connectWiFi() {
  Serial.printf("Connecting to WiFi: %s", WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("\nWiFi connected! IP: %s\n", WiFi.localIP().toString().c_str());
  } else {
    Serial.println("\nWiFi FAILED - continuing offline");
  }
}

void connectMQTT() {
  if (mqtt.connected()) return;
  Serial.print("MQTT: connecting...");
  if (mqtt.connect(MQTT_CLIENT_ID)) {
    Serial.println("connected!");
  } else {
    Serial.printf("failed (rc=%d)\n", mqtt.state());
  }
}

void setup() {
  Serial.begin(115200);
  delay(500);
  pinMode(CURRENT_PIN, INPUT);
  pinMode(LDR_PIN, INPUT);
  pinMode(LASER_PIN, OUTPUT);
  digitalWrite(LASER_PIN, LOW);

  Wire.begin(21, 22);

  Serial.println("\n=== VIDYUTDRISHTI SOLAR NODE ===\n");

  // Auto-calibrate ACS712
  Serial.println("Calibrating ACS712...");
  long calSum = 0;
  for (int i = 0; i < 100; i++) {
    calSum += analogRead(CURRENT_PIN);
    delay(10);
  }
  ACS_ZERO_V = (calSum / 100.0) * ADC_REF / ADC_MAX;
  Serial.printf("ACS712 zero: %.4fV\n\n", ACS_ZERO_V);

  // Init sensors
  bh1750OK = lightMeter.begin(BH1750::CONTINUOUS_HIGH_RES_MODE);
  Serial.printf("BH1750:  %s\n", bh1750OK ? "OK (0x23)" : "FAIL");

  bme280OK = bme.begin(0x76);
  Serial.printf("BME280:  %s\n", bme280OK ? "OK (0x76)" : "FAIL");

  mlxOK = mlx.begin();
  Serial.printf("GY-906:  %s\n", mlxOK ? "OK (0x5A)" : "FAIL");

  oledOK = display.begin(OLED_ADDR, true);
  Serial.printf("OLED:    %s\n", oledOK ? "OK (0x3C)" : "FAIL");

  // WiFi + MQTT
  connectWiFi();
  mqtt.setServer(MQTT_BROKER, MQTT_PORT);

  if (oledOK) {
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(SH110X_WHITE);
    display.setCursor(0, 0);
    display.println("VidyutDrishti Solar");
    display.printf("WiFi: %s\n", WiFi.status() == WL_CONNECTED ? "OK" : "FAIL");
    display.printf("BH1750: %s\n", bh1750OK ? "OK" : "FAIL");
    display.printf("BME280: %s\n", bme280OK ? "OK" : "FAIL");
    display.printf("GY-906: %s\n", mlxOK ? "OK" : "FAIL");
    display.display();
    delay(2000);
  }

  Serial.println("\nPublishing every 2 seconds...\n");
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) connectWiFi();
  if (!mqtt.connected()) connectMQTT();
  mqtt.loop();

  unsigned long now = millis();
  if (now - lastPublish < PUBLISH_INTERVAL) return;
  lastPublish = now;

  // --- BH1750 ---
  float lux = 0;
  if (bh1750OK) {
    lux = lightMeter.readLightLevel();
    if (lux < 0) lux = 0;
  }
  float ghi = lux * LUX_TO_GHI;

  // --- BME280 ---
  float temp = 0, hum = 0, pressure = 0;
  if (bme280OK) {
    temp = bme.readTemperature();
    hum = bme.readHumidity();
    pressure = bme.readPressure() / 100.0;
  }

  // --- GY-906 ---
  float panelTemp = 0;
  if (mlxOK) {
    panelTemp = mlx.readObjectTempC();
  }

  // --- ACS712 ---
  long sum = 0;
  for (int i = 0; i < 50; i++) {
    sum += analogRead(CURRENT_PIN);
    delayMicroseconds(200);
  }
  float acsV = (sum / 50.0) * ADC_REF / ADC_MAX;
  float current = (acsV - ACS_ZERO_V) / ACS_SENSITIVITY;
  if (current < 0.005) current = 0;
  float power = current * PANEL_VOLTAGE;

  // --- Soiling ---
  int baseline = 0;
  for (int i = 0; i < 5; i++) { baseline += analogRead(LDR_PIN); delayMicroseconds(200); }
  baseline /= 5;

  digitalWrite(LASER_PIN, HIGH);
  delay(50);
  int reflected = 0;
  for (int i = 0; i < 5; i++) { reflected += analogRead(LDR_PIN); delayMicroseconds(200); }
  reflected /= 5;
  digitalWrite(LASER_PIN, LOW);

  int diff = reflected - baseline;
  if (diff < 0) diff = 0;
  float soiling = 1.0 - ((float)diff / 2000.0);
  if (soiling < 0) soiling = 0;
  if (soiling > 1) soiling = 1;

  // --- Publish MQTT ---
  if (mqtt.connected()) {
    char json[256];
    snprintf(json, sizeof(json),
      "{\"lux\":%.0f,\"ghi\":%.2f,\"air_temp\":%.1f,\"panel_temp\":%.1f,"
      "\"humidity\":%.1f,\"pressure\":%.1f,\"current\":%.3f,\"power\":%.3f,\"soiling\":%.2f}",
      lux, ghi, temp, panelTemp, hum, pressure, current, power, soiling);
    mqtt.publish(MQTT_TOPIC, json);
    Serial.printf("MQTT> %s\n", json);
  }

  // --- OLED ---
  if (oledOK) {
    display.clearDisplay();
    display.setCursor(0, 0);

    display.println("== SOLAR NODE ==");
    display.printf("GHI:  %.1f W/m2\n", ghi);
    display.printf("Air:%.1fC Pan:%.1fC\n", temp, panelTemp);
    display.printf("H:%.0f%% hPa:%.0f\n", hum, pressure);
    display.printf("Power:%.2fW %.0fmA\n", power, current * 1000);
    display.printf("Soil:%.2f\n", soiling);
    display.printf("MQTT:%s WiFi:%ddBm", mqtt.connected() ? "OK" : "--", WiFi.RSSI());

    display.display();
  }
}
