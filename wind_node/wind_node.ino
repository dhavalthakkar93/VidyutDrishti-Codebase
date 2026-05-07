// VidyutDrishti — Wind Node Firmware (WiFi + MQTT)
// Sensors: 2x Pot (speed/direction) + BME280 + MPU-6050 + OLED
// Publishes JSON to: vidyutdrishti/wind

#include <WiFi.h>
#include <PubSubClient.h>
#include <Wire.h>
#include <Adafruit_BME280.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_SH110X.h>
#include <Adafruit_GFX.h>

// --- WiFi ---
const char* WIFI_SSID = "Dhaval 4G";
const char* WIFI_PASS = "Dhaval@4995";

// --- MQTT ---
const char* MQTT_BROKER = "192.168.29.61";
const int   MQTT_PORT = 1883;
const char* MQTT_TOPIC = "vidyutdrishti/wind";
const char* MQTT_CLIENT_ID = "wind_node_01";

// --- Pins ---
#define SPEED_PIN       34
#define DIRECTION_PIN   35
#define SCREEN_WIDTH    128
#define SCREEN_HEIGHT   64
#define OLED_ADDR       0x3C
#define OLED_RESET      -1

// --- Wind speed mapping ---
const float MAX_WIND_SPEED = 15.0;

// --- Wind direction calibration ---
int NORTH_ADC = 0;

// --- Publish interval ---
const unsigned long PUBLISH_INTERVAL = 2000;
unsigned long lastPublish = 0;

// --- Objects ---
WiFiClient wifiClient;
PubSubClient mqtt(wifiClient);
Adafruit_BME280 bme;
Adafruit_MPU6050 mpu;
Adafruit_SH1106G display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

bool bme280OK = false;
bool mpuOK = false;
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
  pinMode(SPEED_PIN, INPUT);
  pinMode(DIRECTION_PIN, INPUT);

  Wire.begin(21, 22);

  Serial.println("\n=== VIDYUTDRISHTI WIND NODE ===\n");

  // Init sensors
  bme280OK = bme.begin(0x76);
  Serial.printf("BME280:  %s\n", bme280OK ? "OK (0x76)" : "FAIL");

  mpuOK = mpu.begin(0x68);
  if (mpuOK) {
    mpu.setAccelerometerRange(MPU6050_RANGE_4_G);
    mpu.setGyroRange(MPU6050_RANGE_500_DEG);
    mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
    Serial.println("MPU6050: OK (0x68)");
  } else {
    Serial.println("MPU6050: FAIL");
  }

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
    display.println("VidyutDrishti Wind");
    display.printf("WiFi: %s\n", WiFi.status() == WL_CONNECTED ? "OK" : "FAIL");
    display.printf("BME280:  %s\n", bme280OK ? "OK" : "FAIL");
    display.printf("MPU6050: %s\n", mpuOK ? "OK" : "FAIL");
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

  // --- Wind Speed (Pot 1) ---
  int speedRaw = analogRead(SPEED_PIN);
  float windSpeed = speedRaw * MAX_WIND_SPEED / 4095.0;

  // --- Wind Direction (Pot 2) ---
  int dirRaw = analogRead(DIRECTION_PIN);
  float rawDegrees = dirRaw * 270.0 / 4095.0;
  float northDegrees = NORTH_ADC * 270.0 / 4095.0;
  float direction = rawDegrees - northDegrees;
  if (direction < 0) direction += 360.0;
  if (direction >= 360.0) direction -= 360.0;

  const char* compass;
  if (direction < 22.5 || direction >= 337.5)  compass = "N";
  else if (direction < 67.5)   compass = "NE";
  else if (direction < 112.5)  compass = "E";
  else if (direction < 157.5)  compass = "SE";
  else if (direction < 202.5)  compass = "S";
  else if (direction < 247.5)  compass = "SW";
  else if (direction < 292.5)  compass = "W";
  else                         compass = "NW";

  // --- BME280 ---
  float temp = 0, hum = 0, pressure = 0;
  if (bme280OK) {
    temp = bme.readTemperature();
    hum = bme.readHumidity();
    pressure = bme.readPressure() / 100.0;
  }

  // --- MPU-6050 (vibration) ---
  float vibration = 0;
  if (mpuOK) {
    sensors_event_t a, g, t;
    mpu.getEvent(&a, &g, &t);
    float totalG = sqrt(a.acceleration.x * a.acceleration.x +
                        a.acceleration.y * a.acceleration.y +
                        a.acceleration.z * a.acceleration.z);
    vibration = abs(totalG - 9.81);
  }

  // --- Wind Power ---
  float rho = 1.225;
  if (bme280OK && temp > 0) {
    rho = (pressure * 100.0) / (287.05 * (temp + 273.15));
  }
  float windPower = 0.5 * rho * 1.0 * 0.35 * windSpeed * windSpeed * windSpeed;

  // --- Publish MQTT ---
  if (mqtt.connected()) {
    char json[256];
    snprintf(json, sizeof(json),
      "{\"wind_speed\":%.1f,\"wind_direction\":%.0f,\"compass\":\"%s\","
      "\"air_temp\":%.1f,\"humidity\":%.1f,\"pressure\":%.1f,"
      "\"vibration\":%.2f,\"wind_power\":%.2f}",
      windSpeed, direction, compass, temp, hum, pressure, vibration, windPower);
    mqtt.publish(MQTT_TOPIC, json);
    Serial.printf("MQTT> %s\n", json);
  }

  // --- OLED ---
  if (oledOK) {
    display.clearDisplay();
    display.setCursor(0, 0);

    display.println("== WIND NODE ==");
    display.printf("Speed: %.1f m/s\n", windSpeed);
    display.printf("Dir:   %.0f %s\n", direction, compass);
    display.printf("Temp:  %.1fC H:%.0f%%\n", temp, hum);
    display.printf("hPa:   %.0f\n", pressure);
    display.printf("Vib:   %.2f m/s2\n", vibration);
    display.printf("MQTT:%s WiFi:%ddBm", mqtt.connected() ? "OK" : "--", WiFi.RSSI());

    display.display();
  }
}
