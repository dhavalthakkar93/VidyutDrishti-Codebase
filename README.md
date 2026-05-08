# VidyutDrishti — AI-Powered Renewable Energy Forecasting

An edge-to-insight system that forecasts solar and wind power output 1-24 hours ahead using real sensors, physics-informed ML, and a local LLM assistant — running entirely on-premise with zero cloud dependency.

## Key Results

- **Solar 1h MAPE:** 9.1% (88% better than persistence baseline)
- **Wind 1h MAPE:** 13.5% (70% better than persistence baseline)
- **12 physical sensors** across 2 ESP32 nodes
- **End-to-end latency:** Sensor to forecast in <3 seconds
- **Fully on-premise:** No cloud APIs, no data leaving the system

## Architecture

```
[12 Sensors] → [ESP32 x2] → [MQTT/WiFi] → [Raspberry Pi] → [REST API] → [ML + Dashboard + LLM]
     │              │              │              │                              │
  7 Solar       Firmware       2s interval     SQLite +                    XGBoost (5 horizons)
  5 Wind        + OLED         JSON msgs      Flask API                   Conformal Prediction
                                                                          SHAP Explainability
                                                                          Qwen 2.5 7B (local)
```

## Directory Structure

```
codebase/
├── solar_node/
│   ├── firmware/solar_node.ino    # Solar node firmware (ESP32)
├── wind_node/
│   ├── firmware/wind_node.ino     # Wind node firmware (ESP32)
├── raspberry_pi/
│   ├── data_collector.py          # MQTT subscriber → SQLite
│   ├── api_server.py              # Flask REST API
│   └── setup_autostart.sh         # Systemd service setup
├── ml/
│   ├── generate_synthetic_data.py # 30-day synthetic dataset generator
│   ├── feature_engineering.py     # 93 features (lags, rolling, temporal, physics)
│   ├── train_model.py             # XGBoost training (5 horizons x 2 nodes)
│   ├── conformal_prediction.py    # Distribution-free uncertainty intervals
│   ├── explainability.py          # SHAP value computation
│   ├── evaluate.py                # Model evaluation + baseline comparison
│   ├── predict.py                 # Inference API
│   ├── run_pipeline.py            # End-to-end training pipeline
│   ├── models/                    # Trained models (.joblib)
│   └── data/                      # Training data (SQLite + parquet)
├── dashboard/
│   ├── app.py                     # Streamlit dashboard (live data + forecast + LLM)
│   ├── requirements.txt           # Python dependencies
│   └── .streamlit/config.toml     # Streamlit theme config
├── presentation/
│   ├── index.html                 # Main presentation slides
│   └── video_backdrop.html        # Video demo backdrop slides
├── start_demo.sh                  # One-command demo launcher
└── README.md
```

## Hardware

### Solar Node (7 sensors)
| Sensor | Measurement | Role |
|--------|-------------|------|
| BH1750 | Light intensity → GHI (W/m²) | Primary power driver |
| BME280 | Temperature, Humidity, Pressure | Weather context |
| GY-906 (MLX90614) | Panel surface temperature (IR) | Efficiency correction |
| ACS712 | Current (A) | Actual power measurement |
| Laser + LDR | Soiling index (0-1) | Novel dust detection |
| OLED (SSD1306) | Display | On-node readings |

### Wind Node (5 sensors)
| Sensor | Measurement | Role |
|--------|-------------|------|
| Potentiometer (x2) | Wind speed (m/s) + Direction (°) | Anemometer/vane simulation |
| BME280 | Temperature, Pressure | Air density calculation |
| MPU-6050 | 6-axis vibration (m/s²) | Predictive maintenance |
| OLED (SSD1306) | Display | On-node readings |

## Setup & Running

### Prerequisites
- Python 3.9+
- Arduino IDE (for ESP32 firmware)
- Raspberry Pi with Mosquitto MQTT broker
- Ollama (for local LLM)

### Quick Start

```bash
# 1. Install Python dependencies
pip install -r dashboard/requirements.txt

# 2. Train ML models (or use pre-trained in ml/models/)
cd ml && python run_pipeline.py

# 3. Start Ollama LLM
ollama serve &
ollama pull qwen2.5:7b

# 4. Launch dashboard
cd dashboard && streamlit run app.py

# 5. (Optional) Flash ESP32 firmware via Arduino IDE
#    - solar_node/firmware/solar_node.ino
#    - wind_node/firmware/wind_node.ino
```

### Raspberry Pi Setup
```bash
# On the Pi (192.168.29.61):
sudo apt install mosquitto mosquitto-clients
python3 raspberry_pi/data_collector.py &
python3 raspberry_pi/api_server.py &
```

## ML Methodology

- **Model:** XGBoost (gradient-boosted trees, 500 trees, depth 6)
- **Features:** 93 (solar) / 85 (wind) — lags, rolling stats, temporal sin/cos, physics
- **Horizons:** 1h, 3h, 6h, 12h, 24h ahead (separate model per horizon)
- **Uncertainty:** Conformal prediction (distribution-free, verified calibration)
- **Explainability:** SHAP values per prediction
- **Physics constraints:** Solar=0 at night, power>=0, wind P∝v³

## Physics

```
Solar: P = η(T) × A × GHI × (1 - Soiling)
       η drops 0.4% per °C above 25°C

Wind:  P = ½ × ρ × A × Cp × v³
       ρ = air density from ideal gas law (BME280 pressure + temp)
       2× speed = 8× power (cubic relationship)
```

## Team

**GreenMatrix** — Dhaval Thakkar
