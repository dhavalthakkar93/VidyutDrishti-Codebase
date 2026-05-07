import json
import sqlite3
import os
from datetime import datetime
import paho.mqtt.client as mqtt

BROKER = "localhost"
PORT = 1883
TOPICS = [
    ("vidyutdrishti/solar", 0),
    ("vidyutdrishti/wind", 0),
]

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, "vidyutdrishti.db")


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS solar (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            lux REAL,
            ghi REAL,
            air_temp REAL,
            panel_temp REAL,
            humidity REAL,
            pressure REAL,
            current REAL,
            power REAL,
            soiling REAL
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS wind (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            wind_speed REAL,
            wind_direction REAL,
            compass TEXT,
            air_temp REAL,
            humidity REAL,
            pressure REAL,
            vibration REAL,
            wind_power REAL
        )
    """)
    conn.commit()
    conn.close()


def insert_solar(data):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO solar (timestamp, lux, ghi, air_temp, panel_temp, humidity, pressure, current, power, soiling)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data.get("timestamp"),
        data.get("lux"),
        data.get("ghi"),
        data.get("air_temp"),
        data.get("panel_temp"),
        data.get("humidity"),
        data.get("pressure"),
        data.get("current"),
        data.get("power"),
        data.get("soiling"),
    ))
    conn.commit()
    conn.close()


def insert_wind(data):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO wind (timestamp, wind_speed, wind_direction, compass, air_temp, humidity, pressure, vibration, wind_power)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data.get("timestamp"),
        data.get("wind_speed"),
        data.get("wind_direction"),
        data.get("compass"),
        data.get("air_temp"),
        data.get("humidity"),
        data.get("pressure"),
        data.get("vibration"),
        data.get("wind_power"),
    ))
    conn.commit()
    conn.close()


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Connected to MQTT broker")
        client.subscribe(TOPICS)
        print(f"  Subscribed to: {[t[0] for t in TOPICS]}")
    else:
        print(f"Connection failed with code {rc}")


def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
    except json.JSONDecodeError:
        print(f"  Bad JSON on {msg.topic}: {msg.payload}")
        return

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    payload["timestamp"] = timestamp

    if msg.topic == "vidyutdrishti/solar":
        insert_solar(payload)
        print(f"  [{timestamp}] SOLAR | GHI:{payload.get('ghi', '?')} W/m2 | "
              f"Panel:{payload.get('panel_temp', '?')}C | "
              f"Power:{payload.get('power', '?')}W | "
              f"Soil:{payload.get('soiling', '?')}")

    elif msg.topic == "vidyutdrishti/wind":
        insert_wind(payload)
        print(f"  [{timestamp}] WIND  | Speed:{payload.get('wind_speed', '?')} m/s | "
              f"Dir:{payload.get('compass', '?')} | "
              f"Power:{payload.get('wind_power', '?')}W | "
              f"Vib:{payload.get('vibration', '?')}")


def main():
    print("=" * 50)
    print("  VidyutDrishti Data Collector")
    print("=" * 50)
    print(f"  Broker: {BROKER}:{PORT}")
    print(f"  Database: {DB_PATH}")
    print("=" * 50)

    init_db()

    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(BROKER, PORT, 60)
    client.loop_forever()


if __name__ == "__main__":
    main()
