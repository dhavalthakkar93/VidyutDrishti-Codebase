"""
VidyutDrishti — Synthetic Data Generator
Generates realistic solar and wind generation data for Nadiad, Gujarat (22.69N, 72.86E)
30 days, 1-minute resolution. May 2026 weather patterns.
"""

import sqlite3
import numpy as np
import os
from datetime import datetime, timedelta

np.random.seed(42)

# Location: Nadiad, Gujarat
LATITUDE = 22.69
LONGITUDE = 72.86
ELEVATION = 30  # meters

# Simulation parameters
START = datetime(2026, 4, 15, 0, 0, 0)
DAYS = 30
INTERVAL_MIN = 1
TOTAL_POINTS = DAYS * 24 * 60

# Solar panel specs
PANEL_EFFICIENCY = 0.18
PANEL_AREA = 0.5  # m²
TEMP_COEFF = -0.004  # efficiency loss per degree C above 25
PANEL_VOLTAGE = 8.0

# Output paths
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, "training_data.db")


def solar_elevation(hour_decimal, day_of_year):
    declination = 23.45 * np.sin(np.radians((284 + day_of_year) / 365 * 360))
    hour_angle = (hour_decimal - 12) * 15
    lat_rad = np.radians(LATITUDE)
    dec_rad = np.radians(declination)
    ha_rad = np.radians(hour_angle)
    sin_elev = (np.sin(lat_rad) * np.sin(dec_rad) +
                np.cos(lat_rad) * np.cos(dec_rad) * np.cos(ha_rad))
    return np.degrees(np.arcsin(np.clip(sin_elev, -1, 1)))


def clear_sky_ghi(elevation_deg):
    if elevation_deg <= 0:
        return 0.0
    am = 1.0 / (np.sin(np.radians(elevation_deg)) + 0.50572 * (elevation_deg + 6.07995) ** -1.6364)
    am = min(am, 40)
    ghi = 1361 * 0.7 ** (am ** 0.678) * np.sin(np.radians(elevation_deg))
    return max(0, ghi)


def generate_weather_pattern(num_days):
    """Generate a coherent multi-day weather pattern for pre-monsoon Gujarat."""
    # Weather regimes: 0=clear, 1=hazy, 2=partly cloudy, 3=overcast (pre-monsoon)
    # April-May: mostly clear/hazy, occasional pre-monsoon cloud buildup
    regimes = []
    current = 0
    for d in range(num_days):
        if d < 5:
            current = 0  # clear start
        elif d < 8:
            current = 1  # haze builds
        elif d < 10:
            current = 0  # clear again
        elif d < 13:
            current = 2  # pre-monsoon clouds
        elif d == 13:
            current = 3  # overcast day
        elif d < 17:
            current = 1  # hazy recovery
        elif d < 20:
            current = 0  # clear
        elif d < 23:
            current = 2  # another cloudy spell
        elif d == 23:
            current = 3  # overcast
        elif d < 27:
            current = 1  # hazy
        else:
            current = 0  # clear end
        regimes.append(current)
    return regimes


def generate_solar_data():
    records = []
    regimes = generate_weather_pattern(DAYS)

    # Soiling accumulates, resets on day 10 and 20 (cleaning events)
    soiling_acc = 0.0

    for i in range(TOTAL_POINTS):
        ts = START + timedelta(minutes=i)
        hour = ts.hour + ts.minute / 60.0
        day_of_year = ts.timetuple().tm_yday
        day_index = i // (24 * 60)
        regime = regimes[day_index]

        # Solar elevation
        elev = solar_elevation(hour, day_of_year)

        # Cloud factor based on weather regime
        if regime == 0:  # clear
            cloud_factor = 0.95 + np.random.normal(0, 0.02)
        elif regime == 1:  # hazy
            cloud_factor = 0.82 + np.random.normal(0, 0.03)
        elif regime == 2:  # partly cloudy
            cloud_factor = 0.60 + np.random.normal(0, 0.08)
            # Intermittent clouds: random dips
            if np.random.random() < 0.15:
                cloud_factor -= 0.2 * np.random.random()
        else:  # overcast
            cloud_factor = 0.30 + np.random.normal(0, 0.05)

        # Afternoon convective buildup (pre-monsoon pattern)
        if 13 < hour < 17 and regime >= 1:
            cloud_factor -= 0.05 * np.sin((hour - 13) / 4 * np.pi)

        cloud_factor = np.clip(cloud_factor, 0.05, 1.0)

        # GHI
        ghi_clear = clear_sky_ghi(elev)
        ghi = ghi_clear * cloud_factor
        ghi += np.random.normal(0, 0.5) if ghi > 0 else 0
        ghi = max(0, ghi)
        lux = ghi * 120

        # Air temperature: diurnal cycle for pre-monsoon Gujarat
        # Varies by regime: clear days are hotter
        temp_min_base = 26 + day_index * 0.1  # gradual warming through April-May
        temp_max_base = 40 + day_index * 0.08
        if regime == 0:
            temp_max_adj = temp_max_base + 2
        elif regime == 1:
            temp_max_adj = temp_max_base
        elif regime == 2:
            temp_max_adj = temp_max_base - 3
        else:
            temp_max_adj = temp_max_base - 6

        temp_min = temp_min_base + np.random.normal(0, 0.3)
        temp_max = temp_max_adj + np.random.normal(0, 0.5)

        if hour < 5.5:
            temp = temp_min + (hour / 5.5) * 0.5
        elif hour < 15:
            temp = temp_min + (temp_max - temp_min) * ((hour - 5.5) / 9.5) ** 0.85
        else:
            temp = temp_max - (temp_max - temp_min) * ((hour - 15) / 14.5) ** 1.3
        temp += np.random.normal(0, 0.2)

        # Panel temperature
        panel_temp_rise = ghi * 0.03
        panel_temp = temp + panel_temp_rise + np.random.normal(0, 0.3)

        # Humidity: inverse of temp, higher on cloudy/overcast days
        hum_range = temp_max - temp_min if temp_max > temp_min else 1
        hum_base = 55 - 30 * ((temp - temp_min) / hum_range)
        if regime >= 2:
            hum_base += 15
        if regime == 3:
            hum_base += 10
        hum = hum_base + np.random.normal(0, 1.5)
        hum = np.clip(hum, 12, 80)

        # Pressure: semi-diurnal tide pattern
        pressure = 1007 + 2 * np.sin(2 * np.pi * hour / 12) + np.random.normal(0, 0.2)
        # Lower pressure on overcast days
        if regime >= 2:
            pressure -= 2

        # Soiling: accumulates daily, resets on cleaning days
        if i % (24 * 60) == 0:  # start of each day
            if day_index in [10, 20]:  # cleaning events
                soiling_acc = 0.01
            else:
                soiling_acc += 0.005 + np.random.normal(0, 0.001)
        soiling = np.clip(soiling_acc + np.random.normal(0, 0.003), 0, 0.20)

        # Power output
        efficiency = PANEL_EFFICIENCY * (1 + TEMP_COEFF * (panel_temp - 25))
        power_theoretical = efficiency * PANEL_AREA * ghi * (1 - soiling)
        power = power_theoretical * 0.90  # system losses
        current = power / PANEL_VOLTAGE if power > 0 else 0
        current += np.random.normal(0, 0.002) if current > 0 else 0
        current = max(0, current)
        power = current * PANEL_VOLTAGE

        records.append({
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "lux": round(lux, 0),
            "ghi": round(ghi, 2),
            "air_temp": round(temp, 1),
            "panel_temp": round(panel_temp, 1),
            "humidity": round(hum, 1),
            "pressure": round(pressure, 1),
            "current": round(current, 4),
            "power": round(power, 4),
            "soiling": round(soiling, 3),
        })

    return records


def generate_wind_data():
    records = []
    regimes = generate_weather_pattern(DAYS)
    base_direction = 225  # SW pre-monsoon

    for i in range(TOTAL_POINTS):
        ts = START + timedelta(minutes=i)
        hour = ts.hour + ts.minute / 60.0
        day_index = i // (24 * 60)
        regime = regimes[day_index]

        # Wind speed: diurnal + regime-dependent
        if hour < 7:
            speed_base = 1.5 + 0.3 * np.sin(hour / 7 * np.pi * 0.3)
        elif hour < 14:
            speed_base = 2.0 + 4.5 * ((hour - 7) / 7) ** 1.4
        elif hour < 18:
            speed_base = 6.0 + 1.5 * np.sin((hour - 14) / 4 * np.pi)
        else:
            speed_base = 6.0 - 4.0 * ((hour - 18) / 6) ** 0.8

        # Regime affects wind
        if regime == 0:
            speed_factor = 0.9
        elif regime == 1:
            speed_factor = 1.0
        elif regime == 2:
            speed_factor = 1.3  # pre-storm winds
        else:
            speed_factor = 1.5  # strong winds on overcast days

        # Day-to-day variation
        day_variation = 1.0 + 0.2 * np.sin(2 * np.pi * day_index / 7)
        speed_base *= speed_factor * day_variation

        # Turbulence/gusts
        gust = np.random.exponential(0.4)
        wind_speed = speed_base + gust
        wind_speed = np.clip(wind_speed, 0.5, 15)

        # Direction: SW base with slow multi-day drift
        dir_drift = 25 * np.sin(2 * np.pi * day_index / 10)
        dir_hourly = 10 * np.sin(2 * np.pi * hour / 24)  # sea-land breeze effect
        dir_turbulence = np.random.normal(0, 8)
        direction = base_direction + dir_drift + dir_hourly + dir_turbulence
        direction = direction % 360

        # Compass
        if direction < 22.5 or direction >= 337.5:
            compass = "N"
        elif direction < 67.5:
            compass = "NE"
        elif direction < 112.5:
            compass = "E"
        elif direction < 157.5:
            compass = "SE"
        elif direction < 202.5:
            compass = "S"
        elif direction < 247.5:
            compass = "SW"
        elif direction < 292.5:
            compass = "W"
        else:
            compass = "NW"

        # Temperature (same location as solar)
        temp_min_base = 26 + day_index * 0.1
        temp_max_base = 40 + day_index * 0.08
        if regime == 0:
            temp_max_adj = temp_max_base + 2
        elif regime == 1:
            temp_max_adj = temp_max_base
        elif regime == 2:
            temp_max_adj = temp_max_base - 3
        else:
            temp_max_adj = temp_max_base - 6

        temp_min = temp_min_base + np.random.normal(0, 0.3)
        temp_max = temp_max_adj + np.random.normal(0, 0.5)

        if hour < 5.5:
            temp = temp_min + (hour / 5.5) * 0.5
        elif hour < 15:
            temp = temp_min + (temp_max - temp_min) * ((hour - 5.5) / 9.5) ** 0.85
        else:
            temp = temp_max - (temp_max - temp_min) * ((hour - 15) / 14.5) ** 1.3
        temp += np.random.normal(0, 0.2)

        # Humidity
        hum_range = temp_max - temp_min if temp_max > temp_min else 1
        hum_base = 55 - 30 * ((temp - temp_min) / hum_range)
        if regime >= 2:
            hum_base += 15
        if regime == 3:
            hum_base += 10
        hum = hum_base + np.random.normal(0, 1.5)
        hum = np.clip(hum, 12, 80)

        # Pressure
        pressure = 1007 + 2 * np.sin(2 * np.pi * hour / 12) + np.random.normal(0, 0.2)
        if regime >= 2:
            pressure -= 2

        # Air density
        rho = (pressure * 100) / (287.05 * (temp + 273.15))

        # Wind power
        wind_power = 0.5 * rho * 1.0 * 0.35 * (wind_speed ** 3)

        # Vibration: correlated with wind speed
        vibration = 0.1 + wind_speed * 0.12 + np.random.exponential(0.08)
        if wind_speed > 8:
            vibration += np.random.exponential(0.4)

        records.append({
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "wind_speed": round(wind_speed, 1),
            "wind_direction": round(direction, 0),
            "compass": compass,
            "air_temp": round(temp, 1),
            "humidity": round(hum, 1),
            "pressure": round(pressure, 1),
            "vibration": round(vibration, 2),
            "wind_power": round(wind_power, 2),
        })

    return records


def save_to_db(solar_data, wind_data):
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE solar (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            lux REAL, ghi REAL, air_temp REAL, panel_temp REAL,
            humidity REAL, pressure REAL, current REAL, power REAL, soiling REAL
        )
    """)
    c.execute("""
        CREATE TABLE wind (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            wind_speed REAL, wind_direction REAL, compass TEXT,
            air_temp REAL, humidity REAL, pressure REAL,
            vibration REAL, wind_power REAL
        )
    """)

    c.executemany("""
        INSERT INTO solar (timestamp, lux, ghi, air_temp, panel_temp, humidity, pressure, current, power, soiling)
        VALUES (:timestamp, :lux, :ghi, :air_temp, :panel_temp, :humidity, :pressure, :current, :power, :soiling)
    """, solar_data)

    c.executemany("""
        INSERT INTO wind (timestamp, wind_speed, wind_direction, compass, air_temp, humidity, pressure, vibration, wind_power)
        VALUES (:timestamp, :wind_speed, :wind_direction, :compass, :air_temp, :humidity, :pressure, :vibration, :wind_power)
    """, wind_data)

    conn.commit()
    conn.close()


if __name__ == "__main__":
    print("=" * 60)
    print("  VidyutDrishti — Synthetic Data Generator")
    print("=" * 60)
    print(f"  Location: Nadiad, Gujarat ({LATITUDE}N, {LONGITUDE}E)")
    print(f"  Period: {START.strftime('%Y-%m-%d')} to {(START + timedelta(days=DAYS)).strftime('%Y-%m-%d')}")
    print(f"  Resolution: {INTERVAL_MIN} minute")
    print(f"  Total points per node: {TOTAL_POINTS:,}")
    print()

    print("Generating solar data...")
    solar = generate_solar_data()
    print(f"  {len(solar):,} records")

    print("Generating wind data...")
    wind = generate_wind_data()
    print(f"  {len(wind):,} records")

    print(f"\nSaving to: {DB_PATH}")
    save_to_db(solar, wind)

    print(f"Database size: {os.path.getsize(DB_PATH) / 1024 / 1024:.1f} MB")
    print("\nDone.")
