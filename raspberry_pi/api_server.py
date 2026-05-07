"""
VidyutDrishti — API Server (runs on Raspberry Pi)
Serves live sensor data from SQLite over HTTP.
Dashboard on laptop fetches from this.
"""

from flask import Flask, jsonify
import sqlite3
import os

app = Flask(__name__)

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "vidyutdrishti.db")


def query_db(query, args=()):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.execute(query, args)
    rows = [dict(row) for row in cur.fetchall()]
    conn.close()
    return rows


@app.route("/api/solar/latest", methods=["GET"])
def solar_latest():
    rows = query_db("SELECT * FROM solar ORDER BY id DESC LIMIT 200")
    rows.reverse()
    return jsonify(rows)


@app.route("/api/wind/latest", methods=["GET"])
def wind_latest():
    rows = query_db("SELECT * FROM wind ORDER BY id DESC LIMIT 200")
    rows.reverse()
    return jsonify(rows)


@app.route("/api/solar/last", methods=["GET"])
def solar_last():
    rows = query_db("SELECT * FROM solar ORDER BY id DESC LIMIT 1")
    return jsonify(rows[0] if rows else {})


@app.route("/api/wind/last", methods=["GET"])
def wind_last():
    rows = query_db("SELECT * FROM wind ORDER BY id DESC LIMIT 1")
    return jsonify(rows[0] if rows else {})


@app.route("/api/status", methods=["GET"])
def status():
    solar_count = query_db("SELECT COUNT(*) as count FROM solar")[0]["count"]
    wind_count = query_db("SELECT COUNT(*) as count FROM wind")[0]["count"]
    solar_last = query_db("SELECT timestamp FROM solar ORDER BY id DESC LIMIT 1")
    wind_last = query_db("SELECT timestamp FROM wind ORDER BY id DESC LIMIT 1")
    return jsonify({
        "status": "running",
        "solar_records": solar_count,
        "wind_records": wind_count,
        "solar_last_update": solar_last[0]["timestamp"] if solar_last else None,
        "wind_last_update": wind_last[0]["timestamp"] if wind_last else None,
    })


if __name__ == "__main__":
    print("=" * 50)
    print("  VidyutDrishti API Server")
    print("=" * 50)
    print(f"  Database: {DB_PATH}")
    print(f"  Endpoints:")
    print(f"    GET /api/solar/latest  — last 200 solar readings")
    print(f"    GET /api/wind/latest   — last 200 wind readings")
    print(f"    GET /api/solar/last    — single latest solar reading")
    print(f"    GET /api/wind/last     — single latest wind reading")
    print(f"    GET /api/status        — system status")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5000)
