#!/usr/bin/env python3
"""
| File: plot_flight.py
| Author: Steven Martinez Jara
| License: BSD-3-Clause
| Description: Turn an autonomous_flight.py telemetry CSV into a self-contained
|              HTML report: 3D trajectory (orbit + playback), roll/airspeed/
|              altitude traces, and a computed turn-radius feasibility check.
|
| Usage (from the repo root):
|   python3 examples/yoy_trainer_v2/scripts/plot_flight.py                  # newest log
|   python3 examples/yoy_trainer_v2/scripts/plot_flight.py <log.csv>
|   python3 examples/yoy_trainer_v2/scripts/plot_flight.py <log.csv> -o out.html
|
| The CSV needs the columns written by autonomous_flight.py:
|   time_s, roll_deg, pitch_deg, airspeed, altitude, lat, lon
| and, in logs written after 2026-09-23, throttle_pct and alt_agl. Older logs
| still plot, but without a throttle trace or a known ground level.
"""

import argparse
import csv
import glob
import json
import math
import os

# ── Design references (from the paper / vehicle config) ──────────────────────
DESIGN_CRUISE   = 15.0     # m/s
ROLL_LIMIT_DEG  = 35.0     # deg — matches ROLL_LIMIT_DEG in autonomous_flight.py
LOITER_RADIUS   = 80.0     # m   — matches LOITER_RADIUS in autonomous_flight.py
TRIM_THROTTLE   = 58.0     # %   — matches TRIM_THROTTLE in autonomous_flight.py
WING_AREA       = 0.1699   # m²
CD_0            = 0.058    # CFD v2 (v1: 0.104)
RHO             = 1.225    # kg/m³
G               = 9.81

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR    = os.path.join(SCRIPT_DIR, "..", "flight_logs")

# Waypoints the mission script commands — kept in sync with autonomous_flight.py
WAYPOINTS = {
    "WP0":  (-35.36100000, 149.16900000),
    "WP1":  (-35.36327729, 149.16660000),
    "LOIT": (-35.36293680, 149.16507765),
}


def load(path):
    with open(path) as fh:
        rows = [r for r in csv.DictReader(fh) if float(r["lat"]) != 0.0]
    if not rows:
        raise SystemExit(f"{path}: no rows with a GPS fix — nothing to plot.")
    return rows


def build(rows, max_points=800):
    """Project to a local ENU plane and downsample for the browser."""
    lat0, lon0 = float(rows[0]["lat"]), float(rows[0]["lon"])
    k  = 111320.0
    kc = k * math.cos(math.radians(lat0))

    def to_xy(lat, lon):
        return ((lon - lon0) * kc, (lat - lat0) * k)

    has_thr = "throttle_pct" in rows[0]
    has_agl = "alt_agl" in rows[0]

    step = max(1, len(rows) // max_points)
    pts = []
    for r in rows[::step]:
        x, y = to_xy(float(r["lat"]), float(r["lon"]))
        pts.append([
            round(float(r["time_s"]), 1), round(x, 1), round(y, 1),
            round(float(r["altitude"]), 1), round(float(r["roll_deg"]), 1),
            round(float(r["pitch_deg"]), 1), round(float(r["airspeed"]), 1),
            round(float(r["throttle_pct"])) if has_thr else None,
        ])

    wps = {n: [round(v, 1) for v in to_xy(*ll)] for n, ll in WAYPOINTS.items()}

    # Full-resolution statistics (not the downsampled set)
    spd  = [float(r["airspeed"]) for r in rows]
    alt  = [float(r["altitude"]) for r in rows]
    roll = [float(r["roll_deg"]) for r in rows]
    pit  = [float(r["pitch_deg"]) for r in rows]

    lx, ly = to_xy(*WAYPOINTS["LOIT"])
    d_loit = [math.hypot(to_xy(float(r["lat"]), float(r["lon"]))[0] - lx,
                         to_xy(float(r["lat"]), float(r["lon"]))[1] - ly)
              for r in rows]

    v_avg   = sum(spd) / len(spd)
    r_min   = v_avg ** 2 / (G * math.tan(math.radians(ROLL_LIMIT_DEG)))
    q_ratio = (v_avg / DESIGN_CRUISE) ** 2
    if has_agl:
        # MSL minus height above home = elevation of the arming point. Median,
        # because the first samples can carry agl = 0 before a fix arrives.
        offs   = sorted(a - float(r["alt_agl"]) for a, r in zip(alt, rows))
        ground = offs[len(offs) // 2]
    else:
        # Old logs: no ground reference. The lowest sample is only a lower
        # bound on height, NOT the ground; with --loiter-now the log starts
        # after the climb, so this puts the ground tens of metres too high.
        ground = min(alt)

    meta = {
        "wps": wps,
        "altMin": round(min(alt), 1), "altMax": round(max(alt), 1),
        "ground": round(ground, 1), "groundKnown": has_agl,
        "hasThr": has_thr, "trimThr": TRIM_THROTTLE,
        "spdAvg": round(v_avg, 1), "spdMin": round(min(spd), 1), "spdMax": round(max(spd), 1),
        "rollMax": round(max(abs(v) for v in roll), 1),
        "pitchMax": round(max(abs(v) for v in pit), 1),
        "loitMin": round(min(d_loit)), "loitAvg": round(sum(d_loit) / len(d_loit)),
        "tEnd": round(float(rows[-1]["time_s"]), 1),
        "samples": len(rows),
        "rMin": round(r_min), "qRatio": round(q_ratio, 2),
        "cruise": DESIGN_CRUISE, "rollLimit": ROLL_LIMIT_DEG, "loiterRad": LOITER_RADIUS,
        "thrCruise": round(100 * CD_0 * 0.5 * RHO * DESIGN_CRUISE ** 2 * WING_AREA / 15.0),
        "feasible": r_min <= LOITER_RADIUS,
    }
    return {"meta": meta, "pts": pts}


TEMPLATE = os.path.join(SCRIPT_DIR, "report_template.html")


def render(data, name, out_path):
    with open(TEMPLATE) as fh:
        html = fh.read()
    html = (html
            .replace("__FLIGHT_DATA__", json.dumps(data, separators=(",", ":")))
            .replace("__LOG_NAME__", name))
    with open(out_path, "w") as fh:
        fh.write(html)


def main():
    ap = argparse.ArgumentParser(description="Plot an autonomous flight log as HTML.")
    ap.add_argument("csv", nargs="?", help="log to plot (default: newest in flight_logs/)")
    ap.add_argument("-o", "--out", help="output HTML (default: alongside the CSV)")
    args = ap.parse_args()

    path = args.csv
    if not path:
        found = sorted(glob.glob(os.path.join(LOG_DIR, "auto_*.csv")))
        if not found:
            raise SystemExit(f"No auto_*.csv found in {os.path.normpath(LOG_DIR)}")
        path = found[-1]
        print(f"Using newest log: {os.path.basename(path)}")

    rows = load(path)
    data = build(rows)
    out  = args.out or os.path.splitext(path)[0] + ".html"
    render(data, os.path.basename(path), out)

    m = data["meta"]
    print(f"  {m['samples']} muestras · {m['tEnd']} s")
    print(f"  Velocidad media {m['spdAvg']} m/s   (diseño {m['cruise']} m/s)")
    print(f"  Alabeo maximo   {m['rollMax']}°     (limite {m['rollLimit']}°)")
    print(f"  Radio de giro minimo {m['rMin']} m  vs radio de loiter {m['loiterRad']} m"
          f"  -> {'ALCANZABLE' if m['feasible'] else 'IMPOSIBLE'}")
    print(f"  Distancia minima al punto de loiter: {m['loitMin']} m")
    if m["groundKnown"]:
        print(f"  Altura minima sobre el suelo: {m['altMin'] - m['ground']:.1f} m")
    else:
        print("  (log sin alt_agl: el suelo del reporte es la cota minima, no el real)")
    print(f"\n  Reporte: {out}")


if __name__ == "__main__":
    main()
