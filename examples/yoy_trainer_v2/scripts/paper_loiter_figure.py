#!/usr/bin/env python3
"""
Fig. "loiter_quant" of the ETCM 2026 paper: sustained loiter in both turn
directions, (a) orbit radius and (b) bank angle, with the settled window
(last 60 s) shaded.

Radius comes from the GPS track in the mission CSV, measured from the centre
of a circle fitted to the settled window. Bank angle is the TRUE attitude from
Pegasus (auto_<ts>_forces.csv.gz), not ArduPilot's EKF estimate, which reads
0.3-1.5 deg to the left of it; the paper's table uses the same source.

Usage (from the repo root):
  python3 examples/yoy_trainer_v2/scripts/paper_loiter_figure.py \
      --cw auto_20260925_191716 --ccw auto_20260925_190119 -o loiter_quant.png

The two logs above are the published runs; --logs points the script at a
folder other than ../flight_logs.
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "flight_logs")
SETTLED = 60.0        # s, settled window at the end of each run
T_START = 40.0        # s, first sample shown: LOITER entry transient is over the scale before this

BLUE, VERMILION, ORANGE, SHADE = "#0072B2", "#D55E00", "#E69F00", "#009E73"


def load_mission(name):
    with open(os.path.join(LOG_DIR, name + ".csv")) as fh:
        rows = list(csv.DictReader(fh))
    col = lambda k: np.array([float(r[k]) if r[k] else np.nan for r in rows])
    t, lat, lon, spd = col("time_s"), col("lat"), col("lon"), col("airspeed")
    ok = lat != 0
    x = (lon - lon[ok][0]) * 111320 * np.cos(np.radians(lat[ok][0]))
    y = (lat - lat[ok][0]) * 111320
    m = ok & (t > t[-1] - SETTLED)
    A = np.c_[2 * x[m], 2 * y[m], np.ones(m.sum())]
    c = np.linalg.lstsq(A, x[m] ** 2 + y[m] ** 2, rcond=None)[0]
    r = np.hypot(x - c[0], y - c[1])
    return {"t": t[ok], "r": r[ok], "spd": spd, "t_all": t}


def load_truth(name, mission):
    """Pegasus state, re-timed onto the mission clock by matching the instant
    each log first sees 5 m/s of airspeed on the takeoff roll."""
    d = np.genfromtxt(os.path.join(LOG_DIR, name + "_forces.csv.gz"), delimiter=",", names=True)
    tt = d["timestamp"] - d["timestamp"][0]
    V = np.sqrt(d["u"] ** 2 + d["v"] ** 2 + d["w"] ** 2)
    t5_truth = tt[np.argmax(V > 5)]
    t5_mission = mission["t_all"][np.argmax(mission["spd"] > 5)]
    t = tt - t5_truth + t5_mission
    keep = t <= mission["t_all"][-1]
    # 250 Hz -> 25 Hz, plenty for a 260 s trace
    return {"t": t[keep][::10], "roll": d["roll_deg"][keep][::10]}


def settled(t, v, t_end):
    m = t > t_end - SETTLED
    return v[m].mean(), v[m].std()


def main():
    global LOG_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--cw", required=True)
    ap.add_argument("--ccw", required=True)
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--logs", default=LOG_DIR,
                    help="folder holding the auto_<ts> logs (default: ../flight_logs)")
    a = ap.parse_args()
    LOG_DIR = a.logs

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
        "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    })
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.84, 6.44), dpi=200, sharex=True)

    runs = []
    for name, label, color in ((a.cw, "Clockwise", BLUE), (a.ccw, "Counter-clockwise", VERMILION)):
        mis = load_mission(name)
        tru = load_truth(name, mis)
        runs.append((label, color, mis, tru))

    t_end = min(r[2]["t"][-1] for r in runs)
    for ax in (ax1, ax2):
        ax.axvspan(t_end - SETTLED, t_end, color=SHADE, alpha=0.10, lw=0)

    notes_r, notes_b = [], []
    for label, color, mis, tru in runs:
        m = mis["t"] >= T_START
        ax1.plot(mis["t"][m], mis["r"][m], color=color, lw=1.8, label=label)
        R, sR = settled(mis["t"], mis["r"], mis["t"][-1])
        ax1.axhline(R, color=color, ls="--", lw=1.2, alpha=0.9)
        notes_r.append(rf"{label.lower()}: $\bar{{R}}$ = {R:.1f} $\pm$ {sR:.2f} m ({100 * sR / R:.2f}%)")

        mb = tru["t"] >= T_START
        ax2.plot(tru["t"][mb], tru["roll"][mb], color=color, lw=1.8, label=label)
        phi, sphi = settled(tru["t"], tru["roll"], mis["t"][-1])
        ax2.axhline(phi, color=color, ls="--", lw=1.2, alpha=0.9)
        notes_b.append(rf"{label.lower()}: $\bar{{\varphi}}$ = {phi:+.2f}$^\circ$ $\pm$ {sphi:.2f}$^\circ$")

    ax1.axhline(150, color="0.35", ls=":", lw=1.2)
    ax1.text(T_START + 2, 150.8, "commanded 150 m", color="0.35", fontsize=10, va="bottom")
    ax1.set_title("(a) Orbit radius", loc="left")
    ax1.set_ylabel("Orbit radius [m]")
    ax1.legend(loc="upper center", frameon=False, fontsize=10, ncol=2)
    ax1.text(0.98, 0.70, "settled:\n" + "\n".join(notes_r), transform=ax1.transAxes,
             ha="right", va="center", fontsize=9.5, color="0.3", linespacing=1.4,
             bbox=dict(fc="white", ec="none", alpha=0.8, pad=2))

    ax2.axhline(0, color="0.6", lw=0.8)
    ax2.set_title("(b) Bank angle", loc="left")
    ax2.set_ylabel(r"Bank angle $\varphi$ [deg]")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylim(-16, 16)
    ax2.text(0.98, 0.5, "settled:\n" + "\n".join(notes_b), transform=ax2.transAxes,
             ha="right", va="center", fontsize=9.5, color="0.3", linespacing=1.4,
             bbox=dict(fc="white", ec="none", alpha=0.8, pad=2))
    ax2.set_xlim(T_START, t_end + 2)

    fig.tight_layout()
    fig.savefig(a.out, dpi=200)
    print(f"saved {a.out}")
    for n in notes_r + notes_b:
        print("  " + n.replace("$", ""))


if __name__ == "__main__":
    main()
