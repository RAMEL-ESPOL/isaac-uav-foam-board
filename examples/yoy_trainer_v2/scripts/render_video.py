#!/usr/bin/env python3
"""
Render the isometric flight playback of a log to an MP4, for slides / editing.

Same projection, colours and vertical exaggeration as the "Iso" view of the
HTML report (plot_flight.py + report_template.html), but at full log
resolution and with a HUD. Pure Python: OpenCV draws and encodes (its wheel
bundles ffmpeg), Pillow only renders the text.

Usage (from the repo root):
  python3 examples/yoy_trainer_v2/scripts/render_video.py              # newest log
  python3 examples/yoy_trainer_v2/scripts/render_video.py <log.csv> --speed 5 --spin
Options:
  --speed N    playback speed vs real time (default 5: 300 s of flight -> 60 s)
  --fps N      frames per second (default 30)
  --size WxH   resolution (default 1920x1080)
  --spin       rotate the camera slowly around the orbit instead of a fixed iso
  --light      light background (default dark, like the report)
  --radius R   commanded loiter radius drawn as a dashed circle (default: from
               the log's .json sidecar, else 80 m). The circle is centred on the
               fitted centre of the settled orbit, which is where LOITER
               actually engaged with --loiter-now.
"""

import argparse
import csv
import glob
import json
import math
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(SCRIPT_DIR, "..", "flight_logs")
ROLL_LIMIT_DEG = 35.0          # matches plot_flight.py / autonomous_flight.py

# BGR, from the CSS custom properties of report_template.html
THEMES = {
    "dark":  {"ground": "#0A0F14", "line": "#1E2C38", "ink": "#D2DCE5", "dim": "#8B9AA8",
              "faint": "#5C6B79", "trace": "#FF9E2C", "nominal": "#3FBFB4",
              "alarm": "#F0483E", "violet": "#8B7BD8", "panel": "#111A22"},
    "light": {"ground": "#F5F4F0", "line": "#DEDAD2", "ink": "#141C24", "dim": "#5A6672",
              "faint": "#8B96A1", "trace": "#C96A00", "nominal": "#0E7C74",
              "alarm": "#C4241C", "violet": "#5B4BA8", "panel": "#FFFFFF"},
}


def hex_bgr(h):
    return (int(h[5:7], 16), int(h[3:5], 16), int(h[1:3], 16))


def hex_rgb(h):
    return (int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16))


def font(size, bold=False):
    names = (["DejaVuSansMono-Bold.ttf", "LiberationMono-Bold.ttf"] if bold
             else ["DejaVuSansMono.ttf", "LiberationMono-Regular.ttf"])
    for n in names:
        for d in ("/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/truetype/liberation"):
            p = os.path.join(d, n)
            if os.path.exists(p):
                return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def load(path):
    with open(path) as fh:
        rows = [r for r in csv.DictReader(fh) if float(r["lat"]) != 0.0]
    if not rows:
        raise SystemExit(f"{path}: no rows with a GPS fix.")
    col = lambda k: np.array([float(r[k]) if r.get(k) not in (None, "") else np.nan
                              for r in rows])
    lat, lon = col("lat"), col("lon")
    k = 111320.0
    d = {"t": col("time_s"), "roll": col("roll_deg"), "pitch": col("pitch_deg"),
         "spd": col("airspeed"), "alt": col("altitude"),
         "x": (lon - lon[0]) * k * math.cos(math.radians(lat[0])),
         "y": (lat - lat[0]) * k}
    d["thr"] = col("throttle_pct") if "throttle_pct" in rows[0] else None
    d["thrust"] = col("thrust_n") if "thrust_n" in rows[0] else None
    if "alt_agl" in rows[0]:
        d["agl"] = col("alt_agl")
        d["ground"] = float(np.median(d["alt"] - d["agl"]))
    else:
        d["ground"] = float(d["alt"].min())
        d["agl"] = d["alt"] - d["ground"]
    return d


def fit_circle(x, y):
    A = np.c_[2 * x, 2 * y, np.ones_like(x)]
    c = np.linalg.lstsq(A, x ** 2 + y ** 2, rcond=None)[0]
    r = np.hypot(x - c[0], y - c[1])
    return c[0], c[1], r.mean(), r.std()


class Camera:
    def __init__(self, d, W, H, az=-0.62, el=0.52):
        self.az, self.el, self.W, self.H = az, el, W, H
        bx, by = (d["x"].min(), d["x"].max()), (d["y"].min(), d["y"].max())
        self.cx, self.cy = sum(bx) / 2, sum(by) / 2
        self.bx, self.by = bx, by
        span = max(bx[1] - bx[0], by[1] - by[0], 200)
        self.grid = 250 if span > 1200 else 100 if span > 400 else 50
        self.zex = max(2, min(14, span / max(12, d["agl"].max())))
        self.scale, self.ox, self.oy = 1.0, 0.0, 0.0
        self.panel_w = int(380 * H / 1080)
        self.d = d

    def proj(self, x, y, z):
        ca, sa, ce, se = math.cos(self.az), math.sin(self.az), math.cos(self.el), math.sin(self.el)
        X = x * ca - y * sa
        Y = x * sa + y * ca
        return X * self.scale + self.ox, -(Y * se + z * ce) * self.scale + self.oy

    def world(self, x, y, agl):
        return self.proj(x - self.cx, y - self.cy, agl * self.zex)

    def frame(self, top_margin=0.16):
        """Fit trajectory + shadow + grid bounds, like frame() in the report.
        The bounds are taken over every azimuth when spinning, see fit_all()."""
        self.scale, self.ox, self.oy = 1.0, 0.0, 0.0
        d = self.d
        pts = [self.world(x, y, a) for x, y, a in zip(d["x"][::5], d["y"][::5], d["agl"][::5])]
        pts += [self.world(x, y, 0) for x, y in zip(d["x"][::5], d["y"][::5])]
        pts += [self.world(gx, gy, 0) for gx in self.bx for gy in self.by]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        return min(xs), max(xs), min(ys), max(ys)

    def fit_all(self, azimuths, top_margin=0.16):
        az0 = self.az
        x0 = y0 = math.inf
        x1 = y1 = -math.inf
        for a in azimuths:
            self.az = a
            a0, a1, b0, b1 = self.frame()
            x0, x1, y0, y1 = min(x0, a0), max(x1, a1), min(y0, b0), max(y1, b1)
        self.az = az0
        usable_h = self.H * (1 - top_margin - 0.06)
        usable_w = self.W - self.panel_w           # right-hand HUD panel
        self.scale = min(usable_w * 0.9 / max(x1 - x0, 1), usable_h / max(y1 - y0, 1))
        self.ox = usable_w / 2 - (x0 + x1) / 2 * self.scale
        self.oy = self.H * top_margin + usable_h / 2 - (y0 + y1) / 2 * self.scale


def roll_color(r, th):
    a = abs(r)
    return hex_bgr(th["alarm"] if a > ROLL_LIMIT_DEG else
                   th["trace"] if a > ROLL_LIMIT_DEG * 0.6 else th["nominal"])


def color_runs(roll, th):
    """Split sample indices into runs of equal roll colour -> few polylines."""
    cols = [roll_color(r, th) for r in roll]
    runs, start = [], 0
    for i in range(1, len(cols) + 1):
        if i == len(cols) or cols[i] != cols[start]:
            runs.append((start, i, cols[start]))
            start = i
    return runs


def draw_scene(img, cam, d, head, th, circle, runs):
    ip = lambda p: (int(round(p[0] * 4)), int(round(p[1] * 4)))   # 2-bit subpixel
    SH, AA = 2, cv2.LINE_AA
    line_c, violet = hex_bgr(th["line"]), hex_bgr(th["violet"])

    # grid
    g = cam.grid
    g0x, g1x = math.floor(cam.bx[0] / g) * g, math.ceil(cam.bx[1] / g) * g
    g0y, g1y = math.floor(cam.by[0] / g) * g, math.ceil(cam.by[1] / g) * g
    for gx in np.arange(g0x, g1x + 1, g):
        cv2.line(img, ip(cam.world(gx, g0y, 0)), ip(cam.world(gx, g1y, 0)), line_c, 1, AA, SH)
    for gy in np.arange(g0y, g1y + 1, g):
        cv2.line(img, ip(cam.world(g0x, gy, 0)), ip(cam.world(g1x, gy, 0)), line_c, 1, AA, SH)

    # commanded loiter circle (dashed), on the ground
    if circle:
        ccx, ccy, R = circle
        n = 96
        for a in range(0, n, 2):
            t0, t1 = a / n * 2 * math.pi, (a + 1) / n * 2 * math.pi
            p0 = cam.world(ccx + R * math.cos(t0), ccy + R * math.sin(t0), 0)
            p1 = cam.world(ccx + R * math.cos(t1), ccy + R * math.sin(t1), 0)
            cv2.line(img, ip(p0), ip(p1), violet, 2, AA, SH)

    # ground shadow of the flown path
    shadow = np.array([ip(cam.world(x, y, 0)) for x, y in zip(d["x"][:head + 1], d["y"][:head + 1])],
                      np.int32)
    if len(shadow) > 1:
        cv2.polylines(img, [shadow], False, line_c, 3, AA, SH)

    # drop lines every ~1.3 s
    for i in range(0, head + 1, 26):
        cv2.line(img, ip(cam.world(d["x"][i], d["y"][i], 0)),
                 ip(cam.world(d["x"][i], d["y"][i], d["agl"][i])), line_c, 1, AA, SH)

    # flown path, coloured by |roll|
    path = np.array([ip(cam.world(x, y, a)) for x, y, a in
                     zip(d["x"][:head + 1], d["y"][:head + 1], d["agl"][:head + 1])], np.int32)
    for s, e, c in runs:
        if s > head:
            break
        seg = path[max(0, s - 1):min(e, head + 1)]
        if len(seg) > 1:
            cv2.polylines(img, [seg], False, c, 3, AA, SH)

    # aircraft marker, pointed along the projected track
    c = roll_color(d["roll"][head], th)
    q = cam.world(d["x"][head], d["y"][head], d["agl"][head])
    j = max(0, head - 4)
    pq = cam.world(d["x"][j], d["y"][j], d["agl"][j])
    ang = math.atan2(q[1] - pq[1], q[0] - pq[0]) if head > 0 else 0.0
    ca, sa = math.cos(ang), math.sin(ang)
    shape = [(16, 0), (-12, 11), (-6, 0), (-12, -11)]
    tri = np.array([ip((q[0] + u * ca - v * sa, q[1] + u * sa + v * ca)) for u, v in shape], np.int32)
    gq = cam.world(d["x"][head], d["y"][head], 0)
    cv2.line(img, ip(q), ip(gq), c, 1, AA, SH)
    cv2.circle(img, ip(gq), 4 * 4, c, 1, AA, SH)
    cv2.fillPoly(img, [tri], c, AA, SH)

    # takeoff point
    h = cam.world(d["x"][0], d["y"][0], 0)
    cv2.circle(img, ip(h), 5 * 4, hex_bgr(th["faint"]), -1, AA, SH)
    return h


def hud(img, d, head, th, cam, title, subtitle, home_px, circle):
    W, H = img.shape[1], img.shape[0]
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    dr = ImageDraw.Draw(pil)
    s = H / 1080
    fT, fS, fL, fV = font(int(34 * s), True), font(int(20 * s)), font(int(16 * s)), font(int(40 * s), True)
    ink, dim, faint = hex_rgb(th["ink"]), hex_rgb(th["dim"]), hex_rgb(th["faint"])
    m = int(48 * s)
    dr.text((m, int(36 * s)), title, font=fT, fill=ink)
    dr.text((m, int(82 * s)), subtitle, font=fS, fill=dim)

    fields = [("t", f"{d['t'][head]:6.1f}", "s"),
              ("ALTURA", f"{d['agl'][head]:5.1f}", "m"),
              ("VELOCIDAD", f"{d['spd'][head]:5.1f}", "m/s"),
              ("ALABEO", f"{d['roll'][head]:+6.1f}", "°")]
    if d["thr"] is not None:
        fields.append(("ACELERADOR", f"{d['thr'][head]:4.0f}", "%"))
    if d["thrust"] is not None and not np.isnan(d["thrust"][head]):
        fields.append(("EMPUJE", f"{d['thrust'][head]:5.2f}", "N"))
    # right-hand panel: one reading per row
    x = W - cam.panel_w + int(40 * s)
    rowh = int(96 * s)
    y = (H - rowh * len(fields)) // 2
    dr.line([(x - int(28 * s), y), (x - int(28 * s), y + rowh * len(fields) - int(20 * s))],
            fill=hex_rgb(th["line"]), width=max(1, int(2 * s)))
    for lbl, val, unit in fields:
        dr.text((x, y), lbl, font=fL, fill=faint)
        col = hex_rgb(th["alarm"] if lbl == "ALABEO" and abs(d["roll"][head]) > ROLL_LIMIT_DEG else th["ink"])
        dr.text((x, y + int(22 * s)), val.strip(), font=fV, fill=col)
        vw = dr.textlength(val.strip(), font=fV)
        dr.text((x + vw + int(8 * s), y + int(40 * s)), unit, font=fS, fill=dim)
        y += rowh

    dr.text((home_px[0], home_px[1] + int(14 * s)), "DESPEGUE", font=fL, fill=faint, anchor="mt")
    legend = [(th["nominal"], f"|φ| ≤ {ROLL_LIMIT_DEG * 0.6:.0f}°"),
              (th["trace"], f"|φ| ≤ {ROLL_LIMIT_DEG:.0f}°"),
              (th["alarm"], f"|φ| > {ROLL_LIMIT_DEG:.0f}° (límite)")]
    if circle:
        legend.append((th["violet"], f"radio comandado {circle[2]:.0f} m"))
    y = H - m - int(24 * s) * len(legend)
    for c, txt in legend:
        dr.line([(m, y + int(10 * s)), (m + int(26 * s), y + int(10 * s))], fill=hex_rgb(c), width=max(2, int(3 * s)))
        dr.text((m + int(36 * s), y), txt, font=fL, fill=dim)
        y += int(24 * s)
    dr.text((W - m, H - m), f"rejilla {cam.grid} m · altitud ×{cam.zex:.1f}",
            font=fL, fill=faint, anchor="rd")
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def open_writer(path, fps, W, H):
    # H.264 if this OpenCV build has an encoder for it, else MPEG-4 Part 2,
    # which every editor (DaVinci, Premiere, Kdenlive, CapCut) also imports.
    for cc in ("avc1", "mp4v"):
        w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*cc), fps, (W, H))
        if w.isOpened():
            return w, cc
    raise SystemExit("OpenCV could not open an MP4 writer.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("csv", nargs="?")
    ap.add_argument("-o", "--out")
    ap.add_argument("--speed", type=float, default=5.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--spin", action="store_true")
    ap.add_argument("--light", action="store_true")
    ap.add_argument("--radius", type=float)
    a = ap.parse_args()

    path = a.csv or max(glob.glob(os.path.join(LOG_DIR, "auto_*.csv")), key=os.path.getmtime)
    name = os.path.splitext(os.path.basename(path))[0]
    out = a.out or os.path.splitext(path)[0] + ".mp4"
    W, H = (int(v) for v in a.size.lower().split("x"))
    th = THEMES["light" if a.light else "dark"]

    d = load(path)
    side = {}
    if os.path.exists(os.path.splitext(path)[0] + ".json"):
        with open(os.path.splitext(path)[0] + ".json") as fh:
            side = json.load(fh)
    R = a.radius or side.get("loiter_radius_m", 80.0)
    direction = side.get("loiter_dir")

    # Settled orbit = last 60 s; its fitted centre is where LOITER engaged.
    m = d["t"] > d["t"][-1] - 60
    ccx, ccy, r_fit, r_sd = fit_circle(d["x"][m], d["y"][m])
    circle = (ccx, ccy, R)

    sense = {1: "horario", -1: "antihorario"}.get(direction, "")
    title = f"YOY Trainer v2 — loiter {sense}".rstrip(" —") if "--loiter-now" in side.get("argv", []) \
        else "YOY Trainer v2 — misión autónoma"
    thrust_txt = (f" · empuje máx. {side['sim_config']['prop_max_thrust_n']} N"
                  if side.get("sim_config") else "")
    subtitle = (f"SITL ArduPlane + Isaac Sim · {name} · radio medido {r_fit:.1f} m "
                f"(comandado {R:.0f} m){thrust_txt} · ×{a.speed:g}")

    total = d["t"][-1] - d["t"][0]
    n_frames = int(total / a.speed * a.fps) + 1
    hold = a.fps * 2                                   # 2 s still at the end
    az_of = (lambda k: -0.62 + 2 * math.pi * 0.35 * k / n_frames) if a.spin else (lambda k: -0.62)

    cam = Camera(d, W, H)
    cam.fit_all(np.linspace(-0.62, -0.62 + 2 * math.pi * 0.35, 24) if a.spin else [-0.62])
    runs = color_runs(d["roll"], th)
    bg = np.empty((H, W, 3), np.uint8)
    bg[:] = hex_bgr(th["ground"])

    writer, codec = open_writer(out, a.fps, W, H)
    for k in range(n_frames + hold):
        kk = min(k, n_frames - 1)
        t_now = d["t"][0] + kk * a.speed / a.fps
        head = int(np.searchsorted(d["t"], t_now, side="right") - 1)
        head = max(0, min(head, len(d["t"]) - 1))
        cam.az = az_of(kk)
        img = bg.copy()
        home_px = draw_scene(img, cam, d, head, th, circle, runs)
        writer.write(hud(img, d, head, th, cam, title, subtitle, home_px, circle))
        if k % (a.fps * 5) == 0:
            print(f"\r  {100 * k / (n_frames + hold):5.1f} %", end="", flush=True)
    writer.release()
    print(f"\r  {n_frames + hold} frames, {(n_frames + hold) / a.fps:.1f} s, {codec} → {out}")


if __name__ == "__main__":
    main()
