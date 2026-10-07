# YOY Trainer v2 — Foamboard Fixed-Wing

Digital twin of the JoyPlanes YOY Trainer foamboard aircraft, as used in the
ETCM 2026 paper *Integration and Validation of Open-Source Fixed-Wing UAV
Platforms for Low-Cost Autonomous Flight Research*. The airframe is a
dimensional reconstruction of the prototype; all coefficients come from a CFD
characterization of that reconstruction.

## CFD characterization

| | |
|---|---|
| Solver | SolidWorks Flow Simulation, V = 15 m/s |
| Domain | far field, 32.7 m³, ~1 % blockage |
| Alpha sweep | 8 points, −5° to 20°, plus one sideslip point (β = 5°) |
| Convergence | manual goals, 0.02–0.05 N / 0.003 N·m |
| Mesh independence | level 5 → 6, C_L within 1.3 % |
| `C_Lα` | 4.908 rad⁻¹ |
| `C_D0` | 0.058 |
| Lateral derivatives | `CY_β`, `Cn_β`, `Cl_β` measured |
| Static margin | 15.5 % at the as-built CG |

The CAD geometry, the raw CFD results and the flight logs behind the paper are
published in
[RAMEL-ESPOL/OpenSourceFixedWingUAV](https://github.com/RAMEL-ESPOL/OpenSourceFixedWingUAV).


## Aircraft parameters

| Parameter | Value | Source |
|---|---|---|
| Mass (prototype) | 0.644 kg | sum of the components |
| Mass (as simulated) | 0.524 kg | URDF/USD; ESC, GPS, receiver, power module and servos are not modelled yet |
| Wing area | 0.1699 m² | SolidWorks |
| Wing span | 1.058 m | SolidWorks |
| MAC | 0.155 m | SolidWorks |
| `I_xx` roll / `I_yy` pitch / `I_zz` yaw | 0.010315 / 0.012866 / 0.020917 kg·m² | SolidWorks, at CG, ROS FLU axes |
| `C_L0` / `C_Lα` / `C_Lmax` | 0.154 / 4.908 / 1.063 | CFD v2 |
| `C_D0` | 0.058 | CFD v2 |
| `C_m0` / `C_mα` | 0.016 / −1.442 | CFD v2, reference CG |
| `C_m0` / `C_mα` at the as-built CG | 0.037 / −0.761 | translated analytically; static margin 15.5 % |
| `CY_β` / `Cn_β` / `Cl_β` | −0.378 / +0.054 / −0.054 | CFD v2, sideslip case |
| Stall speed (as simulated) | 6.82 m/s | derived |
| Rotation speed | 7.84 m/s | 1.15 × V_stall |
| Max thrust | 8.585 N | 1.1 % below the 8.68 N measured on the static bench |

Control-surface effectiveness and damping derivatives are **not** measured yet;
the launcher uses Pegasus library defaults.

## Directory layout

```
yoy_trainer_v2/
├── 14_yoy_trainer_v2_fixedwing.py   # launcher (manual / thrust_only / autonomous)
├── README.md
├── model/
│   ├── meshes/                      # base_link.STL + body collider (no gear)
│   └── urdf/                        # URDF + generated USD
├── scripts/
│   ├── urdf_to_usd.py               # URDF -> USD (articulation, contact friction)
│   ├── make_body_collision.py       # airframe collider without the landing gear
│   ├── verify_usd.py                # mass properties carried by the generated USD
│   ├── inspect_usd.py               # prim/API structure of the generated USD
│   ├── autonomous_flight.py         # mission + LOITER + CSV logger
│   ├── plot_flight.py               # CSV -> self-contained HTML report
│   ├── report_template.html
│   ├── render_video.py              # CSV -> MP4 of the isometric view
│   └── paper_loiter_figure.py       # two-direction loiter figure (radius, bank)
└── flight_logs/                     # auto_<date>_<time>.csv / .json / .html (generated)
```

## Quick start

Everything from the repo root, always with `isaac_run` (Isaac Sim 5.1.0):

```bash
isaac_run examples/yoy_trainer_v2/14_yoy_trainer_v2_fixedwing.py --mode manual
isaac_run examples/yoy_trainer_v2/14_yoy_trainer_v2_fixedwing.py --mode thrust_only
```


## Autonomous flight (ArduPilot SITL)

**Terminal 1** - simulator + SITL. Wait until the ArduPilot console (separate
window, MAVProxy) shows a heartbeat / "Detected vehicle" before Terminal 2.

```bash
isaac_run examples/yoy_trainer_v2/14_yoy_trainer_v2_fixedwing.py --mode autonomous
```

**Terminal 2** - the mission. Pick one:

```bash
# Sustained loiter: takeoff, climb to 40 m, LOITER until the timer ends.
# These two are the orbits reported in the paper (clockwise, then counter-clockwise).
python3 examples/yoy_trainer_v2/scripts/autonomous_flight.py --loiter-now --duration 300 --radius 150
python3 examples/yoy_trainer_v2/scripts/autonomous_flight.py --loiter-now --duration 300 --radius 150 --ccw

# Waypoints: HOME -> WP0 (east) -> WP1 (south, RIGHT turn) -> LOITER
python3 examples/yoy_trainer_v2/scripts/autonomous_flight.py

# Straight line only, no turns (isolates speed/pitch control)
python3 examples/yoy_trainer_v2/scripts/autonomous_flight.py --straight --duration 120
```

Other mission flags: `--ccw` (counter-clockwise loiter), `--radius <m>`,
`--thr-cap <pct>` (THR_MAX). Without `--duration`, stop with Ctrl+C: the log is
saved either way.

**Report** - each run writes `flight_logs/auto_<date>_<time>.csv`, a `.json`
with the simulated plant and mission settings, and a compressed copy of the
Pegasus ground-truth force log. Turn the CSV into
an HTML page (3D trajectory with playback, roll/airspeed/altitude traces,
turn-radius check) and open it in the browser:

```bash
python3 examples/yoy_trainer_v2/scripts/plot_flight.py            # newest log
python3 examples/yoy_trainer_v2/scripts/plot_flight.py examples/yoy_trainer_v2/flight_logs/auto_XXXX.csv
xdg-open examples/yoy_trainer_v2/flight_logs/auto_XXXX.html
```

`scripts/render_video.py <log.csv>` exports the same view as an MP4.

The mission's ArduPilot parameters (TECS, PIDs, trim and takeoff throttle) live
in `scripts/autonomous_flight.py`.
