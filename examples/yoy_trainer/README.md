# YOY Trainer — Foamboard Fixed-Wing Example (first model)

> Earlier iteration of the digital twin, kept for reference. The current model,
> and the one behind the ETCM 2026 paper, is [`../yoy_trainer_v2`](../yoy_trainer_v2/).

Custom foamboard fixed-wing aircraft (**JoyPlanes YOY Trainer**) integrated into
Isaac Sim via the Pegasus Simulator framework.  All aerodynamic coefficients were
extracted through SolidWorks Flow Simulation CFD and validated in a dual-loop
SITL environment (Isaac Sim + ArduPilot).

## Directory Layout

```
yoy_trainer/
├── 13_yoy_trainer_fixedwing.py   # Main launcher (3 modes)
├── README.md
├── model/
│   ├── meshes/                   # STL geometry (SolidWorks export)
│   └── urdf/
│       ├── foam_board_plane_complete.urdf
│       └── foam_board_plane_complete_physics.usd
├── scripts/
│   ├── autonomous_flight.py      # Waypoint mission + LOITER + CSV logger
│   ├── plot_flight.py            # CSV -> self-contained HTML report
│   └── report_template.html
└── flight_logs/                  # Auto-generated CSV telemetry logs
```

## Quick Start

Run everything from the repository root. `isaac_run` is the launcher alias
described in the Pegasus Simulator installation guide.

### 1. Frame Debugging (manual mode)
```bash
isaac_run examples/yoy_trainer/13_yoy_trainer_fixedwing.py --mode manual
```

### 2. Aero Debugging (thrust only)
```bash
isaac_run examples/yoy_trainer/13_yoy_trainer_fixedwing.py --mode thrust_only
```

### 3. Fully Autonomous (ArduPilot SITL)
```bash
# Terminal 1 — launch simulation
isaac_run examples/yoy_trainer/13_yoy_trainer_fixedwing.py --mode autonomous

# Terminal 2 — run the mission
python examples/yoy_trainer/scripts/autonomous_flight.py
```

With no extra flags the launcher runs the original configuration. Each model
correction studied afterwards is exposed as an independent, opt-in flag
(`--cg`, `--inertia`, `--gear-shift`, `--thrust`, `--damping`, `--gear-fix`,
`--symmetric-inertia`, `--force-offset`); see `--help` for what each one changes.

## Aircraft Parameters

| Parameter | Value | Source |
|---|---|---|
| Mass | 644 g (612.45 g airframe + electronics) | SolidWorks assembly |
| Wing area | 0.1699 m² | SolidWorks |
| Wing span | 1.058 m | SolidWorks |
| MAC | 0.155 m | SolidWorks |
| CL_α | 2.71 rad⁻¹ | CFD (V = 15 m/s) |
| CD_0 | 0.104 | CFD |
| Cm_0 | 0.057 | CFD |
| Cm_α | −0.205 | CFD (static margin 7.5 %) |
| Max thrust | 15.0 N (peak) | Motor bench test |
| Cruise thrust | ≈ 7.7 N | SITL trim condition |

## Camera Hotkeys

Press the number key **after clicking** the 3-D viewport:

| Key | View |
|---|---|
| 1 | Chase (behind & above) |
| 2 | Top-Down |
| 3 | Side |
| 4 | Cockpit / FPV |
| 5 | Isometric 45° |
