#!/usr/bin/env python
"""
YOY Trainer v2 — Isaac Sim launcher (ground takeoff configuration)
==================================================================

Compared with the earlier model in examples/yoy_trainer (v1):
  * Aerodynamic coefficients come from the September 2026 CFD re-characterization
    of the rebuilt CAD model: 9-point alpha sweep (-5 to 20 deg) plus a sideslip
    case, far-field domain, converged goals, mesh-independence study.
  * Lateral-directional derivatives are now measured, not guessed.
  * Damping derivatives are back to physically plausible values. In v1 they were
    Cm_q = -120, Cl_p = -5, Cn_r = -2, i.e. 15x / 10x / 10x the Pegasus library
    defaults. Those were compensating for an inertia tensor that was wrong by
    -61 % in roll and -71 % in yaw, not for real aerodynamics.
  * No force_application_offset: the URDF link frame sits ON the centre of mass,
    so aerodynamic forces are applied where they belong.

Modes:  manual | thrust_only | autonomous
"""

import argparse
import json
import os
import time

from isaacsim import SimulationApp

# Pin to a single GPU; on hybrid systems Isaac Sim may otherwise pick the iGPU.
simulation_app = SimulationApp({"headless": False, "multi_gpu": False,
                                "active_gpu": 0, "max_gpu_count": 1})

import carb                                                    # noqa: E402
import numpy as np                                             # noqa: E402
import omni.appwindow                                          # noqa: E402
import omni.isaac.core.utils.prims as prim_utils               # noqa: E402
import omni.timeline                                           # noqa: E402
from omni.kit.viewport.utility import get_active_viewport      # noqa: E402
from pxr import UsdGeom, UsdPhysics                            # noqa: E402
from scipy.spatial.transform import Rotation                   # noqa: E402

from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.vehicles.fixedwing import FixedWing, FixedWingConfig
from pegasus.simulator.logic.backends.ardupilot_mavlink_backend import (
    ArduPilotMavlinkBackend, ArduPilotMavlinkBackendConfig,
)
from pegasus.simulator.params import SIMULATION_ENVIRONMENTS
from isaacsim.core.api.world.world import World


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_USD = os.path.join(SCRIPT_DIR, "model", "urdf",
                         "yoy_trainer_v2_physics.usd")

# ── Reference data ────────────────────────────────────────────────────────────
# These match the CURRENT CAD state, i.e. exactly what the USD carries:
# structure + landing gear + Pixhawk + battery + motor + propeller = 524.36 g.
# The bill of materials for the finished aircraft is 644 g; ESC, GPS, RC
# receiver, power module and the four servos (~130 g) are not modelled yet.
# When they go in, re-read Mass Properties and update these five constants.
MASS_MODEL = 0.52436       # kg, what the USD actually weighs
MASS_BOM = 0.644           # kg, finished aircraft per the BOM (reference only)
WING_AREA = 0.1699         # m^2
CL_MAX = 1.063             # CFD v2, at alpha = 15 deg
V_STALL = 6.82             # m/s at MASS_MODEL
V_ROTATE = 7.84            # m/s = 1.15 * V_stall

# Centre of gravity at Z = 251.5 mm in the SolidWorks frame -> static margin
# 15.5 %, mid-band for a trainer. The CFD reference CG was Z = 230 mm; the
# pitching moment is translated analytically (Cm_alpha += dz/c * CL_alpha),
# never re-simulated.
CM_ALPHA_AT_CG = -0.7605   # rad^-1 at the as-built CG (CFD value was -1.442)
CM_0_AT_CG = 0.0374        # translated the same way (CFD value was 0.016)

# Centre of mass in the base_link frame, straight from the URDF <inertial>
# <origin>. The exporter CSYS (ROS_FLU) sits at the assembly origin, not at the
# CG, so aerodynamic forces must be applied here or PhysX adds a spurious moment.
CG_IN_LINK_FRAME = [-0.25152, -0.019648, -0.013068]   # m

# Mechanical surface throw at full servo travel (PWM 1100/1900), deg.
# High-rate (full mechanical) setup for the JoyPlanes trainer: ArduPilot
# drives the servos over their whole travel, and dual rates / expo live in
# the RC transmitter, which the autopilot never sees. See build_config().
AILERON_MAX_DEG = 25.0
ELEVATOR_MAX_DEG = 20.0
RUDDER_MAX_DEG = 30.0


class DummyBackend:
    """No-op backend for manual / thrust_only modes."""
    def __init__(self):        self._armed = False
    def initialize(self, v):   pass
    def start(self):           pass
    def stop(self):            pass
    def update_state(self, s): pass
    def update_sensor(self, t, d): pass
    def update(self, dt):      return [0.0] * 6
    def input_reference(self): return [0.0] * 6


def create_camera(name, parent_path, translation, rotation_euler_deg):
    cam_path = f"{parent_path}/{name}"
    quat = Rotation.from_euler("XYZ", rotation_euler_deg, degrees=True).as_quat()
    prim_utils.create_prim(
        prim_path=cam_path,
        prim_type="Camera",
        translation=translation,
        orientation=[quat[3], quat[0], quat[1], quat[2]],
    )
    return cam_path


def build_config(mode):
    config = FixedWingConfig()

    # ── Geometry (SolidWorks, unchanged between v1 and v2) ────────────────
    config.wing_area = WING_AREA
    config.wing_span = 1.058
    config.chord = 0.155
    config.air_density = 1.225

    # ── Longitudinal aerodynamics — CFD v2, V = 15 m/s ───────────────────
    # Linear fit over -5 to 7.5 deg: CL_alpha R^2 = 0.9996, Cm_alpha R^2 = 0.9874
    config.CL_0 = 0.154
    config.CL_alpha = 4.908           # rad^-1  (v1: 2.71)
    config.CL_max = 1.063             # at 15 deg (v1: 0.410, unjustified)
    config.CL_min = -0.40             # from the -5 deg point
    # Drag polar in Pegasus's form CD = CD_0 + CD_alpha*|a| + CD_alpha2*a^2,
    # least-squares over the 7 CFD points from -5 to 15 deg
    # (Resultados_Flow_Simulation_WhatIfAnalysis4-v3.xlsx, forces projected on
    # wind axes, q*S = 23.41 N). Max residual 0.002. The Pegasus defaults
    # (0.30 / 2.0) put 10-17 % too much drag in the 0-7.5 deg cruise range.
    #   alpha:  -5     0     2.5   5     7.5   10    15
    #   CD CFD: .0859 .0585 .0673 .0850 .1168 .1543 .2722
    config.CD_0 = 0.0590              # v1: 0.104
    config.CD_alpha = 0.0476          # rad^-1
    config.CD_alpha2 = 2.921          # rad^-2
    config.Cm_0 = CM_0_AT_CG           # trasladado al CG real (CFD: 0.016)
    config.Cm_alpha = CM_ALPHA_AT_CG   # rad^-1 -> static margin 15.5 %

    # ── Lateral-directional — CFD v2, sideslip case at alpha = 5 deg ─────
    # Measured, not library defaults.
    config.CY_beta = -0.378           # side force due to sideslip
    config.Cn_beta = 0.054            # weathercock stability  (stabilising +)
    config.Cl_beta = -0.054           # dihedral effect        (stabilising -)

    # ── Control-surface effectiveness ────────────────────────────────────
    # Not measured yet: a control-deflection CFD sweep is still pending.
    # The per-radian values are the Pegasus library defaults, which at least
    # are self-consistent. At the 15.5 % static margin, rotating to 10 deg
    # alpha needs Cm = 0.095; 20 deg of elevator delivers 0.392 (4.1x).
    #
    # Pegasus multiplies these by the NORMALISED stick command (-1..+1), not
    # by a deflection angle, so a per-radian derivative used as-is means full
    # stick = 1 rad = 57 deg of surface. That gave a steady-state roll rate of
    # ~740 deg/s at 15 m/s and enough elevator to trim at 82 deg alpha: every
    # autopilot correction overshot (porpoising, a 129 deg roll upset in
    # auto_20260923_222002). Scale each derivative by the real surface throw
    # so +-1 means the mechanical limit. Throws: high-rate setup for the
    # JoyPlanes trainer (ailerons +-20..25, elevator +-18..20, rudder
    # +-25..30 deg), the top of each range and still inside the 25 / 30 deg
    # where a bevelled foamboard hinge stops being proportional. With them:
    # ~325 deg/s of roll at 15 m/s, and full up elevator trims at ~27 deg
    # alpha, well past CL_max (15 deg), so the autopilot can stall it.
    ail = np.radians(AILERON_MAX_DEG)
    ele = np.radians(ELEVATOR_MAX_DEG)
    rud = np.radians(RUDDER_MAX_DEG)
    config.CL_elevator = 0.43 * ele    # per rad -> per full stick
    config.Cm_elevator = -1.122 * ele
    config.Cl_aileron = 0.229 * ail    # v1 had this doubled to 0.3 by hand
    config.Cn_rudder = -0.032 * rud
    config.CY_rudder = 0.870 * rud

    # ── Damping ──────────────────────────────────────────────────────────
    # Library defaults, physically plausible for this class of aircraft.
    # Do NOT raise these to tame an oscillation before checking the inertia
    # tensor in the USD: that is exactly the mistake v1 made.
    config.Cm_q = -8.0
    config.Cl_p = -0.50
    config.Cl_r = 0.15
    config.Cn_p = -0.06
    config.Cn_r = -0.20

    # ── Prim names inside the USD ────────────────────────────────────────
    config.body_name = "/base_link"
    config.propeller_name = "/propeller_link"
    config.propeller_joint_name = "propeller_joint"

    # ── Propulsion ───────────────────────────────────────────────────────
    # A2212 1000 kV + 1045 prop on 3S: 8.585 N (875 g) static, 1.1 % below the
    # 8.68 N measured on the bench. The 15 N (1529 g) used in v1 is beyond
    # this motor.
    config.prop_max_thrust = 8.585    # N
    # Pegasus computes thrust = coef * (throttle * prop_max_rpm)^2 and clips it
    # at prop_max_thrust. With its defaults (1e-5, 8000 rpm) full throttle is
    # 640 N, so an 8.585 N cap is reached at 11.6 % throttle and everything
    # above did nothing: ArduPilot's whole throttle range lived in 0-15 %,
    # TECS could not meter thrust. (This, not
    # C_D0, is why v1 overflew at 29 m/s.) Scale the coefficient so 100 %
    # throttle is exactly prop_max_thrust. Still quadratic, like a real prop.
    config.prop_max_rpm = 8000.0
    config.prop_thrust_coefficient = config.prop_max_thrust / config.prop_max_rpm ** 2
    config.mass_kg = MASS_MODEL       # fallback only; PhysX uses the USD value

    # Apply the aerodynamic forces AT the centre of mass. This is not a tuning
    # knob: it is the CG the URDF reports, in the link frame.
    config.force_application_offset = np.array(CG_IN_LINK_FRAME)

    config.simulation_mode = mode
    return config


SIM_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "flight_logs", "sim_config.json")


def write_sim_config(config, mode):
    """Leave the plant the simulator is ACTUALLY running where the mission
    script can find it. autonomous_flight.py copies this into every log, so a
    CSV says which thrust curve its throttle_pct belongs to. Without it the
    15 N -> 8.585 N change made old and new throttle columns indistinguishable
    and thrust had to be reconstructed from the launcher's git history."""
    os.makedirs(os.path.dirname(SIM_CONFIG_PATH), exist_ok=True)
    with open(SIM_CONFIG_PATH, "w") as f:
        json.dump({
            "written": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": mode,
            "prop_max_thrust_n": config.prop_max_thrust,
            "prop_max_rpm": config.prop_max_rpm,
            "prop_thrust_coefficient": config.prop_thrust_coefficient,
            "mass_model_kg": MASS_MODEL,
            "surface_throw_deg": {"aileron": AILERON_MAX_DEG,
                                  "elevator": ELEVATOR_MAX_DEG,
                                  "rudder": RUDDER_MAX_DEG},
            "CD_0": config.CD_0, "CD_alpha": config.CD_alpha,
            "CD_alpha2": config.CD_alpha2,
        }, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="YOY Trainer v2 — Isaac Sim launcher")
    parser.add_argument("--mode", type=str, default="manual",
                        choices=["manual", "thrust_only", "autonomous"])
    parser.add_argument("--contact-friction", type=float, default=None,
                        help="Friction of the wheel and tail contact spheres. "
                             "Default: the value baked into the USD (0.04, "
                             "rolling resistance on a runway). Lower is more "
                             "slippery: ~0.01-0.02 behaves like ice.")
    parser.add_argument("--ardupilot-dir", type=str,
                        default=os.path.expanduser("~/ardupilot"))
    args = parser.parse_args()

    if not os.path.exists(MODEL_USD):
        raise SystemExit(
            f"\nMissing USD: {MODEL_USD}\n"
            "Export the URDF from SolidWorks into model/urdf/ "
            "and import it with the Isaac URDF importer "
            "(Fix Base Link OFF, Import Inertia Tensor ON).\n")

    pg = PegasusInterface()
    pg._world = World(**pg._world_settings)
    world = pg.world
    pg.load_environment(SIMULATION_ENVIRONMENTS["Flat Plane"])

    config = build_config(args.mode)
    write_sim_config(config, args.mode)

    if args.mode == "autonomous":
        ardupilot_config = ArduPilotMavlinkBackendConfig({
            "vehicle_id": 0,
            "ardupilot_autolaunch": True,
            "ardupilot_dir": args.ardupilot_dir,
            "ardupilot_vehicle_model": "plane",
            "ardupilot_vehicle": "ArduPlane",
            "connection_type": "udpin",
            "connection_ip": "0.0.0.0",
            "connection_baseport": 14550,
            "update_rate": 250,
        })
        config.backends = [ArduPilotMavlinkBackend(config=ardupilot_config)]
    else:
        config.backends = [DummyBackend()]

    # init_pos Z must put the wheels exactly on the ground. Too high and the
    # aircraft drops and bounces; too low and PhysX pushes it out hard. Either
    # way the EKF never settles and ArduPilot refuses to arm.
    airplane = FixedWing(
        stage_prefix="/World/fixedwing0",
        usd_file=MODEL_USD,
        vehicle_id=0,
        init_pos=[0.0, 0.0, 0.20],  # measured resting height of the link origin
        config=config,
    )

    if args.contact_friction is not None:
        material = prim_utils.get_current_stage().GetPrimAtPath(
            "/World/fixedwing0/PhysicsMaterials/contact_low_friction")
        if not material.IsValid():
            raise SystemExit("contact_low_friction material missing: "
                             "regenerate the USD with scripts/urdf_to_usd.py")
        physics_material = UsdPhysics.MaterialAPI(material)
        physics_material.GetStaticFrictionAttr().Set(args.contact_friction)
        physics_material.GetDynamicFrictionAttr().Set(args.contact_friction)
        print(f"  Contact friction = {args.contact_friction}")

    body_path = f"/World/fixedwing0{config.body_name}"
    cameras = {
        "chase":   create_camera("ChaseCam",   body_path, (-6, 0, 2.5),   [0, 15, 0]),
        "top":     create_camera("TopCam",     body_path, (0, 0, 8),      [90, 0, 0]),
        "side":    create_camera("SideCam",    body_path, (0, -4, 0.8),   [0, -5, 90]),
        "cockpit": create_camera("CockpitCam", body_path, (0.35, 0, 0.12), [0, 0, 0]),
        "iso":     create_camera("IsoCam",     body_path, (-5, -5, 4),    [30, 0, 45]),
    }

    viewport = get_active_viewport()
    if viewport:
        viewport.camera_path = cameras["chase"]
    cam_keys = list(cameras.keys())

    appwindow = omni.appwindow.get_default_app_window()
    input_iface = carb.input.acquire_input_interface()
    keyboard = appwindow.get_keyboard()
    KEY_MAP = {
        carb.input.KeyboardInput.KEY_1: 0, carb.input.KeyboardInput.KEY_2: 1,
        carb.input.KeyboardInput.KEY_3: 2, carb.input.KeyboardInput.KEY_4: 3,
        carb.input.KeyboardInput.KEY_5: 4,
    }

    def on_cam_key(event, *a, **kw):
        if event.type != carb.input.KeyboardEventType.KEY_PRESS:
            return
        if event.input in KEY_MAP and KEY_MAP[event.input] < len(cam_keys):
            name = cam_keys[KEY_MAP[event.input]]
            if viewport:
                viewport.camera_path = cameras[name]
            print(f"  Camera -> {name.upper()}")

    cam_sub = input_iface.subscribe_to_keyboard_events(keyboard, on_cam_key)

    world.reset()

    try:
        stage = prim_utils.get_current_stage()
        for path in ["/World/layout", "/World/layout/GroundPlane",
                     "/World/layout/Environment",
                     "/World/layout/Environment/Geometry"]:
            prim = stage.GetPrimAtPath(path)
            if prim.IsValid():
                for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
                    if op.GetOpName() == "xformOp:scale":
                        op.Set((5000.0, 5000.0, 1.0))
                        break
    except Exception as e:
        print(f"  Could not scale ground: {e}")

    omni.timeline.get_timeline_interface().play()

    print(f"\nYOY Trainer v2 — mode: {args.mode.upper()}")
    print(f"  V_stall = {V_STALL:.2f} m/s   rotate at {V_ROTATE:.2f} m/s")
    print("  Cameras: [1] Chase [2] Top [3] Side [4] Cockpit [5] Iso\n")

    frame = 0
    rotated = False
    while simulation_app.is_running():
        world.step(render=True)
        if frame % 30 == 0:
            state = airplane.state
            pos, vel = state.position, state.linear_velocity
            speed = float(np.linalg.norm(vel))
            # Dynamic-pressure-based lift fraction: how much of the weight the
            # wing is carrying right now. Crossing 1.0 is the actual lift-off.
            q = 0.5 * config.air_density * speed ** 2
            frac = q * WING_AREA * CL_MAX / (MASS_MODEL * 9.81) if speed > 0 else 0.0
            flag = ""
            if not rotated and speed >= V_ROTATE:
                flag, rotated = "  <-- V_rotate reached", True
            print(f"[{frame:6d}]  X={pos[0]:+8.2f}  Y={pos[1]:+8.2f}  "
                  f"Alt={pos[2]:7.2f} m  V={speed:6.2f} m/s  "
                  f"L/W_max={frac:5.2f}{flag}")
        frame += 1

    input_iface.unsubscribe_to_keyboard_events(keyboard, cam_sub)
    simulation_app.close()


if __name__ == "__main__":
    main()
