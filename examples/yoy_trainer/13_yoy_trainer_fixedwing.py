#!/usr/bin/env python
"""
| File: 13_yoy_trainer_fixedwing.py
| Author: Steven Martinez Jara
| License: BSD-3-Clause
| Description: Isaac Sim standalone app for the YOY Trainer foamboard fixed-wing
|              aircraft.  Supports three simulation modes:
|                - manual       → Frame Debugging (UI force panel only)
|                - thrust_only  → Aero Debugging  (UI thrust + aerodynamics)
|                - autonomous   → Fully Autonomous (ArduPilot SITL)
|
| Usage (run from the repo root):
|   isaac_run examples/yoy_trainer/13_yoy_trainer_fixedwing.py
|   isaac_run examples/yoy_trainer/13_yoy_trainer_fixedwing.py --mode thrust_only
|   isaac_run examples/yoy_trainer/13_yoy_trainer_fixedwing.py --mode autonomous
|
| Cameras (click viewport + press key):
|   1 → Chase    2 → Top-Down    3 → Side    4 → Cockpit    5 → Isometric
"""

import argparse
import os
import numpy as np

# ── Isaac Sim bootstrap ──────────────────────────────────────────────────────
os.environ.setdefault("ROS_DISTRO", "jazzy")
os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_cyclonedds_cpp")

from isaacsim import SimulationApp
simulation_app = SimulationApp({
    "headless": False,
    "multi_gpu": False,       # Disable multi-GPU (avoids picking an integrated GPU)
    "active_gpu": 0,          # Render on the first discrete GPU
    "max_gpu_count": 1,       # Limit to single GPU
})

import omni.timeline
import carb.input
import omni.appwindow
import omni.usd
import omni.isaac.core.utils.prims as prim_utils
from pxr import UsdGeom, UsdPhysics, PhysxSchema, Gf
from omni.kit.viewport.utility import get_active_viewport
from scipy.spatial.transform import Rotation

from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.vehicles.fixedwing import FixedWing, FixedWingConfig
from pegasus.simulator.logic.backends.ardupilot_mavlink_backend import (
    ArduPilotMavlinkBackend, ArduPilotMavlinkBackendConfig,
)
from pegasus.simulator.params import SIMULATION_ENVIRONMENTS
from isaacsim.core.api.world.world import World


# ── Resolve paths relative to this script ────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_USD  = os.path.join(SCRIPT_DIR, "model", "urdf",
                          "foam_board_plane_complete_physics.usd")


# ── Minimal backend for non-autonomous modes ─────────────────────────────────
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


# ── Camera helper ────────────────────────────────────────────────────────────
# USD's default Camera is a 50mm-equivalent lens (~24° horizontal FOV) — far too
# narrow for a ~1m-wingspan aircraft, which is why every camera looked "zoomed in"
# and needed a lot of manual scroll-out to frame the plane. 18mm gives ~60° FOV.
DEFAULT_FOCAL_LENGTH_MM = 18.0

def create_camera(name, parent_path, translation, rotation_euler_deg, focal_length_mm=DEFAULT_FOCAL_LENGTH_MM):
    """Spawn a camera parented to a body prim so it tracks the aircraft."""
    cam_path = f"{parent_path}/{name}"
    quat = Rotation.from_euler("XYZ", rotation_euler_deg, degrees=True).as_quat()
    prim_utils.create_prim(
        prim_path=cam_path,
        prim_type="Camera",
        translation=translation,
        orientation=[quat[3], quat[0], quat[1], quat[2]],
    )
    stage = omni.usd.get_context().get_stage()
    UsdGeom.Camera(stage.GetPrimAtPath(cam_path)).GetFocalLengthAttr().Set(focal_length_mm)
    return cam_path


# ── Landing-gear stabilisation ───────────────────────────────────────────────
# The URDF gives the wheels inertias far smaller than the fuselage: the tail
# wheel is 4.51e-7 kg·m² against the body's 1.29e-2, a ratio of 28,665:1, on a
# `continuous` joint with no damping and in contact with the ground.  PhysX
# articulations go unstable well before that ratio, and the result was the
# aircraft shaking itself apart on the runway at V < 1 m/s — where the dynamic
# pressure is 0.6 Pa, so aerodynamics cannot be the cause.  That ground jitter
# is what the oversized Cm_q/Cl_p/Cn_r values were really masking.
#
# Three fixes, applied to the loaded stage rather than by editing the binary
# USD asset: floor the wheel inertias to a sane ratio, add joint damping so the
# free-spinning wheels cannot chatter, and give the articulation solver enough
# iterations to resolve the contact.
MIN_WHEEL_INERTIA = 1.0e-4     # kg·m²  → worst-case ratio ~130:1
WHEEL_JOINT_DAMPING = 0.05     # N·m·s/rad — light, still lets them roll

# Wing chord measured off base_link.STL: leading edge X = -0.1760, trailing
# edge X = -0.3349, chord 0.1589 m.  The USD bakes the CG at X = -0.33459,
# i.e. 99.8% of the chord -- right on the trailing edge, far behind the 25-35%
# band a conventional aircraft needs to be pitch-stable.  That single number is
# what the inflated Cm_q, the hand-dialled -0.71 pitch trim and the rest of the
# accumulated patches were all compensating for.
#
# NEW_COM puts it at 30% chord instead.  NEW_INERTIA is the Esamble_nuevo
# tensor (g*mm^2 -> kg*m^2, assembly axes remapped to ROS: the model's Z is
# longitudinal and its Y is vertical, which the identity Iyaw ~= Iroll + Ipitch
# disambiguates -- the new tensor satisfies it to 2.6%, the USD's fails by 2.5x).
NEW_COM     = (-0.22367, 0.08764, 0.09486)      # 30% of chord

# Moving the CG forward exposes a second inconsistency: the main gear sits at
# X=-0.20383, which left 131 mm of nose-over margin behind the OLD (trailing
# edge) CG but only 20 mm behind the correct one.  A taildragger tips forward
# when T*h > W*d; with h=0.120 m and d=0.020 m that happens at 0.99 N, and
# takeoff commands 15 N.  That is why the aircraft went over on its nose and
# never left the runway.  Shifting the mains 36 mm forward puts the CG 25 deg
# behind the contact point, the usual taildragger geometry.
MAIN_GEAR_SHIFT = 0.0362        # m, +X = forward

# prop_max_thrust = 15 N on a 6.01 N aircraft is T/W = 2.5; a trainer runs
# 0.5-0.8.  Drag at the 15 m/s design cruise is only 2.44 N, so 5 N gives
# T/W = 0.83 and puts cruise at ~49% throttle instead of pinning the aircraft
# at 30 m/s the way 15 N did.
NEW_MAX_THRUST = 5.0
NEW_INERTIA = (0.032817, 0.018224, 0.049720)    # roll, pitch, yaw

def symmetrise_inertia(stage_prefix, body_name):
    """Align the inertia tensor's principal axes with the body axes, leaving the
    diagonal moments exactly as published."""
    try:
        stage = prim_utils.get_current_stage()
        prim = stage.GetPrimAtPath(stage_prefix + body_name)
        if not prim or not prim.IsValid() or not prim.HasAPI(UsdPhysics.MassAPI):
            print("  ⚠ No pude simetrizar la inercia: no encontre el cuerpo.")
            return
        m = UsdPhysics.MassAPI(prim)
        old = m.GetPrincipalAxesAttr().Get()
        m.CreatePrincipalAxesAttr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        print(f"  ⚖ ejes principales {old} → identidad "
              f"(productos de inercia a cero; diagonal intacta)")
    except Exception as e:
        print(f"  ⚠ No pude simetrizar la inercia: {e}")


def apply_mass_properties(stage_prefix, body_name, set_com=False, set_inertia=False):
    """Apply the corrected mass properties PIECE BY PIECE.

    CG and inertia used to move together under a single --mass-props flag, which
    made it impossible to attribute a change in behaviour to either one.  They
    are separate arguments now so each can be bisected on its own.
    """
    if not (set_com or set_inertia):
        return
    try:
        stage = prim_utils.get_current_stage()
        prim = stage.GetPrimAtPath(stage_prefix + body_name)
        if not prim or not prim.IsValid():
            print(f"  ⚠ No encontre {stage_prefix}{body_name}; masa sin cambiar.")
            return
        m = (UsdPhysics.MassAPI(prim) if prim.HasAPI(UsdPhysics.MassAPI)
             else UsdPhysics.MassAPI.Apply(prim))
        if set_com:
            old_com = m.GetCenterOfMassAttr().Get()
            m.CreateCenterOfMassAttr().Set(Gf.Vec3f(*NEW_COM))
            print(f"  ⚖ CG {tuple(round(v,5) for v in old_com)} → {NEW_COM}"
                  f"  (99.8% → 30% de cuerda)")
        if set_inertia:
            m.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*NEW_INERTIA))
            m.CreatePrincipalAxesAttr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            print(f"  ⚖ inercia diagonal → {NEW_INERTIA}  (alabeo, cabeceo, guiñada)"
                  f"  + productos a cero")
    except Exception as e:
        print(f"  ⚠ No pude aplicar las propiedades de masa: {e}")


def shift_main_gear(stage_prefix, dx=MAIN_GEAR_SHIFT):
    """Move the main wheels forward so the taildragger geometry matches the
    corrected CG. Leaves the tail wheel alone."""
    try:
        stage = prim_utils.get_current_stage()
        moved = []
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if not path.startswith(stage_prefix):
                continue
            n = path.split("/")[-1].lower()
            if "wheel" not in n or "joint" not in n or "tair" in n or "tail" in n:
                continue
            attr = prim.GetAttribute("physics:localPos0")
            if attr and attr.Get() is not None:
                v = attr.Get()
                attr.Set(Gf.Vec3f(float(v[0]) + dx, float(v[1]), float(v[2])))
                moved.append(f"{path.split('/')[-1]} X {v[0]:+.5f}→{v[0]+dx:+.5f}")
        if moved:
            print(f"  🛞 tren principal {dx*1000:+.1f} mm adelante: " + ", ".join(moved))
        else:
            print("  ⚠ No encontre las juntas del tren principal; sin mover.")
    except Exception as e:
        print(f"  ⚠ No pude mover el tren principal: {e}")


def stabilize_landing_gear(stage_prefix):
    try:
        _stabilize_landing_gear(stage_prefix)
    except Exception as e:
        print(f"  ⚠ No se pudo estabilizar el tren de aterrizaje: {e}")


def _stabilize_landing_gear(stage_prefix):
    stage = prim_utils.get_current_stage()
    floored, damped = [], []

    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith(stage_prefix) or "wheel" not in path.lower():
            continue

        # 1. Floor the diagonal inertia of the wheel bodies.
        if prim.HasAPI(UsdPhysics.MassAPI):
            mass_api = UsdPhysics.MassAPI(prim)
            attr = mass_api.GetDiagonalInertiaAttr()
            if attr and attr.HasAuthoredValue():
                i = attr.Get()
                new = Gf.Vec3f(*[max(float(v), MIN_WHEEL_INERTIA) for v in i])
                if new != i:
                    attr.Set(new)
                    floored.append((path.split("/")[-1], tuple(i), tuple(new)))

        # 2. Damp the continuous wheel joints so they cannot free-spin/chatter.
        if prim.IsA(UsdPhysics.RevoluteJoint):
            drive = (UsdPhysics.DriveAPI(prim, "angular")
                     if prim.HasAPI(UsdPhysics.DriveAPI)
                     else UsdPhysics.DriveAPI.Apply(prim, "angular"))
            drive.CreateTypeAttr().Set("force")
            drive.CreateStiffnessAttr().Set(0.0)          # free to roll
            drive.CreateDampingAttr().Set(WHEEL_JOINT_DAMPING)
            damped.append(path.split("/")[-1])

    # 3. More solver iterations for the whole articulation.
    root = stage.GetPrimAtPath(stage_prefix)
    if root and root.IsValid():
        art = (PhysxSchema.PhysxArticulationAPI(root)
               if root.HasAPI(PhysxSchema.PhysxArticulationAPI)
               else PhysxSchema.PhysxArticulationAPI.Apply(root))
        art.CreateSolverPositionIterationCountAttr().Set(32)
        art.CreateSolverVelocityIterationCountAttr().Set(8)

    for name, old, new in floored:
        print(f"  🛞 {name}: inercia {old} → {new}")
    if damped:
        print(f"  🛞 amortiguamiento {WHEEL_JOINT_DAMPING} en: {', '.join(damped)}")
    print("  🛞 solver de la articulación: 32 pos / 8 vel iteraciones")


# ═══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="YOY Trainer Fixed-Wing — Isaac Sim launcher")
    parser.add_argument(
        "--mode", type=str, default="manual",
        choices=["manual", "thrust_only", "autonomous"],
        help="Simulation mode (default: manual)")
    parser.add_argument(
        "--mass-props", type=str, default="legacy", choices=["legacy", "new"],
        help="Shortcut that turns on --cg new, --inertia new, --gear-shift and "
             "--thrust 5.0 together. Kept so older commands still reproduce. "
             "Prefer the individual flags: bundling them made it impossible to "
             "tell which one changed the behaviour.")
    parser.add_argument(
        "--cg", type=str, default=None, choices=["legacy", "new"],
        help="Center of mass ONLY, nothing else. 'legacy' is the value baked "
             "into the USD, X = -0.33459, which sits at 99.8%% of the wing chord "
             "-- on the trailing edge. 'new' puts it at 30%% chord "
             f"(X = {NEW_COM[0]}), the conventional stable band. WARNING: on its "
             "own this makes the aircraft MORE likely to nose over, because the "
             "main gear is still placed for the old CG and only 20 mm of "
             "nose-over margin remains. Pair it with --gear-shift.")
    parser.add_argument(
        "--inertia", type=str, default=None, choices=["legacy", "new"],
        help="Diagonal inertia tensor ONLY. 'new' is the Esamble_nuevo "
             "SolidWorks tensor, which satisfies Iyaw ~= Iroll + Ipitch to 2.6%% "
             "(the USD's fails it by 2.5x), and zeroes the products of inertia.")
    parser.add_argument(
        "--gear-shift", action="store_true",
        help=f"Move the main wheels forward (default {MAIN_GEAR_SHIFT*1000:.1f} mm, "
             "override with --gear-shift-mm) so the taildragger geometry matches "
             "a forward CG. Needed with --cg new.")
    parser.add_argument(
        "--gear-shift-mm", type=float, default=None,
        help="How far forward to move the mains, in mm. A taildragger noses over "
             "when T*h > W*d, with h = 0.120 m the thrust-line height and d the "
             "CG-to-contact distance. With --cg new the stock 36.2 mm leaves "
             "d = 56 mm, i.e. it tips above 2.8 N -- below both the 5 N and the "
             "15 N thrust settings, which is why every previous attempt to move "
             "the CG ended on the nose. Clearing 5 N needs 80 mm; clearing 15 N "
             "needs 280 mm, which is geometrically absurd, so a forward CG also "
             "requires cutting the thrust. Implies --gear-shift.")
    parser.add_argument(
        "--thrust", type=float, default=None,
        help="prop_max_thrust in N (default 15.0 = T/W 2.5; a real trainer runs "
             "0.5-0.8, i.e. 3-5 N). Independent of the mass properties now.")
    parser.add_argument(
        "--damping", type=str, default="legacy", choices=["legacy", "physical"],
        help="'legacy': Cm_q=-120, Cl_p=-5.0, Cn_r=-2.0 (10-15x literature, "
             "needed only to mask the aft CG). 'physical': the Pegasus defaults "
             "-8.0 / -0.50 / -0.20, which match strip theory. Use with "
             "--mass-props new.")
    parser.add_argument(
        "--gear-fix", action="store_true",
        help="Floor the wheel inertias, damp the wheel joints and raise the "
             "articulation solver iterations. Fixes a real PhysX instability "
             "(the tail wheel is 28,665:1 against the fuselage on an undamped "
             "continuous joint) but it IS a physics change the published runs "
             "did not have, so it is off by default when replicating.")
    parser.add_argument(
        "--symmetric-inertia", action="store_true",
        help="Zero the products of inertia, keeping the published diagonal "
             "values untouched. The USD's inertia tensor has Ixy = -0.001308 "
             "against Ixx = 0.01294 -- a 10%% product of inertia, i.e. principal "
             "axes rotated 7 deg -- on an aircraft whose plane of symmetry sits "
             "7 mm from its CG. It is a modelling artefact, and it bleeds "
             "~4.3 rad/s2 of ROLL per N.m of commanded PITCH, always the same "
             "way: a candidate for the persistent one-sided turn. The "
             "Esamble_nuevo model has products of 0.00/0.11/1.32%%.")
    parser.add_argument(
        "--force-offset", type=str, default="legacy",
        choices=["com", "zero", "legacy"],
        help="Where aero/thrust forces are applied. Defaults to 'legacy' (the "
             "original value, with Z negated) because that is the configuration "
             "the original results were produced with -- the -0.71 N.m "
             "pitch trim it documents is the compensation for it. 'com' places "
             "them at the active CoM and 'zero' at the link origin; both are "
             "more defensible physically but change the published behaviour.")
    parser.add_argument(
        "--ardupilot-dir", type=str,
        default=os.path.expanduser("~/ardupilot"),
        help="Path to local ArduPilot source tree")
    args = parser.parse_args()

    # --mass-props new stays as the bundled shortcut; the individual flags win
    # whenever they are given explicitly.
    _bundle = (args.mass_props == "new")
    if args.cg      is None: args.cg      = "new" if _bundle else "legacy"
    if args.inertia is None: args.inertia = "new" if _bundle else "legacy"
    if _bundle:
        args.gear_shift = True
    if args.gear_shift_mm is not None:
        args.gear_shift = True
        if args.thrust is None:
            args.thrust = NEW_MAX_THRUST
    if args.thrust is None:
        args.thrust = 15.0

    # ── World ────────────────────────────────────────────────────────────
    pg = PegasusInterface()
    pg._world = World(**pg._world_settings)
    world = pg.world
    pg.load_environment(SIMULATION_ENVIRONMENTS["Flat Plane"])

    # ── Aircraft configuration ───────────────────────────────────────────
    config = FixedWingConfig()

    # Geometry (SolidWorks measurements)
    config.wing_area = 0.1699           # m²
    config.wing_span = 1.058            # m
    config.chord     = 0.155            # m (MAC)

    # Longitudinal aero coefficients (SolidWorks Flow Sim, V = 15 m/s)
    config.CL_0     =  0.008           # CL at AoA = 0
    config.CL_alpha =  2.71            # lift-curve slope  [rad⁻¹]
    config.CL_max   =  0.410           # stall limit
    config.CD_0     =  0.104           # parasitic drag
    config.Cm_0     =  0.057           # pitching moment at AoA = 0
    config.Cm_alpha = -0.205           # pitch stability (static margin 7.5 %)

    # Control-surface effectiveness
    config.Cm_elevator = -0.6
    config.Cl_aileron  =  0.3           # doubled from 0.15 for roll authority
                                        # (value the published runs actually flew with)
    config.Cn_rudder   = -0.03

    # Aerodynamic damping — KNOWN-WRONG VALUES, DELIBERATELY KEPT.
    #
    # These are 10-15x larger than both the FixedWingConfig class defaults
    # (Cl_p=-0.50, Cn_r=-0.20, Cm_q=-8.0) and the strip-theory/literature values
    # (-0.45, -0.185, -7.5).  The upstream example
    # 12_ardupilot_fixedwing.py never overrides them and flies stably.
    #
    # They are kept because BOTH attempts to correct them made the aircraft
    # unflyable, and the failure mode identifies where the real bug is:
    #   - correct damping alone            -> log auto_20260816_215527
    #   - correct damping + negated offset -> log auto_20260816_223534
    # In both, the aircraft tumbles (roll +-137 deg, pitch +-72 deg) while still
    # on the runway at V < 1 m/s.  At that speed q = 0.6 Pa, so aerodynamic
    # forces are negligible by three orders of magnitude: the instability is in
    # the ground / landing-gear contact model, NOT in the aerodynamics.  The
    # inflated damping only masks it once airborne.
    #
    # Consequence for results: with this damping the roll channel oscillates at
    # +-87 N.m internally (hitting the +-500 N.m clip in 3.6% of samples, roll
    # rates to 43 rad/s).  The 20 Hz telemetry aliases that into what looks like
    # a steady 28 deg bank which produces no turn at all (measured -0.10 deg/s
    # vs the +9.61 deg/s that bank implies).  Do not read the logged bank angle
    # as a physically meaningful attitude.
    #
    # Fix the ground contact first; only then correct these three values.
    if args.damping == "physical":
        config.Cm_q =  -8.0            # Pegasus default / literature -7.5
        config.Cl_p =  -0.50           # Pegasus default / strip theory -0.45
        config.Cn_r =  -0.20           # Pegasus default / literature -0.185
    else:
        config.Cm_q = -120.0           # published configuration
        config.Cl_p =   -5.0
        config.Cn_r =   -2.0
    print(f"  ⚙ damping = {args.damping} "
          f"(Cm_q={config.Cm_q}, Cl_p={config.Cl_p}, Cn_r={config.Cn_r})")

    # Prim names inside the USD
    config.body_name            = "/base_link"
    config.propeller_name       = "/propeller_link"
    config.propeller_joint_name = "propeller_joint"

    # Propulsion
    config.prop_max_thrust = args.thrust
    print(f"  ⚙ prop_max_thrust = {config.prop_max_thrust} N  "
          f"(T/W = {config.prop_max_thrust / (0.61245 * 9.81):.2f})")

    # Force-application offset — where aero/thrust forces are applied, in the
    # body link's local frame.  Selected by --force-offset; see that flag's
    # help.  The ambiguity is whether apply_force()'s `pos` argument is measured
    # from the link origin or from the CoM, which decides which value is right:
    #   com    = (-0.334590, +0.087640, +0.094860)  URDF CoM, verbatim
    #   zero   = (0, 0, 0)                          i.e. the body's own CoM
    #   legacy = (-0.334590, +0.087640, -0.094860)  Z negated (previous value)
    #
    # The legacy value needed a hand-dialled Torque Y = -0.71 N.m to fly level,
    # and that number was taken as the digital twin's "stable
    # trim".  If either of the other two flies level with Torque Y = 0, that
    # trim was never physical -- it was cancelling a misplaced force.
    # "com" must track whichever CoM is actually active: with --mass-props new
    # the CoM moves 111 mm forward, and pointing this at the old one would
    # re-introduce exactly the parasitic moment the move is meant to remove.
    _active_com = NEW_COM if args.cg == "new" else (-0.334590, 0.087640, 0.094860)
    _OFFSETS = {
        "com":    np.array(_active_com),
        "zero":   np.array([0.0, 0.0, 0.0]),
        "legacy": np.array([-0.334590, 0.087640, -0.094860]),
    }
    config.force_application_offset = _OFFSETS[args.force_offset]
    # Moving the CG without moving the force-application point puts every
    # aerodynamic force on a lever arm equal to the distance between them.
    # With --cg new and the default --force-offset legacy that arm is 111 mm,
    # and lift alone (~6.01 N) then generates ~0.67 N.m of nose-down moment --
    # more than the elevator can trim. The result is not a test of the CG, it
    # is a test of the lever. Refuse to run it silently.
    if args.cg == "new" and args.force_offset == "legacy":
        _arm = abs(_OFFSETS["legacy"][0] - NEW_COM[0])
        print(f"\n  ⚠⚠ ADVERTENCIA: --cg new con --force-offset legacy.")
        print(f"     Las fuerzas se aplican {_arm*1000:.1f} mm por detras del CG nuevo.")
        print(f"     Solo la sustentacion (6.01 N) da {6.01*_arm:.2f} N.m de picado parasito.")
        print(f"     Esto NO mide el CG, mide el brazo de palanca.")
        print(f"     Usa:  --cg new --force-offset com\n")
    print(f"  ⚙ force_application_offset = {args.force_offset} "
          f"{config.force_application_offset}")

    # ── Mode selection ───────────────────────────────────────────────────
    config.simulation_mode = args.mode

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

    # ── Spawn aircraft ───────────────────────────────────────────────────
    airplane = FixedWing(
        stage_prefix="/World/fixedwing0",
        usd_file=MODEL_USD,
        vehicle_id=0,
        init_pos=[0.0, 0.0, 0.15],
        config=config,
    )

    # Must run BEFORE world.reset(): editing physics schema after PhysX has
    # already built the articulation crashes the simulator outright.
    if args.gear_fix:
        stabilize_landing_gear("/World/fixedwing0")
    apply_mass_properties("/World/fixedwing0", config.body_name,
                          set_com=(args.cg == "new"),
                          set_inertia=(args.inertia == "new"))
    if args.gear_shift:
        _dx = (MAIN_GEAR_SHIFT if args.gear_shift_mm is None
               else args.gear_shift_mm / 1000.0)
        shift_main_gear("/World/fixedwing0", dx=_dx)
        _d = abs((-0.20383 + _dx) - (NEW_COM[0] if args.cg == "new" else -0.334590))
        print(f"  🛞 margen contra vuelco: d = {_d*1000:.1f} mm  →  vuelca sobre "
              f"T = {0.61245*9.81*_d/0.120:.2f} N  (empuje configurado "
              f"{args.thrust:.1f} N)")
    if args.symmetric_inertia and args.inertia != "new":
        symmetrise_inertia("/World/fixedwing0", config.body_name)

    # ── Cameras ──────────────────────────────────────────────────────────
    body_path = f"/World/fixedwing0{config.body_name}"
    cameras = {
        "chase":   create_camera("ChaseCam",   body_path, (-6, 0, 2.5),    [0, 15, 0]),
        "top":     create_camera("TopCam",     body_path, (0, 0, 8),       [90, 0, 0]),
        "side":    create_camera("SideCam",    body_path, (0, -4, 0.8),    [0, -5, 90]),
        "cockpit": create_camera("CockpitCam", body_path, (0.35, 0, 0.12), [0, 0, 0]),
        "iso":     create_camera("IsoCam",     body_path, (-5, -5, 4),     [30, 0, 45]),
    }

    viewport = get_active_viewport()
    if viewport:
        viewport.camera_path = cameras["chase"]

    cam_keys = list(cameras.keys())

    # ── Camera hotkeys ───────────────────────────────────────────────────
    appwindow = omni.appwindow.get_default_app_window()
    input_iface = carb.input.acquire_input_interface()
    keyboard = appwindow.get_keyboard()

    KEY_MAP = {
        carb.input.KeyboardInput.KEY_1: 0,
        carb.input.KeyboardInput.KEY_2: 1,
        carb.input.KeyboardInput.KEY_3: 2,
        carb.input.KeyboardInput.KEY_4: 3,
        carb.input.KeyboardInput.KEY_5: 4,
    }

    def on_cam_key(event, *args, **kwargs):
        if event.type != carb.input.KeyboardEventType.KEY_PRESS:
            return
        if event.input in KEY_MAP:
            idx = KEY_MAP[event.input]
            if idx < len(cam_keys):
                name = cam_keys[idx]
                if viewport:
                    viewport.camera_path = cameras[name]
                print(f"  📸 Camera → {name.upper()}")

    cam_sub = input_iface.subscribe_to_keyboard_events(keyboard, on_cam_key)

    # ── Simulation loop ──────────────────────────────────────────────────
    world.reset()

    # Scale ground plane for long-range flights
    try:
        stage = prim_utils.get_current_stage()
        for path in ["/World/layout", "/World/layout/GroundPlane",
                     "/World/layout/Environment",
                     "/World/layout/Environment/Geometry"]:
            prim = stage.GetPrimAtPath(path)
            if prim.IsValid():
                xf = UsdGeom.Xformable(prim)
                for op in xf.GetOrderedXformOps():
                    if op.GetOpName() == "xformOp:scale":
                        op.Set((5000.0, 5000.0, 1.0))
                        break
    except Exception as e:
        print(f"  ⚠ Could not scale ground: {e}")


    timeline = omni.timeline.get_timeline_interface()
    timeline.play()

    frame = 0
    print(f"\n🚀 YOY Trainer — mode: {args.mode.upper()}")
    print("📸 Cameras:  [1] Chase  [2] Top  [3] Side  [4] Cockpit  [5] Iso")
    print("Telemetry active …\n")

    while simulation_app.is_running():
        world.step(render=True)
        if frame % 30 == 0:
            state = airplane.state
            pos = state.position
            vel = state.linear_velocity
            speed = np.linalg.norm(vel)
            print(f"[{frame:6d}]  X={pos[0]:+8.2f}  Y={pos[1]:+8.2f}  "
                  f"Alt={pos[2]:7.2f} m  V={speed:6.2f} m/s")
        frame += 1

    input_iface.unsubscribe_to_keyboard_events(keyboard, cam_sub)
    simulation_app.close()


if __name__ == "__main__":
    main()
