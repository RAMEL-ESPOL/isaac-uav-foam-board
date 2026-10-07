#!/usr/bin/env python3
"""
Convert yoy_trainer_v2.urdf to yoy_trainer_v2_physics.usd using the
Isaac Sim URDF importer in headless mode.

Importer settings:
  - Fix Base Link: OFF
  - Import Inertia Tensor: ON
  - Self Collision: OFF
  - Stage units: 1.0 m/unit

Usage (from the yoy_trainer_v2 directory):
  $ISAACSIM_PYTHON scripts/urdf_to_usd.py
"""

import os
import sys

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
URDF_PATH = os.path.join(SCRIPT_DIR, "model", "urdf", "yoy_trainer_v2.urdf")
USD_PATH  = os.path.join(SCRIPT_DIR, "model", "urdf", "yoy_trainer_v2_physics.usd")

# Ground-contact spheres, by their <collision name=...> in the URDF.
CONTACT_SPHERES = ("wheel_right", "wheel_left", "tail_contact")
# Rolling resistance of small wheels on a hard runway is ~0.02-0.05.
CONTACT_FRICTION = 0.04

print(f"[urdf_to_usd] URDF: {URDF_PATH}")
print(f"[urdf_to_usd] USD:  {USD_PATH}")

if not os.path.exists(URDF_PATH):
    print(f"[urdf_to_usd] ERROR: URDF not found at {URDF_PATH}")
    sys.exit(1)

# ── Isaac Sim bootstrap ───────────────────────────────────────────────────
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True, "multi_gpu": False,
                                "active_gpu": 0, "max_gpu_count": 1})

import omni.kit.commands                                       # noqa: E402
from pxr import PhysxSchema, UsdPhysics, UsdShade, Usd        # noqa: E402

# ── Build import config ───────────────────────────────────────────────────
status, import_config = omni.kit.commands.execute("URDFCreateImportConfig")

import_config.fix_base = False               # avion libre, no soldado al mundo
import_config.import_inertia_tensor = True   # conservar el tensor de SolidWorks
import_config.self_collision = False
import_config.merge_fixed_joints = False
import_config.convex_decomp = False

print("[urdf_to_usd] Config: fix_base=False, import_inertia_tensor=True, self_collision=False")

# ── Execute the import command ────────────────────────────────────────────
# dest_path must be the output file. Importing into the in-memory stage and
# calling stage.Export() leaves the meshes under /visuals, /colliders and
# /meshes, OUTSIDE the robot prim, so they are lost when Pegasus references
# the file into /World/fixedwing0.
omni.kit.commands.execute(
    "URDFParseAndImportFile",
    urdf_path=URDF_PATH,
    import_config=import_config,
    dest_path=USD_PATH,
)

# ── Post-process the top-level layer ──────────────────────────────────────
# 1. The headless importer does not set defaultPrim on the top-level file, and
#    Pegasus references the file by its defaultPrim: without it the vehicle
#    prim ends up empty.
# 2. The URDF has a single link and no joints, so the importer does not add an
#    ArticulationRootAPI anywhere. Pegasus's Vehicle is an Articulation, so
#    world.reset() fails with "Failed to find articulation". The root must go
#    on the rigid body itself (base_link): on the parent Xform PhysX does not
#    create a single-link articulation.
stage = Usd.Stage.Open(USD_PATH)
root_layer = stage.GetRootLayer()
root_layer.defaultPrim = root_layer.rootPrims[0].name
robot_prim = stage.GetDefaultPrim()
base_link = robot_prim.GetChild("base_link")
UsdPhysics.ArticulationRootAPI.Apply(base_link)

# 3. Low-friction material on the three ground-contact spheres declared in the
#    URDF. It stands in for wheel rolling resistance: with the default ground
#    friction (~0.5+) the drag at the main wheels, 13 cm below the CG and only
#    31 mm ahead of it, flips the aircraft onto its nose on the takeoff roll.
#    The airframe hull keeps the default friction, so a crash still scrapes.
#    The importer makes base_link/collisions instanceable, which locks its
#    children; un-instancing it lets the binding be authored here.
material = UsdShade.Material.Define(
    stage, robot_prim.GetPath().AppendPath("PhysicsMaterials/contact_low_friction"))
physx_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
physx_material.CreateStaticFrictionAttr(CONTACT_FRICTION)
physx_material.CreateDynamicFrictionAttr(CONTACT_FRICTION)
physx_material.CreateRestitutionAttr(0.0)
# PhysX averages the two materials' friction by default, which against the
# ground (~0.5-1.0) would undo this. "min" outranks "average", so the
# pair uses CONTACT_FRICTION whatever the ground material is.
combine = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
combine.CreateFrictionCombineModeAttr("min")
combine.CreateRestitutionCombineModeAttr("min")

collisions = base_link.GetChild("collisions")
collisions.SetInstanceable(False)
for name in CONTACT_SPHERES:
    sphere = collisions.GetChild(name).GetChild("sphere")
    if not sphere.IsValid():
        raise SystemExit(f"[urdf_to_usd] ERROR: contact sphere {name} missing in the URDF")
    UsdShade.MaterialBindingAPI.Apply(sphere).Bind(
        material, UsdShade.Tokens.weakerThanDescendants, "physics")

root_layer.Save()

print(f"[urdf_to_usd] Robot prim: {robot_prim.GetPath()} (ArticulationRootAPI on base_link)")
print(f"[urdf_to_usd] Friction {CONTACT_FRICTION} on {', '.join(CONTACT_SPHERES)}")

# ── Verify mass properties ────────────────────────────────────────────────
total_mass = 0.0
for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.MassAPI):
        m = UsdPhysics.MassAPI(prim)
        mass_val = m.GetMassAttr().Get() or 0.0
        total_mass += mass_val
        diag = m.GetDiagonalInertiaAttr().Get()
        axes = m.GetPrincipalAxesAttr().Get()
        com  = m.GetCenterOfMassAttr().Get()
        print(f"  {prim.GetPath()}")
        print(f"    mass            = {mass_val:.5f} kg")
        print(f"    diagonalInertia = {diag}")
        print(f"    principalAxes   = {axes}")
        print(f"    centerOfMass    = {com}")

print(f"\n[urdf_to_usd] TOTAL mass = {total_mass:.5f} kg  (expected 0.52436)")

if abs(total_mass - 0.52436) > 0.001:
    print("[urdf_to_usd] WARNING: mass mismatch!")

print(f"[urdf_to_usd] Saved to {USD_PATH}")

simulation_app.close()
print("[urdf_to_usd] Done.")
