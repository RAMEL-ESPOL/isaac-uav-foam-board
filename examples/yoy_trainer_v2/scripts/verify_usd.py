#!/usr/bin/env python3
"""Verify mass properties in the generated USD file (requires Isaac Sim)."""
import os

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

from pxr import Usd, UsdPhysics, UsdGeom  # noqa: E402

V2_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USD_PATH = os.path.join(V2_DIR, "model", "urdf", "yoy_trainer_v2_physics.usd")

stage = Usd.Stage.Open(USD_PATH)
total = 0.0
has_mesh = False

for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.MassAPI):
        m = UsdPhysics.MassAPI(prim)
        mass_val = m.GetMassAttr().Get() or 0.0
        total += mass_val
        diag = m.GetDiagonalInertiaAttr().Get()
        axes = m.GetPrincipalAxesAttr().Get()
        com  = m.GetCenterOfMassAttr().Get()
        print(f"{prim.GetPath()}")
        print(f"  mass            = {mass_val:.5f} kg")
        print(f"  diagonalInertia = {diag}")
        print(f"  principalAxes   = {axes}")
        print(f"  centerOfMass    = {com}")
    if prim.GetTypeName() == "Mesh":
        has_mesh = True
        bb = UsdGeom.Mesh(prim).GetExtentAttr().Get()
        if bb:
            size = [bb[1][i] - bb[0][i] for i in range(3)]
            print(f"  Mesh bounding box: {size[0]*1000:.1f} x {size[1]*1000:.1f} x {size[2]*1000:.1f} mm")

print(f"\nTOTAL mass = {total:.5f} kg  (expected 0.52436)")
print(f"Mass OK: {abs(total - 0.52436) < 0.001}")
print(f"Has mesh geometry: {has_mesh}")

app.close()
