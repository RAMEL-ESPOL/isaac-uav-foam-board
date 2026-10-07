#!/usr/bin/env python3
"""Inspect the structure of both v1 and v2 USDs for comparison."""
import os

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

from pxr import Usd, UsdPhysics  # noqa: E402

V2_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V1_DIR = os.path.join(os.path.dirname(V2_DIR), "yoy_trainer")

def inspect(path, label):
    print(f"\n{'='*60}")
    print(f"  {label}: {path}")
    print(f"{'='*60}")
    stage = Usd.Stage.Open(path)
    if not stage:
        print("  ERROR: could not open stage")
        return
    dp = stage.GetDefaultPrim()
    print(f"  defaultPrim: {dp.GetPath() if dp else 'NONE'}")
    print(f"  rootPrims: {[str(p.GetPath()) for p in stage.GetPseudoRoot().GetChildren()]}")

    for prim in stage.Traverse():
        apis = []
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            apis.append("ArticulationRoot")
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            apis.append("RigidBody")
        if prim.HasAPI(UsdPhysics.MassAPI):
            apis.append("Mass")
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            apis.append("Collision")
        api_str = f"  [{', '.join(apis)}]" if apis else ""
        print(f"  {prim.GetPath()}  ({prim.GetTypeName()}){api_str}")

inspect(os.path.join(V1_DIR, "model", "urdf", "foam_board_plane_complete_physics.usd"), "V1")
inspect(os.path.join(V2_DIR, "model", "urdf", "yoy_trainer_v2_physics.usd"), "V2")

app.close()
