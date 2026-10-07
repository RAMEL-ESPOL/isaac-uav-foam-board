#!/usr/bin/env python3
"""
Write meshes/base_link_body_collision.STL: base_link.STL without the landing
gear, used as the airframe collider (convex hull) in yoy_trainer_v2.urdf.

Ground contact is handled by three spheres declared in the URDF (two main
wheels + tail). If the airframe hull also contained the wheels it would touch
the ground at the same points, and its friction would flip the aircraft onto
its nose during the takeoff roll.

Keeps every triangle whose three vertices lie above GEAR_CUT_Z. Everything
below it is landing gear: the fuselage belly bottoms out at z = -0.071 m.

Plain numpy, no Isaac Sim needed:
  python3 scripts/make_body_collision.py
"""

import os

import numpy as np

GEAR_CUT_Z = -0.09   # m, link frame

MESH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model", "meshes")
SRC = os.path.join(MESH_DIR, "base_link.STL")
DST = os.path.join(MESH_DIR, "base_link_body_collision.STL")

TRI = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])

data = open(SRC, "rb").read()
count = int(np.frombuffer(data, dtype="<u4", count=1, offset=80)[0])
tris = np.frombuffer(data, dtype=TRI, count=count, offset=84)

body = tris[(tris["v"][:, :, 2] > GEAR_CUT_Z).all(axis=1)]

with open(DST, "wb") as f:
    f.write(b"yoy_trainer_v2 airframe collider (landing gear removed)".ljust(80, b"\0"))
    f.write(np.array([len(body)], dtype="<u4").tobytes())
    f.write(body.tobytes())

print(f"{len(body)} of {count} triangles -> {DST}")
