"""Repro: run_tilt's ground plane tilts ~0.349deg instead of the requested
20deg, because math.radians(spec.angle_deg) is fed into MJCF's `euler`
attribute, which MuJoCo reads in DEGREES (no <compiler angle="radian"/>
override exists in either template). Confirmed identically present in
generate_mujoco.py:386 and generate_gso.py:313.

Doubles as a regression test for the eventual fix: assertion 1 pins
build_mjcf's euler-degrees semantics (radians in -> ~0.349deg scene),
assertion 2 pins that passing degrees directly yields the intended 20deg.
Both stay true after the call-site fix; only run_tilt's inputs change.

Authored by the "free" session during the 2026-08-21 three-session audit
(diagnosis: free; second-site + framing: llama session; scene-measurement
confirmation: main session). Archived verbatim per shared-worktree
convention (one committer). See docs/FINDINGS.md F34 instrument flag.

Run: .venv/bin/python scripts/repro_tilt_bug.py
"""
import math
import sys

sys.path.insert(0, "src")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from pseudomarble import materials as M  # noqa: E402
from pseudomarble.data.generate_mujoco import build_mjcf  # noqa: E402

mat = M.Material(
    name="probe",
    visual=M.VisualProps(base_color=(0.5, 0.5, 0.5, 1), roughness=0.5,
                          metallic=0.5, transmission=0.0, ior=1.5),
    physics=M.PhysicsProps(density=6420.0, friction=0.78, restitution=0.72),
    tags=(),
)

# --- 1. Ground-normal check: this is what run_tilt actually does today ---
buggy_angle = math.radians(20.0)  # the bug: radians computed, degrees expected
xml = build_mjcf("sphere", mat, object_z=0.23, ground_euler=(buggy_angle, 0.0, 0.0))
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

ground_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
normal = data.geom_xmat[ground_id].reshape(3, 3) @ np.array([0, 0, 1.0])
tilt_deg = math.degrees(math.acos(np.clip(np.dot(normal, [0, 0, 1.0]), -1, 1)))
print(f"requested tilt: 20.0 deg  |  actual built tilt: {tilt_deg:.4f} deg")
assert abs(tilt_deg - 0.3491) < 1e-3, "bug signature not reproduced"

# --- 2. Correct call (pass degrees directly) reproduces the intended tilt ---
xml_fixed = build_mjcf("sphere", mat, object_z=0.23, ground_euler=(20.0, 0.0, 0.0))
model2 = mujoco.MjModel.from_xml_string(xml_fixed)
data2 = mujoco.MjData(model2)
mujoco.mj_forward(model2, data2)
normal2 = data2.geom_xmat[ground_id].reshape(3, 3) @ np.array([0, 0, 1.0])
tilt2 = math.degrees(math.acos(np.clip(np.dot(normal2, [0, 0, 1.0]), -1, 1)))
print(f"passing 20.0 directly (no radians()): actual tilt: {tilt2:.4f} deg")
assert abs(tilt2 - 20.0) < 1e-3, "expected fix confirmation failed"

print("\nCONFIRMED: the fix is dropping math.radians() at the two call sites")
print("(generate_mujoco.py:386, generate_gso.py:313) -- NOT adding")
print('<compiler angle="radian"/>, which would flip every other angular')
print("attribute in the template at once.")
