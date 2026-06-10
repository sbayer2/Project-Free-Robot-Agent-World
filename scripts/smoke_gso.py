"""In-sandbox GSO smoke test — runs the real mesh physics path end-to-end.

Builds tiny but *real* meshes (a convex box, a concave ring), runs the GSO data
layer on them — measured-mass parsing, watertight gate, CoACD concave
decomposition, mesh MJCF, and real MuJoCo drop/tilt/push physics — and checks the
MuJoCo body mass equals the measured mass. Rendering needs a GL context and is
skipped headless.

Requires the gso extra + coacd::

    pip install -e ".[gso]" coacd
    python scripts/smoke_gso.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _make_object(root, name, mesh, mass, category):
    d = os.path.join(root, name)
    os.makedirs(os.path.join(d, "meshes"))
    mesh.export(os.path.join(d, "meshes", "model.obj"))
    open(os.path.join(d, "model.sdf"), "w").write(
        f"<sdf><model><link><inertial><mass>{mass}</mass></inertial>"
        f"</link></model></sdf>")
    open(os.path.join(d, "metadata.pbtxt"), "w").write(f'categories: "{category}"\n')


def main() -> int:
    try:
        import mujoco
        import trimesh
    except Exception as exc:
        print(f"needs trimesh + mujoco: {exc}")
        return 0

    from pseudomarble.config import PhysicsConfig
    from pseudomarble.data import generate_gso as G
    from pseudomarble.data import samples
    from pseudomarble.data.generate_mujoco import build_mjcf

    root = tempfile.mkdtemp()
    _make_object(root, "BoxThing", trimesh.creation.box(extents=[0.2, 0.2, 0.2]), 0.25, "box")
    _make_object(root, "RingThing",
                 trimesh.creation.annulus(r_min=0.05, r_max=0.13, height=0.18), 0.40, "ring")

    pc = PhysicsConfig()
    objs = G.discover_objects(root)
    print(f"discovered {len(objs)} objects")
    ok = True
    for o in objs:
        half_h, density = G.mesh_halfheight_and_density(o.mesh_path, o.mass_kg, 1.0)
        parts, meta = G.build_collision(o.mesh_path, os.path.join(root, "_o", o.object_id),
                                        pc.collision_method, pc.collision_max_hulls)
        asset = G.to_mesh_asset(o, half_h, 1.0, 0.5, parts)
        model = mujoco.MjModel.from_xml_string(build_mjcf(mesh=asset))
        body_mass = float(model.body("object").mass[0])
        beh = G.run_probes_mesh(asset, pc)
        outcomes = {r["probe"]: r["outcome"] for r in beh}
        mass_ok = abs(body_mass - o.mass_kg) < 1e-6
        ok = ok and mass_ok
        print(f"  {o.object_id:9s} mass={o.mass_kg}kg body={body_mass:.3f}kg "
              f"{'OK' if mass_ok else 'MASS-MISMATCH'} | "
              f"collision={meta['method']}x{meta['n_parts']} "
              f"concave={meta['concavity_preserved']} | "
              f"drop.settle={outcomes['drop']['settle_time']:.2f}s "
              f"push.toppled={outcomes['push']['toppled']}")
        samples.physics_labels_measured(asset.mass, density)  # honest provenance
    print("SMOKE OK" if ok else "SMOKE FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
