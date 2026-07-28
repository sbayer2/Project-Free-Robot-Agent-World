"""Every PhysicsConfig knob must actually reach a generator.

`PhysicsConfig` is a hub: all three generators (`generate_mujoco`,
`generate_blender`, `generate_gso`) import it, so a knob *looks* global at every
call site. But they are not mirrors of each other — they are different pipelines
that agree only on the `sample.json` contract, and they read **disjoint subsets**
of the config. Two failure modes follow, neither of which any other test covers:

  1. **Inert knob.** A field is declared, accepted, threaded through — and never
     read. Setting it changes nothing, silently. `gravity`, `drop_height` and
     `substeps` have been inert through F1-F21 and nothing noticed.
  2. **Silent asymmetry.** A field only one generator honors. Passing
     `PhysicsConfig(topple_jitter_reps=8)` to `generate_gso` is accepted and
     ignored, so the real-object arm cannot produce F8's soft-topple labels.

These tests are static (stdlib `ast`, no mujoco/bpy) so they run in any session.
The scan is name-based liveness, not dataflow: it answers "is this field ever
consulted on a config object?", which is exactly what catches a dead knob.
"""

import ast
import sys
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pseudomarble import materials as M  # noqa: E402
from pseudomarble.config import PhysicsConfig  # noqa: E402
from pseudomarble.data.generate_mujoco import build_mjcf  # noqa: E402

_SRC = Path(__file__).resolve().parents[1] / "src" / "pseudomarble"

# The modules a PhysicsConfig is legitimately handed to.
GENERATORS = {
    "generate_mujoco": _SRC / "data" / "generate_mujoco.py",
    "generate_blender": _SRC / "data" / "generate_blender.py",
    "generate_gso": _SRC / "data" / "generate_gso.py",
    "generate_abo": _SRC / "data" / "generate_abo.py",
}

# The three that build a config and simulate with it. `generate_abo` is a
# preparation wrapper: it forwards CLI flags into the GSO path and never reads
# a PhysicsConfig itself, so it cannot be part of a "what do they all honor?"
# question. That is asserted below rather than assumed.
SIMULATORS = ("generate_mujoco", "generate_blender", "generate_gso")

# Fields that are declared but reach no generator. Each entry is a standing
# decision to make, not a permanent exemption — this is a ratchet: adding a new
# inert field fails, and wiring one of these up also fails (so the record gets
# corrected rather than drifting).
KNOWN_INERT = {
    "gravity": (
        "Never read off the config. build_mjcf() declares its OWN gravity=-9.81 "
        "parameter and writes that into the MJCF <option> instead, so there are "
        "two sources of truth and the authoritative-looking one is inert. "
        "Decide: thread physics_cfg.gravity into build_mjcf, or delete the field."
    ),
    "drop_height": (
        "Never read in src/, tests/ or scripts/. The DROP probe carries its own "
        "height (probes.DropSpec.height). Decide: wire it, or delete the field."
    ),
    "substeps": (
        "Never read. The MJCF hardcodes timestep=0.002 and nothing derives a "
        "substep count from it. Decide: wire it, or delete the field."
    ),
}


def _fields_read(path: Path, names: set) -> set:
    """Field names in `names` consulted as an attribute in `path`.

    Attribute reads off `args` are skipped: those are argparse results, so
    `args.collision_method` proves a CLI flag exists, not that the *config
    object* is ever consulted.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute) or node.attr not in names:
            continue
        base = node.value
        if isinstance(base, ast.Name) and base.id == "args":
            continue
        found.add(node.attr)
    return found


def _read_map() -> dict:
    names = {f.name for f in fields(PhysicsConfig)}
    return {mod: _fields_read(path, names) for mod, path in GENERATORS.items()}


def test_every_physics_config_field_reaches_a_generator():
    """No knob may be declared without at least one generator consulting it."""
    declared = {f.name for f in fields(PhysicsConfig)}
    consumed = set().union(*_read_map().values())

    inert = declared - consumed
    undocumented = sorted(inert - set(KNOWN_INERT))
    assert not undocumented, (
        f"PhysicsConfig field(s) {undocumented} are declared but no generator "
        f"reads them. A knob nothing consults changes nothing when set — wire it "
        f"into a generator, or delete it. If it is deliberately inert for now, "
        f"add it to KNOWN_INERT with the decision it is waiting on."
    )

    revived = sorted(set(KNOWN_INERT) & consumed)
    assert not revived, (
        f"PhysicsConfig field(s) {revived} are now read by a generator but are "
        f"still listed in KNOWN_INERT. Remove them from that dict — the ratchet "
        f"only tightens."
    )


def test_known_inert_fields_are_exactly_the_documented_three():
    """Pin the debt so it cannot quietly grow between findings."""
    assert set(KNOWN_INERT) == {"gravity", "drop_height", "substeps"}, (
        "KNOWN_INERT changed. That is fine when a field is wired up or deleted "
        "(shrink it), but growing it means new dead config — justify it here."
    )


def test_non_default_gravity_reaches_the_mjcf():
    """The wire that DOES exist must carry a non-default value end to end.

    build_mjcf's own gravity parameter is currently the only path to the MJCF
    (see KNOWN_INERT['gravity']). Guard it, so wiring the config through later
    is a change that keeps a tested contract rather than inventing one.
    """
    xml = build_mjcf("sphere", M.get("steel"), gravity=-3.71)
    assert 'gravity="0 0 -3.71"' in xml, (
        "a non-default gravity did not reach the MJCF <option> block"
    )
    default_xml = build_mjcf("sphere", M.get("steel"))
    assert 'gravity="0 0 -9.81"' in default_xml


def test_generator_field_coverage_is_asymmetric_and_declared():
    """Which generator honors which knob — asserted, not assumed.

    This is the map that makes silent asymmetry visible. If a knob is taught to
    a second generator (e.g. soft topple reaching the GSO path), this test fails
    and the asymmetry record gets updated deliberately.
    """
    read = _read_map()
    soft_topple = {"topple_jitter_reps", "topple_jitter_impulse_rel",
                   "topple_jitter_azimuth_deg"}
    collision = {"collision_method", "collision_max_hulls"}

    # F8's soft topple is MuJoCo-only: the real-object arm cannot produce
    # smooth P(topple) labels, which is load-bearing for reading F14/F16.
    assert soft_topple <= read["generate_mujoco"]
    for mod in ("generate_blender", "generate_gso", "generate_abo"):
        assert not (soft_topple & read[mod]), (
            f"{mod} now reads soft-topple config. Good — but update this test "
            f"and docs/FINDINGS.md F8, which records the MuJoCo-only asymmetry."
        )

    # Mesh decomposition is real-object-only: primitives need no convex hulls.
    assert collision & read["generate_gso"]
    assert not (collision & read["generate_mujoco"])

    # ABO prepares assets and forwards CLI flags into the GSO path; it never
    # consults a PhysicsConfig, so physics knobs cannot be set "for ABO" except
    # through the flags generate_abo chooses to forward.
    assert read["generate_abo"] == set()

    # Of the three that do simulate, the only shared knobs are the two clock
    # settings — five of ten fields are honored by exactly one generator.
    common = set.intersection(*(read[m] for m in SIMULATORS))
    assert common == {"fps", "sim_seconds"}, (
        f"the common subset across {SIMULATORS} changed to {sorted(common)}; "
        f"these are not mirrors, so widening it is a design decision"
    )


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
