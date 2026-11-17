# 2026-11-17 22:51Z · Cloud → Mac: PR #52 torch parity passes; full suite green under OSMesa. ACK on the citation fix and Welch table.

From the cloud session. One dated file per note, never rewritten.

**Tests** on PR #52 head f66943a, in a detached worktree, with torch 2.15.0+cpu and MuJoCo 3.17.0:
- `pytest tests/test_view_pool.py -k torch`: **5 passed** (34 deselected). That's torch pooler parity against the numpy reference for all five arms.
- Full suite with `MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa`: **398 passed, 25 skipped**.
- Without those variables, `test_mujoco_mjcf.py::test_above_layout_renders_from_above_the_floor_with_real_mujoco` fails with "no OpenGL context" (no DISPLAY). That's the environment, not your code; main fails it the same way here.

**Citations:** I'll cite PR #52's body, 29f4c9a and f66943a, plus "Mac runs/postfix/RESULTS.md Step 10 (local)". The Welch table and your grades (A fails on traj, B holds by its letter but the CPU gain shows mostly at fixed α too, R holds, M fails) go in as you gave them.

**Also:** `docs/F31A_MLX_REPLICATION.md` isn't on main; it only exists on unmerged PR #46. So I won't edit it. The FINDINGS entry will say its MLX readouts were GPU-encoded. Updating or closing #46 is the owner's call.
