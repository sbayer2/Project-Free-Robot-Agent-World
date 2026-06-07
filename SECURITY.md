# Security

pseudo-marble is a **personal research codebase**, not a deployed service. It has
**no web server, no authentication, no network endpoints, no database, and no
multi-tenant or untrusted-user surface.** It is a set of Python libraries and CLI
scripts that generate synthetic data, train a model, and compute a metric, run by
the person who cloned it. The threat model is scoped accordingly.

This document maps the usual audit concerns to what actually exists here, records
the standing audit results, and explains the automated checks in
`.github/workflows/security.yml`.

## Threat model in one line

The realistic risks are (1) **dependency CVEs**, (2) **loading untrusted asset
files** (3D meshes / images / dataset JSON) if you point the tools at data you
did not generate, and (3) **accidentally committing a secret**. Everything else
below is N/A by construction, and we say so rather than inventing surface.

## Auth patterns

**Not applicable.** There is no authentication, authorization, session, token,
cookie, or credential handling anywhere in the codebase — there is nothing to log
in to. The only identity-like value in the repo is the maintainer's own contact
email in `CLAUDE.md`. The GitHub Actions workflow runs with
`permissions: contents: read` (least privilege) and stores no secrets.

## Input validation

Inputs are local files and CLI arguments, not network requests. The validation
that exists, and where:

- **Material/config values** (`materials.py`): `VisualProps` / `PhysicsProps`
  validate ranges in `__post_init__` (e.g. `0 ≤ roughness ≤ 1`, `density > 0`).
- **Mesh integrity** (`data/mesh_validate.py`): meshes are gated on
  watertightness before they contribute physics ground truth; bad/empty meshes
  are reported, not trusted.
- **Shape/geometry** (`data/generate_mujoco.py::build_mjcf`): unknown shapes are
  rejected with an explicit error rather than silently mishandled.
- **Splits** (`splits.py`): holdout fractions and grids are validated; empty
  inputs raise.
- **Dataset/model agreement** (`models/train.py`): refuses to train if the
  dataset resolution does not match the model's expected `image_size`.

**Residual risk:** `trimesh.load(...)` (meshes) and `imageio`/`PIL` (images) parse
**arbitrary binary files**. Maliciously crafted assets can trigger
decompression bombs or parser bugs in those libraries. **Only run the generators
and loader on data you generated or otherwise trust.** Dataset `sample.json` is
parsed with the standard `json` module (no code execution).

## Unsafe operations

Audited and currently clean:

- **No dynamic code execution** — no `eval`, `exec`, `compile`, `__import__`, or
  `input()`-driven control flow. (The `eval` matches in a scan are an unrelated
  comment and MLX's `mx.eval`, which evaluates tensors, not strings.)
- **No unsafe deserialization** — no `pickle`, `marshal`, or `yaml.load`. Model
  weights use `safetensors` (MLX) / standard tensors, not pickled blobs. No
  `torch.load`/`np.load` of untrusted files.
- **No shell/command injection** — no `os.system`, no `subprocess`, no
  `shell=True`. The `scripts/*.sh` wrappers invoke `blender`/`python` with
  fixed, non-interpolated arguments.
- **File writes** go to caller-specified output directories via `os.makedirs(...,
  exist_ok=True)` + `json`/image writes; paths come from the operator's own CLI
  args (no untrusted path traversal surface).
- **MJCF generation** builds XML via f-strings; interpolated values are numeric
  physics params and names drawn from the controlled material library. Keep
  material identifiers controlled if you extend the library.

## Standing audit results (latest local run)

- **bandit** (SAST): **0 Medium/High**. Low findings are `B311` (stdlib `random`
  used only for reproducible seeds — not cryptographic; skipped with rationale in
  `pyproject.toml`) and one `B110` benign try/except fallback in image saving.
- **pip-audit**: the project declares **no required dependencies**; advisories
  that appear in a given environment generally belong to the **base image /
  toolchain** (e.g. `pip`, `setuptools`, `cryptography`), not to pseudo-marble's
  optional deps (`numpy`, `trimesh`, `imageio`, `Pillow`, and the platform-specific
  `mlx` / `mujoco` / `torch`).
- **secrets**: no API keys, tokens, or private keys committed.

## Automated checks (CI)

`.github/workflows/security.yml` runs on every push/PR to `main`, weekly, and on
demand:

| Job | Tool | Behavior |
|---|---|---|
| `sast` | bandit | **fails** the build on Medium/High; full low+ report is informational |
| `dependencies` | pip-audit | audits cross-platform deps; informational (transitive CVEs) |
| `secrets` | gitleaks | scans the full git history |

Run the same checks locally:

```bash
pip install "bandit[toml]" pip-audit
bandit -c pyproject.toml -ll -r src scripts     # the CI gate
pip-audit                                        # dependency CVEs
```

## Reporting a vulnerability

This is an independent research project. If you find a security issue, open a
GitHub issue (or, for something sensitive, contact the maintainer privately via
the address in `CLAUDE.md`). There is no SLA — it's a personal project — but
reports are welcome.
