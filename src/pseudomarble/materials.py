"""The principled material library — *the coupling we author on purpose*.

The central honesty problem of this project (see ``docs/ARCHITECTURE.md``):
Blender decouples how a thing *looks* (the Principled BSDF shader) from how it
*behaves* (rigid-body friction / restitution / mass). A ceramic-looking shader
does not automatically get ceramic physics. So any appearance<->physics coupling
the model could learn is coupling that *we* inject into the data.

We make that injection **principled** rather than arbitrary: each material binds
a physically motivated appearance to physically motivated physical properties,
through a single real-world material identity. Glass looks transmissive *and* is
dense and brittle; rubber looks matte-dark *and* is high-friction and bouncy.

The model's job is to recover this appearance->physics map. The *honest* test
(see ``splits.py``) is whether it generalizes the map to material x shape
combinations it never saw paired — interpolating the coupling rather than
memorizing seen pairs.

All values are order-of-magnitude-plausible reference numbers, not measurements.
We are, as the design notes put it, learning *Blender's* eigenvector — a decent
approximation of reality's, not reality's itself.
"""

from __future__ import annotations

import colorsys
import math
import random
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class VisualProps:
    """Inputs to a Blender Principled BSDF (sRGB-ish 0..1 where applicable)."""

    base_color: Tuple[float, float, float, float]  # RGBA
    roughness: float
    metallic: float
    specular: float = 0.5
    transmission: float = 0.0  # 1.0 == fully transmissive (glass/ice)
    ior: float = 1.45

    def __post_init__(self) -> None:  # lightweight validation
        for name, v in (
            ("roughness", self.roughness),
            ("metallic", self.metallic),
            ("specular", self.specular),
            ("transmission", self.transmission),
        ):
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"VisualProps.{name} out of [0,1]: {v}")
        if len(self.base_color) != 4:
            raise ValueError("base_color must be RGBA (4 floats)")


@dataclass(frozen=True)
class PhysicsProps:
    """Inputs to Blender's (Bullet) rigid-body solver, plus density for mass."""

    density: float  # kg / m^3
    friction: float  # dimensionless coefficient
    restitution: float  # bounciness, 0..1
    linear_damping: float = 0.04
    angular_damping: float = 0.10

    def __post_init__(self) -> None:
        if self.density <= 0:
            raise ValueError(f"density must be > 0: {self.density}")
        if not 0.0 <= self.restitution <= 1.0:
            raise ValueError(f"restitution out of [0,1]: {self.restitution}")
        if self.friction < 0:
            raise ValueError(f"friction must be >= 0: {self.friction}")


@dataclass(frozen=True)
class Material:
    """A single material identity binding appearance to physics."""

    name: str
    visual: VisualProps
    physics: PhysicsProps
    tags: Tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tags"] = list(self.tags)
        return d


def _mat(
    name: str,
    color: Tuple[float, float, float, float],
    roughness: float,
    metallic: float,
    density: float,
    friction: float,
    restitution: float,
    *,
    transmission: float = 0.0,
    ior: float = 1.45,
    tags: Tuple[str, ...] = (),
) -> Material:
    return Material(
        name=name,
        visual=VisualProps(
            base_color=color,
            roughness=roughness,
            metallic=metallic,
            transmission=transmission,
            ior=ior,
        ),
        physics=PhysicsProps(density=density, friction=friction, restitution=restitution),
        tags=tags,
    )


# The library. Appearance and physics co-vary through material identity — that
# co-variation *is* the signal the shared latent is supposed to discover.
MATERIALS: Dict[str, Material] = {
    "ceramic": _mat(
        "ceramic", (0.92, 0.90, 0.85, 1.0), roughness=0.20, metallic=0.0,
        density=2300, friction=0.40, restitution=0.30, tags=("brittle", "hard"),
    ),
    "rubber": _mat(
        "rubber", (0.08, 0.08, 0.09, 1.0), roughness=0.95, metallic=0.0,
        density=1100, friction=1.05, restitution=0.80, tags=("soft", "grippy", "bouncy"),
    ),
    "steel": _mat(
        "steel", (0.56, 0.57, 0.58, 1.0), roughness=0.22, metallic=1.0,
        density=7850, friction=0.42, restitution=0.55, tags=("metal", "hard", "heavy"),
    ),
    "wood": _mat(
        "wood", (0.45, 0.30, 0.16, 1.0), roughness=0.70, metallic=0.0,
        density=700, friction=0.50, restitution=0.40, tags=("organic",),
    ),
    "foam": _mat(
        "foam", (0.86, 0.84, 0.78, 1.0), roughness=1.0, metallic=0.0,
        density=50, friction=0.60, restitution=0.08, tags=("soft", "light", "damping"),
    ),
    "glass": _mat(
        "glass", (0.92, 0.96, 0.97, 1.0), roughness=0.02, metallic=0.0,
        density=2500, friction=0.45, restitution=0.40,
        transmission=1.0, ior=1.50, tags=("brittle", "transparent"),
    ),
    "plastic": _mat(
        "plastic", (0.15, 0.45, 0.80, 1.0), roughness=0.40, metallic=0.0,
        density=1050, friction=0.35, restitution=0.55, tags=("synthetic",),
    ),
    "stone": _mat(
        "stone", (0.42, 0.41, 0.40, 1.0), roughness=0.85, metallic=0.0,
        density=2700, friction=0.62, restitution=0.25, tags=("hard", "heavy", "rough"),
    ),
    "cardboard": _mat(
        "cardboard", (0.70, 0.56, 0.36, 1.0), roughness=0.92, metallic=0.0,
        density=250, friction=0.52, restitution=0.10, tags=("light", "damping"),
    ),
    "ice": _mat(
        "ice", (0.80, 0.90, 0.96, 1.0), roughness=0.10, metallic=0.0,
        density=917, friction=0.05, restitution=0.12,
        transmission=0.80, ior=1.31, tags=("slippery", "transparent", "brittle"),
    ),
}


def get(name: str) -> Material:
    """Look up a material by name (raises KeyError with a helpful message)."""
    try:
        return MATERIALS[name]
    except KeyError as exc:  # pragma: no cover - trivial
        raise KeyError(
            f"unknown material {name!r}; known: {sorted(MATERIALS)}"
        ) from exc


def names() -> Tuple[str, ...]:
    """Stable, sorted tuple of material names (deterministic for splits)."""
    return tuple(sorted(MATERIALS))


def physics_vector(name: str) -> Tuple[float, float, float]:
    """The (density, friction, restitution) target the physics decoder predicts.

    Density is returned in kg/m^3; callers that need a normalized target should
    scale it themselves (see ``config.PHYSICS_NORMALIZERS``).
    """
    p = get(name).physics
    return (p.density, p.friction, p.restitution)


# --------------------------------------------------------------------------- #
# Continuous material sampler — the fix for the "10-row lookup table" problem.
# --------------------------------------------------------------------------- #
# The discrete MATERIALS above are useful anchors, but a model can "solve" them
# by recognizing which of 10 names it sees and looking up three numbers. That is
# memorization, not understanding the physical essence of a thing.
#
# Instead we sample a hidden continuous "essence" from a few latent factors, and
# produce BOTH appearance and physics from it. Appearance reveals the essence
# only *partially and noisily*, so the model must infer it from visual evidence
# (the way you infer "heavy" from "looks metallic") rather than read a label.
# Because the essence is continuous, held-out *regions* of essence-space test
# real interpolation/extrapolation (see splits.make_region_split).

# Physical ranges the essence spans (plausible, not measured).
DENSITY_RANGE = (50.0, 8000.0)      # kg/m^3, sampled log-uniform
FRICTION_RANGE = (0.05, 1.10)
RESTITUTION_RANGE = (0.05, 0.85)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


@dataclass(frozen=True)
class MaterialSample:
    """A continuously-sampled material plus the latent factors that produced it.

    ``material`` is the usual appearance+physics object the generators consume.
    ``factors`` and ``nearest_anchor`` are recorded as ground truth for analysis
    (e.g. measuring how well a model recovers the hidden essence).
    """

    material: Material
    factors: Dict[str, float]      # heaviness/grip/hardness/clarity in [0,1]
    nearest_anchor: str            # closest named material, for interpretable eval


class MaterialSampler:
    """Draws continuous materials whose appearance is a noisy function of physics.

    Four latent factors drive everything so the coupling is principled:
      * heaviness -> density (and a metallic look, since metals are dense)
      * grip      -> friction (and a rougher surface)
      * hardness  -> restitution (and a smoother surface)
      * clarity   -> transmission (transparent kinds), mostly low
    Gaussian noise on the *appearance* channels makes the map non-invertible, so
    the essence must be inferred, not read off.
    """

    def __init__(self, seed: int = 0, appearance_noise: float = 0.07) -> None:
        self.rng = random.Random(seed)
        self.noise = appearance_noise

    # -- internals -------------------------------------------------------- #
    def _n(self) -> float:
        return self.rng.gauss(0.0, self.noise)

    def _appearance_from_factors(
        self, heaviness: float, grip: float, hardness: float, clarity: float
    ) -> VisualProps:
        # Metals are dense and metallic; transparent kinds are not metallic.
        metallic = _clip01((1.2 * heaviness - 0.2) * (1.0 - clarity) + self._n())
        # Smooth when hard/clear; rough when grippy.
        roughness = _clip01(0.90 - 0.55 * hardness - 0.70 * clarity + 0.25 * grip + self._n())
        transmission = _clip01(clarity + 0.5 * self._n())
        ior = _lerp(1.30, 1.50, clarity)
        # Colour is a deliberately weak cue: random hue, value dimming with mass,
        # so the model can't rely on colour alone and must use roughness/metallic.
        hue = self.rng.random()
        sat = _clip01(0.15 + 0.45 * self.rng.random() - 0.3 * transmission)
        val = _clip01(0.85 - 0.35 * heaviness + self._n())
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        return VisualProps(
            base_color=(r, g, b, 1.0),
            roughness=max(0.02, roughness),
            metallic=metallic,
            transmission=transmission,
            ior=ior,
        )

    def _physics_from_factors(
        self, heaviness: float, grip: float, hardness: float
    ) -> PhysicsProps:
        log_d = _lerp(math.log(DENSITY_RANGE[0]), math.log(DENSITY_RANGE[1]), heaviness)
        density = math.exp(log_d)
        friction = _lerp(*FRICTION_RANGE, grip)
        restitution = _lerp(*RESTITUTION_RANGE, hardness)
        return PhysicsProps(density=density, friction=friction, restitution=restitution)

    # -- public API ------------------------------------------------------- #
    def sample(self, material_id: Optional[str] = None) -> MaterialSample:
        heaviness = self.rng.random()
        grip = self.rng.random()
        hardness = self.rng.random()
        clarity = self.rng.random() ** 3  # skew toward opaque
        return self._assemble(heaviness, grip, hardness, clarity, material_id)

    def sample_near(self, anchor: str, jitter: float = 0.15,
                    material_id: Optional[str] = None) -> MaterialSample:
        """Sample continuously *around* a named anchor (for controlled studies)."""
        base = get(anchor)
        heaviness = _clip01(_inv_log(base.physics.density, DENSITY_RANGE) + self._jit(jitter))
        grip = _clip01(_inv_lin(base.physics.friction, FRICTION_RANGE) + self._jit(jitter))
        hardness = _clip01(_inv_lin(base.physics.restitution, RESTITUTION_RANGE) + self._jit(jitter))
        clarity = _clip01(base.visual.transmission + self._jit(jitter))
        return self._assemble(heaviness, grip, hardness, clarity, material_id)

    def _jit(self, jitter: float) -> float:
        return self.rng.uniform(-jitter, jitter)

    def _assemble(self, heaviness, grip, hardness, clarity, material_id) -> MaterialSample:
        visual = self._appearance_from_factors(heaviness, grip, hardness, clarity)
        physics = self._physics_from_factors(heaviness, grip, hardness)
        mat = Material(name=material_id or "sampled", visual=visual, physics=physics)
        factors = {
            "heaviness": heaviness, "grip": grip,
            "hardness": hardness, "clarity": clarity,
        }
        return MaterialSample(mat, factors, nearest_anchor(physics))


def _inv_log(value: float, rng: Tuple[float, float]) -> float:
    return (math.log(value) - math.log(rng[0])) / (math.log(rng[1]) - math.log(rng[0]))


def _inv_lin(value: float, rng: Tuple[float, float]) -> float:
    return (value - rng[0]) / (rng[1] - rng[0])


def nearest_anchor(physics: PhysicsProps) -> str:
    """Closest named material to a physics vector, in normalized essence space."""
    from pseudomarble.config import PHYSICS_NORMALIZERS as N

    def norm(p: PhysicsProps) -> Tuple[float, float, float]:
        return (p.density / N["density"], p.friction / N["friction"],
                p.restitution / N["restitution"])

    q = norm(physics)
    best, best_d = "", float("inf")
    for name in names():
        a = norm(get(name).physics)
        d = sum((qi - ai) ** 2 for qi, ai in zip(q, a))
        if d < best_d:
            best, best_d = name, d
    return best
