"""MLX shared-latent model (encoder -> latent -> {render, physics} decoders).

DESIGN STAGE. The data pipeline comes first (see docs/ARCHITECTURE.md, "Build
order"); this package currently holds the agreed interface and the coherence
loss, implemented against MLX when it is available on the Mac. We keep imports
soft so the rest of the package and CI do not require ``mlx``.

The model the rest of the project is building toward::

    scene_description --(encoder)--> z  (single latent)
        z --(render_decoder)--> simplified MLX Gaussian splats  (appearance)
        z --(physics_decoder)--> (density, friction, restitution)  (physics)

    loss = render_loss + physics_loss + coherence_weight * coherence_loss(z, ...)

``coherence_loss`` is the novel, untested contribution: it penalizes the two
decoders for disagreeing about *what the object is*, so the latent is forced to
carry appearance and physics jointly rather than in private subspaces.
"""
