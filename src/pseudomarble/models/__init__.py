"""MLX shared-latent model (encoder -> latent -> {render, physics} decoders).

Status: encoder + behavior head are built. ``mlx_net`` is the trainable model
(MLX, runs on the Mac); ``numpy_net`` mirrors its architecture for a forward-only
check that runs in any session (MLX has no working Linux runtime); ``losses`` is
the framework-agnostic loss reference; ``train`` is the MLX training loop;
``coherence`` is the measurement. The render head (appearance) is next, and is
what enables the full render-vs-behavior coherence experiment. Imports stay soft
so the rest of the package and CI do not require ``mlx``.

The model the rest of the project is building toward::

    scene_description --(encoder)--> z  (single latent)
        z --(render_decoder)--> simplified MLX Gaussian splats  (appearance)
        z --(physics_decoder)--> (density, friction, restitution)  (physics)

    loss = render_loss + physics_loss + coherence_weight * coherence_loss(z, ...)

``coherence_loss`` is the novel, untested contribution: it penalizes the two
decoders for disagreeing about *what the object is*, so the latent is forced to
carry appearance and physics jointly rather than in private subspaces.
"""
