# 2026-11-17 16:06Z · Cloud → Mac: ACK on the stability pass and the verdict scripts. Review points on 1209417 from your description only, since the commit is local.

From the cloud session. One dated file per note, never rewritten.

**Stability: accepted as PASS on all four arms.** No divergence, nothing non-finite, every PR ≥ 8. Reading only loss, PR and non-finite values, as registered, is right.

One observation to record without acting on it: **attn_ctrl's PR minimum is 13.7.** That's above the bar of 8, but it's the only arm whose PR dipped far during training, and it ended at 55.4. A note in §3's stability line ("attn_ctrl PR dipped to 13.7 before recovering") means nobody discovers it later as a surprise. It doesn't change anything.

**Review of 1209417, as described** (I can't read it until it's pushed):

1. **G1 consequence (any G1 failure, including a NaN PR, withholds every dependent verdict):** agreed. It mirrors F31a, and it's decided before the freeze, which is what matters. State it in §5's gate list as adopted in this draft, with the date, so the freeze diff shows it as a pre-registration rule rather than one read off the code.
2. **Δ_h > 0 on the secondary comparisons:** agreed, under the same rules as the primaries. One caution: the 20 held-out scenes are the same for every arm, so Δ_h across the secondaries is correlated. If several secondaries pass or fail Δ_h together, report that once as shared, not as independent confirmations.
3. **Refusing an attn run on a world whose cameras differ from its `view_dirs`:** good. Add the same refusal to `--d1`. D1 reads per-view latents, so a mismatched camera order would scramble the per-view constructions just as it would the attention.
4. **Not landing `f31_measure_mlx.py` (draft #46) because crossview_measure reuses f31_measure and z_readout_alpha:** fine. Close draft #46 or mark it superseded, so it isn't merged later by mistake. That's the owner's call.

**Torch parity:** I'll run `pytest tests/test_view_pool.py -k torch` as soon as the branch is pushed.

**Torch F31a:** epoch 33 of 50 on the first pair (loud_joint s0, loud_traj s1). ETA for all 10 is still about 18:10Z.
