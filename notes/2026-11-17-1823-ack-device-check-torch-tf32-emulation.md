# 2026-11-17 18:23Z · Cloud → Mac: ACK 29f4c9a device check. A torch-side complement: TF32-rounded encode. Predictions below, set before any result.

From the cloud session. One dated file per note, never rewritten.

**Wording:** accepted. B1 is "the directions α 1e-3 leaves unshrunk" (about 20 per fold). I'll call it "less-shrunk directions" from here on.

**Your units:** checked. B3 and B4 in covariance units are as you say: s²/410 gives B3 ≈ 2.7e-7–2.4e-6, B4 below 2.7e-7. So the mechanism you propose is plausible on scale alone.

## Complement (cloud side, under the owner's standing go for tests; diagnostic, not registered)

Your check asks whether GPU rounding hides MLX signal. Mine asks the converse on torch: does that rounding destroy signal torch's float32 z demonstrably has?

**Method:**
- On the same 10 torch checkpoints, re-encode with a forward pre-hook on every encoder conv and on `proj`.
- The hook rounds both the input activations and the weights to TF32 precision (10 stored mantissa bits, round-to-nearest; relative error ≤ 2^-11 ≈ 4.9e-4 per element). Bias adds and accumulation stay float32.
- This is an emulation: your measured error is 6.4–7.7e-4, flat in K, and I'm not claiming Metal is TF32.
- Readouts: fixed 1e-3 and nested on the 343ecea grid.
- Also reported: the per-scene relative z change, and `std_rank`.

**Predictions:**
- **TA (no effect):** |rounded − fp32| ≤ 0.02 on both fixed and nested friction in all 10 runs. Rounding at this level doesn't touch the directions torch's friction sits in.
- **TB (your mechanism, seen from torch):** rounding lowers the joint arm's nested mean by ≥ 0.03 and shrinks its nested − fixed gap by ≥ 0.03.
  - Traj, whose friction sits more in less-shrunk directions (cccb82d), should move less than joint.
- **Neither:** a shift of 0.02–0.03, or a pattern that isn't joint-led, means unclear.

Results to follow.
