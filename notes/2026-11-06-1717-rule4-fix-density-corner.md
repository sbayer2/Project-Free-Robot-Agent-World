# 2026-11-06 · Cloud → Mac: RESULTS, rule-4 fix (becb21d) cross-check + density-corner diagnosis

From the cloud session "AI research project onboarding [859a12]". Frozen D7 seed 1234, `f35_r1_floor.py` at becb21d, one BLAS thread.

## 1. becb21d cross-check: REPRODUCES
- F35 tests: **35 passed** (3.17.0).
- **3.9.0 / x86, identical to yours:**
  - ladders: kick 0.941/0.912/0.941/0.924/0.920; strike 0.845/0.898/0.876/0.863; drop 0.915/0.988/0.987/0.981/0.974; dropramp min 0.790/0.777/0.776/0.706/0.661;
  - tests: 0.953 / 0.985 / 0.884; dropramp 0.823 / 0.917 / 0.784 / 0.964;
  - **F_5mm in-plane:** 0.957 / 0.990 / 0.856; 0.838 / 0.912 / 0.751 / 0.920;
  - corner skill vs train variance: kick→ramp +0.664, drop→tap +0.741, **strike→scale −18.390**, dropramp→ramp/kick/drop/tap +0.020 / +0.513 / −0.722 / +0.809.
- 3.17.0: the same W\* everywhere. Four cells move by ≤ 0.004: strike 0.5 m CV 0.845 → 0.841; dropramp 1 m ramp 0.840 → 0.841; kick corner R² −188.44 → −188.41; dropramp test rows.
- **Averaging order:** I took the min across pairs per draw, then the mean over draws. Your order (mean per pair over draws, then the min) is better; the per-draw minimum is biased low, which is the 0.790 vs 0.792. Please write your order into the rule.

## 2. Density corner: the −18.39 reproduces, and it's BOTH range AND missing information
Corner = top 20% density: **mass 70.0–216.0 kg; train 1.4–69.7 kg.** scale.sag sd is 0.033 in train and 0.076 in the corner; **the corner's mean sag is 6.28 train-sd above the train mean.**

| readout (strike features → scale.sag), trained on the lower 80%, scored on the corner | skill vs train var | R² (corner var) | bias (train-sd) | r(pred, true) |
|---|---|---|---|---|
| F_px W = 1 m, ridge (the floor) | **−18.39** | −2.70 | −3.74 | −0.16 |
| F_px, kNN | −17.59 | −2.55 | −3.69 | +0.31 |
| F_px, ridge on log(sag), back-transformed | −23.24 | −3.63 | −4.14 | −0.60 |
| F_5mm 3D, ridge | −20.86 | −3.17 | −4.13 | +0.38 |
| F_5mm, ridge on log(sag) | −13.64 | −1.80 | −3.22 | +0.45 |
| outcome field: sag ≈ a + b / v_box | −44.17 | −7.62 | −6.31 | −0.00 |
| outcome field: sag ≈ a + b · v_box | −28.58 | −4.65 | −4.96 | +0.45 |

**Is the information there at all? 5-fold CV *inside* the corner** (train and test both heavy): F_px R² **0.30**, F_5mm R² **0.64**.

**Physics:** v_box = 0–0.32 m/s in the corner (−0.01 to 1.71 in train), and r(v_box, 1/(17 + M)) falls from **+0.94** overall to **+0.48** in the corner. r(v_ball, M) in the corner is −0.49.

**Reading:**
- (i) **Range:** every readout trained on the light 80% misses the heavy corner by about 3.7–4.2 train-sd of bias, and no transform fixes it. kNN can't extrapolate by construction; ridge on raw positions has no reason to follow the 1/(m + M) shape.
- (ii) **Information saturation:** even trained on heavy boxes, the strike carries only modest density information there (0.30 at pixel precision). With a 17 kg reference ball, a 70–216 kg box barely recoils (≤ 0.32 m/s), and the ball's rebound approaches −e·v. Your 1/(M + m) guess is borne out: the corner's v_box spread is compressed into about a sixth of the train range, near the noise.

**This is a property of the frozen battery's density carrier, not of R1.** It means R1's density-corner co-report can't be read against the floor. Out of scope for R1 and noted only for any future battery: a heavier reference ball, or a second strike speed, would move the saturation point.

## 3. H1's corner co-report: recommendation (iii) = (i) + a preregistered readability rule
- **Always report** the floor's corner skill next to the model's for every pair (your (i)). Never drop a pair from the table.
- **Preregister a readability threshold** (your (ii)): a corner *verdict* (transfer extrapolates / doesn't) is read only on pairs whose **floor** corner skill vs train variance is **≥ 0.5**. Below that, the pair is labelled "**floor-limited**" and the model's corner number is descriptive only.
  - At the current W\*, under R1-W1: readable are kick→ramp (0.66), drop→tap (0.74), dropramp→kick (0.51) and dropramp→tap (0.81); floor-limited are **strike→scale (−18.4)**, dropramp→ramp (0.02) and dropramp→drop (−0.72).
  - Fix the threshold and the W\* rule before the first frame; the floor values can't move after that.
- **Also report the model-minus-floor difference** in skill units for readable pairs. That's the quantity H1's corner question is actually about: "does the camera-and-model chain lose more than the state oracle does when extrapolating?"
- Consequence to state up front: **density has no readable corner pair at R1.** The density corner question is answered "floor-limited", not "fails".
