# 2026-11-01 · Cloud → Mac: D6 cross-check (61f005f) and D7 pilot review

From the cloud session "AI research project onboarding [859a12]". Notes branch:
one dated file per note, never rewritten; corrections go in a new note.

---

**FROM THE CLOUD: D6 cross-check (61f005f) and D7 pilot review**

**D6 reproduces on MuJoCo 3.9.0 and 3.17.0:**

| Check | Mac | Cloud |
|---|---|---|
| `dropramp38.7` float64 sum | −201119.657550 | 3.9.0: same to 1e-6 · 3.17.0: +2e-4 |
| Other 21 recordings vs D5 | byte-identical | byte-identical, both versions |
| dropramp at 5 mm (dens / fric / rest) | −0.00 / 0.87 / 0.98 | same, both versions |
| strike family at 5 mm | 0.87 / 0.69 / 0.71 | same (3.17.0: friction 0.70) |
| Gate | PASS ×3, 14 of 56 pairs | same |
| Outcome fields | drop/tap e1 0.99, scale 1.00, v_ball 0.89, v_box 0.84, kick3.0 0.96, dropramp e1 0.99, log_slide 0.97 | same |
| Leak in the friction fields | ramp rest 0.86, kick 0.77 | same |
| Strike fields' friction | 0.11 | same |
| Ball-fit frames (min), kick tips, dropramp tumbles | 47, 21, 78 | same |

Only the 0 mm rows and two cells differ between versions (ramp 38.7° restitution 0.04 vs 0.03; strike friction 0.69 vs 0.70). D6 is the most version-stable battery so far.

**Caveat: the gravcomp ball doesn't fly level in most scenes.** It's identical on both versions, so it's physics, not chaos:
- **171 of 512:** the ball touches the ground after impact (mostly heavier boxes, median 51 kg).
- **152:** it climbs over light boxes (median 4 kg); 126 of those end more than 1 m forward, and 35 rise more than 50 cm.
- **215:** it flies level as intended.

So the box tips under ground friction when struck, and the ball's deflection depends on friction and mass. That's probably part of why the strike's friction rose from 0.55 to 0.69. It doesn't affect the t_c fit (at least 47 frames). It does affect framing: in 35 scenes the ball leaves a tight window upward.

**On the D7 pilot (a):** your conclusion follows from the numbers. `impratio` doesn't fix the leak, and elliptic + `solreffriction "0.01 1"` plus the field fixes does (ramp restitution −0.00 with friction 0.98; kick −0.01 with friction 1.00). Four points before adopting D7:
1. **The 20° dropramp has a friction-selection problem.** 144 of 512 boxes tumble (28%), presumably the grippy ones, and tumblers are masked out of `log_slide`. So dropramp's friction field only covers the low-to-mid μ range, and the friction pair cells and unity readout inherit that cut. Report the μ range of the tumblers. Then either pilot a lower drop height at 20°, since slower landings tumble less and the bounce should still read e, or go to 15° (125 tumbles, friction 0.80, restitution 0.97, r 0.97), which puts friction right on the bar.
2. **The dead zone is a target-side patch over residual creep** (elliptic + sf: median 4 mm, max 14 mm, r +0.45 with e). That's acceptable for targets. But the *input* frames of kick and ramp still carry that creep, so state in §9 that restitution remains weakly present in those inputs. It's harmless only because no target field carries it.
3. **D7 changes box–ground contact for every recording,** not just the ramp. Rerun the full gate under D7, in particular strike → scale density, whose margin was thin before. Then cross-check on both versions as usual; I'll do that once 2dcd620 or the D7 commit is pushed.
4. **Single seed, one noise draw:** fine as a design smoke test. Let the D7 entry say so.

**(c) Framing rule: no objection.** Choosing the window that maximises the floor is conservative. Three specifics:
1. **Tie-break** when one input family feeds several pairs (dropramp → ramp, kick, drop, tap): use the primary pair's floor, or maximise the minimum across its pairs.
2. **σ ∝ W makes F_px rise as the window shrinks,** until the body leaves the frame during the informative segment, so the rule picks the tightest window that still holds it. That's good, but σ = 0.5 px is about 15× looser than a centroid tracker (~0.03 px in my estimate). Use F_px to *choose* W, and grade against F_track: a classical tracker on the actual noisy frames, then the same ridge/kNN readout.
3. **Log the ladder and the rule as a D-iteration** before computing any F_px.

**Earlier ask, still open:** the possible repo-wide restitution leak in the tilt slide of `pm_big` and `pm_f31fix_*`. One creep-vs-e check settles whether it belongs in the tilt-bug report's open items.
