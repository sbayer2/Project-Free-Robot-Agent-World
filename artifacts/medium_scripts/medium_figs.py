"""Figures for the pseudo-marble Medium draft. Every number is copied from a
cited project source (FINDINGS, the tilt-bug propagation report, the F35
design note and freeze table, the lab notebook); nothing is recomputed here."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"      # categorical slot 1
ORANGE = "#eb6834"    # categorical slot 2
CONTEXT = "#c3c2b7"   # muted context (the "before" state)
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def frame(ax, ygrid=True):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(length=0)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)


def titles(fig, title, sub):
    fig.text(0.04, 0.955, title, fontsize=15, fontweight="bold", color=INK, va="top")
    fig.text(0.04, 0.885, sub, fontsize=10.5, color=INK2, va="top")


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=200)
    plt.close(fig)


def bar_labels(ax, bars, fmt="{:.2f}", pad=0.015, color=INK2):
    for b in bars:
        h = b.get_height()
        y = h + pad if h >= 0 else h - pad
        ax.text(b.get_x() + b.get_width() / 2, y, fmt.format(h).replace("-", "\u2212"), ha="center",
                va="bottom" if h >= 0 else "top", fontsize=9.5, color=color)


def legend(ax, handles, labels, loc="upper left"):
    lg = ax.legend(handles, labels, loc=loc, frameon=False, fontsize=10,
                   handlelength=1.0, handleheight=0.9)
    for t in lg.get_texts():
        t.set_color(INK2)


# Figure 1: friction's share of behavior, flat vs real ramp
fig, ax = plt.subplots(figsize=(7.2, 4.4))
fig.subplots_adjust(left=0.08, right=0.98, top=0.74, bottom=0.14)
titles(fig, "Friction barely mattered on the flat ramp",
       "Share of the behavior targets' variance explained by one property, same objects\n"
       "before and after the units fix (cubic fit on all 21 behavior fields)")
groups = ["Mixed shapes\nfriction", "Mixed shapes\nrestitution", "One box shape\nfriction", "One box shape\nrestitution"]
flat = [0.01, 0.20, 0.03, 0.75]
real = [0.25, 0.05, 0.87, 0.08]
x = np.arange(len(groups)); w = 0.36
b1 = ax.bar(x - w / 2 - 0.01, flat, w, color=CONTEXT, edgecolor=SURFACE, linewidth=1.5)
b2 = ax.bar(x + w / 2 + 0.01, real, w, color=BLUE, edgecolor=SURFACE, linewidth=1.5)
bar_labels(ax, b1, fmt="{:.0%}"); bar_labels(ax, b2, fmt="{:.0%}")
ax.set_xticks(x, groups, fontsize=10); ax.set_ylim(0, 1.0)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "25%", "50%", "75%", "100%"])
frame(ax)
legend(ax, [b1, b2], ["Ramp as built (0.349°)", "Ramp as intended (20°)"])
save(fig, "fig1_friction_share.png")

# Figure 2: oracle bit curve
fig, ax = plt.subplots(figsize=(7.2, 4.4))
fig.subplots_adjust(left=0.1, right=0.86, top=0.74, bottom=0.14)
titles(fig, "The material signal stopped saturating once the ramp was real",
       "Zero-training oracle: prediction gain over predict-the-mean when the true\n"
       "material is given at a fixed number of bits (shape always given; 512 scenes)")
xs = [0, 1, 2, 3, 4.4]
xl = ["0 bits", "1.6", "3.2", "6.3", "Exact\nmaterial"]
flat = [1.68, 2.02, 2.35, 2.43, 3.44]
real = [1.92, 2.38, 2.95, 3.91, 7.31]
for vals, col, name in ((flat, CONTEXT, "Ramp as built"), (real, BLUE, "Ramp as intended")):
    ax.plot(xs[:4], vals[:4], color=col, linewidth=2, marker="o", markersize=7,
            markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.plot(xs[3:], vals[3:], color=col, linewidth=1.5, linestyle=(0, (2, 3)))
    ax.plot(xs[4], vals[4], marker="o", markersize=7, color=col,
            markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.text(xs[4] + 0.15, vals[4], f"{name}\n{vals[4]:.2f}×", va="center", fontsize=9.5,
            color=INK2)
ax.set_xticks(xs, xl); ax.set_xlim(-0.3, 4.7); ax.set_ylim(0, 8)
ax.set_ylabel("Oracle gain (×)")
frame(ax)
save(fig, "fig2_bit_curve.png")

# Figure 3: co-movement without shared content
fig, ax = plt.subplots(figsize=(7.2, 3.8))
fig.subplots_adjust(left=0.47, right=0.93, top=0.7, bottom=0.12)
titles(fig, "The heads moved together with nothing in common",
       "Learned coherence, a world where render and behavior share no content\n"
       "(one box shape, materials only), versus a content-matched direction test")
labels = ["Joint training\n(heads share one latent)", "Separately trained control",
          "Content-matched direction test\n(highest value, any coupling)"]
vals = [0.434, 0.0001, 0.004]
y = np.arange(len(labels))[::-1]
bars = ax.barh(y, vals, 0.55, color=[BLUE, CONTEXT, CONTEXT], edgecolor=SURFACE, linewidth=1.5)
for yi, v in zip(y, vals):
    ax.text(v + 0.008, yi, f"+{v:.4f}" if v < 0.001 else f"+{v:.3f}", va="center",
            fontsize=10, color=INK2)
ax.set_yticks(y, labels, fontsize=10); ax.set_xlim(0, 0.5)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(AXIS); ax.tick_params(length=0)
ax.xaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
save(fig, "fig3_entanglement.png")

# Figure 4: keyframes in a drop
fig, ax = plt.subplots(figsize=(7.2, 4.6))
fig.subplots_adjust(left=0.09, right=0.98, top=0.74, bottom=0.2)
titles(fig, "The fall carries nothing; the first impact carries everything",
       "How well friction and restitution can be read back (inverse R²) from parts of a\n"
       "drop recording (512 boxes, contact at frame 11 of 64). Below 0 means no signal.")
wins = ["Free fall\n0 mm noise", "First 4 contact\nframes, 0 mm", "First 4 contact\nframes, 1 mm",
        "After contact\n5 mm", "Any 8 frames\n5 mm"]
fr = [0.00, 0.95, 0.09, -0.23, -0.15]
re = [0.00, 1.00, 0.99, 1.00, 0.99]
x = np.arange(len(wins)); w = 0.36
b1 = ax.bar(x - w / 2 - 0.01, fr, w, color=BLUE, edgecolor=SURFACE, linewidth=1.5)
b2 = ax.bar(x + w / 2 + 0.01, re, w, color=ORANGE, edgecolor=SURFACE, linewidth=1.5)
bar_labels(ax, b1); bar_labels(ax, b2)
ax.axhline(0, color=AXIS, linewidth=1)
ax.set_xticks(x, wins, fontsize=9.5); ax.set_ylim(-0.35, 1.15)
frame(ax)
legend(ax, [b1, b2], ["Friction", "Restitution"], loc="upper left")
save(fig, "fig5_keyframes.png")

# Figure 5: same data, three answers
fig, ax = plt.subplots(figsize=(7.2, 4.4))
fig.subplots_adjust(left=0.09, right=0.98, top=0.74, bottom=0.16)
titles(fig, "Same data, three answers",
       "Friction readable from the latent (R²) with and without a trajectory head, on\n"
       "byte-identical worlds. The head predicted one path for every object in all runs.")
tr = ["PyTorch", "MLX, default optimizer", "MLX, bias correction on"]
base = [0.24, 0.27, 0.28]
traj = [0.38, 0.24, 0.07]
x = np.arange(len(tr)); w = 0.34
b1 = ax.bar(x - w / 2 - 0.01, base, w, color=BLUE, edgecolor=SURFACE, linewidth=1.5)
b2 = ax.bar(x + w / 2 + 0.01, traj, w, color=ORANGE, edgecolor=SURFACE, linewidth=1.5)
bar_labels(ax, b1); bar_labels(ax, b2)
ax.set_xticks(x, tr, fontsize=10); ax.set_ylim(0, 0.5)
frame(ax)
legend(ax, [b1, b2], ["Normal training", "Plus trajectory head"], loc="upper right")
save(fig, "fig4_three_answers.png")

# Figure 6: the strike leak
fig, ax = plt.subplots(figsize=(7.2, 4.0))
fig.subplots_adjust(left=0.09, right=0.98, top=0.72, bottom=0.17)
titles(fig, "A better method made a leak pass the bar",
       "Restitution read from the ball-strike test (inverse R², 5 mm noise). The striking ball\n"
       "had inherited each box's bounciness for its own later floor bounces.")
st = ["As first measured\n(leak present)", "Better regression\n(leak present)", "Leak fixed"]
sv = [0.79, 0.88, 0.71]
bars = ax.bar(np.arange(3), sv, 0.5, color=[CONTEXT, CONTEXT, BLUE], edgecolor=SURFACE, linewidth=1.5)
bar_labels(ax, bars)
ax.axhline(0.8, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
ax.text(1.5, 0.815, "pass bar 0.8", ha="center", fontsize=9.5, color=INK2)
ax.set_xticks(np.arange(3), st, fontsize=10); ax.set_ylim(0, 1.0)
frame(ax)
save(fig, "fig6_strike_leak.png")

# Figure 7: the frozen code table
fams = ["drop", "ramp", "push", "strike", "scale", "tap", "kick", "drop onto ramp"]
axes_ = ["density", "friction", "restitution"]
M = np.array([[-0.01, -0.02, 0.98], [-0.01, 0.95, 0.18], [0.57, 0.61, 0.08],
              [0.86, 0.73, 0.69], [1.00, -0.01, -0.01], [-0.01, -0.01, 0.95],
              [-0.01, 0.99, 0.13], [-0.02, 0.82, 0.92]])
designated = {(3, 0), (4, 0), (1, 1), (6, 1), (0, 2), (5, 2)}
fig, ax = plt.subplots(figsize=(7.2, 5.6))
fig.subplots_adjust(left=0.26, right=0.95, top=0.74, bottom=0.04)
titles(fig, "Which interaction reveals which material property",
       "Frozen test battery (Nov 4): inverse R² from each interaction's recorded motion,\n"
       "5 mm noise, mean of 5 seeds × 512 boxes. Outlined cells decide each property.")
from matplotlib.colors import BoundaryNorm, ListedColormap
cmap = ListedColormap(["#f0efec"] + SEQ)
norm = BoundaryNorm([-1, 0.1, 0.3, 0.5, 0.65, 0.8, 0.9, 0.97, 1.01], cmap.N)
ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        v = M[i, j]
        dark = v >= 0.65
        ax.text(j, i, f"{v:.2f}".replace("-", "\u2212"), ha="center", va="center", fontsize=10.5,
                color="#ffffff" if dark else INK2, fontweight="bold" if v >= 0.8 else "normal")
        if (i, j) in designated:
            ax.add_patch(Rectangle((j - 0.47, i - 0.45), 0.94, 0.9, fill=False,
                                   edgecolor=INK, linewidth=2))
ax.set_xticks(range(3), axes_); ax.xaxis.tick_top()
ax.set_yticks(range(len(fams)), fams)
for s in ax.spines.values():
    s.set_visible(False)
ax.tick_params(length=0)
ax.set_xticks(np.arange(-0.5, 3), minor=True); ax.set_yticks(np.arange(-0.5, len(fams)), minor=True)
ax.grid(which="minor", color=SURFACE, linewidth=2.5)
ax.tick_params(which="minor", length=0)
save(fig, "fig7_code_table.png")
print("ok", sorted(os.listdir(OUT)))
