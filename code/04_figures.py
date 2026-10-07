"""Step 4. Figures (light-blue glass style)."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch
import mne
from hypyp import viz
from config import *

mne.set_log_level("ERROR")
res = json.load(open(ROOT / "derivatives" / "results.json"))
G = np.load(SYNC / "group_maps.npz")
bands, ch = list(res["bands"]), res["ch_names"]
N = len(res["pairs"])
LAB = {"plv": "Phase-locking value", "coh": "Coherence", "ccorr": "Circular correlation",
       "imaginary_coh": "Imaginary coherence"}
BLAB = [f"{b}\n{lo}-{hi}" for b, (lo, hi) in res["bands"].items()]

INK, MUTED, GRID = "#0b2a4a", "#4f6f8f", "#9cc4e4"
REAL, SURR, HOT = "#0a6fd8", "#7fa3c4", "#d6456f"
DIV = LinearSegmentedColormap.from_list("ice", ["#0a4fb0", "#5aa9ea", "#e9f1f7", "#f29ab4", "#c2255c"])
plt.rcParams.update({"font.family": "Inter", "font.size": 10.5, "text.color": INK, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": "none",
                     "axes.titleweight": "semibold", "axes.titlesize": 11.5, "axes.titlelocation": "left",
                     "savefig.dpi": 200})


def canvas(w, h, title, sub):
    """Figure with an icy gradient ground; panels are frosted-glass cards."""
    fig = plt.figure(figsize=(w, h))
    bg = fig.add_axes([0, 0, 1, 1]); bg.axis("off")
    yy, xx = np.mgrid[0:1:200j, 0:1:200j]
    glow = 0.55 * np.exp(-((xx - .15) ** 2 + (yy - .9) ** 2) / .12) + 0.45 * np.exp(-((xx - .9) ** 2 + (yy - .1) ** 2) / .2)
    cm = LinearSegmentedColormap.from_list("bg", ["#eef8ff", "#d3ebfd", "#b9defb"])
    bg.imshow(glow, cmap=cm, aspect="auto", extent=[0, 1, 0, 1], origin="lower", zorder=0)
    fig.text(.04, 1 - .34 / h, title.upper(), fontsize=13.5, fontweight="bold", color=INK, va="top")
    fig.text(.04, 1 - .68 / h, sub, fontsize=10.5, color=MUTED, va="top")
    return fig


def glass(fig, rect, pad=.012):
    """Frosted card behind an axes rect [l,b,w,h] (figure fractions); returns the axes."""
    l, b, w, h = rect
    fw, fh = fig.get_size_inches()
    L, Rr, B, T = .62 / fw, .14 / fw, .52 / fh, .42 / fh          # room for tick labels and title (inches)
    fig.axes[0].add_patch(FancyBboxPatch((l - L, b - B), w + L + Rr, h + B + T, transform=fig.transFigure,
                                         boxstyle="round,pad=0,rounding_size=0.014", mutation_aspect=fw / fh,
                                         fc=(1, 1, 1, .46), ec=(1, 1, 1, .95), lw=1.3, zorder=1))
    ax = fig.add_axes(rect, facecolor="none")
    ax.grid(axis="y", color=GRID, lw=.6, alpha=.6); ax.set_axisbelow(True)
    ax.tick_params(length=0)
    return ax


def row(n, top, bottom=.16, left=.075, right=.025, gap=.075):
    w = (1 - left - right - gap * (n - 1)) / n
    return [[left + i * (w + gap), bottom, w, top - bottom] for i in range(n)]


# ---------- Fig 1: raw synchrony, real vs shuffled partners ----------
fig = canvas(11, 4.6, "Real partners vs shuffled partners",
             f"Mean inter-brain synchrony over all 27 x 27 electrode links, {N} dyads. Bars: group mean. Dots: individual dyads.")
x = np.arange(4)
for rect, m in zip(row(4, .72), MODES):
    ax = glass(fig, rect)
    rows = [r for r in res["real_vs_surrogate"] if r["measure"] == m]
    for k, (key, col, lab) in enumerate([("per_dyad_real", REAL, "Real partner"), ("per_dyad_surr", SURR, "Shuffled trials")]):
        v = np.array([r[key] for r in rows])
        ax.bar(x + (k - .5) * .38, v.mean(1), .34, color=col, alpha=.88, label=lab, zorder=2)
        for i in range(4):
            ax.scatter(np.full(N, x[i] + (k - .5) * .38), v[i], s=7, color=INK, alpha=.35, lw=0, zorder=3)
    ax.set_xticks(x, BLAB, fontsize=8.5); ax.set_xlabel("frequency band (Hz)", fontsize=8.5); ax.set_title(LAB[m]); ax.set_ylim(0)
    if m == MODES[0]:
        ax.set_ylabel("synchrony (0-1)")
        ax.legend(frameon=False, fontsize=9, loc="upper right", handlelength=1, borderaxespad=0)
fig.savefig(FIG / "fig1_real_vs_shuffled.png"); plt.close(fig)

# ---------- Fig 2: partner-specific synchrony per dyad ----------
fig = canvas(11, 4.8, "Partner-specific synchrony",
             "Real minus shuffled, as a percent of the shuffled level. Each dot is one dyad; diamond and bar: group mean and 95% CI.")
lim = 0
for rect, m in zip(row(4, .72), MODES):
    ax = glass(fig, rect)
    for i, r in enumerate([r for r in res["real_vs_surrogate"] if r["measure"] == m]):
        a, s = np.array(r["per_dyad_real"]), np.array(r["per_dyad_surr"])
        d = 100 * (a - s) / s
        jit = np.random.default_rng(i).uniform(-.16, .16, N)
        ax.scatter(i + jit, d, s=26, color=REAL, alpha=.55, lw=.8, ec="white", zorder=3)
        ci = 1.96 * d.std(ddof=1) / np.sqrt(N)
        ax.plot([i + .3, i + .3], [d.mean() - ci, d.mean() + ci], color=INK, lw=1.6, zorder=4)
        ax.scatter([i + .3], [d.mean()], marker="D", s=30, color=INK, zorder=5)
        ax.text(i, 1.0, f"p={r['p']:.2f}" if r["p"] >= .01 else f"p={r['p']:.3f}", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=8, color=HOT if r["q_fdr"] < .05 else MUTED)
        lim = max(lim, np.abs(d).max())
    ax.axhline(0, color=INK, lw=1, zorder=2)
    ax.set_xticks(x, BLAB, fontsize=8.5); ax.set_xlabel("frequency band (Hz)", fontsize=8.5); ax.set_title(LAB[m]); ax.set_xlim(-.5, 3.6)
    if m == MODES[0]:
        ax.set_ylabel("real vs shuffled (%)")
for ax in fig.axes[1:]:
    ax.set_ylim(-lim * 1.15, lim * 1.35)
fig.savefig(FIG / "fig2_partner_specific.png"); plt.close(fig)

# ---------- Fig 3: electrode level (strongest band for PLV) ----------
best = max((r for r in res["electrode_level"] if r["measure"] in ("plv", "coh")), key=lambda r: r["max_abs_T"])
key = f"{best['measure']}_{best['band']}"
T, P = G[f"{key}__T"], G[f"{key}__p"]
epo = mne.read_epochs(EPO / f"{res['pairs'][0]}_L-epo.fif")
order = np.argsort([-epo.info["chs"][i]["loc"][1] for i in range(len(ch))])        # front -> back
fig = canvas(11, 5.6, "Where on the scalp?",
             f"{LAB[best['measure']]}, {best['band']} band: real minus shuffled, t-value across {N} dyads for every electrode link. "
             f"Links surviving max-T permutation correction: {best['n_links_p05_maxT']}.")
ax = glass(fig, [.09, .15, .34, .60]); ax.grid(False)
vmax = np.abs(T).max()
im = ax.imshow(T[np.ix_(order, order)], cmap=DIV, vmin=-vmax, vmax=vmax, interpolation="nearest")
ax.set_xticks(range(len(ch)), [ch[i] for i in order], rotation=90, fontsize=6.5)
ax.set_yticks(range(len(ch)), [ch[i] for i in order], fontsize=6.5)
ax.set_xlabel("participant R electrode (front to back)"); ax.set_ylabel("participant L electrode")
cb = fig.colorbar(im, ax=ax, fraction=.04, pad=.03); cb.outline.set_visible(False); cb.set_label("t (real - shuffled)", color=MUTED)
ax2 = glass(fig, [.57, .15, .39, .60]); ax2.grid(False); ax2.axis("off"); ax2.set_aspect(1)
plt.sca(ax2)
viz.plot_2d_topomap_inter(ax2)
viz.plot_sensors_2d_inter(epo, epo, lab=False)
thr = np.sort(np.abs(T).ravel())[-25]
C = np.where(np.abs(T) >= thr, T / vmax, 0)
viz.plot_links_2d_inter(epo, epo, C=C, threshold=1e-6, steps=12)
ax2.set_title("25 strongest links (uncorrected), drawn with HyPyP", fontsize=10.5)
fig.savefig(FIG / "fig3_scalp.png"); plt.close(fig)

# ---------- Fig 4: duets vs constant pitch ----------
fig = canvas(11, 4.8, "Musical duets vs constant-pitch sequences",
             "Difference in partner-specific synchrony (duet minus constant pitch), per dyad. Diamond and bar: group mean and 95% CI.")
lim = 0
for rect, m in zip(row(4, .72), MODES):
    ax = glass(fig, rect)
    for i, r in enumerate([r for r in res["duet_vs_pitch"] if r["measure"] == m]):
        d = np.array(r["per_dyad"])
        jit = np.random.default_rng(i).uniform(-.16, .16, N)
        ax.scatter(i + jit, d, s=26, color=REAL, alpha=.55, lw=.8, ec="white", zorder=3)
        ci = 1.96 * d.std(ddof=1) / np.sqrt(N)
        ax.plot([i + .3, i + .3], [d.mean() - ci, d.mean() + ci], color=INK, lw=1.6, zorder=4)
        ax.scatter([i + .3], [d.mean()], marker="D", s=30, color=INK, zorder=5)
        ax.text(i, 1.0, f"p={r['p']:.2f}" if r["p"] >= .01 else f"p={r['p']:.3f}", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=8, color=HOT if r["q_fdr"] < .05 else MUTED)
        lim = max(lim, np.abs(d).max())
    ax.axhline(0, color=INK, lw=1, zorder=2)
    ax.set_xticks(x, BLAB, fontsize=8.5); ax.set_xlabel("frequency band (Hz)", fontsize=8.5); ax.set_title(LAB[m]); ax.set_xlim(-.5, 3.6)
    if m == MODES[0]:
        ax.set_ylabel("duet - constant pitch")
for ax in fig.axes[1:]:
    ax.set_ylim(-lim * 1.15, lim * 1.35)
fig.savefig(FIG / "fig4_duet_vs_pitch.png"); plt.close(fig)
print("figures written", best)
