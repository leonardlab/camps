"""Supplementary Figure 10 - per-well cell recovery as a confound.

    PYTHONPATH=. python build_figS10.py

Panel (c) uses neutral gray bars; red and blue are reserved throughout for the
high- and low-motility states.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import figstyle010 as F
import si_common as S
from panel_common import nocap_layout, render_nocaption

FIGSIZE = (10.0, 4.4)
TOP, BOTTOM = 0.660, 0.180
LEGY, LETY = 0.960, 0.995


READOUTS = [("w_hi", "high-state occupancy"),
            ("mu_hi", "high-state position"),
            ("med_y", "well median MSD")]


def build_figS10(fname, dpi=300, legend_in=0.30):
    ok = S.load_fits()
    cells = S.load_cells()
    dd, sp, wr = S.figS10_prep(cells, ok)

    figsize, top, bottom, legy, lety = nocap_layout(
            FIGSIZE, TOP, BOTTOM, LEGY, LETY, legend_in=legend_in)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.30, 1.0, 0.80], wspace=0.30,
                          left=0.062, right=0.988, top=top, bottom=bottom)

    # (a) recovery per well, grouped
    ax = fig.add_subplot(gs[0, 0])
    d = dd.sort_values(["construct", "dose_label", "run", "well"]).reset_index(drop=True)
    x = np.arange(len(d))
    ax.bar(x, d.n_cells, width=0.80, color=F.OBS_FILL, edgecolor="black", lw=0.35)
    ax.set_xlim(-1, len(d)); ax.set_ylim(0, d.n_cells.max() * 1.14)
    ax.set_xticks([])
    med = d.n_cells.median()
    ax.axhline(med, color=F.MUTED, ls=(0, (3, 3)), lw=0.8)
    ax.text(len(d) * 0.99, med * 1.10, f"median {med:.0f} cells", ha="right",
            va="bottom", fontsize=6.2, color=F.MUTED)
    ax.tick_params(labelsize=7)
    F.gp_axes(ax, ybounds=(0, d.n_cells.max()), xbounds=(0, len(d) - 1),
              ylabel="cells recovered per well",
              title="a   recovery varies several-fold between replicate wells")
    ax.set_xlabel(f"each bar is one well, ordered by construct \u00d7 dose \u00d7 run "
                  f"({len(d)} wells)", fontsize=7.2, labelpad=3)
    ax.text(0.015, 0.94,
            f"worst within-condition spread {wr.ratio.max():.0f}\u00d7 "
            f"(median {wr.ratio.median():.1f}\u00d7)",
            transform=ax.transAxes, fontsize=6.2, color=F.COUNT_INK, va="top")

    # (b) readouts vs recovery
    ax = fig.add_subplot(gs[0, 1])
    ax2 = ax
    xs = d.n_cells.values
    ax2.plot(xs, d.w_hi, "o", ms=3.4, mfc=F.MUTED, mec="black", mew=0.35, ls="none")
    ax2.set_ylim(-0.03, 1.03); ax2.set_xlim(0, xs.max() * 1.05)
    ax2.tick_params(labelsize=7)
    F.gp_axes(ax2, ybounds=(0, 1), xbounds=(0, xs.max()),
              ylabel="high-state occupancy (w_hi)",
              title="b   recovery tracks occupancy")
    ax2.set_xlabel("cells recovered per well", fontsize=7.5, labelpad=2)
    if np.isfinite(xs).all() and len(xs) > 2:
        z = np.polyfit(np.log10(xs), d.w_hi.values, 1)
        gx = np.linspace(xs.min(), xs.max(), 60)
        ax2.plot(gx, np.polyval(z, np.log10(gx)), lw=1.0, color=F.MUTED,
                 ls=(0, (4, 2)), zorder=1)
    ax2.text(0.03, 0.05, f"Spearman ρ = {sp['w_hi']:+.2f}", transform=ax2.transAxes,
             fontsize=6.4, color=F.MUTED)

    # (c) rho per readout -- neutral ink, sign read from the zero line
    ax = fig.add_subplot(gs[0, 2])
    vals = [sp[k] for k, _ in READOUTS]
    y = np.arange(len(READOUTS))[::-1]
    ax.barh(y, vals, height=0.52, color=F.MUTED, edgecolor="black", lw=0.8)
    ax.axvline(0, color="black", lw=0.7)
    for yi, v in zip(y, vals):
        ax.text(v + (0.03 if v >= 0 else -0.03), yi, f"{v:+.2f}",
                ha="left" if v >= 0 else "right", va="center",
                fontsize=6.4, color=F.COUNT_INK)
    ax.set_yticks(y); ax.set_yticklabels([lab for _, lab in READOUTS], fontsize=6.6)
    ax.set_xlim(-0.45, 0.55); ax.set_ylim(-0.6, len(READOUTS) - 0.4)
    ax.tick_params(labelsize=7)
    F.gp_axes(ax, ybounds=(-0.5, len(READOUTS) - 0.5), xbounds=(-0.4, 0.5),
              title="c   occupancy, not position")
    ax.set_xlabel("Spearman ρ with cells per well", fontsize=7.2, labelpad=2)

    fig.savefig(fname, dpi=dpi)
    return fig


def overlaps(fig, dpi=None):
    """Count pairwise overlaps among visible Axes-owned text. 0 is the pass condition.

    Includes tick labels and loc="left" titles (ax._left_title) -- both are artists a
    naive checker misses, and both have shipped visible collisions in this project.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    boxes = []
    for ax in fig.axes:
        items = list(ax.texts) + [ax.title, ax.xaxis.label, ax.yaxis.label]
        items += ax.get_xticklabels() + ax.get_yticklabels()
        lt = getattr(ax, "_left_title", None)
        if lt is not None:
            items.append(lt)
        for t_ in items:
            if t_ is not None and t_.get_text().strip() and t_.get_visible():
                boxes.append((t_.get_text()[:20], t_.get_window_extent(r)))
    n = 0
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes[i][1].overlaps(boxes[j][1]):
                n += 1
    return n


if __name__ == "__main__":
    import os as _o
    _OUT = _o.environ.get("CAMPS_FIG_OUT",
                          _o.path.join(_o.path.dirname(_o.path.abspath(__file__)),
                                       "..", "rebuilt"))
    _o.makedirs(_OUT, exist_ok=True)
    f, L, gap = render_nocaption(build_figS10, _o.path.join(_OUT, "figS10_recovery_confound_dox.svg"), dpi=300)
    print(f"overlaps: {overlaps(f)}  legend_in={L:.2f} gap={gap:+.3f} in")
    plt.close(f)
    f, _, _ = render_nocaption(build_figS10, _o.path.join(_OUT, "figS10_recovery_confound_dox.png"), dpi=400)
    plt.close(f)
