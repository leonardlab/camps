"""Supplementary Figure 9 - track length as a confound.

    PYTHONPATH=. python build_figS9.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import figstyle010 as F
import si_common as S
from panel_common import nocap_layout, render_nocaption

FIGSIZE = (10.4, 4.6)
TOP, BOTTOM = 0.652, 0.190
LEGY, LETY = 0.960, 0.995



def build_figS9(fname, dpi=300, legend_in=0.30):
    ok = S.load_fits()
    cells = S.load_cells()
    b, tld, drop = S.figS9_prep(cells, ok)

    figsize, top, bottom, legy, lety = nocap_layout(
            FIGSIZE, TOP, BOTTOM, LEGY, LETY, legend_in=legend_in)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.95, 1.25], wspace=0.30,
                          left=0.070, right=0.988, top=top, bottom=bottom)

    # (a) per-cell MSD by track length; y axis spans every plotted point
    ax = fig.add_subplot(gs[0, 0])
    bb = b.groupby("bin", observed=True)
    ylo = np.floor(b.y.min() * 2) / 2
    yhi = np.ceil(b.y.max() * 2) / 2
    ax.set_ylim(ylo - 0.15, yhi + 0.15); ax.set_xlim(-0.6, len(bb) - 0.4)
    labs = []
    for i, (k, g) in enumerate(bb):
        o = F.beeswarm_offsets(g.y.values, half_width=0.30, n_rows=40)
        ax.plot(i + o, g.y.values, "o", ms=1.1, mfc=F.CELL_DOT, mec="none",
                alpha=0.40, ls="none", rasterized=True)
        ax.plot([i - 0.34, i + 0.34], [g.y.median()] * 2, lw=1.8, color=F.MUTED,
                zorder=9, solid_capstyle="butt")
        lo, hi = int(k.left) + 1, int(k.right)
        labs.append(f">{int(k.left)}" if k.right >= 100 else
                    (f"{lo}" if lo == hi else f"{lo}\u2013{hi}"))
    ax.axhline(np.log10(F.THR_UM2), color=F.FAINT, ls=(0, (1, 2.4)), lw=.7)
    ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, rotation=45, ha="right", fontsize=6)
    F.log_ticks(ax, decades=(-1, 0, 1, 2, 3), labels=("0.1", "1", "10", "100", "1000"))
    ax.tick_params(labelsize=7)
    F.gp_axes(ax, ybounds=(ylo, yhi), xbounds=(-0.5, len(labs) - 0.5),
              ylabel="MSD (µm²)", title="a   shorter tracks read as faster")
    ax.set_xlabel("displacement pairs used per cell", fontsize=7.5, labelpad=2)

    # (b) per-well coupling
    ax = fig.add_subplot(gs[0, 1])
    for ds_, g in tld.groupby("ds"):
        ax.plot(g.med_pairs, g.med_y, S.ARMM[ds_], ms=4.0, mfc=S.ARMC[ds_],
                mec="black", mew=0.4, alpha=0.9, ls="none", label=ds_)
    ax.axhline(np.log10(F.THR_UM2), color=F.FAINT, ls=(0, (1, 2.4)), lw=.7)
    ax.set_xlim(2.2, 18.8); ax.set_ylim(-0.10, 2.62)
    F.log_ticks(ax, decades=(0, 1, 2), labels=("1", "10", "100"))
    ax.tick_params(labelsize=7)
    F.gp_axes(ax, ybounds=(0, 2.0), xbounds=(3, 18),
              ylabel="well median MSD (µm²)", title="b   treated wells hold longer tracks")
    ax.set_xticks([3, 6, 9, 12, 15, 18])
    ax.set_xlabel("well median displacement pairs", fontsize=7.5, labelpad=2)
    ax.legend(loc="upper right", fontsize=6, frameon=False, handletextpad=0.3, borderpad=0.2)
    rho = tld[["med_pairs", "med_y"]].corr(method="spearman").iloc[0, 1]
    ax.text(0.03, 0.04, f"Spearman ρ = {rho:+.2f}   ({len(tld)} wells)",
            transform=ax.transAxes, fontsize=6.4, color=F.MUTED)

    # (c) per-arm exclusion rate -- carried constants, see si_common.DROP_PCT
    ax = fig.add_subplot(gs[0, 2])
    keys = [("density", "untreated"), ("density", "PMA ctrl"),
            ("pma", "untreated"), ("pma", "PMA"),
            ("dox", "untreated"), ("dox", "dox"), ("dox", "PMA ctrl")]
    vals = [drop[k] for k in keys]
    x = np.arange(len(keys))
    ax.bar(x, vals, width=0.62, color=F.MUTED, edgecolor="black", lw=0.8)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 1.4, f"{v:.0f}", ha="center", va="bottom",
                fontsize=6.2, color=F.COUNT_INK)
    ax.set_ylim(0, 66); ax.set_xlim(-0.6, len(keys) - 0.4)
    ax.set_xticks(x)
    # One short arm label per bar; the experiment is named once beneath its group,
    # so the tick labels stay single-line and cannot collide.
    ARM_LBL = {"untreated": "untr.", "PMA": "+PMA", "PMA ctrl": "+PMA", "dox": "+dox"}
    ax.set_xticklabels([ARM_LBL[a] for _, a in keys], fontsize=6.4)
    for lo, hi, name in [(0, 1, "density"), (2, 3, "PMA"), (4, 6, "doxycycline")]:
        xc = (lo + hi) / 2
        ax.plot([lo - 0.30, hi + 0.30], [-7.2, -7.2], lw=0.7, color=F.MUTED,
                clip_on=False, solid_capstyle="butt")
        ax.text(xc, -9.4, name, ha="center", va="top", fontsize=6.4,
                color=F.COUNT_INK, clip_on=False)
    ax.tick_params(labelsize=7)
    F.gp_axes(ax, ybounds=(0, 60), xbounds=(-0.5, len(keys) - 0.5),
              ylabel="cells yielding no MSD (%)",
              title="c   untreated wells lose more cells")

    fig.savefig(fname, dpi=dpi)
    return fig


def overlaps(fig, dpi=None):
    """Count pairwise overlaps among visible Axes-owned text. 0 is the pass condition.

    Includes TICK LABELS. An earlier version of this checker walked only ax.texts,
    ax.title and the axis labels, and reported 0 for a panel whose x tick labels
    visibly collided. Same failure class as loc="left" titles living in the private
    ax._left_title: if the checker cannot see an artist, it cannot fail on it.
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
    f, L, gap = render_nocaption(build_figS9, _o.path.join(_OUT, "figS9_track_length_confound.svg"), dpi=300)
    print(f"overlaps: {overlaps(f)}  legend_in={L:.2f} gap={gap:+.3f} in")
    plt.close(f)
    f, _, _ = render_nocaption(build_figS9, _o.path.join(_OUT, "figS9_track_length_confound.png"), dpi=400)
    plt.close(f)
