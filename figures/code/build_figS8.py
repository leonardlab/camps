"""Supplementary Figure 8 - absolute state levels across the three experiments.

    PYTHONPATH=. python build_figS8.py

Open markers denote a state fitted on fewer than ten effective cells; those
positions are not quotable. Phase and GFP blocks use different MSD lags (540 s
and 600 s) and are deliberately not drawn on a shared axis.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import figstyle010 as F
import si_common as S
from panel_common import nocap_layout, render_nocaption, legend_clearance

FIGSIZE = (7.2, 10.0)
TOP, BOTTOM = 0.798, 0.086
LEGY, LETY = 0.876, 0.997
YL = (-0.55, 3.05)



def build_figS8(fname, dpi=300, legend_in=0.55):
    ok = S.load_fits()
    agg = S.figS8_agg(ok)

    figsize, top, bottom, legy, lety = nocap_layout(
            FIGSIZE, TOP, BOTTOM, LEGY, LETY, legend_in=legend_in)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(4, 4, hspace=1.00, wspace=0.10,
                          left=0.115, right=0.985, top=top, bottom=bottom)
    for bi, (btitle, groups, chan, xlab, lagnote) in enumerate(S.BLOCKS):
        axes = []
        for gi, g in enumerate(groups):
            ax = fig.add_subplot(gs[bi, gi]); axes.append(ax)
            order = S.ORDER[g]
            d = agg[agg.group == g].set_index("cond").reindex(order)
            x = np.arange(len(order))
            ax.vlines(x, d.mu_lo, d.mu_hi, color="#C8C8C8", lw=0.9, zorder=1)
            for xi in range(len(order)):
                a, b = d.hi_min.iloc[xi], d.hi_max.iloc[xi]
                if np.isfinite(a):
                    ax.vlines(xi, a, b, color=S.HI, lw=2.6, alpha=.30, zorder=2)
                a, b = d.lo_min.iloc[xi], d.lo_max.iloc[xi]
                if np.isfinite(a):
                    ax.vlines(xi, a, b, color=S.LO, lw=2.6, alpha=.30, zorder=2)
            wk = d.hi_weak.fillna(False).values.astype(bool)
            wl = d.lo_weak.fillna(False).values.astype(bool)
            ax.plot(x[~wk], d.mu_hi.values[~wk], "o", color=S.HI, ms=4.6,
                    zorder=4, clip_on=False)
            ax.plot(x[wk], d.mu_hi.values[wk], "o", mfc="white", mec=S.HI,
                    mew=1.1, ms=4.6, zorder=4, clip_on=False)
            ax.plot(x[~wl], d.mu_lo.values[~wl], "o", color=S.LO, ms=4.6,
                    zorder=4, clip_on=False)
            ax.plot(x[wl], d.mu_lo.values[wl], "o", mfc="white", mec=S.LO,
                    mew=1.1, ms=4.6, zorder=4, clip_on=False)
            ax.plot(x, d.mean_y, "_", color=S.POOL, ms=9, mew=1.5, zorder=5)
            ref = d.mu_hi.iloc[0]
            if np.isfinite(ref) and not wk[0]:
                ax.axhline(ref, color=S.HI, ls=(0, (3, 3)), lw=.7, alpha=.5, zorder=0)
            ax.set_xticks(x); ax.set_xticklabels(order)
            ax.set_xlim(-0.55, len(order) - 0.45); ax.set_ylim(*YL)
            ax.set_yticks([0, 1, 2, 3])
            ax.set_yticklabels(["1", "10", "100", "1000"] if gi == 0 else [])
            ax.set_title(S.TITLE[g], fontsize=7, pad=3, loc="left")
            ax.tick_params(labelsize=6)
            ax.set_xlabel(xlab, fontsize=6.6, labelpad=2)
        axes[0].set_ylabel("MSD (µm²)", fontsize=7.5)
        axes[0].text(0.0, 1.30, f"{'abcd'[bi]}   {btitle}", transform=axes[0].transAxes,
                     fontsize=8, fontweight="bold", ha="left", va="bottom")
        axes[-1].text(1.0, 1.30, lagnote, transform=axes[-1].transAxes,
                      fontsize=6, color=F.MUTED, ha="right", va="bottom", style="italic")

    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], marker="o", color=S.HI, ls="none", ms=4.6, label="high-motility state mean"),
        Line2D([], [], marker="o", color=S.LO, ls="none", ms=4.6, label="low-motility state mean"),
        Line2D([], [], marker="o", mfc="white", mec="black", mew=1.1, ls="none", ms=4.6,
               label=f"state on <{int(S.MINEFF)} effective cells (not quotable)"),
        Line2D([], [], marker="_", color=S.POOL, ls="none", ms=9, mew=1.5, label="all-cell mean"),
        Line2D([], [], color=S.HI, lw=2.6, alpha=.30, label="well-to-well range"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.115, legy),
               ncol=3, frameon=False, fontsize=6.4, handletextpad=0.5,
               columnspacing=1.4, labelspacing=0.5)

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
    f, L, gap = render_nocaption(build_figS8, _o.path.join(_OUT, "figS8_msd_absolute_states.svg"), dpi=300)
    print(f"overlaps: {overlaps(f)}  legend_in={L:.2f} gap={gap:+.3f} in")
    plt.close(f)
    f, _, _ = render_nocaption(build_figS8, _o.path.join(_OUT, "figS8_msd_absolute_states.png"), dpi=400)
    plt.close(f)
