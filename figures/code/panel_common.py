"""Data loading and layout helpers shared by the figure builders."""
import os, sys, numpy as np, pandas as pd
import matplotlib
matplotlib.rcParams["svg.fonttype"] = "none"
# macOS Helvetica.ttc intermittently raises "failed to load glyph" under repeated loads,
# so register Arial by FILE and put it first. See AESTHETICS.md.
import os as _os
from matplotlib import font_manager as _fm
for _p in ["/System/Library/Fonts/Supplemental/Arial.ttf",
           "/Library/Fonts/Arial.ttf"]:
    if _os.path.exists(_p):
        try:
            _fm.fontManager.addfont(_p)
            matplotlib.rcParams["font.family"] = "sans-serif"
            matplotlib.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
            break
        except Exception:
            pass
else:
    matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle010 as F
F.apply_style()

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
cells = pd.read_csv(os.path.join(DATA, "cells_all_datasets.csv.gz"))
fits  = pd.read_csv(os.path.join(DATA, "perwell_state_fits_all.csv"))

# --- canonical data prep
cells = cells[np.isfinite(cells.y)]
GRID = np.linspace(-1.0, 3.5, 320)

def repmap(wells):
    return {w: i + 1 for i, w in enumerate(sorted(wells))}

def fitmap(q):
    """replicate index -> that well's fit dict, with the per-well total width
    sqrt(sigma_bio^2 + median per-cell noise^2). state_assignment expects key `width`."""
    rm = repmap(q.well); out = {}
    for w in q.well:
        fr = fits[fits.well == w]
        if not len(fr):
            continue
        fr = fr.iloc[0]
        if not (np.isfinite(fr.mu_hi) and np.isfinite(fr.mu_lo)):
            continue
        sg = fr.sigma_bio if np.isfinite(fr.sigma_bio) else 0.373
        sn = np.nanmedian(cells[cells.well == w].sd_noise.values)
        out[rm[w]] = dict(mu_lo=fr.mu_lo, mu_hi=fr.mu_hi, w_hi=fr.w_hi,
                          width=float(np.sqrt(sg**2 + (0.0 if not np.isfinite(sn) else sn)**2)),
                          weak_high=bool(fr.weak_high), weak_low=bool(fr.weak_low),
                          n_hi_eff=float(fr.n_hi_eff), n_lo_eff=float(fr.n_lo_eff))
    return out

ACUTE = {"NoPMA": "0", "2ug": "2", "10ug": "10", "50ug": "50"}
CHRON = {"NoPMA": "0", ".08": ".08", ".4": ".4", "2": "2"}
pma = cells[cells.dataset == "pma"].copy(); pf = fits[fits.dataset == "pma"].copy()
for _d in (pma, pf):
    _d["dl"] = np.where(_d.treatment == "acute", _d.dose_label.map(ACUTE), _d.dose_label.map(CHRON))
DOSES = {"acute": ["0", "2", "10", "50"], "chronic": ["0", ".08", ".4", "2"]}
den = cells[cells.dataset == "density"].copy(); dens = den
dox = cells[cells.dataset == "dox"].copy(); dxf = fits[fits.dataset == "dox"].copy()
DOSE_O = {"untreated": "untr.", "20": "20", "100": "100", "500": "500", "PMA": "PMA"}
CORD = {c: [d for d in ["untreated", "20", "100", "500", "PMA"]
            if len(dxf[(dxf.construct.astype(str) == c) & (dxf.dose_label.astype(str) == d)])]
        for c in ["239", "240", "255", "257"]}
LBL = {"239": "239 (DRIVER)", "240": "240", "255": "255", "257": "257"}
SPACING = 1.90; WIDTH = 1.16; REFTAG = "\n(reference)"

# --- stars
def stars(p): return "****" if p<1e-4 else "***" if p<1e-3 else "**" if p<0.01 else "*" if p<0.05 else "ns"

# --- overlaps
def overlaps(fig):
    r=fig.canvas.get_renderer(); bs=[]
    for ax in fig.axes:
        for t in ax.texts:
            if t.get_text().strip(): bs.append(t.get_window_extent(r))
    n=0
    for i in range(len(bs)):
        for j in range(i+1,len(bs)):
            if bs[i].overlaps(bs[j]): n+=1
    return n

# --- LEG
def LEG(extras=()):
    h=[Line2D([],[],marker="o",ls="",mfc=F.STATE_HI,mec=F.CELL_EDGE,mew=0.28,ms=3.4,label="cell, high state"),
       Line2D([],[],marker="o",ls="",mfc=F.STATE_LO,mec=F.CELL_EDGE,mew=0.28,ms=3.4,label="cell, low state"),
       Line2D([],[],marker="o",ls="",mfc="none",mec=F.UNFIT_INK,mew=0.45,ms=3.4,label="cell, well unfit"),
       Line2D([],[],color=F.STATE_HI,lw=1.7,label="geo. mean of assigned cells"),
       Line2D([],[],color=F.POOL_INK,lw=1.9,label="geo. mean, all cells"),
       Patch(fc=F.STATE_HI,alpha=.55,label="fitted high state"),
       Patch(fc=F.STATE_LO,alpha=.55,label="fitted low state"),
       Line2D([],[],color="#B0B0B0",lw=0.9,label="fitted center (inside curve)"),
       Line2D([],[],color=F.FAINT,ls=(0,(1,2.4)),lw=.7,label="25 µm² threshold")]
    return h+list(extras)

# --- CAP_SHARED

# --- col_stats
def col_stats(q,sub):
    """Reference stats for a column: the AVERAGE of the per-replicate geometric means.

    Each replicate well contributes one value per quantity and they are averaged with equal weight,
    matching the well-is-the-unit-of-analysis rule. Pooling the reference column's cells instead
    would let a replicate with more cells dominate the baseline that every other column is measured
    against. A reference column's own replicates are then expressed relative to this cross-replicate
    average, so their scatter around 1.00x is visible rather than being collapsed to "Ref."."""
    g=sub.well.map(repmap(q.well)).values; fm=fitmap(q)
    order=sorted(set(g))
    acc={"overall":[],"hi":[],"lo":[]}
    for v in order:
        gm=F.state_geomeans(sub.y.values[g==v],fm.get(v))
        for k in acc:
            if np.isfinite(gm.get(k,np.nan)): acc[k].append(float(gm[k]))
    return {k:(float(np.mean(vs)) if vs else np.nan) for k,vs in acc.items()}

# --- fit_stats
def fit_stats(q):
    """Reference dict built from FITTED Gaussian centers (the alternative estimator)."""
    fm=fitmap(q); fl=[f for f in fm.values() if f is not None]
    def wm(key,wkey):
        vs=[(float(f[key]),max(float(f.get(wkey,1.0) or 0.0),0.0)) for f in fl
            if np.isfinite(f.get(key,np.nan))]
        if not vs: return np.nan
        tot=sum(w for _,w in vs)
        return float(sum(v*w for v,w in vs)/tot) if tot>0 else float(np.mean([v for v,_ in vs]))
    return dict(hi=wm("mu_hi","n_hi_eff"),lo=wm("mu_lo","n_lo_eff"))

# --- arith_ref
def arith_ref(q,sub):
    """Reference value for a column: mean of its per-well arithmetic means."""
    g=sub.well.map(repmap(q.well)).values
    return F.column_relative(sub.y.values,g)["mean"]

# --- draw_column
def draw_column(ax,xc,q,sub,width=0.86,ms=1.9,count_y=None,unit_gap=0.14,
                pct_y=(3.34,-0.98),ref=None,swarm_frac=0.52,
                pct_fs=5.2,count_fs=5.0,pct_dx=0.30,pct_inset=0.0,halo=0.82,
                rel_y=None,rel_fs=6.0):
    """One condition column. Geometric-mean bars and state proportions stay; the per-replicate
    relative-motility triplets are gone (Josh: the subpopulation-relative metric lacks a clear
    biological hypothesis). A single whole-population value sits above the column instead."""
    g=sub.well.map(repmap(q.well)).values
    fm=fitmap(q)
    _,ct=F.replicate_units(ax,xc,sub.y.values,g,fm,GRID,width=width,ms=ms,count_y=count_y,
                           unit_gap=unit_gap,swarm_frac=swarm_frac,count_fontsize=count_fs)
    F.replicate_percentages(ax,xc,pct_y[0],pct_y[1],fontsize=pct_fs,halo=halo,
                            dx_frac=pct_dx,inset=pct_inset)
    cr=F.column_relative(sub.y.values,g,ref=ref)
    if rel_y is not None:
        F.column_relative_label(ax,xc,cr,rel_y,fontsize=rel_fs,halo=halo)
    return cr,ct

def nocap_layout(figsize, top, bottom, legend_y, letter_y, legend_in=0.60):
    """Layout for a caption-free (repo) render.

    Keeps the AXES BLOCK at its exact physical size and removes only the caption
    band, so repo figures have identical panel proportions to the manuscript ones.
    `legend_in` is the height reserved for the legend band ABOVE the axes. Do not
    hand-tune it -- call render_nocaption(), which fits it by measurement.
    """
    PAD = 0.10      # gap between legend band and panel ink
    LETTER = 0.20   # headroom for the bold panel letter
    w, h = figsize
    axes_in = h * (top - bottom)
    bot_in = h * bottom
    new_h = axes_in + bot_in + PAD + legend_in + LETTER
    return ((w, new_h),
            (axes_in + bot_in) / new_h,                       # top
            bot_in / new_h,                                   # bottom
            (axes_in + bot_in + PAD + legend_in) / new_h,     # legend anchor
            0.998)                                            # panel letter


def _axes_text_top(fig, renderer):
    """Highest ink of any Text belonging to an Axes, in display units.

    Walks Axes CHILDREN rather than ax.texts/ax.title: a title placed with
    loc="left" lives in the private ax._left_title and appears in NEITHER of
    those, which is exactly the text that collides with a figure legend.
    """
    top = None
    def walk(obj):
        nonlocal top
        for ch in getattr(obj, "get_children", lambda: [])():
            s = getattr(ch, "get_text", lambda: None)()
            if s and s.strip() and ch.get_visible():
                y1 = ch.get_window_extent(renderer).y1
                top = y1 if top is None else max(top, y1)
            walk(ch)
    for ax in fig.axes:
        walk(ax)
    return top


def legend_clearance(fig):
    """Signed gap in inches between the figure legend's lowest ink and the
    highest Axes-owned text. Negative means the legend sits on the panel."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    if not fig.legends:
        return float("inf")
    lows = [t.get_window_extent(r).y0 for lg in fig.legends for t in lg.get_texts()]
    lows += [h.get_window_extent(r).y0 for lg in fig.legends
             for h in lg.legend_handles if hasattr(h, "get_window_extent")]
    top = _axes_text_top(fig, r)
    if top is None or not lows:
        return float("inf")
    return (min(lows) - top) / fig.dpi


def render_nocaption(build, fname, dpi=300, min_gap_in=0.05, verbose=False):
    """Render a caption-free figure, FITTING the legend band by measurement.

    Grows legend_in until the legend clears all panel text, so the gap is
    measured rather than guessed. Raises if no value in range works.
    """
    import matplotlib.pyplot as _plt
    for legend_in in [0.35, 0.50, 0.65, 0.80, 0.95, 1.10, 1.30, 1.50, 1.75, 2.00]:
        fig = build(fname, dpi=dpi, legend_in=legend_in)
        gap = legend_clearance(fig)
        if verbose:
            print(f"    legend_in={legend_in:.2f} -> gap {gap:+.3f} in")
        if gap >= min_gap_in:
            fig.savefig(fname, dpi=dpi)
            return fig, legend_in, gap
        _plt.close(fig)
    raise AssertionError(f"no legend_in in range clears {min_gap_in} in for {fname}")


def assert_no_legend_collision(fig, min_gap_in=0.02):
    gap = legend_clearance(fig)
    if gap < min_gap_in:
        raise AssertionError(f"legend clears panel ink by only {gap:+.3f} in")
    return gap
