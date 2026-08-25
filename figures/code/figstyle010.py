"""Shared plotting grammar: colors, axes, markers, and annotation helpers."""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

# ---------------------------------------------------------------- color + constants
STATE_HI  = "#C0392B"   # high-motility state  (never recolor: C04.008 grammar)
STATE_LO  = "#2E6DA4"   # low-motility state — BLUE. Restored once replicate identity moved to
                        # position (one unit per replicate) instead of color: color is now free
                        # to mean state everywhere, including the beeswarm fill.
COUNT_INK = "#3A3A3A"   # per-replicate cell counts: dark gray text, not a color-coded label
UNFIT_INK = "#7A8088"   # cells of a well whose mixture did not converge: hollow gray, unclassified
CELL_DOT  = "#23282D"   # individual cells: near-black, they are the primary data
OBS_FILL  = "#C9CCCF"   # observed-distribution violin
POOL_INK  = "#1A1A1A"
STATE_INK = "#FFFFFF"   # center line drawn INSIDE a fitted curve: reads as part of the curve
MUTED     = "#5A5A5A"
FAINT     = "#8A8A8A"
CTRL_BAND = "#F2EADB"   # positive-control shading
THR_UM2   = 25.0        # historical motility threshold

def apply_style(base=8.0):
    """GraphPad-compatible defaults: white ground, two spines, outward ticks, no grid."""
    mpl.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "savefig.bbox": None,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],  # Helvetica.ttc raises "failed to load glyph" on macOS
        "font.size": base, "axes.labelsize": base, "axes.titlesize": base + 1,
        "xtick.labelsize": base - 1, "ytick.labelsize": base - 1,
        "legend.fontsize": base - 1.4,
        "axes.linewidth": 0.9, "axes.edgecolor": "#000000",
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3.4, "ytick.major.size": 3.4,
        "xtick.major.width": 0.9, "ytick.major.width": 0.9,
        "xtick.color": "#000000", "ytick.color": "#000000",
        "axes.grid": False, "legend.frameon": False,
        "svg.fonttype": "none",          # live text for Illustrator
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "lines.solid_capstyle": "round",
    })

def gp_axes(ax, ybounds=None, xbounds=None, ylabel=None, title=None):
    """GraphPad look: spines bounded to the data range, ticks outward, no box."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if ybounds is not None:
        ax.spines["left"].set_bounds(*ybounds)
    if xbounds is not None:
        ax.spines["bottom"].set_bounds(*xbounds)
    if ylabel: ax.set_ylabel(ylabel)
    if title:  ax.set_title(title, loc="left", pad=4)
    return ax

# ---------------------------------------------------------------- beeswarm
def marker_pitch(ax, ms, pad=1.12):
    """Marker diameter converted into (dx, dy) DATA units for this Axes.

    Beeswarm packing has to be done in data units, but marker size is specified in points. If
    the two are set independently the dots overlap at some panel widths and gap at others. This
    measures the actual points-per-data-unit of the live Axes so the packing pitch is exactly
    one marker diameter (times `pad`), which is what makes a swarm tight AND non-overlapping.

    Call it after the Axes limits are final.
    """
    fig = ax.figure
    fig.canvas.draw_idle()
    bb = ax.get_window_extent()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    dpi = fig.dpi
    d_px = ms * dpi / 72.0 * pad            # marker diameter in pixels (ms is in points)
    dx = d_px * (x1 - x0) / max(bb.width, 1e-9)
    dy = d_px * (y1 - y0) / max(bb.height, 1e-9)
    return dx, dy

def beeswarm_offsets(y, half_width=0.34, point_dy=None, n_rows=40, max_cols=None, step=None):
    """Deterministic horizontal dodge — GraphPad 'scatter dot' / bubble-map placement.

    y is binned into horizontal rows of height point_dy; within a row, points are placed
    symmetrically outward from the center (0, +s, -s, +2s, -2s, ...) so each row reads as a
    centered bar of dots whose width tracks local density. No RNG: identical on every rebuild.

    half_width is the maximum |offset| a point may take. A row denser than max_cols columns
    tightens its own step so the row still fits rather than spilling into the neighbour.
    """
    y = np.asarray(y, float)
    ok = np.isfinite(y)
    off = np.full(y.shape, np.nan)
    yv = y[ok]
    if len(yv) == 0:
        return off
    lo, hi = np.nanmin(yv), np.nanmax(yv)
    span = max(hi - lo, 1e-9)
    if point_dy is None:
        point_dy = span / float(n_rows)
    rows = np.floor((yv - lo) / point_dy).astype(int)
    # step = one marker diameter in x-data units, so a row of k points spans (k-1)*step centered
    # on 0: row WIDTH tracks local density and neighbours touch without overlapping. Pass `step`
    # (and the matching point_dy) from marker_pitch() to make packing exact at any panel width;
    # the fallback just divides the band into a fixed column count.
    if step is None:
        if max_cols is None:
            max_cols = 13
        step = half_width / max(1.0, (max_cols - 1) / 2.0)
    res = np.zeros(len(yv))
    for r in np.unique(rows):
        m = np.where(rows == r)[0]
        k = len(m)
        # centered symmetric ladder: 0, +1, -1, +2, -2, ... in units of `step`
        ladder = np.empty(k)
        for j in range(k):
            t = (j + 1) // 2
            ladder[j] = (t if j % 2 == 1 else -t)
        s = step
        if k > 1 and abs(ladder).max() * s > half_width:      # tighten a dense row to fit
            s = half_width / abs(ladder).max()
        # order within the row by y so the swarm looks settled, not shuffled
        res[m[np.argsort(yv[m], kind="stable")]] = ladder * s
    off[ok] = res
    return off

def draw_cells(ax, xc, y, half_width=0.34, ms=1.9, alpha=0.85, color=CELL_DOT,
               zorder=8, point_dy=None, marker="o", n_rows=40, max_cols=None):
    """The cell-level data, beeswarm-dodged and drawn ON TOP of the distributions."""
    y = np.asarray(y, float); y = y[np.isfinite(y)]
    if not len(y):
        return
    off = beeswarm_offsets(y, half_width=half_width, point_dy=point_dy,
                           n_rows=n_rows, max_cols=max_cols)
    ax.plot(xc + off, y, marker, ms=ms, mfc=color, mec="none",
            alpha=alpha, zorder=zorder, ls="none", rasterized=True)

# ---------------------------------------------------------------- distributions
# Replicate encoding. Each replicate gets its OWN LANE (its own horizontal band inside the
# column) plus its own color, so replicate structure reads at a glance and no two replicates
# are ever interleaved. Colors are a CMY-family set: distinct in hue, similar in weight, and
# none of them collides with the red/blue reserved for the fitted states.
REP_MARKERS = ("o", "^", "s", "D", "v")
REP_TONES   = ("#1B1F23", "#4A5158", "#767E86", "#9AA2AA", "#B6BDC4")
# Cyan / magenta / pastel green. NO YELLOW: it is unreadable as text and near-invisible as a
# small marker on white, and these same colors are reused for the per-replicate count labels.
# Cyan / light orange / pastel green. Light orange rather than magenta: magenta read too close to
# the STATE_HI red. With STATE_LO now gray, cyan is unambiguous too.
REP_COLORS  = ("#22B8CF", "#F5A24B", "#5FBF7F", "#7C6BD8", "#C0679B")
CELL_EDGE   = "#000000"   # every cell dot carries a thin black outline: it reads at print size
                          # against both the pale mixture fills and the white ground

def lane_beeswarm(ax, xc, y, groups, side="left", half_width=0.34, ms=2.0, alpha=0.92,
                  n_rows=40, zorder=8, gap=0.010, lane_gap=0.10, colors=REP_COLORS,
                  markers=REP_MARKERS, label_prefix="rep ", counts=True, count_y=None,
                  count_fontsize=5.2):
    """One-sided beeswarm split into one LANE PER REPLICATE, stacked side by side.

    Cells are never mixed between replicates: lane 1 occupies the band nearest the axis, lane 2
    the next band out, and so on. Each lane is swarm-dodged within its own width so lane width
    still reads as local density. Color and marker both track the replicate.

    Returns (handles, per_lane_counts) — counts is [(x_center, n), ...] so the caller can place
    its own labels if `counts=False`.
    """
    y = np.asarray(y, float)
    ok = np.isfinite(y)
    y = y[ok]
    g = np.asarray(groups, dtype=object)[ok]
    if not len(y):
        return [], []
    gvals = sorted(set(g.tolist()))
    k = len(gvals)
    sgn = -1.0 if side == "left" else 1.0
    # split half_width into k lanes with a small gutter between them
    total_gutter = lane_gap * half_width * max(0, k - 1)
    lane_w = max((half_width - total_gutter) / max(k, 1), 1e-6)
    handles, counts_out = [], []
    for i, gv in enumerate(gvals):
        m = g == gv
        inner = gap + i * (lane_w + lane_gap * half_width)
        off = beeswarm_offsets(y[m], half_width=lane_w / 2.0, n_rows=n_rows)
        center = inner + lane_w / 2.0
        x = xc + sgn * (center + off)
        col = colors[i % len(colors)]
        mk = markers[i % len(markers)]
        ax.plot(x, y[m], mk, ms=ms, mfc=col, mec="none", alpha=alpha, ls="none",
                zorder=zorder + i * 0.01, rasterized=True)
        handles.append(Line2D([], [], marker=mk, ls="", mfc=col, mec="none", ms=ms + 0.9,
                              label=f"{label_prefix}{gv}"))
        counts_out.append((xc + sgn * center, int(m.sum())))
        if counts and count_y is not None:
            ax.text(xc + sgn * center, count_y, f"n={int(m.sum())}", ha="center", va="center",
                    fontsize=count_fontsize, color=col)
    return handles, counts_out

def half_beeswarm(ax, xc, y, groups=None, side="left", half_width=0.34, ms=2.0, alpha=0.9,
                  n_rows=40, max_cols=None, zorder=8, gap=0.012, markers=REP_MARKERS,
                  tones=REP_TONES, single_color=CELL_DOT, label_prefix="rep "):
    """One-sided beeswarm: the cells ARE the distribution on this side of the axis.

    Offsets are computed on the FULL sample (so the swarm's width still reads as local density),
    then folded onto one side. If `groups` is given (e.g. replicate id per cell), each group gets
    its own marker shape and tone so replicate structure is visible inside the swarm rather than
    averaged away. Returns the legend handles for the groups actually drawn.
    """
    y = np.asarray(y, float)
    ok = np.isfinite(y)
    y = y[ok]
    if not len(y):
        return []
    g = None if groups is None else np.asarray(groups, dtype=object)[ok]
    off = beeswarm_offsets(y, half_width=half_width, n_rows=n_rows, max_cols=max_cols)
    sgn = -1.0 if side == "left" else 1.0
    x = xc + sgn * (np.abs(off) + gap)          # fold to one side, small gap off the spine
    handles = []
    if g is None:
        ax.plot(x, y, "o", ms=ms, mfc=single_color, mec="none", alpha=alpha,
                ls="none", zorder=zorder, rasterized=True)
        return handles
    for i, gv in enumerate(sorted(set(g.tolist()))):
        m = g == gv
        mk = markers[i % len(markers)]
        tn = tones[i % len(tones)]
        ax.plot(x[m], y[m], mk, ms=ms, mfc=tn, mec="none", alpha=alpha,
                ls="none", zorder=zorder + i * 0.01, rasterized=True)
        handles.append(Line2D([], [], marker=mk, ls="", mfc=tn, mec="none", ms=ms + 0.8,
                              label=f"{label_prefix}{gv}"))
    return handles

def replicate_units(ax, xc, y, groups, fits_by_group, grid, width=0.44, ms=1.8, alpha=0.92,
                    n_rows=40, swarm_frac=0.52, gap=0.008, unit_gap=0.16, colors=REP_COLORS,
                    markers=REP_MARKERS, label_prefix="rep ", count_y=None, count_fontsize=5.0,
                    zorder=8, edge_lw=0.28):
    """One self-contained unit PER REPLICATE: its own beeswarm on the left, its own fitted
    mixture immediately on the right. Units sit side by side within the column.

    This is the unambiguous pairing — every fit is visually attached to the cells it was fitted
    to, so a reader can never mis-assign a mixture curve to the wrong replicate. `fits_by_group`
    maps a group value to that well's fit dict (mu_lo, mu_hi, w_hi, width, weak_*).

    Returns (handles, unit_centers).
    """
    y = np.asarray(y, float)
    ok = np.isfinite(y)
    y = y[ok]
    g = np.asarray(groups, dtype=object)[ok]
    if not len(y):
        return [], []
    gvals = sorted(set(g.tolist()))
    k = len(gvals)
    total_gap = unit_gap * width * max(0, k - 1)
    uw = max((width - total_gap) / max(k, 1), 1e-6)      # width of one replicate unit
    sw = uw * swarm_frac                                  # swarm half-band
    fw = uw * (1.0 - swarm_frac)                          # fit half-band
    pitch_dx, pitch_dy = marker_pitch(ax, ms)
    handles, centers = [], []
    for i, gv in enumerate(gvals):
        m = g == gv
        left = xc - width / 2.0 + i * (uw + unit_gap * width)
        axis_x = left + sw                                # the unit's own local baseline
        # Marker SHAPE and COLOR no longer encode the replicate — POSITION does (one unit each),
        # which frees the fill to carry state membership instead. A single circle shape keeps the
        # swarm visually even.
        mk = markers[0] if markers else "o"
        f = fits_by_group.get(gv)
        # ONE-SIDED swarm: offsets are folded to a single side so the swarm grows away from the
        # unit's baseline, mirroring the fitted mixture on the other side of the same baseline.
        # Pitch is measured from the live Axes so dots pack tightly without ever overlapping.
        off = np.abs(beeswarm_offsets(y[m], half_width=sw, n_rows=n_rows,
                                     point_dy=pitch_dy, step=pitch_dx))
        xs, ys = axis_x - gap - off, y[m]
        # Cell fill = the state THIS well's own fit assigns the cell to (hard assignment at
        # posterior 0.5). Composition is then readable straight off the swarm, and it always
        # agrees with the mixture drawn beside it because it comes from the same fit.
        hi_mask = state_assignment(ys, f) if f is not None else None
        if hi_mask is None:
            # No usable fit for this well (too few cells to converge). Its cells are still shown —
            # dropping them would misrepresent the replicate — but in hollow gray, so they read as
            # "unclassified" and can never be mistaken for a fitted state.
            ax.plot(xs, ys, mk, ms=ms, mfc="none", mec=UNFIT_INK, mew=max(edge_lw, 0.45),
                    alpha=alpha, ls="none", zorder=zorder + i * 0.01, rasterized=True)
        else:
            for sel, cc in ((~hi_mask, STATE_LO), (hi_mask, STATE_HI)):
                if sel.any():
                    ax.plot(xs[sel], ys[sel], mk, ms=ms, mfc=cc, mec=CELL_EDGE, mew=edge_lw,
                            alpha=alpha, ls="none", zorder=zorder + i * 0.01, rasterized=True)
        gm = state_geomeans(y[m], f)
        if f is not None:
            mixture_halves(ax, axis_x + gap, [f], side="right", half_width=fw,
                           grid=grid, alpha=0.80, mean_len=0.95)
            pass
        # THREE measurement bars per replicate — high, overall, low — drawn on the CELL side,
        # because every one of them is a geometric mean OF THOSE CELLS. Putting them over the swarm
        # they summarize (and the fitted center line inside its own curve on the other side) means
        # each side of the unit carries one kind of quantity and they can never be confused.
        for key, cc in (("hi", STATE_HI), ("overall", POOL_INK), ("lo", STATE_LO)):
            v = gm.get(key)
            if v is None or not np.isfinite(v):
                continue
            ax.plot([axis_x - gap - sw, axis_x - gap], [v, v],
                    lw=(1.9 if key == "overall" else 1.7), color=cc, alpha=0.97,
                    zorder=zorder + 2, solid_capstyle="butt")
        # Record this unit's swarm band so column_summary can restrict its reference lines to the
        # cell side of each unit instead of running them across the fitted curves. Stashed on the
        # Axes because the (handles, centers) return signature is used by existing callers.
        if not hasattr(ax, "_c04_swarm_spans"):
            ax._c04_swarm_spans = {}
        ax._c04_swarm_spans.setdefault(round(float(xc), 6), []).append(
            (float(axis_x - gap - sw), float(axis_x - gap)))
        # Record the unit's MIXTURE-side geometry too, so per-replicate labels can be placed to sit
        # directly on the fitted state-mean bars (the bar then underlines its own number).
        if not hasattr(ax, "_c04_units"):
            ax._c04_units = {}
        ax._c04_units.setdefault(round(float(xc), 6), []).append(dict(
            group=gv, axis_x=float(axis_x + gap), bar_len=float(fw * 0.95),
            # cell-side geometry: where the three measurement bars live, and where the numbers
            # computed FROM those bars are written
            cell_x=float(axis_x - gap - sw), cell_len=float(sw),
            fit_x=float(axis_x + gap), fit_len=float(fw),
            n=int(m.sum()), fit=f, y=y[m], gm=gm))
        centers.append((axis_x, int(m.sum()), COUNT_INK))
    if count_y is not None:
        # ONE combined count row per column ("n = 57, 43, 72"), not a diagonal stagger: the counts
        # belong to the column as a set and read faster on a single line.
        ns = ", ".join(str(int((g == gv).sum())) for gv in gvals)
        ax.text(xc, count_y, f"n = {ns}", ha="center", va="center",
                fontsize=count_fontsize, color=COUNT_INK)
    return handles, centers

def state_assignment(y, fit):
    """Hard two-state assignment for each cell under ONE well's own fit: True = high state.

    Posterior odds at the fitted weights and widths, thresholded at 0.5. Returns None if the fit
    is unusable. This is only ever called with the fit belonging to the cells being colored —
    never a pooled or averaged fit.
    """
    if fit is None:
        return None
    try:
        mu_lo, mu_hi, w_hi, sd = (float(fit["mu_lo"]), float(fit["mu_hi"]),
                                  float(fit["w_hi"]), float(fit["width"]))
    except (KeyError, TypeError, ValueError):
        return None
    if not np.isfinite([mu_lo, mu_hi, w_hi, sd]).all() or sd <= 0:
        return None
    # A fit whose two components have collapsed onto each other (flagged `weak_sep` by the fitter,
    # or separated by less than half a component width) carries NO state information: at equal
    # means the assignment is decided by the mixing weight alone and every cell lands in one state,
    # which would be reported as a real 0% / 100% composition. Treat it as unclassified instead —
    # the cells are still drawn and still counted, they just get no state decomposition.
    try:
        if fit.get("weak_sep"):
            return None
    except AttributeError:
        pass
    if abs(mu_hi - mu_lo) < 0.5 * sd:
        return None
    y = np.asarray(y, float)
    lo = (1.0 - w_hi) * np.exp(-0.5 * ((y - mu_lo) / sd) ** 2)
    hi = w_hi * np.exp(-0.5 * ((y - mu_hi) / sd) ** 2)
    return hi > lo

def column_summary(ax, xc, y_all, fits, half_width=0.44, ratio_y=None, lw=1.15,
                   ratio_fontsize=6.0, zorder=11, label_ratio=True, dash=(0, (4.5, 2.0)),
                   side="mixture"):
    """Column-level reference marks: the pooled mean of every cell in the column, and the mean
    high- and low-state centers across that column's per-well fits, plus the high:low separation.

    These are SUMMARIES OF the per-well fits, not a fit to pooled cells — the distinction matters
    and is why they are drawn with a long-dash pattern that no per-well element uses. (The short
    dash inside a unit means something else entirely: a state on <10 effective cells.)

    Weighting: state centers are averaged with each well's effective cell count in that state, so
    a well contributing three cells to the high state cannot drag the column center.

    Returns dict(mean_all, mu_lo, mu_hi, ratio) with NaN where a component is unavailable.
    """
    out = {"mean_all": np.nan, "mu_lo": np.nan, "mu_hi": np.nan, "ratio": np.nan}
    y_all = np.asarray(y_all, float); y_all = y_all[np.isfinite(y_all)]
    # Reference lines are drawn on ONE side of each unit only, so they never run the full width of
    # the column and read as a single bar across unrelated replicates. `side="mixture"` puts them
    # on the fit side, aligned with the per-well state bars and the relative-motility labels that
    # sit on them, so every horizontal mark in a unit shares one baseline; `side="swarm"` puts them
    # over the cells instead. Falls back to the full column width if no geometry was recorded.
    if side == "mixture":
        units = getattr(ax, "_c04_units", {}).get(round(float(xc), 6)) or []
        spans = [(u["axis_x"], u["axis_x"] + u["bar_len"]) for u in units]
    else:
        spans = getattr(ax, "_c04_swarm_spans", {}).get(round(float(xc), 6))
    if not spans:
        spans = [(xc - half_width, xc + half_width)]
    def seg(v, **kw):
        for a, b in spans:
            ax.plot([a, b], [v] * 2, **kw)
    if len(y_all):
        out["mean_all"] = float(np.mean(y_all))
        seg(out["mean_all"], lw=lw + 0.25, color=POOL_INK, solid_capstyle="butt",
            zorder=zorder + 0.2)
    def wmean(key, wkey):
        vs = [(float(f[key]), max(float(f.get(wkey, 1.0) or 0.0), 0.0))
              for f in fits if f is not None and np.isfinite(f.get(key, np.nan))]
        vs = [(v, w) for v, w in vs if np.isfinite(v)]
        if not vs:
            return np.nan
        tot = sum(w for _, w in vs)
        if tot <= 0:
            return float(np.mean([v for v, _ in vs]))
        return float(sum(v * w for v, w in vs) / tot)
    # Column state centers are pooled ASSIGNED-CELL geometric means, matching the per-replicate
    # bars and the relative-motility labels. (The fitted-center weighted mean, `wmean`, is kept
    # available above but is no longer what the column line reports — mixing the two estimators
    # across a figure was the ambiguity removed in the 08.11 revision.)
    units = getattr(ax, "_c04_units", {}).get(round(float(xc), 6)) or []
    hi_pool = [u["gm"]["hi"] for u in units
               if u.get("gm") and np.isfinite(u["gm"].get("hi", np.nan))]
    lo_pool = [u["gm"]["lo"] for u in units
               if u.get("gm") and np.isfinite(u["gm"].get("lo", np.nan))]
    hi_w = [u["gm"]["n_hi"] for u in units
            if u.get("gm") and np.isfinite(u["gm"].get("hi", np.nan))]
    lo_w = [u["gm"]["n_lo"] for u in units
            if u.get("gm") and np.isfinite(u["gm"].get("lo", np.nan))]
    out["mu_hi"] = (float(np.average(hi_pool, weights=hi_w)) if hi_pool and sum(hi_w) > 0
                    else wmean("mu_hi", "n_hi_eff"))
    out["mu_lo"] = (float(np.average(lo_pool, weights=lo_w)) if lo_pool and sum(lo_w) > 0
                    else wmean("mu_lo", "n_lo_eff"))
    for v, c in ((out["mu_lo"], STATE_LO), (out["mu_hi"], STATE_HI)):
        if np.isfinite(v):
            seg(v, lw=lw, color=c, ls=dash, zorder=zorder, dash_capstyle="butt")
    if np.isfinite(out["mu_hi"]) and np.isfinite(out["mu_lo"]):
        out["ratio"] = float(10.0 ** (out["mu_hi"] - out["mu_lo"]))
        if label_ratio and ratio_y is not None:
            ax.text(xc, ratio_y, f"{out['ratio']:.0f}×", ha="center", va="center",
                    fontsize=ratio_fontsize, color=MUTED, style="italic")
    return out

def half_violin(ax, xc, y, side="left", half_width=0.34, grid=None, color=OBS_FILL,
                alpha=0.95, zorder=2, bw=0.30, edge=True):
    """KDE of the OBSERVED per-cell values — a true violin of the data, not a model curve."""
    y = np.asarray(y, float); y = y[np.isfinite(y)]
    if len(y) < 5 or grid is None:
        return
    kde = gaussian_kde(y, bw_method=bw)
    d = kde(grid); mx = d.max()
    if mx <= 0: return
    d = d / mx * half_width
    sgn = -1.0 if side == "left" else 1.0
    ax.fill_betweenx(grid, xc, xc + sgn * d, color=color, alpha=alpha, lw=0, zorder=zorder)
    if edge:
        ax.plot(xc + sgn * d, grid, lw=0.5, color="#9AA0A6", alpha=0.9, zorder=zorder + 0.1)

def mixture_halves(ax, xc, fits, side="right", half_width=0.34, grid=None,
                   alpha=None, zorder=3, lw=0.7, means=True, mean_len=0.55):
    """Two-state mixture curves, ONE PER WELL, overlaid — never a fit to pooled replicates.

    `fits` is a list of dicts with mu_lo, mu_hi, w_hi and width (total component sd for that
    well). Each well contributes its own pair of half-curves, so replicate disagreement is
    visible as spread between curves rather than hidden inside an averaged fit.
    """
    if grid is None or not fits:
        return
    sgn = -1.0 if side == "left" else 1.0
    if alpha is None:
        alpha = 0.85 if len(fits) == 1 else (0.42 if len(fits) == 2 else 0.30)
    # common scale so curve heights are comparable within the column
    peaks = []
    for f in fits:
        w = f.get("width", 0.466)
        dh = f["w_hi"] * np.exp(-0.5 * ((grid - f["mu_hi"]) / w) ** 2) / w
        dl = (1 - f["w_hi"]) * np.exp(-0.5 * ((grid - f["mu_lo"]) / w) ** 2) / w
        peaks.append(np.nanmax(dh + dl))
    scale = half_width / max(max(peaks), 1e-9)
    for f in fits:
        w = f.get("width", 0.466)
        dh = f["w_hi"] * np.exp(-0.5 * ((grid - f["mu_hi"]) / w) ** 2) / w * scale
        dl = (1 - f["w_hi"]) * np.exp(-0.5 * ((grid - f["mu_lo"]) / w) ** 2) / w * scale
        ax.fill_betweenx(grid, xc, xc + sgn * dl, color=STATE_LO, alpha=alpha, lw=0, zorder=zorder)
        ax.fill_betweenx(grid, xc, xc + sgn * dh, color=STATE_HI, alpha=alpha, lw=0, zorder=zorder + 0.1)
        # Outline only where the curve stands off the baseline, so no vertical stroke appears
        # along x = xc (which would read as a spurious axis).
        for d, c in ((dl, STATE_LO), (dh, STATE_HI)):
            vis = d > (half_width * 0.004)
            xs = np.where(vis, xc + sgn * d, np.nan)
            ax.plot(xs, grid, lw=lw, color=c, alpha=min(1, alpha + 0.35),
                    zorder=zorder + 0.25, solid_capstyle="round")
        if means:
            # The fitted Gaussian center is drawn INSIDE its own curve — spanning only the width of
            # that component at its peak — so it reads as part of the curve rather than as a bar
            # competing with the cell-side measurement bars. It marks where the model places the
            # component; it is not a reported measurement.
            for mu, dens_, weak_key in ((f["mu_hi"], dh, "weak_high"),
                                        (f["mu_lo"], dl, "weak_low")):
                if not np.isfinite(mu):
                    continue
                w_at_mu = float(np.interp(mu, grid, dens_))
                ax.plot([xc, xc + sgn * w_at_mu], [mu, mu], lw=0.9, color=STATE_INK,
                        alpha=0.85, zorder=zorder + 1,
                        ls=((0, (1.6, 1.2)) if f.get(weak_key) else "-"),
                        solid_capstyle="butt")

# ---------------------------------------------------------------- annotation
def stars_for_q(q):
    if q is None or not np.isfinite(q): return "ns"
    return "****" if q < 1e-4 else "***" if q < 1e-3 else "**" if q < 1e-2 else "*" if q < 0.05 else "ns"

def sig_bracket(ax, x0, x1, y, stars, drop=0.055, lw=0.9, color="#000000", fontsize=8.0,
                extra=None):
    """Asterisk bracket. Significance is shown with asterisks, never as inline q-value text."""
    ax.plot([x0, x0, x1, x1], [y - drop, y, y, y - drop], lw=lw, color=color,
            clip_on=False, solid_joinstyle="miter")
    lab = stars if extra is None else f"{stars} {extra}"
    ax.text((x0 + x1) / 2, y + drop * 0.35, lab, ha="center", va="bottom",
            fontsize=fontsize, color=color, clip_on=False)

def n_label(ax, xc, n, y, fontsize=5.4, color=MUTED, prefix="n="):
    ax.text(xc, y, f"{prefix}{int(n)}", ha="center", va="center", fontsize=fontsize, color=color)

def log_ticks(ax, decades=(0, 1, 2, 3), labels=("1", "10", "100", "1000"), show=True):
    ax.set_yticks(list(decades))
    ax.set_yticklabels(list(labels) if show else [])

def minor_log_ticks(ax, lo=-1, hi=4):
    mn = []
    for d in range(lo, hi):
        mn += [d + np.log10(k) for k in range(2, 10)]
    ax.set_yticks([m for m in mn if ax.get_ylim()[0] <= m <= ax.get_ylim()[1]], minor=True)
    ax.tick_params(axis="y", which="minor", length=1.8, width=0.7)

def legend_handles(states=True, cells=True, obs=True, extras=()):
    h = []
    if obs:    h.append(Patch(fc=OBS_FILL, ec="#9AA0A6", lw=0.5, label="observed distribution"))
    if states: h += [Patch(fc=STATE_HI, alpha=.6, label="fitted high state"),
                     Patch(fc=STATE_LO, alpha=.6, label="fitted low state"),
                     Line2D([], [], color=STATE_HI, lw=1.6, label="state mean (per well)")]
    if cells:  h.append(Line2D([], [], marker="o", ls="", mfc=CELL_DOT, mec="none", ms=2.6,
                               label="one cell"))
    return h + list(extras)


# ---------------------------------------------------------------- Yannick 2026.08.11 revisions

def p_stars(p, ns="ns"):
    """GraphPad significance vocabulary: **** <1e-4, *** <1e-3, ** <0.01, * <0.05, else 'ns'."""
    if p is None or not np.isfinite(p):
        return ns
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ns


def sig_bracket(ax, x1, x2, y, label, lw=0.9, fontsize=8.0, color=None, dy_frac=0.010,
                halo=None):
    """A GraphPad-style significance annotation: a flat horizontal rule with the label above it.

    Matches the convention in the CAMP bar-graph figures — a plain line spanning the compared
    columns with no end caps or serifs, asterisks (or 'ns') centered above it. Asterisks sit closer
    to the rule than a word does, because the glyphs hang from their top edge; 'ns' is nudged up so
    both read as sitting on the line rather than touching it.
    """
    col = POOL_INK if color is None else color
    ax.plot([x1, x2], [y, y], color=col, lw=lw, solid_capstyle="butt",
            clip_on=False, zorder=15)
    star = label not in ("ns", "n.s.")
    # Offset is fixed in POINTS, not in axis-fraction: an axis-fraction offset silently grows
    # when the y limits are enlarged to fit more bracket levels, floating the label off its rule.
    # Asterisk glyphs carry their ink in the upper half of the em box, so they need a NEGATIVE
    # offset from the baseline to sit on the rule; 'ns' is a normal x-height word and sits above it.
    # Asterisk ink sits +3.4..+5.8 pt above the BASELINE (measured with TextPath on the render face),
    # so a baseline offset of -2.4 pt puts the bottom of the ink ~1 pt clear of the rule. 'ns' is a
    # normal x-height word whose ink starts at the baseline, so it takes a small positive offset.
    off_pt = -2.4 if star else 1.4
    t = ax.annotate(label, xy=((x1 + x2) / 2.0, y), xycoords="data",
                    xytext=(0, off_pt), textcoords="offset points", ha="center",
                    va="baseline" if star else "bottom",
                    fontsize=fontsize if star else fontsize * 0.80,
                    color=col, zorder=16, clip_on=False)
    if halo:
        t.set_bbox(dict(boxstyle="square,pad=0.10", fc="white", ec="none", alpha=halo))
    return t


def well_arith_means(y, groups):
    """Per-well ARITHMETIC mean MSD in linear units (um^2), one value per replicate well.

    `y` is log10 MSD, so this exponentiates first and then averages: the mean of the cells' MSD,
    not the geometric mean. This is the whole-population estimator Yannick and Josh specified for
    relative motility, and it is deliberately NOT the estimator behind the geometric-mean bars —
    an arithmetic mean in a right-skewed, two-state distribution is pulled toward the high state,
    which is the point: it is a whole-population summary, not a per-state one.
    """
    y = np.asarray(y, float); groups = np.asarray(groups)
    out = {}
    for v in sorted(set(groups.tolist())):
        vals = y[groups == v]; vals = vals[np.isfinite(vals)]
        if len(vals):
            out[v] = float(np.mean(10.0 ** vals))
    return out


def column_relative(y, groups, ref=None):
    """Whole-population relative motility for one condition column, with SEM across wells.

    Returns {mean, sem, n, rel, rel_sem}. `mean` is the arithmetic mean over the column's per-well
    arithmetic means (each well weighted equally — the well is the unit of analysis). `sem` is the
    standard error of those per-well values. `rel` divides by the reference column's `mean`; its
    error propagates only this column's SEM, so the reference row's own SEM is reported alongside
    it rather than folded in — the reference is assigned unity by construction, and its scatter is
    a separate quantity a reader needs in order to judge any comparison against it.
    """
    w = well_arith_means(y, groups); vals = list(w.values()); n = len(vals)
    if not n:
        return dict(mean=np.nan, sem=np.nan, n=0, rel=np.nan, rel_sem=np.nan, wells=w)
    m = float(np.mean(vals))
    sem = float(np.std(vals, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    if ref is None or not np.isfinite(ref) or ref == 0:
        return dict(mean=m, sem=sem, n=n, rel=np.nan, rel_sem=np.nan, wells=w)
    return dict(mean=m, sem=sem, n=n, rel=m / ref, rel_sem=(sem / ref if np.isfinite(sem) else np.nan),
                wells=w)


def column_relative_label(ax, xc, cr, y, fontsize=6.0, color=None, halo=0.85, is_ref=False):
    """One whole-population relative-motility label centered over a condition column.

    The reference column is drawn as "1.00 +/- sem" rather than "Ref.": Josh asked for the
    reference to carry a SEM so comparisons against it are interpretable, which means it needs a
    number, not a word.
    """
    if not np.isfinite(cr.get("rel", np.nan)):
        return None
    col = POOL_INK if color is None else color
    s = f"{cr['rel']:.2f}"
    if np.isfinite(cr.get("rel_sem", np.nan)):
        s += f" \u00b1 {cr['rel_sem']:.2f}"
    t = ax.text(xc, y, s, ha="center", va="center", fontsize=fontsize, color=col, zorder=14)
    t.set_bbox(dict(boxstyle="square,pad=0.16", fc="white", ec="none", alpha=halo))
    return t


def replicate_percentages(ax, xc, hi_y, lo_y, fontsize=5.2, halo=0.82, dx_frac=0.30,
                          inset=0.0, symbol="\u03c1"):
    """Percent of cells assigned to each state, PER REPLICATE, centered on that replicate's unit.

    One pair of numbers per violin, sitting directly above and below the unit it describes, so the
    reader attaches a composition to a distribution by position rather than by counting along a row
    at the column edge. Percentages come from the same hard assignment that colors the dots
    (`state_geomeans` -> n_hi / n_lo), so they always agree with what is drawn.

    Returns [(group, pct_hi, pct_lo), ...].
    """
    units = getattr(ax, "_c04_units", {}).get(round(float(xc), 6), [])
    out = []
    for u in units:
        gm = u.get("gm") or {}
        n_hi, n_lo = int(gm.get("n_hi", 0)), int(gm.get("n_lo", 0))
        tot = n_hi + n_lo
        if not tot:
            continue                      # unconverged well: no composition to report
        p_hi = 100.0 * n_hi / tot
        # Sit over the CELLS, not the midpoint of the unit: these are proportions of cells, and
        # centring them between swarm and curve made them read as describing the fitted mixture.
        # `dx_frac` slides them from the unit center toward the swarm; `inset` pulls both rows in
        # toward the data so each pair visibly belongs to the distribution between them.
        mid = (u.get("cell_x", xc) + u.get("fit_x", xc) + u.get("fit_len", 0.0)) / 2.0
        ux = mid - dx_frac * (mid - u.get("cell_x", xc))
        for val, yy, col in ((p_hi, hi_y - inset, STATE_HI),
                             (100.0 - p_hi, lo_y + inset, STATE_LO)):
            # Italic ρ = ..% marks these as population proportions rather than another x-fold value.
            t = ax.text(ux, yy, f"{symbol} = {val:.0f}%", color=col, fontsize=fontsize,
                        ha="center", va="center", zorder=13, style="italic")
            if halo:
                t.set_bbox(dict(boxstyle="square,pad=0.08", fc="white", ec="none", alpha=halo))
            if not hasattr(ax, "_c04_rel_texts"):
                ax._c04_rel_texts = []
            ax._c04_rel_texts.append(t)
        out.append((u.get("group"), p_hi, 100.0 - p_hi))
    return out


def state_percentages(ax, x_right, y_all, fits, hi_y, lo_y, fontsize=5.8, ha="left", pad=0.02):
    """Percent of cells assigned to each state, in state color, at the right of a column.

    Computed by HARD-ASSIGNING every cell in the column under its OWN well's fit (the same
    `state_assignment` used to color the dots), then pooling the assignments. It is deliberately
    NOT the mean of the fitted w_hi values: the annotation must describe the cells the reader can
    see, so a well with more cells contributes more, exactly as it does visually.

    Returns (pct_hi, pct_lo); NaN if no cell could be assigned.
    """
    n_hi = n_tot = 0
    for y, f in zip(y_all, fits):
        y = np.asarray(y, float); y = y[np.isfinite(y)]
        m = state_assignment(y, f)
        if m is None:
            continue                      # unconverged well: excluded from the percentage
        n_hi += int(m.sum()); n_tot += int(m.size)
    if not n_tot:
        return (np.nan, np.nan)
    p_hi = 100.0 * n_hi / n_tot
    p_lo = 100.0 - p_hi
    ax.text(x_right + pad, hi_y, f"{p_hi:.0f}%", color=STATE_HI, fontsize=fontsize,
            ha=ha, va="center", zorder=12)
    ax.text(x_right + pad, lo_y, f"{p_lo:.0f}%", color=STATE_LO, fontsize=fontsize,
            ha=ha, va="center", zorder=12)
    return (p_hi, p_lo)

def geo_mean_log(y):
    """Geometric mean of MSD for cells given in log10 — i.e. 10**mean(log10 y)."""
    y = np.asarray(y, float); y = y[np.isfinite(y)]
    return float(10.0 ** np.mean(y)) if len(y) else np.nan

def control_band(ax, x0, x1, color=None, zorder=0):
    """Shade a positive-control column and record its extent for the axis spine.

    The shaded band must not run past the end of the drawn axis, and the axis must not stop short
    inside it — either reads as the band being outside the plot. Callers pass the accumulated
    extents to `gp_axes(..., xbounds=...)` so the spine spans every band it contains.

    Returns (x0, x1).
    """
    ax.axvspan(x0, x1, color=(CTRL_BAND if color is None else color), zorder=zorder)
    if not hasattr(ax, "_c04_bands"):
        ax._c04_bands = []
    ax._c04_bands.append((float(x0), float(x1)))
    return (float(x0), float(x1))


def band_bounds(ax, lo, hi):
    """Widen an (lo, hi) spine bound to cover any control bands drawn on this Axes."""
    bands = getattr(ax, "_c04_bands", [])
    if not bands:
        return (lo, hi)
    return (min([lo] + [b[0] for b in bands]), max([hi] + [b[1] for b in bands]))


def state_geomeans(y, fit):
    """Geometric mean log10 MSD of the cells ASSIGNED to each state, plus over all cells.

    Returns dict(hi, lo, overall, n_hi, n_lo) with NaN where a state has no assigned cells.

    This is the measurement the figure reports: it is computed from the cells the reader can see,
    each hard-assigned under its own well's fit. The fitted Gaussian center is a different quantity
    — a parameter of the model, pulled by the assumed component width and by cells in the tail that
    the hard assignment gives to the other state — and the two can differ by a few tenths of a dex
    in wells with heavy overlap. Drawing bars at one and labeling them with the other is the bug
    this function exists to remove.
    """
    out = dict(hi=np.nan, lo=np.nan, overall=np.nan, n_hi=0, n_lo=0)
    y = np.asarray(y, float); y = y[np.isfinite(y)]
    if not len(y):
        return out
    out["overall"] = float(np.mean(y))
    hi_mask = state_assignment(y, fit)
    if hi_mask is None:
        return out
    if hi_mask.any():
        out["hi"] = float(np.mean(y[hi_mask])); out["n_hi"] = int(hi_mask.sum())
    if (~hi_mask).any():
        out["lo"] = float(np.mean(y[~hi_mask])); out["n_lo"] = int((~hi_mask).sum())
    return out


def relative_motility(y_all, fits, ref):
    """Overall / high / low relative motility for one column against a reference column.

    `ref` is the dict returned for the reference column by this same function (pass None to GET
    that dict). ALL THREE values are geometric means of cells: overall over every cell in the
    column, and the state values over the cells assigned to that state under their own well's fit
    (pooled across the column's wells). This is the same estimator `state_geomeans` computes per
    replicate, so a relative value always describes the bar it is written on.

    `fits` is positionally matched to `y_all` — one fit per well, used only to assign that well's
    own cells.
    """
    y_all = [np.asarray(v, float) for v in y_all]
    y = np.concatenate(y_all) if len(y_all) else np.array([])
    y = y[np.isfinite(y)]
    hi_pool, lo_pool = [], []
    for yv, f in zip(y_all, list(fits) + [None] * max(0, len(y_all) - len(fits))):
        yv = yv[np.isfinite(yv)]
        if not len(yv):
            continue
        m = state_assignment(yv, f)
        if m is None:
            continue
        hi_pool.append(yv[m]); lo_pool.append(yv[~m])
    def pooled(chunks):
        v = np.concatenate(chunks) if chunks else np.array([])
        return float(np.mean(v)) if len(v) else np.nan
    cur = dict(overall=float(np.mean(y)) if len(y) else np.nan,
               hi=pooled(hi_pool), lo=pooled(lo_pool))
    if ref is None:
        return cur
    return {k: (10.0 ** (cur[k] - ref[k]) if np.isfinite(cur[k]) and np.isfinite(ref[k]) else np.nan)
            for k in cur}, cur

def rel_label(v, is_ref=False, dp=2):
    """Relative-motility text. The reference column reads 'Ref.', never '1.00x'."""
    if is_ref:
        return "Ref."
    if not np.isfinite(v):
        return ""
    return f"{v:.{dp}f}×"

def replicate_relative(ax, xc, ref, fontsize=5.0, dp=2, is_ref=False, halo=0.82,
                       overall_bar=True, bar_lw=1.4, pad_frac=0.55, fit_ref=None, label_dy=0.012):
    """Per-REPLICATE relative motility, each value sitting on the bar it describes.

    Reads the unit geometry recorded by `replicate_units`, so every replicate in the column gets
    its own high / overall / low numbers rather than one set for the pooled column. Labels are
    left-aligned at the start of each state-mean bar on the mixture side, so the bar runs underneath
    the number and visually underlines it.

    `ref` is a reference dict from `relative_motility(..., None)`. The overall value has no fitted
    bar of its own, so one is drawn (POOL_INK) at the column's geometric mean for that replicate.

    A white halo keeps the text readable where it crosses a Gaussian curve. `pad_frac` dodges the
    label outward along its bar, away from the unit's baseline, so the number sits over the thinning
    tail of the curve rather than its dense body — that placement does most of the legibility work,
    with the halo only cleaning up what remains.
    """
    units = getattr(ax, "_c04_units", {}).get(round(float(xc), 6), [])
    out = []
    # Occupancy is shared across the whole column: labels from ADJACENT replicate units can be
    # close enough in x to collide, so a per-unit check is not sufficient.
    col_placed = []          # (x_left, x_right, y)
    for u in units:
        # Values and label positions both come from the ASSIGNED-CELL geometric means recorded by
        # `replicate_units` — the same numbers its bars are drawn at. Reading the fitted Gaussian
        # center here instead would place a label on a bar that measures something else.
        gm = u.get("gm") or state_geomeans(u.get("y"), u.get("fit"))
        rows, vals = [], {}
        for key, col in (("hi", STATE_HI), ("overall", POOL_INK), ("lo", STATE_LO)):
            cur = gm.get(key, np.nan)
            if not np.isfinite(cur):
                vals[key] = np.nan
                continue
            rows.append((float(cur), col, key))
            vals[key] = (10.0 ** (cur - ref[key])
                         if ref is not None and np.isfinite(ref.get(key, np.nan)) else np.nan)
        # CELL-SIDE labels: ratios of assigned-cell geometric means, written over the bars that
        # mark those means, right-aligned at the bar's outer end so the bar underlines the number.
        rows = sorted(rows, key=lambda r: -r[0])
        cx0, clen = u.get("cell_x", u["axis_x"]), u.get("cell_len", u["bar_len"])
        # Lift the label clear of its own bar. `va="bottom"` alone leaves the glyph box resting ON
        # the line, so the halo clips the bar's end and the two read as one smudged mark; a small
        # offset in y separates them while keeping the number unambiguously attached to its bar.
        y0_, y1_ = ax.get_ylim()
        dy = (y1_ - y0_) * label_dy
        for mu, col, key in rows:
            t = ax.text(cx0 + clen * (1.0 - pad_frac), mu + dy,
                        rel_label(vals.get(key, np.nan), is_ref=is_ref, dp=dp),
                        color=col, fontsize=fontsize, ha="right", va="bottom", zorder=14)
            if halo:
                t.set_bbox(dict(boxstyle="square,pad=0.10", fc="white", ec="none", alpha=halo))
            if not hasattr(ax, "_c04_rel_texts"):
                ax._c04_rel_texts = []
            ax._c04_rel_texts.append(t)

        # CURVE-SIDE labels (optional): the SAME ratios computed from the fitted Gaussian centers
        # instead of the cells. Two estimators of the same quantity, each written beside the mark
        # it comes from, so the two can be compared directly and one chosen. `fit_ref` must be a
        # reference dict built from fitted centers, never the cell-based one.
        if fit_ref is not None:
            f = u.get("fit")
            fx0, flen = u.get("fit_x", u["axis_x"]), u.get("fit_len", u["bar_len"])
            for key, mukey, col in (("hi", "mu_hi", STATE_HI), ("lo", "mu_lo", STATE_LO)):
                cur = (f or {}).get(mukey, np.nan)
                if not np.isfinite(cur) or not np.isfinite(fit_ref.get(key, np.nan)):
                    continue
                t = ax.text(fx0 + flen * pad_frac, cur,
                            rel_label(10.0 ** (cur - fit_ref[key]), is_ref=is_ref, dp=dp),
                            color=col, fontsize=fontsize * 0.92, ha="left", va="bottom",
                            zorder=14, style="italic")
                if halo:
                    t.set_bbox(dict(boxstyle="square,pad=0.10", fc="white", ec="none", alpha=halo))
                ax._c04_rel_texts.append(t)
        out.append((u["group"], vals))
    return out

def resolve_label_overlaps(fig, max_passes=6, pad_px=1.0):
    """Nudge relative-motility labels apart using MEASURED bounding boxes.

    Data-space width estimates for text are unreliable — a glyph box depends on the font, the
    figure size and the dpi, so the same estimate is wrong at every other panel width. This runs
    after the figure exists, measures real extents, and moves only the text (never the bar) by the
    smallest amount that clears the collision. Call once, immediately before saving.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for _ in range(max_passes):
        moved = 0
        for ax in fig.axes:
            ts = getattr(ax, "_c04_rel_texts", [])
            if not ts:
                continue
            items = sorted(ts, key=lambda t: -t.get_window_extent(r).y0)
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    A = items[i].get_window_extent(r)
                    B = items[j].get_window_extent(r)
                    if not A.overlaps(B):
                        continue
                    shift_px = (B.y1 - A.y0) + pad_px          # push the lower label down
                    inv = ax.transData.inverted()
                    _, y_at_0 = inv.transform((0, 0))
                    _, y_at_s = inv.transform((0, shift_px))
                    dy = y_at_s - y_at_0
                    x, y = items[j].get_position()
                    items[j].set_position((x, y - abs(dy)))
                    moved += 1
        if not moved:
            break
        fig.canvas.draw()
    return moved

def axis_break(ax, x, height=0.018, gap=0.55, lean=0.55, lw=1.0, color="black", zorder=20,
               erase=True, erase_pad=1.05):
    """Draw a two-stroke break ON the bottom spine at data-x `x`.

    Marks that the x axis is NOT continuous across that point — used where a categorical axis
    switches to a different kind of condition (e.g. a density series next to a PMA control), so
    the neighbouring column is not read as a further step along the same scale.

    Geometry is anchored to the spine's OWN position, read back from the Axes: the spine is
    routinely moved to a data-y (`spines["bottom"].set_position(("data", y))`), and drawing the
    strokes in axes-fraction coordinates instead puts them somewhere below the visible line. Both
    strokes and the erased segment therefore live in data coordinates on the spine.

    `height` is the stroke half-height as a fraction of the y range, `gap` the clear space between
    the two strokes as a fraction of that height, and `lean` their slant. With `erase`, the spine
    itself is broken by overplotting a short white segment between the strokes, so the axis runs UP
    TO the glyph from each side and stops there — the conventional reading — rather than passing
    through it.

    The glyph is small on purpose: it is punctuation on the axis, not a data mark. Sized to roughly
    the height of a tick, it stays legible without competing with the plotted content.

    Use it ONLY where the axis genuinely stops being one continuous scale — a dose or density series
    meeting a differently-treated control. Marking every categorical step turns a meaningful signal
    into decoration.
    """
    y0, y1 = ax.get_ylim()
    pos = ax.spines["bottom"].get_position()
    ys = pos[1] if isinstance(pos, tuple) and pos[0] == "data" else y0
    # Size the glyph in DISPLAY pixels and convert back, so the strokes are the same physical
    # length and slant on every panel. Deriving the horizontal offsets from a y-data height instead
    # would shear the glyph differently in each panel, since x and y have unrelated scales here.
    h_px = abs(ax.transData.transform((0, y0 + (y1 - y0) * height))[1]
               - ax.transData.transform((0, y0))[1])
    ox, oy = ax.transData.transform((x, ys))
    def d(px, py):
        return ax.transData.inverted().transform((ox + px, oy + py))
    half_gap_px = h_px * gap
    lean_px = h_px * lean
    if erase:
        xa, _ = d(-(half_gap_px + lean_px * erase_pad), 0)
        xb, _ = d(+(half_gap_px + lean_px * erase_pad), 0)
        ax.plot([xa, xb], [ys, ys], color="white", lw=lw + 2.0, clip_on=False,
                zorder=zorder - 0.1, solid_capstyle="butt")
    for sgn in (-1.0, 1.0):
        p0 = d(sgn * half_gap_px - lean_px, -h_px)
        p1 = d(sgn * half_gap_px + lean_px, +h_px)
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, lw=lw, clip_on=False,
                zorder=zorder, solid_capstyle="round")
