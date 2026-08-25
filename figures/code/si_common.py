"""Shared data preparation for the Supplementary Figure builders.

    load_fits()        per-well mixture fits, converged only
    load_cells()       per-cell MSD table
    figS8_agg(ok)      per-condition aggregate with weak-state flags
    figS9_prep(cells)  binned cells, per-well table, dropout by arm
    figS10_prep(...)   per-well dox table, Spearman correlations, spread

A state fitted on fewer than ten effective cells is flagged and treated as
unquotable. Phase and GFP data use different MSD lags (540 s and 600 s) and are
not placed on a shared axis.
"""
import numpy as np
import pandas as pd

DATA = "../data"

MINEFF = 10.0                   # effective-cell floor for a quotable state
ACUTE_UNIT = "ng/mL"            # PMA dose unit; filenames say "ug" and are WRONG (see README)
ACUTE_MAP = {"2ug": "2", "10ug": "10", "50ug": "50"}

# figS8 colors predate figstyle010 and are kept verbatim so the panel is unchanged
HI, LO, POOL = "#C0392B", "#2E6DA4", "#333333"

#
ORDER = {
    "density: 24h prior": ["25E4", "5E5", "1E6", "PMA"],
    "density: in-assay":  ["25E4", "5E5", "1E6", "PMA"],
    "PMA: activated / acute":   ["0", "2", "10", "50"],
    "PMA: naive / acute":       ["0", "2", "10", "50"],
    "PMA: activated / chronic": ["0", ".08", ".4", "2"],
    "PMA: naive / chronic":     ["0", ".08", ".4", "2"],
    "dox: 239": ["untr.", "500", "PMA"],
    "dox: 240": ["untr.", "20", "100", "500"],
    "dox: 255": ["untr.", "20", "100", "500"],
    "dox: 257": ["untr.", "20", "100", "500"],
}

GROUPS = [
    ("density: 24h prior", "phase", "activated 24 h prior"),
    ("density: in-assay",  "phase", "activated in-assay"),
    ("PMA: activated / acute",   "phase", "activated"),
    ("PMA: naive / acute",       "phase", "naive"),
    ("PMA: activated / chronic", "phase", "activated"),
    ("PMA: naive / chronic",     "phase", "naive"),
    ("dox: 239", "GFP", "239 (DRIVER)"), ("dox: 240", "GFP", "240"),
    ("dox: 255", "GFP", "255"),          ("dox: 257", "GFP", "257"),
]
TITLE = {g: t for g, _, t in GROUPS}

# -- four blocks, one per experiment; lag note is per block by design
BLOCKS = [
    ("Cell density x activation timing",
     ["density: 24h prior", "density: in-assay"], "phase",
     "cells plated per gel", "540 s lag (phase contrast)"),
    ("Acute PMA",
     ["PMA: activated / acute", "PMA: naive / acute"], "phase",
     f"PMA ({ACUTE_UNIT}), dosed before imaging", "540 s lag (phase contrast)"),
    ("Chronic PMA",
     ["PMA: activated / chronic", "PMA: naive / chronic"], "phase",
     "PMA (ng/mL), 48 h exposure", "540 s lag (phase contrast)"),  # literal in
    ("Doxycycline titration",
     ["dox: 239", "dox: 240", "dox: 255", "dox: 257"], "GFP",
     "doxycycline (ng/mL)", "600 s lag (GFP)"),
]

# figS9 arm styling. Neutral inks only -- red/blue are reserved for
# the two motility states and must not be borrowed for experimental arms.
ARMC = {"density": "#1B1F23", "pma": "#5A5A5A", "dox": "#9AA0A6"}
ARMM = {"density": "o", "pma": "s", "dox": "^"}
PAIR_BINS = [0, 1, 2, 3, 5, 8, 12, 17, 100]

# Median displacement pairs PER WELL, untreated vs treated, verified per experiment.
# This is the convention panel (b) plots, so a reader can check it against the panel.
#   density   untreated 14.0 (12 wells)   treated 17.0 (4 wells)
#   PMA expt  untreated 14.5 ( 8 wells)   treated 17.0 (24 wells)
# Do NOT quote 16: that is the density experiment's ALL-wells median (untreated and
# treated pooled), and equally the cell-level treated median pooled over both
# experiments. It is not either experiment's treated arm -- both of those are 17.
MED_PAIRS_PER_WELL = {("density", "untreated"): 14.0, ("density", "treated"): 17.0,
                      ("pma", "untreated"): 14.5, ("pma", "treated"): 17.0}

# Panel (c) dropout rates, by (dataset, arm), as PERCENTAGES.
# These are NOT recomputable from the deposited data: cells_all_datasets.csv.gz
# contains only the 9,746 cells that yielded a usable MSD, while the 6,264 dropped
# rows exist only in the 137 raw per-well source files. The values below were
# computed from those source files and are carried as a constant, exactly as the
# original builder did. Recomputing requires the raw TrackMate output.
DROP_PCT = {("density", "untreated"): 19.7, ("density", "PMA ctrl"): 13.6,
            ("pma", "untreated"): 17.6, ("pma", "PMA"): 13.5,
            ("dox", "untreated"): 54.7, ("dox", "dox"): 53.4, ("dox", "PMA ctrl"): 38.5}


def _cond_label(r):
    """Condition label per dataset."""
    if r.dataset == "density":
        return "PMA" if r.treatment == "PMA" else r.dose_label
    if r.dataset == "pma":
        return "0" if r.dose_label == "NoPMA" else r.dose_label
    return {"untreated": "untr."}.get(r.dose_label, r.dose_label)


def _armlab(r):
    """Experimental arm for S9 panel (c) — verbatim from.

    Note the density PMA control is labelled "PMA ctrl", not "PMA": it is a separate
    control arm of the density experiment, not a dose of the PMA titration. DROP_PCT
    is keyed on these exact strings.
    """
    if r.dataset == "density":
        return "PMA ctrl" if r.treatment == "PMA" else "untreated"
    if r.dataset == "pma":
        return "untreated" if r.dose_label == "NoPMA" else "PMA"
    return ("PMA ctrl" if r.dose_label == "PMA" else
            ("untreated" if r.dose_label == "untreated" else "dox"))


def load_fits():
    wf = pd.read_csv(f"{DATA}/perwell_state_fits_all.csv")
    ok = wf[wf.converged == True].copy()
    ok["cond"] = ok.apply(_cond_label, axis=1)
    ok["cond"] = ok["cond"].map(lambda c: ACUTE_MAP.get(c, c))
    ok["group"] = np.where(
        ok.dataset == "density", "density: " + ok.activation,
        np.where(ok.dataset == "pma",
                 "PMA: " + ok.activation + " / " + ok.treatment,
                 "dox: " + ok.construct.astype(str)))
    ok["arm"] = ok.apply(_armlab, axis=1)
    return ok


def load_cells():
    cells = pd.read_csv(f"{DATA}/cells_all_datasets.csv.gz")
    return cells[np.isfinite(cells.y)].copy()


def figS8_agg(ok):
    """Per-condition state positions, with the weak-state flags S8 draws open."""
    ok = ok.copy()
    ok["hi_weak"] = ok.n_hi_eff < MINEFF
    ok["lo_weak"] = ok.n_lo_eff < MINEFF
    return (ok.groupby(["group", "cond"]).agg(
        mu_lo=("mu_lo", "mean"), mu_hi=("mu_hi", "mean"),
        lo_min=("mu_lo", "min"), lo_max=("mu_lo", "max"),
        hi_min=("mu_hi", "min"), hi_max=("mu_hi", "max"),
        w_hi=("w_hi", "mean"), mean_y=("mean_y", "mean"), n=("well", "size"),
        hi_weak=("hi_weak", "any"), lo_weak=("lo_weak", "any"),
        min_nhi=("n_hi_eff", "min")).reset_index())


def figS9_prep(cells, ok):
    """Track-length tables for S9, 529).

    Returns (b, tld, droparm):
      b       per-cell rows with a displacement-pair bin
      tld     one row per well: median displacement pairs and median MSD
      droparm dropout percentage by (dataset, arm) -- see DROP_PCT note below

    IMPORTANT -- tld spans ALL 137 wells present in the cell table, not the 131
    with a converged mixture fit. The published Spearman rho = -0.67 is the
    137-well value; restricting to converged fits gives -0.68 and does not match
    the caption. Panel (b) is a per-well property and does not require a fit.
    """
    b = cells.dropna(subset=["n_lag_pairs", "y"]).copy()
    b["bin"] = pd.cut(b.n_lag_pairs, PAIR_BINS)

    tld = (cells.groupby(["dataset", "well"])
                .agg(med_pairs=("n_lag_pairs", "median"), med_y=("y", "median"))
                .reset_index())
    tld["ds"] = tld.dataset
    arm = ok.set_index("well").arm
    tld["arm"] = tld.well.map(arm)
    return b, tld, DROP_PCT


def figS10_prep(cells, ok=None):
    """Recovery tables for S10.

    Returns (dd, sp, wr):
      dd  per-well dox rows with cells recovered
      sp  Spearman correlations of each readout with cells per well
      wr  within construct x dose x run max/min recovery ratio

    IMPORTANT -- this uses ALL 89 dox well rows, including the 6 whose mixture did
    not converge, because panel (a) reports CELL RECOVERY, which needs no fit. The
    published n = 89 wells and median max/min = 2.0x are the unfiltered values;
    restricting to the 83 converged wells gives 1.8x and contradicts the caption.
    The `ok` argument is accepted for call-signature symmetry and ignored.
    """
    wf = pd.read_csv(f"{DATA}/perwell_state_fits_all.csv")
    dox = cells[cells.dataset == "dox"]
    dd = wf[wf.dataset == "dox"].copy()
    dd["n_cells"] = dd.well.map(dox.groupby("well").size())
    dd = dd[np.isfinite(dd.n_cells)]
    sp = dd[["n_cells", "mu_hi", "w_hi", "med_y"]].corr(method="spearman")["n_cells"]
    wr = dd.groupby(["construct", "dose_label", "run"]).n_cells.agg(["min", "max"])
    wr["ratio"] = wr["max"] / wr["min"]
    return dd, sp, wr
