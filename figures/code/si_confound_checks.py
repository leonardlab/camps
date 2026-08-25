"""Recomputes the values quoted in the Supplementary Figure 9 and 10 captions.

    PYTHONPATH=. python si_confound_checks.py

Exits non-zero if any published value fails to reproduce.
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

DATA = "../data"
PAIR_MIN = 12          # "well-tracked" threshold, in displacement pairs
ACUTE = {"NoPMA": "0", "2ug": "2", "10ug": "10", "50ug": "50"}
CHRON = {"NoPMA": "0", ".08": ".08", ".4": ".4", "2": "2"}


def load():
    cells = pd.read_csv(f"{DATA}/cells_all_datasets.csv.gz")
    fits = pd.read_csv(f"{DATA}/perwell_state_fits_all.csv")
    cells = cells[np.isfinite(cells.y)]
    return cells, fits


def pooled_mean_msd(cells, wells):
    """Arithmetic mean per-cell MSD over all cells of the listed wells, pooled.
    This is the S9 convention -- NOT the per-well estimator used in Figure 4."""
    sub = cells[cells.well.isin(wells)]
    return float(np.mean(10 ** sub.y.values))


def wellwise_mean_msd(cells, wells):
    """Mean of per-well arithmetic means -- the Figure 4 legend's estimator.
    Provided for comparison so the two conventions can be seen side by side."""
    return float(np.mean([np.mean(10 ** cells[cells.well == w].y.values)
                          for w in wells if (cells.well == w).any()]))


def density_wells(fits, activation, density, pma):
    f = fits[fits.dataset == "density"].copy()
    f["dl"] = f.well.str.extract(r"^(?:PMA_)?(25E4|5E5|1E6)")[0]
    f["is_pma"] = f.well.str.startswith("PMA")
    q = f[(f.activation == activation) & (f.dl == density) & (f.is_pma == pma)]
    return list(q.well)


def pma_wells(fits, activation, timing, dose):
    f = fits[fits.dataset == "pma"].copy()
    f["dl"] = np.where(f.treatment == "acute",
                       f.dose_label.map(ACUTE), f.dose_label.map(CHRON))
    q = f[(f.activation == activation) & (f.treatment == timing) & (f.dl == dose)]
    return list(q.well)


def fig9_fold_reductions(cells, fits):
    """The three fold reductions quoted in the Supplementary Figure 9 caption.

    Each is (untreated / PMA-treated) mean per-cell MSD, pooled across the
    condition's cells, computed twice: on all cells, and on well-tracked cells
    only (>= PAIR_MIN displacement pairs).
    """
    den = cells[cells.dataset == "density"]
    pma = cells[cells.dataset == "pma"]
    comparisons = [
        ("Fig 4b  pre-activated 5E5, untreated vs +PMA",
         den, density_wells(fits, "24h prior", "5E5", False),
              density_wells(fits, "24h prior", "5E5", True)),
        ("Fig 4c  activated, PMA at 0 h, 0 vs 50 ng/mL",
         pma, pma_wells(fits, "activated", "acute", "0"),
              pma_wells(fits, "activated", "acute", "50")),
        ("Fig 4c  activated, PMA at -48 h, 0 vs 2 ng/mL",
         pma, pma_wells(fits, "activated", "chronic", "0"),
              pma_wells(fits, "activated", "chronic", "2")),
    ]
    rows = []
    for label, src, untr, treated in comparisons:
        wt = src[src.n_lag_pairs >= PAIR_MIN]
        rows.append(dict(
            comparison=label,
            n_wells_untreated=len(untr), n_wells_treated=len(treated),
            fold_all_cells=pooled_mean_msd(src, untr) / pooled_mean_msd(src, treated),
            fold_well_tracked=pooled_mean_msd(wt, untr) / pooled_mean_msd(wt, treated),
            fold_all_cells_wellwise=wellwise_mean_msd(src, untr) / wellwise_mean_msd(src, treated),
        ))
    return pd.DataFrame(rows)


def fig9_track_length_summary(cells):
    """The median displacement-pair counts contrasted in the S9 caption."""
    sub = cells[cells.dataset.isin(["density", "pma"])].copy()
    sub["treated"] = np.where(sub.dose_label.isin(["NoPMA", "untreated"]), "untreated", "treated")
    med = sub.groupby("treated").n_lag_pairs.median()
    return med.to_dict()


def fig10_occupancy_adjustment(cells, fits):
    """The naive high-state occupancy shift of the Supplementary Figure 10 caption.

    Readout is w_hi, the fitted fraction of a well's cells in the high-motility
    state. Recovery enters as log10(cells recovered per well) -- see module docstring.
    """
    pma = cells[cells.dataset == "pma"]
    f = fits[fits.dataset == "pma"].copy()
    f["dl"] = np.where(f.treatment == "acute",
                       f.dose_label.map(ACUTE), f.dose_label.map(CHRON))
    nv = f[f.activation == "naive"].copy()
    nv["ncells"] = [int((pma.well == w).sum()) for w in nv.well]
    nv["treated"] = (nv.dl != "0").astype(int)

    mean_untr = nv.loc[nv.treated == 0, "w_hi"].mean()
    mean_treated = nv.loc[nv.treated == 1, "w_hi"].mean()
    unadjusted = mean_treated - mean_untr

    model = smf.ols("w_hi ~ treated + np.log10(ncells)", data=nv).fit()
    adjusted = model.params["treated"]
    return dict(
        n_wells=int(len(nv)),
        n_untreated=int((nv.treated == 0).sum()),
        n_treated=int((nv.treated == 1).sum()),
        mean_w_hi_untreated=mean_untr,
        mean_w_hi_treated=mean_treated,
        unadjusted_diff=unadjusted,
        adjusted_diff=adjusted,
        attributed_to_recovery=unadjusted - adjusted,
        p_value=model.pvalues["treated"],
        residual_df=int(model.df_resid),
    )


def assert_published(fold_table, occupancy):
    """Fail loudly if any published caption value has drifted."""
    f = fold_table.set_index("comparison")
    checks = [
        ("S9 Fig 4b all cells",      f.iloc[0].fold_all_cells,    10.5,  0.05),
        ("S9 Fig 4b well-tracked",   f.iloc[0].fold_well_tracked, 11.3,  0.05),
        ("S9 Fig 4c 0h all cells",   f.iloc[1].fold_all_cells,    10.7,  0.05),
        ("S9 Fig 4c 0h well-track",  f.iloc[1].fold_well_tracked, 13.2,  0.05),
        ("S9 Fig 4c -48h all cells", f.iloc[2].fold_all_cells,     9.4,  0.05),
        ("S9 Fig 4c -48h well-trk",  f.iloc[2].fold_well_tracked,  8.0,  0.05),
        ("S10 unadjusted diff",      occupancy["unadjusted_diff"], 0.30, 0.005),
        ("S10 adjusted diff",        occupancy["adjusted_diff"],   0.25, 0.005),
        ("S10 p value",              occupancy["p_value"],       0.0005, 0.00005),
    ]
    bad = [(n, got, want) for n, got, want, tol in checks if abs(got - want) > tol]
    for n, got, want, tol in checks:
        print(f"  {'OK  ' if abs(got-want) <= tol else 'DRIFT'}  {n:26s} {got:9.5f}  (published {want})")
    if bad:
        raise AssertionError(f"{len(bad)} published value(s) no longer reproduce: {bad}")
    return True


if __name__ == "__main__":
    cells, fits = load()
    folds = fig9_fold_reductions(cells, fits)
    occ = fig10_occupancy_adjustment(cells, fits)

    print("Supplementary Figure 9 -- fold reductions in mean per-cell MSD")
    print("(pooled across each condition's cells; wellwise column shown for contrast)\n")
    print(folds.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("\nmedian displacement pairs by arm:", fig9_track_length_summary(cells))

    print("\nSupplementary Figure 10 -- naive high-state occupancy, recovery-adjusted")
    for k, v in occ.items():
        print(f"  {k:26s} {v}")

    print("\nVerifying against published caption values:")
    assert_published(folds, occ)
    print("\nAll published values reproduce.")
