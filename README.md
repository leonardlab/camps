# camps

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22089114.svg)](https://doi.org/10.5281/zenodo.22089114)

Analysis software for the CAMP cell-motility study.

Cell detection and tracking from timelapse movies, per-cell mean squared
displacement (MSD), and the code that produced the published figures.

Raw movies and spot tables are archived separately in a Zenodo repository. The
derived tables the figure code needs are included here, so the figures rebuild
from a fresh clone.

## Code behind each figure

| Paper element | Code |
|---|---|
| Detection and tracking | `pipeline/stage1_trackmate.py` |
| Per-cell MSD | `pipeline/stage2_compute_msd.py` |
| Figure 4b | `figures/code/build_fig4b.py` |
| Figure 4c | `figures/code/build_fig4c.py` |
| Figure 6b | `figures/code/build_fig6b.py` |
| Supplementary Figure 8 | `figures/code/build_figS8.py` |
| Supplementary Figure 9 | `figures/code/build_figS9.py` |
| Supplementary Figure 10 | `figures/code/build_figS10.py` |
| Supplementary Table S2 | `figures/panel_data/tableS2_significance.csv` |

## Pipeline

Requires Fiji with TrackMate, and Python 3 with numpy and pandas.

```
cd pipeline
make PROFILE=phase     # phase contrast: density and PMA experiments
make PROFILE=gfp       # GFP: doxycycline titration
make print             # show all settings for the selected profile
```

The two imaging channels use different detector and tracker settings, so a
profile must be selected; `make print` shows the resolved values. If Fiji is
not on the default search path, pass `FIJI=/path/to/fiji`, or run
`make install-fiji`.

## Figures

```
conda env create -f environment.yml
conda activate c04-panels
cd figures/code
PYTHONPATH=. python build_fig4b.py
```

Rebuilds are written to `figures/rebuilt/`; the published renders are in
`figures/output/`. Each builder prints a text-overlap count, where zero is the
pass condition.

`build_fig4b.py` and `build_fig4c.py` refit the significance model from the
per-well data rather than reading stored values, so running them also
reproduces `figures/panel_data/tableS2_significance.csv`.

`si_confound_checks.py` recomputes the values quoted in the Supplementary
Figure 9 and 10 captions and exits non-zero if any of them fail to reproduce.

Figures in the paper were assembled and typeset from these renders, so a
rebuild reproduces the data, statistics and panel geometry but not the final
typography.

## Layout

```
pipeline/     movies to tracks to per-cell MSD
figures/
  code/       figure builders and shared plotting code
  data/       derived inputs used by the builders
  panel_data/ per-panel values and the published tables
  output/     the published figure renders
docs/         software versions used for the published analysis
```

## Notes

Significance uses a pooled-error model: one within-condition variance estimated
from all eight wells of a subplot, giving four residual degrees of freedom, with
Benjamini-Hochberg correction within each panel. An earlier Welch-based analysis
is retained at `figures/panel_data/supporting/` for comparison only.

Pixel size and MSD lag differ by channel: 0.68626 um/px and 3 frames for phase
contrast, 1.02939 um/px and 5 frames for GFP.

PMA doses are in ng/mL.

## Citation and archived versions

Each release is archived on Zenodo. Cite the DOI for the specific version you
used; the concept DOI always resolves to the most recent release.

| | DOI |
|---|---|
| v1.0.0 (published with the study) | [10.5281/zenodo.22089115](https://doi.org/10.5281/zenodo.22089115) |
| All versions (concept) | [10.5281/zenodo.22089114](https://doi.org/10.5281/zenodo.22089114) |

The supporting data are archived separately at
[10.5281/zenodo.22073097](https://doi.org/10.5281/zenodo.22073097).

See `CITATION.cff` for author and citation metadata.

## License

MIT. See `LICENSE`.
