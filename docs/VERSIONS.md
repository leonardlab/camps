# Software versions

Software versions used to produce the published results.

## Tracking and per-cell MSD

| | |
|---|---|
| macOS | 26.5.2 (arm64) |
| ImageJ | 1.54p |
| TrackMate | 8.1.6 |
| Jython | 2.7.4 |
| Java | 21.0.7 (Azul Zulu) |
| Python | 3.11.4 |
| numpy | 2.4.6 |
| pandas | 2.3.3 |

## Figures and statistics

| | |
|---|---|
| Python | 3.11.15 (conda-forge) |
| numpy | 2.4.6 |
| pandas | 2.3.3 |
| scipy | 1.17.1 |
| statsmodels | 0.14.6 |
| scikit-learn | 1.9.0 |
| matplotlib | 3.11.0 |

Reproduced by `environment.yml`. **This is not a lock file.** Direct
dependencies are pinned to the versions actually used, which is enough to
rebuild a working environment, but transitive dependencies are unpinned and
there is no platform or build-hash pinning. Figure output is not expected to be
bit-identical across matplotlib patch versions.

## TrackMate settings

The two imaging channels used different settings. Both were read directly from
the TrackMate XML files that produced the published data.

| | phase contrast | GFP |
|---|---|---|
| datasets | density, PMA | doxycycline titration |
| detector | LoG | LoG |
| target channel | 1 | 1 |
| radius | 18.0 px | 10.0 px |
| threshold | 0.2 | 0.2 |
| median filtering | on | on |
| sub-pixel localization | on | on |
| initial quality filter | 0.0 | 0.0 |
| tracker | Sparse LAP | Sparse LAP |
| linking max distance | 46.0 px | 25.0 px |
| gap-closing max distance | 55.0 px | 30.0 px |
| max frame gap | 2 | 2 |
| splitting / merging | off / off | off / off |
| pixel size | 0.68626 um/px | 1.02939 um/px |
| frame interval | 180 s | 120 s |
| MSD lag | 3 frames (540 s) | 5 frames (600 s) |

These are the defaults in `pipeline/Makefile`, selected with `PROFILE`.
