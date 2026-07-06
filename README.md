# Wi2SAR Artifact Package

This directory contains the executable artifact package for the Wi2SAR
submission. The package is self-contained for the supported data-backed
figures: it includes package-local CSV inputs, the direction-finding helpers,
the visualization functions, reference images, and reproduced outputs.

The primary entry point is:

```bash
jupyter nbconvert --to notebook --execute --inplace mobicom26_artifact_evaluation.ipynb
```

The notebook calls Python functions directly. It does not require command-line
wrappers for individual figures.

## Directory Layout

- `mobicom26_artifact_evaluation.ipynb`: one-click executable notebook.
- `algorithm/`: direction-finding CLI, algorithm modules, example beam-pattern
  data, and tests.
- `algorithm/main.py`: offline and real-time direction-finding CLI entry.
- `algorithm/module/`: direction-finding implementation.
- `algorithm/data/case5-3d/`: example layout, beam-pattern RSSI table, and test
  RSSI input used by the CLI.
- `data/`: package-local CSV inputs for all supported figures.
- `viz/`: plotting functions. Each supported figure is exposed as a Python
  function in `viz/charts.py`.
- `frontend/reference_images/`: manuscript reference images used in notebook
  markdown cells for visual comparison.
- `output/`: reproduced figures written by the notebook and plotting
  functions.
- `requirements.txt`: Python package requirements.

## Environment

Use Python 3 with the packages listed in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

No external data download is required for the notebook. The CSV files under
`data/` are the package inputs used by the executable sections.

The direction-finding CLI can also be run from `algorithm/`:

```bash
cd algorithm
python main.py --log INFO --data-dir ./data/case5-3d --pos-elevation --calibration --no-plot
```

## One-Click Reproduction

Run from this directory:

```bash
jupyter nbconvert --to notebook --execute --inplace mobicom26_artifact_evaluation.ipynb
```

The notebook executes in place and leaves the rendered outputs in the notebook.
Figure files are also written under `output/`. Existing files in `output/` may
be replaced by newly computed versions.

## Supported Results

The executable notebook focuses on the data-backed results below.

- Figure 4(b): measured Luneburg Lens beam pattern from an RSSI sweep, plus an
  inline interactive Plotly sphere generated from the same beam-pattern data.
- Figure 10(b-c): Luneburg Lens RSS gain and victim-discovery range.
- Figure 11(a-h): direction-finding score summaries, parameter studies, and
  the ArrayTrack comparison.
- Figure 12(a-c): trajectory visualizations from flight-log CSVs, including
  interactive 3D views for the supported search trajectories.

Reference-only figures are shown only where they help compare a reproduced
result with the manuscript figure. System diagrams and other figures without a
data-backed executable reproduction are not part of the one-click notebook.

## Data Inputs

The notebook uses package-local CSV files:

- `data/fig04_ll_beam_pattern/`: beam-pattern RSSI sweep and interpolated
  beam-pattern table.
- `data/fig10_ll_effectiveness/`: RSS gain and VDM working-range summaries.
- `data/fig11_direction_finding/`: direction-finding dataset and antenna
  layouts.
- `data/fig11h_arraytrack_comparison/`: ArrayTrack comparison errors.
- `data/fig12_trajectory/`: flight-log trajectory extracts and metadata.

These files are treated as inputs. Reproduction code reads from `data/` and
writes reproduced figures to `output/`.

## Visualization API

The notebook imports `viz.charts` and calls figure-specific functions. The main
entry points and helpers are:

- `fig04_beam_pattern(...)`
- `load_grid(...)` and `build_surface(...)` for the interactive Figure 4(b)
  beam-pattern sphere
- `fig10_ll_effectiveness(...)`
- `fig11_direction_finding(...)`
- `fig12_trajectory(...)`

The direction-finding CLI and algorithm modules are kept in `algorithm/`. The
figure-specific statistics and plotting helpers live in `viz/charts.py` so the
notebook can show data loading, intermediate processing, and visualization as
separate steps.
