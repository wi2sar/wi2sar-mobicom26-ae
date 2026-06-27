"""Reusable figure-generation functions for the Wi2SAR notebook.

The public figure functions in this module are the importable one-click entry
points used by ``mobicom26_artifact_evaluation.ipynb``. They read only
package-local CSV files under ``data/`` and write regenerated artifacts under
``output/``.
"""
# ruff: noqa: E402,F811

from __future__ import annotations



# ---- Figure 4 static beam pattern ----
"""Recompute Fig.04(b) measured Luneburg Lens beam pattern from package CSV files."""


import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def bp_form_cyclic_matrix(old_matrix: np.ndarray) -> np.ndarray:
    new_matrix = np.zeros((old_matrix.shape[0], old_matrix.shape[1] * 2 - 2))
    new_matrix[:, : old_matrix.shape[1]] = old_matrix
    old_matrix_to_cat = old_matrix[:, 1:-1]
    old_matrix_to_cat_roll_a = np.roll(old_matrix_to_cat, 180, axis=0)
    old_matrix_to_cat_roll_a_flip = np.flip(old_matrix_to_cat_roll_a, axis=1)
    new_matrix[:, old_matrix.shape[1] :] = old_matrix_to_cat_roll_a_flip
    return new_matrix


def smooth(matrix: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    try:
        from scipy.ndimage import gaussian_filter

        return gaussian_filter(matrix, sigma=sigma)
    except Exception:
        return matrix


def load_interpolated_matrix(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    required = {"azimuth", "elevation", "response"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"missing required interpolated columns: {sorted(missing)}")
    matrix = np.zeros((360, 181))
    for row in df.itertuples(index=False):
        azimuth = int(getattr(row, "azimuth"))
        elevation = int(getattr(row, "elevation"))
        if 0 <= azimuth < 360 and -90 <= elevation <= 90:
            matrix[azimuth, elevation + 90] = float(getattr(row, "response"))
    return matrix


def plot_original_style_beam_pattern_matrix(matrix: np.ndarray) -> plt.Figure:
    beam_pattern_avg_array_cyclic = smooth(bp_form_cyclic_matrix(matrix), sigma=3)
    fig, ax = plt.subplots(figsize=(4, 4))
    plt.rcParams["font.sans-serif"] = ["Arial"]
    ax.imshow(beam_pattern_avg_array_cyclic.T, cmap="viridis", interpolation="nearest")
    ax.set_xlabel("Azimuth", fontsize=12)
    ax.set_ylabel("Elevation", fontsize=12)
    ax.set_yticks([0, 90, 180, 270, 359], ["-90", "0", "90", "0", "-90"], fontsize=12)
    ax.yaxis.label.set_fontsize(12)
    ax.yaxis.set_tick_params(labelcolor="red")
    ax.yaxis.get_ticklabels()[2].set_color("black")
    ax.yaxis.get_ticklabels()[1].set_color("mediumblue")
    ax.yaxis.get_ticklabels()[0].set_color("mediumblue")
    ax.set_xlim(0, 359)
    ax.set_xticks([0, 90, 180, 270, 359], ["180", "-90", "0", "90", "180"], fontsize=12, color="red")
    ax2 = ax.twiny()
    ax2.set_xlim(0, 359)
    ax2.set_xticks([0, 90, 180, 270, 359], ["0", "90", "180", "-90", "0"], fontsize=12, color="mediumblue")
    return fig


def save_original_style_beam_pattern(matrix: np.ndarray, out_base: Path) -> None:
    fig = plot_original_style_beam_pattern_matrix(matrix)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", dpi=300)
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def summarize_raw_sweep(raw_csv: Path, out_dir: Path) -> None:
    df = pd.read_csv(raw_csv)
    required = {"angle_deg", "trial", "stat", "channel", "rssi"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"missing required raw columns: {sorted(missing)}")
    df["antenna"] = df["stat"].astype(str) + df["channel"].astype(str)
    summary = (
        df.groupby(["angle_deg", "antenna"], as_index=False)["rssi"]
        .median()
        .rename(columns={"rssi": "median_rssi"})
    )
    summary.to_csv(out_dir / "fig04b_measured_beam_summary.csv", index=False)

    angle_profile = summary.groupby("angle_deg", as_index=False)["median_rssi"].max()
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    ax.plot(angle_profile["angle_deg"], angle_profile["median_rssi"], marker="o", linewidth=1.6)
    ax.set_xlabel("Incident angle (deg)")
    ax.set_ylabel("Peak median RSS")
    ax.grid(True, linestyle="--", alpha=0.45)
    fig.tight_layout()
    fig.savefig(out_dir / "fig04b_peak_response_by_angle.png", dpi=220)
    fig.savefig(out_dir / "fig04b_peak_response_by_angle.pdf")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="fig04_ll_rssi_sweep.csv")
    parser.add_argument("--interpolated", type=Path, default=None, help="fig04_interpolated_beam_pattern.csv")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    interpolated = args.interpolated or (args.data.parent / "fig04_interpolated_beam_pattern.csv")
    if not interpolated.exists():
        raise SystemExit(f"missing interpolated beam-pattern CSV: {interpolated}")
    matrix = load_interpolated_matrix(interpolated)
    save_original_style_beam_pattern(matrix, args.out / "fig04b_measured_ll_beam_pattern")
    summarize_raw_sweep(args.data, args.out)
    print(f"wrote Fig.04 outputs to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



# ---- Figure 4 interactive beam pattern ----
"""Generate the interactive 3D sphere view for Fig.04(b).

The input is the package CSV containing the measured interpolated beam-pattern
data. The function does not read from or mutate experiment directories.
"""


import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def load_grid(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    required = {"azimuth", "elevation", "response"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"missing required columns in {csv_path}: {sorted(missing)}")

    grid = (
        df.pivot_table(index="elevation", columns="azimuth", values="response", aggfunc="mean")
        .sort_index()
        .sort_index(axis=1)
    )
    azimuth = grid.columns.to_numpy(dtype=float)
    elevation = grid.index.to_numpy(dtype=float)
    response = grid.to_numpy(dtype=float)
    if np.isnan(response).any():
        response = pd.DataFrame(response).interpolate(axis=0).interpolate(axis=1).to_numpy()
    return azimuth, elevation, response


def squared_relative_energy(response: np.ndarray) -> np.ndarray:
    relative = np.maximum(response - float(np.nanmin(response)), 0.0)
    return relative**2


def build_surface(azimuth_deg: np.ndarray, elevation_deg: np.ndarray, response: np.ndarray) -> go.Figure:
    display_azimuth_deg = azimuth_deg + 90.0
    azimuth = np.deg2rad(display_azimuth_deg)
    elevation = np.deg2rad(elevation_deg)
    az_grid, el_grid = np.meshgrid(azimuth, elevation)
    custom_az_grid, custom_el_grid = np.meshgrid(display_azimuth_deg, elevation_deg)
    energy = squared_relative_energy(response)

    x = np.cos(el_grid) * np.cos(az_grid)
    y = np.cos(el_grid) * np.sin(az_grid)
    z = np.sin(el_grid)

    fig = go.Figure(
        data=[
            go.Surface(
                x=x,
                y=y,
                z=z,
                surfacecolor=energy,
                colorscale="Jet",
                cmin=float(np.nanmin(energy)),
                cmax=float(np.nanmax(energy)),
                colorbar={
                    "title": "squared rel. energy",
                    "len": 0.72,
                    "thickness": 14,
                },
                hovertemplate=(
                    "visualized longitude=%{customdata[0]:.0f} deg<br>"
                    "elevation=%{customdata[1]:.0f} deg<br>"
                    "response=%{customdata[2]:.2f}<br>"
                    "squared relative energy=%{surfacecolor:.2f}<extra></extra>"
                ),
                customdata=np.dstack((custom_az_grid, custom_el_grid, response)),
                lighting={"ambient": 0.72, "diffuse": 0.7, "specular": 0.18, "roughness": 0.78},
                showscale=True,
            )
        ]
    )
    fig.update_layout(
        title="Fig.04(b) measured LL beam pattern, interactive sphere",
        margin={"l": 0, "r": 0, "t": 48, "b": 0},
        paper_bgcolor="white",
        scene={
            "aspectmode": "data",
            "xaxis": {"title": "x", "showgrid": False, "zeroline": False},
            "yaxis": {"title": "y", "showgrid": False, "zeroline": False},
            "zaxis": {"title": "z", "showgrid": False, "zeroline": False},
            "camera": {"eye": {"x": 1.55, "y": -1.85, "z": 1.05}},
        },
        font={"family": "Arial, sans-serif", "size": 12},
    )
    return fig


def save_beam_pattern_sphere_preview(
    azimuth_deg: np.ndarray,
    elevation_deg: np.ndarray,
    response: np.ndarray,
    out_path: Path,
) -> None:
    """Save a static 3D preview of the same squared-energy sphere used by Plotly."""

    display_azimuth_deg = azimuth_deg + 90.0
    azimuth = np.deg2rad(display_azimuth_deg)
    elevation = np.deg2rad(elevation_deg)
    az_grid, el_grid = np.meshgrid(azimuth, elevation)
    energy = squared_relative_energy(response)

    x = np.cos(el_grid) * np.cos(az_grid)
    y = np.cos(el_grid) * np.sin(az_grid)
    z = np.sin(el_grid)

    norm = plt.Normalize(vmin=float(np.nanmin(energy)), vmax=float(np.nanmax(energy)))
    colors = plt.cm.jet(norm(energy))

    fig = plt.figure(figsize=(6.2, 5.1))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(x, y, z, facecolors=colors, linewidth=0, antialiased=False, shade=False)
    ax.set_title("Fig.04(b) measured LL beam pattern, 3D preview", fontsize=10)
    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=20, azim=-55)

    mappable = plt.cm.ScalarMappable(norm=norm, cmap="jet")
    mappable.set_array(energy)
    colorbar = fig.colorbar(mappable, ax=ax, shrink=0.68, pad=0.03)
    colorbar.set_label("squared rel. energy")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interpolated", type=Path, required=True, help="fig04_interpolated_beam_pattern.csv")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    azimuth, elevation, response = load_grid(args.interpolated)
    preview_path = args.out / "fig04b_measured_ll_beam_pattern_3d.png"
    save_beam_pattern_sphere_preview(azimuth, elevation, response, preview_path)
    print(f"wrote {preview_path}")
    print("Render the interactive Plotly sphere directly in the notebook with fig04_beam_pattern_figures().")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



# ---- Figure 10 LL effectiveness ----
"""Recompute Fig.10(b,c) Luneburg-Lens effectiveness plots.

The numeric inputs are stored as public CSV tables so reviewers can inspect and
rerun the figures without depending on the original notebook state.
"""


import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_both(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), dpi=300, bbox_inches="tight")


def lens_label(value: str) -> str:
    return "w/ LL" if value == "with_ll" else "w/o LL"


def plot_rss_gain(csv_path: Path, out_base: Path | None = None) -> plt.Figure:
    df = pd.read_csv(csv_path)
    distances = np.array(sorted(df["distance_m"].unique()), dtype=float)

    def series(tx_power_dbm: int, lens: str, include_only: bool = False) -> tuple[np.ndarray, np.ndarray]:
        subset = df[(df["tx_power_dbm"] == tx_power_dbm) & (df["lens"] == lens)].sort_values("distance_m")
        if include_only:
            subset = subset[subset["used_for_fit"] == 1]
        return subset["distance_m"].to_numpy(dtype=float), subset["rssi_dbm"].to_numpy(dtype=float)

    def fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        return tuple(np.polyfit(np.log10(x), y, 1))

    d24_wo, y24_wo = series(24, "without_ll")
    d24_w, y24_w = series(24, "with_ll")
    d1_wo_all, y1_wo_all = series(1, "without_ll")
    d1_w, y1_w = series(1, "with_ll", include_only=True)
    d1_wo, y1_wo = series(1, "without_ll", include_only=True)

    a24_wo, b24_wo = fit_line(d24_wo, y24_wo)
    a24_w, b24_w = fit_line(d24_w, y24_w)
    a1_wo, b1_wo = fit_line(d1_wo, y1_wo)
    a1_w, b1_w = fit_line(d1_w, y1_w)

    d_fit = np.logspace(np.log10(distances.min()), np.log10(distances.max()), 200)
    logd_fit = np.log10(d_fit)

    fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
    ax.plot(d_fit, a24_w * logd_fit + b24_w, linestyle="-", color="red", label="24dBm w/ LL")
    ax.plot(d_fit, a24_wo * logd_fit + b24_wo, linestyle="-", color="black", label="24dBm w/o LL")
    ax.plot(d_fit, a1_w * logd_fit + b1_w, linestyle="--", color="red", label="1dBm  w/ LL")
    ax.plot(d_fit, a1_wo * logd_fit + b1_wo, linestyle="--", color="black", label="1dBm  w/o LL")

    ax.scatter(d24_wo, y24_wo, marker="o", color="black", edgecolors="none")
    ax.scatter(d24_w, y24_w, marker="s", color="red", edgecolors="none")
    ax.scatter(d1_wo, y1_wo, marker="^", color="black", edgecolors="none")
    ax.scatter(d1_w, y1_w, marker="D", color="red", edgecolors="none")

    x384 = 384.0
    y384_1wo = a1_wo * np.log10(x384) + b1_wo
    ax.scatter([x384], [y384_1wo], marker="x", color="blue", zorder=5, s=100)
    ax.text(480, -102, "signal lost", fontsize=14, ha="right", color="blue")
    ax.annotate(
        "~10dB gain",
        xy=(56, -47),
        xytext=(88, -33),
        color="red",
        fontsize=12,
        arrowprops=dict(arrowstyle="-|>", color="red", lw=1.5),
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75),
    )
    ax.annotate("", xy=(96, -53), xytext=(96, -62), arrowprops=dict(arrowstyle="-|>", color="red", lw=1.5))
    ax.annotate("", xy=(192, -60), xytext=(192, -72), arrowprops=dict(arrowstyle="-|>", color="red", lw=1.5))

    ax.set_xscale("log", base=2)
    ax.set_xticks(distances)
    ax.set_xticklabels([str(int(d)) for d in distances], fontsize=12)
    ax.set_xlabel("Distance (m)", fontsize=16)
    ax.set_ylabel("RSS (dBm)", fontsize=16, labelpad=-2)
    ax.set_yticks([-110, -100, -90, -80, -70, -60, -50, -40, -30])
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(axis="both", linestyle=":", linewidth=0.5)
    ax.legend(loc="lower left", fontsize=13)
    ax.set_ylim(-110, -25)
    fig.tight_layout()

    # Preserve a fit summary for AE inspection.
    summary = pd.DataFrame(
        [
            {"tx_power_dbm": 24, "lens": lens_label("without_ll"), "slope": a24_wo, "intercept": b24_wo},
            {"tx_power_dbm": 24, "lens": lens_label("with_ll"), "slope": a24_w, "intercept": b24_w},
            {"tx_power_dbm": 1, "lens": lens_label("without_ll"), "slope": a1_wo, "intercept": b1_wo},
            {"tx_power_dbm": 1, "lens": lens_label("with_ll"), "slope": a1_w, "intercept": b1_w},
        ]
    )
    if out_base is not None:
        save_both(fig, out_base)
        summary.to_csv(out_base.parent / "fig10b_rss_gain_ground_fit_summary.csv", index=False)
        plt.close(fig)
    return fig


def plot_vdm_range(csv_path: Path, out_base: Path | None = None) -> plt.Figure:
    df = pd.read_csv(csv_path)
    order = [
        ("LoS", 2.4),
        ("LoS", 5.0),
        ("NLoS", 2.4),
        ("NLoS", 5.0),
    ]
    categories = ["2.4GHz", "5GHz", "2.4GHz", "5GHz"]
    y = np.arange(len(order))
    height = 0.35

    def values(lens: str) -> list[float]:
        result = []
        for scenario, frequency in order:
            row = df[
                (df["scenario"] == scenario)
                & (np.isclose(df["frequency_ghz"].astype(float), frequency))
                & (df["lens"] == lens)
            ]
            result.append(float(row.iloc[0]["working_range_m"]))
        return result

    w_o_ll = values("without_ll")
    w_ll = values("with_ll")

    fig, ax = plt.subplots(figsize=(4, 4))
    bars1 = ax.barh(
        y + height / 2,
        w_o_ll,
        height,
        label="w/o LL",
        color="skyblue",
        hatch="//",
        linewidth=1,
        edgecolor="black",
    )
    bars2 = ax.barh(
        y - height / 2,
        w_ll,
        height,
        label="w/ LL",
        color="salmon",
        hatch="xx",
        linewidth=1,
        edgecolor="black",
    )

    ax.axhline(1.5, color="gray", linewidth=0.5, linestyle="--")
    ax.text(
        630,
        1.9,
        "NLoS",
        fontsize=16,
        color="k",
        ha="center",
        va="center",
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.5"),
    )
    ax.text(
        630,
        1.1,
        "LoS",
        fontsize=16,
        color="k",
        ha="center",
        va="center",
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.5"),
    )
    ax.set_xlabel("VDM Working Range (m)", fontsize=16)
    ax.set_yticks(y)
    ax.set_yticklabels(categories, fontsize=16, rotation=90, ha="center", va="center")
    ax.legend(fontsize=16)
    ax.set_xlim(0, 750)
    xticks_labels = [0, 100, 200, 300, 400, 500, 600, 700]
    ax.set_xticks(xticks_labels)
    ax.set_xticklabels(xticks_labels, fontsize=16)

    def add_labels(bars) -> None:
        for bar in bars:
            width = bar.get_width()
            ax.annotate(
                f"{width:.1f}",
                xy=(width, bar.get_y() + bar.get_height() / 2),
                xytext=(3, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=16,
            )

    add_labels(bars1)
    add_labels(bars2)
    fig.tight_layout()
    if out_base is not None:
        save_both(fig, out_base)
        plt.close(fig)
    return fig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    plot_rss_gain(args.data / "fig10_rss_gain_ground.csv", args.out / "fig10b_rss_gain_ground")
    plot_vdm_range(args.data / "fig10_vdm_working_range.csv", args.out / "fig10c_aerial_victim_discovery")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



# ---- Figure 12 trajectory panels ----
"""Recompute Fig.12(a-c) trajectory plots from package CSV files."""


import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LAT_TO_METERS = 111000.0
MAC_PHONE_MAP = {
    "IWCH_MAC": "IWCH",
    "IP06_MAC": "IP06",
    "IP15_MAC": "IP15",
    "RMN10_MAC": "RMN10",
    "IP11_MAC": "IP11",
    "IPAD_MAC": "IPAD",
    "IP13_MAC": "IP13",
    "HONR_MAC": "HONR",
    "RMN5_MAC": "RMN5",
    "OPPO_MAC": "OPPO",
    "MI10_MAC": "MI10",
    "IP16_MAC": "IP16",
    "IP15XY_MAC": "IP15XY",
    "DEFAULT_MAC": "DEFAULT",
}
PHONE_MAC_MAP = {name: mac for mac, name in MAC_PHONE_MAP.items()}

PANELS = [
    ("fig12a", "fig12a_zigzag.csv", "Controlled zigzag flight"),
    ("fig12b", "fig12b_large_area_search.csv", "Large-area exploratory search"),
    ("fig12c", "fig12c_full_wi2sar_trial.csv", "Full Wi2SAR trial"),
]
FIG12B_PHONES = ["IP15", "RMN10", "RMN5", "IPAD", "IP11"]
FIG12_PHONE_COLORS = {
    "IP15": "#d62728",
    "RMN10": "#1f77b4",
    "RMN5": "#2ca02c",
    "IPAD": "#9467bd",
    "IP11": "#ff7f0e",
}


def to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def latlon_to_meters(lat_diff: float, lon_diff: float, latitude: float) -> tuple[float, float]:
    lat_meters = lat_diff * LAT_TO_METERS
    lon_meters = lon_diff * LAT_TO_METERS * np.cos(np.radians(latitude))
    return lat_meters, lon_meters


def fix_coordinates(df: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    df = df.copy()
    for col in ["lat_deg", "lon_deg"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    valid = (df["lat_deg"].notna()) & (df["lon_deg"].notna()) & (df["lat_deg"] != 0) & (df["lon_deg"] != 0)
    if not valid.any():
        raise ValueError("no valid GPS coordinates")
    first_idx = valid.idxmax()
    takeoff_lat = float(df.loc[first_idx, "lat_deg"])
    takeoff_lon = float(df.loc[first_idx, "lon_deg"])
    df = df[valid].copy()
    df["lat_deg_diff"] = df["lat_deg"] - takeoff_lat
    df["lon_deg_diff"] = df["lon_deg"] - takeoff_lon
    lat_m = []
    lon_m = []
    for _, row in df.iterrows():
        lat_delta, lon_delta = latlon_to_meters(float(row["lat_deg_diff"]), float(row["lon_deg_diff"]), float(row["lat_deg"]))
        lat_m.append(lat_delta)
        lon_m.append(lon_delta)
    df["lat_m_diff"] = lat_m
    df["lon_m_diff"] = lon_m
    return df, takeoff_lat, takeoff_lon


def nonempty_float(value) -> float | None:
    if pd.isna(value) or value == "":
        return None
    return float(value)


def apply_metadata_filters(df: pd.DataFrame, meta_rows: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if meta_rows.empty:
        return out
    row = meta_rows.iloc[0]
    filter_cols = [
        ("alt_m_diff", "alt_m_diff_min", "alt_m_diff_max"),
        ("yaw", "yaw_min", "yaw_max"),
        ("gps_ts_ms", "gps_ts_ms_min", "gps_ts_ms_max"),
        ("roll", "roll_min", "roll_max"),
    ]
    for data_col, min_col, max_col in filter_cols:
        if data_col not in out.columns:
            continue
        out[data_col] = pd.to_numeric(out[data_col], errors="coerce")
        min_val = nonempty_float(row.get(min_col))
        max_val = nonempty_float(row.get(max_col))
        if min_val is not None:
            out = out[out[data_col] >= min_val]
        if max_val is not None:
            out = out[out[data_col] <= max_val]
    return out


def phone_positions(meta_rows: pd.DataFrame, takeoff_lat: float, takeoff_lon: float) -> dict[str, dict[str, float | str]]:
    phones = {}
    for _, row in meta_rows.iterrows():
        lat = float(row["lat"])
        lon = float(row["lon"])
        lat_m, lon_m = latlon_to_meters(lat - takeoff_lat, lon - takeoff_lon, lat)
        phones[str(row["phone_name"])] = {
            "lat_m_diff": lat_m,
            "lon_m_diff": lon_m,
            "pose": str(row.get("pose", "")),
        }
    return phones


def plot_original_style(
    ax: plt.Axes,
    data_pd: pd.DataFrame,
    meta_rows: pd.DataFrame,
    title: str,
    takeoff_lat: float,
    takeoff_lon: float,
    title_fontsize: int = 12,
    show_legend: bool = True,
) -> bool:
    phones = phone_positions(meta_rows, takeoff_lat, takeoff_lon)
    valid_macs = []
    for phone_name, phone_info in phones.items():
        mac = PHONE_MAC_MAP.get(phone_name)
        if mac:
            valid_macs.append(mac)
        ax.scatter(phone_info["lon_m_diff"], phone_info["lat_m_diff"], marker="x", c="r", s=100)
        ax.text(phone_info["lon_m_diff"], phone_info["lat_m_diff"], f"  {phone_name}", fontsize=8)

    data_pd = data_pd.copy()
    if valid_macs and "mac" in data_pd.columns:
        filtered = data_pd[data_pd["mac"].isin(valid_macs)].copy()
    elif "device_name" in data_pd.columns:
        filtered = data_pd[data_pd["device_name"].astype(str).isin(phones.keys())].copy()
    else:
        filtered = data_pd.copy()
    if filtered.empty:
        ax.text(0.5, 0.5, "No matching trajectory samples", ha="center", va="center", transform=ax.transAxes)
        return False

    path_df = data_pd.drop_duplicates(subset=["gps_ts_ms"]) if "gps_ts_ms" in data_pd.columns else data_pd
    ax.plot(path_df["lon_m_diff"], path_df["lat_m_diff"], c="k", alpha=0.5, label="Drone Trajectory")
    ax.scatter(filtered["lon_m_diff"], filtered["lat_m_diff"], s=6, c="gray", alpha=0.25, label="All Samples")

    head_width = float(meta_rows.iloc[0].get("head_width", 1) if not meta_rows.empty else 1)
    head_length = float(meta_rows.iloc[0].get("head_length", 2) if not meta_rows.empty else 2)
    max_arrows = 220
    step_size = max(1, len(filtered) // max_arrows)
    for i in range(0, len(filtered), step_size):
        row = filtered.iloc[i]
        if not np.isfinite(row.get("earth_azimuth", np.nan)) or not np.isfinite(row.get("yaw", np.nan)):
            continue
        fc = "r" if i != 0 else "k"
        ec = "r" if i != 0 else "k"
        alpha = 0.5 if i != 0 else 1.0
        ax.arrow(
            row["lon_m_diff"],
            row["lat_m_diff"],
            np.sin(np.radians(row["earth_azimuth"])) * head_length,
            np.cos(np.radians(row["earth_azimuth"])) * head_length,
            head_width=head_width,
            head_length=head_length,
            fc=fc,
            ec=ec,
            alpha=alpha,
        )
        ax.arrow(
            row["lon_m_diff"],
            row["lat_m_diff"],
            np.sin(np.radians(row["yaw"])) * head_length,
            np.cos(np.radians(row["yaw"])) * head_length,
            head_width=head_width,
            head_length=head_length,
            fc="y",
            ec="y",
            alpha=0.3,
        )

    valid_scores = pd.to_numeric(filtered.get("score"), errors="coerce").dropna()
    median_score = f"{float(valid_scores.median()):.3f}" if len(valid_scores) else "N/A"
    ax.set_xlabel("Longitude (m)")
    ax.set_ylabel("Latitude (m)")
    ax.set_title(f"{title} | median PR: {median_score} | samples: {len(valid_scores)}", fontsize=title_fontsize)
    ax.grid()
    if show_legend:
        ax.legend()
    ax.axis("equal")
    return True


def filtered_phone_samples(data_pd: pd.DataFrame, meta_rows: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows whose phone appears in the figure metadata."""

    phones = [str(phone) for phone in meta_rows.get("phone_name", pd.Series(dtype=str)).dropna().tolist()]
    if not phones:
        return data_pd.copy()
    valid_macs = [PHONE_MAC_MAP[phone] for phone in phones if phone in PHONE_MAC_MAP]
    if valid_macs and "mac" in data_pd.columns:
        return data_pd[data_pd["mac"].isin(valid_macs)].copy()
    if "device_name" in data_pd.columns:
        return data_pd[data_pd["device_name"].astype(str).isin(phones)].copy()
    return data_pd.iloc[0:0].copy()


def plot_static_3d_trajectory(
    ax,
    data_pd: pd.DataFrame,
    meta_rows: pd.DataFrame,
    title: str,
    takeoff_lat: float,
    takeoff_lon: float,
    title_fontsize: int = 12,
) -> bool:
    """Plot a 3D trajectory view for Fig.12(b) from CSV rows, not image files."""

    phones = phone_positions(meta_rows, takeoff_lat, takeoff_lon)
    phone_names = list(phones.keys())
    filtered = filtered_phone_samples(data_pd, meta_rows)
    if filtered.empty:
        ax.text2D(0.5, 0.5, "No matching trajectory samples", ha="center", va="center", transform=ax.transAxes)
        return False

    path_df = data_pd.drop_duplicates(subset=["gps_ts_ms"]) if "gps_ts_ms" in data_pd.columns else data_pd
    path_z = path_df["alt_m_diff"] if "alt_m_diff" in path_df.columns else pd.Series(0.0, index=path_df.index)
    ax.plot(path_df["lon_m_diff"], path_df["lat_m_diff"], path_z, c="black", alpha=0.55, linewidth=1.2, label="Drone Trajectory")

    for phone_name in phone_names:
        mac = PHONE_MAC_MAP.get(phone_name)
        phone_df = filtered[filtered["mac"] == mac].copy() if mac and "mac" in filtered.columns else filtered.iloc[0:0].copy()
        if phone_df.empty and "device_name" in filtered.columns:
            phone_df = filtered[filtered["device_name"].astype(str) == phone_name].copy()
        if phone_df.empty:
            continue
        color = FIG12_PHONE_COLORS.get(phone_name, "gray")
        z = phone_df["alt_m_diff"] if "alt_m_diff" in phone_df.columns else pd.Series(0.0, index=phone_df.index)
        ax.scatter(phone_df["lon_m_diff"], phone_df["lat_m_diff"], z, s=10, color=color, alpha=0.45, label=phone_name)

    for phone_name, phone_info in phones.items():
        color = FIG12_PHONE_COLORS.get(phone_name, "red")
        ax.scatter(float(phone_info["lon_m_diff"]), float(phone_info["lat_m_diff"]), 0.0, marker="x", color=color, s=80)
        ax.text(float(phone_info["lon_m_diff"]), float(phone_info["lat_m_diff"]), 0.0, f" {phone_name}", fontsize=8, color=color)

    head_length = float(meta_rows.iloc[0].get("head_length", 1) if not meta_rows.empty else 1)
    max_arrows = 120
    step_size = max(1, len(filtered) // max_arrows)
    for i in range(0, len(filtered), step_size):
        row = filtered.iloc[i]
        if not np.isfinite(row.get("earth_azimuth", np.nan)):
            continue
        elevation = -float(row.get("phone_rel_elevation", row.get("elevation", 0.0)))
        azimuth = float(row["earth_azimuth"])
        dx = np.sin(np.radians(azimuth)) * np.cos(np.radians(elevation)) * head_length
        dy = np.cos(np.radians(azimuth)) * np.cos(np.radians(elevation)) * head_length
        dz = np.sin(np.radians(elevation)) * head_length
        ax.quiver(
            row["lon_m_diff"],
            row["lat_m_diff"],
            row.get("alt_m_diff", 0.0),
            dx,
            dy,
            dz,
            color="red",
            alpha=0.55,
            length=1.0,
            normalize=False,
            linewidth=0.7,
        )

    valid_scores = pd.to_numeric(filtered.get("score"), errors="coerce").dropna()
    median_score = f"{float(valid_scores.median()):.3f}" if len(valid_scores) else "N/A"
    ax.set_xlabel("Longitude (m)")
    ax.set_ylabel("Latitude (m)")
    ax.set_zlabel("Altitude (m)")
    ax.set_title(f"{title} | median PR: {median_score} | samples: {len(valid_scores)}", fontsize=title_fontsize)
    ax.legend(loc="upper left", fontsize=8)
    return True


def plot_panel(ax, figure_id: str, panel_df: pd.DataFrame, meta_rows: pd.DataFrame, title: str, takeoff_lat: float, takeoff_lon: float, title_fontsize: int = 12) -> bool:
    return plot_static_3d_trajectory(ax, panel_df, meta_rows, title, takeoff_lat, takeoff_lon, title_fontsize=title_fontsize)


def prepare_panel_dataframe(df: pd.DataFrame, meta_rows: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    """Prepare a Fig.12 panel from a dataframe already in memory."""

    df = df.copy()
    numeric_cols = [
        "lat_deg",
        "lon_deg",
        "lat_m_diff",
        "lon_m_diff",
        "gps_ts_ms",
        "earth_azimuth",
        "yaw",
        "score",
        "alt_m_diff",
        "roll",
        "vel_abs",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df, takeoff_lat, takeoff_lon = fix_coordinates(df)
    df = apply_metadata_filters(df, meta_rows)
    return df, takeoff_lat, takeoff_lon


def load_panel(csv_path: Path, meta_rows: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except pd.errors.ParserError:
        # Some original log-derived CSV rows contain a NUL byte. Keep the CSV
        # unchanged and use pandas' Python parser to skip only unreadable rows.
        try:
            df = pd.read_csv(csv_path, engine="python", on_bad_lines="warn")
        except TypeError:
            df = pd.read_csv(csv_path, engine="python", error_bad_lines=False, warn_bad_lines=True)
    return prepare_panel_dataframe(df, meta_rows)


def load_logged_direction_panel(csv_path: Path, meta_rows: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    """Load Fig.12 CSV and keep logged direction estimates and scores."""

    try:
        raw = pd.read_csv(csv_path, low_memory=False)
    except pd.errors.ParserError:
        try:
            raw = pd.read_csv(csv_path, engine="python", on_bad_lines="warn")
        except TypeError:
            raw = pd.read_csv(csv_path, engine="python", error_bad_lines=False, warn_bad_lines=True)
    return prepare_panel_dataframe(logged_direction_columns(raw), meta_rows)


def save_both(fig: plt.Figure, base: Path) -> None:
    fig.tight_layout()
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), dpi=300, bbox_inches="tight")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=None)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    metadata_path = args.metadata or (args.data_root / "fig12_metadata.csv")
    metadata = pd.read_csv(metadata_path)
    root = args.data_root.resolve().parents[1]

    fig = plt.figure(figsize=(15, 4.8))
    axes = [fig.add_subplot(1, 3, idx + 1, projection="3d") for idx, _ in enumerate(PANELS)]
    for ax, (figure_id, filename, title) in zip(axes, PANELS):
        meta_rows = metadata[metadata["figure_id"] == figure_id].copy()
        panel_df, takeoff_lat, takeoff_lon = load_logged_direction_panel(args.data_root / filename, meta_rows)
        plot_panel(ax, figure_id, panel_df, meta_rows, figure_id, takeoff_lat, takeoff_lon, title_fontsize=9)

        stem = filename[:-4] if filename.endswith(".csv") else filename
        single_fig = plt.figure(figsize=(12, 10))
        single_ax = single_fig.add_subplot(1, 1, 1, projection="3d")
        plot_panel(single_ax, figure_id, panel_df, meta_rows, title, takeoff_lat, takeoff_lon, title_fontsize=12)
        save_both(single_fig, args.out / f"{figure_id}_{stem}")
        plt.close(single_fig)

    fig.tight_layout()
    save_both(fig, args.out / "fig12_abc_combined")
    plt.close(fig)
    print(f"wrote Fig.12 outputs to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



# ---- Figure 12 interactive trajectory ----
"""Generate the Fig.12(a) IP15-only interactive Plotly trajectory view."""


import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
except ImportError as exc:  # pragma: no cover - exercised only in missing envs
    raise SystemExit("plotly is required for this script. Install plotly or use the static Fig.12 outputs.") from exc

def direction_vectors(df: pd.DataFrame, step_size: int, arrow_length: float) -> tuple[list[float], ...]:
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    us: list[float] = []
    vs: list[float] = []
    ws: list[float] = []
    for i in range(0, len(df), step_size):
        row = df.iloc[i]
        azimuth = float(row.get("earth_azimuth", 0.0))
        if "phone_rel_elevation" in df.columns:
            elevation = -float(row.get("phone_rel_elevation", 0.0))
        else:
            elevation = -float(row.get("elevation", 0.0))
        xs.append(float(row["lon_m_diff"]))
        ys.append(float(row["lat_m_diff"]))
        zs.append(float(row.get("alt_m_diff", 0.0)))
        us.append(float(np.sin(np.radians(azimuth)) * np.cos(np.radians(elevation)) * arrow_length))
        vs.append(float(np.cos(np.radians(azimuth)) * np.cos(np.radians(elevation)) * arrow_length))
        ws.append(float(np.sin(np.radians(elevation)) * arrow_length))
    return xs, ys, zs, us, vs, ws


def yaw_vectors(df: pd.DataFrame, step_size: int, arrow_length: float) -> tuple[list[float], ...]:
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    us: list[float] = []
    vs: list[float] = []
    ws: list[float] = []
    for i in range(0, len(df), step_size):
        row = df.iloc[i]
        azimuth = float(row.get("yaw", 0.0))
        xs.append(float(row["lon_m_diff"]))
        ys.append(float(row["lat_m_diff"]))
        zs.append(float(row.get("alt_m_diff", 0.0)))
        us.append(float(np.sin(np.radians(azimuth)) * arrow_length))
        vs.append(float(np.cos(np.radians(azimuth)) * arrow_length))
        ws.append(0.0)
    return xs, ys, zs, us, vs, ws


def add_prediction_arrows(
    fig: go.Figure,
    vectors: tuple[list[float], ...],
    color: str,
    line_name: str,
    cone_name: str,
) -> None:
    """Draw Wi2SAR direction estimates as the Fig.12 line-plus-cone arrow style."""

    xs, ys, zs, us, vs, ws = vectors
    for i, (sx, sy, sz, ux, uy, uz) in enumerate(zip(xs, ys, zs, us, vs, ws)):
        ex, ey, ez = sx + ux, sy + uy, sz + uz
        fig.add_trace(go.Scatter3d(
            x=[sx, ex],
            y=[sy, ey],
            z=[sz, ez],
            mode="lines",
            line=dict(color=color, width=7),
            name=line_name if i == 0 else None,
            showlegend=(i == 0),
            hoverinfo="skip",
        ))
        fig.add_trace(go.Cone(
            x=[ex],
            y=[ey],
            z=[ez],
            u=[ux],
            v=[uy],
            w=[uz],
            anchor="tail",
            colorscale=[[0, color], [1, color]],
            showscale=False,
            sizemode="scaled",
            sizeref=0.5,
            name=cone_name if i == 0 else None,
            showlegend=(i == 0),
            opacity=0.9,
        ))


def metadata_float(meta_rows: pd.DataFrame, key: str, default: float) -> float:
    if meta_rows.empty:
        return default
    value = meta_rows.iloc[0].get(key, default)
    if pd.isna(value) or value == "":
        return default
    return float(value)


def metadata_int(meta_rows: pd.DataFrame, key: str, default: int | None = None) -> int | None:
    if meta_rows.empty or key not in meta_rows.columns:
        return default
    value = meta_rows.iloc[0].get(key)
    if pd.isna(value) or value == "":
        return default
    return int(value)


def arrow_step_size(df: pd.DataFrame, meta_rows: pd.DataFrame, default_max_arrows: int = 120) -> int:
    """Keep interactive Fig.12 arrows readable without dropping trajectory samples."""

    sample_step = metadata_int(meta_rows, "sample_step")
    if sample_step:
        return max(1, sample_step)
    max_arrows = metadata_int(meta_rows, "max_arrows", default_max_arrows) or default_max_arrows
    return max(1, int(np.ceil(len(df) / max_arrows)))


def build_figure(panel_df: pd.DataFrame, meta_rows: pd.DataFrame, takeoff_lat: float, takeoff_lon: float, phone_name: str) -> go.Figure:
    mac = PHONE_MAC_MAP.get(phone_name)
    if not mac:
        raise ValueError(f"unknown phone name: {phone_name}")
    filtered = panel_df[panel_df["mac"] == mac].copy() if "mac" in panel_df.columns else panel_df.iloc[0:0].copy()
    if filtered.empty:
        raise ValueError(f"no rows for phone {phone_name}")

    path_df = panel_df.drop_duplicates(subset=["gps_ts_ms"]) if "gps_ts_ms" in panel_df.columns else panel_df
    phones = phone_positions(meta_rows, takeoff_lat, takeoff_lon)
    phone = phones.get(phone_name)
    if phone is None:
        raise ValueError(f"no metadata position for phone {phone_name}")

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=path_df["lon_m_diff"],
        y=path_df["lat_m_diff"],
        z=path_df.get("alt_m_diff", pd.Series(0.0, index=path_df.index)),
        mode="lines",
        line=dict(color="black", width=3),
        name="Drone Trajectory",
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter3d(
        x=filtered["lon_m_diff"],
        y=filtered["lat_m_diff"],
        z=filtered.get("alt_m_diff", pd.Series(0.0, index=filtered.index)),
        mode="markers",
        marker=dict(size=2, color="gray", opacity=0.25),
        name="All Samples",
        hoverinfo="skip",
    ))

    head_length = metadata_float(meta_rows, "head_length", 2.0)
    step_size = arrow_step_size(filtered, meta_rows)
    vectors = direction_vectors(filtered, step_size, head_length)
    if vectors[0]:
        add_prediction_arrows(fig, vectors, "red", "Predicted Direction", "Predicted Direction (arrowheads)")

    xs, ys, zs, us, vs, ws = yaw_vectors(filtered, step_size, max(head_length * 0.8, 0.1))
    if xs:
        fig.add_trace(go.Cone(
            x=xs,
            y=ys,
            z=zs,
            u=us,
            v=vs,
            w=ws,
            anchor="tail",
            colorscale=[[0, "yellow"], [1, "yellow"]],
            showscale=False,
            sizemode="absolute",
            sizeref=max(0.1, head_length / 4.0),
            name="Drone Yaw",
            opacity=0.35,
        ))

    valid_scores = pd.to_numeric(filtered.get("score"), errors="coerce").dropna()
    mean_score = f"{float(valid_scores.mean()):.3f}" if len(valid_scores) else "N/A"
    fig.add_trace(
        go.Scatter3d(
            x=[float(phone["lon_m_diff"])],
            y=[float(phone["lat_m_diff"])],
            z=[0.0],
            mode="markers+text",
            marker=dict(size=6, color="red", symbol="x"),
            text=[phone_name],
            textposition="top center",
            name="Phone",
        )
    )
    fig.update_layout(
        title=f"3D Path with Predicted Directions - {phone_name} | mean projection rate: {mean_score}",
        margin=dict(l=0, r=0, t=40, b=0),
        scene=dict(
            xaxis_title="Longitude (m)",
            yaxis_title="Latitude (m)",
            zaxis_title="Altitude (m)",
            aspectmode="data",
        ),
        showlegend=True,
    )
    return fig


def build_multi_phone_figure(panel_df: pd.DataFrame, meta_rows: pd.DataFrame, takeoff_lat: float, takeoff_lon: float, phone_names: list[str]) -> go.Figure:
    """Build a notebook-native 3D trajectory view for multiple target phones."""

    filtered = filtered_phone_samples(panel_df, meta_rows)
    if filtered.empty:
        raise ValueError("no rows for requested phones")

    path_df = panel_df.drop_duplicates(subset=["gps_ts_ms"]) if "gps_ts_ms" in panel_df.columns else panel_df
    phones = phone_positions(meta_rows, takeoff_lat, takeoff_lon)
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=path_df["lon_m_diff"],
        y=path_df["lat_m_diff"],
        z=path_df.get("alt_m_diff", pd.Series(0.0, index=path_df.index)),
        mode="lines",
        line=dict(color="black", width=4),
        name="Drone Trajectory",
        hoverinfo="skip",
    ))

    for phone_name in phone_names:
        mac = PHONE_MAC_MAP.get(phone_name)
        phone_df = filtered[filtered["mac"] == mac].copy() if mac and "mac" in filtered.columns else filtered.iloc[0:0].copy()
        if phone_df.empty and "device_name" in filtered.columns:
            phone_df = filtered[filtered["device_name"].astype(str) == phone_name].copy()
        if phone_df.empty:
            continue
        color = FIG12_PHONE_COLORS.get(phone_name, "#888888")
        fig.add_trace(go.Scatter3d(
            x=phone_df["lon_m_diff"],
            y=phone_df["lat_m_diff"],
            z=phone_df.get("alt_m_diff", pd.Series(0.0, index=phone_df.index)),
            mode="markers",
            marker=dict(size=3, color=color, opacity=0.45),
            name=f"{phone_name} samples",
            hoverinfo="skip",
        ))

    for phone_name in phone_names:
        phone = phones.get(phone_name)
        if phone is None:
            continue
        color = FIG12_PHONE_COLORS.get(phone_name, "red")
        fig.add_trace(go.Scatter3d(
            x=[float(phone["lon_m_diff"])],
            y=[float(phone["lat_m_diff"])],
            z=[0.0],
            mode="markers+text",
            marker=dict(size=7, color=color, symbol="x"),
            text=[phone_name],
            textposition="top center",
            name=f"{phone_name} location",
        ))

    head_length = metadata_float(meta_rows, "head_length", 1.0)
    for phone_name in phone_names:
        mac = PHONE_MAC_MAP.get(phone_name)
        phone_df = filtered[filtered["mac"] == mac].copy() if mac and "mac" in filtered.columns else filtered.iloc[0:0].copy()
        if phone_df.empty and "device_name" in filtered.columns:
            phone_df = filtered[filtered["device_name"].astype(str) == phone_name].copy()
        if phone_df.empty:
            continue
        color = FIG12_PHONE_COLORS.get(phone_name, "#888888")
        step_size = arrow_step_size(phone_df, meta_rows)
        vectors = direction_vectors(phone_df, step_size, head_length)
        if vectors[0]:
            add_prediction_arrows(
                fig,
                vectors,
                color,
                f"{phone_name} predicted direction",
                f"{phone_name} direction arrowhead",
            )

    step_size = arrow_step_size(filtered, meta_rows)
    xs, ys, zs, us, vs, ws = yaw_vectors(filtered, step_size, max(head_length * 0.8, 0.1))
    if xs:
        fig.add_trace(go.Cone(
            x=xs,
            y=ys,
            z=zs,
            u=us,
            v=vs,
            w=ws,
            anchor="tail",
            colorscale=[[0, "yellow"], [1, "yellow"]],
            showscale=False,
            sizemode="absolute",
            sizeref=max(0.1, head_length / 4.0),
            name="Drone Yaw",
            opacity=0.32,
        ))

    valid_scores = pd.to_numeric(filtered.get("score"), errors="coerce").dropna()
    median_score = f"{float(valid_scores.median()):.3f}" if len(valid_scores) else "N/A"
    fig.update_layout(
        title=f"Figure 12(b) multi-device discovery | median PR: {median_score} | devices: {', '.join(phone_names)}",
        margin=dict(l=0, r=0, t=45, b=0),
        scene=dict(
            xaxis_title="Longitude (m)",
            yaxis_title="Latitude (m)",
            zaxis_title="Altitude (m)",
            aspectmode="data",
        ),
        showlegend=True,
    )
    return fig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--phone", default="IP15")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    metadata_path = args.metadata or (args.data_root / "fig12_metadata.csv")
    metadata = pd.read_csv(metadata_path)
    meta_rows = metadata[(metadata["figure_id"] == "fig12a") & (metadata["phone_name"] == args.phone)].copy()
    if meta_rows.empty:
        raise ValueError(f"no fig12a metadata row for {args.phone}")
    panel_df, takeoff_lat, takeoff_lon = load_panel(args.data_root / "fig12a_zigzag.csv", meta_rows)
    build_figure(panel_df, meta_rows, takeoff_lat, takeoff_lon, args.phone)
    print("Render the interactive Plotly trajectory directly in the notebook with fig12_trajectory_figures().")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



# ---- Figure 11 ArrayTrack comparison ----
"""Recompute Fig.11(h) ArrayTrack comparison from the package CSV."""


import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({"font.size": 12, "font.family": "Arial", "pdf.fonttype": 42, "ps.fonttype": 42})

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


LABEL_SIZE = 14
COLOR_LL = "red"
COLOR_AT_CONT = "black"
COLOR_AT_ONE = "black"


def x_at_ecdf_p(sorted_vals: np.ndarray, p: float) -> float:
    n = sorted_vals.size
    y = np.arange(1, n + 1, dtype=float) / float(n)
    if p <= y[0]:
        return float(sorted_vals[0])
    idx = int(np.searchsorted(y, p, side="left"))
    if idx >= n:
        return float(sorted_vals[-1])
    if y[idx] == p:
        return float(sorted_vals[idx])
    x0 = float(sorted_vals[idx - 1])
    x1 = float(sorted_vals[idx])
    y0 = float(y[idx - 1])
    y1 = float(y[idx])
    return x0 + (p - y0) / (y1 - y0) * (x1 - x0)


def series_values(df: pd.DataFrame, name: str) -> np.ndarray:
    values = pd.to_numeric(df.loc[df["series"] == name, "angular_error_deg"], errors="coerce").dropna().to_numpy(dtype=float)
    return values[np.isfinite(values)]


def plot_cdf(df: pd.DataFrame, out_base: Path | None = None) -> plt.Figure:
    series = [
        ("Wi2SAR (2D)", COLOR_LL, "-", r"Wi$^{2}$SAR (2D)", 10, 0.5),
        ("ArrayTrack-Cont", COLOR_AT_CONT, "-", "ArrayTrack-Cont", 10, 0.4),
        ("ArrayTrack-One", COLOR_AT_ONE, "--", "ArrayTrack-One", None, 0.4),
    ]

    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    max_x = 5.0
    for name, color, linestyle, label, fixed_text_x, text_y in series:
        vals = series_values(df, name)
        if vals.size == 0:
            continue
        sorted_vals = np.sort(vals)
        cdf = np.arange(1, sorted_vals.size + 1) / float(sorted_vals.size)
        ax.plot(sorted_vals, cdf, color=color, linewidth=2.0, linestyle=linestyle, label=label)
        x_half = x_at_ecdf_p(sorted_vals, 0.5)
        ax.scatter(x_half, 0.5, color=color, marker="o", s=10)
        ax.vlines(x_half, 0.0, 0.5, colors=color, linestyles="--", linewidth=1.2)
        text_x = float(fixed_text_x) if fixed_text_x is not None else x_half + 0.2
        ax.text(text_x, text_y, f"{x_half:.1f}°", rotation=0, va="bottom", ha="left", color=color, fontsize=LABEL_SIZE - 2)
        max_x = max(max_x, float(np.nanmax(sorted_vals)))

    max_x = max(5.0, math.ceil(max_x / 5.0) * 5.0)
    ax.set_xlim(0.0, max_x)
    ax.set_ylim(0.0, 1.0)
    ax.hlines(0.5, 0.0, max_x, colors="black", linestyles="--", linewidth=1.0)
    ax.text(75, 0.5, "median", rotation=0, va="bottom", ha="center", color="black")
    ax.set_xlabel("Angular Error (deg)", fontsize=LABEL_SIZE)
    ax.set_ylabel("Empirical CDF", fontsize=LABEL_SIZE)
    ax.set_yticks([0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0])
    ax.grid(True, alpha=1, linestyle="--", zorder=0)
    ax.legend(loc="lower right")
    plt.tight_layout()
    if out_base is not None:
        out_base.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_base.with_suffix(".png"), dpi=220, bbox_inches="tight")
        fig.savefig(out_base.with_suffix(".pdf"), dpi=220, bbox_inches="tight", format="pdf")
        plt.close(fig)
    return fig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.data)
    plot_cdf(df, args.out / "fig11h_ll_vs_arraytrack_cdf")
    print(f"wrote Fig.11(h) output to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



# ---- Figure 11 direction-finding panels ----
"""Recompute Fig.11(a-g) from the package CSV dataset."""


import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams.update({"font.size": 12, "font.family": "Arial"})

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

AE_ROOT = Path(__file__).resolve().parents[1]
if str(AE_ROOT) not in sys.path:
    sys.path.insert(0, str(AE_ROOT))


DEFAULT_FONT_SIZE = 12
MEDIUM_FONT_SIZE = DEFAULT_FONT_SIZE - 2
LABEL_SIZE = DEFAULT_FONT_SIZE + 2
BAR_COLOR = "#97cce8"
UNDERFLOW_BAR_COLOR = "#c6dbef"
OVERFLOW_BAR_COLOR = "#6baed6"
CI_COLOR = "black"
CI_ALPHA = 0.8
FIG11_DISTANCE_COLLECTION_IDS = {
    "60m1-real_time_20240829_152601",
    "60m2-real_time_20240829_145600",
    "125m1-real_time_20240829_153805",
    "125m2-real_time_20240829_153805",
    "high-real_time_20240829_162426",
    "far5g_ip15-real_time_20240916_095543",
    "far5g_ipad-real_time_20240916_095543",
}

CMP_COLOR_LIST_LIGHT = [
    "#D6E4F0",
    "#D9E2CF",
    "#F1E0D6",
    "#F7E5B7",
    "#A8C0D6",
    "#D2C6D6",
    "#E6B8B0",
    "#C8D2B8",
    "#A2BBC8",
    "#F7F3E8",
    "#EBC5B8",
    "#C0C0C0",
]


def save_both(fig: plt.Figure, base: Path, dpi: int = 220, bbox_inches: str | None = "tight") -> None:
    kwargs = {"dpi": dpi}
    if bbox_inches is not None:
        kwargs["bbox_inches"] = bbox_inches
    fig.savefig(base.with_suffix(".pdf"), format="pdf", **kwargs)
    fig.savefig(base.with_suffix(".png"), **kwargs)


def numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def average_rssi(df: pd.DataFrame) -> pd.Series:
    rssi_cols = [col for col in df.columns if col.startswith("rssi_")]
    if not rssi_cols:
        return pd.Series(np.nan, index=df.index)
    return df[rssi_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)


def azimuth_error_deg(true_azimuth_deg: pd.Series, estimated_azimuth_deg: pd.Series) -> pd.Series:
    true_az = pd.to_numeric(true_azimuth_deg, errors="coerce")
    est_az = pd.to_numeric(estimated_azimuth_deg, errors="coerce")
    return ((est_az - true_az + 180.0) % 360.0 - 180.0).abs()


def angular_error_deg(
    true_azimuth_deg: pd.Series,
    true_elevation_deg: pd.Series,
    estimated_azimuth_deg: pd.Series,
    estimated_elevation_deg: pd.Series,
) -> pd.Series:
    true_az = np.deg2rad(pd.to_numeric(true_azimuth_deg, errors="coerce"))
    true_el = np.deg2rad(pd.to_numeric(true_elevation_deg, errors="coerce"))
    est_az = np.deg2rad(pd.to_numeric(estimated_azimuth_deg, errors="coerce"))
    est_el = np.deg2rad(pd.to_numeric(estimated_elevation_deg, errors="coerce"))

    true_x = np.cos(true_el) * np.cos(true_az)
    true_y = np.cos(true_el) * np.sin(true_az)
    true_z = np.sin(true_el)
    est_x = np.cos(est_el) * np.cos(est_az)
    est_y = np.cos(est_el) * np.sin(est_az)
    est_z = np.sin(est_el)
    dot = np.clip(true_x * est_x + true_y * est_y + true_z * est_z, -1.0, 1.0)
    return pd.Series(np.rad2deg(np.arccos(dot)), index=true_azimuth_deg.index)


def bootstrap_ci_stat(values: pd.Series | np.ndarray, agg: str = "median", n_boot: int = 500, seed: int = 7) -> tuple[float, float]:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if len(arr) == 0:
        return np.nan, np.nan
    if len(arr) == 1:
        return float(arr[0]), float(arr[0])
    rng = np.random.default_rng(seed)
    samples = rng.choice(arr, size=(n_boot, len(arr)), replace=True)
    if agg == "mean":
        stats = np.mean(samples, axis=1)
    else:
        stats = np.median(samples, axis=1)
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def validation_counts(df: pd.DataFrame) -> pd.DataFrame:
    checks = {
        "rows": pd.Series(True, index=df.index),
        "score": numeric(df, "score").notna(),
        "logged_azimuth": numeric(df, "azimuth").notna(),
        "logged_elevation": numeric(df, "elevation").notna(),
        "earth_azimuth": numeric(df, "earth_azimuth").notna(),
        "phone_relative_angle": numeric(df, "phone_rel_azimuth").notna() & numeric(df, "phone_rel_elevation").notna(),
        "distance": numeric(df, "distance").notna(),
        "velocity": numeric(df, "vel_abs").notna()
        | (numeric(df, "vel_x").notna() & numeric(df, "vel_y").notna() & numeric(df, "vel_z").notna()),
        "rssi": average_rssi(df).notna(),
    }
    total = len(df)
    return pd.DataFrame(
        {
            "field": list(checks.keys()),
            "valid_rows": [int(mask.sum()) for mask in checks.values()],
            "total_rows": total,
        }
    )


def derive_columns(df: pd.DataFrame) -> pd.DataFrame:
    return logged_direction_columns(df)



def logged_direction_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare display metrics from logged direction and score columns."""

    out = df.copy()
    out["score"] = pd.to_numeric(out.get("score"), errors="coerce")
    if "similarity" not in out.columns:
        out["similarity"] = out["score"]
    if "direction_source" not in out.columns:
        out["direction_source"] = "logged"
    if "drone_speed" not in out.columns:
        if "vel_abs" in out.columns:
            out["drone_speed"] = pd.to_numeric(out["vel_abs"], errors="coerce")
        else:
            out["drone_speed"] = np.sqrt(
                pd.to_numeric(out.get("vel_x"), errors="coerce") ** 2
                + pd.to_numeric(out.get("vel_y"), errors="coerce") ** 2
                + pd.to_numeric(out.get("vel_z"), errors="coerce") ** 2
            )
    out["avg_rssi"] = average_rssi(out)

    required = ["phone_rel_azimuth", "phone_rel_elevation", "earth_azimuth", "elevation"]
    if all(col in out.columns for col in required):
        valid = out[required].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
        out["angular_error_deg"] = np.nan
        out["azimuth_error_deg"] = np.nan
        if valid.any():
            out.loc[valid, "angular_error_deg"] = angular_error_deg(
                true_azimuth_deg=out.loc[valid, "phone_rel_azimuth"],
                true_elevation_deg=out.loc[valid, "phone_rel_elevation"],
                estimated_azimuth_deg=out.loc[valid, "earth_azimuth"],
                estimated_elevation_deg=out.loc[valid, "elevation"],
            )
            out.loc[valid, "azimuth_error_deg"] = azimuth_error_deg(
                out.loc[valid, "phone_rel_azimuth"],
                out.loc[valid, "earth_azimuth"],
            )
    return out


def fig11_distance_samples(all_df: pd.DataFrame) -> pd.Series:
    """Select Fig.11(f) drone-target distance experiment samples."""

    collection_id = all_df.get("collection_id", pd.Series("", index=all_df.index)).astype(str)
    distance = pd.to_numeric(all_df.get("distance"), errors="coerce")
    distance_mask = collection_id.isin(FIG11_DISTANCE_COLLECTION_IDS)
    elevation_filter = pd.to_numeric(all_df.get("elevation"), errors="coerce")
    if elevation_filter.dropna().empty:
        elevation_filter = pd.to_numeric(all_df.get("phone_rel_elevation"), errors="coerce")
    azimuth_filter = pd.to_numeric(all_df.get("phone_rel_azimuth"), errors="coerce") % 360.0
    distance_mask &= elevation_filter.between(0.0, 90.0) & azimuth_filter.between(0.0, 360.0)
    return distance[distance_mask]


def binned_values(x: pd.Series, y: pd.Series, edges: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    x_num = pd.to_numeric(x, errors="coerce")
    y_num = pd.to_numeric(y, errors="coerce")
    centers = []
    vals_per_bin = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (~x_num.isna()) & (~y_num.isna()) & (x_num >= lo) & (x_num < hi)
        centers.append((float(lo) + float(hi)) / 2.0)
        vals_per_bin.append(y_num[mask].values)
    return np.asarray(centers), vals_per_bin


def bar_with_bootstrap_ci(
    y: pd.Series,
    x: pd.Series,
    edges: np.ndarray,
    xlabel: str,
    out_base: Path | None,
    xtick_min: float | None = None,
    xtick_max: float | None = None,
    xtick_step: float | None = None,
    ylim_min: float = 0.5,
    ylim_max: float = 1.0,
    hide_xticks: bool = False,
    label_texts: list[str] | None = None,
    colors: list[str] | None = None,
    margin_ratio: float = 0.0,
    add_x_padding: bool = False,
    overflow_lower: float | None = None,
    overflow_upper: float | None = None,
    show_legend: bool = False,
) -> plt.Figure:
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    y_num = pd.to_numeric(y, errors="coerce")
    x_num = pd.to_numeric(x, errors="coerce")
    centers, vals_per_bin = binned_values(x_num, y_num, edges)
    means = np.array([np.nan if len(v) == 0 else float(np.median(v)) for v in vals_per_bin])
    ci_lows = np.array([np.nan if len(v) == 0 else bootstrap_ci_stat(v, agg="median")[0] for v in vals_per_bin])
    ci_highs = np.array([np.nan if len(v) == 0 else bootstrap_ci_stat(v, agg="median")[1] for v in vals_per_bin])

    bin_width = float(edges[1] - edges[0]) if len(edges) > 1 else 1.0
    eff_width = bin_width * (1.0 - max(0.0, min(0.9, margin_ratio)))
    centers_list = list(centers)
    means_list = list(means)
    ci_lows_list = list(ci_lows)
    ci_highs_list = list(ci_highs)

    left_center = right_center = None
    if overflow_lower is not None:
        left_center = float(edges[0]) - bin_width / 2.0
        vals = y_num[(~y_num.isna()) & (~x_num.isna()) & (x_num < float(overflow_lower))].values
        means_list = [np.nan if len(vals) == 0 else float(np.median(vals))] + means_list
        low, high = bootstrap_ci_stat(vals, agg="median") if len(vals) else (np.nan, np.nan)
        ci_lows_list = [low] + ci_lows_list
        ci_highs_list = [high] + ci_highs_list
        centers_list = [left_center] + centers_list
    if overflow_upper is not None:
        right_center = float(edges[-1]) + bin_width / 2.0
        vals = y_num[(~y_num.isna()) & (~x_num.isna()) & (x_num > float(overflow_upper))].values
        means_list.append(np.nan if len(vals) == 0 else float(np.median(vals)))
        low, high = bootstrap_ci_stat(vals, agg="median") if len(vals) else (np.nan, np.nan)
        ci_lows_list.append(low)
        ci_highs_list.append(high)
        centers_list.append(right_center)

    centers = np.asarray(centers_list)
    means = np.asarray(means_list)
    ci_lows = np.asarray(ci_lows_list)
    ci_highs = np.asarray(ci_highs_list)
    if colors is not None and len(colors) == len(centers) and overflow_lower is None and overflow_upper is None:
        bar_colors = colors
    else:
        bar_colors = [BAR_COLOR] * len(centers)
        if overflow_lower is not None and bar_colors:
            bar_colors[0] = UNDERFLOW_BAR_COLOR
        if overflow_upper is not None and bar_colors:
            bar_colors[-1] = OVERFLOW_BAR_COLOR

    ax.bar(centers, means, width=eff_width, color=bar_colors, alpha=1, align="center", edgecolor="black", linewidth=1.2, zorder=2)
    ci_mask = (~np.isnan(ci_lows)) & (~np.isnan(ci_highs)) & (~np.isnan(means))
    if np.any(ci_mask):
        means_valid = means[ci_mask]
        ax.errorbar(
            centers[ci_mask],
            means_valid,
            yerr=[means_valid - ci_lows[ci_mask], ci_highs[ci_mask] - means_valid],
            fmt="none",
            ecolor=CI_COLOR,
            elinewidth=1.2,
            capsize=4,
            alpha=CI_ALPHA,
            zorder=4,
        )

    text_margin = 0.2 * (ylim_max - ylim_min)
    for i, (c, m) in enumerate(zip(centers, means)):
        if np.isnan(m):
            continue
        ci_lo = ci_lows[i] if i < len(ci_lows) and not np.isnan(ci_lows[i]) else m
        ax.text(c, min(0.98, min(m, ci_lo) - text_margin), f"{m:.2f}", ha="center", va="bottom", rotation=90, color="black", fontsize=LABEL_SIZE - 2, zorder=5)

    ax.set_ylim(ylim_min, ylim_max)
    ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)
    ax.set_ylabel("Median Projection Rate", fontsize=LABEL_SIZE)
    if xtick_step is not None and xtick_min is not None and xtick_max is not None:
        ticks = np.arange(xtick_min, xtick_max + 1e-9, xtick_step)
        labels = [f"{int(round(t))}" if abs(t - round(t)) < 1e-9 else f"{t:g}" for t in ticks]
        if overflow_lower is not None and labels:
            labels[0] = f"{overflow_lower:g}>="
        if overflow_upper is not None and labels:
            labels[-1] = f"<={overflow_upper:g}"
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
        pad = 0.5 * eff_width if add_x_padding else 0.0
        left_lim = xtick_min - pad
        right_lim = xtick_max + pad
        if left_center is not None:
            left_lim = min(left_lim, left_center - 0.6 * eff_width)
        if right_center is not None:
            right_lim = max(right_lim, right_center + 0.6 * eff_width)
        ax.set_xlim(left_lim, right_lim)
    if hide_xticks:
        ax.set_xticks([])
    if label_texts is not None and len(label_texts) == len(centers):
        for c, text in zip(centers, label_texts):
            if text:
                ax.text(c, ylim_min + 0.02, str(text), ha="center", va="bottom", rotation=90, fontsize=LABEL_SIZE - 2, color="black", zorder=5)
    ax.grid(True, alpha=1, linestyle="--", zorder=1)
    if show_legend:
        legend_items = []
        if overflow_lower is not None:
            legend_items.append(Patch(facecolor=UNDERFLOW_BAR_COLOR, edgecolor="black", label=f"<= {overflow_lower:g}"))
        legend_items.append(Patch(facecolor=BAR_COLOR, edgecolor="black", label="Normal bins"))
        if overflow_upper is not None:
            legend_items.append(Patch(facecolor=OVERFLOW_BAR_COLOR, edgecolor="black", label=f">= {overflow_upper:g}"))
        ax.legend(handles=legend_items, loc="upper right")
    plt.tight_layout()
    if out_base is not None:
        save_both(fig, out_base)
        plt.close(fig)
    return fig


def radar_azimuth(
    y: pd.Series,
    azimuth_deg: pd.Series,
    edges: np.ndarray,
    out_base: Path | None,
    ylim_min: float = 0.5,
    ylim_max: float = 1.0,
) -> plt.Figure | None:
    centers, vals_per_bin = binned_values(pd.to_numeric(azimuth_deg, errors="coerce") % 360.0, pd.to_numeric(y, errors="coerce"), edges)
    means = np.array([np.nan if len(v) == 0 else float(np.median(v)) for v in vals_per_bin])
    ci_lows = np.array([np.nan if len(v) == 0 else bootstrap_ci_stat(v, agg="median")[0] for v in vals_per_bin])
    ci_highs = np.array([np.nan if len(v) == 0 else bootstrap_ci_stat(v, agg="median")[1] for v in vals_per_bin])
    theta = np.radians(centers)
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(111, polar=True)
    ax.set_position([0.0, 0.0, 1.0, 1.0])
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    bar_angle_width = np.radians(edges[1] - edges[0]) if len(edges) > 1 else np.radians(10.0)
    valid = means[np.isfinite(means)]
    if len(valid) == 0:
        plt.close(fig)
        return None
    cmap = plt.get_cmap("Blues")
    norm = matplotlib.colors.Normalize(vmin=float(np.nanmin(means)), vmax=float(np.nanmax(means)))
    colors = cmap(np.clip(0.2 + 0.7 * norm(means), 0.0, 1.0))
    heights = np.maximum(0.0, np.clip(means, ylim_min, ylim_max) - ylim_min)
    ax.bar(theta, heights, width=bar_angle_width, bottom=ylim_min, color=colors, alpha=1, edgecolor="black", linewidth=1.0, align="center")
    ax.set_rmax(ylim_max)
    ax.set_rmin(ylim_min)
    ax.set_rticks([0.5, 0.6, 0.7, 0.8, 0.9])
    angles = np.arange(0, 360, 10)
    ax.set_xticks(np.radians(angles))
    ax.set_xticklabels([str(a) for a in angles])
    ax.tick_params(axis="x", pad=-3, labelsize=DEFAULT_FONT_SIZE - 4)
    ax.set_xlabel("Azimuth (deg)", fontsize=LABEL_SIZE)
    ax.xaxis.set_label_coords(-0.18, 0.5)
    ax.xaxis.label.set_rotation(90)
    ax.xaxis.label.set_horizontalalignment("center")
    ax.xaxis.label.set_verticalalignment("center")
    ax.set_rlabel_position(180)
    ci_mask = (~np.isnan(ci_lows)) & (~np.isnan(ci_highs))
    if np.any(ci_mask):
        theta_ci = np.radians(centers[ci_mask] % 360.0)
        cap_half_angle = min(bar_angle_width * 0.4, np.radians(3.0))
        for th, r_lo, r_hi in zip(theta_ci, ci_lows[ci_mask], ci_highs[ci_mask]):
            ax.plot([th, th], [r_lo, r_hi], color=CI_COLOR, alpha=CI_ALPHA, linewidth=1.2, zorder=5)
            ax.plot([th - cap_half_angle, th + cap_half_angle], [r_lo, r_lo], color=CI_COLOR, alpha=CI_ALPHA, linewidth=1.2, zorder=5)
            ax.plot([th - cap_half_angle, th + cap_half_angle], [r_hi, r_hi], color=CI_COLOR, alpha=CI_ALPHA, linewidth=1.2, zorder=5)
    ax.grid(True, axis="y", alpha=0.5, color="blue", linestyle="--")
    ax.grid(True, axis="x", alpha=0.5, color="black", linestyle="--")
    ax.tick_params(axis="y", colors="blue", rotation=0, labelsize=DEFAULT_FONT_SIZE - 2)
    for label in ax.get_yticklabels():
        label.set_bbox(dict(boxstyle="round,pad=0.1", facecolor="white", alpha=1, zorder=10))
        label.set_horizontalalignment("center")
        label.set_verticalalignment("center")
    sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    sm.set_clim(0.89, 0.99)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.05, pad=0.15, ticks=[0.89, 0.9, 0.93, 0.96, 0.99])
    cbar.ax.tick_params(labelsize=DEFAULT_FONT_SIZE - 2)
    cbar.ax.text(-1, 0.5, "Median Projection Rate", transform=cbar.ax.transAxes, ha="center", va="center", rotation=90)
    if out_base is not None:
        save_both(fig, out_base, bbox_inches=None)
        plt.close(fig)
    return fig


def plot_scene_cdf(all_df: pd.DataFrame, out_base: Path | None = None) -> plt.Figure:
    scenes_order = ["Scene-P", "Scene-F", "Scene-T", "Scene-S"]
    scene_colors = {
        "Scene-P": "#E8A33D",
        "Scene-F": "#984EA3",
        "Scene-T": "#E41A1C",
        "Scene-S": "#377EB8",
    }
    scene_df = all_df[all_df["scene"].notna()] if "scene" in all_df.columns else all_df.iloc[0:0]
    y_all = pd.to_numeric(scene_df.get("score"), errors="coerce").dropna().clip(lower=-1.0, upper=1.0)
    overall_mpr = float(np.median(y_all)) if len(y_all) else np.nan
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    for scene_label in scenes_order:
        mask_scene = scene_df.get("scene").astype(str) == scene_label
        pr_scene = pd.to_numeric(scene_df.loc[mask_scene, "score"], errors="coerce").dropna().clip(lower=-1.0, upper=1.0).sort_values()
        if len(pr_scene) == 0:
            continue
        cdf_scene = np.arange(1, len(pr_scene) + 1) / float(len(pr_scene))
        ax.plot(
            pr_scene.values,
            cdf_scene,
            linewidth=2,
            label=f"{scene_label} (MedPR={float(np.median(pr_scene)):.2f})",
            color=scene_colors.get(scene_label),
            alpha=0.8,
            linestyle="--",
        )

    pr_all = y_all.sort_values()
    if len(pr_all):
        cdf_all = np.arange(1, len(pr_all) + 1) / float(len(pr_all))
        ax.plot(pr_all.values, cdf_all, linewidth=1, label=f"All (MedPR={overall_mpr:.2f}, N={len(pr_all)})", color="black", alpha=1)
        for cdf_target, text_y, valign in [(0.2, 0.21, "bottom"), (0.5, 0.49, "top")]:
            idx = np.searchsorted(cdf_all, cdf_target, side="left")
            if idx < len(pr_all):
                pr_at = pr_all.iloc[idx]
                ax.hlines(cdf_target, 0.0, pr_at, colors="black", linestyles="--", linewidth=1.5)
                ax.vlines(pr_at, 0.0, cdf_target if cdf_target == 0.5 else 0.18, colors="black", linestyles="--", linewidth=1.5)
                ax.scatter([pr_at], [cdf_target], color="black", s=12, zorder=3)
                pct = int(round((1.0 - cdf_target) * 100))
                ax.text(0.43, text_y, f"{pct}% samples' PR >= {pr_at:.2f} ({np.degrees(np.arccos(np.clip(pr_at, -1, 1))):.1f}deg)", ha="center", va=valign, color="black", fontsize=LABEL_SIZE - 1)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0, 0.05, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0])
    ax.set_xlabel("Projection Rate", fontsize=LABEL_SIZE)
    ax.set_ylabel("Empirical CDF", fontsize=LABEL_SIZE)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=MEDIUM_FONT_SIZE)
    if out_base is not None:
        save_both(fig, out_base)
        plt.close(fig)
    return fig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    all_df = logged_direction_columns(pd.read_csv(args.data, low_memory=False))
    all_df["score"] = pd.to_numeric(all_df["score"], errors="coerce")
    y = pd.to_numeric(all_df.get("score"), errors="coerce")
    all_samples = len(all_df)
    print(f"Total samples including all files: {all_samples}")

    plot_scene_cdf(all_df, args.out / "fig11a_score_vs_scene")

    selected_placements = [
        "In backpack",
        "Human held",
        "In bushes",
        "Under tree",
        "By the wall",
        "In box",
        "Lying on ground",
        "Under rugged terrain",
        "By the curb",
        "Beside tetrapod",
        "In rock crevice",
        "At the shoreline",
    ]
    if "placement" in all_df.columns:
        df_p = all_df.copy()
        df_p["score"] = pd.to_numeric(df_p["score"], errors="coerce")
        df_p = df_p[df_p["placement"].isin(selected_placements) & (~df_p["placement"].isna()) & (~df_p["score"].isna())]
        present = [p for p in selected_placements if p in set(df_p["placement"].astype(str).unique())]
        if present:
            placement_to_idx = {p: i for i, p in enumerate(present)}
            x_codes = df_p["placement"].astype(str).map(placement_to_idx).astype(float)
            colors = [CMP_COLOR_LIST_LIGHT[i % len(CMP_COLOR_LIST_LIGHT)] for i in range(len(present))]
            bar_with_bootstrap_ci(
                y=df_p["score"],
                x=x_codes,
                edges=np.arange(0.0, float(len(present)) + 1.0, 1.0),
                xlabel="Placement",
                out_base=args.out / "fig11b_score_vs_placement",
                hide_xticks=True,
                label_texts=present,
                margin_ratio=0.15,
                colors=colors,
            )

    az = pd.to_numeric(all_df.get("phone_rel_azimuth"), errors="coerce") % 360.0
    elev_true = pd.to_numeric(all_df.get("elevation"), errors="coerce")
    if elev_true.dropna().empty:
        elev_true = pd.to_numeric(all_df.get("phone_rel_elevation"), errors="coerce")
    dist = pd.to_numeric(all_df.get("distance"), errors="coerce")
    mask = elev_true.between(0.0, 90.0) & dist.between(0.0, 280.0)
    radar_azimuth(y=y[mask], azimuth_deg=az[mask], edges=np.arange(0.0, 370.0, 10.0), out_base=args.out / "fig11c_score_vs_azimuth")

    te = pd.to_numeric(all_df.get("phone_rel_elevation"), errors="coerce")
    mx = np.ceil(max(15.0, float(te.max(skipna=True) if te.notna().any() else 90.0)) / 5.0) * 5.0
    bar_with_bootstrap_ci(
        y=y.loc[te.index],
        x=te,
        edges=np.arange(15.0, mx + 5.0, 5.0),
        xlabel="Elevation (deg)",
        out_base=args.out / "fig11d_score_vs_elevation",
        xtick_min=15.0,
        xtick_max=90.0,
        xtick_step=5.0,
        add_x_padding=True,
    )

    ds = pd.to_numeric(all_df.get("drone_speed"), errors="coerce")
    ds = ds[ds.between(0.0, 5.5)]
    bar_with_bootstrap_ci(
        y=y.loc[ds.index],
        x=ds,
        edges=np.arange(0.0, 5.5 + 0.5, 0.5),
        xlabel="Drone speed (m/s)",
        out_base=args.out / "fig11e_score_vs_speed",
        xtick_min=0.0,
        xtick_max=5.5,
        xtick_step=0.5,
        add_x_padding=True,
    )

    distance = fig11_distance_samples(all_df)
    bar_with_bootstrap_ci(
        y=y.loc[distance.index],
        x=distance,
        edges=np.arange(30.0, 500.0, 50.0),
        xlabel="Distance (m)",
        out_base=args.out / "fig11f_score_vs_distance",
        xtick_min=30.0,
        xtick_max=430.0,
        xtick_step=50.0,
        add_x_padding=True,
        overflow_lower=None,
        overflow_upper=None,
        show_legend=False,
    )

    avg_rssi = pd.to_numeric(all_df.get("avg_rssi"), errors="coerce")
    bar_with_bootstrap_ci(
        y=y.loc[avg_rssi.index],
        x=avg_rssi,
        edges=np.arange(-96.0, -66.0 + 3.0, 3.0),
        xlabel="Average RSS (dBm)",
        out_base=args.out / "fig11g_score_vs_avg_rssi",
        xtick_min=-96.0,
        xtick_max=-66.0,
        xtick_step=6.0,
        add_x_padding=True,
    )

    print(f"wrote Fig.11(a-g) outputs to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


PACKAGE_ROOT = Path(__file__).resolve().parents[1]

REFERENCE_IMAGES = {
    "fig04": ["frontend/reference_images/beam_pattern.png"],
    "fig10": ["frontend/reference_images/ll_eff.png"],
    "fig11": ["frontend/reference_images/fig11_direction_finding_reference.png"],
    "fig12": ["frontend/reference_images/search_trajectory.png"],
}


def _root(root: str | Path | None = None) -> Path:
    return Path(root).resolve() if root is not None else PACKAGE_ROOT


def _rel(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _paths(root: Path, *paths: Path) -> list[str]:
    return [_rel(root, path) for path in paths]


def reference_images(root: str | Path | None = None, key: str | None = None) -> list[str]:
    r = _root(root)
    keys = [key] if key else sorted(REFERENCE_IMAGES)
    return [p for k in keys for p in REFERENCE_IMAGES.get(k, []) if (r / p).exists()]


def data_summary(root: str | Path | None = None) -> pd.DataFrame:
    """Return package-local CSV tables and row counts."""

    r = _root(root)
    rows = []
    for path in sorted((r / "data").glob("*/*.csv")):
        try:
            count = sum(1 for _ in path.open("r", encoding="utf-8")) - 1
        except UnicodeDecodeError:
            count = pd.read_csv(path, low_memory=False).shape[0]
        rows.append({"path": _rel(r, path), "rows": max(count, 0)})
    return pd.DataFrame(rows)


def fig04_beam_pattern(root: str | Path | None = None, force: bool = False) -> list[str]:
    """Regenerate Figure 4(b) from RSS sweep and interpolated beam-pattern CSVs."""

    r = _root(root)
    data_dir = r / "data" / "fig04_ll_beam_pattern"
    out_dir = r / "output" / "fig04_ll_beam_pattern"
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "fig04b_measured_ll_beam_pattern.png"
    sphere_png = out_dir / "fig04b_measured_ll_beam_pattern_3d.png"
    if force or not png.exists():
        matrix = load_interpolated_matrix(data_dir / "fig04_interpolated_beam_pattern.csv")
        save_original_style_beam_pattern(matrix, out_dir / "fig04b_measured_ll_beam_pattern")
        summarize_raw_sweep(data_dir / "fig04_ll_rssi_sweep.csv", out_dir)
    if force or not sphere_png.exists():
        azimuth, elevation, response = load_grid(data_dir / "fig04_interpolated_beam_pattern.csv")
        save_beam_pattern_sphere_preview(azimuth, elevation, response, sphere_png)
    return _paths(r, png, sphere_png)


def fig04_beam_pattern_figures(root: str | Path | None = None) -> list[tuple[str, object]]:
    """Build Fig.04(b) display figures directly from CSV data."""

    r = _root(root)
    data_dir = r / "data" / "fig04_ll_beam_pattern"
    matrix = load_interpolated_matrix(data_dir / "fig04_interpolated_beam_pattern.csv")
    static_fig = plot_original_style_beam_pattern_matrix(matrix)
    azimuth, elevation, response = load_grid(data_dir / "fig04_interpolated_beam_pattern.csv")
    sphere = build_surface(azimuth, elevation, response)
    return [("Figure 4(b), Matplotlib beam pattern", static_fig), ("Figure 4(b), Plotly interactive sphere", sphere)]


def fig10_ll_effectiveness(root: str | Path | None = None, force: bool = False) -> list[str]:
    """Regenerate Figure 10(b,c) from package-local CSV tables."""

    r = _root(root)
    data_dir = r / "data" / "fig10_ll_effectiveness"
    out_dir = r / "output" / "fig10_ll_effectiveness"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_b = out_dir / "fig10b_rss_gain_ground.png"
    out_c = out_dir / "fig10c_aerial_victim_discovery.png"
    if force or not out_b.exists():
        plot_rss_gain(data_dir / "fig10_rss_gain_ground.csv", out_dir / "fig10b_rss_gain_ground")
    if force or not out_c.exists():
        plot_vdm_range(data_dir / "fig10_vdm_working_range.csv", out_dir / "fig10c_aerial_victim_discovery")
    return _paths(r, out_b, out_c)


def fig10_ll_effectiveness_figures(root: str | Path | None = None) -> list[tuple[str, plt.Figure]]:
    """Build Fig.10(b,c) Matplotlib figures directly from CSV data."""

    r = _root(root)
    data_dir = r / "data" / "fig10_ll_effectiveness"
    return [
        ("Figure 10(b), RSS gain", plot_rss_gain(data_dir / "fig10_rss_gain_ground.csv")),
        ("Figure 10(c), aerial victim discovery range", plot_vdm_range(data_dir / "fig10_vdm_working_range.csv")),
    ]


def fig11_direction_finding(root: str | Path | None = None, force: bool = False) -> list[str]:
    """Regenerate Figure 11(a-h) from the direction-finding flight-log CSV."""

    r = _root(root)
    data_path = r / "data" / "fig11_direction_finding" / "fig11_direction_finding_dataset.csv"
    out_dir = r / "output" / "fig11_direction_finding"
    out_dir.mkdir(parents=True, exist_ok=True)
    expected = [
        out_dir / "fig11a_score_vs_scene.png",
        out_dir / "fig11b_score_vs_placement.png",
        out_dir / "fig11c_score_vs_azimuth.png",
        out_dir / "fig11d_score_vs_elevation.png",
        out_dir / "fig11e_score_vs_speed.png",
        out_dir / "fig11f_score_vs_distance.png",
        out_dir / "fig11g_score_vs_avg_rssi.png",
        r / "output" / "fig11h_arraytrack_comparison" / "fig11h_ll_vs_arraytrack_cdf.png",
    ]
    if force or not all(path.exists() for path in expected[:7]):
        all_df = logged_direction_columns(pd.read_csv(data_path, low_memory=False))
        all_df["score"] = pd.to_numeric(all_df["score"], errors="coerce")
        y = pd.to_numeric(all_df.get("score"), errors="coerce")

        plot_scene_cdf(all_df, out_dir / "fig11a_score_vs_scene")

        selected_placements = [
            "In backpack",
            "Human held",
            "In bushes",
            "Under tree",
            "By the wall",
            "In box",
            "Lying on ground",
            "Under rugged terrain",
            "By the curb",
            "Beside tetrapod",
            "In rock crevice",
            "At the shoreline",
        ]
        if "placement" in all_df.columns:
            df_p = all_df.copy()
            df_p = df_p[df_p["placement"].isin(selected_placements) & (~df_p["score"].isna())]
            present = [p for p in selected_placements if p in set(df_p["placement"].astype(str).unique())]
            if present:
                placement_to_idx = {p: i for i, p in enumerate(present)}
                x_codes = df_p["placement"].astype(str).map(placement_to_idx).astype(float)
                colors = [CMP_COLOR_LIST_LIGHT[i % len(CMP_COLOR_LIST_LIGHT)] for i in range(len(present))]
                bar_with_bootstrap_ci(
                    y=df_p["score"],
                    x=x_codes,
                    edges=np.arange(0.0, float(len(present)) + 1.0, 1.0),
                    xlabel="Placement",
                    out_base=out_dir / "fig11b_score_vs_placement",
                    hide_xticks=True,
                    label_texts=present,
                    margin_ratio=0.15,
                    colors=colors,
                )

        az = pd.to_numeric(all_df.get("phone_rel_azimuth"), errors="coerce") % 360.0
        elev_true = pd.to_numeric(all_df.get("elevation"), errors="coerce")
        if elev_true.dropna().empty:
            elev_true = pd.to_numeric(all_df.get("phone_rel_elevation"), errors="coerce")
        dist = pd.to_numeric(all_df.get("distance"), errors="coerce")
        mask = elev_true.between(0.0, 90.0) & dist.between(0.0, 280.0)
        radar_azimuth(
            y=y[mask],
            azimuth_deg=az[mask],
            edges=np.arange(0.0, 370.0, 10.0),
            out_base=out_dir / "fig11c_score_vs_azimuth",
        )

        te = pd.to_numeric(all_df.get("phone_rel_elevation"), errors="coerce")
        mx = np.ceil(max(15.0, float(te.max(skipna=True) if te.notna().any() else 90.0)) / 5.0) * 5.0
        bar_with_bootstrap_ci(
            y=y.loc[te.index],
            x=te,
            edges=np.arange(15.0, mx + 5.0, 5.0),
            xlabel="Elevation (deg)",
            out_base=out_dir / "fig11d_score_vs_elevation",
            xtick_min=15.0,
            xtick_max=90.0,
            xtick_step=5.0,
            add_x_padding=True,
        )

        ds = pd.to_numeric(all_df.get("drone_speed"), errors="coerce")
        ds = ds[ds.between(0.0, 5.5)]
        bar_with_bootstrap_ci(
            y=y.loc[ds.index],
            x=ds,
            edges=np.arange(0.0, 5.5 + 0.5, 0.5),
            xlabel="Drone speed (m/s)",
            out_base=out_dir / "fig11e_score_vs_speed",
            xtick_min=0.0,
            xtick_max=5.5,
            xtick_step=0.5,
            add_x_padding=True,
        )

        distance = fig11_distance_samples(all_df)
        bar_with_bootstrap_ci(
            y=y.loc[distance.index],
            x=distance,
            edges=np.arange(30.0, 500.0, 50.0),
            xlabel="Distance (m)",
            out_base=out_dir / "fig11f_score_vs_distance",
            xtick_min=30.0,
            xtick_max=430.0,
            xtick_step=50.0,
            add_x_padding=True,
        )

        avg_rssi = pd.to_numeric(all_df.get("avg_rssi"), errors="coerce")
        bar_with_bootstrap_ci(
            y=y.loc[avg_rssi.index],
            x=avg_rssi,
            edges=np.arange(-96.0, -66.0 + 3.0, 3.0),
            xlabel="Average RSS (dBm)",
            out_base=out_dir / "fig11g_score_vs_avg_rssi",
            xtick_min=-96.0,
            xtick_max=-66.0,
            xtick_step=6.0,
            add_x_padding=True,
        )

    h_dir = r / "output" / "fig11h_arraytrack_comparison"
    h_dir.mkdir(parents=True, exist_ok=True)
    if force or not expected[7].exists():
        df_h = pd.read_csv(r / "data" / "fig11h_arraytrack_comparison" / "fig11h_arraytrack_errors.csv")
        plot_cdf(df_h, h_dir / "fig11h_ll_vs_arraytrack_cdf")

    return _paths(r, *expected)


def fig11_direction_finding_figures(root: str | Path | None = None) -> list[tuple[str, plt.Figure]]:
    """Build Fig.11 display figures directly from the flight-log CSV."""

    r = _root(root)
    data_path = r / "data" / "fig11_direction_finding" / "fig11_direction_finding_dataset.csv"
    all_df = logged_direction_columns(pd.read_csv(data_path, low_memory=False))
    all_df["score"] = pd.to_numeric(all_df["score"], errors="coerce")
    y = pd.to_numeric(all_df.get("score"), errors="coerce")
    figures: list[tuple[str, plt.Figure]] = [
        ("Figure 11(a), projection-rate CDF by scene", plot_scene_cdf(all_df)),
    ]

    selected_placements = [
        "In backpack",
        "Human held",
        "In bushes",
        "Under tree",
        "By the wall",
        "In box",
        "Lying on ground",
        "Under rugged terrain",
        "By the curb",
        "Beside tetrapod",
        "In rock crevice",
        "At the shoreline",
    ]
    if "placement" in all_df.columns:
        df_p = all_df[all_df["placement"].isin(selected_placements) & (~all_df["score"].isna())].copy()
        present = [p for p in selected_placements if p in set(df_p["placement"].astype(str).unique())]
        if present:
            placement_to_idx = {p: i for i, p in enumerate(present)}
            x_codes = df_p["placement"].astype(str).map(placement_to_idx).astype(float)
            colors = [CMP_COLOR_LIST_LIGHT[i % len(CMP_COLOR_LIST_LIGHT)] for i in range(len(present))]
            figures.append(
                (
                    "Figure 11(b), median projection rate by placement",
                    bar_with_bootstrap_ci(
                        y=df_p["score"],
                        x=x_codes,
                        edges=np.arange(0.0, float(len(present)) + 1.0, 1.0),
                        xlabel="Placement",
                        out_base=None,
                        hide_xticks=True,
                        label_texts=present,
                        margin_ratio=0.15,
                        colors=colors,
                    ),
                )
            )

    az = pd.to_numeric(all_df.get("phone_rel_azimuth"), errors="coerce") % 360.0
    elev_true = pd.to_numeric(all_df.get("elevation"), errors="coerce")
    if elev_true.dropna().empty:
        elev_true = pd.to_numeric(all_df.get("phone_rel_elevation"), errors="coerce")
    dist = pd.to_numeric(all_df.get("distance"), errors="coerce")
    mask = elev_true.between(0.0, 90.0) & dist.between(0.0, 280.0)
    azimuth_fig = radar_azimuth(y=y[mask], azimuth_deg=az[mask], edges=np.arange(0.0, 370.0, 10.0), out_base=None)
    if azimuth_fig is not None:
        figures.append(("Figure 11(c), median projection rate by incident azimuth", azimuth_fig))

    te = pd.to_numeric(all_df.get("phone_rel_elevation"), errors="coerce")
    mx = np.ceil(max(15.0, float(te.max(skipna=True) if te.notna().any() else 90.0)) / 5.0) * 5.0
    figures.append(
        (
            "Figure 11(d), median projection rate by incident elevation",
            bar_with_bootstrap_ci(
                y=y.loc[te.index],
                x=te,
                edges=np.arange(15.0, mx + 5.0, 5.0),
                xlabel="Elevation (deg)",
                out_base=None,
                xtick_min=15.0,
                xtick_max=90.0,
                xtick_step=5.0,
                add_x_padding=True,
            ),
        )
    )

    ds = pd.to_numeric(all_df.get("drone_speed"), errors="coerce")
    ds = ds[ds.between(0.0, 5.5)]
    figures.append(
        (
            "Figure 11(e), median projection rate by drone speed",
            bar_with_bootstrap_ci(
                y=y.loc[ds.index],
                x=ds,
                edges=np.arange(0.0, 5.5 + 0.5, 0.5),
                xlabel="Drone speed (m/s)",
                out_base=None,
                xtick_min=0.0,
                xtick_max=5.5,
                xtick_step=0.5,
                add_x_padding=True,
            ),
        )
    )

    distance = fig11_distance_samples(all_df)
    figures.append(
        (
            "Figure 11(f), median projection rate by drone-target distance",
            bar_with_bootstrap_ci(
                y=y.loc[distance.index],
                x=distance,
                edges=np.arange(30.0, 500.0, 50.0),
                xlabel="Distance (m)",
                out_base=None,
                xtick_min=30.0,
                xtick_max=430.0,
                xtick_step=50.0,
                add_x_padding=True,
            ),
        )
    )

    avg_rssi = pd.to_numeric(all_df.get("avg_rssi"), errors="coerce")
    figures.append(
        (
            "Figure 11(g), median projection rate by average RSS",
            bar_with_bootstrap_ci(
                y=y.loc[avg_rssi.index],
                x=avg_rssi,
                edges=np.arange(-96.0, -66.0 + 3.0, 3.0),
                xlabel="Average RSS (dBm)",
                out_base=None,
                xtick_min=-96.0,
                xtick_max=-66.0,
                xtick_step=6.0,
                add_x_padding=True,
            ),
        )
    )

    df_h = pd.read_csv(r / "data" / "fig11h_arraytrack_comparison" / "fig11h_arraytrack_errors.csv")
    figures.append(("Figure 11(h), Wi2SAR and ArrayTrack angular-error CDF", plot_cdf(df_h)))
    return figures


def fig12_trajectory(root: str | Path | None = None, force: bool = False) -> list[str]:
    """Regenerate Figure 12(a-c) trajectory plot files."""

    r = _root(root)
    data_dir = r / "data" / "fig12_trajectory"
    out_dir = r / "output" / "fig12_trajectory"
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        out_dir / "fig12a_fig12a_zigzag.png",
        out_dir / "fig12b_fig12b_large_area_search.png",
        out_dir / "fig12c_fig12c_full_wi2sar_trial.png",
        out_dir / "fig12_abc_combined.png",
    ]
    if force or not all(path.exists() for path in outputs[:4]):
        metadata = pd.read_csv(data_dir / "fig12_metadata.csv")
        fig = plt.figure(figsize=(15, 4.8))
        axes = [fig.add_subplot(1, 3, idx + 1, projection="3d") for idx, _ in enumerate(PANELS)]
        for ax, (figure_id, filename, title) in zip(axes, PANELS):
            meta_rows = metadata[metadata["figure_id"] == figure_id].copy()
            panel_df, takeoff_lat, takeoff_lon = load_logged_direction_panel(data_dir / filename, meta_rows)
            plot_panel(
                ax,
                figure_id,
                panel_df,
                meta_rows,
                figure_id,
                takeoff_lat,
                takeoff_lon,
                title_fontsize=9,
            )
            stem = filename[:-4] if filename.endswith(".csv") else filename
            single_fig = plt.figure(figsize=(12, 10))
            single_ax = single_fig.add_subplot(1, 1, 1, projection="3d")
            plot_panel(
                single_ax,
                figure_id,
                panel_df,
                meta_rows,
                title,
                takeoff_lat,
                takeoff_lon,
                title_fontsize=12,
            )
            save_both(single_fig, out_dir / f"{figure_id}_{stem}")
            plt.close(single_fig)
        fig.tight_layout()
        save_both(fig, out_dir / "fig12_abc_combined")
        plt.close(fig)

    return _paths(r, *outputs)


def fig12_trajectory_figures(root: str | Path | None = None) -> list[tuple[str, object]]:
    """Build notebook-native Plotly Fig.12 trajectory views from trajectory CSV data."""

    r = _root(root)
    data_dir = r / "data" / "fig12_trajectory"
    metadata = pd.read_csv(data_dir / "fig12_metadata.csv")
    figures: list[tuple[str, object]] = []

    meta_rows = metadata[(metadata["figure_id"] == "fig12a") & (metadata["phone_name"] == "IP15")].copy()
    panel_df, takeoff_lat, takeoff_lon = load_logged_direction_panel(data_dir / "fig12a_zigzag.csv", meta_rows)
    figures.append(("Figure 12(a), IP15 interactive 3D trajectory", build_figure(panel_df, meta_rows, takeoff_lat, takeoff_lon, "IP15")))
    meta_rows = metadata[(metadata["figure_id"] == "fig12b") & (metadata["phone_name"].isin(FIG12B_PHONES))].copy()
    panel_df, takeoff_lat, takeoff_lon = load_logged_direction_panel(data_dir / "fig12b_large_area_search.csv", meta_rows)
    figures.append(("Figure 12(b), multi-device discovery", build_multi_phone_figure(panel_df, meta_rows, takeoff_lat, takeoff_lon, FIG12B_PHONES)))
    meta_rows = metadata[(metadata["figure_id"] == "fig12c") & (metadata["phone_name"] == "IP15")].copy()
    panel_df, takeoff_lat, takeoff_lon = load_logged_direction_panel(data_dir / "fig12c_full_wi2sar_trial.csv", meta_rows)
    figures.append(("Figure 12(c), IP15 interactive 3D trajectory", build_figure(panel_df, meta_rows, takeoff_lat, takeoff_lon, "IP15")))
    return figures


def regenerate_all(root: str | Path | None = None, force: bool = False) -> dict[str, list[str]]:
    """Regenerate all data-backed figure artifacts."""

    r = _root(root)
    return {
        "Figure 4": fig04_beam_pattern(r, force=force),
        "Figure 10": fig10_ll_effectiveness(r, force=force),
        "Figure 11": fig11_direction_finding(r, force=force),
        "Figure 12": fig12_trajectory(r, force=force),
    }
