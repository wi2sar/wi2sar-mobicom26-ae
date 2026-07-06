"""Configured RSSI-to-direction replay used by the Figure 11 notebook."""

import time
from pathlib import Path

import numpy as np
import pandas as pd

from module.basic_classes import AntennaLayout, SphericalMapper
from module.data_processing import construct_beam_pattern, load_layout, load_signals
from module.evaluation import bp_form_cyclic_matrix, conv2d_fft, cross_correlation_2d_fft


RSSI_COLUMNS = [f"rssi_{i}" for i in range(1, 11)]
RSSI_NORMALIZATION = "z-score-nonzero"
NOISE_FLOOR_DBM = -127
POSITIVE_ELEVATION_ONLY = True


def load_interpolated_beam_pattern(calibration_sheet_tsv, layout_yaml):
    """Build the measured Luneburg Lens beam pattern from calibration RSSI."""

    layout = load_layout(layout_yaml)
    dataset, rssi_calibration, beam_pattern_layout = load_signals(str(Path(calibration_sheet_tsv)), layout)
    raw = construct_beam_pattern(dataset, beam_pattern_layout, rssi_calibration, calibration=True).interpolated_beam_pattern
    beam_pattern = {(int(az), int(el)): float(value) for (az, el), value in raw.items()}
    assert len(beam_pattern) == 360 * 181, len(beam_pattern)
    return beam_pattern


def z_score_nonzero(values):
    """Normalize present RSSI values while leaving zero placeholders unchanged."""

    out = np.asarray(values, dtype=float).copy()
    mask = out != 0
    if not mask.any():
        return out
    std = out[mask].std()
    if std == 0:
        return out
    out[mask] = (out[mask] - out[mask].mean()) / std
    return out


def standardize(values, method=RSSI_NORMALIZATION):
    """Apply the normalization mode used by the replayed Wi2SAR direction search."""

    arr = np.asarray(values, dtype=float)
    if method == "z-score-nonzero":
        return z_score_nonzero(arr)
    if method == "z-score":
        return arr if arr.std() == 0 else (arr - arr.mean()) / arr.std()
    if method == "none":
        return arr.copy()
    raise ValueError(f"unsupported replay normalization: {method}")


def prepare_beam_pattern_fft(beam_pattern, normalization=RSSI_NORMALIZATION):
    """Convert measured beam-pattern samples into the cyclic FFT grid used by a_BcRxB."""

    beam_matrix = np.zeros((360, 181))
    for (azimuth, elevation), value in beam_pattern.items():
        beam_matrix[int(azimuth), int(elevation) + 90] = float(value)
    beam_matrix = standardize(beam_matrix, method=normalization)
    return np.fft.rfft2(bp_form_cyclic_matrix(beam_matrix))


def build_replay_context(
    calibration_sheet_tsv,
    default_layout_yaml,
    layout_20250709_yaml,
    normalization=RSSI_NORMALIZATION,
    noise_floor_dbm=NOISE_FLOOR_DBM,
    positive_elevation_only=POSITIVE_ELEVATION_ONLY,
):
    """Load static inputs once before replaying rows."""

    beam_pattern = load_interpolated_beam_pattern(calibration_sheet_tsv, default_layout_yaml)
    return {
        "beam_pattern_fft": prepare_beam_pattern_fft(beam_pattern, normalization=normalization),
        "default_layout": load_layout(default_layout_yaml),
        "layout_20250709": load_layout(layout_20250709_yaml),
        "normalization": normalization,
        "noise_floor_dbm": noise_floor_dbm,
        "positive_elevation_only": positive_elevation_only,
        "spherical_mapper": SphericalMapper(),
    }


def tags_from_header(header):
    """Read antenna location tags from one released RSSI header."""

    parts = str(header).split("\t")
    assert parts[:2] == ["A", "E"], f"unexpected RSSI header: {header!r}"
    return parts[2:]


def measured_layout_from_header(header, context):
    """Select the released antenna layout matching one RSSI header."""

    tags = tags_from_header(header)
    source_layout = context["layout_20250709"] if any(
        tag.startswith("TRd") or tag.startswith("TR3") or tag.startswith("TR64") for tag in tags
    ) else context["default_layout"]

    antenna_tags = []
    missing = []
    for tag in tags:
        antenna = source_layout.find_tag(tag)
        if antenna is None:
            missing.append(tag)
        else:
            antenna_tags.append(antenna)
    assert not missing, f"RSSI header tags not found in selected layout: {missing}"
    return AntennaLayout(antenna_tags)


def rssi_vector_from_row(row, rssi_columns=RSSI_COLUMNS):
    """Extract the RSSI vector logged for one sample."""

    return np.array([float(row[col]) for col in rssi_columns], dtype=float)


def estimate_direction_one_row(row, context, rssi_columns=RSSI_COLUMNS):
    """Replay Wi2SAR a_BcRxB direction finding for one released RSSI sample."""

    start = time.perf_counter()
    measured_layout = measured_layout_from_header(row["rssi_header_raw"], context)
    measured_rssi = rssi_vector_from_row(row, rssi_columns=rssi_columns)

    measured_rssi_calibrated = measured_rssi - context["noise_floor_dbm"]
    measured_rssi_normalized = standardize(measured_rssi_calibrated, method=context["normalization"])

    measured_pad = np.zeros((360, 360))
    for value, antenna in zip(measured_rssi_normalized, measured_layout.antenna_tags):
        measured_pad[int(antenna.location.azimuth), int(antenna.location.elevation) + 90] = value

    conv = conv2d_fft(context["beam_pattern_fft"], measured_pad, input_type="fAB")
    similarity = cross_correlation_2d_fft(conv, context["beam_pattern_fft"], input_type="AfB")

    if context["positive_elevation_only"]:
        min_val = np.nanmin(similarity)
        similarity[:, :90] = min_val
        similarity[:, 270:] = min_val

    max_i, max_j = np.unravel_index(np.nanargmax(similarity), similarity.shape)
    azimuth, elevation = context["spherical_mapper"].offset(max_i, max_j, start_from=(0, 0))
    elapsed = time.perf_counter() - start
    return int(round(azimuth)) % 360, int(round(elevation)), elapsed


def recompute_estimated_direction(
    df,
    context,
    collection_labels=None,
    max_rows=None,
    progress_every=500,
    rssi_columns=RSSI_COLUMNS,
):
    """Replay direction finding for the selected Figure 11 samples."""

    rows = df if max_rows is None else df.head(max_rows)
    outputs = []
    total = len(rows)
    start_all = time.perf_counter()
    collection_labels = collection_labels or {}

    for n, (idx, row) in enumerate(rows.iterrows(), start=1):
        azimuth, elevation, elapsed = estimate_direction_one_row(row, context, rssi_columns=rssi_columns)
        collection_id = str(row["collection_id"])
        outputs.append({
            "source_index": idx,
            "collection_id": collection_id,
            "trial_id": collection_labels.get(collection_id, collection_id),
            "recomputed_azimuth": azimuth,
            "recomputed_elevation": elevation,
            "recompute_time_s": elapsed,
        })
        if progress_every and (n % progress_every == 0 or n == total):
            rate = n / max(time.perf_counter() - start_all, 1e-9)
            print(f"[{n:>6}/{total:<6}] {rate:6.1f} rows/s")

    return pd.DataFrame(outputs).set_index("source_index")
