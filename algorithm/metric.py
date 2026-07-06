"""Wi2SAR direction-prediction metric helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _float_array(values) -> np.ndarray:
    return np.asarray(pd.to_numeric(values, errors="coerce"), dtype=float)


def earth_azimuth_from_body(azimuth, yaw):
    """Convert Wi2SAR body-frame azimuth to earth-frame azimuth."""
    return (-_float_array(azimuth) + _float_array(yaw)) % 360


def projection_rate_score(gt_elevation, gt_azimuth, pred_elevation, pred_earth_azimuth) -> np.ndarray:
    """Compute the released projection-rate score from two spherical directions."""

    gt_elevation = np.radians(_float_array(gt_elevation))
    gt_azimuth = np.radians(_float_array(gt_azimuth))
    pred_elevation = np.radians(_float_array(pred_elevation))
    pred_earth_azimuth = np.radians(_float_array(pred_earth_azimuth))

    dtheta = pred_elevation - gt_elevation
    dphi = pred_earth_azimuth - gt_azimuth
    a = (np.sin(dtheta / 2.0) ** 2) + (
        np.cos(gt_elevation) * np.cos(pred_elevation) * (np.sin(dphi / 2.0) ** 2)
    )
    a = np.clip(a, 0.0, 1.0)
    central_angle = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return np.cos(central_angle)
