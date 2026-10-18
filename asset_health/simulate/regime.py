"""Operating regimes (SYNTHETIC): load, speed, ambient temperature and stoppages."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter

from asset_health.catalog import AssetTypeProfile
from asset_health.config import SiteClimate


@dataclass
class Regime:
    hours: np.ndarray  # elapsed hours since start
    load: np.ndarray  # fraction of rated, 0..1.1 (0 when stopped)
    speed_ratio: np.ndarray  # shaft speed / nominal (0 when stopped)
    ambient: np.ndarray  # degC
    running: np.ndarray  # bool


def ar1(rng: np.random.Generator, n: int, phi: float, sigma: float) -> np.ndarray:
    """Stationary AR(1) with marginal std ``sigma``."""
    e = rng.normal(0.0, sigma * np.sqrt(1 - phi**2), n)
    return lfilter([1.0], [1.0, -phi], e)


def first_order_lag(x: np.ndarray, dt_h: float, tau_h: float) -> np.ndarray:
    """Exponential smoothing with time constant ``tau_h``, initialised at steady state."""
    alpha = 1.0 - np.exp(-dt_h / tau_h)
    y, _ = lfilter([alpha], [1.0, -(1.0 - alpha)], x, zi=[(1.0 - alpha) * x[0]])
    return y


def stoppage_mask(
    rng: np.random.Generator, n: int, dt_h: float, rate_per_day: float, hours: tuple[float, float]
) -> np.ndarray:
    """Boolean mask, True while the asset is stopped (blockage, feed starvation, shutdown)."""
    stopped = np.zeros(n, dtype=bool)
    n_events = rng.poisson(rate_per_day * n * dt_h / 24.0)
    for _ in range(n_events):
        start = int(rng.integers(0, n))
        dur = float(np.exp(rng.uniform(np.log(hours[0]), np.log(hours[1]))))
        stopped[start : start + max(1, round(dur / dt_h))] = True
    return stopped


def ambient_series(rng: np.random.Generator, hours: np.ndarray, climate: SiteClimate) -> np.ndarray:
    day = hours / 24.0
    seasonal = climate.seasonal_amp_c * np.cos(
        2 * np.pi * (day - climate.seasonal_peak_day) / 365.25
    )
    diurnal = climate.diurnal_amp_c * np.cos(2 * np.pi * ((hours % 24.0) - 15.0) / 24.0)
    weather = ar1(rng, len(hours), 0.98, 1.6)
    return climate.mean_c + seasonal + diurnal + weather


def generate_regime(
    rng: np.random.Generator,
    profile: AssetTypeProfile,
    n: int,
    dt_h: float,
    climate: SiteClimate,
) -> Regime:
    hours = np.arange(n) * dt_h
    hod = hours % 24.0
    dow = (hours // 24.0) % 7

    base = {"conveyor_idler": 0.70, "conveyor_pulley": 0.72, "primary_crusher": 0.62}.get(
        profile.asset_type.value, 0.68
    )
    shift = 1.0 + 0.07 * np.sin(2 * np.pi * (hod - 9.0) / 24.0)
    weekly = np.where(dow >= 5, 0.90, 1.0)

    # campaign effects: ore tonnage / hardness change every ~10 days (piecewise constant, smoothed)
    n_seg = max(2, int(n * dt_h / 24.0 / 10.0))
    levels = rng.normal(0.0, 0.06, n_seg + 1)
    campaign = np.repeat(levels, int(np.ceil(n / n_seg)))[:n]
    campaign = first_order_lag(campaign, dt_h, 12.0)

    noise = ar1(rng, n, np.exp(-dt_h / 10.0), 0.06)
    load = base * shift * weekly * (1.0 + campaign) + noise
    if profile.speed_mode == "engine":
        # haul cycles average out hourly: wider spread, queueing at shift change
        load = load + ar1(rng, n, np.exp(-dt_h / 3.0), 0.07)
        load = np.where(
            (hod >= 5.5) & (hod < 6.5) | (hod >= 17.5) & (hod < 18.5), load * 0.35, load
        )
    load = np.clip(load, 0.08, 1.08)

    # starved / low-feed periods
    low = stoppage_mask(rng, n, dt_h, 0.12, (3.0, 14.0))
    load = np.where(low, rng.uniform(0.12, 0.28, n), load)

    stop_rate, stop_hours = (
        (0.5, (1.0, 3.0)) if profile.speed_mode == "engine" else (0.10, (2.0, 30.0))
    )
    stopped = stoppage_mask(rng, n, dt_h, stop_rate, stop_hours)

    smooth_load = first_order_lag(np.where(stopped, 0.0, load), dt_h, 2.0)
    if profile.speed_mode == "fixed":
        speed = 1.0 - 0.012 * load + rng.normal(0, 0.002, n)
    elif profile.speed_mode == "vsd":
        speed = 0.86 + 0.14 * np.clip(smooth_load, 0, 1.05) + rng.normal(0, 0.003, n)
    else:  # engine
        speed = 0.68 + 0.50 * np.clip(smooth_load, 0, 1.05) + rng.normal(0, 0.01, n)

    load = np.where(stopped, 0.0, load)
    speed = np.where(stopped, 0.0, speed)
    ambient = ambient_series(rng, hours, climate)
    return Regime(hours=hours, load=load, speed_ratio=speed, ambient=ambient, running=~stopped)
