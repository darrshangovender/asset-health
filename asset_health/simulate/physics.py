"""Vibration spectrum synthesis (SYNTHETIC).

The simulator works in the *order domain*: spectra are order-tracked (resampled against shaft
speed), so a bearing defect line sits at a fixed multiple of shaft frequency (BPFO, BPFI ...)
whatever the speed. Band energies are integrals of the simulated spectrum over order windows,
exactly what an edge analyser would publish to a historian.

Spectrum model (power spectral density, (mm/s)^2 per order):
    floor      broadband, ~1/(1+o)^1.1 plus a flat part, with a high-frequency resonance hump
    shaft      lines at 1x, 2x, 3x
    process    lines at the process order (vane pass, mesh, firing) and its 2nd harmonic
    defects    lines at bearing defect orders (+ sidebands for inner-race defects)
    noise      chi-square (gamma) multiplicative estimation noise per bin, as from averaged FFTs
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from asset_health.catalog import (
    HF_BAND_ORDERS,
    SPECTRUM_BINS,
    SPECTRUM_MAX_ORDER,
    VIB_BAND_CHANNELS,
    band_orders,
)
from asset_health.models import AssetSpec

DELTA_ORDER = SPECTRUM_MAX_ORDER / SPECTRUM_BINS
ORDER_AXIS = (np.arange(SPECTRUM_BINS) + 0.5) * DELTA_ORDER
LINE_SIGMA_ORDERS = 0.05  # Gaussian line width (about 0.8 bin)
AVERAGES = 24  # spectral averages -> per-bin relative noise 1/sqrt(24)


def _unit_area(v: np.ndarray) -> np.ndarray:
    return v / (v.sum() * DELTA_ORDER)


FLOOR_BASE = _unit_area(1.0 / (1.0 + ORDER_AXIS) ** 1.1 + 0.04)
FLOOR_HUMP = _unit_area(np.exp(-0.5 * ((ORDER_AXIS - 26.0) / 3.5) ** 2))


def line_kernel(order: float) -> np.ndarray:
    """Unit-power Gaussian line at ``order`` (integrates to 1 over the order axis)."""
    k = np.exp(-0.5 * ((ORDER_AXIS - order) / LINE_SIGMA_ORDERS) ** 2)
    return _unit_area(k)


def band_matrix(spec: AssetSpec) -> tuple[np.ndarray, list[str]]:
    """Membership matrix (bins x bands) for the bands this asset reports."""
    windows = band_orders(spec)
    names = [b for b in VIB_BAND_CHANNELS if b in windows]
    m = np.zeros((SPECTRUM_BINS, len(names)))
    for j, name in enumerate(names):
        for lo, hi in windows[name]:
            m[(ORDER_AXIS >= lo) & (ORDER_AXIS < hi), j] = 1.0
    return m, names


def hf_bin_mask() -> np.ndarray:
    return (ORDER_AXIS >= HF_BAND_ORDERS[0]) & (ORDER_AXIS < HF_BAND_ORDERS[1])


@dataclass
class SpectralDrivers:
    """Time-varying multipliers (all arrays of length T) that act on the spectrum."""

    machine_mult: np.ndarray  # overall regime-driven power (load/speed/stopped)
    proc_mult: np.ndarray  # extra process-line multiplier from regime
    defect_mult: np.ndarray  # regime scaling of defect-line power
    disturbance: np.ndarray  # unmodelled common-mode power factor (ore hardness, slugging)
    hf_mult: np.ndarray = field(default=None)  # type: ignore[assignment]
    floor_mult: np.ndarray = field(default=None)  # type: ignore[assignment]
    shaft_mult: np.ndarray = field(default=None)  # type: ignore[assignment]
    proc_fault_mult: np.ndarray = field(default=None)  # type: ignore[assignment]
    outer_power: np.ndarray = field(default=None)  # type: ignore[assignment]
    inner_power: np.ndarray = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        n = len(self.machine_mult)
        for name in ("hf_mult", "floor_mult", "shaft_mult", "proc_fault_mult"):
            if getattr(self, name) is None:
                setattr(self, name, np.ones(n))
        for name in ("outer_power", "inner_power"):
            if getattr(self, name) is None:
                setattr(self, name, np.zeros(n))


@dataclass
class VibrationResult:
    rms: np.ndarray
    bands: dict[str, np.ndarray]
    spectra: np.ndarray | None = None  # (T, bins) only when requested


def synthesize_vibration(
    rng: np.random.Generator,
    spec: AssetSpec,
    base_rms: float,
    drivers: SpectralDrivers,
    keep_spectra: bool = False,
) -> VibrationResult:
    """Synthesize rms + band energies for T samples.

    Healthy power budget at nominal conditions: floor 45-55%, shaft lines 35-45%, process 0-20%.
    ``outer_power`` / ``inner_power`` are defect-line powers as a fraction of the healthy total.
    """
    t = len(drivers.machine_mult)
    p_total = (base_rms * spec.baseline_scale) ** 2
    has_proc = spec.proc_order is not None
    f_floor, f_shaft, f_proc = (0.45, 0.35, 0.20) if has_proc else (0.55, 0.45, 0.0)
    hump_share = 0.18

    # --- line bank: shaft (3), process (2), outer (3), inner (1 + 2 sidebands + 2x)
    orders: list[float] = [1.0, 2.0, 3.0]
    if has_proc:
        orders += [spec.proc_order, 2.0 * spec.proc_order]  # type: ignore[operator]
    bpfo, bpfi = spec.bearing.bpfo_order, spec.bearing.bpfi_order
    orders += [bpfo, 2 * bpfo, 3 * bpfo]
    orders += [bpfi, bpfi - 1.0, bpfi + 1.0, 2 * bpfi]
    kernels = np.stack([line_kernel(o) for o in orders])  # (J, bins)

    amp = np.zeros((t, len(orders)))
    reg = drivers.machine_mult * drivers.disturbance
    shaft_p = f_shaft * p_total * reg * drivers.shaft_mult
    amp[:, 0] = 0.60 * shaft_p
    amp[:, 1] = 0.28 * shaft_p
    amp[:, 2] = 0.12 * shaft_p
    col = 3
    if has_proc:
        proc_p = f_proc * p_total * reg * drivers.proc_mult * drivers.proc_fault_mult
        amp[:, col] = 0.75 * proc_p
        amp[:, col + 1] = 0.25 * proc_p
        col += 2
    outer = p_total * drivers.outer_power * drivers.defect_mult * drivers.disturbance
    amp[:, col] = 0.55 * outer
    amp[:, col + 1] = 0.30 * outer
    amp[:, col + 2] = 0.15 * outer
    col += 3
    inner = p_total * drivers.inner_power * drivers.defect_mult * drivers.disturbance
    amp[:, col] = 0.50 * inner
    amp[:, col + 1] = 0.15 * inner
    amp[:, col + 2] = 0.15 * inner
    amp[:, col + 3] = 0.20 * inner

    floor_p = f_floor * p_total * reg * drivers.floor_mult
    base_p = floor_p * (1.0 - hump_share)
    hump_p = floor_p * hump_share * drivers.hf_mult

    spectrum = (
        amp @ kernels
        + base_p[:, None] * FLOOR_BASE[None, :]
        + hump_p[:, None] * FLOOR_HUMP[None, :]
    )
    noise = rng.gamma(AVERAGES, 1.0 / AVERAGES, size=spectrum.shape)
    spectrum *= noise

    total = spectrum.sum(axis=1) * DELTA_ORDER
    mat, names = band_matrix(spec)
    energy = (spectrum @ mat) * DELTA_ORDER
    bands = {n: energy[:, j] for j, n in enumerate(names)}
    return VibrationResult(
        rms=np.sqrt(total), bands=bands, spectra=spectrum if keep_spectra else None
    )
