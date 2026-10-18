"""Static engineering knowledge: asset-type profiles, channel definitions and physical limits.

Everything here is a modelling assumption for the synthetic fleet, written down in one place so
that the simulator, the ingest range checks and the feature pipeline agree. Replace the numbers
with the plant's own register values for a real deployment.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from asset_health.models import AssetSpec, AssetType, BearingGeometry, FailureMode

# ---- channel names -------------------------------------------------------------------------

REGIME_CHANNELS: tuple[str, ...] = ("load_pct", "speed_rpm", "ambient_c")

VIB_BAND_CHANNELS: tuple[str, ...] = (
    "vib_band_low",
    "vib_band_bpfo",
    "vib_band_bpfi",
    "vib_band_proc",
    "vib_band_hf",
)

SENSOR_CHANNELS: tuple[str, ...] = (
    "vib_rms",
    *VIB_BAND_CHANNELS,
    "temp_c",
    "current_a",
    "power_kw",
    "pressure_kpa",
    "flow_m3h",
)

ALL_CHANNELS: tuple[str, ...] = (*REGIME_CHANNELS, *SENSOR_CHANNELS)

# Bands are defined in orders (multiples of shaft speed). Half-width of a defect window,
# as a fraction of the centre order. 4% keeps every window disjoint for all asset types
# (asserted in tests/test_simulate_physics.py).
LINE_WINDOW_FRAC = 0.04
LOW_BAND_ORDERS = (0.5, 2.5)
HF_BAND_ORDERS = (20.0, 32.0)
SPECTRUM_MAX_ORDER = 32.0
SPECTRUM_BINS = 512

# Direction in which a residual is considered "worse".
Direction = Literal["up", "both"]

CHANNEL_DIRECTION: dict[str, Direction] = {
    "vib_rms": "up",
    "vib_band_low": "up",
    "vib_band_bpfo": "up",
    "vib_band_bpfi": "up",
    "vib_band_proc": "up",
    "vib_band_hf": "up",
    "temp_c": "up",
    "current_a": "both",
    "power_kw": "both",
    "pressure_kpa": "both",
    "flow_m3h": "both",
}

CHANNEL_UNITS: dict[str, str] = {
    "load_pct": "%",
    "speed_rpm": "rpm",
    "ambient_c": "degC",
    "vib_rms": "mm/s",
    "vib_band_low": "(mm/s)^2",
    "vib_band_bpfo": "(mm/s)^2",
    "vib_band_bpfi": "(mm/s)^2",
    "vib_band_proc": "(mm/s)^2",
    "vib_band_hf": "(mm/s)^2",
    "temp_c": "degC",
    "current_a": "A",
    "power_kw": "kW",
    "pressure_kpa": "kPa",
    "flow_m3h": "m3/h",
}

CHANNEL_DESCRIPTIONS: dict[str, str] = {
    "vib_rms": "overall vibration velocity RMS",
    "vib_band_low": "low-order band energy (imbalance, misalignment, looseness)",
    "vib_band_bpfo": "outer-race defect band energy (BPFO and 2xBPFO)",
    "vib_band_bpfi": "inner-race defect band energy (BPFI and 2xBPFI)",
    "vib_band_proc": "process-line band energy (vane pass, mesh or firing order)",
    "vib_band_hf": "high-frequency broadband energy",
    "temp_c": "monitored temperature",
    "current_a": "motor current",
    "power_kw": "power",
    "pressure_kpa": "discharge pressure",
    "flow_m3h": "flow",
    "bpfo_share": "outer-race band share of total vibration energy",
    "bpfi_share": "inner-race band share of total vibration energy",
    "hf_share": "high-frequency share of total vibration energy",
    "proc_share": "process-line share of total vibration energy",
    "low_share": "low-order share of total vibration energy",
}

# Hard physical limits used by ingest (out-of-range => sentinel / sensor fault, never a reading).
# Per-type scaling happens in ``channel_limits``.
_GENERIC_LIMITS: dict[str, tuple[float, float]] = {
    "load_pct": (0.0, 120.0),
    "ambient_c": (-10.0, 60.0),
    "vib_rms": (0.0, 60.0),
    "vib_band_low": (0.0, 3600.0),
    "vib_band_bpfo": (0.0, 3600.0),
    "vib_band_bpfi": (0.0, 3600.0),
    "vib_band_proc": (0.0, 3600.0),
    "vib_band_hf": (0.0, 3600.0),
    "temp_c": (-10.0, 160.0),
}


class AssetTypeProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    asset_type: AssetType
    label: str
    shaft_hz: float
    speed_mode: Literal["fixed", "vsd", "engine"]
    bearing: BearingGeometry
    proc_order: float | None
    proc_label: str | None
    base_vib_rms: float
    channels: tuple[str, ...]
    rated_power_kw: float | None
    rated_flow_m3h: float | None = None
    rated_pressure_kpa: float | None = None
    voltage_v: float | None = None
    temp_idle_rise_c: float
    temp_load_rise_c: float
    temp_tau_h: float
    ambient_coupling: float = 1.0  # 1 = passively cooled (tracks ambient), 0 = thermostat-regulated
    failure_modes: tuple[tuple[FailureMode, float], ...]
    degradation_days: dict[FailureMode, tuple[float, float]]
    typical_names: tuple[str, ...]
    cost_multiplier: float
    monitored: tuple[str, ...]
    life_weibull: tuple[float, float] = (2.5, 1000.0)
    """Wear-out life of the monitored component: Weibull (shape beta > 1, scale eta in days).
    Illustrative. It sets each component's age (and so what age-based replacement can prevent);
    it does not change the simulated signals."""

    @property
    def typical_degradation_days(self) -> float:
        """Prior P-F interval for the asset class: failure-mode-weighted mean time from first
        detectable degradation to functional failure (what an FMEA/RCM study would record)."""
        total = sum(w for _, w in self.failure_modes)
        return sum(w * sum(self.degradation_days[m]) / 2.0 for m, w in self.failure_modes) / total


def _bearing(n: int, d: float, big_d: float, angle: float = 0.0) -> BearingGeometry:
    return BearingGeometry(
        n_elements=n, ball_diameter_mm=d, pitch_diameter_mm=big_d, contact_angle_deg=angle
    )


_BANDS_NO_PROC = ("vib_band_low", "vib_band_bpfo", "vib_band_bpfi", "vib_band_hf")
_BANDS_ALL = VIB_BAND_CHANNELS

PROFILES: dict[AssetType, AssetTypeProfile] = {
    AssetType.CONVEYOR_IDLER: AssetTypeProfile(
        asset_type=AssetType.CONVEYOR_IDLER,
        life_weibull=(2.0, 1500.0),
        label="Conveyor idler roller set",
        shaft_hz=8.2,  # belt 4.0 m/s over a 152 mm roller
        speed_mode="vsd",
        bearing=_bearing(8, 9.5, 38.0),
        proc_order=None,
        proc_label=None,
        base_vib_rms=1.3,
        channels=("vib_rms", *_BANDS_NO_PROC, "temp_c"),
        rated_power_kw=None,
        temp_idle_rise_c=6.0,
        temp_load_rise_c=14.0,
        temp_tau_h=1.5,
        failure_modes=((FailureMode.BEARING_SPALL, 1.0),),
        degradation_days={FailureMode.BEARING_SPALL: (20.0, 45.0)},
        typical_names=("CV-{n:02d} Idler Station",),
        cost_multiplier=0.25,
        monitored=("vib_rms", *_BANDS_NO_PROC, "temp_c"),
    ),
    AssetType.CONVEYOR_PULLEY: AssetTypeProfile(
        asset_type=AssetType.CONVEYOR_PULLEY,
        life_weibull=(2.5, 1200.0),
        label="Conveyor drive pulley",
        shaft_hz=1.3,  # ~78 rpm
        speed_mode="vsd",
        bearing=_bearing(16, 30.0, 170.0, 10.0),
        proc_order=None,
        proc_label=None,
        base_vib_rms=1.9,
        channels=("vib_rms", *_BANDS_NO_PROC, "temp_c", "current_a", "power_kw"),
        rated_power_kw=450.0,
        voltage_v=690.0,
        temp_idle_rise_c=8.0,
        temp_load_rise_c=18.0,
        temp_tau_h=2.5,
        failure_modes=((FailureMode.BEARING_SPALL, 1.0),),
        degradation_days={FailureMode.BEARING_SPALL: (25.0, 55.0)},
        typical_names=("CV-{n:02d} Head Pulley",),
        cost_multiplier=1.5,
        monitored=("vib_rms", *_BANDS_NO_PROC, "temp_c", "current_a", "power_kw"),
    ),
    AssetType.PRIMARY_CRUSHER: AssetTypeProfile(
        asset_type=AssetType.PRIMARY_CRUSHER,
        life_weibull=(3.0, 800.0),
        label="Primary crusher",
        shaft_hz=4.0,  # countershaft
        speed_mode="fixed",
        bearing=_bearing(14, 40.0, 210.0, 12.0),
        proc_order=3.4,
        proc_label="mantle impact order",
        base_vib_rms=4.6,
        channels=("vib_rms", *_BANDS_ALL, "temp_c", "current_a", "power_kw"),
        rated_power_kw=1000.0,
        voltage_v=6600.0,
        temp_idle_rise_c=10.0,
        temp_load_rise_c=22.0,
        temp_tau_h=3.5,
        failure_modes=((FailureMode.LINER_WEAR, 0.7), (FailureMode.BEARING_SPALL, 0.3)),
        degradation_days={
            FailureMode.LINER_WEAR: (45.0, 85.0),
            FailureMode.BEARING_SPALL: (25.0, 55.0),
        },
        typical_names=("CR-{n:02d} Primary Crusher",),
        cost_multiplier=4.0,
        monitored=("vib_rms", *_BANDS_ALL, "temp_c", "current_a", "power_kw"),
    ),
    AssetType.SLURRY_PUMP: AssetTypeProfile(
        asset_type=AssetType.SLURRY_PUMP,
        life_weibull=(3.0, 700.0),
        label="Slurry pump",
        shaft_hz=24.7,  # 1480 rpm
        speed_mode="vsd",
        bearing=_bearing(10, 12.0, 66.0),
        proc_order=5.0,
        proc_label="vane-pass order (5 vanes)",
        base_vib_rms=3.3,
        channels=(
            "vib_rms",
            *_BANDS_ALL,
            "temp_c",
            "current_a",
            "power_kw",
            "pressure_kpa",
            "flow_m3h",
        ),
        rated_power_kw=250.0,
        rated_flow_m3h=900.0,
        rated_pressure_kpa=500.0,
        voltage_v=690.0,
        temp_idle_rise_c=9.0,
        temp_load_rise_c=20.0,
        temp_tau_h=2.0,
        failure_modes=((FailureMode.IMPELLER_WEAR, 0.7), (FailureMode.BEARING_SPALL, 0.3)),
        degradation_days={
            FailureMode.IMPELLER_WEAR: (40.0, 75.0),
            FailureMode.BEARING_SPALL: (20.0, 45.0),
        },
        typical_names=("SP-{n:02d} Cyclone Feed Pump",),
        cost_multiplier=1.0,
        monitored=(
            "vib_rms",
            *_BANDS_ALL,
            "temp_c",
            "current_a",
            "power_kw",
            "pressure_kpa",
            "flow_m3h",
        ),
    ),
    AssetType.HAUL_TRUCK_ENGINE: AssetTypeProfile(
        asset_type=AssetType.HAUL_TRUCK_ENGINE,
        life_weibull=(2.0, 1000.0),
        label="Haul-truck engine",
        shaft_hz=30.0,  # 1800 rpm nominal
        speed_mode="engine",
        bearing=_bearing(9, 11.0, 50.0),
        proc_order=3.0,
        proc_label="cylinder firing order",
        base_vib_rms=5.2,
        channels=("vib_rms", *_BANDS_ALL, "temp_c", "power_kw"),
        rated_power_kw=1800.0,
        temp_idle_rise_c=0.0,
        temp_load_rise_c=0.0,
        temp_tau_h=0.5,
        ambient_coupling=0.0,
        failure_modes=((FailureMode.OVERHEATING, 1.0),),
        degradation_days={FailureMode.OVERHEATING: (14.0, 32.0)},
        typical_names=("HT-{n:03d} Engine",),
        cost_multiplier=2.0,
        monitored=("vib_rms", *_BANDS_ALL, "temp_c", "power_kw"),
    ),
}

# Derived (share) features that the feature stage adds on top of the raw channels.
SHARE_FEATURES: dict[str, str] = {
    "vib_band_low": "low_share",
    "vib_band_bpfo": "bpfo_share",
    "vib_band_bpfi": "bpfi_share",
    "vib_band_proc": "proc_share",
    "vib_band_hf": "hf_share",
}


def profile_for(asset_type: AssetType) -> AssetTypeProfile:
    return PROFILES[asset_type]


def band_orders(spec: AssetSpec | AssetTypeProfile) -> dict[str, list[tuple[float, float]]]:
    """Order-domain windows (lo, hi) for each vibration band of an asset."""
    b = spec.bearing

    def lines(centre: float, harmonics: int = 2) -> list[tuple[float, float]]:
        return [
            (h * centre * (1 - LINE_WINDOW_FRAC), h * centre * (1 + LINE_WINDOW_FRAC))
            for h in range(1, harmonics + 1)
        ]

    out: dict[str, list[tuple[float, float]]] = {
        "vib_band_low": [LOW_BAND_ORDERS],
        "vib_band_bpfo": lines(b.bpfo_order),
        "vib_band_bpfi": lines(b.bpfi_order),
        "vib_band_hf": [HF_BAND_ORDERS],
    }
    if spec.proc_order is not None:
        out["vib_band_proc"] = lines(spec.proc_order)
    return out


def spec_from_profile(
    profile: AssetTypeProfile, asset_id: str, name: str, baseline_scale: float = 1.0
) -> AssetSpec:
    return AssetSpec(
        asset_id=asset_id,
        asset_type=profile.asset_type,
        name=name,
        shaft_hz=profile.shaft_hz,
        bearing=profile.bearing,
        proc_order=profile.proc_order,
        channels=profile.channels,
        rated_power_kw=profile.rated_power_kw,
        baseline_scale=baseline_scale,
    )


def channel_limits(spec: AssetSpec) -> dict[str, tuple[float, float]]:
    """Physical (min, max) limits for every channel the asset reports, regime channels included."""
    profile = PROFILES[spec.asset_type]
    limits: dict[str, tuple[float, float]] = {
        c: _GENERIC_LIMITS[c] for c in _GENERIC_LIMITS if c in (*REGIME_CHANNELS, *spec.channels)
    }
    limits["speed_rpm"] = (0.0, 1.25 * spec.nominal_rpm)
    if "current_a" in spec.channels and profile.rated_power_kw and profile.voltage_v:
        limits["current_a"] = (0.0, 1.6 * rated_current(profile))
    if "power_kw" in spec.channels and profile.rated_power_kw:
        limits["power_kw"] = (0.0, 1.4 * profile.rated_power_kw)
    if "pressure_kpa" in spec.channels and profile.rated_pressure_kpa:
        limits["pressure_kpa"] = (0.0, 1.8 * profile.rated_pressure_kpa)
    if "flow_m3h" in spec.channels and profile.rated_flow_m3h:
        limits["flow_m3h"] = (0.0, 1.5 * profile.rated_flow_m3h)
    return limits


def rated_current(profile: AssetTypeProfile) -> float:
    """Rated line current (A) from rated power, voltage and an assumed 0.88 pf * 0.95 efficiency."""
    assert profile.rated_power_kw and profile.voltage_v
    return profile.rated_power_kw * 1000.0 / (3**0.5 * profile.voltage_v * 0.88 * 0.95)
