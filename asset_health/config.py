"""Configuration models. Every tunable lives here with its default and a one-line reason."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator

from asset_health.models import AssetType


class _Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------------- simulator


class DataFaultConfig(_Cfg):
    """Rates of *data* faults (independent of equipment degradation)."""

    dropout_per_channel_day: float = Field(1 / 90, ge=0)
    dropout_hours: tuple[float, float] = (3.0, 96.0)
    stuck_per_channel_day: float = Field(1 / 150, ge=0)
    stuck_hours: tuple[float, float] = (10.0, 160.0)
    sentinel_prob: float = Field(0.0007, ge=0, le=0.05)
    glitch_prob: float = Field(0.0006, ge=0, le=0.05)
    missing_row_prob: float = Field(0.004, ge=0, le=0.2)
    comm_outage_per_day: float = Field(1 / 45, ge=0)
    comm_outage_hours: tuple[float, float] = (2.0, 72.0)
    clock_drift_asset_fraction: float = Field(0.25, ge=0, le=1)
    clock_drift_s_per_day: tuple[float, float] = (15.0, 150.0)
    clock_steps_max: int = Field(3, ge=0)
    clock_step_s: tuple[float, float] = (60.0, 600.0)


class SiteClimate(_Cfg):
    """Hot-desert style climate. Illustrative, not a specific site."""

    mean_c: float = 29.0
    seasonal_amp_c: float = 9.0
    seasonal_peak_day: float = 200.0
    diurnal_amp_c: float = 7.0


class SimConfig(_Cfg):
    seed: int = 42
    start: datetime = datetime(2025, 1, 1)
    days: int = Field(270, ge=60, le=1500)
    sample_interval_s: int = Field(3600, ge=60)
    fleet: dict[AssetType, int] = Field(
        default_factory=lambda: {
            AssetType.CONVEYOR_IDLER: 3,
            AssetType.CONVEYOR_PULLEY: 3,
            AssetType.PRIMARY_CRUSHER: 3,
            AssetType.SLURRY_PUMP: 4,
            AssetType.HAUL_TRUCK_ENGINE: 3,
        }
    )
    failure_fraction: float = Field(0.65, ge=0, le=1)
    degradation_scale: float = Field(
        1.0,
        gt=0,
        le=2.0,
        description=(
            "Multiplies every injected failure's signal amplitude (1.0 = as designed). The "
            "benchmark's severity sweep uses 0.5, 0.3 and 0.15 to show how detectability falls "
            "as the signature shrinks; failure times are unchanged."
        ),
    )
    baseline_days: int = Field(21, ge=7, le=60)
    settle_hours: int = Field(48, ge=0, description="Run-in after commissioning / repair.")
    service_interval_days: tuple[float, float] = (80.0, 110.0)
    faults: DataFaultConfig = Field(default_factory=DataFaultConfig)
    climate: SiteClimate = Field(default_factory=SiteClimate)

    @model_validator(mode="after")
    def _check(self) -> SimConfig:
        if sum(self.fleet.values()) < 1:
            raise ValueError("fleet must contain at least one asset")
        if self.days < self.baseline_days + 40:
            raise ValueError("days must exceed baseline_days by at least 40")
        return self

    @property
    def n_samples(self) -> int:
        return int(self.days * 86400 // self.sample_interval_s)

    @property
    def dt_hours(self) -> float:
        return self.sample_interval_s / 3600.0


# --------------------------------------------------------------------------------- ingest


class IngestConfig(_Cfg):
    """Data-quality thresholds. Outcomes are explicit: every check returns pass/warn/fail."""

    sample_interval_s: int = Field(3600, ge=60)
    stuck_min_run: int = Field(8, ge=3, description="Identical consecutive readings => stuck.")
    spike_ratio: float = Field(3.5, gt=1.0, description="Isolated upward jump vs trailing median.")
    spike_history: int = Field(48, ge=8)
    spike_min_history: int = Field(8, ge=3)
    min_channel_valid: float = Field(0.5, ge=0, le=1)
    warn_channel_valid: float = Field(0.9, ge=0, le=1)
    warn_missing_fraction: float = Field(0.05, ge=0, le=1)
    fail_missing_fraction: float = Field(0.40, ge=0, le=1)
    clock_drift_warn_s_per_day: float = 20.0
    clock_drift_fail_s_per_day: float = 600.0
    advisory_hours: float = Field(24.0, gt=0)
    min_rows: int = Field(24 * 14, ge=24)
    stopped_speed_frac: float = Field(0.5, gt=0, le=1)
    # ---- structure of the export (checked before any grid is built; failures block the asset)
    max_backwards_fraction: float = Field(
        0.01,
        ge=0,
        lt=0.5,
        description=(
            "Backward timestamp steps tolerated in file order (logger clock corrections). More "
            "than this, and not a clean newest-first file, blocks the asset as out of order."
        ),
    )
    max_conflicting_fraction: float = Field(
        0.01,
        ge=0,
        lt=0.5,
        description="Rows sharing a timestamp with different values (a second collector).",
    )
    cadence_tolerance: float = Field(
        0.1, gt=0, lt=1, description="Allowed relative gap between median and configured cadence."
    )
    resample: bool = Field(
        False,
        description=(
            "Opt-in: average a finer cadence onto the configured grid instead of blocking. "
            "Resampling uses the reported timestamps, so logger clock drift is not corrected."
        ),
    )
    max_retime_overrun_h: float = Field(
        24.0,
        gt=0,
        description=(
            "The rebuilt grid may end at most this long after the last reported timestamp "
            "(plus the measured drift); otherwise the asset is blocked, so no alert can be dated "
            "after the data ends."
        ),
    )
    # ---- vibration-sensor integrity (loose mount, cable fault): erratic hour-to-hour readings
    erratic_window_h: int = Field(8, ge=4, description="Running hours in the roughness window.")
    erratic_ratio: float = Field(
        4.0, gt=1, description="Roughness above this multiple of its trailing median is erratic."
    )
    erratic_floor: float = Field(
        0.08, gt=0, description="Minimum median |hour-to-hour log change| treated as erratic."
    )
    erratic_reference_h: int = Field(
        14 * 24, ge=48, description="Trailing window for the roughness reference."
    )
    gating: bool = Field(
        True, description="False = ablation: skip range/stuck/spike masking (never in production)."
    )


# --------------------------------------------------------------------------------- features


class FeatureConfig(_Cfg):
    baseline_days: int = Field(21, ge=7, le=90)
    settle_hours: int = Field(48, ge=0)
    min_baseline_rows: int = Field(120, ge=30)
    running_speed_frac: float = Field(0.5, gt=0, le=1)
    min_load_pct: float = Field(5.0, ge=0)
    smooth_h: int = Field(6, ge=1)
    med_h: int = Field(24, ge=2)
    slope_h: int = Field(168, ge=24)
    sigma_floor_log: float = Field(0.04, gt=0, description="Min residual sigma, log channels.")
    sigma_floor_temp_c: float = Field(1.0, gt=0)
    sigma_inflate: float = Field(
        1.25, ge=1.0, description="Baseline sigma multiplier: 3 weeks under-states later variation."
    )
    envelope_margin: float = Field(
        0.25, ge=0, description="Operating-envelope widening, as a fraction of the baseline span."
    )
    ridge_alpha: float = Field(1.0, gt=0)
    hold_hours_after_service: int = Field(36, ge=0)
    regime_normalise: bool = Field(
        True, description="False = ablation: residuals against a constant baseline level only."
    )


# --------------------------------------------------------------------------------- detect


class DetectConfig(_Cfg):
    cusum_k: float = Field(1.5, gt=0, description="Reference value (slack), in smoothed-z units.")
    cusum_h: float = Field(60.0, gt=0, description="Decision interval (z-hours).")
    if_trees: int = Field(30, ge=10)
    if_quantile: float = Field(0.995, gt=0.5, lt=1)
    if_margin: float = Field(1.25, ge=1.0)
    weight_cusum: float = 0.4
    weight_if: float = 0.3
    weight_rule: float = 0.3
    clip: float = Field(3.0, gt=1.0)
    min_valid_fraction: float = Field(0.4, ge=0, le=1, description="Of monitored channels.")
    random_state: int = 7


# --------------------------------------------------------------------------------- RUL


class RULConfig(_Cfg):
    """Remaining-useful-life estimation. All durations in days unless stated."""

    top_k: int = Field(3, ge=1, description="Channels averaged into the health index.")
    hi_min: float = Field(
        2.0, gt=0, description="Below this the asset is not considered degrading."
    )
    trend_window_days: int = Field(8, ge=3, le=30)
    min_points: int = Field(5, ge=3, description="Daily samples needed for a trend.")
    min_slope_per_day: float = Field(0.05, gt=0, description="Health-index units per day.")
    slope_ci: float = Field(0.8, gt=0.5, lt=1.0, description="Theil-Sen interval level.")
    horizon_days: float = Field(90.0, gt=1.0, description="RUL is never extrapolated beyond this.")
    mc_draws: int = Field(400, ge=50)
    quantiles: tuple[float, float, float] = (0.1, 0.5, 0.9)
    target_coverage: float = Field(0.8, gt=0.5, lt=1.0)
    max_staleness_h: int = Field(36, ge=1, description="Newest health-index value allowed.")


# --------------------------------------------------------------------------------- alerting


class PolicyConfig(_Cfg):
    """Alert policy. ``alert_threshold`` and ``cost_exponent`` are tuned on development seeds by
    ``benchmarks/run.py`` (see ``evaluate.tuning``); the defaults here are those tuned values."""

    alert_threshold: float = Field(1.0, gt=0, description="Fused score that opens an alert.")
    cost_exponent: float = Field(
        0.0, ge=0, le=1.0, description="Threshold scales with (cost multiplier) ** -exponent."
    )
    min_detectors: int = Field(2, ge=1, le=3, description="Detectors that must agree.")
    confirm_hours: int = Field(4, ge=1, description="Consecutive valid hours above threshold.")
    clear_ratio: float = Field(0.6, gt=0, lt=1, description="Hysteresis: clear below this x thr.")
    clear_hours: int = Field(72, ge=1, description="Consecutive valid hours below clear level.")
    watch_ratio: float = Field(0.6, gt=0, lt=1, description="WATCH when score >= this x thr.")
    critical_days: float = Field(2.0, gt=0, description="CRITICAL when RUL p10 is below this.")
    urgent_margin_days: float = Field(4.0, ge=0, description="URGENT within lead + this.")
    checkpoint_hours: int = Field(24, ge=1, description="RUL re-estimation cadence in an alert.")
    no_rul_action_days: float = Field(
        3.0, ge=0, description="Act this long after opening if no RUL interval can be formed."
    )
    hypothesis_window_h: int = Field(72, ge=6)


class PipelineConfig(_Cfg):
    """Everything the stages before alerting need. Alerting and RUL carry their own configs."""

    ingest: IngestConfig = Field(default_factory=IngestConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    detect: DetectConfig = Field(default_factory=DetectConfig)
