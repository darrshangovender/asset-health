"""Seeded synthetic fleet generator (SYNTHETIC).

``generate_fleet(SimConfig)`` is a pure function of the config: same config, same bytes out.
It returns the *dirty* data a plant would export (reported timestamps, gaps, stuck sensors ...),
the CMMS-style maintenance log the pipeline may see, and the ground truth it may not.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import asset_health
from asset_health.catalog import (
    PROFILES,
    AssetTypeProfile,
    rated_current,
    spec_from_profile,
)
from asset_health.config import SimConfig
from asset_health.models import (
    SYNTHETIC_NOTICE,
    AssetSpec,
    AssetType,
    DataFaultEvent,
    FailureEvent,
    FailureMode,
    FleetManifest,
    GroundTruth,
    MaintenanceEvent,
)
from asset_health.simulate.degradation import Effects, effects_for
from asset_health.simulate.faults import apply_data_faults
from asset_health.simulate.physics import SpectralDrivers, synthesize_vibration
from asset_health.simulate.regime import (
    Regime,
    ar1,
    first_order_lag,
    generate_regime,
)

ID_PREFIX = {
    AssetType.CONVEYOR_IDLER: "IDL",
    AssetType.CONVEYOR_PULLEY: "PUL",
    AssetType.PRIMARY_CRUSHER: "CRU",
    AssetType.SLURRY_PUMP: "PMP",
    AssetType.HAUL_TRUCK_ENGINE: "ENG",
}

QUANT = {  # decimals kept by the (simulated) historian
    "load_pct": 1,
    "speed_rpm": 1,
    "ambient_c": 1,
    "vib_rms": 3,
    "vib_band_low": 4,
    "vib_band_bpfo": 4,
    "vib_band_bpfi": 4,
    "vib_band_proc": 4,
    "vib_band_hf": 4,
    "temp_c": 1,
    "current_a": 1,
    "power_kw": 1,
    "pressure_kpa": 1,
    "flow_m3h": 1,
}


@dataclass
class SyntheticFleet:
    manifest: FleetManifest
    raw: dict[str, pd.DataFrame]
    maintenance: list[MaintenanceEvent]
    truth: GroundTruth
    latent: dict[str, pd.DataFrame] = field(default_factory=dict)

    @property
    def specs(self) -> list[AssetSpec]:
        return self.manifest.assets

    def spec(self, asset_id: str) -> AssetSpec:
        for s in self.manifest.assets:
            if s.asset_id == asset_id:
                return s
        raise KeyError(asset_id)


# ------------------------------------------------------------------------------ planning


def build_specs(cfg: SimConfig, rng: np.random.Generator) -> list[AssetSpec]:
    specs: list[AssetSpec] = []
    for atype, count in cfg.fleet.items():
        profile = PROFILES[atype]
        for k in range(1, count + 1):
            n_label = k + (100 if atype == AssetType.HAUL_TRUCK_ENGINE else 0)
            scale = float(np.exp(rng.normal(0.0, 0.22)))
            specs.append(
                spec_from_profile(
                    profile,
                    asset_id=f"{ID_PREFIX[atype]}-{k:02d}",
                    name=profile.typical_names[0].format(n=n_label),
                    baseline_scale=scale,
                )
            )
    return specs


def _pick_mode(rng: np.random.Generator, profile: AssetTypeProfile) -> FailureMode:
    modes, weights = zip(*profile.failure_modes, strict=True)
    w = np.array(weights) / np.sum(weights)
    return modes[int(rng.choice(len(modes), p=w))]


def plan_failures(
    rng: np.random.Generator, cfg: SimConfig, specs: list[AssetSpec]
) -> dict[str, FailureEvent]:
    n = len(specs)
    n_fail = round(cfg.failure_fraction * n)
    chosen: dict[int, FailureMode] = {}
    all_modes = {m for s in specs for m, _ in PROFILES[s.asset_type].failure_modes}
    for mode in sorted(all_modes, key=lambda m: m.value):
        if len(chosen) >= n_fail:
            break
        eligible = [
            i
            for i in rng.permutation(n)
            if i not in chosen and mode in dict(PROFILES[specs[i].asset_type].failure_modes)
        ]
        if eligible:
            chosen[int(eligible[0])] = mode
    for i in rng.permutation(n):
        if len(chosen) >= n_fail:
            break
        if int(i) not in chosen:
            chosen[int(i)] = _pick_mode(rng, PROFILES[specs[int(i)].asset_type])

    events: dict[str, FailureEvent] = {}
    settle_days = cfg.settle_hours / 24.0
    for i, mode in sorted(chosen.items()):
        spec = specs[i]
        lo, hi = PROFILES[spec.asset_type].degradation_days[mode]
        dur = float(rng.uniform(lo, hi))
        earliest = cfg.baseline_days + settle_days + dur + 8.0
        latest = cfg.days - 22.0
        if earliest >= latest:
            continue
        fail_day = float(rng.uniform(earliest, latest))
        downtime = float(rng.uniform(36.0, 96.0))
        k = float(
            rng.uniform(2.0, 4.5) if mode == FailureMode.BEARING_SPALL else rng.uniform(1.2, 3.0)
        )
        amp = float(np.clip(np.exp(rng.normal(0.0, 0.3)), 0.55, 1.8)) * cfg.degradation_scale
        race = (
            ("outer" if rng.random() < 0.6 else "inner")
            if mode == FailureMode.BEARING_SPALL
            else None
        )
        failure = cfg.start + timedelta(days=fail_day)
        events[spec.asset_id] = FailureEvent(
            asset_id=spec.asset_id,
            mode=mode,
            race=race,
            onset=failure - timedelta(days=dur),
            failure=failure,
            shape_k=k,
            amplitude=amp,
            downtime_hours=downtime,
        )
    return events


def _weibull_sf(t: np.ndarray, beta: float, eta: float) -> np.ndarray:
    return np.exp(-((np.maximum(t, 0.0) / eta) ** beta))


def component_ages(
    rng: np.random.Generator,
    cfg: SimConfig,
    specs: list[AssetSpec],
    failures: dict[str, FailureEvent],
) -> dict[str, float]:
    """Age in days of each monitored component at the start of the record (SYNTHETIC).

    Components wear out: life ~ Weibull(beta > 1, eta) per asset class
    (``AssetTypeProfile.life_weibull``). The ages are those a fleet of such parts in steady
    operation would show, given what happens in the record:

    * a part that fails on day ``f`` of the record has a life ``L`` drawn from the Weibull
      conditioned on ``L > f`` (it was installed before the record began), so its age at the start
      is ``L - f``. Old parts fail early in the record, young ones late or not at all;
    * a part that survives the record has an age drawn with density proportional to
      ``S(age + days)`` (the steady-state age of parts that outlive the window).

    Ages never change the simulated signals; they decide what age-based (calendar) replacement
    could have prevented. A part fitted during the record is assumed to outlive it; the class lives
    are long enough that the chance it would not is small (see ``renewal_failure_probability``).
    """
    out: dict[str, float] = {}
    for spec in specs:
        beta, eta = PROFILES[spec.asset_type].life_weibull
        f = failures.get(spec.asset_id)
        if f is not None:
            f_day = (f.failure - cfg.start).total_seconds() / 86400.0
            u = float(rng.uniform(1e-12, 1.0))
            life = eta * ((f_day / eta) ** beta - np.log(u)) ** (1.0 / beta)
            out[spec.asset_id] = float(max(life - f_day, 0.0))
        else:
            grid = np.linspace(0.0, 6.0 * eta, 4001)
            dens = _weibull_sf(grid + cfg.days, beta, eta)
            cdf = np.cumsum(dens)
            cdf /= cdf[-1]
            out[spec.asset_id] = float(np.interp(rng.uniform(), cdf, grid))
    return out


def renewal_failure_probability(asset_type: AssetType, days: float) -> float:
    """P(a new component fails within ``days``): the error of assuming a part fitted during the
    record outlives it."""
    beta, eta = PROFILES[asset_type].life_weibull
    return float(1.0 - _weibull_sf(np.array([days]), beta, eta)[0])


def plan_services(
    rng: np.random.Generator, cfg: SimConfig, spec: AssetSpec, failure: FailureEvent | None
) -> list[MaintenanceEvent]:
    events: list[MaintenanceEvent] = []
    day = float(rng.uniform(45.0, 95.0))
    while day < cfg.days - 5:
        ts = cfg.start + timedelta(days=day)
        blocked = False
        if failure is not None:
            end = failure.failure + timedelta(hours=failure.downtime_hours + 72.0)
            blocked = failure.failure - timedelta(days=1) <= ts <= end
        if not blocked:
            events.append(
                MaintenanceEvent(
                    asset_id=spec.asset_id,
                    timestamp=ts,
                    kind="planned_service",
                    note="lubrication and inspection round",
                )
            )
        day += float(rng.uniform(*cfg.service_interval_days))
    return events


# ------------------------------------------------------------------------------ one asset


def _surges(rng: np.random.Generator, n: int, dt_h: float, rate_per_day: float) -> np.ndarray:
    out = np.ones(n)
    for _ in range(rng.poisson(rate_per_day * n * dt_h / 24.0)):
        s = int(rng.integers(0, n))
        d = max(1, round(rng.uniform(2.0, 6.0) / dt_h))
        out[s : s + d] = rng.uniform(1.3, 2.0)
    return out


def simulate_asset(
    rng: np.random.Generator,
    cfg: SimConfig,
    spec: AssetSpec,
    failure: FailureEvent | None,
    services: list[MaintenanceEvent],
    keep_spectra: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, MaintenanceEvent | None]:
    """Return ``(clean_measurements, latent, repair_event)`` on the true-time grid."""
    profile = PROFILES[spec.asset_type]
    n, dt_h = cfg.n_samples, cfg.dt_hours
    index = pd.date_range(cfg.start, periods=n, freq=f"{cfg.sample_interval_s}s")
    reg: Regime = generate_regime(rng, profile, n, dt_h, cfg.climate)
    hours = reg.hours
    start_ns = int(np.datetime64(cfg.start, "ns").astype("int64"))
    eff: Effects = effects_for(failure, hours, start_ns)

    running = reg.running.copy()
    load = reg.load.copy()
    speed = reg.speed_ratio.copy()

    repair: MaintenanceEvent | None = None
    perm_vib = np.ones(n)
    perm_temp = np.zeros(n)
    trans_vib = np.zeros(n)
    trans_temp = np.zeros(n)

    def at(ts: datetime) -> int:
        return round((ts - cfg.start).total_seconds() / cfg.sample_interval_s)

    if failure is not None:
        i_fail = at(failure.failure)
        i_rep = min(n, i_fail + round(failure.downtime_hours / dt_h))
        running[i_fail:i_rep] = False
        repair = MaintenanceEvent(
            asset_id=spec.asset_id,
            timestamp=cfg.start + timedelta(seconds=i_rep * cfg.sample_interval_s),
            kind="repair",
            note=f"component replaced after {failure.mode.value}",
        )
        if i_rep < n:
            # new component: slightly different signature and a run-in transient
            perm_vib[i_rep:] *= float(np.exp(rng.normal(0.0, 0.08)))
            perm_temp[i_rep:] += float(rng.normal(0.0, 0.8))
            since = hours[i_rep:] - hours[i_rep]
            trans_vib[i_rep:] += 0.30 * np.exp(-since / 14.0)
            trans_temp[i_rep:] += 4.0 * np.exp(-since / 14.0)
        # severity must not keep acting on the new component
        eff.severity[i_rep:] = 0.0
        for name in ("outer_power", "inner_power", "temp_delta"):
            getattr(eff, name)[i_rep:] = 0.0
        for name in (
            "hf_mult",
            "floor_mult",
            "shaft_mult",
            "proc_mult",
            "power_mult",
            "head_mult",
            "flow_mult",
            "eff_mult",
        ):
            getattr(eff, name)[i_rep:] = 1.0

    for sv in services:
        i = at(sv.timestamp)
        if not 0 <= i < n:
            continue
        running[i : i + 3] = False
        since = np.clip(hours[i:] - hours[i], 0, None)
        perm_vib[i:] *= 0.96
        perm_temp[i:] -= 0.4
        trans_vib[i:] += 0.20 * np.exp(-since / 12.0)
        trans_temp[i:] += 3.0 * np.exp(-since / 10.0)

    load = np.where(running, load, 0.0)
    speed = np.where(running, speed, 0.0)
    run_f = running.astype(float)

    # ---- vibration
    m_load = (0.35 + 0.65 * load) ** 1.6
    m_speed = np.clip(speed, 0.0, None) ** 2.2
    machine = np.maximum(m_load * m_speed, 0.002)
    machine = machine * perm_vib**2 * (1.0 + trans_vib) ** 2
    proc_reg = ((0.2 + 0.8 * load) / (0.35 + 0.65 * load)) ** 2
    defect_reg = np.maximum((0.3 + 0.7 * load) ** 2 * m_speed, 0.002)
    disturbance = np.exp(ar1(rng, n, np.exp(-dt_h / 60.0), 0.08))
    surge = np.ones(n)
    if spec.asset_type in (AssetType.PRIMARY_CRUSHER, AssetType.SLURRY_PUMP):
        surge = _surges(rng, n, dt_h, 1 / 18.0)
    drivers = SpectralDrivers(
        machine_mult=machine,
        proc_mult=proc_reg,
        defect_mult=defect_reg,
        disturbance=disturbance * surge,
        hf_mult=eff.hf_mult,
        floor_mult=eff.floor_mult,
        shaft_mult=eff.shaft_mult,
        proc_fault_mult=eff.proc_mult,
        outer_power=eff.outer_power,
        inner_power=eff.inner_power,
    )
    vib = synthesize_vibration(rng, spec, profile.base_vib_rms, drivers, keep_spectra)
    data: dict[str, np.ndarray] = {
        "vib_rms": vib.rms * (1.0 + rng.normal(0, 0.025, n)),
    }
    for band, arr in vib.bands.items():
        data[band] = arr * (1.0 + rng.normal(0, 0.04, n))

    # ---- temperature
    if spec.asset_type == AssetType.HAUL_TRUCK_ENGINE:
        t_eq = np.where(
            running,
            84.0 + 7.0 * load + 0.08 * (reg.ambient - 30.0) + eff.temp_delta,
            reg.ambient,
        )
        t_eq = t_eq + perm_temp * run_f + trans_temp * run_f
    else:
        rise = profile.temp_idle_rise_c + profile.temp_load_rise_c * (load**1.5) * np.sqrt(
            np.clip(speed, 0, None)
        )
        t_eq = reg.ambient + run_f * (rise + eff.temp_delta + perm_temp + trans_temp)
    temp = first_order_lag(t_eq, dt_h, profile.temp_tau_h) + rng.normal(0, 0.5, n)
    data["temp_c"] = temp

    # ---- power / current / hydraulics
    pr = profile.rated_power_kw
    power = None
    if spec.asset_type == AssetType.CONVEYOR_PULLEY:
        power = pr * (0.18 + 0.72 * load**1.1 * speed)
    elif spec.asset_type == AssetType.PRIMARY_CRUSHER:
        power = pr * (0.30 + 0.62 * load) * surge**0.5
        power = np.where(running, power, 0.0)
    elif spec.asset_type == AssetType.HAUL_TRUCK_ENGINE:
        power = pr * load * np.sqrt(np.clip(speed, 0, None)) * 0.95
    elif spec.asset_type == AssetType.SLURRY_PUMP:
        qr, hr = profile.rated_flow_m3h, profile.rated_pressure_kpa
        density = np.exp(ar1(rng, n, np.exp(-dt_h / 30.0), 0.025))
        q = qr * np.minimum(load, 1.05) * speed * eff.flow_mult
        head = hr * density * (1.35 * speed**2 - 0.35 * (q / qr) ** 2) * eff.head_mult
        head = np.where(running, np.maximum(head, 0.0), 0.0)
        qn = (q / qr) / np.maximum(speed, 0.3)
        eta = 0.72 * eff.eff_mult * (1.0 - 0.6 * (qn - 0.95) ** 2)
        power = (q / 3600.0) * head / np.maximum(eta, 0.2) + np.where(running, 6.0, 0.0)
        data["pressure_kpa"] = head * (1.0 + rng.normal(0, 0.012, n))
        data["flow_m3h"] = np.where(running, q, 0.0) * (1.0 + rng.normal(0, 0.015, n))
    if power is not None:
        power = np.where(running, power * eff.power_mult, 0.0)
        data["power_kw"] = power * (1.0 + rng.normal(0, 0.01, n))
        if "current_a" in spec.channels:
            pf = 0.50 + 0.38 * np.sqrt(np.clip(power / pr, 0, 1.1))
            amps = power * 1000.0 / (np.sqrt(3.0) * profile.voltage_v * pf * 0.95)
            data["current_a"] = np.where(running, amps, 0.0) * (1.0 + rng.normal(0, 0.012, n))
            _ = rated_current(profile)

    # ---- regime channels as logged
    data["load_pct"] = 100.0 * load + rng.normal(0, 0.8, n) * run_f
    data["speed_rpm"] = spec.nominal_rpm * speed * (1.0 + rng.normal(0, 0.002, n))
    data["ambient_c"] = reg.ambient + rng.normal(0, 0.3, n)

    cols = [c for c in ("load_pct", "speed_rpm", "ambient_c", *spec.channels) if c in data]
    clean = pd.DataFrame(
        {
            c: np.clip(data[c], 0.0, None) if c != "temp_c" and c != "ambient_c" else data[c]
            for c in cols
        },
        index=index,
    )
    for c in clean.columns:
        clean[c] = clean[c].round(QUANT[c])
    latent = pd.DataFrame(
        {
            "severity": eff.severity,
            "running": running,
            "temp_fault_delta": eff.temp_delta,
            "load": load,
            "speed_ratio": speed,
        },
        index=index,
    )
    return clean, latent, repair


# ------------------------------------------------------------------------------ fleet


def generate_fleet(cfg: SimConfig | None = None, keep_latent: bool = True) -> SyntheticFleet:
    cfg = cfg or SimConfig()
    root = np.random.SeedSequence(cfg.seed)
    plan_rng = np.random.default_rng(root.spawn(1)[0])
    specs = build_specs(cfg, plan_rng)
    failures = plan_failures(plan_rng, cfg, specs)
    asset_seeds = root.spawn(len(specs) + 1)[1:]
    # a separate stream, drawn last, so adding component ages left every signal unchanged
    ages = component_ages(np.random.default_rng(root.spawn(1)[0]), cfg, specs, failures)
    specs = [s.model_copy(update={"component_age_days": round(ages[s.asset_id], 1)}) for s in specs]

    raw: dict[str, pd.DataFrame] = {}
    latent: dict[str, pd.DataFrame] = {}
    maintenance: list[MaintenanceEvent] = []
    faults: list[DataFaultEvent] = []
    for spec, ss in zip(specs, asset_seeds, strict=True):
        rng = np.random.default_rng(ss)
        failure = failures.get(spec.asset_id)
        services = plan_services(rng, cfg, spec, failure)
        clean, lat, repair = simulate_asset(rng, cfg, spec, failure, services)
        dirty, ev = apply_data_faults(rng, clean, spec.asset_id, cfg.faults, cfg.sample_interval_s)
        raw[spec.asset_id] = dirty
        if keep_latent:
            latent[spec.asset_id] = pd.concat([clean, lat], axis=1)
        maintenance.extend(services)
        if repair is not None and repair.timestamp < cfg.start + timedelta(days=cfg.days):
            maintenance.append(repair)
        faults.extend(ev)

    manifest = FleetManifest(
        generator=f"asset_health.simulate {asset_health.__version__}",
        seed=cfg.seed,
        start=cfg.start,
        days=cfg.days,
        sample_interval_s=cfg.sample_interval_s,
        baseline_days=cfg.baseline_days,
        assets=specs,
    )
    maintenance.sort(key=lambda m: (m.asset_id, m.timestamp))
    truth = GroundTruth(
        seed=cfg.seed,
        failures=sorted(failures.values(), key=lambda f: f.asset_id),
        maintenance=list(maintenance),
        data_faults=faults,
    )
    return SyntheticFleet(manifest, raw, maintenance, truth, latent)


# ------------------------------------------------------------------------------ IO

README_TEXT = f"""{SYNTHETIC_NOTICE}

Files
  manifest.json       fleet description (assets, bearing geometry, sampling)
  maintenance.json    CMMS-style maintenance log the pipeline may read
  ground_truth.json   injected failures and data faults (the answer key; never read by the pipeline)
  assets/<id>.csv     one historian export per asset (reported timestamps, UTC, with defects)
"""


def write_dataset(fleet: SyntheticFleet, out_dir: str | Path, fmt: str = "csv") -> Path:
    if fmt not in {"csv", "parquet"}:
        raise ValueError("fmt must be 'csv' or 'parquet'")
    out = Path(out_dir)
    (out / "assets").mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(fleet.manifest.model_dump_json(indent=2), encoding="utf-8")
    (out / "maintenance.json").write_text(
        json.dumps([m.model_dump(mode="json") for m in fleet.maintenance], indent=1),
        encoding="utf-8",
    )
    (out / "ground_truth.json").write_text(fleet.truth.model_dump_json(indent=1), encoding="utf-8")
    (out / "SYNTHETIC_DATA.txt").write_text(README_TEXT, encoding="utf-8")
    for asset_id, df in fleet.raw.items():
        path = out / "assets" / f"{asset_id}.{fmt}"
        if fmt == "csv":
            df.to_csv(path, index=False, date_format="%Y-%m-%dT%H:%M:%S")
        else:
            df.to_parquet(path, index=False)
    return out
