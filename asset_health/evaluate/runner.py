"""The benchmark: calibrate on development seeds, evaluate on held-out seeds.

Seed roles (pairwise disjoint, asserted):

* ``dev``   - RUL model training and policy/baseline tuning;
* ``cal``   - split-conformal calibration of the RUL interval;
* ``test``  - everything reported in the README. Never used for any fitting or tuning choice.

Per-asset healthy baselines are learned inside the pipeline from each asset's own commissioning
window, so there is no cross-asset or cross-seed learning other than the global parameters above.
"""

from __future__ import annotations

import copy
import hashlib
import json
import platform
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import asset_health
from asset_health.alerting.cost import CostModel
from asset_health.alerting.policy import episode_hypothesis
from asset_health.catalog import PROFILES
from asset_health.config import PipelineConfig, PolicyConfig, RULConfig, SimConfig
from asset_health.evaluate.artifact import (
    AssetArtifact,
    SeedArtifact,
    _source_fingerprint,
    build_seed_artifact,
)
from asset_health.evaluate.baselines import NAIVE_VARIANTS
from asset_health.evaluate.costing import CostBreakdown, charge_failure
from asset_health.evaluate.policy_eval import (
    Memo,
    SeedEval,
    evaluate_calendar,
    evaluate_fixed_threshold,
    evaluate_policy,
)
from asset_health.evaluate.rul_eval import constant_baseline, rul_metrics, rul_rows
from asset_health.evaluate.stats import bootstrap_ci, mean_sd, quartiles, wilson
from asset_health.evaluate.tuning import (
    DEFAULT_GRID,
    TuneResult,
    select_best,
    tune_calendar,
    tune_fixed_threshold,
    tune_policy,
)
from asset_health.models import HYPOTHESIS_FAMILY, FailureMode
from asset_health.rul.health_index import health_index
from asset_health.rul.model import RULModel
from asset_health.rul.training import RunToFailure, fit_rul_model
from asset_health.simulate.fleet import renewal_failure_probability

Progress = Callable[[str], None]


@dataclass
class BenchConfig:
    dev_seeds: tuple[int, ...] = (1001, 1002, 1003, 1004, 1005, 1006)
    cal_seeds: tuple[int, ...] = (1101, 1102, 1103)
    test_seeds: tuple[int, ...] = tuple(range(3001, 3009))
    low_failure_seeds: tuple[int, ...] = tuple(range(3101, 3105))
    low_failure_fraction: float = 0.25
    # severity sweep: every injected signature scaled down, on the held-out seeds, nothing re-tuned
    severity_scales: tuple[float, ...] = (0.5, 0.3, 0.15)
    ablation_seeds: int = 4  # pipeline ablations re-run the pipeline: use the first N test seeds
    sim: SimConfig = field(default_factory=SimConfig)
    pipe: PipelineConfig = field(default_factory=PipelineConfig)
    base_policy: PolicyConfig = field(default_factory=PolicyConfig)
    rul: RULConfig = field(default_factory=RULConfig)
    costs: CostModel = field(default_factory=CostModel)
    grid: dict[str, tuple] = field(default_factory=lambda: dict(DEFAULT_GRID))
    ablations: bool = True
    min_actionable_lead_days: float = 3.0

    def quick(self) -> BenchConfig:
        """A smoke-test configuration: fewer seeds, a coarser grid, no pipeline ablations."""
        c = copy.deepcopy(self)
        c.dev_seeds, c.cal_seeds, c.test_seeds = (1001, 1002), (1101,), (3001, 3002)
        c.low_failure_seeds = ()
        c.severity_scales = (0.3,)
        c.grid = {
            "alert_threshold": (0.8, 1.0, 1.25),
            "cost_exponent": (0.0, 0.3),
            "min_detectors": (1,),
            "confirm_hours": (6,),
        }
        c.ablations = False
        return c


def assert_disjoint(dev, cal, test, *others) -> None:
    sets = {"dev": set(dev), "cal": set(cal), "test": set(test)}
    for i, o in enumerate(others):
        sets[f"other{i}"] = set(o)
    for a in sets:
        for b in sets:
            if a < b and sets[a] & sets[b]:
                raise ValueError(f"seed sets {a} and {b} overlap: {sorted(sets[a] & sets[b])}")


# ------------------------------------------------------------------ helpers


def run_to_failure_episodes(arts: list[SeedArtifact], rcfg: RULConfig) -> list[RunToFailure]:
    eps = []
    for art in arts:
        for f in art.truth.failures:
            a = art.assets[f.asset_id]
            if a.blocked or a.z24.empty:
                continue
            hi = health_index(a.z24, rcfg.top_k)
            if hi.notna().sum() < 100:
                continue
            eps.append(
                RunToFailure(art.seed, f.asset_id, hi, pd.Timestamp(f.failure), a.spec.asset_type)
            )
    return eps


def with_fused(art: SeedArtifact, weights: dict[str, float]) -> SeedArtifact:
    """Copy of an artefact whose fused score uses only the given detector weights (ablation)."""
    new_assets: dict[str, AssetArtifact] = {}
    total = sum(weights.values())
    for k, a in art.assets.items():
        f = a.frame.copy()
        fused = sum(w * f[c].clip(upper=3.0) for c, w in weights.items()) / total
        f["fused"] = fused.where(f["decision_valid"])
        firing = sum((f[c] >= 1).astype(int) for c in weights)
        f["n_firing"] = firing.where(f["decision_valid"], 0)
        new_assets[k] = AssetArtifact(**{**a.__dict__, "frame": f})
    return SeedArtifact(art.seed, art.sim, art.truth, new_assets)


def _detection_summary(evals: list[SeedEval], min_lead: float) -> dict:
    outcomes = [o for e in evals for o in e.outcomes]
    n = len(outcomes)
    det = sum(o.detected for o in outcomes)
    act = sum(o.actionable(min_lead) for o in outcomes)
    leads = [o.lead_days for o in outcomes if o.lead_days is not None]
    per_seed_det = [
        sum(o.detected for o in e.outcomes) / len(e.outcomes) for e in evals if e.outcomes
    ]
    per_seed_act = [
        sum(o.actionable(min_lead) for o in e.outcomes) / len(e.outcomes)
        for e in evals
        if e.outcomes
    ]
    fa = sum(len(e.false_alarms) for e in evals)
    months = sum(e.healthy_months for e in evals)
    per_seed_fa = [e.fa_per_asset_month for e in evals if e.healthy_months > 0]
    return {
        "n_failures": n,
        "detected": det,
        "detection_rate": det / n if n else float("nan"),
        "detection_rate_ci95": wilson(det, n),
        "detection_rate_by_seed": mean_sd(per_seed_det),
        f"detected_with_{min_lead:g}d_lead": act,
        "actionable_rate": act / n if n else float("nan"),
        "actionable_rate_ci95": wilson(act, n),
        "actionable_rate_by_seed": mean_sd(per_seed_act),
        "lead_time_days": quartiles(leads),
        "false_alarms": fa,
        "healthy_asset_months": months,
        "false_alarms_per_asset_month": fa / months if months else float("nan"),
        "false_alarms_per_asset_month_by_seed": mean_sd(per_seed_fa),
        "false_alarms_near_data_fault": sum(e.n_fa_near_data_fault for e in evals),
        "false_alarms_coincidental": sum(e.n_coincidental for e in evals),
    }


def _by_mode(evals: list[SeedEval], min_lead: float) -> dict:
    out = {}
    for mode in FailureMode:
        outs = [o for e in evals for o in e.outcomes if o.mode == mode]
        if not outs:
            continue
        n = len(outs)
        det = sum(o.detected for o in outs)
        act = sum(o.actionable(min_lead) for o in outs)
        leads = [o.lead_days for o in outs if o.lead_days is not None]
        durations = [o.failure.duration_days for o in outs]
        out[mode.value] = {
            "n": n,
            "detected": det,
            "detection_rate": det / n,
            "detection_rate_ci95": wilson(det, n),
            "actionable": act,
            "actionable_rate": act / n,
            "lead_time_days": quartiles(leads),
            "median_degradation_days": float(np.median(durations)),
            "median_lead_fraction_of_degradation": float(
                np.median(
                    [o.lead_days / o.failure.duration_days for o in outs if o.lead_days is not None]
                )
            )
            if leads
            else float("nan"),
        }
    return out


def _cost_summary(evals: list[SeedEval], arts: list[SeedArtifact]) -> dict:
    per_seed = [e.cost.total for e in evals]
    total = sum((e.cost for e in evals), CostBreakdown())
    months = sum(len(a.assets) * a.sim.days / 30.4375 for a in arts)
    return {
        "per_seed": mean_sd(per_seed),
        "per_seed_bootstrap_ci95": bootstrap_ci(per_seed),
        "per_asset_month": total.total / months,
        "breakdown_total": total.as_dict(),
    }


def _diagnosis(evals: list[SeedEval], arts: dict[int, SeedArtifact], window_h: int) -> dict:
    ok = unexplained = wrong = n = 0
    nt_n = nt_ok = 0  # asset types that can fail in more than one family (crusher, pump)
    confusion: dict[str, dict[str, int]] = {}
    for e in evals:
        art = arts[e.seed]
        for o in e.outcomes:
            if o.episode is None:
                continue
            a = art.assets[o.failure.asset_id]
            t = min(
                pd.Timestamp(o.episode.opened_at) + pd.Timedelta(days=3),
                pd.Timestamp(o.failure.failure) - pd.Timedelta(hours=1),
            )
            hyp, _ = episode_hypothesis(a.frame, t, window_h)
            fam = HYPOTHESIS_FAMILY[hyp]
            n += 1
            nontrivial = len(PROFILES[a.spec.asset_type].failure_modes) > 1
            nt_n += nontrivial
            nt_ok += nontrivial and fam == o.mode
            confusion.setdefault(o.mode.value, {}).setdefault(hyp.value, 0)
            confusion[o.mode.value][hyp.value] += 1
            if fam is None:
                unexplained += 1
            elif fam == o.mode:
                ok += 1
            else:
                wrong += 1
    return {
        "n": n,
        "correct_family": ok,
        "unexplained": unexplained,
        "wrong_family": wrong,
        "accuracy_when_stated": ok / (ok + wrong) if ok + wrong else float("nan"),
        "n_multi_family_assets": int(nt_n),
        "correct_multi_family_assets": int(nt_ok),
        "confusion": confusion,
    }


def _decisions(evals: list[SeedEval]) -> dict:
    """When the system acted on its detections: does the RUL interval ever defer the work?"""
    delays, before_fail, at_first = [], 0, 0
    for e in evals:
        for o in e.outcomes:
            if o.episode is None:
                continue
            tr = e.traces.get((o.failure.asset_id, pd.Timestamp(o.episode.opened_at)))
            if tr is None or tr.action_at is None:
                continue
            d = (tr.action_at - pd.Timestamp(o.episode.opened_at)).total_seconds() / 86400.0
            delays.append(d)
            at_first += d < 0.5
            before_fail += tr.execution_at is not None and tr.execution_at < pd.Timestamp(
                o.failure.failure
            )
    return {
        "n_decisions": len(delays),
        "decision_delay_days": quartiles(delays),
        "acted_at_first_checkpoint": int(at_first),
        "replaced_before_failure": int(before_fail),
    }


def _reference_costs(arts: list[SeedArtifact], costs: CostModel) -> dict:
    """Run-to-failure (do nothing) and an oracle that replaces every failing part ``plan_lead``
    days before it fails (and never inspects a healthy asset): the two ends of the scale."""
    rtf, oracle = [], []
    for art in arts:
        a = CostBreakdown()
        b = CostBreakdown()
        for f in art.truth.failures:
            t = art.assets[f.asset_id].spec.asset_type
            fail = pd.Timestamp(f.failure)
            charge_failure(a, costs, t, fail, None)
            charge_failure(b, costs, t, fail, fail - pd.Timedelta(days=costs.plan_lead_days))
        rtf.append(a.total)
        oracle.append(b.total)
    return {"run_to_failure": mean_sd(rtf), "oracle_replace_3d_before": mean_sd(oracle)}


# ------------------------------------------------------------------ main entry point


def _code_fingerprint() -> str:
    """Hash of every source file whose behaviour feeds the tuning tables."""
    here = Path(__file__).resolve().parents[1]
    files = ["config.py", "catalog.py", "models.py"]
    files += [f"alerting/{n}" for n in ("policy.py", "cost.py")]
    files += [
        f"evaluate/{n}"
        for n in (
            "policy_eval.py",
            "costing.py",
            "events.py",
            "baselines.py",
            "tuning.py",
            "artifact.py",
        )
    ]
    files += [
        f"rul/{n}"
        for n in ("estimate.py", "model.py", "trend.py", "training.py", "health_index.py")
    ]
    h = hashlib.sha1()
    for rel in files:
        h.update((here / rel).read_bytes())
    return h.hexdigest()[:12]


def _tuned(bc, dev, model, cache_dir, progress):
    """Policy and baseline tuning on development seeds, cached next to the pipeline artefacts.

    The cache key hashes the configuration, the seeds, the grid and the source of every module the
    tables depend on, so a stale entry cannot be read after a relevant change.
    """
    key = None
    if cache_dir is not None:
        blob = json.dumps(
            [
                bc.dev_seeds,
                bc.cal_seeds,
                bc.sim.model_dump_json(),
                bc.pipe.model_dump_json(),
                bc.base_policy.model_dump_json(),
                bc.rul.model_dump_json(),
                bc.costs.model_dump_json(),
                {k: list(v) for k, v in bc.grid.items()},
                _code_fingerprint(),
                _source_fingerprint(),
                model.model_dump_json(),
            ],
            default=str,
        )
        key = cache_dir / f"tuning_{hashlib.sha1(blob.encode()).hexdigest()[:12]}.json"
        if key.exists():
            c = json.loads(key.read_text(encoding="utf-8"))
            best = select_best(c["tuning_table"])
            pcfg = bc.base_policy.model_copy(update={k: best[k] for k in bc.grid})
            progress("  (tuning tables read from cache)")
            return (
                pcfg,
                c["tuning_table"],
                c["k_fixed"],
                c["fixed_table"],
                c["t_cal"],
                c["cal_table"],
            )
    tuned = tune_policy(dev, bc.base_policy, bc.rul, bc.costs, model, bc.grid)
    k_fixed, fixed_table = {}, {}
    for variant, channels in NAIVE_VARIANTS.items():
        progress(f"  fixed limits ({variant}): per-channel k on dev seeds")
        k_fixed[variant], fixed_table[variant] = tune_fixed_threshold(
            dev, bc.costs, channels=channels
        )
    t_cal, cal_table = tune_calendar(dev, bc.costs)
    if key is not None:
        key.write_text(
            json.dumps(
                {
                    "tuning_table": tuned.table,
                    "k_fixed": k_fixed,
                    "fixed_table": fixed_table,
                    "t_cal": t_cal,
                    "cal_table": cal_table,
                }
            ),
            encoding="utf-8",
        )
    return tuned.best, tuned.table, k_fixed, fixed_table, t_cal, cal_table


def run_benchmark(
    bc: BenchConfig,
    cache_dir: Path | None = None,
    n_jobs: int = 1,
    progress: Progress = print,
) -> tuple[dict, RULModel]:
    assert_disjoint(bc.dev_seeds, bc.cal_seeds, bc.test_seeds, bc.low_failure_seeds)
    t_start = time.time()
    pool = ProcessPoolExecutor(n_jobs) if n_jobs > 1 else None
    try:

        def build(seeds, pipe=bc.pipe, label=""):
            arts = []
            for s in seeds:
                t0 = time.time()
                arts.append(build_seed_artifact(s, bc.sim, pipe, pool, cache_dir))
                progress(f"  {label}seed {s}: {time.time() - t0:.0f}s")
            return arts

        progress("dev seeds")
        dev = build(bc.dev_seeds)
        progress("calibration seeds")
        cal = build(bc.cal_seeds)
        progress("test seeds")
        test = build(bc.test_seeds)
        test_by_seed = {a.seed: a for a in test}

        progress("fitting RUL model (dev) and conformal margin (cal)")
        model = fit_rul_model(
            run_to_failure_episodes(dev, bc.rul), run_to_failure_episodes(cal, bc.rul), bc.rul
        )
        progress(
            f"  level {model.level_mean:.2f}+-{model.level_sd:.2f}, "
            f"margin {model.conformal_margin:.2f}, "
            f"{model.n_train_events} train / {model.n_cal_events} cal events"
        )

        progress("tuning policy and baselines on dev seeds")
        pcfg, tuning_table, k_fixed, fixed_table, t_cal, cal_table = _tuned(
            bc, dev, model, cache_dir, progress
        )
        tuned = TuneResult(pcfg, tuning_table)
        progress(f"  policy {pcfg.model_dump()} | fixed k={k_fixed} | calendar {t_cal} x life")

        def fixed(arts, variant):
            ch = NAIVE_VARIANTS[variant]
            return [
                evaluate_fixed_threshold(a, k_fixed[variant], bc.costs, channels=ch) for a in arts
            ]

        progress("evaluating on held-out test seeds")
        memo = Memo()
        sys_evals = [evaluate_policy(a, pcfg, bc.rul, bc.costs, model, memo) for a in test]
        fix_evals = fixed(test, "raw")
        rise_evals = fixed(test, "rise")
        cal_evals = [evaluate_calendar(a, t_cal, bc.costs) for a in test]

        # RUL
        z24_of = {(a.seed, k): v.z24 for a in test for k, v in a.assets.items()}
        df = rul_rows(sys_evals, z24_of, bc.rul)
        dev_evals = [evaluate_policy(a, pcfg, bc.rul, bc.costs, model, memo) for a in dev]
        dev_rows = rul_rows(
            dev_evals,
            {(a.seed, k): v.z24 for a in dev for k, v in a.assets.items()},
            bc.rul,
            with_trend_only=False,
        )
        dev_df_true = dev_rows["true"].to_numpy() if len(dev_rows) else np.array([5.0])
        const = constant_baseline(dev_df_true)

        results: dict = {
            "synthetic": True,
            "package_version": asset_health.__version__,
            "python": platform.python_version(),
            "seeds": {
                "dev": list(bc.dev_seeds),
                "cal": list(bc.cal_seeds),
                "test": list(bc.test_seeds),
            },
            "fleet": {k.value: v for k, v in bc.sim.fleet.items()},
            "days": bc.sim.days,
            "failure_fraction": bc.sim.failure_fraction,
            "costs": bc.costs.model_dump(),
            "tuned_policy": pcfg.model_dump(),
            "tuning_table": tuned.table,
            "fixed_threshold_k": k_fixed,
            "fixed_threshold_table": fixed_table,
            "calendar_life_fraction": t_cal,
            "calendar_table": cal_table,
            "life_model": {
                t.value: {
                    "weibull_beta": PROFILES[t].life_weibull[0],
                    "weibull_eta_days": PROFILES[t].life_weibull[1],
                    "p_new_part_fails_within_record": renewal_failure_probability(t, bc.sim.days),
                }
                for t in bc.sim.fleet
            },
            "rul_model": {
                "level_mean": model.level_mean,
                "level_sd": model.level_sd,
                "conformal_margin": model.conformal_margin,
                "n_train_events": model.n_train_events,
                "n_cal_events": model.n_cal_events,
                "n_train_rows": model.n_train_rows,
            },
        }
        ml = bc.min_actionable_lead_days
        results["system"] = {
            "overall": _detection_summary(sys_evals, ml),
            "by_mode": _by_mode(sys_evals, ml),
            "diagnosis": _diagnosis(sys_evals, test_by_seed, bc.base_policy.hypothesis_window_h),
            "cost": _cost_summary(sys_evals, test),
            "rul": rul_metrics(df, const),
            "decisions": _decisions(sys_evals),
            "per_seed": [
                {
                    "seed": e.seed,
                    "failures": len(e.outcomes),
                    "detected": sum(o.detected for o in e.outcomes),
                    "false_alarms": len(e.false_alarms),
                    "cost": e.cost.total,
                }
                for e in sys_evals
            ],
        }
        results["fixed_threshold"] = {
            "overall": _detection_summary(fix_evals, ml),
            "by_mode": _by_mode(fix_evals, ml),
            "cost": _cost_summary(fix_evals, test),
        }
        results["fixed_threshold_rise"] = {
            "overall": _detection_summary(rise_evals, ml),
            "by_mode": _by_mode(rise_evals, ml),
            "cost": _cost_summary(rise_evals, test),
        }
        cal_total = [e.cost.total for e in cal_evals]
        months = sum(len(a.assets) * a.sim.days / 30.4375 for a in test)
        results["calendar"] = {
            "life_fraction": t_cal,
            "cost": {
                "per_seed": mean_sd(cal_total),
                "per_seed_bootstrap_ci95": bootstrap_ci(cal_total),
                "per_asset_month": sum(cal_total) / months,
            },
            "replacements_per_seed": mean_sd([e.n_replacements for e in cal_evals]),
            "failures_prevented": sum(e.n_prevented for e in cal_evals),
            "failures": sum(e.n_failures for e in cal_evals),
        }
        results["reference_costs"] = _reference_costs(test, bc.costs)

        if bc.low_failure_seeds:
            progress("scenario: low failure rate (policy and baselines NOT re-tuned)")
            sim_lo = bc.sim.model_copy(update={"failure_fraction": bc.low_failure_fraction})
            lo = [
                build_seed_artifact(s, sim_lo, bc.pipe, pool, cache_dir)
                for s in bc.low_failure_seeds
            ]
            lo_sys = [evaluate_policy(a, pcfg, bc.rul, bc.costs, model) for a in lo]
            lo_fix = fixed(lo, "raw")
            lo_rise = fixed(lo, "rise")
            lo_cal = [evaluate_calendar(a, t_cal, bc.costs) for a in lo]
            lo_months = sum(len(a.assets) * a.sim.days / 30.4375 for a in lo)
            results["scenario_low_failure_rate"] = {
                "failure_fraction": bc.low_failure_fraction,
                "seeds": list(bc.low_failure_seeds),
                "system": {
                    "overall": _detection_summary(lo_sys, ml),
                    "cost": _cost_summary(lo_sys, lo),
                },
                "fixed_threshold": {
                    "overall": _detection_summary(lo_fix, ml),
                    "cost": _cost_summary(lo_fix, lo),
                },
                "fixed_threshold_rise": {
                    "overall": _detection_summary(lo_rise, ml),
                    "cost": _cost_summary(lo_rise, lo),
                },
                "calendar": {
                    "per_seed": mean_sd([e.cost.total for e in lo_cal]),
                    "per_asset_month": sum(e.cost.total for e in lo_cal) / lo_months,
                },
                "reference_costs": _reference_costs(lo, bc.costs),
            }

        if bc.severity_scales:
            progress("severity sweep: injected signatures scaled down (nothing re-tuned)")

            def row(scale, arts, s_ev, f_ev, r_ev):
                return {
                    "scale": scale,
                    "system": _sweep_cell(s_ev, ml),
                    "fixed_threshold": _sweep_cell(f_ev, ml),
                    "fixed_threshold_rise": _sweep_cell(r_ev, ml),
                    "calendar_cost_per_seed": mean_sd(
                        [evaluate_calendar(a, t_cal, bc.costs).cost.total for a in arts]
                    ),
                    "run_to_failure_cost_per_seed": _reference_costs(arts, bc.costs)[
                        "run_to_failure"
                    ],
                }

            sweep = [row(1.0, test, sys_evals, fix_evals, rise_evals)]
            for scale in bc.severity_scales:
                sim_s = bc.sim.model_copy(update={"degradation_scale": scale})
                arts = []
                for sd in bc.test_seeds:
                    t0 = time.time()
                    arts.append(build_seed_artifact(sd, sim_s, bc.pipe, pool, cache_dir))
                    progress(f"  scale {scale} seed {sd}: {time.time() - t0:.0f}s")
                sweep.append(
                    row(
                        scale,
                        arts,
                        [evaluate_policy(a, pcfg, bc.rul, bc.costs, model) for a in arts],
                        fixed(arts, "raw"),
                        fixed(arts, "rise"),
                    )
                )
            results["severity_sweep"] = {"seeds": list(bc.test_seeds), "rows": sweep}

        if bc.ablations:
            results["ablation_seeds"] = list(bc.test_seeds[: bc.ablation_seeds])
            results["ablations"] = _ablations(
                bc, dev, test, model, pcfg, sys_evals, pool, cache_dir, progress
            )
        results["runtime_s"] = time.time() - t_start
        return results, model
    finally:
        if pool is not None:
            pool.shutdown()


def _sweep_cell(evals: list[SeedEval], ml: float) -> dict:
    ov = _detection_summary(evals, ml)
    return {
        "n_failures": ov["n_failures"],
        "detected": ov["detected"],
        "actionable": ov[f"detected_with_{ml:g}d_lead"],
        "median_lead_days": ov["lead_time_days"]["median"],
        "false_alarms": ov["false_alarms"],
        "false_alarms_per_asset_month": ov["false_alarms_per_asset_month"],
        "cost_per_seed": mean_sd([e.cost.total for e in evals]),
    }


def _variant_row(
    label: str, evals: list[SeedEval], arts: list[SeedArtifact], ml: float, extra=None
) -> dict:
    ov = _detection_summary(evals, ml)
    return {
        "variant": label,
        "detection_rate": ov["detection_rate"],
        "actionable_rate": ov["actionable_rate"],
        "median_lead_days": ov["lead_time_days"]["median"],
        "false_alarms_per_asset_month": ov["false_alarms_per_asset_month"],
        "false_alarms": ov["false_alarms"],
        "false_alarms_near_data_fault": ov["false_alarms_near_data_fault"],
        "cost_per_seed": mean_sd([e.cost.total for e in evals]),
        **(extra or {}),
    }


def _ablations(bc, dev, test, model, pcfg, sys_evals, pool, cache_dir, progress) -> list[dict]:
    """Every row is scored on the same first ``ablation_seeds`` held-out seeds."""
    ml = bc.min_actionable_lead_days
    test = test[: bc.ablation_seeds]
    ids = {a.seed for a in test}
    sys_evals = [e for e in sys_evals if e.seed in ids]
    rows = [_variant_row("full system", sys_evals, test, ml)]
    # detector ablations: re-tune the alert threshold for each variant on dev seeds
    variants = {
        "CUSUM only": ({"cusum": 1.0}, (1,)),
        "Isolation Forest only": ({"iforest": 1.0}, (1,)),
        "rule layer only": ({"rule": 1.0}, (1,)),
        "no rule layer (CUSUM + IF)": ({"cusum": 0.4, "iforest": 0.3}, (1, 2)),
    }
    small_grid = {
        "alert_threshold": DEFAULT_GRID["alert_threshold"],
        "cost_exponent": (pcfg.cost_exponent,),
        "min_detectors": (1,),
        "confirm_hours": (pcfg.confirm_hours,),
    }
    for label, (w, md_options) in variants.items():
        progress(f"  ablation: {label}")
        dev_v = [with_fused(a, w) for a in dev]
        test_v = [with_fused(a, w) for a in test]
        grid = {**small_grid, "min_detectors": md_options}
        tuned = tune_policy(dev_v, pcfg, bc.rul, bc.costs, model, grid)
        evs = [evaluate_policy(a, tuned.best, bc.rul, bc.costs, model) for a in test_v]
        rows.append(
            _variant_row(label, evs, test, ml, {"threshold_retuned": tuned.best.alert_threshold})
        )
    # policy ablations on the same detections
    progress("  ablation: single-detector trigger")
    solo_grid = {**small_grid, "min_detectors": (1,), "confirm_hours": (pcfg.confirm_hours,)}
    solo = tune_policy(dev, pcfg, bc.rul, bc.costs, model, solo_grid)
    evs = [evaluate_policy(a, solo.best, bc.rul, bc.costs, model) for a in test]
    rows.append(
        _variant_row(
            "no detector agreement (1 of 3 may trigger)",
            evs,
            test,
            ml,
            {"threshold_retuned": solo.best.alert_threshold},
        )
    )
    progress("  ablation: no debounce / hysteresis")
    nodb = pcfg.model_copy(update={"confirm_hours": 1, "clear_hours": 1})
    rows.append(
        _variant_row(
            "no debounce, no hysteresis",
            [evaluate_policy(a, nodb, bc.rul, bc.costs, model) for a in test],
            test,
            ml,
        )
    )
    # pipeline ablations: re-run the pipeline on the test seeds (policy fixed, NOT re-tuned)
    for label, pipe in (
        (
            "no data-quality gating",
            bc.pipe.model_copy(
                update={"ingest": bc.pipe.ingest.model_copy(update={"gating": False})}
            ),
        ),
        (
            "no regime normalisation",
            bc.pipe.model_copy(
                update={"features": bc.pipe.features.model_copy(update={"regime_normalise": False})}
            ),
        ),
    ):
        progress(f"  ablation: {label}")
        arts = []
        for a0 in test:  # the same seeds as every other ablation row
            arts.append(build_seed_artifact(a0.seed, bc.sim, pipe, pool, cache_dir))
        evs = [evaluate_policy(a, pcfg, bc.rul, bc.costs, model) for a in arts]
        rows.append(_variant_row(label, evs, arts, ml, {"policy_retuned": False}))
    return rows


def save_results(results: dict, path: Path) -> None:
    def default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, tuple):
            return list(o)
        raise TypeError(type(o))

    path.write_text(json.dumps(results, indent=1, default=default), encoding="utf-8")
