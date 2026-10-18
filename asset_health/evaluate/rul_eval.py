"""RUL evaluation on held-out seeds: error and interval coverage at the system's own checkpoints.

Checkpoints are the daily re-estimates inside each *detected* failure's alert episode (so RUL is
scored only where the system would have produced one). True RUL is ``failure - checkpoint``.

The primary rows are the **decision-time** checkpoints: those up to the planned replacement
(``EpisodeTrace.execution_at``). In operation the part is gone after that, so later checkpoints are
counterfactual; they are easy (close to failure) and would flatter the error. The first estimate of
each episode and the pooled set (every checkpoint up to failure) are reported as well.

Three predictors are compared on the same rows:

* **system**: trend extrapolation + quantile regression + conformal margin (what ships);
* **trend only**: the uncalibrated trend extrapolation and its Monte-Carlo interval;
* **constant**: the median true RUL of the training rows, with their 10-90% range as its interval.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from asset_health.config import RULConfig
from asset_health.evaluate.policy_eval import SeedEval
from asset_health.models import RULStatus
from asset_health.rul.estimate import estimate_rul
from asset_health.rul.health_index import health_index


@dataclass
class ConstantRUL:
    p10: float
    p50: float
    p90: float


def rul_rows(
    seed_evals: list[SeedEval],
    z24_of: dict[tuple[int, str], pd.DataFrame],
    rcfg: RULConfig,
    with_trend_only: bool = True,
) -> pd.DataFrame:
    rows = []
    for ev in seed_evals:
        for o in ev.outcomes:
            if o.episode is None:
                continue
            asset = o.failure.asset_id
            tr = ev.traces.get((asset, pd.Timestamp(o.episode.opened_at)))
            if tr is None:
                continue
            fail = pd.Timestamp(o.failure.failure)
            hi = None
            first = True
            for cp in tr.checkpoints:
                if cp.t >= fail or cp.rul.status != RULStatus.ESTIMATED:
                    continue
                true = (fail - cp.t).total_seconds() / 86400.0
                r = cp.rul
                row = {
                    "seed": ev.seed,
                    "asset": asset,
                    "mode": o.failure.mode.value,
                    "t": cp.t,
                    "decision": tr.execution_at is None or cp.t <= tr.execution_at,
                    "first": first,
                    "true": true,
                    "p10": r.p10_days,
                    "p50": r.p50_days,
                    "p90": r.p90_days,
                    "trend_p50": r.method_trend_p50,
                }
                if with_trend_only:
                    if hi is None:
                        hi = health_index(z24_of[(ev.seed, asset)], rcfg.top_k)
                    e0 = estimate_rul(asset, hi, cp.t, None, rcfg)
                    row.update(t10=e0.p10_days, t50=e0.p50_days, t90=e0.p90_days)
                rows.append(row)
                first = False
    return pd.DataFrame(rows)


def constant_baseline(train_true: np.ndarray) -> ConstantRUL:
    p10, p50, p90 = np.quantile(train_true, [0.1, 0.5, 0.9])
    return ConstantRUL(float(p10), float(p50), float(p90))


def _metrics(true, p10, p50, p90) -> dict[str, float]:
    err = np.abs(p50 - true)
    cover = (true >= p10) & (true <= p90)
    return {
        "n": len(true),
        "mae_days": float(err.mean()),
        "median_ae_days": float(np.median(err)),
        "coverage": float(cover.mean()),
        "mean_width_days": float((p90 - p10).mean()),
    }


BUCKETS = ((0.0, 7.0), (7.0, 14.0), (14.0, 30.0), (30.0, 1e9))


SUBSETS = {
    "decision_time": "decision",  # primary: checkpoints up to the planned replacement
    "first_estimate": "first",
    "all": None,  # every checkpoint up to failure (pooled, includes counterfactual rows)
}


def _block(sub: pd.DataFrame, a: str, b: str, c: str) -> dict:
    out: dict = {
        "overall": _metrics(
            sub["true"].to_numpy(), sub[a].to_numpy(), sub[b].to_numpy(), sub[c].to_numpy()
        )
    }
    sub = sub.copy()
    sub["err"] = (sub[b] - sub["true"]).abs()
    sub["cover"] = ((sub["true"] >= sub[a]) & (sub["true"] <= sub[c])).astype(float)
    per_ep = sub.groupby(["seed", "asset"])[["err", "cover"]].mean()
    out["n_episodes"] = len(per_ep)
    out["per_episode_mae_mean"] = float(per_ep["err"].mean())
    out["per_episode_coverage_mean"] = float(per_ep["cover"].mean())
    by_seed = sub.groupby("seed")["cover"].mean()
    out["coverage_by_seed"] = {int(k): float(v) for k, v in by_seed.items()}
    out["coverage_seed_sd"] = float(by_seed.std(ddof=1)) if len(by_seed) > 1 else 0.0
    for lo, hi_ in BUCKETS:
        s = sub[(sub["true"] >= lo) & (sub["true"] < hi_)]
        if len(s):
            key = f"true_rul_{int(lo)}_{'inf' if hi_ > 1e6 else int(hi_)}d"
            out[key] = _metrics(
                s["true"].to_numpy(), s[a].to_numpy(), s[b].to_numpy(), s[c].to_numpy()
            )
    return out


def rul_metrics(df: pd.DataFrame, const: ConstantRUL | None = None) -> dict:
    """Metrics per predictor for each subset (decision time, first estimate, all checkpoints),
    with per-episode means, coverage by seed and by horizon bucket."""
    if df.empty:
        return {"n": 0}
    out: dict = {
        "n_checkpoints": len(df),
        "n_checkpoints_decision_time": int(df["decision"].sum()) if "decision" in df else len(df),
        "n_episodes": int(df.groupby(["seed", "asset"]).ngroups),
    }
    if const is not None:
        df = df.assign(c10=const.p10, c50=const.p50, c90=const.p90)
    preds = {"system": ("p10", "p50", "p90")}
    if "t50" in df:
        preds["trend_only"] = ("t10", "t50", "t90")
    if const is not None:
        preds["constant"] = ("c10", "c50", "c90")
    for name, (a, b, c) in preds.items():
        base = df.dropna(subset=[a, b, c])
        out[name] = {}
        for label, col in SUBSETS.items():
            sub = base if col is None or col not in base else base[base[col].astype(bool)]
            if len(sub):
                out[name][label] = _block(sub, a, b, c)
    return out
