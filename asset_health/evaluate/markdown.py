"""Render ``results.json`` as the Markdown tables used in the README (single source of numbers)."""

from __future__ import annotations


def _pct(x: float) -> str:
    return "n/a" if x != x else f"{100 * x:.0f}%"


def _ci(ci) -> str:
    return f"{100 * ci[0]:.0f}-{100 * ci[1]:.0f}%"


def _ms(d: dict, fmt: str = "{:.0f}") -> str:
    return f"{fmt.format(d['mean'])} ± {fmt.format(d['sd'])}"


def _k(k) -> str:
    if isinstance(k, dict):
        return ", ".join(f"{c} {v:g}" for c, v in k.items())
    return f"{k:g}"


def _lead(x) -> str:
    return "n/a" if x is None or x != x else f"{x:.1f}"


SEVERITY_NOTE = (
    "Every injected failure signature (vibration lines, broadband energy, temperature, hydraulic "
    "and power changes) is multiplied by the scale; failure times, data faults and everything "
    "tuned stay as they are."
)


def _severity_sweep(r: dict) -> list[str]:
    sw = r.get("severity_sweep")
    if not sw:
        return []
    ml = _min_lead(r)
    out = [
        f"### Detectability vs failure severity ({len(sw['seeds'])} held-out seeds per row, "
        "nothing re-tuned)\n",
        SEVERITY_NOTE + "\n",
        f"| Signature scale | Failures | Detected with >= {ml} lead: asset-health / fixed raw / "
        "fixed rise | Median lead, days: asset-health / raw / rise | Cost per seed, kUSD: "
        "asset-health / raw / rise | Calendar, kUSD | Run to failure, kUSD |",
        "|---|---|---|---|---|---|---|",
    ]
    verdicts = []
    for row in sw["rows"]:
        s, f, g = row["system"], row["fixed_threshold"], row["fixed_threshold_rise"]
        n = s["n_failures"]
        out.append(
            f"| {row['scale']:g}x | {n} | {s['actionable']} / {f['actionable']} / {g['actionable']} "
            f"| {_lead(s['median_lead_days'])} / {_lead(f['median_lead_days'])} / "
            f"{_lead(g['median_lead_days'])} "
            f"| {_ms(s['cost_per_seed'])} / {_ms(f['cost_per_seed'])} / {_ms(g['cost_per_seed'])} "
            f"| {_ms(row['calendar_cost_per_seed'])} | {_ms(row['run_to_failure_cost_per_seed'])} |"
        )
        best_fixed = min(f["cost_per_seed"]["mean"], g["cost_per_seed"]["mean"])
        best_act = max(f["actionable"], g["actionable"])
        verdicts.append(
            (row["scale"], s["cost_per_seed"]["mean"] < best_fixed, s["actionable"] >= best_act)
        )
    wins = [sc for sc, c, _ in verdicts if c]
    loses = [sc for sc, c, _ in verdicts if not c]
    act_lose = [sc for sc, _, a in verdicts if not a]
    parts = []
    if wins:
        parts.append(
            "asset-health costs less than the cheaper fixed limit at "
            + ", ".join(f"{x:g}x" for x in wins)
        )
    if loses:
        parts.append("it does **not** at " + ", ".join(f"{x:g}x" for x in loses))
    if act_lose:
        parts.append(
            "a fixed limit warns at least "
            + ml
            + " ahead on more failures at "
            + ", ".join(f"{x:g}x" for x in act_lose)
        )
    else:
        parts.append(
            f"it warns {ml} ahead on at least as many failures as either fixed limit at every scale"
        )
    out.append("\n" + "; ".join(parts) + ".\n")
    return out


def _min_lead(r: dict) -> str:
    ml = next(k for k in r["system"]["overall"] if k.startswith("detected_with_"))
    return ml.replace("detected_with_", "").replace("_lead", "")


def results_to_markdown(r: dict) -> str:
    s, f, c = r["system"], r["fixed_threshold"], r["calendar"]
    g = r.get("fixed_threshold_rise")
    lead = _min_lead(r)
    nt = len(r["seeds"]["test"])
    ref = r["reference_costs"]
    out: list[str] = []

    out.extend(_severity_sweep(r))

    so, fo = s["overall"], f["overall"]
    go = g["overall"] if g else None
    frac = c["life_fraction"]
    out.append(
        f"### Policy comparison at the designed signature size ({nt} held-out seeds, "
        f"{so['n_failures']} injected failures)\n"
    )
    cols = ["asset-health", "fixed limit, raw temperature"]
    if g:
        cols.append("fixed limit, temperature rise over ambient")
    cols.append(f"age-based replacement ({frac:g} x characteristic life)")
    out.append("| | " + " | ".join(cols) + " |\n|" + "---|" * (len(cols) + 1))

    def row(label, sys_cell, fix_cell, rise_cell, cal_cell):
        cells = [sys_cell, fix_cell] + ([rise_cell] if g else []) + [cal_cell]
        out.append(f"| {label} | " + " | ".join(cells) + " |")

    def det(o):
        return (
            f"{o['detected']}/{o['n_failures']} ({_pct(o['detection_rate'])}, "
            f"95% CI {_ci(o['detection_rate_ci95'])})"
        )

    def iqr(o):
        q = o["lead_time_days"]
        return f"{q['median']:.1f} ({q['q25']:.1f}-{q['q75']:.1f})"

    def fa(o):
        return (
            f"{o['false_alarms_per_asset_month']:.3f} ({o['false_alarms']} alarms, "
            f"{o.get('false_alarms_coincidental', 0)} coincidental)"
        )

    cal_prev = f"prevents {c['failures_prevented']}/{c['failures']}"
    row("Failures detected", det(so), det(fo), det(go) if go else "", cal_prev)
    row(
        f"Detected with >= {lead} lead",
        _pct(so["actionable_rate"]),
        _pct(fo["actionable_rate"]),
        _pct(go["actionable_rate"]) if go else "",
        "n/a",
    )
    row("Median lead time, days (IQR)", iqr(so), iqr(fo), iqr(go) if go else "", "n/a")
    row(
        "False alarms per healthy asset-month",
        fa(so),
        fa(fo),
        fa(go) if go else "",
        f"{_ms(c['replacements_per_seed'], '{:.1f}')} replacements per seed",
    )
    row(
        "Policy cost per seed, kUSD (mean ± sd)",
        _ms(s["cost"]["per_seed"]),
        _ms(f["cost"]["per_seed"]),
        _ms(g["cost"]["per_seed"]) if g else "",
        _ms(c["cost"]["per_seed"]),
    )
    row(
        "Policy cost per asset-month, kUSD",
        f"{s['cost']['per_asset_month']:.2f}",
        f"{f['cost']['per_asset_month']:.2f}",
        f"{g['cost']['per_asset_month']:.2f}" if g else "",
        f"{c['cost']['per_asset_month']:.2f}",
    )
    life = r.get("life_model", {})
    worst = max((v["p_new_part_fails_within_record"] for v in life.values()), default=float("nan"))
    out.append(
        f"\nReference points per seed: run to failure {_ms(ref['run_to_failure'])} kUSD; "
        f"oracle that replaces every failing part {r['costs']['plan_lead_days']:g} days before it "
        f"fails {_ms(ref['oracle_replace_3d_before'])} kUSD. Costs are illustrative (see Design "
        "decisions). 'Coincidental' alarms opened inside a failure window before the injected "
        "degradation reached 5% severity; they count as false alarms, not detections. Each "
        "baseline's false-alarm rate uses the hours it was itself watching. Component lives are "
        "Weibull wear-out per asset class; a part fitted during the record is assumed to outlive "
        f"it (true with probability at least {_pct(1 - worst)} for every class).\n"
    )

    out.append("### Detection by failure mode (system, held-out seeds)\n")
    out.append(
        "| Failure mode | Injected | Detected (95% CI) | Median lead, days "
        "| Median degradation, days |\n|---|---|---|---|---|"
    )
    for mode, v in s["by_mode"].items():
        out.append(
            f"| {mode} | {v['n']} | {v['detected']} ({_pct(v['detection_rate'])}, "
            f"{_ci(v['detection_rate_ci95'])}) | {_lead(v['lead_time_days']['median'])} "
            f"| {v['median_degradation_days']:.0f} |"
        )

    out.append("\n### Variation across seeds\n")
    out.append("| Metric | mean ± sd over seeds |\n|---|---|")
    out.append(
        f"| Detection rate | {_pct(so['detection_rate_by_seed']['mean'])} ± "
        f"{100 * so['detection_rate_by_seed']['sd']:.0f} pts |"
    )
    out.append(
        f"| False alarms per asset-month | "
        f"{so['false_alarms_per_asset_month_by_seed']['mean']:.3f} ± "
        f"{so['false_alarms_per_asset_month_by_seed']['sd']:.3f} |"
    )
    ci = s["cost"]["per_seed_bootstrap_ci95"]
    out.append(
        f"| Policy cost per seed, kUSD | {_ms(s['cost']['per_seed'])} (bootstrap 95% CI of the "
        f"mean {ci[0]:.0f}-{ci[1]:.0f}) |"
    )

    out.extend(_rul_section(s))

    dec = s.get("decisions")
    if dec and dec["n_decisions"]:
        q = dec["decision_delay_days"]
        out.append(
            f"\n### When the system acted\n\nOf {dec['n_decisions']} detections, "
            f"{dec['acted_at_first_checkpoint']} were acted on at the first checkpoint after the "
            f"alert (median delay {q['median']:.1f} days, IQR {q['q25']:.1f}-{q['q75']:.1f}) and "
            f"{dec['replaced_before_failure']} parts were replaced before they failed. At these "
            "cost ratios the RUL interval rarely defers work: it grades severity and ranks assets."
        )

    d = s["diagnosis"]
    out.append(
        f"\n### Fault diagnosis three days after the alert\n\n{d['correct_family']} of {d['n']} "
        f"alerts named the right failure family, {d['unexplained']} stayed 'unexplained anomaly', "
        f"{d['wrong_family']} named a wrong family. On the asset types that can fail in more than "
        f"one family (crusher, pump), {d.get('correct_multi_family_assets', 0)} of "
        f"{d.get('n_multi_family_assets', 0)} were right; elsewhere the family follows from the "
        "asset type."
    )

    if "scenario_low_failure_rate" in r:
        lo = r["scenario_low_failure_rate"]
        ls, lf = lo["system"], lo["fixed_threshold"]
        lg = lo.get("fixed_threshold_rise")
        out.append(
            f"\n### Sensitivity: rarer failures (failure fraction {lo['failure_fraction']}, "
            f"{len(lo['seeds'])} further seeds, nothing re-tuned)\n"
        )
        hdr = [
            "asset-health",
            "fixed raw",
            *(["fixed rise"] if lg else []),
            "calendar",
            "run to failure",
        ]
        out.append("| | " + " | ".join(hdr) + " |\n|" + "---|" * (len(hdr) + 1))
        rise = [lg] if lg else []
        out.append(
            "| Detected | "
            + " | ".join(
                f"{x['overall']['detected']}/{x['overall']['n_failures']}" for x in [ls, lf, *rise]
            )
            + " | n/a | n/a |"
        )
        out.append(
            "| False alarms per healthy asset-month | "
            + " | ".join(
                f"{x['overall']['false_alarms_per_asset_month']:.3f}" for x in [ls, lf, *rise]
            )
            + " | n/a | n/a |"
        )
        out.append(
            "| Cost per seed, kUSD | "
            + " | ".join(_ms(x["cost"]["per_seed"]) for x in [ls, lf, *rise])
            + f" | {_ms(lo['calendar']['per_seed'])} | {_ms(lo['reference_costs']['run_to_failure'])} |"
        )

    if "ablations" in r:
        abl = r.get("ablation_seeds", r["seeds"]["test"])
        out.append(
            f"\n### Ablations (held-out seeds {abl}; detector rows re-tuned on development seeds)\n"
        )
        out.append(
            "| Variant | Detected | Detected with lead | Median lead, d | False alarms / asset-month "
            "| Cost per seed, kUSD |\n|---|---|---|---|---|---|"
        )
        for a in r["ablations"]:
            out.append(
                f"| {a['variant']} | {_pct(a['detection_rate'])} | {_pct(a['actionable_rate'])} "
                f"| {_lead(a['median_lead_days'])} | {a['false_alarms_per_asset_month']:.3f} "
                f"| {_ms(a['cost_per_seed'])} |"
            )

    out.append(_tuning_summary(r))
    t = r["tuned_policy"]
    kf = r["fixed_threshold_k"]
    kf_txt = (
        "; ".join(f"{v}: median + k robust sigma with k = {_k(k)}" for v, k in kf.items())
        if isinstance(kf, dict) and all(isinstance(x, dict) for x in kf.values())
        else f"median + {_k(kf)} robust sigma"
    )
    out.append(
        f"\nTuned on development seeds {r['seeds']['dev']}: alert threshold {t['alert_threshold']}, "
        f"cost exponent {t['cost_exponent']}, {t['min_detectors']} detector(s) must agree, "
        f"{t['confirm_hours']} h debounce; fixed limits {kf_txt}; age-based replacement at "
        f"{frac:g} x characteristic life. RUL conformal margin fitted on seeds {r['seeds']['cal']}; "
        f"evaluated on {r['seeds']['test']}."
    )
    return "\n".join(out) + "\n"


def _rul_section(s: dict) -> list[str]:
    rul = s["rul"]
    if not rul.get("n_checkpoints"):
        return []
    sysr = rul["system"]
    dt = sysr.get("decision_time")
    if dt is None:
        return []
    out = [
        f"\n### Remaining useful life at decision time ({dt['n_episodes']} detected failures, "
        f"{dt['overall']['n']} daily checkpoints up to the planned replacement)\n",
        "| Predictor | MAE, days | Interval coverage (nominal 80%) | Mean interval width, days |",
        "|---|---|---|---|",
    ]
    for key, label in (
        ("system", "asset-health (trend + quantile regression + conformal)"),
        ("trend_only", "trend extrapolation only"),
        ("constant", "constant (training-set median)"),
    ):
        blk = rul.get(key, {}).get("decision_time")
        if blk:
            a = blk["overall"]
            out.append(
                f"| {label} | {a['mae_days']:.1f} | {_pct(a['coverage'])} "
                f"| {a['mean_width_days']:.1f} |"
            )
    out.append(
        "\n| True RUL at the checkpoint | n | MAE, days | Coverage | Width, days |\n"
        "|---|---|---|---|---|"
    )
    for k, v in dt.items():
        if k.startswith("true_rul_"):
            label = k.replace("true_rul_", "").replace("_inf", "+").replace("_", "-")
            label = label.replace("d", " d")
            out.append(
                f"| {label} | {v['n']} | {v['mae_days']:.1f} | {_pct(v['coverage'])} "
                f"| {v['mean_width_days']:.1f} |"
            )
    fe, al = sysr.get("first_estimate"), sysr.get("all")
    extra = []
    if fe:
        o = fe["overall"]
        extra.append(
            f"first estimate of each episode: MAE {o['mae_days']:.1f} d, coverage "
            f"{_pct(o['coverage'])}, width {o['mean_width_days']:.1f} d"
        )
    if al:
        o = al["overall"]
        extra.append(
            f"every checkpoint up to failure ({o['n']}, most of them after the part would have "
            f"been replaced): MAE {o['mae_days']:.1f} d, coverage {_pct(o['coverage'])}, width "
            f"{o['mean_width_days']:.1f} d"
        )
    if extra:
        out.append("\nFor comparison, " + "; ".join(extra) + ".")
    out.append(
        "\nCoverage by seed (system, decision time): "
        + ", ".join(f"{k}: {_pct(v)}" for k, v in dt["coverage_by_seed"].items())
        + "."
    )
    return out


def _tuning_summary(r: dict) -> str:
    """What the development-seed grid search saw (so the choice can be audited)."""
    table = r["tuning_table"]
    lines = ["", "### Policy tuning on the development seeds", ""]
    lines.append(
        "| Detectors that must agree | Best cost per seed, kUSD | Detected | False alarms "
        "| Cost range across the alert-threshold grid |"
    )
    lines.append("|---|---|---|---|---|")
    for md in sorted({row["min_detectors"] for row in table}):
        rows = [row for row in table if row["min_detectors"] == md]
        best = min(rows, key=lambda x: x["mean_cost_per_seed"])
        costs = [x["mean_cost_per_seed"] for x in rows]
        lines.append(
            f"| {md} of 3 | {best['mean_cost_per_seed']:.0f} "
            f"| {best['detected']}/{best['failures']} | {best['false_alarms']} "
            f"| {min(costs):.0f}-{max(costs):.0f} |"
        )
    return "\n".join(lines) + "\n"
