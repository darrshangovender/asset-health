"""Work-order narrative: a deterministic template, and an optional LLM behind a Protocol.

Guardrail. A narrative is accepted only if everything it asserts can be traced to the structured
``WorkOrder``: every number must match a number in the object (evidence values, RUL quantiles,
dates, counts), every feature name must be one of the evidence features, and every named fault
must be the primary or an alternative hypothesis. Anything else makes the order fall back to the
template (``narrative_source = "template_after_guardrail"``). The guardrail is a conservative
text check, not a proof; the structured object remains the source of truth.
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

from asset_health.catalog import CHANNEL_DESCRIPTIONS
from asset_health.models import FaultHypothesis, Severity, WorkOrder

HYPOTHESIS_TEXT: dict[FaultHypothesis, str] = {
    FaultHypothesis.BEARING_OUTER_RACE: "bearing outer-race defect",
    FaultHypothesis.BEARING_INNER_RACE: "bearing inner-race defect",
    FaultHypothesis.BEARING_GENERIC: "early bearing wear or lubrication loss",
    FaultHypothesis.IMPELLER_WEAR: "impeller wear",
    FaultHypothesis.LINER_WEAR: "crusher liner wear",
    FaultHypothesis.OVERHEATING: "overheating",
    FaultHypothesis.UNEXPLAINED: "unexplained anomaly",
}


@runtime_checkable
class NarrativeGenerator(Protocol):
    """Anything that can turn a structured work order into prose (an LLM, a template ...)."""

    def generate(self, work_order: WorkOrder) -> str: ...


def _fmt(x: float) -> str:
    return f"{x:.1f}"


def template_narrative(wo: WorkOrder) -> str:
    """Deterministic narrative using only fields of the work order."""
    parts = [
        f"{wo.asset_name} ({wo.asset_id}) is at severity {wo.severity.value}: most likely "
        f"{HYPOTHESIS_TEXT[wo.failure_mode_hypothesis]}."
    ]
    if wo.evidence:
        ev = "; ".join(
            f"{e.feature} is {_fmt(e.observed)} {e.unit} against {_fmt(e.expected)} {e.unit} "
            f"expected for the current load and speed (z = {_fmt(e.z_score)})".replace("  ", " ")
            for e in wo.evidence[:3]
        )
        parts.append(f"Evidence: {ev}.")
    r = wo.rul
    if (
        r is not None
        and r.p10_days is not None
        and r.p50_days is not None
        and r.p90_days is not None
    ):
        parts.append(
            f"Remaining useful life is estimated at {_fmt(r.p50_days)} days "
            f"(80% interval {_fmt(r.p10_days)} to {_fmt(r.p90_days)} days)."
        )
    if wo.latest_safe_date is not None:
        parts.append(f"Latest safe date for the intervention: {wo.latest_safe_date:%Y-%m-%d}.")
    parts.append(f"Recommended action: {wo.recommended_action}")
    if wo.data_quality_note:
        parts.append(f"Data quality: {wo.data_quality_note}")
    return " ".join(parts)


# ---------------------------------------------------------------------------- guardrail

_NUM = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_KNOWN_FEATURES = set(CHANNEL_DESCRIPTIONS) | {
    "vib_band_low", "vib_band_bpfo", "vib_band_bpfi", "vib_band_proc", "vib_band_hf",
    "vib_rms", "temp_c", "current_a", "power_kw", "pressure_kpa", "flow_m3h",
}  # fmt: skip


_MONTHS = (
    "jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|"
    "sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
)
_LONG_DATE = re.compile(
    rf"\b(?:\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{_MONTHS})\.?,?\s+\d{{4}}"
    rf"|(?:{_MONTHS})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}})\b",
    flags=re.IGNORECASE,
)

# Spelled-out quantities are claims too ("fails in two days", "forty-eight hours"). "one" is only
# a quantity next to a unit ("one day"); on its own it is ordinary prose.
_NUMBER_WORDS = re.compile(
    r"\b(?:two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|"
    r"fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|"
    r"eighty|ninety|hundred|thousand|dozen|half|twice|double|triple|several|couple)\b"
    r"|\bone\s+(?:day|hour|week|month|shift)s?\b"
    r"|\b(?:a|an)\s+(?:day|hour|week|month|fortnight)\b",
    flags=re.IGNORECASE,
)

# Fault vocabulary. Each pattern names what it implies; a match is allowed only if the order's
# primary or alternative hypotheses include one of those, or if the phrase is part of the order's
# own text (its recommended action, for instance, mentions "liner clearances" on a pump).
_FAULT_TERMS: list[tuple[re.Pattern[str], frozenset[FaultHypothesis]]] = [
    (
        re.compile(r"outer[- ]?race|\bbpfo\b", re.IGNORECASE),
        frozenset({FaultHypothesis.BEARING_OUTER_RACE}),
    ),
    (
        re.compile(r"inner[- ]?race|\bbpfi\b", re.IGNORECASE),
        frozenset({FaultHypothesis.BEARING_INNER_RACE}),
    ),
    (
        re.compile(r"\bbearings?\b|\bspall\w*|\brolling[- ]element", re.IGNORECASE),
        frozenset(
            {
                FaultHypothesis.BEARING_OUTER_RACE,
                FaultHypothesis.BEARING_INNER_RACE,
                FaultHypothesis.BEARING_GENERIC,
            }
        ),
    ),
    (
        re.compile(r"\bimpellers?\b|\bvane[- ]pass", re.IGNORECASE),
        frozenset({FaultHypothesis.IMPELLER_WEAR}),
    ),
    (
        re.compile(r"\bliners?\b|\breline\w*|\bconcaves?\b|\bmantles?\b", re.IGNORECASE),
        frozenset({FaultHypothesis.LINER_WEAR}),
    ),
    (
        re.compile(r"overheat\w*|over-heat\w*|\bcoolant\b|\bcooling\b|\bradiator\b", re.IGNORECASE),
        frozenset({FaultHypothesis.OVERHEATING}),
    ),
]
# Fault names outside the rule layer's vocabulary: never a hypothesis of any order.
_FOREIGN_FAULTS = re.compile(
    r"cavitat\w*|misalign\w*|\bunbalance\w*|\bimbalance\w*|\blooseness\b|\bsoft[- ]foot\b|"
    r"\bgear\w*|\bcrack\w*|\bcorro\w*|contaminat\w*|\brotor[- ]bar\w*|\bresonan\w*|"
    r"\bcoupling\w*|\beccentric\w*|\brub(?:bing)?\b|\binsulation\b|\bwinding\w*|\bseals?\b|"
    r"\bfatigue\b|\bfracture\w*|\bshaft (?:bend|bow|crack)\w*",
    flags=re.IGNORECASE,
)
_SEVERITY_WORDS = {s: re.compile(rf"\b{s.value}\b", re.IGNORECASE) for s in Severity}
_ESCALATION = re.compile(
    r"\btrip\w*\b|\bshut\s*(?:it\s+)?down\b|\bshutdown\b|\bemergency\b|\bimmediate(?:ly)?\b|"
    r"\bstop (?:the|it)\b|\bat once\b|\bright now\b",
    flags=re.IGNORECASE,
)
# A number bound to a named RUL quantity must be that quantity.
_RUL_BINDINGS = (
    (re.compile(r"\b(?:median|p50)\b", re.IGNORECASE), "p50_days"),
    (
        re.compile(
            r"\b(?:worst[- ]case|pessimistic|p10|at the earliest|as early as)\b", re.IGNORECASE
        ),
        "p10_days",
    ),
    (
        re.compile(r"\b(?:best[- ]case|optimistic|p90|at the latest|as late as)\b", re.IGNORECASE),
        "p90_days",
    ),
)


def _allowed_numbers(wo: WorkOrder) -> set[float]:
    nums: set[float] = {float(len(wo.evidence))}
    for e in wo.evidence:
        nums.update({e.observed, e.expected, e.z_score})
    r = wo.rul
    if r is not None:
        for v in (r.p10_days, r.p50_days, r.p90_days, r.health_index, r.failure_threshold):
            if v is not None:
                nums.add(float(v))
    # numbers that are themselves fields of the order: identifiers, the data-quality note and the
    # recommended action (for example "re-inspect within 72 hours")
    for token in re.findall(r"\d+(?:\.\d+)?", _order_text(wo)):
        nums.add(float(token))
    nums.update({80.0})  # the stated interval level
    # Dates are checked as whole dates (ISO or written out); their year, month and day are NOT
    # free numbers ("27 bearings", "risen 5-fold" must not pass because a date has a 27 or a 5).
    return nums


def _order_text(wo: WorkOrder) -> str:
    return " ".join(
        [wo.asset_id, wo.asset_name, wo.work_order_id, wo.data_quality_note, wo.recommended_action]
    )


def _close(x: float, allowed: set[float]) -> bool:
    return any(abs(x - a) <= max(0.06, 0.006 * abs(a)) for a in allowed)


def _long_dates(wo: WorkOrder) -> set[str]:
    out: set[str] = set()
    for d in (wo.issued_at, wo.latest_safe_date):
        if d is not None:
            out.add(f"{d.year}-{d.month:02d}-{d.day:02d}")
    return out


def _parse_long_date(s: str) -> str | None:
    import pandas as pd

    try:
        ts = pd.Timestamp(re.sub(r"(\d)(st|nd|rd|th)", r"\1", s).replace(",", " "))
    except (ValueError, TypeError):
        return None
    return f"{ts.year}-{ts.month:02d}-{ts.day:02d}"


def check_narrative(text: str, wo: WorkOrder) -> list[str]:
    """Return a list of guardrail violations; an empty list means the narrative is acceptable.

    Checked: dates (ISO or written out), every digit-number, spelled-out quantities, feature
    names, fault vocabulary (the order's hypotheses only, and nothing outside the rule layer's
    vocabulary), severity words above the order's severity, escalation language on an order that
    is not CRITICAL, and numbers bound to a named RUL quantity (median, worst case, best case).
    It remains a text check: a claim made without any of these words is not caught.
    """
    problems: list[str] = []
    own = _order_text(wo).lower()
    allowed_dates = {f"{d:%Y-%m-%d}" for d in (wo.issued_at, wo.latest_safe_date) if d is not None}
    for d in _DATE.findall(text):
        if d not in allowed_dates:
            problems.append(f"date {d} is not in the work order")
    scrubbed = _DATE.sub(" ", text)
    long_ok = _long_dates(wo)
    for m in _LONG_DATE.finditer(scrubbed):
        iso = _parse_long_date(m.group(0))
        if iso is None or iso not in long_ok:
            problems.append(f"date '{m.group(0)}' is not in the work order")
    scrubbed = _LONG_DATE.sub(" ", scrubbed)
    for e in wo.evidence:
        if e.unit:  # units such as "(mm/s)^2" contain digits that are not claims
            scrubbed = scrubbed.replace(e.unit, " ")
    allowed = _allowed_numbers(wo)
    for tok in _NUM.findall(scrubbed):
        if not _close(float(tok), allowed):
            problems.append(f"number {tok} is not in the work order")
    for m in _NUMBER_WORDS.finditer(text):
        if m.group(0).lower() not in own:
            problems.append(f"spelled-out quantity '{m.group(0)}' is not a field of the order")
    problems.extend(_rul_binding_problems(text, wo))

    cited = {f for f in _KNOWN_FEATURES if re.search(rf"\b{re.escape(f)}\b", text)}
    allowed_feats = {e.feature for e in wo.evidence}
    for f in sorted(cited - allowed_feats):
        problems.append(f"feature {f} is not among the evidence")
    # feature names contain fault words ("vib_band_bpfo"); judge the prose without them
    prose = text
    for f in cited:
        prose = re.sub(rf"\b{re.escape(f)}\b", " ", prose)
    allowed_h = {wo.failure_mode_hypothesis, *wo.alternative_hypotheses}
    for h, label in HYPOTHESIS_TEXT.items():
        if h not in allowed_h and re.search(re.escape(label), prose, flags=re.IGNORECASE):
            problems.append(f"fault '{label}' is not a hypothesis of this work order")
    for pattern, implies in _FAULT_TERMS:
        for m in pattern.finditer(prose):
            if not (implies & allowed_h) and m.group(0).lower() not in own:
                problems.append(f"fault term '{m.group(0)}' does not match the hypotheses")
    for m in _FOREIGN_FAULTS.finditer(prose):
        if m.group(0).lower() not in own:
            problems.append(f"fault '{m.group(0)}' is not a hypothesis of this work order")

    for sev, pattern in _SEVERITY_WORDS.items():
        if sev.rank > wo.severity.rank and pattern.search(prose):
            problems.append(f"severity '{sev.value}' is above the order's '{wo.severity.value}'")
    if wo.severity != Severity.CRITICAL:
        for m in _ESCALATION.finditer(prose):
            if m.group(0).lower() not in own:
                problems.append(
                    f"escalation '{m.group(0)}' is not warranted at {wo.severity.value}"
                )
    return problems


def _rul_binding_problems(text: str, wo: WorkOrder) -> list[str]:
    """A number written next to "median", "worst case" ... must be that RUL quantile."""
    r = wo.rul
    if r is None:
        return []
    out: list[str] = []
    for sentence in re.split(r"(?<=[.;:!?])\s+", text):
        clean = _DATE.sub(" ", sentence)
        for pattern, attr in _RUL_BINDINGS:
            for m in pattern.finditer(clean):
                want = getattr(r, attr)
                after = _NUM.search(clean, m.end())
                if after is None or after.start() - m.end() > 40:
                    continue
                got = float(after.group(0))
                if want is None or abs(got - want) > max(0.06, 0.006 * abs(want)):
                    out.append(
                        f"'{m.group(0)}' is followed by {after.group(0)}, but the order's "
                        f"{attr.replace('_days', '')} is {want}"
                    )
    return out


def narrate(wo: WorkOrder, generator: NarrativeGenerator | None = None) -> WorkOrder:
    """Attach a narrative. ``generator`` output is used only if it passes the guardrail."""
    fallback = template_narrative(wo)
    if generator is None:
        return wo.model_copy(update={"narrative": fallback, "narrative_source": "template"})
    try:
        text = generator.generate(wo)
    except Exception:  # an LLM outage must never block a work order
        text = ""
    if text and not check_narrative(text, wo):
        return wo.model_copy(update={"narrative": text, "narrative_source": "llm"})
    return wo.model_copy(
        update={"narrative": fallback, "narrative_source": "template_after_guardrail"}
    )
