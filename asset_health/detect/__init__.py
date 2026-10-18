"""Detection: robust CUSUM, Isolation Forest and an explainable rule layer, fused."""

from asset_health.detect.cusum import cusum_arrays, cusum_frame
from asset_health.detect.ensemble import Detection, detect_asset
from asset_health.detect.rules import DEFAULT_RULES, Cond, Rule, best_hypothesis, rule_strengths

__all__ = [
    "DEFAULT_RULES",
    "Cond",
    "Detection",
    "Rule",
    "best_hypothesis",
    "cusum_arrays",
    "cusum_frame",
    "detect_asset",
    "rule_strengths",
]
