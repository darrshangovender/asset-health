# Evaluation methodology

This page is what to read before trusting, or attacking, a number in the README. Everything here is
**synthetic data**: the point is a reproducible, leak-free protocol, not a claim about any plant.

## 1. What is being scored

The unit is the **event**, not the sample. Per-sample accuracy is meaningless for maintenance
(thousands of healthy hours per failure; a detector that always says "healthy" scores 99.9%).

* A **failure** is an injected degradation run `[onset, failure]` with a ground-truth timestamp
  (`simulate/degradation.py`).
* An **alert episode** opens after a debounce (`confirm_hours` consecutive valid hours above the
  threshold). The time it opens is the time an operator would actually have been told.
* A **detection** is an episode that opens inside a failure window on the same asset once the
  injected degradation has reached a latent severity of at least 0.05 (on the 0-1 scale). **Lead
  time** is `failure - opened_at`. Only the first such episode in a window counts; later ones are
  re-alerts.
* An episode that opens inside the window *before* that (nothing physical to find yet) is a
  **coincidental** alarm: scored as a false alarm and counted separately. Without this rule a noisy
  baseline earns "detections" from alarms that merely fall inside a 45-90 day liner or impeller
  window.
* Everything else is a **false alarm**: opened before onset, after the failure, or on an asset that
  never fails. The rate is per **healthy monitored asset-month**, where "monitored" means the hours
  that policy was itself watching: valid-decision hours for the system, armed-limit hours with a
  reading for a fixed limit; failure windows are excluded.
* RUL is scored at the system's own daily checkpoints inside detected episodes, against
  `failure - checkpoint`. The primary rows are the **decision-time** checkpoints, up to the planned
  replacement; after that the part is gone in operation, so later rows are counterfactual (and
  easy). The first estimate per episode and all checkpoints up to failure are reported alongside.
  Metrics: MAE of the median, empirical coverage of the 10-90% interval and its mean width, overall
  and by true-RUL bucket, against a trend-only and a constant predictor. Consecutive checkpoints of
  one episode are strongly correlated, so the effective sample is closer to the number of episodes
  than the number of rows.
* Cost is charged identically to every alarm-driven policy (inspection per alert, planned
  replacement plus wasted life, or unplanned failure). Costs are illustrative.

## 2. Seeds and leakage controls

| Role | Seeds | Used for |
|---|---|---|
| development | 1001-1006 | training the RUL quantile model; tuning the alert policy and both baselines |
| calibration | 1101-1103 | split-conformal margin of the RUL interval only |
| held-out test | 3001-3008 | **every number in the README** |
| rarer-failure scenario | 3101-3104 | sensitivity row, nothing re-tuned |

* `assert_disjoint` raises if any two sets overlap; a test also checks that the **shipped** RUL
  model's recorded training and calibration seeds never intersect the test seeds.
* There is no cross-asset or cross-seed learning inside the pipeline. Each asset's healthy
  baseline (regime regression, residual sigma, Isolation Forest, its alarm calibration) is learned
  from that asset's own first weeks after commissioning or repair, and is re-learned after a
  repair record.
* All detector and feature maths is **causal**: a score at time `t` uses rows up to `t` only.
  `tests/test_e2e.py` re-runs the pipeline on a history truncated mid-degradation and checks that
  every earlier score is identical. The RUL trend uses a trailing window; `fit_trend` is tested to
  ignore the future.
* Design iteration (feature set for RUL, detector parameters) used development and calibration
  seeds only. The held-out seeds were generated once, after the pipeline was frozen.
* Ground truth (`ground_truth.json`) is read only by the evaluation code and, optionally, by the
  report to draw red markers. No detector, rule or model takes it as input. The CMMS maintenance
  log (planned services, completed repairs) *is* an input, as it would be in a plant.

## 3. Baselines, and why they are fair

* **Fixed statistical limits**, two variants: median + `k` robust sigma from the commissioning
  window on (a) raw overall vibration and raw temperature, and (b) overall vibration and
  temperature *rise above the logged ambient* (which removes the summer false alarms a winter
  baseline produces on raw temperature). 3 h debounce, no regime model, no data-quality gate. `k`
  is tuned **per channel** on the development seeds by cost, with the same 1% plateau rule the
  system's tuning uses (on the plateau, fewest false alarms wins). These are statistical limits;
  engineering limits (ISO 20816 vibration zones, OEM bearing and coolant trip temperatures) were
  not modelled.
* **Age-based (calendar) replacement**: replace the component when its age reaches a fraction of
  its class's characteristic Weibull life (fraction tuned on the development seeds; on the plateau
  the longest interval wins). Components wear out (Weibull shape > 1) and have ages: a part that
  fails on day `f` of the record has a life drawn from the class Weibull conditioned on exceeding
  `f`, so old parts fail early; surviving parts have the steady-state age of parts that outlive the
  record. A part already over the age limit at the start is replaced on day 0. Any replacement
  before the failure prevents it (the new part has a fresh life). Like every part fitted during the
  record, including the system's replacements, the new part is assumed to outlive the record;
  `results.json` (`life_model`) gives the probability that this is wrong for each class.
* **Run to failure** and an **oracle** (replaces every failing part three days before it fails,
  never inspects a healthy asset) bracket the scale.

## 3a. Severity sweep

The designed signatures are large (a bearing's broadband energy reaches 4-10x baseline at
failure), so "detected" at the designed size says little. The benchmark rebuilds the held-out seeds
with every failure's signal amplitude multiplied by 0.5, 0.3 and 0.15
(`SimConfig.degradation_scale`; failure times, data faults and seeds unchanged) and scores the
system and both fixed limits with nothing re-tuned. The README headline is this curve.

## 4. Ablations

Detector ablations (CUSUM only, Isolation Forest only, rule layer only, CUSUM + IF) re-tune the
alert threshold on the development seeds, so each variant gets its own best operating point. The
"no debounce / hysteresis" row changes the policy only. The "no data-quality gating" and "no regime
normalisation" rows re-run the pipeline on a subset of the test seeds with the component disabled
and the **full system's policy unchanged** (not re-tuned), which is the usual meaning of removing a
component; it can flatter the full system, so read the false-alarm column as the signal.

## 5. Known weaknesses of the protocol

* Synthetic failures are generated by the same family of curves the RUL model and health index
  assume (smooth monotone degradation). Real degradation is lumpier. Detection and RUL numbers here
  are an upper bound on what the same code does on real data.
* The inspection step in the cost model is an oracle (an inspection always distinguishes a degrading
  part from a healthy one, once the fault has reached the 0.05 severity floor).
* Unplanned failure is a flat cost per asset class; the simulated downtime (36-96 h) is not priced,
  so lead time matters only through the three-day mobilisation cliff. A false alarm costs 1/80 of a
  failure, so alarm fatigue does not show up in the cost.
* The RUL model's class prior (typical P-F interval) is the range the simulator draws degradation
  durations from, and its failure level is learned from runs that end at severity exactly 1.0. Both
  are regularities a plant does not hand over; on real data failures are censored by preventive
  replacement.
* With 8 test seeds, about 80 failures, rate confidence intervals are wide: read the Wilson
  intervals, not the point estimates.
* Several quantities (policy grid, RUL features) were chosen on seven-plus seeds with a coarse
  grid. Some optimism from selecting the minimum-cost grid point survives; the held-out seeds are
  what protect against it.
