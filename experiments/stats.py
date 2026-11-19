
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

STATS_VERSION = "scope-exp-stats-v1"


def wilson_interval(successes: int, total: int, *, alpha: float = 0.05) -> tuple[float, float] | None:
    if total <= 0:
        return None
    z = _z_for_alpha(alpha)
    phat = successes / total
    denominator = 1.0 + z * z / total
    centre = (phat + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(phat * (1.0 - phat) / total + z * z / (4.0 * total * total)) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def _z_for_alpha(alpha: float) -> float:
    p = 1.0 - alpha / 2.0
    if not 0.0 < p < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    t = math.sqrt(-2.0 * math.log(1.0 - p)) if p > 0.5 else math.sqrt(-2.0 * math.log(p))
    coefficients = (2.515517, 0.802853, 0.010328)
    denominators = (1.0, 1.432788, 0.189269, 0.001308)
    numerator = coefficients[0] + coefficients[1] * t + coefficients[2] * t * t
    denominator = 1.0 + denominators[1] * t + denominators[2] * t * t + \
        denominators[3] * t * t * t
    z = t - numerator / denominator
    return z if p > 0.5 else -z


def rate_summary(successes: int, total: int, *, unresolved: int = 0,
                 alpha: float = 0.05) -> dict[str, Any]:
    interval = wilson_interval(successes, total, alpha=alpha)
    record: dict[str, Any] = {
        "successes": int(successes),
        "total": int(total),
        "rate": (successes / total) if total else None,
        "wilson_lower": interval[0] if interval else None,
        "wilson_upper": interval[1] if interval else None,
        "alpha": alpha,
        "unresolved": int(unresolved),
        "complete_queue_range": None,
        "interval_scope": None,
    }
    if total:
        upper_complete = (successes + unresolved) / total
        record["complete_queue_range"] = [successes / total, upper_complete]
        record["interval_scope"] = ("complete queue" if unresolved == 0 else
                                    "conditional on resolved tasks; the complete-queue "
                                    "range accounts for the unresolved tasks")
    if total and successes == total and total == 10:
        record["note"] = ("10/10 gives a Wilson 95% lower bound of about 0.722; this is "
                          "not evidence of 99% reliability")
    return record


def describe(values: Sequence[float], *, alpha: float = 0.05,
             bootstrap_resamples: int = 10000, seed: int = 20261008) -> dict[str, Any]:
    cleaned = [float(value) for value in values if value is not None
               and math.isfinite(float(value))]
    n = len(cleaned)
    record: dict[str, Any] = {"n": n, "mean": None, "sd": None, "median": None,
                              "cv": None, "sem": None, "ci_low": None, "ci_high": None,
                              "min": None, "max": None, "values": cleaned,
                              "bootstrap_resamples": bootstrap_resamples,
                              "note": None}
    if n == 0:
        record["note"] = "n=0: no mean is reported"
        return record
    array = np.array(cleaned, dtype=np.float64)
    record.update(mean=float(np.mean(array)), median=float(np.median(array)),
                  min=float(np.min(array)), max=float(np.max(array)))
    if n == 1:
        record["note"] = "n=1: no estimable SD or interval"
        return record
    sd = float(np.std(array, ddof=1))
    record.update(sd=sd, sem=sd / math.sqrt(n),
                  cv=(sd / float(np.mean(array))) if np.mean(array) != 0 else None)
    low, high = bootstrap_mean_interval(array, resamples=bootstrap_resamples, seed=seed,
                                        alpha=alpha)
    record.update(ci_low=low, ci_high=high)
    return record


def bootstrap_mean_interval(values: np.ndarray, *, resamples: int = 10000,
                            seed: int = 20261008, alpha: float = 0.05) -> tuple[float, float]:
    rng = np.random.Generator(np.random.PCG64(seed))
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return (float("nan"), float("nan"))
    draws = rng.integers(0, array.size, size=(resamples, array.size))
    means = array[draws].mean(axis=1)
    return (float(np.quantile(means, alpha / 2.0)),
            float(np.quantile(means, 1.0 - alpha / 2.0)))


def paired_bootstrap(differences: Sequence[float], *, resamples: int = 10000,
                     seed: int = 20261008, alpha: float = 0.05) -> dict[str, Any]:
    cleaned = [float(value) for value in differences if value is not None
               and math.isfinite(float(value))]
    n = len(cleaned)
    if n == 0:
        return {"n": 0, "mean_difference": None, "ci_low": None, "ci_high": None,
                "note": "n=0: no paired difference is reported"}
    array = np.array(cleaned, dtype=np.float64)
    if n == 1:
        return {"n": 1, "mean_difference": float(array[0]), "ci_low": None,
                "ci_high": None, "note": "n=1: no estimable interval"}
    low, high = bootstrap_mean_interval(array, resamples=resamples, seed=seed, alpha=alpha)
    return {"n": n, "mean_difference": float(np.mean(array)), "sd": float(np.std(array, ddof=1)),
            "ci_low": low, "ci_high": high, "resamples": resamples,
            "note": "paired by parent task; resampling keeps the pairs together"}


def effective_throughput(deliveries: Sequence[bool],
                         throughputs: Sequence[float | None]) -> dict[str, Any]:
    values: list[float] = []
    undelivered = 0
    unknown = 0
    for delivered, throughput in zip(deliveries, throughputs):
        if delivered is None:
            unknown += 1
            values.append(0.0)
            continue
        if not delivered:
            undelivered += 1
            values.append(0.0)
            continue
        if throughput is None:
            unknown += 1
            values.append(0.0)
            continue
        values.append(float(throughput))
    summary = describe(values)
    summary.update(undelivered=undelivered, unknown_throughput=unknown,
                   definition="Y_i = delivered_i * throughput_i (0 when undelivered)",
                   interpretation="expected throughput per generation attempt, not a "
                                  "kernel running speed")
    return summary


def holm_correction(p_values: Sequence[float], *, alpha: float = 0.05) -> list[dict[str, Any]]:
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    m = len(indexed)
    results: list[dict[str, Any]] = [{} for _ in indexed]
    previous_reject = True
    for rank, (original_index, p_value) in enumerate(indexed):
        threshold = alpha / (m - rank)
        reject = previous_reject and p_value <= threshold
        previous_reject = reject
        results[original_index] = {"p_value": p_value, "rank": rank + 1,
                                   "holm_threshold": threshold,
                                   "reject_null": bool(reject)}
    return results


def rate_difference(a_successes: int, a_total: int,
                    b_successes: int, b_total: int) -> dict[str, Any]:
    a = rate_summary(a_successes, a_total)
    b = rate_summary(b_successes, b_total)
    difference = None
    if a["rate"] is not None and b["rate"] is not None:
        difference = a["rate"] - b["rate"]
    return {"a": a, "b": b, "difference": difference,
            "note": "intervals are marginal; overlap is not an equivalence test"}
