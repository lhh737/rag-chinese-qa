"""统计工具（设计 §4.8 实现要求）。

- 比例型指标**在计算模块直接输出 Wilson 95% 区间**（不是事后补）
- 对照实验用配对检验（McNemar），禁止"区间重叠目测"判定
- 不引入重依赖：statistics + 朴素实现
"""
from __future__ import annotations

import math
from statistics import mean

Z95 = 1.959963984540054


def wilson_ci(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson 95% 置信区间（k 成功 / n 总数）。n=0 时返回 (0, 1)。"""
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def proportion(k: int, n: int) -> dict:
    """比例型指标的标准输出：value + Wilson CI + 原始计数。"""
    lo, hi = wilson_ci(k, n)
    return {"value": (k / n if n else 0.0), "ci95": [round(lo, 4), round(hi, 4)], "k": k, "n": n}


def percentile(values: list[float], q: float) -> float:
    """朴素分位数（线性插值）。"""
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def latency_summary(ms_values: list[float]) -> dict:
    return {
        "p50_ms": round(percentile(ms_values, 0.50), 1),
        "p95_ms": round(percentile(ms_values, 0.95), 1),
        "mean_ms": round(mean(ms_values), 1) if ms_values else 0.0,
        "n": len(ms_values),
    }


def mcnemar(b: int, c: int) -> dict:
    """配对检验（McNemar，连续修正）。

    b = 甲对乙错的配上数，c = 甲错乙对的配上数。
    返回卡方值与近似 p（1 自由度卡方，双侧；小样本请以精确检验为准并在报告注明）。
    """
    if b + c == 0:
        return {"chi2": 0.0, "p": 1.0, "b": b, "c": c}
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    # 1 自由度卡方的双侧 p：p = erfc(sqrt(chi2/2))
    p = math.erfc(math.sqrt(chi2 / 2))
    return {"chi2": round(chi2, 4), "p": round(p, 4), "b": b, "c": c}
