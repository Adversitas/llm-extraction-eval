"""Entity-level precision / recall / F1 and operational metrics."""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd

from .data import FIELDS, Doc
from .pipeline import Result


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().casefold()


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def count(pred: dict, gold: dict) -> dict[str, tuple[int, int, int]]:
    """Per field (tp, fp, fn) with multiset matching on normalised strings."""
    out = {}
    for f in FIELDS:
        p = Counter(norm(x) for x in pred.get(f, []))
        g = Counter(norm(x) for x in gold.get(f, []))
        tp = sum((p & g).values())
        out[f] = (tp, sum(p.values()) - tp, sum(g.values()) - tp)
    return out


def score(results: list[Result], docs: list[Doc]) -> pd.DataFrame:
    gold = {d.id: d.gold for d in docs}
    rows = []
    by_strategy: dict[str, list[Result]] = {}
    for r in results:
        by_strategy.setdefault(r.strategy, []).append(r)
    for name, rs in by_strategy.items():
        totals = {f: np.zeros(3, dtype=int) for f in FIELDS}
        for r in rs:
            for f, c in count(r.prediction, gold[r.doc_id]).items():
                totals[f] += np.array(c)
        all_c = sum(totals.values())
        p, rc, f1 = prf(*all_c)
        row = {
            "strategy": name,
            "precision": p,
            "recall": rc,
            "micro_f1": f1,
            **{f"f1_{f}": prf(*totals[f])[2] for f in FIELDS},
            "valid_first_pass": float(np.mean([r.first_pass_valid for r in rs])),
            "valid_final": float(np.mean([r.final_valid for r in rs])),
            "hallucinated_per_100_docs": 100 * float(np.mean([r.hallucinated for r in rs])),
            "calls_per_doc": float(np.mean([r.calls for r in rs])),
            "prompt_tokens_per_doc": float(np.mean([r.prompt_tokens for r in rs])),
            "completion_tokens_per_doc": float(np.mean([r.completion_tokens for r in rs])),
            "ms_per_doc_p50": 1000 * float(np.median([r.seconds for r in rs])),
            "ms_per_doc_p95": 1000 * float(np.quantile([r.seconds for r in rs], 0.95)),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def cohen_kappa(a: list[int], b: list[int]) -> float:
    a_arr, b_arr = np.asarray(a), np.asarray(b)
    po = float((a_arr == b_arr).mean())
    pa, pb = a_arr.mean(), b_arr.mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0
