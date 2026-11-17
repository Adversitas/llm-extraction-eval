"""LLM-as-judge, checked against human annotations.

Each item is one extracted entity with its type. The *human* verdict is whether it matches an
annotated entity of that type in CoNLL-2003. The judge model sees the sentence and the
candidate and answers yes or no. We then measure how far the judge can be trusted: agreement
and Cohen's kappa with the human verdicts, its bias (how often it says yes), and run-to-run
consistency when sampling at temperature > 0.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

import numpy as np

from .data import FIELDS, Doc
from .llm import Backend
from .metrics import cohen_kappa, norm
from .pipeline import Result

LABEL = {
    "persons": "a person",
    "organizations": "an organization",
    "locations": "a location",
    "misc": "a miscellaneous named entity (nationality, event, product, ...)",
}

JUDGE_SYSTEM = (
    "You are a strict annotation reviewer for named-entity extraction. Answer with a single "
    "word: yes or no."
)


@dataclass
class Item:
    doc_id: str
    text: str
    entity: str
    field: str
    human: int  # 1 = correct according to the gold annotation


def build_items(results: list[Result], docs: list[Doc], n: int, seed: int = 0) -> list[Item]:
    """Balanced sample of correct and incorrect extracted entities."""
    by_id = {d.id: d for d in docs}
    pos, neg = [], []
    for r in results:
        doc = by_id[r.doc_id]
        gold = {f: {norm(x) for x in doc.gold[f]} for f in FIELDS}
        for f in FIELDS:
            for e in r.prediction.get(f, []):
                item = Item(doc.id, doc.text, e, f, int(norm(e) in gold[f]))
                (pos if item.human else neg).append(item)
    rng = random.Random(seed)
    rng.shuffle(pos)
    rng.shuffle(neg)
    half = n // 2
    items = pos[:half] + neg[: n - half]
    rng.shuffle(items)
    return items


def prompt(item: Item) -> list[dict[str, str]]:
    q = (
        f'Sentence: {item.text}\n\nCandidate entity: "{item.entity}"\n'
        f'Question: is "{item.entity}" a complete named entity that appears in the sentence '
        f"and refers to {LABEL[item.field]}? Answer yes or no."
    )
    return [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": q}]


_YES = re.compile(r"^\W*(yes|oui)\b", re.I)


def run_judge(backend: Backend, items: list[Item], temperature: float = 0.0) -> list[int]:
    gens = backend.generate([prompt(i) for i in items], max_new_tokens=4, temperature=temperature)
    return [int(bool(_YES.match(g.text))) for g in gens]


def agreement(human: list[int], judge: list[int]) -> dict:
    h, j = np.asarray(human), np.asarray(judge)
    return {
        "items": len(h),
        "accuracy_vs_human": float((h == j).mean()),
        "cohen_kappa": cohen_kappa(list(h), list(j)),
        "judge_yes_rate": float(j.mean()),
        "human_yes_rate": float(h.mean()),
        "true_positive_rate": float(j[h == 1].mean()) if (h == 1).any() else float("nan"),
        "true_negative_rate": float(1 - j[h == 0].mean()) if (h == 0).any() else float("nan"),
    }
