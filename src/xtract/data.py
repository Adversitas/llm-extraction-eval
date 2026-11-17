"""CoNLL-2003 (English news, human-annotated named entities) as an extraction benchmark.

The corpus comes from Reuters news and may not be redistributed, so it is downloaded at run
time from the Hugging Face Hub parquet export and never committed to this repository.
"""

from __future__ import annotations

import random
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

HF_URL = (
    "https://huggingface.co/datasets/eriktks/conll2003/resolve/refs%2Fconvert%2Fparquet/"
    "conll2003/{split}/0000.parquet"
)
DATA_DIR = Path("data")
# CoNLL-2003 BIO tag ids -> entity type
TAGS = {
    1: "B-PER",
    2: "I-PER",
    3: "B-ORG",
    4: "I-ORG",
    5: "B-LOC",
    6: "I-LOC",
    7: "B-MISC",
    8: "I-MISC",
}
TYPE_TO_FIELD = {"PER": "persons", "ORG": "organizations", "LOC": "locations", "MISC": "misc"}
FIELDS = list(TYPE_TO_FIELD.values())


@dataclass
class Doc:
    id: str
    text: str
    gold: dict[str, list[str]] = field(default_factory=dict)


def download(split: str, data_dir: Path = DATA_DIR) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / f"conll2003_{split}.parquet"
    if not path.exists():
        urllib.request.urlretrieve(HF_URL.format(split=split), path)
    return path


def bio_to_entities(tokens: list[str], tags: list[int]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {f: [] for f in FIELDS}
    current: list[str] = []
    etype: str | None = None

    def flush() -> None:
        if current and etype:
            out[TYPE_TO_FIELD[etype]].append(" ".join(current))

    for tok, tag_id in zip(tokens, tags, strict=True):
        tag = TAGS.get(int(tag_id), "O")
        if tag.startswith("B-") or (tag.startswith("I-") and tag[2:] != etype):
            flush()
            current, etype = [tok], tag[2:]
        elif tag.startswith("I-"):
            current.append(tok)
        else:
            flush()
            current, etype = [], None
    flush()
    return out


def frame_to_docs(df: pd.DataFrame) -> list[Doc]:
    docs = []
    for row in df.itertuples():
        tokens = list(row.tokens)
        if not tokens or tokens[0] == "-DOCSTART-":
            continue
        docs.append(Doc(str(row.id), " ".join(tokens), bio_to_entities(tokens, list(row.ner_tags))))
    return docs


def load_docs(split: str, data_dir: Path = DATA_DIR) -> list[Doc]:
    return frame_to_docs(pd.read_parquet(download(split, data_dir)))


def sample(docs: list[Doc], n: int, seed: int = 0, min_tokens: int = 6) -> list[Doc]:
    """Seeded sample of sentences long enough to be realistic (headers and scores dropped)."""
    pool = [d for d in docs if len(d.text.split()) >= min_tokens]
    rng = random.Random(seed)
    return rng.sample(pool, min(n, len(pool)))


def few_shot_examples(train: list[Doc], k: int = 3, seed: int = 0) -> list[Doc]:
    """Pick k short training sentences that together cover all four entity types."""
    rng = random.Random(seed)
    pool = [d for d in train if 8 <= len(d.text.split()) <= 25]
    rng.shuffle(pool)
    chosen: list[Doc] = []
    covered: set[str] = set()
    for d in pool:
        types = {f for f in FIELDS if d.gold[f]}
        if len(types - covered) >= 1 and len(types) >= 2:
            chosen.append(d)
            covered |= types
        if len(chosen) == k:
            break
    return chosen
