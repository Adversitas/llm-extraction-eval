import json

import pandas as pd
import pytest

from xtract import data
from xtract.judge import Item, agreement, build_items, run_judge
from xtract.llm import FakeBackend
from xtract.metrics import cohen_kappa, count, prf, score
from xtract.pipeline import run_all, run_strategy, verify_spans
from xtract.schema import Entities, extract_json_block, parse

DOCS = [
    data.Doc(
        "1",
        "Peter Blackburn met EU officials in Brussels on Monday .",
        {
            "persons": ["Peter Blackburn"],
            "organizations": ["EU"],
            "locations": ["Brussels"],
            "misc": [],
        },
    ),
    data.Doc(
        "2",
        "German striker Klinsmann joined Bayern Munich .",
        {
            "persons": ["Klinsmann"],
            "organizations": ["Bayern Munich"],
            "locations": [],
            "misc": ["German"],
        },
    ),
]


def perfect(msgs):
    text = msgs[-1]["content"].removeprefix("Sentence: ")
    doc = next(d for d in DOCS if d.text == text)
    return "```json\n" + json.dumps(doc.gold) + "\n```"


def test_bio_to_entities_handles_adjacent_and_broken_spans():
    tokens = ["EU", "rejects", "German", "call", "by", "Peter", "Blackburn", "in", "New", "York"]
    tags = [3, 0, 7, 0, 0, 1, 2, 0, 5, 6]
    ents = data.bio_to_entities(tokens, tags)
    assert ents == {
        "persons": ["Peter Blackburn"],
        "organizations": ["EU"],
        "locations": ["New York"],
        "misc": ["German"],
    }
    # an I- tag of a new type starts a new entity
    assert data.bio_to_entities(["A", "B"], [1, 4]) == {
        "persons": ["A"],
        "organizations": ["B"],
        "locations": [],
        "misc": [],
    }


def test_frame_to_docs_skips_docstart():
    df = pd.DataFrame(
        {"id": ["0", "1"], "tokens": [["-DOCSTART-"], ["Paris", "wins"]], "ner_tags": [[0], [5, 0]]}
    )
    docs = data.frame_to_docs(df)
    assert len(docs) == 1 and docs[0].gold["locations"] == ["Paris"]


def test_extract_json_block_and_parse():
    assert extract_json_block('noise {"a": "}"} tail') == '{"a": "}"}'
    ok, err = parse('Here: {"persons": ["X"], "organizations": [], "locations": [], "misc": []}')
    assert err is None and ok.persons == ["X"]
    assert parse("no json")[1] == "no JSON object found"
    assert parse('{"persons": "X"}')[1].startswith("schema error")
    assert parse('{"persons": [], "dates": []}')[1].startswith("schema error")
    assert parse("{'persons': []}")[1].startswith("invalid JSON")


def test_verify_spans_drops_hallucinations_and_duplicates():
    ents = Entities(
        persons=["Peter Blackburn", "peter blackburn", "John Smith"], locations=["Brussels"]
    )
    clean, dropped = verify_spans(ents, DOCS[0].text)
    assert clean.persons == ["Peter Blackburn"] and clean.locations == ["Brussels"]
    assert dropped == 1


def test_metrics_counts_and_prf():
    c = count({"persons": ["A", "B"]}, {"persons": ["a", "C"]})
    assert c["persons"] == (1, 1, 1)
    assert prf(1, 1, 1) == pytest.approx((0.5, 0.5, 0.5))
    assert prf(0, 0, 0) == (0.0, 0.0, 0.0)
    assert cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert cohen_kappa([1, 1, 0, 0], [1, 0, 1, 0]) == pytest.approx(0.0)


def test_perfect_backend_scores_one():
    results = run_all(FakeBackend(perfect), DOCS, examples=[])
    table = score(results, DOCS).set_index("strategy")
    assert set(table.index) == {
        "zero-shot",
        "zero-shot+repair",
        "few-shot+repair",
        "few-shot+repair+verify",
    }
    assert (table["micro_f1"] == 1.0).all()


def test_repair_loop_recovers_invalid_json():
    calls = []

    def flaky(msgs):
        calls.append(msgs)
        if len(msgs) <= 2:  # first pass
            return "Sure! persons: Peter"
        return json.dumps({"persons": ["Peter Blackburn"]})

    res, _ = run_strategy(FakeBackend(flaky), DOCS[:1], "z", repair=True)
    r = res[0]
    assert not r.first_pass_valid and r.final_valid and r.repaired and r.calls == 2
    assert "rejected" in calls[-1][-1]["content"]


def test_hallucinations_counted_without_verify():
    fake = FakeBackend(lambda m: json.dumps({"persons": ["Nobody Here"]}))
    res, _ = run_strategy(fake, DOCS[:1], "z")
    assert res[0].hallucinated == 1
    res, _ = run_strategy(fake, DOCS[:1], "z", verify=True)
    assert res[0].hallucinated == 0 and res[0].prediction["persons"] == []


def test_judge_items_are_balanced_and_scored():
    wrong = FakeBackend(
        lambda m: json.dumps({"persons": ["Brussels", "Peter Blackburn"], "locations": ["Monday"]})
    )
    res, _ = run_strategy(wrong, DOCS[:1], "z")
    items = build_items(res, DOCS, n=2)
    assert sorted(i.human for i in items) == [0, 1]
    always_yes = FakeBackend(lambda m: "Yes.")
    verdicts = run_judge(always_yes, items)
    a = agreement([i.human for i in items], verdicts)
    assert a["judge_yes_rate"] == 1.0 and a["true_negative_rate"] == 0.0
    assert isinstance(items[0], Item)


def test_few_shot_examples_cover_types():
    train = DOCS * 5
    for i, d in enumerate(train):
        train[i] = data.Doc(str(i), d.text + " extra words here to pass length", d.gold)
    ex = data.few_shot_examples(train, k=2)
    assert 1 <= len(ex) <= 2
