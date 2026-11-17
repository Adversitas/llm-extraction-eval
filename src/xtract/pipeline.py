"""Prompting strategies: zero-shot, few-shot, validate-and-repair, span verification."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .data import FIELDS, Doc
from .llm import Backend, Generation, Messages
from .schema import SCHEMA_HINT, Entities, parse

SYSTEM = (
    "You extract named entities from English news sentences. Return only a JSON object with "
    f"exactly these keys: {SCHEMA_HINT}. persons = people; organizations = companies, teams, "
    "institutions, governments; locations = countries, cities, regions, places; misc = "
    "nationalities, events, languages, products and other proper names. Copy every entity "
    "exactly as written in the sentence. Use an empty list when there is none."
)


def user_turn(text: str) -> str:
    return f"Sentence: {text}"


def zero_shot(doc: Doc) -> Messages:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_turn(doc.text)}]


def few_shot(doc: Doc, examples: list[Doc]) -> Messages:
    msgs: Messages = [{"role": "system", "content": SYSTEM}]
    for ex in examples:
        msgs.append({"role": "user", "content": user_turn(ex.text)})
        msgs.append({"role": "assistant", "content": json.dumps({f: ex.gold[f] for f in FIELDS})})
    msgs.append({"role": "user", "content": user_turn(doc.text)})
    return msgs


def repair_turn(messages: Messages, bad_output: str, error: str) -> Messages:
    return [
        *messages,
        {"role": "assistant", "content": bad_output},
        {
            "role": "user",
            "content": f"That answer was rejected ({error}). Reply with only the JSON object "
            f"with keys {', '.join(FIELDS)}, each a list of strings.",
        },
    ]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().casefold()


def verify_spans(ents: Entities, text: str) -> tuple[Entities, int]:
    """Drop entities that do not occur in the sentence (hallucinations) and duplicates."""
    haystack = _norm(text)
    dropped = 0
    clean = {}
    for f in FIELDS:
        seen: list[str] = []
        for e in getattr(ents, f):
            if not isinstance(e, str) or not e.strip() or _norm(e) not in haystack:
                dropped += 1
                continue
            if _norm(e) not in {_norm(s) for s in seen}:
                seen.append(e.strip())
        clean[f] = seen
    return Entities(**clean), dropped


@dataclass
class Result:
    doc_id: str
    strategy: str
    first_pass_valid: bool
    final_valid: bool
    repaired: bool
    prediction: dict
    hallucinated: int
    prompt_tokens: int
    completion_tokens: int
    seconds: float
    calls: int
    errors: list[str] = field(default_factory=list)


def _empty() -> dict:
    return {f: [] for f in FIELDS}


def run_strategy(
    backend: Backend,
    docs: list[Doc],
    name: str,
    examples: list[Doc] | None = None,
    repair: bool = False,
    verify: bool = False,
    max_new_tokens: int = 200,
    first_pass: list[Generation] | None = None,
) -> tuple[list[Result], list[Generation]]:
    """Run one strategy over ``docs``. ``first_pass`` lets strategies that share a prompt
    (e.g. few-shot with and without verification) reuse the same generations."""
    build = (lambda d: few_shot(d, examples)) if examples else zero_shot
    prompts = [build(d) for d in docs]
    gens = first_pass or backend.generate(prompts, max_new_tokens=max_new_tokens)

    parsed = [parse(g.text) for g in gens]
    to_fix = [i for i, (v, _) in enumerate(parsed) if v is None] if repair else []
    fixed: dict[int, Generation] = {}
    if to_fix:
        rprompts = [repair_turn(prompts[i], gens[i].text, parsed[i][1] or "") for i in to_fix]
        fixed = dict(zip(to_fix, backend.generate(rprompts, max_new_tokens), strict=True))

    results = []
    for i, doc in enumerate(docs):
        value, err = parsed[i]
        errors = [err] if err else []
        g = gens[i]
        p_tok, c_tok, secs, calls = g.prompt_tokens, g.completion_tokens, g.seconds, 1
        repaired = False
        if value is None and i in fixed:
            r = fixed[i]
            p_tok, c_tok, secs, calls = (
                p_tok + r.prompt_tokens,
                c_tok + r.completion_tokens,
                secs + r.seconds,
                2,
            )
            value, err2 = parse(r.text)
            repaired = value is not None
            if err2:
                errors.append(err2)
        hallucinated = 0
        if value is not None:
            checked, hallucinated = verify_spans(value, doc.text)
            if verify:
                value = checked
        results.append(
            Result(
                doc_id=doc.id,
                strategy=name,
                first_pass_valid=parsed[i][0] is not None,
                final_valid=value is not None,
                repaired=repaired,
                prediction=value.model_dump() if value is not None else _empty(),
                hallucinated=0 if verify else hallucinated,
                prompt_tokens=p_tok,
                completion_tokens=c_tok,
                seconds=secs,
                calls=calls,
                errors=errors,
            )
        )
    return results, gens


STRATEGIES = {
    # name: (few-shot?, repair?, verify?)
    "zero-shot": (False, False, False),
    "zero-shot+repair": (False, True, False),
    "few-shot+repair": (True, True, False),
    "few-shot+repair+verify": (True, True, True),
}


def run_all(
    backend: Backend, docs: list[Doc], examples: list[Doc], strategies: list[str] | None = None
) -> list[Result]:
    """Run the strategies, generating each distinct first-pass prompt only once."""
    cache: dict[bool, list[Generation]] = {}
    results: list[Result] = []
    for name in strategies or list(STRATEGIES):
        use_few, repair, verify = STRATEGIES[name]
        res, gens = run_strategy(
            backend,
            docs,
            name,
            examples=examples if use_few else None,
            repair=repair,
            verify=verify,
            first_pass=cache.get(use_few),
        )
        cache.setdefault(use_few, gens)
        results.extend(res)
    return results
