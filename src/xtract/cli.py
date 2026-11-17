"""Command line: ``xtract download | run | judge | report``."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from . import data
from .judge import Item, agreement, build_items, run_judge
from .metrics import score
from .pipeline import STRATEGIES, Result, run_all


def slug(model_id: str) -> str:
    return model_id.split("/")[-1].lower()


def _backend(model_id: str, batch_size: int):
    from .llm import HFBackend

    return HFBackend(model_id, batch_size=batch_size)


def cmd_download(args: argparse.Namespace) -> int:
    for split in ("train", "test"):
        print(data.download(split, Path(args.data_dir)))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    ddir = Path(args.data_dir)
    test = data.sample(data.load_docs("test", ddir), args.n, seed=args.seed)
    examples = data.few_shot_examples(data.load_docs("train", ddir), k=3, seed=args.seed)
    backend = _backend(args.model, args.batch_size)
    results = run_all(backend, test, examples, args.strategies)
    out = Path(args.out) / slug(args.model)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "results.jsonl").open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(asdict(r)) + "\n")
    table = score(results, test)
    table.insert(0, "model", slug(args.model))
    table.to_csv(out / "scores.csv", index=False)
    print(
        table[
            [
                "strategy",
                "micro_f1",
                "valid_first_pass",
                "valid_final",
                "hallucinated_per_100_docs",
                "ms_per_doc_p50",
            ]
        ].to_string(index=False)
    )
    return 0


def _load_results(path: Path) -> list[Result]:
    return [Result(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines()]


def cmd_judge(args: argparse.Namespace) -> int:
    ddir = Path(args.data_dir)
    docs = data.load_docs("test", ddir)
    results = [r for r in _load_results(Path(args.results)) if r.strategy == args.strategy]
    items = build_items(results, docs, args.n, seed=args.seed)
    backend = _backend(args.judge_model, args.batch_size)
    human = [i.human for i in items]
    greedy = run_judge(backend, items)
    s1 = run_judge(backend, items, temperature=args.temperature)
    s2 = run_judge(backend, items, temperature=args.temperature)
    report = {
        "judge_model": slug(args.judge_model),
        "judged_results": args.results,
        "strategy": args.strategy,
        "greedy": agreement(human, greedy),
        f"sampled_t{args.temperature}_run1": agreement(human, s1),
        f"sampled_t{args.temperature}_run2": agreement(human, s2),
        "sampled_run_to_run_agreement": float(pd.Series(s1).eq(pd.Series(s2)).mean()),
        "greedy_vs_sampled_agreement": float(pd.Series(greedy).eq(pd.Series(s1)).mean()),
        "per_field_greedy_accuracy": {
            f: float(
                pd.Series(
                    [g == h for g, h, i in zip(greedy, human, items, strict=True) if i.field == f]
                ).mean()
            )
            for f in data.FIELDS
            if any(i.field == f for i in items)
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    with out.with_suffix(".items.jsonl").open("w", encoding="utf-8") as fh:
        for i, g in zip(items, greedy, strict=True):
            fh.write(json.dumps({**asdict(i), "judge_greedy": g}) + "\n")
    print(json.dumps(report["greedy"], indent=2))
    return 0


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append(
            "| "
            + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r.tolist())
            + " |"
        )
    return "\n".join(lines)


def cmd_report(args: argparse.Namespace) -> int:
    root = Path(args.reports)
    scores = pd.concat([pd.read_csv(p) for p in sorted(root.glob("*/scores.csv"))])
    scores.to_csv(root / "all_scores.csv", index=False)
    main_cols = [
        "model",
        "strategy",
        "precision",
        "recall",
        "micro_f1",
        "valid_first_pass",
        "valid_final",
        "hallucinated_per_100_docs",
        "calls_per_doc",
        "prompt_tokens_per_doc",
        "completion_tokens_per_doc",
        "ms_per_doc_p50",
    ]
    field_cols = ["model", "strategy"] + [f"f1_{f}" for f in data.FIELDS]
    lines = [
        "# Results",
        "",
        "## Extraction quality and cost",
        "",
        _md(scores[main_cols]),
        "",
        "## F1 per entity type",
        "",
        _md(scores[field_cols]),
        "",
    ]
    judges = sorted(root.glob("judge_*.json"))
    if judges:
        rows = []
        for p in judges:
            j = json.loads(p.read_text())
            g = j["greedy"]
            rows.append(
                {
                    "judge": j["judge_model"],
                    "judged": Path(j["judged_results"]).parent.name,
                    "items": g["items"],
                    "accuracy_vs_human": g["accuracy_vs_human"],
                    "cohen_kappa": g["cohen_kappa"],
                    "judge_yes_rate": g["judge_yes_rate"],
                    "TPR": g["true_positive_rate"],
                    "TNR": g["true_negative_rate"],
                    "sampled_run_to_run_agreement": j["sampled_run_to_run_agreement"],
                }
            )
        lines += ["## LLM-as-judge vs human annotation", "", _md(pd.DataFrame(rows)), ""]
    (root / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((root / "results.md").read_text())
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="xtract", description=__doc__)
    p.add_argument("--data-dir", default="data")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download", help="fetch CoNLL-2003 parquet files (not redistributed)")
    d.set_defaults(func=cmd_download)

    r = sub.add_parser("run", help="run extraction strategies with a local HF model")
    r.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    r.add_argument("--n", type=int, default=300)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--batch-size", type=int, default=16)
    r.add_argument("--strategies", nargs="*", default=list(STRATEGIES))
    r.add_argument("--out", default="reports")
    r.set_defaults(func=cmd_run)

    j = sub.add_parser("judge", help="LLM-as-judge agreement with human annotations")
    j.add_argument("--judge-model", default="Qwen/Qwen2.5-3B-Instruct")
    j.add_argument("--results", required=True)
    j.add_argument("--strategy", default="zero-shot")
    j.add_argument("--n", type=int, default=300)
    j.add_argument("--seed", type=int, default=0)
    j.add_argument("--temperature", type=float, default=0.7)
    j.add_argument("--batch-size", type=int, default=16)
    j.add_argument("--out", required=True)
    j.set_defaults(func=cmd_judge)

    rep = sub.add_parser("report", help="merge all runs into reports/results.md")
    rep.add_argument("--reports", default="reports")
    rep.set_defaults(func=cmd_report)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["Item", "main"]
