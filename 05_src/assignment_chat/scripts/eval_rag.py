from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

# Allow "python scripts/eval_rag.py" to import project modules.
PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from config import KB_CSV_PATH  # noqa: E402
from services.rag_service import query_issue_kb  # noqa: E402


DEFAULT_TEST_CASES: list[dict[str, str]] = [
    {
        "query": "AHU commissioning issue: OA damper command changes but OA flow stays low",
        "expected_top1": "MEP-001",
    },
    {
        "query": "AHU startup static pressure swings and supply fan VFD speed oscillates",
        "expected_top1": "MEP-006",
    },
    {
        "query": "VAV perimeter zone overheats, reheat command is 0% but leaving air is still warm",
        "expected_top1": "MEP-012",
    },
    {
        "query": "CHW pump room differential pressure setpoint swings and pumps keep ramping",
        "expected_top1": "MEP-017",
    },
    {
        "query": "BAS startup has alarm flooding and nuisance alarms everywhere",
        "expected_top1": "MEP-024",
    },
    {
        "query": "Cleanroom pressure cascade is unstable with frequent differential pressure alarms",
        "expected_top1": "MEP-028",
    },
]


def _load_kb(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8").fillna("")
    if "id" not in df.columns:
        raise ValueError("CSV must include an 'id' column.")
    return df


def _parse_citation_id(citation: str) -> str:
    if ":" in citation:
        return citation.split(":", 1)[0].strip()
    return citation.strip()


def _contains_all(answer: str, snippets: list[str]) -> bool:
    return all((not s) or (s in answer) for s in snippets)


def evaluate(
    kb_df: pd.DataFrame,
    test_cases: list[dict[str, str]],
    top_k: int,
) -> dict[str, Any]:
    kb_ids = set(kb_df["id"].astype(str).tolist())
    rows_by_id = {
        str(row["id"]): row
        for _, row in kb_df.iterrows()
    }

    per_case: list[dict[str, Any]] = []
    top1_hits = 0
    topk_hits = 0
    grounded_hits = 0
    citation_valid_hits = 0

    for case in test_cases:
        query = case["query"]
        expected_top1 = case["expected_top1"]

        result = query_issue_kb(query, top_k=top_k)
        citation_ids = [_parse_citation_id(c) for c in result.citations]
        top1_id = citation_ids[0] if citation_ids else ""

        top1_ok = top1_id == expected_top1
        topk_ok = expected_top1 in citation_ids
        citations_valid = all(cid in kb_ids for cid in citation_ids)

        grounding_ok = False
        if top1_id in rows_by_id:
            row = rows_by_id[top1_id]
            grounding_ok = _contains_all(
                result.answer,
                [
                    str(row.get("likely_root_causes", "")),
                    str(row.get("field_checks", "")),
                    str(row.get("recommended_actions", "")),
                ],
            )

        top1_hits += int(top1_ok)
        topk_hits += int(topk_ok)
        grounded_hits += int(grounding_ok)
        citation_valid_hits += int(citations_valid)

        per_case.append(
            {
                "query": query,
                "expected_top1": expected_top1,
                "actual_top1": top1_id,
                "top1_ok": top1_ok,
                "topk_ok": topk_ok,
                "grounding_ok": grounding_ok,
                "citations_valid": citations_valid,
                "citations": result.citations,
            }
        )

    n = len(test_cases) or 1
    summary = {
        "num_cases": len(test_cases),
        "top1_accuracy": round(top1_hits / n, 4),
        "topk_recall": round(topk_hits / n, 4),
        "grounded_answer_rate": round(grounded_hits / n, 4),
        "citation_valid_rate": round(citation_valid_hits / n, 4),
    }
    return {"summary": summary, "cases": per_case}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Service 2 (RAG) retrieval and grounding quality."
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=KB_CSV_PATH,
        help=f"Path to the KB CSV file (default: {KB_CSV_PATH})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-k retrieval count for each test case (default: 5)",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to save full JSON report.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    kb_df = _load_kb(args.csv_path)
    report = evaluate(kb_df=kb_df, test_cases=DEFAULT_TEST_CASES, top_k=args.top_k)

    print("RAG Evaluation Summary")
    for k, v in report["summary"].items():
        print(f"- {k}: {v}")

    print("\nPer-case (expected -> actual):")
    for row in report["cases"]:
        print(
            f"- {row['expected_top1']} -> {row['actual_top1']} "
            f"(top1={row['top1_ok']}, topk={row['topk_ok']}, "
            f"grounded={row['grounding_ok']}, citations_valid={row['citations_valid']})"
        )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nSaved report to: {args.json_out}")


if __name__ == "__main__":
    main()

