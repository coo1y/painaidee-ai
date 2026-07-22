"""Retrieval evaluation: compare multiple retrieval approaches.

Approaches compared:
  - vector   : dense embeddings only
  - bm25     : lexical only
  - hybrid   : dense + BM25 (Reciprocal Rank Fusion)
  - hybrid+rr: hybrid then LLM re-ranking

Metrics: Recall@k and MRR@k on a ground-truth set. If ``eval/ground_truth.json``
exists it is used; otherwise a proxy set is auto-generated (each document's name
should retrieve that document). With the small sample data this is a smoke test;
swap in the full dataset + curated queries for meaningful numbers.

Run:  python -m eval.retrieval_eval
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.retrieval import (
    _all_docs,
    bm25_search,
    dense_search,
    hybrid_search,
    rerank,
)

K = 3
GT_PATH = Path(__file__).resolve().parent / "ground_truth.json"


def build_ground_truth() -> list[dict]:
    """Load curated ground truth or auto-generate a proxy set from names."""
    if GT_PATH.exists():
        return json.loads(GT_PATH.read_text(encoding="utf-8"))

    gt = []
    for coll in (config.ATTRACTIONS_COLLECTION, config.EVENTS_COLLECTION):
        ids, _docs, metas = _all_docs(coll)
        for _id, meta in zip(ids, metas):
            name = meta.get("name_en") or meta.get("name") or meta.get("name_th")
            if name:
                gt.append({"collection": coll, "query": name, "relevant": [_id]})
    return gt


def _ids(hits) -> list[str]:
    return [h["id"] for h in hits]


def evaluate(method: str, gt: list[dict]) -> dict:
    recalls, rrs = [], []
    for case in gt:
        coll, query, relevant = case["collection"], case["query"], set(case["relevant"])
        if method == "vector":
            hits = dense_search(coll, query, k=K)
        elif method == "bm25":
            hits = bm25_search(coll, query, k=K)
        elif method == "hybrid":
            hits = hybrid_search(coll, query, k=K)
        elif method == "hybrid+rr":
            cands = hybrid_search(coll, query, k=config.CANDIDATE_K)
            hits = rerank(query, cands, top_k=K)
        else:
            raise ValueError(method)

        ranked = _ids(hits)[:K]
        recalls.append(1.0 if relevant & set(ranked) else 0.0)
        rr = 0.0
        for rank, _id in enumerate(ranked, 1):
            if _id in relevant:
                rr = 1.0 / rank
                break
        rrs.append(rr)

    n = max(len(gt), 1)
    return {"recall@k": sum(recalls) / n, "mrr@k": sum(rrs) / n, "n": len(gt)}


def main() -> None:
    gt = build_ground_truth()
    print(f"Ground-truth cases: {len(gt)} | k={K} | embeddings={config.EMBEDDING_BACKEND}\n")

    methods = ["vector", "bm25", "hybrid", "hybrid+rr"]
    results = {m: evaluate(m, gt) for m in methods}

    print(f"{'method':<12}{'recall@k':>10}{'mrr@k':>10}")
    print("-" * 32)
    for m in methods:
        r = results[m]
        print(f"{m:<12}{r['recall@k']:>10.3f}{r['mrr@k']:>10.3f}")

    best = max(results, key=lambda m: (results[m]["recall@k"], results[m]["mrr@k"]))
    print(f"\nBest approach: {best}")
    print("(The app uses hybrid + re-ranking; see README for rationale.)")


if __name__ == "__main__":
    main()
