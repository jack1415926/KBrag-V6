"""Run one end-to-end four-path retrieval smoke case and save safe evidence."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from config_runtime import apply_default_env
from retrieval_engine import RetrievalEngine
from visual_retrieval import (
    ROOT,
    VisualRetrievalEngine,
    analysis_query_text,
    analyze_query_images,
    file_to_data_url,
)


GOLD = ROOT / "evaluation" / "visual_relevance_gold_30_derived.jsonl"
OUTPUT = ROOT / "validation_outputs" / "visual_retrieval_smoke.json"


def rank_of(values: list[str], relevant: list[str]) -> int | None:
    """Return the first one-based relevant rank."""
    expected = set(relevant)
    return next((rank for rank, value in enumerate(values, 1) if value in expected), None)


def main() -> None:
    """Exercise Vision text extraction, both visual queries, fusion, and VL rerank."""
    apply_default_env()
    os.environ["VISUAL_RETRIEVAL_ENABLED"] = "1"
    os.environ["VISUAL_RETRIEVAL_STRICT"] = "1"
    case = json.loads(GOLD.read_text(encoding="utf-8").splitlines()[0])
    image = file_to_data_url(ROOT / case["query_image"])
    started = time.time()
    analysis = analyze_query_images([image])
    query = analysis_query_text(case["question"], analysis)
    visual = VisualRetrievalEngine()
    text_hits = visual.search_text(query, top_n=5)
    image_hits = visual.search_image(image, top_n=5)
    engine = RetrievalEngine()
    engine.ensure_index()
    results, _ = engine.search_manual(
        [], semantic_query=query, original_query=query, top_k=5,
        query_images=[image], visual_query_text=query,
    )
    fused_sections = [f"{result.product}|{result.source['parent_section_id']}" for result in results]
    record = {
        "case_id": case["case_id"],
        "expected_product": case["expected_product"],
        "relevant_sections": case["relevant_sections"],
        "relevant_images": case["relevant_images"],
        "vision_analysis": analysis,
        "text_to_image": {
            "images": [hit.image_id for hit in text_hits],
            "image_rank": rank_of([hit.image_id for hit in text_hits], case["relevant_images"]),
        },
        "image_to_image": {
            "images": [hit.image_id for hit in image_hits],
            "image_rank": rank_of([hit.image_id for hit in image_hits], case["relevant_images"]),
        },
        "fusion": {
            "sections": fused_sections,
            "section_rank": rank_of(fused_sections, case["relevant_sections"]),
            "channels": [result.source.get("retrieval_channels", []) for result in results],
            "vl_rerank_fallback": any(result.source.get("vl_rerank_fallback") for result in results),
        },
        "elapsed": round(time.time() - started, 3),
    }
    OUTPUT.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "visual_smoke=OK "
        f"case={case['case_id']} text_image_rank={record['text_to_image']['image_rank']} "
        f"image_image_rank={record['image_to_image']['image_rank']} "
        f"fusion_section_rank={record['fusion']['section_rank']} "
        f"fallback={record['fusion']['vl_rerank_fallback']} elapsed={record['elapsed']}"
    )


if __name__ == "__main__":
    main()
