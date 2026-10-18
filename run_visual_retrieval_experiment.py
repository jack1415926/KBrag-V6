"""Run the fixed four-way visual retrieval comparison.

This script calls the configured Vision, embedding, and rerank services. It
requires a completed visual index and explicit authorization for data egress.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

import requests

from product_router import ProductRouter
from retrieval_engine import RetrievalEngine, SearchResult
from visual_retrieval import (
    ROOT,
    VisualRetrievalEngine,
    analysis_query_text,
    analyze_query_images,
    file_to_data_url,
    rrf_rank,
)


DERIVED_GOLD = ROOT / "evaluation" / "visual_relevance_gold_30_derived.jsonl"
EXTERNAL_GOLD = ROOT / "evaluation" / "visual_relevance_gold_10_external.jsonl"
MODES = (
    "current_text",
    "vision_to_text",
    "visual_only",
    "fusion",
    "fusion_merged",
    "fusion_text_serial",
)
_PROFILE_CONTEXT: ContextVar[dict[str, Any]] = ContextVar("visual_latency_profile", default={})


def load_jsonl(path: Path) -> list[dict]:
    """Load an optional JSONL file."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def percentile_nearest_rank(values: list[float], percentile: float) -> float:
    """Return a nearest-rank percentile."""
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def section_key(result: SearchResult) -> str:
    """Return the common cross-channel parent-section key."""
    return f"{result.product}|{result.source['parent_section_id']}"


def text_search(engine: RetrievalEngine, router: ProductRouter, query: str) -> tuple[list[SearchResult], dict]:
    """Run the existing routed text retrieval baseline."""
    route = router.route(query)
    results, _filtered = engine._search_text_manual(
        [], semantic_query=query, original_query=query, top_k=5, products=route.products or None
    )
    return results, {"products": route.products, "reason": route.reason, "confidence": route.confidence}


def visual_only_search(visual: VisualRetrievalEngine, question: str, images: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Fuse text-to-image and image-to-image rankings without text chunks."""
    ranked_lists: list[list[str]] = []
    image_lists: list[list[str]] = []
    products: list[str] = []
    if question:
        hits = visual.search_text(question)
        ranked_lists.append(list(dict.fromkeys(key for hit in hits for key in hit.section_keys)))
        image_lists.append([hit.image_id for hit in hits])
        products.extend(hit.product for hit in hits)
    for image in images:
        hits = visual.search_image(image)
        ranked_lists.append(list(dict.fromkeys(key for hit in hits for key in hit.section_keys)))
        image_lists.append([hit.image_id for hit in hits])
        products.extend(hit.product for hit in hits)
    section_keys, _scores = rrf_rank(ranked_lists, top_n=5)
    image_ids, _image_scores = rrf_rank(image_lists, top_n=5)
    return section_keys, image_ids, list(dict.fromkeys(products))[:5]


def _merged_rerank_query(original: str, expanded: str) -> str:
    supplement = expanded[len(original):].strip() if expanded.startswith(original) else expanded.strip()
    return f"用户原始问题：{original.strip()}\n图像分析补充：{supplement}" if supplement else original.strip()


def merged_fusion_search(
    engine: RetrievalEngine,
    visual: VisualRetrievalEngine,
    router: ProductRouter,
    original_query: str,
    expanded_query: str,
    images: list[str],
) -> tuple[list[SearchResult], dict]:
    """Merge four raw text rankings, then run one text rerank before visual fusion."""
    expanded_route = router.route(expanded_query)
    original_route = router.route(original_query)
    route = {
        "products": expanded_route.products,
        "reason": expanded_route.reason,
        "confidence": expanded_route.confidence,
    }
    visual_products = (
        expanded_route.products
        if expanded_route.reason in {"explicit_product_name", "explicit_product_nickname"}
        else None
    )

    def recall(query: str, products: list[str] | None) -> list[list[int]]:
        rankings, _rerank_query = engine._collect_text_recall_rankings(
            [], semantic_query=query, original_query=query, products=products
        )
        return rankings

    def collect_visual_hits() -> tuple[list, list[list]]:
        return (
            visual.search_text(expanded_query, products=visual_products),
            [visual.search_image(image, products=visual_products) for image in images],
        )

    with ThreadPoolExecutor(max_workers=3) as pool:
        original_future = pool.submit(copy_context().run, recall, original_query, original_route.products or None)
        expanded_future = pool.submit(copy_context().run, recall, expanded_query, expanded_route.products or None)
        visual_future = pool.submit(copy_context().run, collect_visual_hits)
        ranked_lists = original_future.result() + expanded_future.result()
        visual_text_hits, visual_image_hits = visual_future.result()

    candidates = engine._rrf_merge(ranked_lists, top_n=engine.rerank_candidate_pool)
    merged_query = _merged_rerank_query(original_query, expanded_query)
    _PROFILE_CONTEXT.get()["merged_query"] = merged_query
    reranked = engine._rerank_candidates(merged_query, candidates, top_n=min(12, len(candidates)))
    seen: set[str] = set()
    final_ids: list[int] = []
    for doc_id in reranked:
        key = engine._result_dedup_key(doc_id)
        if key in seen:
            continue
        seen.add(key)
        final_ids.append(doc_id)
        if len(final_ids) == 10:
            break
    text_results = engine._build_results(final_ids)
    results = engine._merge_visual_results(
        text_results,
        query=expanded_query,
        images=images,
        products=visual_products,
        top_k=5,
        visual_text_hits=visual_text_hits,
        visual_image_hits=visual_image_hits,
    )
    return results, route


def metrics(section_keys: list[str], image_ids: list[str], products: list[str], gold: dict) -> dict:
    """Calculate per-case product, section, and image retrieval metrics."""
    relevant_sections = set(gold["relevant_sections"])
    relevant_images = set(gold["relevant_images"])
    expected_product = gold["expected_product"]
    section_rank = next((i for i, key in enumerate(section_keys, 1) if key in relevant_sections), None)
    image_rank = next((i for i, key in enumerate(image_ids, 1) if key in relevant_images), None)
    product_rank = next((i for i, product in enumerate(products, 1) if product == expected_product), None)
    return {
        "product_r1": int(product_rank == 1),
        "product_r5": int(product_rank is not None and product_rank <= 5),
        "section_r1": int(section_rank == 1),
        "section_r5": int(section_rank is not None and section_rank <= 5),
        "image_r1": int(image_rank == 1),
        "image_r5": int(image_rank is not None and image_rank <= 5),
        "section_rr": 0.0 if section_rank is None else 1.0 / section_rank,
        "image_rr": 0.0 if image_rank is None else 1.0 / image_rank,
        "wrong_image": int(bool(image_ids) and not relevant_images.intersection(image_ids)),
    }


def append_jsonl(path: Path, record: dict) -> None:
    """Append one durable experiment record."""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict) -> None:
    """Atomically replace one optional frontend progress file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


class LatencyProfiler:
    """Durably record call-level timings without changing retrieval behavior."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def call(
        self,
        stage: str,
        function: Callable,
        *args,
        payload_items: int = 0,
        payload_bytes: int = 0,
        details: dict[str, Any] | None = None,
        result_metadata: Callable[[Any], dict[str, Any]] | None = None,
        **kwargs,
    ):
        started_at = time.time()
        started = time.perf_counter()
        error_type = None
        dynamic_details: dict[str, Any] = {}
        try:
            result = function(*args, **kwargs)
            if result_metadata is not None:
                dynamic_details = result_metadata(result)
            return result
        except Exception as exc:
            error_type = type(exc).__name__
            raise
        finally:
            ended_at = time.time()
            context = _PROFILE_CONTEXT.get()
            record = {
                "repeat": context.get("repeat"),
                "mode": context.get("mode"),
                "case_id": context.get("case_id"),
                "stage": stage,
                "started_at": round(started_at, 6),
                "ended_at": round(ended_at, 6),
                "elapsed": round(time.perf_counter() - started, 6),
                "payload_items": payload_items,
                "payload_bytes": payload_bytes,
                "thread_id": threading.get_ident(),
                "error_type": error_type,
            }
            record.update(details or {})
            record.update(dynamic_details)
            with self._lock:
                append_jsonl(self.path, record)


def _payload_bytes(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, default=str).encode("utf-8"))


def _text_stage(operation: str, query: str) -> str:
    context = _PROFILE_CONTEXT.get()
    if query == context.get("merged_query"):
        return f"text_merged_{operation}"
    if query == context.get("original_question"):
        return f"text_original_{operation}"
    if query == context.get("expanded_query"):
        return f"text_expanded_{operation}"
    return f"text_other_{operation}"


def install_latency_profile(
    profiler: LatencyProfiler,
    engine: RetrievalEngine,
    visual: VisualRetrievalEngine,
    router: ProductRouter,
) -> None:
    """Wrap existing clients in-place so the experiment records a call timeline."""
    text_embed = engine.client.embed_texts
    text_rerank = engine.rerank_client.rerank
    visual_embed = visual.embedding_client.embed_inputs
    visual_rerank = visual.rerank_client.rerank
    route = router.route
    http_post = requests.Session.post

    def profiled_text_embed(texts, model, timing_trace=None):
        query = texts[0] if len(texts) == 1 else ""
        call_args = (texts, model) if timing_trace is None else (texts, model, timing_trace)
        return profiler.call(
            _text_stage("embedding", query),
            text_embed,
            *call_args,
            payload_items=len(texts),
            payload_bytes=_payload_bytes(texts),
        )

    def profiled_text_rerank(query, documents, top_n):
        return profiler.call(
            _text_stage("rerank", query),
            text_rerank,
            query,
            documents,
            top_n,
            payload_items=len(documents),
            payload_bytes=_payload_bytes({"query": query, "documents": documents}),
        )

    def profiled_visual_embed(inputs, model, dimension=4096):
        if inputs and all(isinstance(item, str) for item in inputs):
            stage = "vl_text_embedding"
        elif inputs and all(isinstance(item, dict) and "image" in item for item in inputs):
            stage = "vl_image_embedding"
        else:
            stage = "vl_mixed_embedding"
        return profiler.call(
            stage,
            visual_embed,
            inputs,
            model,
            dimension,
            payload_items=len(inputs),
            payload_bytes=_payload_bytes(inputs),
        )

    def profiled_visual_rerank(query, documents, top_n, model=None):
        call_args = (query, documents, top_n) if model is None else (query, documents, top_n, model)
        return profiler.call(
            "vl_rerank", visual_rerank, *call_args,
            payload_items=len(documents), payload_bytes=_payload_bytes({"query": query, "documents": documents}),
        )

    def profiled_route(query):
        return profiler.call(
            _text_stage("product_route", query),
            route,
            query,
            payload_items=1,
            payload_bytes=len(query.encode("utf-8")),
        )

    def profiled_http_post(session, url, *args, **kwargs):
        body = kwargs.get("json") if isinstance(kwargs.get("json"), dict) else {}
        model = str(body.get("model") or "")
        path = urlsplit(url).path
        if "rerank" in path or "reranking" in path:
            stage = "vl_rerank_http" if "VL" in model.upper() else "text_rerank_http"
        elif "embedding" in path:
            stage = "vl_embedding_http" if "VL" in model.upper() else "text_embedding_http"
        else:
            stage = "other_http"
        return profiler.call(
            stage,
            http_post,
            session,
            url,
            *args,
            payload_items=len(body.get("documents") or body.get("input") or []),
            payload_bytes=_payload_bytes(body),
            details={"endpoint": path, "model": model},
            result_metadata=lambda response: {"status_code": getattr(response, "status_code", None)},
            **kwargs,
        )

    engine.client.embed_texts = profiled_text_embed
    engine.rerank_client.rerank = profiled_text_rerank
    visual.embedding_client.embed_inputs = profiled_visual_embed
    visual.rerank_client.rerank = profiled_visual_rerank
    engine.visual_engine = visual
    router.route = profiled_route
    requests.Session.post = profiled_http_post


def install_text_remote_limit(engine: RetrievalEngine) -> None:
    """Serialize text remote calls only for the fusion_text_serial experiment mode."""
    semaphore = threading.Semaphore(1)
    embed = engine.client.embed_texts
    rerank = engine.rerank_client.rerank

    def limited_embed(*args, **kwargs):
        if _PROFILE_CONTEXT.get().get("mode") != "fusion_text_serial":
            return embed(*args, **kwargs)
        with semaphore:
            return embed(*args, **kwargs)

    def limited_rerank(*args, **kwargs):
        if _PROFILE_CONTEXT.get().get("mode") != "fusion_text_serial":
            return rerank(*args, **kwargs)
        with semaphore:
            return rerank(*args, **kwargs)

    engine.client.embed_texts = limited_embed
    engine.rerank_client.rerank = limited_rerank


def write_latency_summary(events_path: Path, summary_path: Path) -> None:
    """Aggregate call-level timing records after a completed experiment."""
    events = load_jsonl(events_path)
    summary: dict[str, dict] = {}
    for stage in sorted({event["stage"] for event in events}):
        selected = [event for event in events if event["stage"] == stage]
        elapsed = [event["elapsed"] for event in selected]
        payloads = [event["payload_bytes"] for event in selected]
        slowest = max(selected, key=lambda event: event["elapsed"])
        summary[stage] = {
            "calls": len(selected),
            "errors": sum(bool(event["error_type"]) for event in selected),
            "latency_median": statistics.median(elapsed),
            "latency_p95": percentile_nearest_rank(elapsed, 0.95),
            "latency_max": max(elapsed),
            "payload_bytes_median": statistics.median(payloads),
            "payload_bytes_max": max(payloads),
            "slowest_call": {
                "repeat": slowest["repeat"],
                "mode": slowest["mode"],
                "case_id": slowest["case_id"],
                "elapsed": slowest["elapsed"],
            },
        }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    """Run all four configurations in interleaved repeated order."""
    parser = argparse.ArgumentParser(description="运行视觉检索四组对照")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--modes", default=",".join(MODES), help="comma-separated subset of modes")
    parser.add_argument("--profile-latency", action="store_true", help="record per-call timing timeline")
    parser.add_argument("--precompute-vision", action="store_true", help="freeze Vision analysis before timed modes")
    parser.add_argument("--progress-path", type=Path, help="optional JSON progress file for the local web UI")
    args = parser.parse_args()
    selected_modes = tuple(item.strip() for item in args.modes.split(",") if item.strip())
    if not selected_modes or any(mode not in MODES for mode in selected_modes):
        raise ValueError(f"modes must be selected from {MODES}")
    gold = load_jsonl(DERIVED_GOLD) + load_jsonl(EXTERNAL_GOLD)
    if len(gold) != 40:
        raise RuntimeError(f"visual gold is incomplete: expected 40, got {len(gold)}")
    progress_current = 0
    progress_errors = 0
    progress_total = args.repeats * len(gold) * (len(selected_modes) + int(args.precompute_vision))

    def save_progress(status: str, *, mode: str | None = None, repeat: int = 0, case_id: str | None = None) -> None:
        if args.progress_path is None:
            return
        write_json(args.progress_path, {
            "status": status,
            "current": progress_current,
            "total": progress_total,
            "mode": mode,
            "repeat": repeat,
            "case_id": case_id,
            "errors": progress_errors,
            "out_dir": str(args.out_dir),
            "updated_at": time.time(),
        })

    save_progress("loading_index")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    runs_path = args.out_dir / "runs.jsonl"
    summary_path = args.out_dir / "summary.json"
    latency_events_path = args.out_dir / "latency_events.jsonl"
    latency_summary_path = args.out_dir / "latency_summary.json"
    output_paths = (runs_path, summary_path, latency_events_path, latency_summary_path)
    if any(path.exists() for path in output_paths) and not args.reset:
        raise FileExistsError("output already exists; use a new directory or --reset")
    for path in output_paths:
        if args.reset and path.exists():
            path.unlink()

    os.environ["VISUAL_RETRIEVAL_ENABLED"] = "1"
    os.environ["VISUAL_RETRIEVAL_STRICT"] = "1"
    engine = RetrievalEngine()
    engine.ensure_index()
    router = ProductRouter(engine.catalog, engine=engine)
    visual = VisualRetrievalEngine()
    visual.ensure_index()
    profiler = LatencyProfiler(latency_events_path) if args.profile_latency else None
    if profiler is not None:
        install_latency_profile(profiler, engine, visual, router)
    install_text_remote_limit(engine)
    analysis_cache: dict[tuple[int, str], dict] = {}

    def get_analysis(repeat: int, case_id: str, images: list[str]) -> dict:
        """Reuse one stochastic Vision analysis across compared modes."""
        key = (repeat, case_id)
        if key not in analysis_cache:
            if profiler is None:
                analysis_cache[key] = analyze_query_images(images)
            else:
                analysis_cache[key] = profiler.call(
                    "vision_analysis",
                    analyze_query_images,
                    images,
                    payload_items=len(images),
                    payload_bytes=_payload_bytes(images),
                )
        return analysis_cache[key]

    if args.precompute_vision:
        for repeat in range(args.repeats):
            cases = list(reversed(gold)) if repeat % 2 else list(gold)
            for case in cases:
                context = {
                    "repeat": repeat,
                    "mode": "vision_precompute",
                    "case_id": case["case_id"],
                    "original_question": case["question"],
                    "expanded_query": None,
                    "merged_query": None,
                }
                token = _PROFILE_CONTEXT.set(context)
                try:
                    image = file_to_data_url(ROOT / case["query_image"])
                    get_analysis(repeat, case["case_id"], [image])
                finally:
                    _PROFILE_CONTEXT.reset(token)
                progress_current += 1
                save_progress(
                    "precomputing_vision",
                    mode="vision_precompute",
                    repeat=repeat + 1,
                    case_id=case["case_id"],
                )
                print(f"vision_precompute repeat={repeat + 1} case={case['case_id']}")

    for repeat in range(args.repeats):
        modes = list(selected_modes[repeat % len(selected_modes):] + selected_modes[:repeat % len(selected_modes)])
        cases = list(reversed(gold)) if repeat % 2 else list(gold)
        for mode in modes:
            for case in cases:
                save_progress("running", mode=mode, repeat=repeat + 1, case_id=case["case_id"])
                started = time.time()
                profile_context = {
                    "repeat": repeat,
                    "mode": mode,
                    "case_id": case["case_id"],
                    "original_question": case["question"],
                    "expanded_query": None,
                    "merged_query": None,
                }
                profile_token = _PROFILE_CONTEXT.set(profile_context)
                error = None
                section_keys: list[str] = []
                image_ids: list[str] = []
                products: list[str] = []
                route: dict = {}
                fallback = False
                try:
                    image = file_to_data_url(ROOT / case["query_image"])
                    images = [image]
                    if mode == "current_text":
                        results, route = text_search(engine, router, case["question"])
                        section_keys = [section_key(result) for result in results]
                        image_ids = list(dict.fromkeys(pic for result in results for pic in result.pics))[:5]
                        products = list(dict.fromkeys(result.product for result in results))[:5]
                    elif mode == "vision_to_text":
                        analysis = get_analysis(repeat, case["case_id"], images)
                        query = analysis_query_text(case["question"], analysis)
                        profile_context["expanded_query"] = query
                        results, route = text_search(engine, router, query)
                        section_keys = [section_key(result) for result in results]
                        image_ids = list(dict.fromkeys(pic for result in results for pic in result.pics))[:5]
                        products = list(dict.fromkeys(result.product for result in results))[:5]
                    elif mode == "visual_only":
                        section_keys, image_ids, products = visual_only_search(visual, case["question"], images)
                    elif mode in {"fusion", "fusion_text_serial"}:
                        analysis = get_analysis(repeat, case["case_id"], images)
                        query = analysis_query_text(case["question"], analysis)
                        profile_context["expanded_query"] = query
                        route_decision = router.route(query)
                        original_route = router.route(case["question"])
                        route = {"products": route_decision.products, "reason": route_decision.reason, "confidence": route_decision.confidence}
                        visual_products = route_decision.products if route_decision.reason in {"explicit_product_name", "explicit_product_nickname"} else None
                        results, _filtered = engine.search_manual(
                            [], semantic_query=query, original_query=query, top_k=5,
                            products=route_decision.products or None, query_images=images,
                            visual_query_text=query, visual_products=visual_products,
                            original_text_query=case["question"],
                            original_text_products=original_route.products or None,
                        )
                        section_keys = [section_key(result) for result in results]
                        image_ids = list(dict.fromkeys(pic for result in results for pic in result.pics))[:5]
                        products = list(dict.fromkeys(result.product for result in results))[:5]
                        fallback = any(result.source.get("vl_rerank_fallback") for result in results)
                    else:
                        analysis = get_analysis(repeat, case["case_id"], images)
                        query = analysis_query_text(case["question"], analysis)
                        profile_context["expanded_query"] = query
                        results, route = merged_fusion_search(
                            engine, visual, router, case["question"], query, images
                        )
                        section_keys = [section_key(result) for result in results]
                        image_ids = list(dict.fromkeys(pic for result in results for pic in result.pics))[:5]
                        products = list(dict.fromkeys(result.product for result in results))[:5]
                        fallback = any(result.source.get("vl_rerank_fallback") for result in results)
                except Exception as exc:
                    error = str(exc)
                    progress_errors += 1
                finally:
                    _PROFILE_CONTEXT.reset(profile_token)
                record = {
                    "repeat": repeat,
                    "mode": mode,
                    "case_id": case["case_id"],
                    "source_group": case.get("source_group", "derived"),
                    "source_type": case["source_type"],
                    "category": case["category"],
                    "section_keys": section_keys,
                    "image_ids": image_ids,
                    "products": products,
                    "route": route,
                    "elapsed": round(time.time() - started, 3),
                    "error": error,
                    "fallback": fallback,
                    **metrics(section_keys, image_ids, products, case),
                }
                append_jsonl(runs_path, record)
                progress_current += 1
                save_progress("running", mode=mode, repeat=repeat + 1, case_id=case["case_id"])
                print(f"visual_eval repeat={repeat + 1} mode={mode} case={case['case_id']} error={bool(error)}")

    rows = load_jsonl(runs_path)
    summary: dict[str, dict] = {}
    for mode in selected_modes:
        selected = [row for row in rows if row["mode"] == mode]
        valid = [row for row in selected if not row["error"]]
        entry = {
            "records": len(selected),
            "valid": len(valid),
            "errors": len(selected) - len(valid),
            "fallbacks": sum(bool(row.get("fallback")) for row in valid),
        }
        for field in ("product_r1", "product_r5", "section_r1", "section_r5", "image_r1", "image_r5", "section_rr", "image_rr", "wrong_image"):
            entry[field] = 0.0 if not valid else sum(row[field] for row in valid) / len(valid)
        elapsed = [row["elapsed"] for row in valid]
        entry["latency_median"] = 0.0 if not elapsed else statistics.median(elapsed)
        entry["latency_p95"] = percentile_nearest_rank(elapsed, 0.95)
        subset_rows = {
            "derived": [row for row in valid if row["source_group"] == "derived"],
            "external_style": [row for row in valid if row["source_group"] == "external_style"],
            "synthetic": [row for row in valid if row["source_type"] == "synthetic"],
            "user_photo": [row for row in valid if row["source_type"] == "user_photo"],
            "public_photo": [row for row in valid if row["source_type"] == "public_photo"],
        }
        entry["subsets"] = {
            name: {
                "records": len(subset),
                "section_r5": sum(row["section_r5"] for row in subset) / max(1, len(subset)),
                "image_r5": sum(row["image_r5"] for row in subset) / max(1, len(subset)),
                "section_mrr": sum(row["section_rr"] for row in subset) / max(1, len(subset)),
                "image_mrr": sum(row["image_rr"] for row in subset) / max(1, len(subset)),
            }
            for name, subset in subset_rows.items()
        }
        summary[mode] = entry
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if profiler is not None:
        write_latency_summary(latency_events_path, latency_summary_path)
        print(f"latency_profile=OK events={latency_events_path} summary={latency_summary_path}")
    save_progress("complete")
    print(f"visual_experiment=OK records={len(rows)} summary={summary_path}")


if __name__ == "__main__":
    main()
