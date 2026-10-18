"""Focused offline checks for the optional visual retrieval channel."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import faiss
import numpy as np
import requests

from agent import _question_images, run_agent
from build_visual_index import collect_items
from retrieval_engine import RetrievalEngine, SearchResult
from run_visual_retrieval_experiment import (
    LatencyProfiler,
    _PROFILE_CONTEXT,
    install_text_remote_limit,
    install_latency_profile,
    load_jsonl,
    merged_fusion_search,
    write_latency_summary,
)
from visual_retrieval import (
    VisualHit,
    VisualEmbeddingClient,
    VisualRerankClient,
    VisualRetrievalEngine,
    has_visual_intent,
    parse_vision_analysis,
    rrf_rank,
)


class FakeEmbeddingClient:
    def __init__(self, vector: list[float]) -> None:
        self.vector = vector

    def embed_inputs(self, inputs, model, dimension=4096):
        return [self.vector for _item in inputs]


class FakeRerankClient:
    def rerank(self, query, documents, top_n, model=None):
        return [(index, 1.0 - index / 10) for index in range(min(top_n, len(documents)))]


class VisualRetrievalTests(unittest.TestCase):
    def test_visual_clients_reuse_one_session_per_thread(self) -> None:
        class EmbeddingResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {"data": [{"index": 0, "embedding": [1.0]}]}

        embedding_session = SimpleNamespace(trust_env=True, post=lambda *_args, **_kwargs: EmbeddingResponse())
        with patch("requests.Session", return_value=embedding_session) as session_factory:
            client = VisualEmbeddingClient("https://example.invalid", "test")
            client.embed_inputs(["a"], dimension=1)
            client.embed_inputs(["b"], dimension=1)
        session_factory.assert_called_once()
        self.assertFalse(embedding_session.trust_env)

        class RerankResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {"results": [{"index": 0, "relevance_score": 1.0}]}

        rerank_session = SimpleNamespace(trust_env=True, post=lambda *_args, **_kwargs: RerankResponse())
        with patch("requests.Session", return_value=rerank_session) as session_factory:
            client = VisualRerankClient("https://example.invalid", "test")
            client.rerank("q1", ["d"], 1)
            client.rerank("q2", ["d"], 1)
        session_factory.assert_called_once()
        self.assertFalse(rerank_session.trust_env)
    def test_latency_profiler_durably_records_success_and_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            events_path = Path(tmp) / "events.jsonl"
            summary_path = Path(tmp) / "summary.json"
            profiler = LatencyProfiler(events_path)

            self.assertEqual(profiler.call("ok", lambda value: value + 1, 1, payload_items=1), 2)
            with self.assertRaisesRegex(ValueError, "boom"):
                profiler.call("failed", lambda: (_ for _ in ()).throw(ValueError("boom")))

            events = load_jsonl(events_path)
            self.assertEqual([event["stage"] for event in events], ["ok", "failed"])
            self.assertIsNone(events[0]["error_type"])
            self.assertEqual(events[1]["error_type"], "ValueError")
            write_latency_summary(events_path, summary_path)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["ok"]["calls"], 1)
            self.assertEqual(summary["failed"]["errors"], 1)

    def test_latency_profile_wrappers_preserve_client_results_and_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            events_path = Path(tmp) / "events.jsonl"
            engine = SimpleNamespace(
                client=SimpleNamespace(embed_texts=lambda texts, model: [[1.0]]),
                rerank_client=SimpleNamespace(rerank=lambda query, documents, top_n: [0]),
                visual_engine=None,
            )
            visual = SimpleNamespace(
                embedding_client=SimpleNamespace(embed_inputs=lambda inputs, model, dimension=4096: [[1.0]]),
                rerank_client=SimpleNamespace(rerank=lambda query, documents, top_n, model="vl": [(0, 1.0)]),
            )
            router = SimpleNamespace(route=lambda query: {"query": query})
            original_http_post = requests.Session.post
            token = _PROFILE_CONTEXT.set(
                {"repeat": 0, "mode": "fusion", "case_id": "V001", "original_question": "原问题", "expanded_query": "扩展问题"}
            )
            try:
                install_latency_profile(LatencyProfiler(events_path), engine, visual, router)
                self.assertEqual(engine.client.embed_texts(["原问题"], "text-model"), [[1.0]])
                self.assertEqual(engine.rerank_client.rerank("扩展问题", ["doc"], 1), [0])
                self.assertEqual(visual.embedding_client.embed_inputs([{"image": "x"}], "vl-model"), [[1.0]])
                self.assertEqual(visual.rerank_client.rerank("q", ["doc"], 1), [(0, 1.0)])
                self.assertEqual(router.route("原问题"), {"query": "原问题"})
            finally:
                requests.Session.post = original_http_post
                _PROFILE_CONTEXT.reset(token)

            stages = {event["stage"] for event in load_jsonl(events_path)}
            self.assertEqual(
                stages,
                {"text_original_embedding", "text_expanded_rerank", "vl_image_embedding", "vl_rerank", "text_original_product_route"},
            )

    def test_merged_fusion_uses_one_text_rrf_and_one_text_rerank(self) -> None:
        result = SearchResult(0, "P", "H", "T", [], 1.0, {"parent_section_id": 1})

        class FakeEngine:
            rerank_candidate_pool = 80

            def __init__(self):
                self.rrf_lists = None
                self.rerank_calls = 0

            def _collect_text_recall_rankings(self, keywords, **kwargs):
                return [[1], [2], [], []], kwargs["semantic_query"]

            def _rrf_merge(self, ranked_lists, top_n):
                self.rrf_lists = ranked_lists
                return [1, 2]

            def _rerank_candidates(self, query, candidates, top_n):
                self.rerank_calls += 1
                return candidates

            def _result_dedup_key(self, doc_id):
                return str(doc_id)

            def _build_results(self, doc_ids):
                return [result]

            def _merge_visual_results(self, text_results, **kwargs):
                return text_results

        class FakeVisual:
            def search_text(self, *args, **kwargs):
                return []

            def search_image(self, *args, **kwargs):
                return []

        route = SimpleNamespace(products=["P"], reason="explicit_product_name", confidence="high")
        engine = FakeEngine()
        actual, _route = merged_fusion_search(
            engine, FakeVisual(), SimpleNamespace(route=lambda query: route), "原问题", "原问题\n图像补充", ["image"]
        )
        self.assertEqual(actual, [result])
        self.assertEqual(len(engine.rrf_lists), 8)
        self.assertEqual(engine.rerank_calls, 1)

    def test_serial_mode_limits_text_remote_calls_to_one(self) -> None:
        state = {"active": 0, "maximum": 0}
        lock = threading.Lock()

        def remote(*args, **kwargs):
            with lock:
                state["active"] += 1
                state["maximum"] = max(state["maximum"], state["active"])
            time.sleep(0.02)
            with lock:
                state["active"] -= 1
            return []

        engine = SimpleNamespace(
            client=SimpleNamespace(embed_texts=remote),
            rerank_client=SimpleNamespace(rerank=remote),
        )
        install_text_remote_limit(engine)
        token = _PROFILE_CONTEXT.set({"mode": "fusion_text_serial"})
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [
                    pool.submit(copy_context().run, engine.client.embed_texts, ["q"], "m"),
                    pool.submit(copy_context().run, engine.rerank_client.rerank, "q", ["d"], 1),
                ]
                for future in futures:
                    future.result()
        finally:
            _PROFILE_CONTEXT.reset(token)
        self.assertEqual(state["maximum"], 1)

    def test_repository_image_inventory_matches_plan(self) -> None:
        items, stats = collect_items()
        self.assertEqual(len(items), 2504)
        self.assertEqual(stats["referenced_images"], 2563)
        self.assertEqual(stats["excluded_noise"], 59)
        self.assertEqual(stats["unreferenced_files"], 42)
        self.assertEqual(len({item["image_id"] for item in items}), len(items))

    def test_visual_intent_and_vision_json_fallback(self) -> None:
        self.assertTrue(has_visual_intent("这个按钮在哪里？"))
        self.assertTrue(has_visual_intent("show the port location"))
        self.assertFalse(has_visual_intent("为什么会自动关机？"))
        parsed = parse_vision_analysis('{"description":"面板","ocr_text":"E1","keywords":["按钮"]}')
        self.assertEqual(parsed, {"description": "面板", "ocr_text": "E1", "keywords": ["按钮"]})
        self.assertEqual(parse_vision_analysis("plain description")["description"], "plain description")

    def test_question_images_are_internal_only(self) -> None:
        image = "data:image/png;base64,AAAA"
        content = [{"type": "text", "text": "q"}, {"type": "image_url", "image_url": {"url": image}}]
        self.assertEqual(_question_images(content), [image])

    def test_service_image_request_skips_visual_retrieval_and_trace_base64(self) -> None:
        image = "data:image/png;base64,AAAA"
        content = [{"type": "text", "text": "包装破损怎么办"}, {"type": "image_url", "image_url": {"url": image}}]

        class FakeEngine:
            def ensure_index(self):
                pass

        response = SimpleNamespace(content=[SimpleNamespace(type="text", text="请保留照片并联系平台客服。")])
        with (
            patch.dict(os.environ, {"VISUAL_RETRIEVAL_ENABLED": "1"}),
            patch("agent.analyze_query_images") as analyze,
            patch("agent.create_message_with_fallback", return_value=(response, SimpleNamespace(name="test"))),
        ):
            result = run_agent(content, FakeEngine(), question_id=0, collect_trace=True)
        analyze.assert_not_called()
        self.assertNotIn(image, json.dumps(result.trace, ensure_ascii=False))

    def test_visual_index_search_and_product_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            index_path = root / "visual.faiss"
            metadata_path = root / "visual_metadata.json"
            vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
            index = faiss.IndexFlatIP(2)
            index.add(vectors)
            faiss.write_index(index, str(index_path))
            metadata_path.write_text(
                json.dumps(
                    {
                        "embedding_model": "test",
                        "embedding_dimension": 2,
                        "items": [
                            {"image_id": "a", "product": "P1", "section_keys": ["P1|1"], "relative_path": "a.jpg"},
                            {"image_id": "b", "product": "P2", "section_keys": ["P2|2"], "relative_path": "b.jpg"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            engine = VisualRetrievalEngine(
                index_path=index_path,
                metadata_path=metadata_path,
                embedding_client=FakeEmbeddingClient([0.9, 0.1]),
                rerank_client=FakeRerankClient(),
            )
            self.assertEqual(engine.search_text("query", top_n=1)[0].image_id, "a")
            self.assertEqual(engine.search_text("query", top_n=1, products=["P2"])[0].image_id, "b")

    def test_rrf_uses_section_keys(self) -> None:
        ordered, scores = rrf_rank([["P|1", "P|2"], ["P|2", "P|3"]], top_n=3)
        self.assertEqual(ordered[0], "P|2")
        self.assertGreater(scores["P|2"], scores["P|1"])

    def test_disabled_feature_preserves_text_results(self) -> None:
        result = SearchResult(0, "P", "H", "T", [], 1.0, {"parent_section_id": 1})
        engine = RetrievalEngine()
        with (
            patch.object(engine, "_search_text_manual", return_value=([result], 0)),
            patch.dict(os.environ, {"VISUAL_RETRIEVAL_ENABLED": "0"}),
        ):
            actual, filtered = engine.search_manual(["x"], original_query="按钮")
        self.assertIs(actual[0], result)
        self.assertEqual(filtered, 0)

    def test_enabled_wrapper_runs_text_and_visual_without_changing_tool_schema(self) -> None:
        result = SearchResult(0, "P", "H", "T", [], 1.0, {"parent_section_id": 1})

        class FakeVisual:
            def ensure_index(self):
                pass

            def search_text(self, *args, **kwargs):
                return []

            def search_image(self, *args, **kwargs):
                return []

        engine = RetrievalEngine()
        engine.visual_engine = FakeVisual()
        with (
            patch.object(engine, "_search_text_manual", return_value=([result], 0)) as text_search,
            patch.object(engine, "_merge_visual_results", return_value=[result]) as merge,
            patch.dict(os.environ, {"VISUAL_RETRIEVAL_ENABLED": "1"}),
        ):
            actual, _ = engine.search_manual(
                ["按钮"],
                original_query="扩展后的按钮视觉描述",
                visual_query_text="扩展后的按钮视觉描述",
                original_text_query="按钮在哪里",
                original_text_products=["P"],
            )
        self.assertEqual(actual, [result])
        self.assertEqual(text_search.call_count, 2)
        merge.assert_called_once()
        self.assertEqual(merge.call_args.kwargs["original_text_results"], [result])

    def test_visual_only_section_can_join_text_results(self) -> None:
        image_path = next((Path(__file__).resolve().parents[1] / "手册" / "插图").glob("*.jpg"))
        relative = image_path.relative_to(Path(__file__).resolve().parents[1]).as_posix()
        text = SearchResult(0, "P", "H1", "T1", [], 1.0, {"parent_section_id": 1})
        hit = VisualHit(0, "img", "P", ("P|2",), relative, "caption", "part_view", 0.9)

        class FakeVisual:
            rerank_client = FakeRerankClient()

            def ensure_index(self):
                pass

            def search_text(self, *args, **kwargs):
                return [hit]

            def search_image(self, *args, **kwargs):
                return [hit]

        engine = RetrievalEngine()
        engine.visual_engine = FakeVisual()
        engine.section_lookup = {
            ("P", 1): {"product": "P", "section_id": 1, "heading": "H1", "text": "T1", "pics": [], "summary": "", "tags": []},
            ("P", 2): {"product": "P", "section_id": 2, "heading": "H2", "text": "T2", "pics": ["img"], "summary": "", "tags": []},
        }
        merged = engine._merge_visual_results([text], query="按钮", images=["data:image/png;base64,AAAA"], products=None, top_k=2)
        self.assertEqual({result.source["parent_section_id"] for result in merged}, {1, 2})
        visual = next(result for result in merged if result.source["parent_section_id"] == 2)
        self.assertIn("visual_image", visual.source["retrieval_channels"])
        self.assertEqual(visual.source["matched_visual_images"], ["img"])


if __name__ == "__main__":
    unittest.main()
