"""Offline checks for detailed search_manual timing trace output."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import faiss
import numpy as np

from agent import _build_trace_tool_event
from retrieval_engine import EmbeddingClient, RetrievalEngine
from run_embedding_request_mode_experiment import balanced_order, summarize as summarize_modes
from run_search_trace_baseline import summarize as summarize_baseline
from run_text_query_embed_mode_gold import query_keywords, summarize as summarize_gold_modes
from run_text_query_strategy_gold import strategy_inputs, summarize as summarize_query_strategies


class FakeRerankClient:
    def rerank(self, query, documents, top_n):
        return [SimpleNamespace(index=index, score=1.0) for index in range(min(top_n, len(documents)))]


class SearchManualTraceTests(unittest.TestCase):
    def test_query_strategy_inputs_and_hybrid_gate(self) -> None:
        question = "椅子的扶手为什么松动？"
        keywords = ["扶手", "松动"]
        self.assertEqual(strategy_inputs("full_only", question, keywords), ([], question, question))
        self.assertEqual(strategy_inputs("keywords_only", question, keywords), (keywords, "", question))
        self.assertEqual(strategy_inputs("hybrid", question, keywords), (keywords, question, question))
        rows = []
        for strategy, inputs, reranks in (("full_only", 1, 1), ("keywords_only", 3, 2), ("hybrid", 3, 1)):
            rows.append({
                "repeat": 1, "id": 1, "strategy": strategy, "error": None,
                "elapsed": 2.0, "embedding_elapsed": 1.0, "embedding_inputs": inputs,
                "rerank_calls": reranks, "hit_at_5": 1, "reciprocal_rank": 1.0,
                "top_sections": ["P|1"], "first_relevant_rank": 1,
            })
        summary, gate = summarize_query_strategies(rows, repeats=1)
        self.assertTrue(gate["passed"])
        self.assertEqual(summary["strategies"]["hybrid"]["rerank_calls_median"], 1)

    def test_gold_mode_keywords_and_gate_are_deterministic(self) -> None:
        keywords = query_keywords("如何给空调遥控器安装电池？")
        self.assertLessEqual(len(keywords), 4)
        self.assertEqual(len(keywords), len(set(keywords)))
        self.assertNotIn("如何", keywords)
        rows = [
            {
                "repeat": 1, "id": 72, "mode": "serial", "error": None,
                "elapsed": 4.0, "embedding_elapsed": 3.0, "embedding_requests": 5,
                "hit_at_5": 1, "reciprocal_rank": 1.0, "top_sections": ["空调手册|8"],
                "first_relevant_rank": 1,
            },
            {
                "repeat": 1, "id": 72, "mode": "batch", "error": None,
                "elapsed": 2.0, "embedding_elapsed": 1.0, "embedding_requests": 1,
                "hit_at_5": 1, "reciprocal_rank": 1.0, "top_sections": ["空调手册|8"],
                "first_relevant_rank": 1,
            },
        ]
        summary, gate = summarize_gold_modes(rows, repeats=1)
        self.assertTrue(gate["passed"])
        self.assertEqual(summary["comparison"]["top5_equal"], 1)

    def test_invalid_text_query_embed_mode_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "TEXT_QUERY_EMBED_MODE"):
            RetrievalEngine(text_query_embed_mode="parallel")

    def test_batch_mode_embeds_main_and_keywords_once(self) -> None:
        engine = RetrievalEngine(text_query_embed_mode="batch")
        engine.product_chunk_ids = {"P": [0, 1, 2]}
        trace: dict = {}
        with (
            patch.object(engine, "ensure_index"),
            patch.object(engine, "_sparse_recall", return_value=[0]),
            patch.object(engine.client, "embed_texts", return_value=[[1.0], [2.0], [3.0]]) as embed,
            patch.object(engine, "_dense_recall_from_vector", side_effect=[[0, 1], [1], [2]]),
        ):
            rankings, rerank_query = engine._collect_text_recall_rankings(
                ["扶手", "螺丝"],
                semantic_query="扶手松动",
                products=["P"],
                timing_trace=trace,
            )

        self.assertEqual(embed.call_count, 1)
        self.assertEqual(embed.call_args.args[0], ["扶手松动", "扶手", "螺丝"])
        self.assertEqual(rankings, [[0], [0, 1], [0], [1, 2]])
        self.assertEqual(rerank_query, "扶手松动")
        self.assertEqual(trace["text_query_embed_mode"], "batch")
        self.assertEqual(trace["main_dense"]["input_count"], 3)
        self.assertTrue(all(call["shared_batch"] for call in trace["keyword_dense"]["calls"]))

    def test_precomputed_vector_search_matches_index_order(self) -> None:
        engine = RetrievalEngine()
        engine.dense_vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        engine.dense_index = faiss.IndexFlatIP(2)
        engine.dense_index.add(engine.dense_vectors)
        self.assertEqual(engine._dense_recall_from_vector([0.1, 0.9], 2), [1, 0])
        self.assertEqual(engine._dense_recall_from_vector([0.1, 0.9], 2, [0]), [0])

    def test_mode_experiment_rotates_order_and_summarizes_requests(self) -> None:
        self.assertEqual(balanced_order(0), ["serial", "batch", "parallel"])
        self.assertEqual(balanced_order(1), ["batch", "parallel", "serial"])
        rows = []
        for mode, request_count in (("serial", 5), ("batch", 1), ("parallel", 5)):
            rows.append({
                "repeat": 1,
                "mode": mode,
                "elapsed": 2.0,
                "error": None,
                "signature": ["P|1"],
                "timings": {"embedding": {"elapsed": 1.0, "request_count": request_count, "http_statuses": [200]}},
            })
        summary = summarize_modes(rows)
        self.assertEqual(summary["batch"]["request_count_per_run"], 1)
        self.assertEqual(summary["parallel"]["request_count_per_run"], 5)
        self.assertEqual(summary["batch"]["serial_signature_matches"], 1)

    def test_baseline_summary_uses_persisted_stage_timings(self) -> None:
        rows = [
            {"elapsed": 2.0, "error": None, "retrieval_timings": {"main_dense": {"elapsed": 1.5}}},
            {"elapsed": 4.0, "error": None, "retrieval_timings": {"main_dense": {"elapsed": 3.0}}},
            {"elapsed": 1.0, "error": "boom", "retrieval_timings": {}},
        ]
        summary = summarize_baseline(rows)
        self.assertEqual(summary["successful"], 2)
        self.assertEqual(summary["errors"], 1)
        self.assertEqual(summary["total_median"], 3.0)
        self.assertEqual(summary["stages"]["main_dense"]["median"], 2.25)

    def test_embedding_trace_records_http_status_without_payload(self) -> None:
        class Response:
            status_code = 200

            def raise_for_status(self) -> None:
                pass

            def json(self) -> dict:
                return {"data": [{"index": 0, "embedding": [1.0]}]}

        trace: dict = {}
        client = EmbeddingClient(base_url="https://example.invalid", api_key="test")
        with patch("requests.Session.post", return_value=Response()):
            self.assertEqual(client.embed_texts(["query"], "model", timing_trace=trace), [[1.0]])

        self.assertEqual(trace["input_count"], 1)
        self.assertEqual(trace["http_status"], 200)
        self.assertGreaterEqual(trace["http_elapsed"], 0.0)
        self.assertNotIn("query", trace)

    def test_text_search_records_recall_rerank_rrf_and_parent_stages(self) -> None:
        engine = RetrievalEngine(text_query_embed_mode="serial")
        engine.rerank_enabled = True
        engine.rerank_candidate_pool = 80
        engine.rerank_client = FakeRerankClient()
        engine.product_chunk_ids = {"P": [0, 1, 2, 3]}
        engine.retrieval_chunks = [
            {
                "product": "P",
                "heading": f"H{index}",
                "caption_aux": "",
                "summary": "",
                "text": f"T{index}",
                "pics": [],
                "chunk_id": index,
                "parent_section_id": index,
            }
            for index in range(4)
        ]
        engine.section_lookup = {
            ("P", index): {
                "product": "P",
                "section_id": index,
                "heading": f"H{index}",
                "text": f"Section {index}",
                "pics": [],
                "summary": "",
                "tags": [],
            }
            for index in range(4)
        }
        trace: dict = {}

        with (
            patch.object(engine, "ensure_index"),
            patch.object(engine, "_sparse_recall", return_value=[0, 1]),
            patch.object(engine, "_dense_recall", return_value=[2, 3]),
        ):
            results, filtered = engine._search_text_manual(
                ["扶手", "螺丝"],
                semantic_query="扶手松动",
                original_query="椅子的扶手为什么松动",
                products=["P"],
                timing_trace=trace,
            )

        self.assertEqual(filtered, 0)
        self.assertTrue(results)
        self.assertEqual(trace["mode"], "text")
        self.assertEqual(trace["keyword_count"], 2)
        self.assertEqual(trace["main_dense"]["input_count"], 1)
        self.assertEqual(trace["main_dense"]["status"], "ok")
        self.assertEqual(trace["keyword_dense"]["call_count"], 2)
        self.assertEqual(trace["keyword_sparse"]["call_count"], 2)
        self.assertEqual(trace["rrf"]["ranked_list_count"], 4)
        self.assertTrue(trace["keyword_rerank"]["executed"])
        self.assertTrue(trace["original_rerank"]["executed"])
        self.assertEqual(trace["parent_build"]["returned_sections"], len(results))
        self.assertGreaterEqual(trace["total_elapsed"], 0.0)

    def test_tool_event_includes_retrieval_timings(self) -> None:
        event = _build_trace_tool_event(
            index=1,
            name="search_manual",
            input_data={"keywords": ["扶手"]},
            default_products=["人体工学椅手册"],
            default_query_context="扶手为什么松动",
            elapsed=1.25,
            pics=[],
            result_text="(无检索结果)",
            retrieval_timings={"main_dense": {"elapsed": 1.0, "status": "ok"}},
        )

        self.assertEqual(event["retrieval_timings"]["main_dense"]["status"], "ok")
        self.assertEqual(event["retrieval_timings"]["main_dense"]["elapsed"], 1.0)


if __name__ == "__main__":
    unittest.main()
