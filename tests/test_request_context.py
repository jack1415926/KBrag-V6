"""Focused regression tests for request identity and cooperative timeout handling."""

from __future__ import annotations

import asyncio
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

import api_server
from llm_router import AnthropicRoute, _create_openai_message
from rerank_client import RerankClient
from request_context import (
    RequestCancelled,
    check_cancelled,
    get_current_qid,
    has_deadline,
    remaining_timeout,
    request_scope,
)
from retrieval_engine import EmbeddingClient


class RequestContextTests(unittest.TestCase):
    """Verify request-local identity and bounded worker cancellation."""

    def test_qid_is_isolated_between_threads(self) -> None:
        barrier = threading.Barrier(2)

        def read_twice(label: str) -> tuple[str | None, str | None]:
            with request_scope(qid=label):
                barrier.wait()
                first = get_current_qid()
                time.sleep(0.01)
                return first, get_current_qid()

        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(read_twice, "qid-a")
            second = pool.submit(read_twice, "qid-b")
        self.assertEqual(first.result(), ("qid-a", "qid-a"))
        self.assertEqual(second.result(), ("qid-b", "qid-b"))
        self.assertIsNone(get_current_qid())

    def test_deadline_caps_timeout_and_event_cancels(self) -> None:
        cancel_event = threading.Event()
        self.assertFalse(has_deadline())
        with request_scope(deadline=time.monotonic() + 0.5, cancel_event=cancel_event):
            self.assertTrue(has_deadline())
            self.assertLessEqual(remaining_timeout(300), 0.5)
            cancel_event.set()
            with self.assertRaises(RequestCancelled):
                check_cancelled()
        self.assertFalse(has_deadline())

    def test_worker_stops_after_cancel_event(self) -> None:
        cancel_event = threading.Event()

        def fake_agent(_question: str, _session: str, _images: list[str]):
            while True:
                check_cancelled()
                time.sleep(0.005)

        with patch.object(api_server, "_run_agent_sync_impl", side_effect=fake_agent):
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    api_server._run_agent_sync,
                    "q",
                    "s",
                    [],
                    cancel_event,
                    time.monotonic() + 1,
                )
                time.sleep(0.02)
                cancel_event.set()
                with self.assertRaises(RequestCancelled):
                    future.result(timeout=0.5)

    def test_embedding_timeout_uses_remaining_request_budget(self) -> None:
        captured: dict[str, float] = {}

        class Response:
            def raise_for_status(self) -> None:
                pass

            def json(self) -> dict:
                return {"data": [{"index": 0, "embedding": [1.0]}]}

        def fake_post(*_args, **kwargs):
            captured["timeout"] = kwargs["timeout"]
            return Response()

        client = EmbeddingClient(base_url="https://example.invalid", api_key="test", timeout=300)
        with request_scope(deadline=time.monotonic() + 0.5):
            with patch("requests.Session.post", side_effect=fake_post):
                self.assertEqual(client.embed_texts(["x"], "model"), [[1.0]])
        self.assertGreater(captured["timeout"], 0)
        self.assertLessEqual(captured["timeout"], 0.5)

    def test_text_clients_reuse_one_session_per_thread(self) -> None:
        class EmbeddingResponse:
            status_code = 200

            def raise_for_status(self) -> None:
                pass

            def json(self) -> dict:
                return {"data": [{"index": 0, "embedding": [1.0]}]}

        embedding_session = SimpleNamespace(
            trust_env=True,
            post=lambda *_args, **_kwargs: EmbeddingResponse(),
        )
        with patch("requests.Session", return_value=embedding_session) as session_factory:
            client = EmbeddingClient(base_url="https://example.invalid", api_key="test")
            client.embed_texts(["a"], "model")
            client.embed_texts(["b"], "model")
        session_factory.assert_called_once()
        self.assertFalse(embedding_session.trust_env)

        rerank_session = SimpleNamespace(
            trust_env=True,
            post=lambda *_args, **_kwargs: SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {"results": [{"index": 0, "relevance_score": 1.0}]},
            ),
        )
        with patch("requests.Session", return_value=rerank_session) as session_factory:
            client = RerankClient(base_url="https://example.invalid", api_key="test")
            client.rerank("q1", ["d"], 1)
            client.rerank("q2", ["d"], 1)
        session_factory.assert_called_once()
        self.assertFalse(rerank_session.trust_env)

    def test_rerank_timeout_uses_remaining_request_budget(self) -> None:
        captured: dict[str, float] = {}

        class Response:
            def raise_for_status(self) -> None:
                pass

            def json(self) -> dict:
                return {"results": [{"index": 0, "relevance_score": 1.0}]}

        def fake_post(*_args, **kwargs):
            captured["timeout"] = kwargs["timeout"]
            return Response()

        client = RerankClient(base_url="https://example.invalid", api_key="test", timeout=120)
        with request_scope(deadline=time.monotonic() + 0.5):
            with patch("requests.Session.post", side_effect=fake_post):
                client._post("https://example.invalid/rerank", {"documents": ["x"]})
        self.assertGreater(captured["timeout"], 0)
        self.assertLessEqual(captured["timeout"], 0.5)

    def test_llm_timeout_uses_remaining_request_budget(self) -> None:
        captured: dict[str, float | int] = {}

        class FakeCompletions:
            def create(self, **_kwargs):
                message = SimpleNamespace(content="ok", tool_calls=None)
                return SimpleNamespace(choices=[SimpleNamespace(message=message)])

        class FakeOpenAI:
            def __init__(self, **kwargs):
                captured["timeout"] = kwargs["timeout"]
                captured["max_retries"] = kwargs["max_retries"]
                self.chat = SimpleNamespace(completions=FakeCompletions())

        route = AnthropicRoute("test", "https://example.invalid", "test", "model", "openai")
        with patch("openai.OpenAI", FakeOpenAI):
            with request_scope(deadline=time.monotonic() + 0.5):
                _create_openai_message(
                    route=route,
                    system="system",
                    messages=[{"role": "user", "content": "x"}],
                    max_tokens=8,
                    tools=None,
                    tool_choice=None,
                    model=None,
                    timeout=remaining_timeout(30),
                )
        self.assertGreater(captured["timeout"], 0)
        self.assertLessEqual(captured["timeout"], 0.5)
        self.assertEqual(captured["max_retries"], 0)


class ApiTimeoutTests(unittest.IsolatedAsyncioTestCase):
    """Exercise the actual /chat timeout wrapper without external services."""

    async def test_chat_timeout_cancels_and_reaps_worker(self) -> None:
        def fake_agent(_question: str, _session: str, _images: list[str]):
            while True:
                check_cancelled()
                time.sleep(0.005)

        request = api_server.Request({"type": "http", "headers": []})
        payload = api_server.ChatRequest(question="timeout test")
        with (
            patch.object(api_server, "_run_agent_sync_impl", side_effect=fake_agent),
            patch.object(api_server, "REQUEST_TIMEOUT_S", 0.05),
            patch.object(api_server, "_write_api_error_trace"),
        ):
            with self.assertRaises(api_server.HTTPException) as caught:
                await api_server.chat(payload, request)
            self.assertEqual(caught.exception.status_code, 504)
            for _ in range(20):
                if not api_server._BACKGROUND_AGENT_TASKS:
                    break
                await asyncio.sleep(0.01)
            self.assertFalse(api_server._BACKGROUND_AGENT_TASKS)
            self.assertEqual((await api_server.health())["background_agents"], 0)


if __name__ == "__main__":
    unittest.main()
