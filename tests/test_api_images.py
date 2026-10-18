"""Contract tests for exposing manual images through the FastAPI response."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import api_server
from submission_utils import format_submission_ret


ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "手册" / "插图"
SECTION_CHUNKS_PATH = ROOT / "data" / "section_chunks.json"


def referenced_image_ids() -> list[str]:
    sections = json.loads(SECTION_CHUNKS_PATH.read_text(encoding="utf-8"))
    return list(dict.fromkeys(pic for section in sections for pic in section.get("pics", [])))


def image_files_by_id() -> dict[str, Path]:
    return {path.stem: path for path in IMAGE_DIR.iterdir() if path.is_file()}


class ApiImageContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(api_server.app)
        cls.headers = {"Authorization": f"Bearer {api_server.EXPECTED_TOKEN}"}
        cls.referenced_ids = referenced_image_ids()
        cls.files_by_id = image_files_by_id()
        cls.jpg_id = next(image_id for image_id, path in cls.files_by_id.items() if path.suffix.lower() == ".jpg")
        cls.png_id = next(image_id for image_id, path in cls.files_by_id.items() if path.suffix.lower() == ".png")

    def _chat(self, *, answer: str, pics: list[str], route: str) -> dict:
        trace = {"result": {"tool_calls": 1 if route == "tech" else 0, "turns": 1}}
        with (
            patch.object(
                api_server,
                "_run_agent_sync",
                return_value=(answer, pics, route, trace),
            ),
            patch.object(api_server, "_write_api_success_trace"),
            patch.object(api_server, "_append_session_turn"),
        ):
            response = self.client.post(
                "/chat",
                headers=self.headers,
                json={"question": "test", "session_id": "api-image-contract"},
            )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["data"]

    def test_all_2563_referenced_image_ids_resolve(self) -> None:
        self.assertEqual(len(self.referenced_ids), 2563)
        mapping = getattr(api_server, "MANUAL_IMAGE_BY_ID", None)
        self.assertIsNotNone(mapping, "api_server.MANUAL_IMAGE_BY_ID 尚未实现")
        self.assertEqual([image_id for image_id in self.referenced_ids if image_id not in mapping], [])

    def test_image_filename_stems_are_unique(self) -> None:
        files = [path for path in IMAGE_DIR.iterdir() if path.is_file()]
        self.assertEqual(len(files), 2605)
        self.assertEqual(len({path.stem for path in files}), len(files))

    def test_jpg_endpoint_returns_image_jpeg(self) -> None:
        response = self.client.get(f"/manual-images/{self.jpg_id}")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.headers["content-type"], "image/jpeg")

    def test_png_endpoint_returns_image_png(self) -> None:
        response = self.client.get(f"/manual-images/{self.png_id}")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.headers["content-type"], "image/png")

    def test_unknown_image_id_returns_404(self) -> None:
        response = self.client.get("/manual-images/not-a-real-manual-image")
        self.assertEqual(response.status_code, 404)

    def test_path_traversal_cannot_read_outside_image_directory(self) -> None:
        for path in (
            "/manual-images/../config_runtime.py",
            "/manual-images/%2E%2E%2Fconfig_runtime.py",
            "/manual-images/%2e%2e%5cconfig_runtime.py",
        ):
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 404)
                self.assertNotIn("DEFAULT_ENV", response.text)

    def test_technical_image_order_matches_pic_marker_order(self) -> None:
        pics = [self.jpg_id, self.png_id]
        data = self._chat(answer="第一张<PIC>第二张<PIC>", pics=pics, route="tech")
        self.assertIn("images", data)
        self.assertEqual([image["id"] for image in data["images"]], pics)
        self.assertEqual([image["marker_index"] for image in data["images"]], [0, 1])
        self.assertTrue(data["images"][0]["url"].endswith(f"/manual-images/{self.jpg_id}"))
        self.assertTrue(data["images"][1]["url"].endswith(f"/manual-images/{self.png_id}"))
        self.assertEqual(data["display_answer"], "第一张<PIC>第二张<PIC>")

    def test_service_response_always_has_empty_images(self) -> None:
        data = self._chat(answer="请保留照片并联系平台客服。", pics=[self.jpg_id], route="service")
        self.assertIn("images", data)
        self.assertEqual(data["images"], [])

    def test_legacy_answer_field_is_unchanged(self) -> None:
        answer = "充电状态<PIC>"
        pics = [self.jpg_id]
        data = self._chat(answer=answer, pics=pics, route="tech")
        self.assertEqual(data["answer"], format_submission_ret(64, answer, pics))

    def test_technical_response_without_pics_has_empty_images(self) -> None:
        data = self._chat(answer="手册中未提供相关图片。", pics=[], route="tech")
        self.assertIn("images", data)
        self.assertEqual(data["images"], [])

    def test_web_index_is_served(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn("KBrag 图文问答", response.text)

    def test_latency_status_is_idle_without_progress_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.json"
            with patch.object(api_server, "LATENCY_PROGRESS_PATH", missing):
                response = self.client.get("/latency-baseline/status")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "idle", "current": 0, "total": 0})


if __name__ == "__main__":
    unittest.main()
