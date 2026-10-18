"""Independent Qwen3-VL image retrieval helpers for KBrag.

The visual index stores manual-image vectors only.  Text chunks remain in the
existing dense.faiss; the two channels meet at parent-section keys.
"""

from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import requests

from request_context import RequestCancelled, check_cancelled, remaining_timeout, sleep_with_cancel

try:
    from config_runtime import apply_default_env

    apply_default_env()
except Exception:
    pass


ROOT = Path(__file__).resolve().parent
VISUAL_INDEX_DIR = ROOT / "data" / "visual_index"
VISUAL_FAISS_PATH = VISUAL_INDEX_DIR / "visual.faiss"
VISUAL_METADATA_PATH = VISUAL_INDEX_DIR / "visual_metadata.json"
IMAGE_DIR = ROOT / "手册" / "插图"

DEFAULT_VISUAL_EMBEDDING_MODEL = os.getenv(
    "VISUAL_EMBEDDING_MODEL", "Qwen/Qwen3-VL-Embedding-8B"
)
DEFAULT_VISUAL_RERANK_MODEL = os.getenv(
    "VISUAL_RERANK_MODEL", "Qwen/Qwen3-VL-Reranker-8B"
)
VISUAL_DIMENSION = 4096
VISUAL_RECALL_TOP_N = 20
VISUAL_SECTION_POOL = 20
VISUAL_FINAL_TOP_N = 6

_VISUAL_INTENT_RE = re.compile(
    r"图片|图示|示意图|按钮|部件|组件|接口|位置|外观|指示灯|屏幕|标签|铭牌|"
    r"\b(?:image|picture|diagram|button|port|component|part|located|location|view|indicator|label|screen)\b",
    re.IGNORECASE,
)


class VisualRetrievalError(RuntimeError):
    """Raised when a visual service or index cannot produce a usable result."""


@dataclass(frozen=True)
class VisualHit:
    """One matched manual image and its parent-section mappings."""

    vector_id: int
    image_id: str
    product: str
    section_keys: tuple[str, ...]
    relative_path: str
    caption: str
    category: str
    score: float


def visual_retrieval_enabled() -> bool:
    """Return whether the optional visual channel is enabled."""
    return os.getenv("VISUAL_RETRIEVAL_ENABLED", "0").lower() in {"1", "true", "yes"}


def has_visual_intent(text: str) -> bool:
    """Gate text-to-image retrieval to explicitly visual questions."""
    return bool(_VISUAL_INTENT_RE.search(text or ""))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_to_data_url(path: Path) -> str:
    """Encode a local image for the remote VL APIs."""
    media_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def image_input_sha256(image: str) -> str:
    """Hash a data URL without retaining it in traces."""
    try:
        payload = image.split(",", 1)[1] if image.startswith("data:") else image
        return sha256_bytes(base64.b64decode(payload, validate=True))
    except Exception:
        return sha256_bytes(image.encode("utf-8", errors="replace"))


class VisualEmbeddingClient:
    """Small SiliconFlow-compatible client for text/image VL embeddings."""

    def __init__(self, base_url: str, api_key: str, timeout: float = 300) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._session_local = threading.local()

    def _session(self) -> requests.Session:
        """Return one reusable requests session per worker thread."""
        session = getattr(self._session_local, "value", None)
        if session is None:
            session = requests.Session()
            session.trust_env = False
            self._session_local.value = session
        return session

    def embed_inputs(
        self,
        inputs: list[str | dict[str, str]],
        model: str = DEFAULT_VISUAL_EMBEDDING_MODEL,
        dimension: int = VISUAL_DIMENSION,
    ) -> list[list[float]]:
        if not inputs:
            return []
        check_cancelled()
        body: dict[str, Any] = {"model": model, "input": inputs, "dimensions": dimension}
        try:
            return self._post(body)
        except requests.HTTPError as exc:
            # Some compatible deployments expose the 4096-d model but reject
            # the optional dimensions field. Retry once without changing data.
            status = getattr(exc.response, "status_code", None)
            if status not in {400, 422}:
                raise
            body.pop("dimensions", None)
            return self._post(body)

    def _post(self, body: dict[str, Any]) -> list[list[float]]:
        response = self._session().post(
            f"{self.base_url}/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=body,
            timeout=remaining_timeout(self.timeout),
        )
        response.raise_for_status()
        payload = response.json()
        data = sorted(payload.get("data") or [], key=lambda item: item.get("index", 0))
        vectors = [item.get("embedding") for item in data]
        if len(vectors) != len(body["input"]) or any(not isinstance(v, list) for v in vectors):
            raise VisualRetrievalError("VL embedding returned an unexpected vector count")
        return vectors


class VisualRerankClient:
    """Multimodal reranker with bounded retries and request cancellation."""

    def __init__(self, base_url: str, api_key: str, timeout: float = 120) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._session_local = threading.local()

    def _session(self) -> requests.Session:
        """Return one reusable requests session per worker thread."""
        session = getattr(self._session_local, "value", None)
        if session is None:
            session = requests.Session()
            session.trust_env = False
            self._session_local.value = session
        return session

    def rerank(
        self,
        query: str | dict[str, str],
        documents: list[str | dict[str, str]],
        top_n: int,
        model: str = DEFAULT_VISUAL_RERANK_MODEL,
    ) -> list[tuple[int, float]]:
        if not documents:
            return []
        last_error: Exception | None = None
        for attempt in range(2):
            check_cancelled()
            try:
                response = self._session().post(
                    f"{self.base_url}/rerank",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "query": query,
                        "documents": documents,
                        "top_n": min(top_n, len(documents)),
                        "return_documents": False,
                    },
                    timeout=remaining_timeout(self.timeout),
                )
                response.raise_for_status()
                items = response.json().get("results") or []
                ranked = [
                    (int(item["index"]), float(item.get("relevance_score", item.get("score", 0.0))))
                    for item in items
                    if isinstance(item, dict) and "index" in item
                ]
                if ranked:
                    return ranked
                raise VisualRetrievalError("VL rerank returned no candidates")
            except RequestCancelled:
                raise
            except Exception as exc:
                last_error = exc
                if attempt == 0:
                    sleep_with_cancel(0.5)
        raise VisualRetrievalError(f"VL rerank failed: {last_error}")


class VisualRetrievalEngine:
    """Load visual.faiss and retrieve manual images by text or image."""

    def __init__(
        self,
        *,
        index_path: Path = VISUAL_FAISS_PATH,
        metadata_path: Path = VISUAL_METADATA_PATH,
        base_url: str | None = None,
        api_key: str | None = None,
        embedding_client: VisualEmbeddingClient | None = None,
        rerank_client: VisualRerankClient | None = None,
    ) -> None:
        base_url = base_url or os.getenv("EMBEDDING_BASE_URL", "https://api.siliconflow.cn/v1")
        api_key = api_key if api_key is not None else os.getenv("EMBEDDING_API_KEY", "")
        rerank_base = os.getenv("RERANK_BASE_URL", base_url)
        rerank_key = os.getenv("RERANK_API_KEY", api_key)
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.embedding_client = embedding_client or VisualEmbeddingClient(base_url, api_key)
        self.rerank_client = rerank_client or VisualRerankClient(rerank_base, rerank_key)
        self.embedding_model = DEFAULT_VISUAL_EMBEDDING_MODEL
        self.items: list[dict[str, Any]] = []
        self.index: faiss.Index | None = None
        self.vectors: np.ndarray | None = None

    @property
    def available(self) -> bool:
        return self.index_path.exists() and self.metadata_path.exists()

    def ensure_index(self) -> None:
        if self.index is not None:
            return
        if not self.available:
            raise FileNotFoundError("visual index is not built")
        payload = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        self.items = list(payload.get("items") or [])
        self.embedding_model = payload.get("embedding_model") or DEFAULT_VISUAL_EMBEDDING_MODEL
        self.index = faiss.read_index(str(self.index_path))
        dimension = int(payload.get("embedding_dimension") or self.index.d)
        if self.index.ntotal != len(self.items) or self.index.d != dimension:
            raise VisualRetrievalError("visual index and metadata do not match")
        self.vectors = np.empty((self.index.ntotal, self.index.d), dtype=np.float32)
        self.index.reconstruct_n(0, self.index.ntotal, self.vectors)

    @staticmethod
    def _normalize(vector: list[float]) -> np.ndarray:
        array = np.asarray([vector], dtype=np.float32)
        if array.ndim != 2 or not np.isfinite(array).all():
            raise VisualRetrievalError("VL embedding contains invalid values")
        norm = np.linalg.norm(array, axis=1, keepdims=True)
        if np.any(norm == 0):
            raise VisualRetrievalError("VL embedding has zero norm")
        return array / norm

    def search_text(self, text: str, *, top_n: int = VISUAL_RECALL_TOP_N, products: list[str] | None = None) -> list[VisualHit]:
        if not text.strip():
            return []
        vector = self.embedding_client.embed_inputs([text], self.embedding_model)[0]
        return self._search(vector, top_n=top_n, products=products)

    def search_image(self, image: str, *, top_n: int = VISUAL_RECALL_TOP_N, products: list[str] | None = None) -> list[VisualHit]:
        if not image:
            return []
        vector = self.embedding_client.embed_inputs([{"image": image}], self.embedding_model)[0]
        return self._search(vector, top_n=top_n, products=products)

    def _search(self, vector: list[float], *, top_n: int, products: list[str] | None) -> list[VisualHit]:
        self.ensure_index()
        assert self.index is not None and self.vectors is not None
        query = self._normalize(vector)
        if query.shape[1] != self.index.d:
            raise VisualRetrievalError(
                f"visual query dimension {query.shape[1]} does not match index {self.index.d}"
            )
        allowed = [
            idx for idx, item in enumerate(self.items)
            if not products or item.get("product") in products
        ]
        if not allowed:
            return []
        allowed_array = np.asarray(allowed, dtype=np.int64)
        scores = self.vectors[allowed_array] @ query[0]
        order = np.argsort(scores)[::-1][:top_n]
        hits: list[VisualHit] = []
        for offset in order:
            vector_id = int(allowed_array[int(offset)])
            item = self.items[vector_id]
            hits.append(
                VisualHit(
                    vector_id=vector_id,
                    image_id=str(item["image_id"]),
                    product=str(item["product"]),
                    section_keys=tuple(item.get("section_keys") or ()),
                    relative_path=str(item["relative_path"]),
                    caption=str(item.get("caption") or ""),
                    category=str(item.get("category") or ""),
                    score=float(scores[int(offset)]),
                )
            )
        return hits


def parse_vision_analysis(raw: str) -> dict[str, Any]:
    """Normalize the structured image description without a second model call."""
    cleaned = (raw or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    try:
        payload = json.loads(cleaned)
    except Exception:
        payload = {"description": cleaned, "ocr_text": "", "keywords": []}
    description = str(payload.get("description") or "").strip()[:300]
    ocr_text = str(payload.get("ocr_text") or "").strip()[:500]
    keywords = [str(item).strip() for item in payload.get("keywords", []) if str(item).strip()][:8]
    return {"description": description, "ocr_text": ocr_text, "keywords": keywords}


def analyze_query_images(images: list[str], *, model: str | None = None) -> dict[str, Any]:
    """Ask the configured vision model for retrieval-only description/OCR terms."""
    if not images:
        return {"description": "", "ocr_text": "", "keywords": []}
    from llm_router import create_message_with_fallback

    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "只描述图片中明确可见、可用于检索产品手册的内容。提取可见OCR文字和最多8个关键词；"
                "不得猜测图片中看不到的型号、故障、原因或操作步骤。只输出JSON："
                '{"description":"","ocr_text":"","keywords":[]}'
            ),
        }
    ]
    content.extend({"type": "image_url", "image_url": {"url": image}} for image in images)
    response, _route = create_message_with_fallback(
        system="你是只做图像事实提取的检索预处理器。",
        messages=[{"role": "user", "content": content}],
        max_tokens=700,
        tools=None,
        model=model or os.getenv("SILICONFLOW_VISION_MODEL"),
    )
    raw = "\n".join(
        str(getattr(block, "text", "")) for block in response.content if getattr(block, "text", "")
    )
    return parse_vision_analysis(raw)


def analysis_query_text(original: str, analysis: dict[str, Any]) -> str:
    """Build the text used by product routing and the existing text index."""
    parts = [original.strip(), str(analysis.get("description") or "").strip(), str(analysis.get("ocr_text") or "").strip()]
    keywords = [str(item).strip() for item in analysis.get("keywords", []) if str(item).strip()]
    if keywords:
        parts.append(" ".join(keywords))
    return "\n".join(part for part in parts if part)


def rrf_rank(ranked_lists: list[list[str]], *, top_n: int, k: int = 60) -> tuple[list[str], dict[str, float]]:
    """Fuse arbitrary section keys without mixing incomparable raw scores."""
    scores: dict[str, float] = {}
    first_seen: dict[str, tuple[int, int]] = {}
    for list_idx, ranked in enumerate(ranked_lists):
        for rank, key in enumerate(ranked):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            first_seen.setdefault(key, (list_idx, rank))
    ordered = sorted(scores, key=lambda key: (-scores[key], *first_seen[key]))
    return ordered[:top_n], scores
