"""
V5 检索引擎：

- 稀疏检索：jieba + rank_bm25
- 稠密检索：硅基流动 Qwen/Qwen3-Embedding-8B + FAISS(IP)
- 融合：RRF
- 重排：硅基流动 Qwen/Qwen3-Reranker-8B

说明：
- 当前默认 embedding/rerank 均走硅基流动线上服务，代码内保留默认 key。
- 本地 8091/8090 仅作为显式 fallback：只有通过 EMBEDDING_BASE_URL/RERANK_BASE_URL 覆盖到本地时才会使用。
- rerank 不再做全库预编码，而是在召回后对 `(query, doc)` 候选对打分。
"""

from __future__ import annotations

import json
import os
import pickle
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import faiss
import jieba
import numpy as np
import requests
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from dotenv import load_dotenv
from rerank_client import RerankClient, RerankError
from request_context import RequestCancelled, check_cancelled, get_current_qid, remaining_timeout

load_dotenv()

try:
    from config_runtime import apply_default_env

    apply_default_env()
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
INDEX_DIR = DATA_DIR / "index"

RETRIEVAL_CHUNKS_PATH = DATA_DIR / "retrieval_chunks.json"
SECTION_CHUNKS_PATH = DATA_DIR / "section_chunks.json"
CATALOG_PATH = DATA_DIR / "catalog.json"

FAISS_PATH = INDEX_DIR / "dense.faiss"
METADATA_PATH = INDEX_DIR / "retrieval_index.pkl"

DEFAULT_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "https://api.siliconflow.cn/v1")
DEFAULT_API_KEY = os.getenv("EMBEDDING_API_KEY", "")
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B")
DEFAULT_MAX_CONCURRENCY = int(os.getenv("EMBEDDING_MAX_CONCURRENCY", "4"))
DEFAULT_TEXT_QUERY_EMBED_MODE = os.getenv("TEXT_QUERY_EMBED_MODE", "serial").strip().lower()
DEFAULT_RERANK_BASE_URL = os.getenv("RERANK_BASE_URL", "https://api.siliconflow.cn/v1")
DEFAULT_RERANK_API_KEY = os.getenv("RERANK_API_KEY", "")
DEFAULT_RERANK_MODEL = os.getenv("RERANK_MODEL_ALIAS", "Qwen/Qwen3-Reranker-8B")
DEFAULT_RERANK_ENABLED = os.getenv("RERANK_ENABLED", "1").lower() not in {"0", "false", "no"}
DEFAULT_RERANK_CANDIDATE_POOL = max(1, int(os.getenv("RERANK_CANDIDATE_POOL", "80")))
RERANK_FALLBACK_LOG_PATH = Path(os.getenv("RERANK_FALLBACK_LOG", "")).expanduser() if os.getenv("RERANK_FALLBACK_LOG") else None
RERANK_TIMING_LOG_PATH = Path(os.getenv("RERANK_TIMING_LOG", "")).expanduser() if os.getenv("RERANK_TIMING_LOG") else None
_RERANK_FALLBACK_LOCK = threading.Lock()
RETURN_PARENT_SECTION = os.getenv("RETURN_PARENT_SECTION", "1").lower() not in {"0", "false", "no"}


def normalize_product_name(product: str) -> str:
    return product.lower().replace("手册", "").strip()


# 段级别名：让非手册术语（如 "battery conversion"）在 BM25/dense 全库搜索时
# 仍能命中正确段。不改产品路由，只影响 search_manual(products=[]) 全库搜的排名。
_SECTION_ALIASES: dict[str, list[str]] = {}
_aliases_path = DATA_DIR / "section_aliases.json"
if _aliases_path.exists():
    try:
        with open(_aliases_path) as f:
            _SECTION_ALIASES = json.load(f).get("aliases", {})
    except Exception:
        pass


def build_searchable_text(chunk: dict) -> str:
    # caption_aux（info_table 表格 OCR 全文）仅用于召回，不进模型正文。
    # 不截断：长文本由 _embed_corpus 分片并做 mean-pooling，避免删除表格尾部信息；极少数超模型 token
    # 上限的由 _embed 分片平均处理，不丢尾部信息。
    parts = [
        chunk.get("product", ""),
        chunk.get("heading", ""),
        chunk.get("summary", ""),
        chunk.get("text", ""),
        chunk.get("caption_aux", ""),
    ]
    # 注入段级业务别名（如 "battery conversion" → Battery switches 段）
    alias_key = f"{chunk.get('product', '')}|{chunk.get('heading', '')}"
    aliases = _SECTION_ALIASES.get(alias_key, [])
    if aliases:
        parts.append("\n".join(aliases))
    return "\n".join(part for part in parts if part).strip()


# rerank 输入长度上限：reranker 实测 ~1000 字符内安全（10117 字才 500）。截到 900 留余量。
_RERANK_MAX_CHARS = 900


def build_rerank_text(chunk: dict) -> str:
    """rerank 输入文本：用完整 caption_aux（含 info_table 表格数据，如 'MAXIMUM LOAD 160kg'），
    并把 caption_aux 提前，保证表格数据进 rerank（否则 query='max load' 匹配不到中文表名）；
    整体截断到 _RERANK_MAX_CHARS 避免超 reranker 上限触发 500（500 会让整批回退 RRF）。
    关键规格/数据通常在表头/前部，截尾影响小。召回端仍用完整 caption_aux，不受此截断影响。"""
    parts = [
        chunk.get("product", ""),
        chunk.get("heading", ""),
        chunk.get("caption_aux", ""),   # 提前：info_table 表格数据优先进 rerank
        chunk.get("summary", ""),
        chunk.get("text", ""),
    ]
    return "\n".join(part for part in parts if part).strip()[:_RERANK_MAX_CHARS]


# dense 单片字符上限：中文 1 字≈1 token，保留现有小片段策略以控制单次请求长度。
# 注意：这只切 dense 的 embedding 输入，BM25 仍用完整 search_texts（关键词全覆盖）。
_EMBED_MAX_SEG_CHARS = 380


def split_text_for_embedding(text: str, max_chars: int = _EMBED_MAX_SEG_CHARS) -> list[str]:
    """按行（markdown 表格行/段边界）切片，每片 ≤ max_chars，不切断整行。
    长 info_table 表格靠多片 + mean-pooling 完整进 dense 向量，零删尾部信息。"""
    text = text or ""
    if len(text) <= max_chars:
        return [text] if text.strip() else []
    segs: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for line in text.split("\n"):
        ll = len(line) + 1
        if cur and cur_len + ll > max_chars:
            segs.append("\n".join(cur))
            cur, cur_len = [], 0
        if len(line) > max_chars:  # 单行超长（罕见），硬切
            for j in range(0, len(line), max_chars):
                segs.append(line[j:j + max_chars])
            continue
        cur.append(line)
        cur_len += ll
    if cur:
        segs.append("\n".join(cur))
    return [s for s in segs if s.strip()] or [text[:max_chars]]


def contains_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def _tokenize_english(text: str) -> list[str]:
    """纯英文/数字/型号文本的简单空白分词，不做 stemming 以避免破坏型号/缩写匹配。"""
    import re
    tokens: list[str] = []
    for raw in re.split(r'[\s,;:!?()\[\]{}"\'<>]+', text):
        w = raw.strip().lower()
        if not w:
            continue
        # 保留原始词项（型号/缩写对 BM25 关键词匹配至关重要）
        # 同时补充去标点版本（如 "safety." → "safety"）
        cleaned = "".join(ch for ch in w if ch.isalnum() or ch in "._-+/")
        if cleaned and cleaned != w:
            tokens.append(cleaned)
        tokens.append(w)
    return list(dict.fromkeys(tokens))  # 去重保序


def tokenize_mixed(text: str) -> list[str]:
    """
    中英混合分词：
    - 中文：jieba 分词 + 双字片段补充
    - 英文/型号/数字：简单空白分词，保留原始词项（不做 stemming，避免破坏型号和缩写匹配）
    """
    text = text.strip().lower()
    if not text:
        return []

    # 快速检测是否包含中文
    has_cjk = contains_cjk(text)

    tokens: list[str] = []
    for word in jieba.cut_for_search(text):
        word = word.strip().lower()
        if not word:
            continue
        if contains_cjk(word):
            if len(word) == 1:
                tokens.append(word)
            else:
                tokens.append(word)
                for i in range(len(word) - 1):
                    tokens.append(word[i:i + 2])
        else:
            cleaned = "".join(ch for ch in word if ch.isalnum() or ch in "._-+/")
            if cleaned:
                tokens.append(cleaned)

    # 对英文文本补充空白分词，弥补 jieba 对纯英文分词过粗的缺陷
    if not has_cjk or any(ch.isascii() and ch.isalpha() for ch in text):
        eng_tokens = _tokenize_english(text)
        for t in eng_tokens:
            if t not in tokens:
                tokens.append(t)

    return tokens


_STRUCTURE_MARKER_RE = re.compile(r"<PIC>|\[\[PIC:[^\]]+\]\]", re.IGNORECASE)
_ASCII_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[._+/-][a-z0-9]+)*", re.IGNORECASE)
_MAX_CLEAN_TOKEN_CHARS = 64


def tokenize_mixed_clean(text: str) -> list[str]:
    """实验用清理分词：去结构噪声和同位置重复，同时保留真实词频。"""
    text = _STRUCTURE_MARKER_RE.sub(" ", (text or "").strip().lower())
    if not text:
        return []

    positioned: list[tuple[int, int, str]] = []
    seen_positions: set[tuple[int, int, str]] = set()
    for word, start, end in jieba.tokenize(text, mode="search"):
        word = word.strip().lower()
        if not word or not contains_cjk(word):
            continue
        candidates = [(start, end, word)]
        if len(word) > 1:
            candidates.extend(
                (start + i, start + i + 2, word[i:i + 2])
                for i in range(len(word) - 1)
            )
        for item in candidates:
            if item not in seen_positions and len(item[2]) <= _MAX_CLEAN_TOKEN_CHARS:
                seen_positions.add(item)
                positioned.append(item)

    for match in _ASCII_TOKEN_RE.finditer(text):
        token = match.group(0).lower()
        if len(token) <= _MAX_CLEAN_TOKEN_CHARS:
            positioned.append((match.start(), match.end(), token))

    positioned.sort(key=lambda item: (item[0], item[1]))
    return [token for _, _, token in positioned]



@dataclass
class SearchResult:
    """检索结果的统一返回结构。

    chunk_id/source 保留召回单元信息；text/pics 通常来自完整 parent section，使 agent 看到同一主题下的正文、警告和图片锚点，而不是只看到命中的短 chunk。
    """
    chunk_id: int
    product: str
    heading: str
    text: str
    pics: list[str]
    score: float
    source: dict


class EmbeddingClient:
    """硅基流动 OpenAI-compatible embedding 客户端。

    只封装 /embeddings 调用和返回顺序恢复；并发、分片、mean-pooling 和索引构建由 RetrievalEngine 统一控制。
    """
    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = DEFAULT_API_KEY,
        timeout: int = 300,
    ) -> None:
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

    def embed_texts(
        self,
        texts: list[str],
        model: str,
        timing_trace: dict | None = None,
    ) -> list[list[float]]:
        if not texts:
            return []
        check_cancelled()

        started = time.perf_counter()
        if timing_trace is not None:
            timing_trace.update({"input_count": len(texts), "http_status": None})
        try:
            response = self._session().post(
                f"{self.base_url}/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "input": texts,
                },
                timeout=remaining_timeout(self.timeout),
            )
            if timing_trace is not None:
                timing_trace["http_status"] = getattr(response, "status_code", None)
            response.raise_for_status()
            payload = response.json()
            data = sorted(payload["data"], key=lambda item: item["index"])
            return [item["embedding"] for item in data]
        except Exception as exc:
            if timing_trace is not None:
                timing_trace["error_type"] = type(exc).__name__
                response = getattr(exc, "response", None)
                if response is not None:
                    timing_trace["http_status"] = getattr(response, "status_code", None)
            raise
        finally:
            if timing_trace is not None:
                timing_trace["http_elapsed"] = round(time.perf_counter() - started, 6)


class RetrievalEngine:
    """手册 RAG 的统一检索后端。

    构建 BM25 与 FAISS dense 两套索引，查询时按产品范围召回候选、用 RRF 融合稀疏/稠密结果，再用 reranker 精排。默认 chunk 负责定位，最终返回 parent section，兼顾召回精度和证据完整性。
    """
    def __init__(
        self,
        retrieval_chunks_path: Path = RETRIEVAL_CHUNKS_PATH,
        section_chunks_path: Path = SECTION_CHUNKS_PATH,
        catalog_path: Path = CATALOG_PATH,
        index_dir: Path = INDEX_DIR,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = DEFAULT_API_KEY,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        rerank_base_url: str = DEFAULT_RERANK_BASE_URL,
        rerank_api_key: str = DEFAULT_RERANK_API_KEY,
        rerank_model: str = DEFAULT_RERANK_MODEL,
        rerank_enabled: bool = DEFAULT_RERANK_ENABLED,
        rerank_candidate_pool: int = DEFAULT_RERANK_CANDIDATE_POOL,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        text_query_embed_mode: str | None = None,
    ) -> None:
        self.retrieval_chunks_path = Path(retrieval_chunks_path)
        self.section_chunks_path = Path(section_chunks_path)
        self.catalog_path = Path(catalog_path)
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.faiss_path = self.index_dir / FAISS_PATH.name
        self.metadata_path = self.index_dir / METADATA_PATH.name

        self.embedding_model = embedding_model
        self.max_concurrency = max(1, max_concurrency)
        self.text_query_embed_mode = (text_query_embed_mode or DEFAULT_TEXT_QUERY_EMBED_MODE).strip().lower()
        if self.text_query_embed_mode not in {"serial", "batch"}:
            raise ValueError("TEXT_QUERY_EMBED_MODE must be 'serial' or 'batch'")
        self.client = EmbeddingClient(base_url=base_url, api_key=api_key)
        self.rerank_enabled = rerank_enabled
        self.rerank_candidate_pool = max(1, rerank_candidate_pool)
        self.rerank_client = RerankClient(
            base_url=rerank_base_url,
            api_key=rerank_api_key,
            model=rerank_model,
        )

        self.retrieval_chunks: list[dict] = []
        self.section_chunks: list[dict] = []
        self.catalog: dict = {}
        self.section_lookup: dict[tuple[str, int], dict] = {}

        self.search_texts: list[str] = []
        self.bm25: BM25Okapi | None = None
        self.tokenized_docs: list[list[str]] = []

        self.dense_index: faiss.Index | None = None
        self.dense_vectors: np.ndarray | None = None
        self.visual_engine = None

    def load_documents(self) -> None:
        with open(self.retrieval_chunks_path, "r", encoding="utf-8") as f:
            self.retrieval_chunks = json.load(f)
        with open(self.section_chunks_path, "r", encoding="utf-8") as f:
            self.section_chunks = json.load(f)
        # 图文合一 LLM 章节总结（gen_section_summaries.py 产物）：挂到 llm_summary，
        # 供章节元数据、审计和后续检索上下文使用，旧截断 summary 仅作兜底。
        summ_path = self.section_chunks_path.parent / "section_summaries.json"
        if summ_path.exists():
            try:
                llm_summaries = json.loads(summ_path.read_text(encoding="utf-8"))
                for section in self.section_chunks:
                    s = llm_summaries.get(f"{section['product']}|{section['section_id']}", "")
                    if s and not s.startswith("__ERROR__"):
                        section["llm_summary"] = s
            except Exception:
                pass
        with open(self.catalog_path, "r", encoding="utf-8") as f:
            self.catalog = json.load(f)

        self.section_lookup = {
            (section["product"], section["section_id"]): section
            for section in self.section_chunks
        }
        self.search_texts = [build_searchable_text(chunk) for chunk in self.retrieval_chunks]

        # 产品路由索引：product_name -> [chunk_id, ...]
        self.product_chunk_ids: dict[str, list[int]] = {}
        for i, chunk in enumerate(self.retrieval_chunks):
            product = chunk["product"]
            if product not in self.product_chunk_ids:
                self.product_chunk_ids[product] = []
            self.product_chunk_ids[product].append(i)

    def build_index(self, batch_size: int = 32) -> None:
        self.load_documents()

        print(f"构建 BM25 索引: {len(self.search_texts)} 文档")
        self.tokenized_docs = [
            tokenize_mixed(text)
            for text in tqdm(self.search_texts, desc="BM25 分词", unit="doc")
        ]
        self.bm25 = BM25Okapi(self.tokenized_docs)

        print(
            f"构建 dense 向量索引: {len(self.search_texts)} 文档, "
            f"batch_size={batch_size}, concurrency={self.max_concurrency}"
        )
        dense_vectors = self._embed_corpus(self.search_texts, self.embedding_model, batch_size)
        dense_vectors = self._l2_normalize(dense_vectors)

        self.dense_vectors = dense_vectors

        dimension = dense_vectors.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(dense_vectors)
        self.dense_index = index

        faiss.write_index(index, str(self.faiss_path))
        with open(self.metadata_path, "wb") as f:
            pickle.dump(
                {
                    "embedding_model": self.embedding_model,
                    "retrieval_chunks": self.retrieval_chunks,
                    "section_chunks": self.section_chunks,
                    "catalog": self.catalog,
                    "search_texts": self.search_texts,
                    "tokenized_docs": self.tokenized_docs,
                    "dense_vectors": self.dense_vectors,
                },
                f,
            )

    def load_index(self) -> None:
        if not self.faiss_path.exists() or not self.metadata_path.exists():
            raise FileNotFoundError("索引文件不存在，请先运行 build_index。")

        with open(self.metadata_path, "rb") as f:
            data = pickle.load(f)

        self.embedding_model = data["embedding_model"]
        self.retrieval_chunks = data["retrieval_chunks"]
        self.section_chunks = data["section_chunks"]
        # 缓存里的 section_chunks 不含 llm_summary（索引早于总结生成），这里补挂。
        summ_path = self.section_chunks_path.parent / "section_summaries.json"
        if summ_path.exists():
            try:
                llm_summaries = json.loads(summ_path.read_text(encoding="utf-8"))
                for section in self.section_chunks:
                    s = llm_summaries.get(f"{section['product']}|{section['section_id']}", "")
                    if s and not s.startswith("__ERROR__"):
                        section["llm_summary"] = s
            except Exception:
                pass
        self.catalog = data["catalog"]
        self.search_texts = data["search_texts"]
        self.tokenized_docs = data["tokenized_docs"]
        self.dense_vectors = data["dense_vectors"]

        self.section_lookup = {
            (section["product"], section["section_id"]): section
            for section in self.section_chunks
        }
        self.bm25 = BM25Okapi(self.tokenized_docs)
        self.dense_index = faiss.read_index(str(self.faiss_path))

        self.product_chunk_ids = {}
        for i, chunk in enumerate(self.retrieval_chunks):
            product = chunk["product"]
            if product not in self.product_chunk_ids:
                self.product_chunk_ids[product] = []
            self.product_chunk_ids[product].append(i)

    def ensure_index(self) -> None:
        if self.dense_index is not None and self.bm25 is not None:
            return
        self.load_index()

    def search_manual(
        self,
        keywords: list[str],
        *,
        semantic_query: str = "",
        original_query: str = "",
        top_k: int = 8,
        products: list[str] | None = None,
        query_images: list[str] | None = None,
        visual_query_text: str = "",
        visual_products: list[str] | None = None,
        precomputed_visual_hits: list[list] | None = None,
        original_text_query: str = "",
        original_text_products: list[str] | None = None,
        timing_trace: dict | None = None,
    ) -> tuple[list[SearchResult], int]:
        """Run text and optional visual retrieval, preserving the old fallback."""
        from visual_retrieval import has_visual_intent, visual_retrieval_enabled

        search_started = time.perf_counter()
        images = list(query_images or [])
        visual_query = (visual_query_text or semantic_query or original_query or " ".join(keywords)).strip()
        if not visual_retrieval_enabled() or (not images and not has_visual_intent(visual_query)):
            return self._search_text_manual(
                keywords,
                semantic_query=semantic_query,
                original_query=original_query,
                top_k=top_k,
                products=products,
                timing_trace=timing_trace,
            )
        try:
            visual = self._get_visual_engine()
            visual.ensure_index()
        except RequestCancelled:
            raise
        except Exception as exc:
            if os.getenv("VISUAL_RETRIEVAL_STRICT", "0").lower() in {"1", "true", "yes"}:
                raise
            print(f"视觉索引不可用，回退到现有文本链路: {exc}")
            return self._search_text_manual(
                keywords,
                semantic_query=semantic_query,
                original_query=original_query,
                top_k=top_k,
                products=products,
                timing_trace=timing_trace,
            )

        expanded_text_trace: dict | None = {} if timing_trace is not None else None
        original_text_trace: dict | None = {} if timing_trace is not None and original_text_query.strip() else None
        if timing_trace is not None:
            timing_trace.update({
                "mode": "visual",
                "expanded_text": expanded_text_trace,
                "original_text": original_text_trace,
            })

        def collect_visual_hits() -> tuple[list, list[list]]:
            text_hits = visual.search_text(visual_query, products=visual_products) if visual_query else []
            image_hits = list(precomputed_visual_hits or [])
            if precomputed_visual_hits is None:
                image_hits = [visual.search_image(image, products=visual_products) for image in images]
            return text_hits, image_hits

        keep_original = bool(original_text_query.strip() and original_text_query.strip() != visual_query)
        with ThreadPoolExecutor(max_workers=3 if keep_original else 2) as pool:
            text_future = pool.submit(
                copy_context().run,
                self._search_text_manual,
                keywords,
                semantic_query=semantic_query,
                original_query=original_query,
                top_k=top_k,
                products=products,
                timing_trace=expanded_text_trace,
            )
            visual_future = pool.submit(copy_context().run, collect_visual_hits)
            original_future = None
            if keep_original:
                original_future = pool.submit(
                    copy_context().run,
                    self._search_text_manual,
                    [],
                    semantic_query=original_text_query,
                    original_query=original_text_query,
                    top_k=top_k,
                    products=original_text_products,
                    timing_trace=original_text_trace,
                )
            results, filtered = text_future.result()
            original_text_results = original_future.result()[0] if original_future is not None else []
            try:
                visual_text_hits, visual_image_hits = visual_future.result()
            except RequestCancelled:
                raise
            except Exception as exc:
                if os.getenv("VISUAL_RETRIEVAL_STRICT", "0").lower() in {"1", "true", "yes"}:
                    raise
                print(f"视觉召回不可用，回退到现有文本链路: {exc}")
                return original_text_results or results, filtered
        try:
            merged = self._merge_visual_results(
                results,
                query=visual_query,
                images=images,
                products=visual_products,
                top_k=top_k,
                visual_text_hits=visual_text_hits,
                visual_image_hits=visual_image_hits,
                original_text_results=original_text_results,
            )
            if timing_trace is not None:
                timing_trace["total_elapsed"] = round(time.perf_counter() - search_started, 6)
            return merged, filtered
        except RequestCancelled:
            raise
        except Exception as exc:
            if os.getenv("VISUAL_RETRIEVAL_STRICT", "0").lower() in {"1", "true", "yes"}:
                raise
            print(f"视觉融合不可用，回退到现有文本链路: {exc}")
            return original_text_results or results, filtered

    def _search_text_manual(
        self,
        keywords: list[str],
        *,
        semantic_query: str = "",
        original_query: str = "",
        top_k: int = 8,
        products: list[str] | None = None,
        timing_trace: dict | None = None,
    ) -> tuple[list[SearchResult], int]:
        """统一检索入口：BM25 20 + 向量 20 + 逐关键词各5 → RRF候选池 → rerank → parent section去重。
        返回 (结果列表, 被过滤数量)；当前不再使用固定 rerank 分数阈值。
        """
        search_started = time.perf_counter()
        if timing_trace is not None:
            timing_trace.update({"mode": "text", "keyword_count": len(keywords)})
        ranked_lists, keyword_rerank_query = self._collect_text_recall_rankings(
            keywords,
            semantic_query=semantic_query,
            original_query=original_query,
            products=products,
            timing_trace=timing_trace,
        )
        rrf_started = time.perf_counter()
        candidates = self._rrf_merge(ranked_lists, top_n=self.rerank_candidate_pool)
        if timing_trace is not None:
            timing_trace["rrf"] = {
                "elapsed": round(time.perf_counter() - rrf_started, 6),
                "ranked_list_count": len(ranked_lists),
                "input_items": sum(len(ranked) for ranked in ranked_lists),
                "candidate_count": len(candidates),
            }
        if not candidates:
            if timing_trace is not None:
                timing_trace["keyword_rerank"] = {"executed": False, "reason": "empty_candidates"}
                timing_trace["original_rerank"] = {"executed": False, "reason": "empty_candidates"}
                timing_trace["parent_build"] = {"elapsed": 0.0, "returned_sections": 0}
                timing_trace["total_elapsed"] = round(time.perf_counter() - search_started, 6)
            return [], 0

        original_query = (original_query or "").strip()
        keyword_rerank_trace: dict | None = {} if timing_trace is not None else None
        keyword_top = self._rerank_candidates(
            keyword_rerank_query,
            candidates,
            top_n=6,
            timing_trace=keyword_rerank_trace,
        )[:6]
        if timing_trace is not None:
            timing_trace["keyword_rerank"] = keyword_rerank_trace

        user_top: list[int] = []
        if original_query and original_query != keyword_rerank_query:
            original_rerank_trace: dict | None = {} if timing_trace is not None else None
            user_top = self._rerank_candidates(
                original_query,
                candidates,
                top_n=4,
                timing_trace=original_rerank_trace,
            )[:4]
            if timing_trace is not None:
                timing_trace["original_rerank"] = original_rerank_trace
        elif timing_trace is not None:
            timing_trace["original_rerank"] = {"executed": False, "reason": "same_or_empty_query"}

        seen: set[int] = set()
        final_ids: list[int] = []
        for doc_id in keyword_top + user_top:
            heading_key = self._result_dedup_key(doc_id)
            if heading_key in seen:
                continue
            seen.add(heading_key)
            final_ids.append(doc_id)

        build_started = time.perf_counter()
        results = self._build_results(final_ids[:top_k])
        if timing_trace is not None:
            timing_trace["parent_build"] = {
                "elapsed": round(time.perf_counter() - build_started, 6),
                "input_ids": len(final_ids[:top_k]),
                "returned_sections": len(results),
            }
            timing_trace["total_elapsed"] = round(time.perf_counter() - search_started, 6)
        return results, 0

    def _collect_text_recall_rankings(
        self,
        keywords: list[str],
        *,
        semantic_query: str = "",
        original_query: str = "",
        products: list[str] | None = None,
        timing_trace: dict | None = None,
    ) -> tuple[list[list[int]], str]:
        """Return raw BM25/Dense rankings before local RRF and remote rerank."""
        ensure_started = time.perf_counter()
        self.ensure_index()
        if timing_trace is not None:
            timing_trace["ensure_index_elapsed"] = round(time.perf_counter() - ensure_started, 6)
        sparse_query = " ".join(keyword.strip() for keyword in keywords if keyword.strip())
        original_query = (original_query or "").strip()
        semantic_query = (semantic_query or "").strip()

        dense_query = semantic_query or sparse_query
        if not sparse_query and not dense_query:
            if timing_trace is not None:
                timing_trace["recall_status"] = "empty_query"
            return [], ""

        recall_n = 20
        per_keyword_recall_n = 5
        allowed_doc_ids: list[int] | None = None
        if products:
            allowed: set[int] = set()
            for product in products:
                allowed.update(self.product_chunk_ids.get(product, []))
            allowed_doc_ids = sorted(allowed)

        main_sparse_started = time.perf_counter()
        sparse_doc_ids = self._sparse_recall(
            sparse_query or dense_query,
            top_n=recall_n,
            allowed_doc_ids=allowed_doc_ids,
        )
        if timing_trace is not None:
            timing_trace["main_sparse"] = {
                "elapsed": round(time.perf_counter() - main_sparse_started, 6),
                "candidate_count": len(sparse_doc_ids),
            }
        keyword_phrases = [
            keyword.strip()
            for keyword in keywords
            if keyword and keyword.strip()
        ]
        extra_sparse_doc_ids: list[int] = []
        keyword_sparse_calls: list[dict] = []
        keyword_sparse_started = time.perf_counter()
        for index, phrase in enumerate(keyword_phrases, start=1):
            sparse_started = time.perf_counter()
            sparse_ids = self._sparse_recall(
                phrase,
                top_n=per_keyword_recall_n,
                allowed_doc_ids=allowed_doc_ids,
            )
            extra_sparse_doc_ids.extend(sparse_ids)
            keyword_sparse_calls.append({
                "index": index,
                "elapsed": round(time.perf_counter() - sparse_started, 6),
                "candidate_count": len(sparse_ids),
            })
        if timing_trace is not None:
            timing_trace["keyword_sparse"] = {
                "elapsed": round(time.perf_counter() - keyword_sparse_started, 6),
                "call_count": len(keyword_sparse_calls),
                "calls": keyword_sparse_calls,
            }

        main_dense_query = dense_query if dense_query else sparse_query
        extra_dense_doc_ids: list[int] = []
        keyword_dense_calls: list[dict] = []
        if self.text_query_embed_mode == "batch":
            batch_trace: dict = {}
            batch_started = time.perf_counter()
            try:
                vectors = self.client.embed_texts(
                    [main_dense_query, *keyword_phrases],
                    self.embedding_model,
                    timing_trace=batch_trace,
                )
                if len(vectors) != 1 + len(keyword_phrases):
                    raise RuntimeError("batch embedding returned an unexpected vector count")
                dense_doc_ids = self._dense_recall_from_vector(vectors[0], recall_n, allowed_doc_ids)
                for index, vector in enumerate(vectors[1:], start=1):
                    dense_ids = self._dense_recall_from_vector(vector, per_keyword_recall_n, allowed_doc_ids)
                    extra_dense_doc_ids.extend(dense_ids)
                    keyword_dense_calls.append({
                        "index": index,
                        "input_count": 1,
                        "status": "ok",
                        "candidate_count": len(dense_ids),
                        "elapsed": 0.0,
                        "shared_batch": True,
                    })
                batch_trace["status"] = "ok"
            except Exception as exc:
                batch_trace.update({"status": "error", "error_type": type(exc).__name__})
                raise
            finally:
                batch_elapsed = round(time.perf_counter() - batch_started, 6)
                batch_trace["elapsed"] = batch_elapsed
                if timing_trace is not None:
                    timing_trace["dense_batch"] = batch_trace
            main_dense_trace = {
                "mode": "batch",
                "input_count": len(vectors),
                "status": "ok",
                "candidate_count": len(dense_doc_ids),
                "elapsed": batch_elapsed,
                "http_status": batch_trace.get("http_status"),
            }
            keyword_dense_elapsed = 0.0
        else:
            main_dense_trace = {"mode": "serial", "input_count": 1}
            main_dense_started = time.perf_counter()
            try:
                dense_doc_ids = self._dense_recall(
                    main_dense_query,
                    top_n=recall_n,
                    allowed_doc_ids=allowed_doc_ids,
                    timing_trace=main_dense_trace,
                )
                main_dense_trace.update({"status": "ok", "candidate_count": len(dense_doc_ids)})
            except Exception as exc:
                main_dense_trace.update({"status": "error", "error_type": type(exc).__name__})
                raise
            finally:
                main_dense_trace["elapsed"] = round(time.perf_counter() - main_dense_started, 6)

            keyword_dense_elapsed = 0.0
            for index, phrase in enumerate(keyword_phrases, start=1):
                dense_call = {"index": index, "input_count": 1}
                dense_started = time.perf_counter()
                try:
                    dense_ids = self._dense_recall(
                        phrase,
                        top_n=per_keyword_recall_n,
                        allowed_doc_ids=allowed_doc_ids,
                        timing_trace=dense_call,
                    )
                    extra_dense_doc_ids.extend(dense_ids)
                    dense_call.update({"status": "ok", "candidate_count": len(dense_ids)})
                except Exception as exc:
                    dense_call.update({"status": "error", "error_type": type(exc).__name__})
                    raise
                finally:
                    elapsed = time.perf_counter() - dense_started
                    keyword_dense_elapsed += elapsed
                    dense_call["elapsed"] = round(elapsed, 6)
                    keyword_dense_calls.append(dense_call)

        if timing_trace is not None:
            timing_trace["text_query_embed_mode"] = self.text_query_embed_mode
            timing_trace["main_dense"] = main_dense_trace
            timing_trace["keyword_dense"] = {
                "elapsed": round(keyword_dense_elapsed, 6),
                "call_count": len(keyword_dense_calls),
                "calls": keyword_dense_calls,
            }

        if not products:
            reorder_query = original_query or semantic_query or sparse_query
            sparse_doc_ids = self._reorder_by_lang(reorder_query, sparse_doc_ids)
            dense_doc_ids = self._reorder_by_lang(reorder_query, dense_doc_ids)

        if not products:
            reorder_query = original_query or semantic_query or sparse_query
            extra_sparse_doc_ids = self._reorder_by_lang(reorder_query, extra_sparse_doc_ids)
            extra_dense_doc_ids = self._reorder_by_lang(reorder_query, extra_dense_doc_ids)

        return [
            list(dict.fromkeys(sparse_doc_ids)),
            list(dict.fromkeys(dense_doc_ids)),
            list(dict.fromkeys(extra_sparse_doc_ids)),
            list(dict.fromkeys(extra_dense_doc_ids)),
        ], semantic_query or sparse_query

    def _get_visual_engine(self):
        if self.visual_engine is None:
            from visual_retrieval import VisualRetrievalEngine

            self.visual_engine = VisualRetrievalEngine()
        return self.visual_engine

    def prepare_visual_images(self, images: list[str], products: list[str] | None = None) -> list[list]:
        """Embed user images once so the later tool call can reuse their hits."""
        visual = self._get_visual_engine()
        visual.ensure_index()
        return [visual.search_image(image, products=products) for image in images]

    @staticmethod
    def _section_key(result: SearchResult) -> str:
        return f"{result.product}|{int(result.source['parent_section_id'])}"

    def _result_from_section(self, section_key: str) -> SearchResult | None:
        product, separator, raw_section_id = section_key.rpartition("|")
        if not separator:
            return None
        try:
            section_id = int(raw_section_id)
        except ValueError:
            return None
        section = self.section_lookup.get((product, section_id))
        if section is None:
            return None
        return SearchResult(
            chunk_id=-1,
            product=product,
            heading=section.get("heading", ""),
            text=section.get("text", ""),
            pics=list(section.get("pics") or []),
            score=0.0,
            source={
                "matched_chunk_id": None,
                "matched_chunk_text": "",
                "matched_chunk_pics": [],
                "matched_subchunk_id": None,
                "matched_split_kind": "visual",
                "parent_section_id": section_id,
                "source_section_ids": [section_id],
                "section_summary": section.get("summary", ""),
                "section_tags": section.get("tags", []),
                "section_text": section.get("text", ""),
                "section_pics": list(section.get("pics") or []),
                "section_heading": section.get("heading", ""),
                "return_mode": "parent_section",
            },
        )

    def _merge_visual_results(
        self,
        text_results: list[SearchResult],
        *,
        query: str,
        images: list[str],
        products: list[str] | None,
        top_k: int,
        visual_text_hits: list | None = None,
        visual_image_hits: list[list] | None = None,
        original_text_results: list[SearchResult] | None = None,
    ) -> list[SearchResult]:
        """Fuse text/image retrieval at parent-section granularity."""
        from visual_retrieval import (
            ROOT,
            VISUAL_FINAL_TOP_N,
            VISUAL_RECALL_TOP_N,
            VISUAL_SECTION_POOL,
            file_to_data_url,
            rrf_rank,
        )

        visual = self._get_visual_engine()
        visual.ensure_index()
        original_text_results = list(original_text_results or [])
        original_text_keys = list(dict.fromkeys(self._section_key(result) for result in original_text_results))
        text_keys = list(dict.fromkeys(self._section_key(result) for result in text_results))
        result_by_key = {self._section_key(result): result for result in original_text_results}
        for result in text_results:
            result_by_key.setdefault(self._section_key(result), result)
        ranked_lists: list[list[str]] = []
        if original_text_keys:
            ranked_lists.append(original_text_keys)
        if text_keys:
            ranked_lists.append(text_keys)
        base_list_count = len(ranked_lists)
        matched_images: dict[str, list] = {}
        channels: dict[str, set[str]] = {key: {"text_original"} for key in original_text_keys}
        for key in text_keys:
            channels.setdefault(key, set()).update({"bm25", "text_dense", "vision_text"})

        def add_hits(hits, channel: str) -> None:
            keys: list[str] = []
            seen: set[str] = set()
            for hit in hits:
                for key in hit.section_keys:
                    if key not in seen:
                        seen.add(key)
                        keys.append(key)
                    matched_images.setdefault(key, []).append(hit)
                    channels.setdefault(key, set()).add(channel)
            if keys:
                ranked_lists.append(keys)

        if query:
            add_hits(
                visual_text_hits if visual_text_hits is not None else visual.search_text(
                    query, top_n=VISUAL_RECALL_TOP_N, products=products
                ),
                "visual_text",
            )
        if visual_image_hits is None:
            visual_image_hits = [
                visual.search_image(image, top_n=VISUAL_RECALL_TOP_N, products=products)
                for image in images
            ]
        for hits in visual_image_hits:
            add_hits(hits, "visual_image")
        if len(ranked_lists) == base_list_count:
            return original_text_results or text_results

        candidate_keys, rrf_scores = rrf_rank(ranked_lists, top_n=VISUAL_SECTION_POOL)
        for key in candidate_keys:
            result = result_by_key.get(key) or self._result_from_section(key)
            if result is None:
                continue
            result_by_key[key] = result
            unique_hits = []
            seen_images: set[str] = set()
            for hit in matched_images.get(key, []):
                if hit.image_id not in seen_images:
                    seen_images.add(hit.image_id)
                    unique_hits.append(hit)
            result.source["retrieval_channels"] = sorted(channels.get(key, set()))
            result.source["matched_visual_images"] = [hit.image_id for hit in unique_hits]
            result.source["section_rrf_score"] = float(rrf_scores.get(key, 0.0))
            result.source["vl_rerank_score"] = 0.0
            result.source["vl_rerank_fallback"] = False

        candidate_keys = [key for key in candidate_keys if key in result_by_key]
        reranked_lists: list[list[str]] = []
        if original_text_keys:
            reranked_lists.append([key for key in original_text_keys if key in result_by_key])
        if text_keys:
            reranked_lists.append([key for key in text_keys if key in result_by_key])
        rerank_scores: dict[str, float] = {}
        rerank_failed = False
        visual_doc_keys = list(candidate_keys)
        if visual_doc_keys:
            documents: list[str | dict[str, str]] = []
            for key in visual_doc_keys:
                result = result_by_key[key]
                text = "\n".join(
                    part for part in (
                        result.product,
                        result.heading,
                        str(result.source.get("section_summary") or ""),
                        result.text[:900],
                    ) if part
                )
                hits = matched_images.get(key) or []
                if hits:
                    documents.append({"text": text, "image": file_to_data_url(ROOT / hits[0].relative_path)})
                else:
                    documents.append({"text": text})

            # SiliconFlow's probed contract accepts mixed text+image objects.
            # Multiple user images are still reranked separately, then fused.
            queries: list[str | dict[str, str]] = (
                [{"text": query, "image": image} for image in images]
                if images else [query]
            )
            for rerank_query in queries:
                try:
                    ranked = visual.rerank_client.rerank(
                        rerank_query,
                        documents,
                        top_n=min(VISUAL_FINAL_TOP_N, len(documents)),
                    )
                except RequestCancelled:
                    raise
                except Exception as exc:
                    if os.getenv("VISUAL_RETRIEVAL_STRICT", "0").lower() in {"1", "true", "yes"}:
                        raise
                    rerank_failed = True
                    print(f"视觉精排不可用，本路回退Section RRF: {exc}")
                    continue
                keys = [visual_doc_keys[index] for index, _score in ranked if 0 <= index < len(visual_doc_keys)]
                if keys:
                    reranked_lists.append(keys)
                for index, score in ranked:
                    if 0 <= index < len(visual_doc_keys):
                        key = visual_doc_keys[index]
                        rerank_scores[key] = max(rerank_scores.get(key, float("-inf")), score)

        if len(reranked_lists) > base_list_count:
            final_keys, _ = rrf_rank(reranked_lists, top_n=top_k)
            final_keys.extend(key for key in candidate_keys if key not in final_keys)
        else:
            final_keys = candidate_keys

        results: list[SearchResult] = []
        for rank, key in enumerate(final_keys[:top_k], start=1):
            result = result_by_key[key]
            result.score = 1.0 / rank
            if key in rerank_scores:
                result.source["vl_rerank_score"] = float(rerank_scores[key])
            result.source["vl_rerank_fallback"] = rerank_failed
            results.append(result)
        return results

    def keyword_search(
        self,
        keywords: list[str],
        top_k: int = 8,
        products: list[str] | None = None,
        semantic_query: str = "",
    ) -> tuple[list[SearchResult], int]:
        """兼容旧接口：内部复用统一检索。"""
        return self.search_manual(
            keywords,
            semantic_query=semantic_query,
            original_query="",
            top_k=top_k,
            products=products,
        )

    def vector_search(
        self,
        query: str,
        top_k: int = 8,
        products: list[str] | None = None,
    ) -> tuple[list[SearchResult], int]:
        """兼容旧接口：内部复用统一检索。"""
        keywords = tokenize_mixed(query)
        return self.search_manual(
            keywords,
            semantic_query=query,
            original_query="",
            top_k=top_k,
            products=products,
        )

    def _filter_by_products(self, doc_ids: list[int], products: list[str]) -> list[int]:
        """按产品名过滤候选 chunk。"""
        allowed = set()
        for p in products:
            allowed.update(self.product_chunk_ids.get(p, []))
        return [doc_id for doc_id in doc_ids if doc_id in allowed]

    def _reorder_by_lang(self, query: str, doc_ids: list[int]) -> list[int]:
        """全库召回时按问题语言重排：同语言 chunk 优先，跨语言保留兜底。

        判定语言用产品名而非 chunk lang 字段（后者不可靠 — Earphones 等英文产品被标 zh）。
        中文产品名以"手册"结尾；其余视为英文产品。
        软优先：稳定按 (lang_match_priority, original_rank) 排序，rerank 仍能让
        跨语言的高相关章节回到 top。
        """
        if not doc_ids:
            return doc_ids
        question_is_zh = contains_cjk(query)
        scored: list[tuple[int, int, int]] = []
        for rank, doc_id in enumerate(doc_ids):
            product = self.retrieval_chunks[doc_id].get("product", "")
            product_is_zh = product.endswith("手册")
            priority = 0 if product_is_zh == question_is_zh else 1
            scored.append((priority, rank, doc_id))
        scored.sort(key=lambda x: (x[0], x[1]))
        return [doc_id for _, _, doc_id in scored]

    def _build_results(self, doc_ids: list[int]) -> list[SearchResult]:
        """把命中的 chunk id 转成 agent 可读证据。

        默认返回完整 parent section 的正文和图片，同时在 source 中保留实际命中的 chunk，方便 trace 解释“为什么召回到这一节”。
        """
        results = []
        for rank, doc_id in enumerate(doc_ids, start=1):
            chunk = self.retrieval_chunks[doc_id]
            section = self.section_lookup.get((chunk["product"], chunk["parent_section_id"]))

            use_parent = RETURN_PARENT_SECTION and section is not None
            return_text = section["text"] if use_parent else chunk["text"]
            return_pics = section.get("pics", []) if use_parent else chunk["pics"]
            return_heading = section.get("heading", chunk["heading"]) if use_parent else chunk["heading"]

            score = 1.0 / rank
            results.append(
                SearchResult(
                    chunk_id=doc_id,
                    product=chunk["product"],
                    heading=return_heading,
                    text=return_text,
                    pics=return_pics,
                    score=score,
                    source={
                        "matched_chunk_id": chunk.get("chunk_id", doc_id),
                        "matched_chunk_text": chunk["text"],
                        "matched_chunk_pics": chunk["pics"],
                        "matched_subchunk_id": chunk.get("subchunk_id"),
                        "matched_split_kind": chunk.get("split_kind"),
                        "parent_section_id": chunk["parent_section_id"],
                        "source_section_ids": chunk.get("source_section_ids", []),
                        "section_summary": section["summary"] if section else "",
                        "section_tags": section["tags"] if section else [],
                        "section_text": section["text"] if section else "",
                        "section_pics": (section.get("pics") or []) if section else [],
                        "section_heading": (section.get("heading") or "") if section else "",
                        "return_mode": "parent_section" if use_parent else "matched_chunk",
                    },
                )
            )
        return results

    def _sparse_recall(self, query: str, top_n: int, allowed_doc_ids: list[int] | None = None) -> list[int]:
        assert self.bm25 is not None
        query_tokens = tokenize_mixed(query)
        if not query_tokens:
            return []
        scores = self.bm25.get_scores(query_tokens)
        if allowed_doc_ids is not None:
            ranked_ids = sorted(
                allowed_doc_ids,
                key=lambda idx: float(scores[idx]),
                reverse=True,
            )
        else:
            ranked_ids = [int(idx) for idx in np.argsort(scores)[::-1]]
        return ranked_ids[:top_n]

    def _dense_recall(
        self,
        query: str,
        top_n: int,
        allowed_doc_ids: list[int] | None = None,
        timing_trace: dict | None = None,
    ) -> list[int]:
        assert self.dense_index is not None
        assert self.dense_vectors is not None
        if timing_trace is None:
            query_vector = self.client.embed_texts([query], self.embedding_model)[0]
        else:
            query_vector = self.client.embed_texts(
                [query],
                self.embedding_model,
                timing_trace=timing_trace,
            )[0]
        return self._dense_recall_from_vector(query_vector, top_n, allowed_doc_ids)

    def _dense_recall_from_vector(
        self,
        query_vector: list[float] | np.ndarray,
        top_n: int,
        allowed_doc_ids: list[int] | None = None,
    ) -> list[int]:
        """Search the existing index with a precomputed query vector."""
        assert self.dense_index is not None
        assert self.dense_vectors is not None
        query_array = self._l2_normalize(np.asarray([query_vector], dtype=np.float32))
        if allowed_doc_ids is not None:
            if not allowed_doc_ids:
                return []
            allowed_vectors = self.dense_vectors[np.asarray(allowed_doc_ids, dtype=np.int64)]
            scores = allowed_vectors @ query_array[0]
            top_idx = np.argsort(scores)[::-1][:top_n]
            return [int(allowed_doc_ids[int(i)]) for i in top_idx]
        _, indices = self.dense_index.search(query_array, top_n)
        return [int(idx) for idx in indices[0] if idx >= 0]

    def _rrf_merge(self, ranked_lists: list[list[int]], top_n: int, k: int = 60) -> list[int]:
        """Reciprocal Rank Fusion：融合 BM25 与 dense 的排名而不依赖分数同尺度。

        BM25 分数、向量相似度和不同产品子集的分布不可直接相加；RRF 只看名次，用于合并互补召回列表。
        """
        scores: dict[int, float] = {}
        first_seen: dict[int, tuple[int, int]] = {}
        for list_idx, ranked in enumerate(ranked_lists):
            for rank, doc_id in enumerate(ranked):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
                first_seen.setdefault(doc_id, (list_idx, rank))
        ordered = sorted(
            scores.items(),
            key=lambda item: (-item[1], first_seen[item[0]][0], first_seen[item[0]][1]),
        )
        return [doc_id for doc_id, _score in ordered[:top_n]]

    def _rerank_candidates(
        self,
        query: str,
        candidate_ids: list[int],
        top_n: int,
        timing_trace: dict | None = None,
    ) -> list[int]:
        """对 RRF 候选做 cross-encoder rerank，失败时可解释地回退原排序。

        rerank 是精排层，不改变召回池；线上偶发 5xx/超时时记录 fallback，避免单个上游错误中断整批提交。
        """
        if not candidate_ids:
            if timing_trace is not None:
                timing_trace.update({"executed": False, "reason": "empty_candidates"})
            return []
        if not self.rerank_enabled:
            if timing_trace is not None:
                timing_trace.update({
                    "executed": False,
                    "reason": "disabled",
                    "candidate_count": len(candidate_ids),
                    "returned_count": min(top_n, len(candidate_ids)),
                })
            return candidate_ids[:top_n]

        documents = [build_rerank_text(self.retrieval_chunks[doc_id]) for doc_id in candidate_ids]
        rerank_elapsed = None
        if timing_trace is not None:
            timing_trace.update({
                "executed": True,
                "candidate_count": len(candidate_ids),
                "document_count": len(documents),
                "top_n": top_n,
                "fallback": False,
            })
        try:
            t0 = time.time()
            ranked = self.rerank_client.rerank(query=query, documents=documents, top_n=min(top_n, len(documents)))
            rerank_elapsed = time.time() - t0
            if RERANK_TIMING_LOG_PATH:
                payload = {
                    "ts": time.time(),
                    "qid": get_current_qid(),
                    "query": query,
                    "top_n": top_n,
                    "candidate_count": len(candidate_ids),
                    "document_count": len(documents),
                    "elapsed": rerank_elapsed,
                }
                with _RERANK_FALLBACK_LOCK:
                    RERANK_TIMING_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
                    with RERANK_TIMING_LOG_PATH.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except RerankError as exc:
            print(f"rerank 不可用，回退到 RRF 排序: {exc}")
            if timing_trace is not None:
                timing_trace.update({
                    "elapsed": round(time.time() - t0, 6),
                    "fallback": True,
                    "error_type": type(exc).__name__,
                    "returned_count": min(top_n, len(candidate_ids)),
                })
            if RERANK_FALLBACK_LOG_PATH:
                payload = {
                    "ts": time.time(),
                    "qid": get_current_qid(),
                    "query": query,
                    "top_n": top_n,
                    "candidate_count": len(candidate_ids),
                    "error": str(exc),
                }
                with _RERANK_FALLBACK_LOCK:
                    RERANK_FALLBACK_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
                    with RERANK_FALLBACK_LOG_PATH.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            return candidate_ids[:top_n]

        reranked_ids: list[int] = []
        for item in ranked:
            if 0 <= item.index < len(candidate_ids):
                reranked_ids.append(candidate_ids[item.index])

        if len(reranked_ids) < top_n:
            seen = set(reranked_ids)
            reranked_ids.extend(doc_id for doc_id in candidate_ids if doc_id not in seen)

        result = reranked_ids[:top_n]
        if timing_trace is not None:
            timing_trace.update({
                "elapsed": round(rerank_elapsed or 0.0, 6),
                "returned_count": len(result),
            })
        return result

    def _apply_rerank_threshold(self, doc_ids: list[int], query: str) -> list[int]:
        """兼容旧调用点：固定阈值已停用，当前直接原样返回。"""
        return doc_ids

    def _result_dedup_key(self, doc_id: int) -> str:
        chunk = self.retrieval_chunks[doc_id]
        product = (chunk.get("product") or "").strip().lower()
        parent = str(chunk.get("parent_section_id") or "")
        return f"{product}::{parent}"

    def _embed_batch(self, batch_id: int, texts: list[str], model: str) -> tuple[int, list[list[float]]]:
        vectors = self.client.embed_texts(texts, model)
        return batch_id, vectors

    def _embed_corpus(self, texts: Iterable[str], model: str, batch_size: int) -> np.ndarray:
        texts = list(texts)

        # 切片展开：长文本（主要是含整表 caption_aux 的 chunk）按行切成 ≤380 字符的片，
        # 每片单独 embedding，最后按 owner 做 mean-pooling 还原为 1 chunk 1 向量。
        # 这样长 info_table 表格完整进 dense 向量（零删尾），且每片都在 ubatch 内不触发 500。
        seg_texts: list[str] = []
        owners: list[int] = []
        for i, text in enumerate(texts):
            segs = split_text_for_embedding(text) or [text]
            for s in segs:
                seg_texts.append(s)
                owners.append(i)

        batches: list[tuple[int, list[str]]] = []
        current_batch: list[str] = []
        for seg in seg_texts:
            current_batch.append(seg)
            if len(current_batch) >= batch_size:
                batches.append((len(batches), current_batch))
                current_batch = []
        if current_batch:
            batches.append((len(batches), current_batch))

        ordered_vectors: list[list[list[float]] | None] = [None] * len(batches)

        with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
            futures = {
                executor.submit(self._embed_batch, batch_id, batch_texts, model): batch_id
                for batch_id, batch_texts in batches
            }
            with tqdm(total=len(batches), desc="Embedding", unit="batch") as pbar:
                for future in as_completed(futures):
                    batch_id, vectors = future.result()
                    ordered_vectors[batch_id] = vectors
                    pbar.update(1)

        seg_vectors: list[list[float]] = []
        for batch_vectors in ordered_vectors:
            if batch_vectors is None:
                raise RuntimeError(f"{model} 存在未完成 batch，索引构建中断。")
            seg_vectors.extend(batch_vectors)

        seg_arr = np.asarray(seg_vectors, dtype=np.float32)
        dim = seg_arr.shape[1]
        out = np.zeros((len(texts), dim), dtype=np.float32)
        cnt = np.zeros(len(texts), dtype=np.float32)
        for owner, vec in zip(owners, seg_arr):
            out[owner] += vec
            cnt[owner] += 1.0
        cnt = np.clip(cnt, 1.0, None)
        out /= cnt[:, None]  # mean-pooling；L2 归一化在 build_index 后续统一做
        return out

    @staticmethod
    def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        return vectors / norms


def format_results(results: list[SearchResult]) -> str:
    lines = []
    for i, item in enumerate(results, start=1):
        lines.append(f"[{i}] {item.product} / {item.heading}")
        lines.append(f"chunk_id={item.chunk_id} pics={item.pics}")
        lines.append(item.text[:280].replace("\n", " ") + ("..." if len(item.text) > 280 else ""))
        lines.append("")
    return "\n".join(lines).strip()


if __name__ == "__main__":
    sample = "扶手<PIC>USB-C 使用扶手"
    clean_tokens = tokenize_mixed_clean(sample)
    assert "pic" not in clean_tokens
    assert clean_tokens.count("扶手") == 2
    assert "usb-c" in clean_tokens
    assert RetrievalEngine(rerank_candidate_pool=30).rerank_candidate_pool == 30
    print("retrieval_engine_self_check=OK")
