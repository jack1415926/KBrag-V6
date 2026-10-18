"""主办方 /chat 接口的薄包装层。

- 不修改 agent.py / submission_utils.py 等核心代码
- 复用 run_agent + format_submission_ret，确保线上输出与 CSV 提交完全一致
- 文本/多模态使用可配置统一截止时间，并向下游调用传播剩余预算和协作取消信号

启动：
    KAFU_API_TOKEN=sk-xxx \\
    python -m uvicorn api_server:app --host 0.0.0.0 --port 8000 --workers 1

也可以直接执行 ./run_api.sh（默认参数与此一致）。
"""
from __future__ import annotations

import asyncio
import base64
import binascii
import concurrent.futures
import json
import logging
import os
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, field_validator
from request_context import RequestCancelled, check_cancelled, remaining_timeout, request_scope

load_dotenv()

try:
    from config_runtime import apply_default_env

    apply_default_env()
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("api_server")

ROOT = Path(__file__).resolve().parent
MANUAL_IMAGE_DIR = ROOT / "手册" / "插图"
WEB_INDEX_PATH = ROOT / "web" / "index.html"
LATENCY_PROGRESS_PATH = ROOT / "validation_outputs" / "search_trace_baseline_progress.json"
_MANUAL_IMAGE_MEDIA_TYPES = {".jpg": "image/jpeg", ".png": "image/png"}


def _build_manual_image_map() -> dict[str, Path]:
    """Map stable image IDs to local files without accepting user-supplied paths."""
    mapping: dict[str, Path] = {}
    if not MANUAL_IMAGE_DIR.is_dir():
        return mapping
    for path in sorted(MANUAL_IMAGE_DIR.iterdir()):
        if not path.is_file() or path.suffix.lower() not in _MANUAL_IMAGE_MEDIA_TYPES:
            continue
        if path.stem in mapping:
            raise RuntimeError(f"重复的手册图片ID: {path.stem}")
        mapping[path.stem] = path
    return mapping


MANUAL_IMAGE_BY_ID = _build_manual_image_map()

REQUEST_TIMEOUT_S = float(os.getenv("CHAT_TIMEOUT_S", "20"))
MULTIMODAL_REQUEST_TIMEOUT_S = float(os.getenv("CHAT_MULTIMODAL_TIMEOUT_S", "30"))
EXPECTED_TOKEN = os.getenv("KAFU_API_TOKEN", "").strip()

# 客服/技术分类器：DeepSeek V4 Flash 关闭 thinking，三路二分类投票。
CLASSIFIER_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "").strip()
CLASSIFIER_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
CLASSIFIER_MODEL = os.getenv("DEEPSEEK_BINARY_MODEL", os.getenv("DEEPSEEK_INTENT_MODEL", "deepseek-v4-flash")).strip()
CLASSIFIER_TIMEOUT_S = float(os.getenv("DEEPSEEK_BINARY_TIMEOUT", os.getenv("DEEPSEEK_INTENT_TIMEOUT", "20")))
CLASSIFIER_MAX_TOKENS = int(os.getenv("DEEPSEEK_BINARY_MAX_TOKENS", "4"))
_LABEL_RE = re.compile(r"^\s*([01])\s*$")
API_RAW_PATH = Path(os.getenv("CHAT_API_RAW_PATH")) if os.getenv("CHAT_API_RAW_PATH") else None
API_TRACE_PATH = Path(os.getenv("CHAT_API_TRACE_PATH", "/tmp/kbrag_chat_api_server.trace.jsonl"))
MAX_CHAT_IMAGES = int(os.getenv("CHAT_MAX_IMAGES", "3"))
MAX_CHAT_IMAGE_BYTES = int(os.getenv("CHAT_MAX_IMAGE_BYTES", str(5 * 1024 * 1024)))
_IMAGE_DATA_URL_RE = re.compile(
    r"^data:image/(?P<media_type>png|jpg|jpeg|webp);base64,(?P<data>[A-Za-z0-9+/=\r\n]+)$",
    re.IGNORECASE,
)
_SESSION_HISTORY_LIMIT = int(os.getenv("CHAT_SESSION_HISTORY_LIMIT", "6"))
_SESSION_HISTORY: dict[str, list[dict[str, str]]] = {}
_SESSION_LOCK = threading.Lock()
_API_OUTPUT_LOCK = threading.Lock()
_BACKGROUND_AGENT_TASKS: set[asyncio.Task] = set()


# ───────── 引擎初始化（懒加载到 lifespan） ─────────

_engine = None
_engine_lock = asyncio.Lock()


async def get_engine():
    global _engine
    if _engine is not None:
        return _engine
    async with _engine_lock:
        if _engine is None:
            from retrieval_engine import RetrievalEngine
            log.info("初始化 RetrievalEngine（首次请求）...")
            t0 = time.time()
            engine = RetrievalEngine()
            engine.ensure_index()
            log.info("RetrievalEngine 就绪 (%.1fs)", time.time() - t0)
            _engine = engine
    return _engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not EXPECTED_TOKEN:
        log.warning(
            "环境变量 KAFU_API_TOKEN 为空，鉴权将拒绝所有请求。"
            "请设置后重启。"
        )
    else:
        log.info("KAFU_API_TOKEN 已配置（长度=%d）", len(EXPECTED_TOKEN))
    log.info("CHAT_TIMEOUT_S=%.0fs CHAT_MULTIMODAL_TIMEOUT_S=%.0fs", REQUEST_TIMEOUT_S, MULTIMODAL_REQUEST_TIMEOUT_S)

    asyncio.create_task(_warmup_engine())
    yield


async def _warmup_engine() -> None:
    try:
        await get_engine()
    except Exception:  # noqa: BLE001
        log.exception("引擎预热失败（首次请求时会重试）")


app = FastAPI(
    title="客服智能体 /chat API",
    version="1.0.0",
    lifespan=lifespan,
)

bearer = HTTPBearer(auto_error=False)


def auth(creds: Optional[HTTPAuthorizationCredentials] = Depends(bearer)) -> None:
    """Bearer Token 鉴权。

    官方只要求请求头 Authorization: Bearer {token}；服务端合法 token 由 KAFU_API_TOKEN 配置，未配置时直接 503 防止误开放。
    """
    if not EXPECTED_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="server token not configured",
        )
    if creds is None or creds.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if creds.credentials != EXPECTED_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ───────── 请求 / 响应模型 ─────────

class ChatRequest(BaseModel):
    """官方 /chat 请求体。

    question 是唯一必填核心字段；images 按官方 data URL 口径校验并透传给多模态模型；session_id 用于内存态短历史续接；stream 当前兼容接收但仍同步返回。
    """
    question: str = Field(..., min_length=1, description="用户问题字符串")
    images: list[str] = Field(default_factory=list, description="Base64 图片列表，支持 0-3 张，每张不超过 5MB")
    session_id: Optional[str] = Field(default=None, description="客服会话 ID")
    stream: bool = Field(default=False, description="是否流式响应（当前同步返回完整答案）")

    @field_validator("question")
    @classmethod
    def validate_question(cls, question: str) -> str:
        question = (question or "").strip()
        if not question:
            raise ValueError("question 不能为空")
        return question

    @field_validator("images")
    @classmethod
    def validate_images(cls, images: list[str]) -> list[str]:
        if len(images) > MAX_CHAT_IMAGES:
            raise ValueError(f"images 最多支持 {MAX_CHAT_IMAGES} 张")
        for idx, image in enumerate(images):
            match = _IMAGE_DATA_URL_RE.match(image or "")
            if not match:
                raise ValueError(
                    "images 必须使用 data:image/{png/jpg/jpeg/webp};base64,{编码内容} 格式"
                )
            try:
                raw = base64.b64decode(match.group("data"), validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ValueError(f"第 {idx + 1} 张图片 base64 编码无效") from exc
            if len(raw) > MAX_CHAT_IMAGE_BYTES:
                raise ValueError(f"第 {idx + 1} 张图片超过 {MAX_CHAT_IMAGE_BYTES // (1024 * 1024)}MB")
        return images


class ChatImage(BaseModel):
    """One answer image resolved from the Agent's ordered picture IDs."""
    id: str
    url: str
    mime_type: str
    marker_index: int


class ChatResponseData(BaseModel):
    """成功响应 data 字段，与官方接口定义保持一致。"""
    answer: str
    display_answer: str
    images: list[ChatImage] = Field(default_factory=list)
    session_id: str
    timestamp: int


class ChatResponse(BaseModel):
    """统一 JSON 包装：code/msg/data。错误情况由 FastAPI HTTPException 返回。"""
    code: int = 0
    msg: str = "success"
    data: ChatResponseData


# ───────── 业务逻辑 ─────────

_BINARY_PROMPTS: dict[str, str] = {
    "service_guard": """你是在线客服系统的客服服务边界识别器。只输出一个字符：0 或 1，不要解释。

0 = 商家客服/平台服务答案。
用户问的是商家、平台、店铺或售后团队能否提供某项服务、如何办理某个流程、交易售后怎么处理，答案应来自客服政策、订单/售后系统或商家承诺。
包括：订单、物流、退款、退换货、试用期、延长试用、商品更换、发票、价格、购买渠道、投诉、人工客服、联系方式、上门安装服务、维修/终身维修服务流程或费用、是否提供纸质版说明书、电子版说明书在哪里获取、商品生产日期/批次/供给信息。即使问题提到故障、安装、维修、说明书，只要是在问“商家是否提供/能否办理/可以吗/在哪里获取/什么时候”，也输出 0。

1 = 产品手册/产品知识答案。
用户问的是产品本身怎么操作、安装、设置、维护、清洁、排障、更换部件、安全规则、部件/按钮/参数/图示，或手册中的保修/免责声明/法规声明/maintenance and care 内容本身。

关键边界：
- 问“你们/商家/平台/售后能否提供或如何办理” => 0。
- 问“产品本身怎么做、怎么用、怎么排障、怎么更换部件、手册条款写什么” => 1。

只输出 0 或 1。""",
    "tech_recall": """你是一个极快的客服/产品手册技术二分类路由器。必须只输出一个字符：0 或 1，不要解释。

输出 0：客服/平台服务/交易售后问题。包括订单、物流、退款、退换货流程、发票、价格、购买渠道、人工客服、投诉、联系方式、平台售后政策、真实维修服务流程或费用咨询。

输出 1：产品手册技术问题。包括产品安装、使用、设置、按钮/部件、参数、清洁维护、故障排查、安全操作、图示说明、随机附带说明、法规声明、保修条款、免责声明、维护保养政策、搬运/移动注意事项、产品自带支付/连接/功能操作、更换保险丝/滤网/电池/灯泡/门/按钮等产品部件。

关键边界：如果问题明确围绕某个具体产品、部件或手册内容，询问 warranty/保修、policy/政策、statement/声明、disclaimer/免责、maintenance and care/维护保养、safety/安全、move/搬运、payment/支付功能 等内容，属于产品手册技术题，输出 1。
只有在询问商家/平台的订单、退款、退换货、发票、物流、人工、投诉、购买、售后维修服务流程时，才输出 0。
如果需要查产品手册才能回答，输出 1。""",
    "answer_source": """你是在线客服系统的前置二分类路由器。只输出一个字符：0 或 1。

判断依据是“这道问题的正确答案应该来自哪里”：

0 = 商家客服/平台服务答案。
问题在问商家、平台、店铺或售后团队的服务承诺、办理流程、交易信息或人工支持。典型特征是“你们/你们家/商家是否提供、如何办理、多久到账、怎么联系客服”。包括订单、物流、退款、退换货、发票、价格、购买渠道、投诉、人工客服、联系方式、上门服务、维修服务流程/费用、是否提供纸质或电子材料、商品生产日期/批次等商家供给信息。

1 = 产品手册/产品知识答案。
问题在问某个产品本身的知识、操作、安装、使用、设置、部件、参数、维护、清洁、排障、安全、规则、原因、条件、要求、图示、声明或手册条款。即使没有明确说“手册”，只要答案应来自产品说明书/使用指南/安装指南/安全说明/保修或法规条款，就输出 1。包括“如何做/怎么用/需要注意什么/有哪些组成部件/规则是什么/原因是什么/要求是什么/如何排查”等产品知识问题。

关键区分：
- 问“你们/商家能否提供某项服务或如何办理” => 0。
- 问“产品本身如何操作、有哪些要求/规则/原因/部件/条款/声明” => 1。
- warranty/policy/statement/disclaimer 如果是某个产品手册中的条款内容 => 1；如果是商家售后服务政策/办理流程 => 0。
- safety、payment、move/load、repair、troubleshooting 等词不要按词判；看是在问产品功能/操作/规则/原因，还是问商家服务。

只输出 0 或 1。""",
}

_SERVICE_FALLBACK_RE = re.compile(
    r"(订单|物流|快递|发货|到货|退款|退货|换货|退换|售后|保修服务|维修服务|"
    r"发票|价格|优惠|购买|下单|店铺|商家|平台|人工客服|联系客服|投诉|"
    r"纸质版说明书|电子版说明书|生产日期|批次|延长试用|上门安装)",
    re.IGNORECASE,
)
_TECH_FALLBACK_RE = re.compile(
    r"(安装|使用|设置|操作|清洁|维护|保养|排障|故障|更换|拆卸|组装|"
    r"按钮|部件|螺丝|滤网|电池|保险丝|灯泡|参数|规格|安全|警告|"
    r"warranty|policy|statement|disclaimer|maintenance|troubleshooting|install|replace|clean)",
    re.IGNORECASE,
)


def _local_fallback_route(question: str) -> tuple[str, dict[str, Any]]:
    """DeepSeek 分类器不可用时的保守本地兜底。

    明显商家/平台/订单/售后问题走 service；明显产品操作/维护/排障问题走 tech；
    边界不清时仍按 tech 处理，保持“技术链路可查证据”的安全兜底。
    """
    service_hit = bool(_SERVICE_FALLBACK_RE.search(question))
    tech_hit = bool(_TECH_FALLBACK_RE.search(question))
    route = "service" if service_hit and not tech_hit else "tech"
    return route, {
        "kind": "classifier_fallback",
        "strategy": "local_rule",
        "route": route,
        "service_hit": service_hit,
        "tech_hit": tech_hit,
    }


def _parse_binary_label(text: str | None) -> int | None:
    match = _LABEL_RE.match(text or "")
    return int(match.group(1)) if match else None


def _classifier_extra_body() -> dict[str, Any]:
    if "deepseek" in CLASSIFIER_BASE_URL.lower():
        return {"thinking": {"type": "disabled"}}
    return {"enable_thinking": False}


def _classify_one_prompt(
    name: str,
    prompt: str,
    question: str,
    timeout: float = CLASSIFIER_TIMEOUT_S,
) -> tuple[str, int | None, str, str | None, float]:
    started = time.time()
    try:
        from openai import OpenAI
        client = OpenAI(
            base_url=CLASSIFIER_BASE_URL,
            api_key=CLASSIFIER_API_KEY,
            timeout=timeout,
            max_retries=0,
        )
        resp = client.chat.completions.create(
            model=CLASSIFIER_MODEL,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": question.strip()},
            ],
            max_tokens=CLASSIFIER_MAX_TOKENS,
            extra_body=_classifier_extra_body(),
        )
        raw = (resp.choices[0].message.content or "").strip()
        return name, _parse_binary_label(raw), raw, None, round(time.time() - started, 3)
    except Exception as exc:  # noqa: BLE001
        return name, None, "", repr(exc), round(time.time() - started, 3)


def _classify_question(question: str) -> tuple[str, dict[str, Any]]:
    """用 DeepSeek 三路投票分 service / tech，失败时走本地保守规则兜底。"""
    started = time.time()
    if not CLASSIFIER_BASE_URL or not CLASSIFIER_API_KEY:
        log.warning("DeepSeek 分类器未配置（DEEPSEEK_*），所有问题默认按 tech 处理")
        return "tech", {
            "kind": "classifier",
            "provider": "deepseek_binary_vote",
            "model": CLASSIFIER_MODEL,
            "route": "tech",
            "elapsed": 0.0,
            "error": "classifier_not_configured",
        }

    check_cancelled()
    classifier_timeout = remaining_timeout(CLASSIFIER_TIMEOUT_S)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(_BINARY_PROMPTS)) as pool:
        futures = [
            pool.submit(_classify_one_prompt, name, prompt, question, classifier_timeout)
            for name, prompt in _BINARY_PROMPTS.items()
        ]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    check_cancelled()

    votes = {name: pred for name, pred, _raw, _err, _elapsed in results}
    raw_outputs = {name: raw for name, _pred, raw, _err, _elapsed in results}
    timings = {name: elapsed for name, _pred, _raw, _err, elapsed in results}
    errors = {name: err for name, _pred, _raw, err, _elapsed in results if err}
    valid = [pred for pred in votes.values() if pred is not None]
    if not valid:
        fallback_route, fallback_trace = _local_fallback_route(question)
        log.warning("DeepSeek 三路分类全失败 errors=%s，本地规则兜底 %s", errors, fallback_route)
        return fallback_route, {
            "kind": "classifier",
            "provider": "deepseek_binary_vote",
            "model": CLASSIFIER_MODEL,
            "route": fallback_route,
            "votes": votes,
            "raw_outputs": raw_outputs,
            "prompt_elapsed": timings,
            "errors": errors,
            "elapsed": round(time.time() - started, 3),
            "fallback": True,
            "fallback_detail": fallback_trace,
        }

    ones = sum(valid)
    zeros = len(valid) - ones
    label = 1 if ones >= zeros else 0
    route = "tech" if label == 1 else "service"
    if len(set(valid)) > 1 or errors:
        log.info("分类分歧 route=%s votes=%s raw=%s errors=%s", route, votes, raw_outputs, errors)
    return route, {
        "kind": "classifier",
        "provider": "deepseek_binary_vote",
        "model": CLASSIFIER_MODEL,
        "route": route,
        "label": label,
        "votes": votes,
        "raw_outputs": raw_outputs,
        "prompt_elapsed": timings,
        "errors": errors,
        "elapsed": round(time.time() - started, 3),
        "fallback": False,
    }


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _API_OUTPUT_LOCK:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _get_session_history(session_id: str) -> list[dict[str, str]]:
    """读取同一 session 的短历史副本，避免请求线程直接修改共享列表。"""
    with _SESSION_LOCK:
        return [dict(item) for item in _SESSION_HISTORY.get(session_id, [])]


def _append_session_turn(session_id: str, question: str, answer: str) -> None:
    """写入一轮用户/客服消息；当前为内存态，便于演示追问，生产可替换为 Redis。"""
    with _SESSION_LOCK:
        history = _SESSION_HISTORY.setdefault(session_id, [])
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        if len(history) > _SESSION_HISTORY_LIMIT:
            del history[: len(history) - _SESSION_HISTORY_LIMIT]


def _build_question_with_history(question: str, history: list[dict[str, str]]) -> str:
    """把历史会话压成文本前缀，让无状态 /chat 具备基本追问理解能力。"""
    if not history:
        return question
    lines = ["以下是同一客服会话的历史对话，仅用于理解用户追问；请优先回答最后一个问题。"]
    for item in history:
        role = "用户" if item.get("role") == "user" else "客服"
        lines.append(f"{role}: {item.get('content', '')}")
    lines.append(f"用户当前问题: {question}")
    return "\n".join(lines)


def _build_multimodal_question(question: str, images: list[str]) -> str:
    if not images:
        return question
    image_note = (
        f"用户本轮上传了 {len(images)} 张图片。图片已随本轮消息一并提供；"
        "请结合图片内容和文字问题回答。若图片内容与问题无关或无法识别，请说明需要用户补充更清晰的信息。"
    )
    return f"{question}\n\n{image_note}"


def _build_multimodal_content(question: str, images: list[str]) -> list[dict[str, Any]]:
    """构造 OpenAI-compatible 多模态 content。

    同时保留 image_url 和 source 字段：OpenAI 兼容端点读 image_url，Anthropic 风格调试/trace 仍能看出原始 base64 类型。
    """
    content: list[dict[str, Any]] = [{"type": "text", "text": question}]
    for image in images:
        match = _IMAGE_DATA_URL_RE.match(image)
        if not match:
            continue
        media_type = match.group("media_type").lower().replace("jpg", "jpeg")
        content.append({
            "type": "image_url",
            "image_url": {"url": image},
            "source": {
                "type": "base64",
                "media_type": f"image/{media_type}",
                "data": match.group("data"),
            },
        })
    return content


def _write_api_success_trace(
    *,
    request_id: str,
    session_id: str,
    question: str,
    images_count: int,
    stream: bool,
    route: str,
    formatted_answer: str,
    pics: list[str],
    elapsed: float,
    agent_trace: dict[str, Any],
) -> None:
    result = agent_trace.get("result") or {}
    raw_record = {
        "request_id": request_id,
        "session_id": session_id,
        "question": question,
        "images_count": images_count,
        "stream": stream,
        "route": route,
        "answer": formatted_answer,
        "pics": pics,
        "tool_calls": int(result.get("tool_calls") or 0),
        "turns": int(result.get("turns") or 0),
        "elapsed": round(elapsed, 3),
        "error": None,
        "timestamp": int(time.time()),
    }
    trace_record = {
        **agent_trace,
        "request_id": request_id,
        "session_id": session_id,
        "route": route,
        "api_elapsed": round(elapsed, 3),
        "formatted_answer": formatted_answer,
    }
    if API_RAW_PATH is not None:
        _append_jsonl(API_RAW_PATH, raw_record)
    _append_jsonl(API_TRACE_PATH, trace_record)


def _write_api_error_trace(
    *,
    request_id: str,
    session_id: str,
    question: str,
    images_count: int,
    stream: bool,
    elapsed: float,
    error: str,
) -> None:
    record = {
        "request_id": request_id,
        "session_id": session_id,
        "question": question,
        "images_count": images_count,
        "stream": stream,
        "answer": "",
        "pics": [],
        "tool_calls": 0,
        "turns": 0,
        "elapsed": round(elapsed, 3),
        "error": error[:500],
        "timestamp": int(time.time()),
    }
    if API_RAW_PATH is not None:
        _append_jsonl(API_RAW_PATH, record)
    _append_jsonl(API_TRACE_PATH, {**record, "kind": "chat_api_error"})


def _run_agent_sync_impl(question: str, session_id: str, images: list[str]) -> tuple[str, list[str], str, dict[str, Any]]:
    """在 worker 线程里跑 ReAct Agent，同时返回仅供服务端落盘的内部 trace。"""
    from agent import run_agent
    from retrieval_engine import RetrievalEngine

    # 同步上下文里取 engine（这里走全局变量，已由 lifespan 预热）
    global _engine
    if _engine is None:
        _engine = RetrievalEngine()
        _engine.ensure_index()

    history = _get_session_history(session_id)
    routed_question = _build_question_with_history(question, history)
    agent_question_text = _build_multimodal_question(routed_question, images)
    agent_question = _build_multimodal_content(agent_question_text, images) if images else agent_question_text

    # 用 DeepSeek V4 Flash 三路二分类，再用 fake_qid 把 run_agent 路由到正确 prompt：
    #   service -> fake_qid=0 (run_agent 内部 qid<64 走 SERVICE_SYSTEM_PROMPT)
    #   tech    -> fake_qid=64 (qid>=64 走 TECH_SYSTEM_PROMPT + 强制检索)
    route, classifier_trace = _classify_question(routed_question)
    fake_qid = 0 if route == "service" else 64
    model = os.getenv("SILICONFLOW_VISION_MODEL") if images else None
    result = run_agent(
        agent_question,
        _engine,
        model=model,
        question_id=fake_qid,
        session_id=session_id,
        collect_trace=True,
    )
    agent_trace = dict(result.trace or {})
    agent_trace["classifier"] = classifier_trace
    agent_trace["session_history_turns"] = len(history) // 2
    agent_trace["input_images_count"] = len(images)
    return result.answer or "", list(result.pics or []), route, agent_trace


def _run_agent_sync(
    question: str,
    session_id: str,
    images: list[str],
    cancel_event: threading.Event | None = None,
    deadline: float | None = None,
) -> tuple[str, list[str], str, dict[str, Any]]:
    """Run one worker request under isolated identity, deadline, and cancellation state."""
    with request_scope(deadline=deadline, cancel_event=cancel_event):
        check_cancelled()
        return _run_agent_sync_impl(question, session_id, images)


def _consume_background_agent(task: asyncio.Task) -> None:
    """Consume a timed-out worker result and remove it from the tracked set."""
    _BACKGROUND_AGENT_TASKS.discard(task)
    try:
        task.result()
    except asyncio.CancelledError:
        pass
    except RequestCancelled:
        pass
    except Exception:  # noqa: BLE001
        log.exception("Timed-out Agent worker ended with an error")


def _format_answer(answer: str, pics: list[str], route: str) -> str:
    """复用 CSV 提交逻辑：按分类结果走 service / tech normalize。"""
    from submission_utils import format_submission_ret

    fake_qid = 0 if route == "service" else 64
    return format_submission_ret(fake_qid, answer, pics)


def _format_display_answer(answer: str, pics: list[str], route: str) -> tuple[str, list[str]]:
    """Return a browser-friendly answer while leaving the legacy answer untouched."""
    from submission_utils import normalize_customer_service_answer, normalize_technical_answer

    if route == "service":
        return normalize_customer_service_answer(answer), []
    return normalize_technical_answer(answer, pics)


def _build_chat_images(pics: list[str], route: str) -> list[ChatImage]:
    """Expose only resolved technical-answer images, preserving marker order."""
    if route != "tech":
        return []
    images: list[ChatImage] = []
    for marker_index, image_id in enumerate(pics):
        path = MANUAL_IMAGE_BY_ID.get(image_id)
        if path is None:
            log.warning("答案图片ID不存在，跳过: %s", image_id)
            continue
        images.append(
            ChatImage(
                id=image_id,
                url=f"/manual-images/{image_id}",
                mime_type=_MANUAL_IMAGE_MEDIA_TYPES[path.suffix.lower()],
                marker_index=marker_index,
            )
        )
    return images


@app.get("/manual-images/{image_id}", response_class=FileResponse)
async def manual_image(image_id: str) -> FileResponse:
    """Return a known manual image by ID; arbitrary filesystem paths are never resolved."""
    path = MANUAL_IMAGE_BY_ID.get(image_id)
    if path is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="image not found")
    return FileResponse(path, media_type=_MANUAL_IMAGE_MEDIA_TYPES[path.suffix.lower()])


@app.get("/", response_class=FileResponse, include_in_schema=False)
async def web_index() -> FileResponse:
    """Serve the minimal same-origin chat UI."""
    if not WEB_INDEX_PATH.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="web UI not found")
    return FileResponse(WEB_INDEX_PATH, media_type="text/html; charset=utf-8")


@app.get("/latency-baseline/status")
async def latency_baseline_status() -> dict[str, Any]:
    """Read the progress file written by the standalone latency baseline runner."""
    if not LATENCY_PROGRESS_PATH.is_file():
        return {"status": "idle", "current": 0, "total": 0}
    try:
        return json.loads(LATENCY_PROGRESS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "error", "current": 0, "total": 0, "error": type(exc).__name__}


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(auth)])
async def chat(req: ChatRequest, request: Request) -> ChatResponse:
    """官方核心端点：同步返回一轮客服/技术答案。

    这里是薄包装层：鉴权、参数校验、超时、trace 和 session 在 API 层处理；真正回答仍复用 run_agent + format_submission_ret，保证线上线下格式同源。
    """
    request_id = request.headers.get("X-Request-Id") or f"kf_req_{uuid.uuid4()}"
    session_id = req.session_id or f"kf_session_{uuid.uuid4().hex[:12]}"
    log.info(
        "REQ id=%s sess=%s images=%d stream=%s q=%r",
        request_id, session_id, len(req.images), req.stream, req.question[:80],
    )
    if req.images:
        log.info("REQ id=%s 收到 %d 张图片，已接入本轮多模态消息", request_id, len(req.images))
    if req.stream:
        log.info("REQ id=%s stream=true，当前同步返回（最终回答首token时间见 run_agent 日志）", request_id)

    t0 = time.time()
    timeout_s = MULTIMODAL_REQUEST_TIMEOUT_S if req.images else REQUEST_TIMEOUT_S
    cancel_event = threading.Event()
    deadline = time.monotonic() + timeout_s
    worker_task = asyncio.create_task(
        asyncio.to_thread(
            _run_agent_sync,
            req.question,
            session_id,
            req.images,
            cancel_event,
            deadline,
        )
    )
    try:
        answer, pics, route, agent_trace = await asyncio.wait_for(
            asyncio.shield(worker_task),
            timeout=timeout_s,
        )
    except (asyncio.TimeoutError, RequestCancelled):
        cancel_event.set()
        if not worker_task.done():
            _BACKGROUND_AGENT_TASKS.add(worker_task)
            worker_task.add_done_callback(_consume_background_agent)
        elapsed = time.time() - t0
        log.warning("REQ id=%s TIMEOUT after %.1fs", request_id, elapsed)
        _write_api_error_trace(
            request_id=request_id,
            session_id=session_id,
            question=req.question,
            images_count=len(req.images),
            stream=req.stream,
            elapsed=elapsed,
            error=f"agent timeout after {timeout_s:.0f}s",
        )
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"agent timeout after {timeout_s:.0f}s",
        )
    except asyncio.CancelledError:
        cancel_event.set()
        if not worker_task.done():
            _BACKGROUND_AGENT_TASKS.add(worker_task)
            worker_task.add_done_callback(_consume_background_agent)
        raise
    except Exception as exc:  # noqa: BLE001
        elapsed = time.time() - t0
        log.exception("REQ id=%s ERROR", request_id)
        _write_api_error_trace(
            request_id=request_id,
            session_id=session_id,
            question=req.question,
            images_count=len(req.images),
            stream=req.stream,
            elapsed=elapsed,
            error=str(exc),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"agent error: {exc}",
        ) from exc

    formatted = _format_answer(answer, pics, route)
    display_answer, display_pics = _format_display_answer(answer, pics, route)
    response_images = _build_chat_images(display_pics, route)
    _append_session_turn(session_id, req.question, formatted)
    elapsed = time.time() - t0
    agent_result = agent_trace.get("result") or {}
    tool_calls = int(agent_result.get("tool_calls") or 0)
    agent_turns = int(agent_result.get("turns") or 0)
    _write_api_success_trace(
        request_id=request_id,
        session_id=session_id,
        question=req.question,
        images_count=len(req.images),
        stream=req.stream,
        route=route,
        formatted_answer=formatted,
        pics=pics,
        elapsed=elapsed,
        agent_trace=agent_trace,
    )
    log.info(
        "RES id=%s sess=%s route=%s elapsed=%.1fs pics=%d ans_len=%d tool_calls=%d agent_turns=%d",
        request_id, session_id, route, elapsed, len(pics), len(formatted), tool_calls, agent_turns,
    )

    return ChatResponse(
        code=0,
        msg="success",
        data=ChatResponseData(
            answer=formatted,
            display_answer=display_answer,
            images=response_images,
            session_id=session_id,
            timestamp=int(time.time()),
        ),
    )


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "engine_ready": _engine is not None,
        "timeout_s": REQUEST_TIMEOUT_S,
        "multimodal_timeout_s": MULTIMODAL_REQUEST_TIMEOUT_S,
        "background_agents": len(_BACKGROUND_AGENT_TASKS),
        "auth_configured": bool(EXPECTED_TOKEN),
        "classifier_provider": "deepseek_binary_vote",
        "classifier_configured": bool(CLASSIFIER_BASE_URL and CLASSIFIER_API_KEY),
        "classifier_model": CLASSIFIER_MODEL,
    }
