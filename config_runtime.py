"""V6 非敏感默认运行配置。

该文件用于降低评审启动成本：`api_server.py` / `llm_router.py` 会在启动时
调用 `apply_default_env()`，把这里的配置写入尚未设置的环境变量。

真实凭据仅通过本机 .env 或环境变量提供；不要写入源码或示例。
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


DEFAULT_ENV = {
    # 在线接口超时：公网/评审环境保守放宽，减少上游波动导致的误判。
    "CHAT_TIMEOUT_S": "50",
    "CHAT_MULTIMODAL_TIMEOUT_S": "60",
    "CHAT_MAX_IMAGES": "3",
    "CHAT_MAX_IMAGE_BYTES": str(5 * 1024 * 1024),

    # 无 qid 在线场景的 service / tech 二分类路由器。
    "DEEPSEEK_BASE_URL": "https://api.deepseek.com",
    "DEEPSEEK_BINARY_MODEL": "deepseek-v4-flash",
    "DEEPSEEK_BINARY_TIMEOUT": "3",
    "DEEPSEEK_BINARY_MAX_TOKENS": "4",

    # 主回答模型：文本请求走 DeepSeek Flash，上传图片时由 API 层切换视觉 Flash。
    "SILICONFLOW_BASE_URL": "https://api.deepseek.com",
    "SILICONFLOW_MODEL": "deepseek-v4-flash",
    "SILICONFLOW_VISION_MODEL": "deepseek-v4-flash-vision-exp",
    "SILICONFLOW_ONLY": "1",
    "SILICONFLOW_MAX_CONCURRENCY": "3",
    "AGENT_MAX_TOKENS": "8192",
    "LLM_TIMEOUT_SECONDS": "30",
    "LLM_TRANSIENT_RETRY_ATTEMPTS": "3",

    # 检索：使用硅基流动 Qwen3 文本 embedding / rerank，评审不需要本地启动 8091/8090。
    "EMBEDDING_BASE_URL": "https://api.siliconflow.cn/v1",
    "EMBEDDING_MODEL": "Qwen/Qwen3-Embedding-8B",
    "EMBEDDING_MAX_CONCURRENCY": "4",
    "TEXT_QUERY_EMBED_MODE": "batch",
    "RERANK_BASE_URL": "https://api.siliconflow.cn/v1",
    "RERANK_MODEL_ALIAS": "Qwen/Qwen3-Reranker-8B",
    "RERANK_ENABLED": "1",
    "RERANK_CANDIDATE_POOL": "20",

    # 独立视觉检索：评测达标前默认关闭；地址和密钥复用现有Embedding/Rerank配置。
    "VISUAL_RETRIEVAL_ENABLED": "0",
    "VISUAL_EMBEDDING_MODEL": "Qwen/Qwen3-VL-Embedding-8B",
    "VISUAL_RERANK_MODEL": "Qwen/Qwen3-VL-Reranker-8B",

    # chunk 命中后返回完整 parent section。
    "RETURN_PARENT_SECTION": "1",
}


def apply_default_env() -> None:
    """Apply delivery defaults without overriding variables already set by the caller."""
    load_dotenv(Path(__file__).with_name(".env"), override=False)
    for key, value in DEFAULT_ENV.items():
        os.environ.setdefault(key, value)
