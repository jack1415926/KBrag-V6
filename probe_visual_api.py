"""Probe SiliconFlow VL embedding/rerank payload compatibility with public data."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from config_runtime import apply_default_env
from visual_retrieval import VisualEmbeddingClient, VisualRerankClient


PUBLIC_IMAGE = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"


def main() -> None:
    """Probe documented payloads without sending project data."""
    parser = argparse.ArgumentParser(description="探测VL Embedding/Rerank接口契约")
    parser.add_argument("--out", type=Path, default=Path("validation_outputs/visual_api_probe.json"))
    args = parser.parse_args()
    apply_default_env()
    base_url = os.getenv("EMBEDDING_BASE_URL", "https://api.siliconflow.cn/v1")
    api_key = os.getenv("EMBEDDING_API_KEY", "")
    embedding = VisualEmbeddingClient(base_url, api_key)
    rerank = VisualRerankClient(
        os.getenv("RERANK_BASE_URL", base_url),
        os.getenv("RERANK_API_KEY", api_key),
    )
    text_vector = embedding.embed_inputs(["a dog on a beach"])[0]
    image_vector = embedding.embed_inputs([{"image": PUBLIC_IMAGE}])[0]
    mixed_supported = True
    try:
        mixed_vector = embedding.embed_inputs([{"text": "a dog", "image": PUBLIC_IMAGE}])[0]
        mixed_dimension = len(mixed_vector)
    except Exception:
        mixed_supported = False
        mixed_dimension = 0
    ranked = rerank.rerank(
        {"image": PUBLIC_IMAGE},
        [{"image": PUBLIC_IMAGE}, "an unrelated text document"],
        top_n=2,
    )
    mixed_rerank_supported = True
    try:
        mixed_ranked = rerank.rerank(
            {"text": "find the dog", "image": PUBLIC_IMAGE},
            [
                {"text": "a dog on a beach", "image": PUBLIC_IMAGE},
                {"text": "an unrelated document"},
            ],
            top_n=2,
        )
    except Exception:
        mixed_rerank_supported = False
        mixed_ranked = []
    result = {
        "status": "OK",
        "timestamp": int(time.time()),
        "sample": "Qwen official public demo image",
        "text_dimension": len(text_vector),
        "image_dimension": len(image_vector),
        "mixed_embedding_supported": mixed_supported,
        "mixed_embedding_dimension": mixed_dimension,
        "rerank_results": len(ranked),
        "mixed_rerank_supported": mixed_rerank_supported,
        "mixed_rerank_results": len(mixed_ranked),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "visual_api_probe=OK "
        f"text_dim={result['text_dimension']} image_dim={result['image_dimension']} "
        f"mixed_supported={str(mixed_supported).lower()} mixed_dim={mixed_dimension} "
        f"rerank_results={len(ranked)} mixed_rerank_supported={str(mixed_rerank_supported).lower()} "
        f"mixed_rerank_results={len(mixed_ranked)} out={args.out}"
    )


if __name__ == "__main__":
    main()
