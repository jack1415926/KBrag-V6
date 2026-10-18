"""
启动本地 GGUF embedding fallback 服务。

当前主线默认使用硅基流动线上 `Qwen/Qwen3-Embedding-8B`，不依赖本地8091；本文件保留为本地BGE-M3兼容回退服务。
本脚本仅用于显式本地 fallback/离线复现。

依赖：
- llama-server 二进制
- 本地 GGUF embedding 模型（bge-m3）

示例：
    python embedding_server.py

自定义端口：
    python embedding_server.py --port 8091
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


# 注意：下面两个默认路径来自原作者的本机环境，仅作占位示例。
# 在其它机器使用本地回退时，请通过 --llama-server / --model 或对应环境变量指定自己的路径。
DEFAULT_LLAMA_SERVER = Path("/Users/alian/.docker/bin/inference/llama-server")
DEFAULT_MODEL = Path("/Users/alian/llm_model/gpustack/bge-m3-GGUF/bge-m3-FP16.gguf")


def build_command(args: argparse.Namespace) -> list[str]:
    """Build the llama-server command used for the local embedding fallback."""
    cmd = [
        str(args.llama_server),
        "--model", str(args.model),
        "--host", args.host,
        "--port", str(args.port),
        "--ctx-size", str(args.ctx_size),
        "--batch-size", str(args.batch_size),
        "--ubatch-size", str(args.ubatch_size),
        "--parallel", str(args.parallel),
        "--threads", str(args.threads),
        "--pooling", "mean",
        "--embedding",
        "--alias", args.alias,
        "--no-webui",
    ]

    if args.n_gpu_layers is not None:
        cmd.extend(["--gpu-layers", str(args.n_gpu_layers)])

    if args.verbose:
        cmd.append("--verbose")

    return cmd


def parse_args() -> argparse.Namespace:
    """Parse local embedding fallback server options."""
    parser = argparse.ArgumentParser(description="启动本地 GGUF embedding 服务")
    parser.add_argument("--llama-server", type=Path, default=DEFAULT_LLAMA_SERVER)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--host", default=os.getenv("EMBEDDING_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("EMBEDDING_PORT", "8091")))
    parser.add_argument("--alias", default=os.getenv("EMBEDDING_MODEL", "bge-m3"))
    parser.add_argument("--ctx-size", type=int, default=int(os.getenv("EMBEDDING_CTX_SIZE", "8192")))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("EMBEDDING_BATCH_SIZE", "4096")))
    parser.add_argument("--ubatch-size", type=int, default=int(os.getenv("EMBEDDING_UBATCH_SIZE", "2048")))
    parser.add_argument("--parallel", type=int, default=int(os.getenv("EMBEDDING_PARALLEL", "4")))
    parser.add_argument("--threads", type=int, default=int(os.getenv("EMBEDDING_THREADS", "8")))
    parser.add_argument("--n-gpu-layers", type=str, default=os.getenv("EMBEDDING_N_GPU_LAYERS", "auto"))
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Validate local model paths and run the embedding fallback process."""
    args = parse_args()

    if not args.llama_server.exists():
        print(f"找不到 llama-server: {args.llama_server}", file=sys.stderr)
        return 1
    if not args.model.exists():
        print(f"找不到 embedding 模型: {args.model}", file=sys.stderr)
        return 1

    cmd = build_command(args)
    print("即将启动 embedding 服务：")
    print(" ".join(shlex.quote(part) for part in cmd))
    print(f"\nAPI 预期地址: http://{args.host}:{args.port}/v1/embeddings")

    if args.dry_run:
        return 0

    proc = subprocess.Popen(cmd)
    try:
        return proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        try:
            return proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            return proc.wait()


if __name__ == "__main__":
    raise SystemExit(main())
