"""交付包自检脚本。

该脚本只做本地静态检查，不访问远程模型服务。它覆盖初赛交付最容易漏的点：
核心文件、Markdown 材料、索引资产、脚本权限、Python 语法、注释覆盖率口径、运行痕迹和
源码中的分散密钥。提交前运行一次，可快速判断 `V6_submit_code/` 是否仍像一个干净交付包。
"""
from __future__ import annotations

import ast
import json
import re
import stat
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent

REQUIRED_FILES = [
    "README.md",
    "API.md",
    "MANIFEST.md",
    ".env.example",
    "requirements.txt",
    "config_runtime.py",
    "run_api.sh",
    "smoke_test.sh",
    "api_server.py",
    "agent.py",
    "request_context.py",
    "llm_router.py",
    "retrieval_engine.py",
    "product_router.py",
    "rerank_client.py",
    "submission_utils.py",
    "generate_submission_v5.py",
    "parse_manuals.py",
    "build_retrieval_index.py",
    "visual_retrieval.py",
    "build_visual_index.py",
    "prepare_visual_gold.py",
    "validate_visual_gold.py",
    "validate_visual_index.py",
    "run_visual_retrieval_experiment.py",
    "run_visual_text_regression.py",
    "validate_visual_experiment.py",
    "probe_visual_api.py",
    "smoke_visual_retrieval.py",
    "clean_section_summaries.py",
    "tests/test_request_context.py",
    "tests/test_visual_retrieval.py",
    "tests/test_search_manual_trace.py",
    "tests/test_api_images.py",
    "run_search_trace_baseline.py",
    "run_embedding_request_mode_experiment.py",
    "run_text_query_embed_mode_gold.py",
    "run_text_query_strategy_gold.py",
    "web/index.html",
    "validate_performance.py",
    "skills/search_manual.md",
    "data/catalog.json",
    "data/section_chunks.json",
    "data/section_summaries.json",
    "data/retrieval_chunks.json",
    "data/product_mapping_final.csv",
    "data/image_captions_v4_final.json",
    "data/index/dense.faiss",
    "data/index/retrieval_index.pkl",
    "data/visual_index/visual.faiss",
    "data/visual_index/visual_metadata.json",
    "data/visual_index/visual_build_summary.json",
    "data/visual_index/visual_build_progress.jsonl",
    "evaluation/visual_relevance_gold_30_derived.jsonl",
    "evaluation/visual_external_style_template.json",
    "evaluation/visual_relevance_gold_10_external.jsonl",
    "validation_outputs/visual_api_probe.json",
    "validation_outputs/visual_retrieval_smoke.json",
    "experiments/visual_faiss_20260915/operation_record.md",
    "experiments/visual_retrieval_20260916/operation_record.md",
    "experiments/visual_retrieval_20260916/config.json",
    "experiments/visual_retrieval_20260916/runs.jsonl",
    "experiments/visual_retrieval_20260916/summary.json",
    "experiments/visual_retrieval_20260916/gate_result.json",
    "experiments/visual_text_20260916/config.json",
    "experiments/visual_text_20260916/runs.jsonl",
    "experiments/visual_text_20260916/summary.json",
    "experiments/visual_original_text_anchor_20260916/config.json",
    "experiments/visual_original_text_anchor_20260916/runs.jsonl",
    "experiments/visual_original_text_anchor_20260916/summary.json",
    "experiments/visual_original_text_anchor_20260916/gate_result.json",
    "experiments/visual_original_text_anchor_20260916/operation_record.md",
    "项目技术架构说明文档/01_API接口说明.md",
    "项目技术架构说明文档/02_源码运行说明.md",
    "项目技术架构说明文档/03_技术方案说明.md",
    "项目技术架构说明文档/04_验证报告.md",
    "项目技术架构说明文档/08_视觉FAISS四路检索实验与优化复盘.md",
    "项目面试，运行问题说明文档/项目当前进展与下一步.md",
    "validation_outputs/dataset_overview.csv",
    "validation_outputs/routing_validation.csv",
    "validation_outputs/product_router_summary.csv",
    "validation_outputs/product_router_details.csv",
    "validation_outputs/multimodal_format_summary.csv",
    "validation_outputs/multimodal_format_details.csv",
    "validation_outputs/dialogue_api_validation.csv",
    "validation_outputs/validation_summary.md",
    "validation_outputs/validation_summary.json",
]

REQUIRED_DIRS = [
    "data/manual_sections",
    "手册_v4",
    "手册/插图",
    "submissions",
    "validation_outputs",
]

EXECUTABLE_FILES = [
    "run_api.sh",
    "smoke_test.sh",
]

FORBIDDEN_SUFFIXES = (
    ".pyc",
    ".pyo",
    ".bak",
    ".tmp",
    ".trace.jsonl",
    ".raw.jsonl",
    ".ckpt.json",
)

SECRET_RE = re.compile(
    r"sk-[A-Za-z0-9][A-Za-z0-9_-]{18,}|gh[pousr]_[A-Za-z0-9]{20,}|"
    r"github_pat_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16}|"
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
)
LOCAL_ONLY_DIRS = {".git", ".venv", "__pycache__", ".idea", ".run", ".tmp", "tmp"}


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def fail(errors: list[str], message: str) -> None:
    errors.append(f"FAIL {message}")


def warn(warnings: list[str], message: str) -> None:
    warnings.append(f"WARN {message}")


def check_required(errors: list[str]) -> None:
    for item in REQUIRED_FILES:
        path = ROOT / item
        if not path.is_file():
            fail(errors, f"缺少文件: {item}")
    for item in REQUIRED_DIRS:
        path = ROOT / item
        if not path.is_dir():
            fail(errors, f"缺少目录: {item}")


def check_executable(errors: list[str]) -> None:
    if sys.platform == "win32":
        return
    for item in EXECUTABLE_FILES:
        path = ROOT / item
        if path.exists() and not (path.stat().st_mode & stat.S_IXUSR):
            fail(errors, f"脚本没有可执行权限: {item}")


def check_no_generated_artifacts(errors: list[str]) -> None:
    for path in ROOT.rglob("*"):
        if LOCAL_ONLY_DIRS.intersection(path.relative_to(ROOT).parts):
            continue
        if path.is_file() and path.name == ".DS_Store":
            # Ignored local cache is not part of the GitHub delivery.
            continue
        if path.is_file() and path.name.endswith(FORBIDDEN_SUFFIXES):
            fail(errors, f"包含不应提交的运行/备份文件: {rel(path)}")


def python_files() -> list[Path]:
    return sorted(path for path in ROOT.glob("*.py") if path.name != "validate_delivery.py") + [ROOT / "validate_delivery.py"]


def check_python_syntax(errors: list[str]) -> None:
    for path in python_files():
        try:
            source = path.read_text(encoding="utf-8")
            compile(source, str(path), "exec")
        except Exception as exc:  # noqa: BLE001
            fail(errors, f"Python 语法检查失败: {rel(path)} :: {exc}")


def check_comment_coverage(errors: list[str], warnings: list[str]) -> None:
    """用“有 docstring 的函数/类占比”作为交付注释覆盖率口径。

    该口径比简单的 `#` 行数更适合本工程：核心说明集中在模块、函数、类 docstring 和少量
    关键代码块注释中。目标阈值按初赛要求设为 30%。
    """
    total_defs = 0
    documented_defs = 0
    low_files: list[str] = []
    for path in python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        defs = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]
        if not defs:
            continue
        documented = [node for node in defs if ast.get_docstring(node)]
        total_defs += len(defs)
        documented_defs += len(documented)
        ratio = len(documented) / len(defs)
        if ratio < 0.20 and path.name not in {"validate_delivery.py"}:
            low_files.append(f"{path.name}={ratio:.0%}")

    ratio = documented_defs / total_defs if total_defs else 1.0
    if ratio < 0.30:
        fail(errors, f"注释覆盖率不足 30%: {ratio:.1%} ({documented_defs}/{total_defs})")
    elif low_files:
        warn(warnings, "部分辅助脚本文档化比例偏低，但整体覆盖率达标: " + ", ".join(low_files))
    print(f"comment_coverage={ratio:.1%} documented_defs={documented_defs} total_defs={total_defs}")


def is_scanned_text_file(path: Path) -> bool:
    if LOCAL_ONLY_DIRS.intersection(path.relative_to(ROOT).parts):
        return False
    if path.name != ".env.example" and (path.name == ".env" or path.name.startswith(".env.")):
        return False
    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".faiss", ".pkl", ".pdf", ".npy"}:
        return False
    return path.is_file()


def check_secret_sprawl(errors: list[str]) -> None:
    for path in ROOT.rglob("*"):
        if not is_scanned_text_file(path):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if SECRET_RE.search(text):
            fail(errors, f"待发布文件含疑似密钥: {rel(path)}")


def check_section_summaries(errors: list[str]) -> None:
    """Require one non-error summary for every current Section and no stale keys."""
    try:
        sections = json.loads((ROOT / "data/section_chunks.json").read_text(encoding="utf-8"))
        summaries = json.loads((ROOT / "data/section_summaries.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(errors, f"Section Summary无法读取: {exc}")
        return
    expected = {f"{section['product']}|{section['section_id']}" for section in sections}
    actual = set(summaries)
    if missing := expected - actual:
        fail(errors, f"Section Summary缺失 {len(missing)} 条")
    if stale := actual - expected:
        fail(errors, f"Section Summary存在 {len(stale)} 条过期键")
    invalid = [key for key, value in summaries.items() if not value or value.startswith("__ERROR__")]
    if invalid:
        fail(errors, f"Section Summary存在 {len(invalid)} 条空值或生成错误")


def check_optional_visual_index(errors: list[str]) -> None:
    """Validate the visual pair when a real remote build has been completed."""
    index_path = ROOT / "data/visual_index/visual.faiss"
    metadata_path = ROOT / "data/visual_index/visual_metadata.json"
    if index_path.exists() != metadata_path.exists():
        fail(errors, "视觉FAISS与Metadata必须同时存在")
        return
    if not metadata_path.exists():
        return
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        items = payload["items"]
        assert payload["embedding_dimension"] == 4096
        assert len(items) == 2504
        assert [item["vector_id"] for item in items] == list(range(2504))
    except (OSError, KeyError, TypeError, AssertionError, json.JSONDecodeError) as exc:
        fail(errors, f"视觉索引Metadata无效: {exc}")


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []

    check_required(errors)
    check_executable(errors)
    check_no_generated_artifacts(errors)
    check_python_syntax(errors)
    check_comment_coverage(errors, warnings)
    check_secret_sprawl(errors)
    check_section_summaries(errors)
    check_optional_visual_index(errors)

    for message in warnings:
        print(message)
    if errors:
        for message in errors:
            print(message, file=sys.stderr)
        print(f"delivery_check=FAILED errors={len(errors)} warnings={len(warnings)}", file=sys.stderr)
        return 1

    print(f"delivery_check=OK warnings={len(warnings)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
