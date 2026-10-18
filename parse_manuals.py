"""
V3 手册解析器：直接解析 `手册_v3/*.md`，生成新的三层数据：

- data/catalog.json            — 产品目录（供 skill 章节导航）
- data/section_chunks.json     — 章节块（按 `# / ##` 真目录切）
- data/retrieval_chunks.json   — 检索块（按 section 内语义 block 中粒度切）

兼容输出：
- data/chunks.json             — retrieval_chunks 的兼容别名
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

MANUAL_DIR = Path("手册_v4")
OUTPUT_DIR = Path("data")

SECTION_FILE = "section_chunks.json"
RETRIEVAL_FILE = "retrieval_chunks.json"
CATALOG_FILE = "catalog.json"
LEGACY_FILE = "chunks.json"

WHOLE_SECTION_MAX_CHARS = 320
MIN_RETRIEVAL_CHARS = 180
TARGET_RETRIEVAL_CHARS = 240
MAX_RETRIEVAL_CHARS = 300
SPLIT_SEARCH_MARGIN = 80

# 句子感知滑动窗口（长度单位：去空白去 PIC 后字符数）
CHUNK_TARGET_UNITS = 160   # 细粒度（对标 V4 级 2-3 chunk/section、中位~200 字符）
CHUNK_OVERLAP_UNITS = 24    # ~15% 重叠
CHUNK_MIN_UNITS = 70       # 短尾并入前块，不留孤立短句
CHUNK_MAX_UNITS = 240      # 单句超此则按词/字细切

IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
PIC_REF_RE = re.compile(r"<PIC(?::([^>]+))?>")
HEADING_RE = re.compile(r"^(#{1,3})\s+(.*)$")
SPLIT_BOUNDARY_RE = re.compile(r"\n{2,}|[。！？!?；;]\s*|[:：]\s+|\.\s+|\n")
CLAUSE_BOUNDARY_RE = re.compile(r"[，,、]\s*")
_CJK_CHAR_RE = re.compile(r"[一-鿿]")
_WORD_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['._-][A-Za-z0-9]+)*")
LIST_LINE_RE = re.compile(r"^\s*(?:[-*+]|(?:\d+\.))\s+")
TABLE_LINE_RE = re.compile(r"^\s*\|.*\|\s*$")
ONLY_PIC_LINE_RE = re.compile(r"^(?:<PIC>\s*)+$")

SPECIAL_HEADING_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\bcontents?\b",
        r"\btable of contents\b",
        r"\bindex\b",
        r"\bsafety\b",
        r"\bwarning\b",
        r"\bcaution\b",
        r"\bdanger\b",
        r"\bimportant\b",
        r"\bnotice\b",
        r"目录",
        r"注意事项",
        r"安全",
        r"警告",
        r"危险",
        r"声明",
    ]
]


def contains_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def infer_lang(product: str, text: str) -> str:
    if contains_cjk(product):
        return "zh"
    sample = text[:800]
    return "zh" if contains_cjk(sample) else "en"


def extract_pic_name(candidate: str, fallback: str = "") -> str | None:
    raw = (candidate or fallback).strip()
    if not raw:
        return None
    raw = raw.split("?", 1)[0].split("#", 1)[0].strip()
    if "/" in raw or "\\" in raw:
        raw = Path(raw).name
    stem = Path(raw).stem.strip()
    return stem or None


def replace_inline_assets(raw: str) -> tuple[str, list[str]]:
    pics: list[str] = []

    def image_repl(match: re.Match[str]) -> str:
        alt, url = match.group(1), match.group(2)
        pic_name = extract_pic_name(url, alt)
        if pic_name:
            pics.append(pic_name)
        return " <PIC> "

    def pic_repl(match: re.Match[str]) -> str:
        pic_name = extract_pic_name(match.group(1) or "")
        if pic_name:
            pics.append(pic_name)
        return " <PIC> "

    text = IMAGE_RE.sub(image_repl, raw)
    text = PIC_REF_RE.sub(pic_repl, text)
    return text, pics


def clean_markdown_line(line: str) -> str:
    line = line.rstrip()
    line = re.sub(r"^\s*>\s?", "", line)
    line = re.sub(r"^\s*[-*+]\s+", "• ", line)
    line = re.sub(r"^\s*(\d+)\.\s+", r"\1. ", line)
    line = line.replace("**", "").replace("__", "")
    line = re.sub(r"`([^`]*)`", r"\1", line)
    line = re.sub(r"\\([*_`\[\]()])", r"\1", line)
    line = re.sub(r"\s*<PIC>\s*", " <PIC> ", line)
    line = re.sub(r"[ \t]+", " ", line)
    return line.strip()


def normalize_section_body(lines: list[str]) -> tuple[str, list[str]]:
    raw = "\n".join(lines).strip()
    if not raw:
        return "", []

    text, pics = replace_inline_assets(raw)

    normalized_lines: list[str] = []
    for line in text.splitlines():
        cleaned = clean_markdown_line(line)
        if not cleaned:
            if normalized_lines and normalized_lines[-1] != "":
                normalized_lines.append("")
            continue

        if ONLY_PIC_LINE_RE.fullmatch(cleaned):
            if normalized_lines:
                normalized_lines[-1] = normalized_lines[-1].rstrip() + "\n" + cleaned
            else:
                normalized_lines.append(cleaned)
            continue

        normalized_lines.append(cleaned)

    while normalized_lines and normalized_lines[0] == "":
        normalized_lines.pop(0)
    while normalized_lines and normalized_lines[-1] == "":
        normalized_lines.pop()

    body = "\n".join(normalized_lines)
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    return body, pics


def summarize_text(text: str, limit: int = 120) -> str:
    cleaned = re.sub(r"\s+", " ", text.replace("<PIC>", " ")).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + "..."


def dense_text(text: str) -> str:
    return re.sub(r"\s+", "", text.replace("<PIC>", ""))


# 图说明兜底：让"短/纯图" section 靠图的 caption 进入检索文本，避免召不回
try:
    _IMAGE_CAPTIONS = json.loads(
        (OUTPUT_DIR / "image_captions_v4_final.json").read_text(encoding="utf-8")
    ).get("items", {})
except Exception:
    _IMAGE_CAPTIONS = {}


def caption_aux_text(product: str, pics: list[str]) -> str:
    """第二变量：三类有用图 caption 都进检索文本（只跳 noise）。
    - info_table → content（表格/规格/标签 OCR 全文，召回主力）
    - part_view / schematic → short_caption + content（部件名/操作名 + 视觉描述，
      让操作图/部件图也能被关键词召回，针对 90/204 这类 part_view 操作图掉图）。
    BM25 吃完整文本，dense 由 _embed_corpus 按行切片 mean-pooling，长表不超 ubatch。"""
    parts: list[str] = []
    for pic in pics or []:
        item = _IMAGE_CAPTIONS.get(f"{product}|{pic}")
        if not item:
            continue
        cat = item.get("category")
        if cat == "noise":
            continue
        short = (item.get("short_caption") or "").strip()
        content = (item.get("content") or "").strip()
        if cat == "info_table":
            seg = content
        else:  # part_view / schematic
            seg = f"{short} {content}".strip()
        if seg:
            parts.append(seg)
    return " ".join(parts).strip()


def approx_units(text: str) -> int:
    """切分长度单位：去图去空白后的字符数（中英文统一）。
    不断英文单词由切分点保证（preferred_split_end / _word_safe_end 落在句/词边界）。"""
    return len(re.sub(r"\s", "", text.replace("<PIC>", "")))


def is_special_heading(heading: str) -> bool:
    heading = heading.strip()
    return any(pattern.search(heading) for pattern in SPECIAL_HEADING_PATTERNS)


def extract_tags(heading: str, text: str, is_special: bool) -> list[str]:
    tags = []
    heading_lower = heading.lower()
    text_lower = text.lower()

    if is_special:
        tags.append("special")
    if "目录" in heading or "contents" in heading_lower or "index" in heading_lower:
        tags.append("contents")
    if any(token in heading for token in ["安全", "警告", "危险", "注意"]) or any(
        token in heading_lower for token in ["safety", "warning", "caution", "danger", "important", "notice"]
    ):
        tags.append("warning")
    if any(token in heading for token in ["步骤", "安装", "使用", "操作", "更换", "清洁"]) or any(
        token in text for token in ["步骤", "安装", "使用", "操作", "更换", "清洁"]
    ) or any(
        token in heading_lower for token in ["install", "operation", "setup", "replace", "cleaning", "how to"]
    ):
        tags.append("procedure")
    if any(token in heading for token in ["参数", "规格", "尺寸", "容量", "环境条件"]) or any(
        token in heading_lower for token in ["specification", "specifications", "size", "capacity", "dimension"]
    ):
        tags.append("spec")
    if any(token in heading for token in ["故障", "异常", "报错", "问题"]) or any(
        token in text for token in ["故障", "异常", "报错", "问题"]
    ) or any(
        token in heading_lower for token in ["troubleshooting", "problem", "problems", "error", "fault"]
    ):
        tags.append("troubleshooting")
    if any(token in heading for token in ["部件", "结构", "概览", "按键", "显示", "控制"]) or any(
        token in heading_lower for token in ["overview", "parts", "component", "components", "controls", "display"]
    ):
        tags.append("parts")
    if "?" in text or "？" in text or "faq" in heading_lower:
        tags.append("faq_like")
    if "<PIC>" in text:
        tags.append("has_pic")

    return sorted(set(tags))


def heading_label(doc_title: str | None, current_h1: str | None, current_h2: str | None, current_h3: str | None) -> tuple[str, list[str]]:
    path: list[str] = []
    if current_h1 and current_h1 != doc_title:
        path.append(current_h1)
    if current_h2:
        path.append(current_h2)
    if current_h3:
        path.append(current_h3)

    if path:
        return " / ".join(path), path

    return "(前言)", []


def parse_markdown_manual(path: Path) -> tuple[str, str, list[dict], list[str], int]:
    product = path.stem
    raw_text = path.read_text(encoding="utf-8")
    lang = infer_lang(product, raw_text)

    doc_title: str | None = None
    current_h1: str | None = None
    current_h2: str | None = None
    current_h3: str | None = None
    buffer: list[str] = []

    sections: list[dict] = []
    all_pics: list[str] = []

    def flush_section() -> None:
        nonlocal buffer
        body, pics = normalize_section_body(buffer)
        buffer = []
        if not body:
            return

        heading, heading_path = heading_label(doc_title, current_h1, current_h2, current_h3)
        is_special = is_special_heading(heading)
        all_pics.extend(pics)

        sections.append(
            {
                "product": product,
                "lang": lang,
                "section_id": len(sections),
                "heading": heading,
                "heading_path": heading_path,
                "chapter": current_h1 if current_h1 and current_h1 != doc_title else None,
                "subheading": current_h2,
                "subsubheading": current_h3,
                "heading_level": len(heading_path),
                "text": body,
                "pics": pics,
                "char_len": len(body),
                "pic_count": len(pics),
                "is_special": is_special,
                "summary": summarize_text(body),
                "tags": extract_tags(heading, body, is_special),
            }
        )

    for line in raw_text.splitlines():
        match = HEADING_RE.match(line)
        if match:
            level = len(match.group(1))
            title = match.group(2).strip()
            flush_section()
            if level == 1:
                if doc_title is None:
                    doc_title = title
                current_h1 = title
                current_h2 = None
                current_h3 = None
            elif level == 2:
                current_h2 = title
                current_h3 = None
            else:
                current_h3 = title
            continue

        buffer.append(line)

    flush_section()

    total_chars = sum(section["char_len"] for section in sections)
    return product, lang, sections, all_pics, total_chars


def pic_spans(text: str, pics: list[str]) -> list[tuple[int, int, str]]:
    spans = []
    for i, match in enumerate(re.finditer(r"<PIC>", text)):
        pic_name = pics[i] if i < len(pics) else None
        if pic_name is not None:
            spans.append((match.start(), match.end(), pic_name))
    return spans


def select_pics_for_span(text: str, pics: list[str], start: int, end: int) -> list[str]:
    selected = []
    for pic_start, pic_end, pic_name in pic_spans(text, pics):
        if pic_start >= start and pic_end <= end:
            selected.append(pic_name)
    return selected


def _word_safe_end(text: str, lo: int, hard_end: int) -> int:
    """在 [lo, hard_end] 找一个不切断英文单词/数字 token 的断点。"""
    if hard_end <= lo:
        return hard_end
    # 优先在空格处断开（英文单词之间）
    for k in range(hard_end, lo, -1):
        if text[k - 1].isspace():
            return k
    # 无空格（如纯中文）：在「非连续 ASCII 字母数字」处断，避免切断英文/数字 token。
    # k == len(text) 时没有右侧字符，说明已经到文本末尾，直接返回即可。
    max_k = min(hard_end, len(text) - 1)
    for k in range(max_k, lo, -1):
        a, b = text[k - 1], text[k]
        if not (a.isascii() and a.isalnum() and b.isascii() and b.isalnum()):
            return k
    return hard_end


def preferred_split_end(text: str, start: int, hard_end: int) -> int:
    """分级找切分点：句子边界 > 子句边界 > 词/字边界（英文绝不切断单词）。"""
    lo = start + MIN_RETRIEVAL_CHARS
    if lo >= hard_end:
        return _word_safe_end(text, start, hard_end)
    # 1) 句子边界（句号/问号/感叹号/分号/冒号/换行），取范围内最靠后的
    best_end = None
    for match in SPLIT_BOUNDARY_RE.finditer(text, lo, hard_end):
        best_end = match.end()
    if best_end is not None:
        return best_end
    # 2) 子句边界（逗号/顿号）
    for match in CLAUSE_BOUNDARY_RE.finditer(text, lo, hard_end):
        best_end = match.end()
    if best_end is not None:
        return best_end
    # 3) 词/字边界，英文不切断单词
    return _word_safe_end(text, lo, hard_end)


def split_long_text(text: str, pics: list[str]) -> list[dict]:
    if len(text) <= MAX_RETRIEVAL_CHARS:
        return [{"text": text, "pics": pics[:], "char_start": 0, "char_end": len(text)}]

    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        hard_end = min(start + MAX_RETRIEVAL_CHARS, text_len)
        end = hard_end if hard_end == text_len else preferred_split_end(text, start, hard_end)
        if end <= start:
            end = hard_end

        raw = text[start:end]
        leading = len(raw) - len(raw.lstrip())
        trailing = len(raw) - len(raw.rstrip())
        adj_start = start + leading
        adj_end = end - trailing
        chunk_text = text[adj_start:adj_end].strip()
        if chunk_text:
            chunks.append(
                {
                    "text": chunk_text,
                    "pics": select_pics_for_span(text, pics, adj_start, adj_end),
                    "char_start": adj_start,
                    "char_end": adj_end,
                }
            )

        if end >= text_len:
            break
        start = end

    return chunks


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    """按句子边界切，返回每句的 (start, end) 字符范围。"""
    spans: list[tuple[int, int]] = []
    last = 0
    for m in SPLIT_BOUNDARY_RE.finditer(text):
        if text[last:m.end()].strip():
            spans.append((last, m.end()))
        last = m.end()
    if text[last:].strip():
        spans.append((last, len(text)))
    return spans


def _split_long_span(text: str, s: int, e: int) -> list[tuple[int, int]]:
    """超长无标点句子：按词/字边界细切，不切断英文单词。"""
    spans: list[tuple[int, int]] = []
    start = s
    while start < e:
        if approx_units(text[start:e]) <= CHUNK_TARGET_UNITS:
            spans.append((start, e))
            break
        hard = min(start + CHUNK_TARGET_UNITS * 4, e)  # 粗略字符上限
        cut = _word_safe_end(text, start + 1, hard)
        if cut <= start:
            cut = hard
        spans.append((start, cut))
        start = cut
    return spans


def merge_pic_only_lines(lines: list[str]) -> list[str]:
    merged: list[str] = []
    for line in lines:
        compact = line.strip()
        if not compact:
            continue
        if ONLY_PIC_LINE_RE.fullmatch(compact):
            if merged:
                merged[-1] = merged[-1].rstrip() + "\n" + compact
            else:
                merged.append(compact)
            continue
        merged.append(compact)
    return merged


def expand_structured_block(raw_block: str) -> list[str]:
    lines = merge_pic_only_lines(raw_block.splitlines())
    if not lines:
        return []

    if len(lines) >= 2 and all(TABLE_LINE_RE.match(line) for line in lines):
        return ["\n".join(lines)]

    list_like_count = sum(1 for line in lines if LIST_LINE_RE.match(line))
    if list_like_count >= 2:
        expanded: list[str] = []
        current: list[str] = []
        for line in lines:
            if LIST_LINE_RE.match(line):
                if current:
                    expanded.append("\n".join(current).strip())
                    current = []
                expanded.append(line.strip())
            else:
                current.append(line.strip())
        if current:
            expanded.append("\n".join(current).strip())
        return [block for block in expanded if block]

    return ["\n".join(lines).strip()]


def split_section_into_blocks(section: dict) -> list[dict]:
    raw_blocks = re.split(r"\n{2,}", section["text"].strip())
    blocks: list[dict] = []
    pic_offset = 0

    for raw_block in raw_blocks:
        for block_text in expand_structured_block(raw_block):
            block_text = block_text.strip()
            if not block_text:
                continue

            pic_count = block_text.count("<PIC>")
            block_pics = section["pics"][pic_offset:pic_offset + pic_count]
            pic_offset += pic_count
            blocks.append({"text": block_text, "pics": block_pics})

    return blocks


def build_candidate_chunk(
    section: dict,
    text: str,
    pics: list[str],
    split_kind: str,
    subchunk_id: int,
    char_start: int | None = None,
    char_end: int | None = None,
) -> dict:
    return {
        "product": section["product"],
        "lang": section["lang"],
        "parent_section_id": section["section_id"],
        "source_section_ids": [section["section_id"]],
        "subchunk_id": subchunk_id,
        "heading": section["heading"],
        "text": text,
        "pics": pics,
        "char_start": char_start,
        "char_end": char_end,
        "char_len": len(text),
        "pic_count": len(pics),
        "is_special": section["is_special"],
        "summary": summarize_text(text),
        "tags": section["tags"][:],
        "split_kind": split_kind,
    }


def build_section_retrieval_candidates(section: dict) -> list[dict]:
    """句子感知滑动窗口切分：按句子滚动组装到目标 size，相邻 chunk 重叠 overlap，
    切分点只落在句/子句/词边界（英文不切断单词），不留孤立短尾。"""
    text = section["text"].strip()
    if not text:
        return []
    pics = section["pics"]

    # 整段不超目标 → 单块
    if approx_units(text) <= CHUNK_TARGET_UNITS:
        return [
            build_candidate_chunk(
                section=section,
                text=text,
                pics=pics[:],
                split_kind="whole_section",
                subchunk_id=0,
                char_start=0,
                char_end=len(text),
            )
        ]

    # 句子级 span（超长无标点句子再按词/字细切）
    spans: list[tuple[int, int]] = []
    for (s, e) in _sentence_spans(text):
        if approx_units(text[s:e]) > CHUNK_MAX_UNITS:
            spans.extend(_split_long_span(text, s, e))
        else:
            spans.append((s, e))
    units = [approx_units(text[s:e]) for (s, e) in spans]

    candidates: list[dict] = []
    subchunk_id = 0
    i = 0
    n = len(spans)
    while i < n:
        cur = 0
        j = i
        while j < n and (cur < CHUNK_TARGET_UNITS or j == i):
            cur += units[j]
            j += 1
        # 防短尾：剩余不足一个最小块时全部并入当前块
        if 0 < sum(units[j:]) < CHUNK_MIN_UNITS:
            j = n
        cs, ce = spans[i][0], spans[j - 1][1]
        seg = text[cs:ce]
        adj_s = cs + (len(seg) - len(seg.lstrip()))
        adj_e = ce - (len(seg) - len(seg.rstrip()))
        chunk_text = text[adj_s:adj_e].strip()
        if chunk_text:
            candidates.append(
                build_candidate_chunk(
                    section=section,
                    text=chunk_text,
                    pics=select_pics_for_span(text, pics, adj_s, adj_e),
                    split_kind="window",
                    subchunk_id=subchunk_id,
                    char_start=adj_s,
                    char_end=adj_e,
                )
            )
            subchunk_id += 1
        if j >= n:
            break
        # 滑动窗口：从块尾回退 overlap 个单位作为下一块起点
        back = 0
        k = j
        while k - 1 > i and back < CHUNK_OVERLAP_UNITS:
            k -= 1
            back += units[k]
        i = k if k > i else i + 1

    return candidates


def merge_small_candidates(candidates: list[dict]) -> list[dict]:
    merged = []
    i = 0

    while i < len(candidates):
        current = candidates[i]
        if current["char_len"] >= MIN_RETRIEVAL_CHARS or current["is_special"] or current.get("split_kind") == "whole_section":
            merged.append(current)
            i += 1
            continue

        bundle = [current]
        bundle_len = current["char_len"]
        j = i + 1
        while j < len(candidates):
            nxt = candidates[j]
            if nxt["product"] != current["product"] or nxt["parent_section_id"] != current["parent_section_id"]:
                break
            if nxt["is_special"]:
                break
            bundle.append(nxt)
            bundle_len += 2 + nxt["char_len"]
            j += 1
            if bundle_len >= MIN_RETRIEVAL_CHARS:
                break

        if len(bundle) == 1:
            merged.append(current)
            i += 1
            continue

        merged_text = "\n\n".join(item["text"] for item in bundle if item["text"])
        merged_pics: list[str] = []
        merged_tags = set()
        source_section_ids = []
        headings = []
        for item in bundle:
            merged_pics.extend(item["pics"])
            merged_tags.update(item["tags"])
            source_section_ids.extend(item["source_section_ids"])
            headings.append(item["heading"])

        merged.append(
            {
                "product": bundle[0]["product"],
                "lang": bundle[0]["lang"],
                "parent_section_id": bundle[0]["parent_section_id"],
                "source_section_ids": sorted(set(source_section_ids)),
                "subchunk_id": bundle[0]["subchunk_id"],
                "heading": " / ".join(dict.fromkeys(headings)),
                "text": merged_text,
                "pics": merged_pics,
                "char_start": None,
                "char_end": None,
                "char_len": len(merged_text),
                "pic_count": len(merged_pics),
                "is_special": False,
                "summary": summarize_text(merged_text),
                "tags": sorted(merged_tags | {"merged"}),
                "split_kind": "merged",
            }
        )
        i = j

    return merged


def build_retrieval_chunks(section_chunks: list[dict]) -> list[dict]:
    candidates = []
    for section in section_chunks:
        compact = dense_text(section["text"])
        if len(compact) < 12 and not section.get("pics"):
            continue
        candidates.extend(build_section_retrieval_candidates(section))

    retrieval_chunks = []
    for chunk_id, chunk in enumerate(candidates):
        chunk["chunk_id"] = chunk_id
        chunk["caption_aux"] = caption_aux_text(
            chunk.get("product", ""), chunk.get("pics") or []
        )
        retrieval_chunks.append(chunk)

    return retrieval_chunks


def catalog_entry(
    section_chunks: list[dict],
    retrieval_chunks: list[dict],
    product_text_len: int,
    pics: list[str],
    lang: str,
) -> dict:
    return {
        "lang": lang,
        "total_chars": product_text_len,
        "total_pics": len(pics),
        "section_count": len(section_chunks),
        "retrieval_chunk_count": len(retrieval_chunks),
        "sections": [
            {
                "id": section["section_id"],
                "title": section["heading"],
                "summary": section["summary"],
                "char_len": section["char_len"],
                "pic_count": section["pic_count"],
                "tags": section["tags"],
            }
            for section in section_chunks
        ],
    }


def validate_pic_alignment(product: str, section_chunks: list[dict], original_pics: list[str]) -> None:
    total = sum(len(section["pics"]) for section in section_chunks)
    if total != len(original_pics):
        print(f"  ⚠ {product}: PIC 不匹配 section={total} vs 原始={len(original_pics)}")


def print_stats(section_chunks: list[dict], retrieval_chunks: list[dict]) -> None:
    section_sizes = [len(chunk["text"]) for chunk in section_chunks]
    retrieval_sizes = [len(chunk["text"]) for chunk in retrieval_chunks]
    total_pics = sum(len(chunk["pics"]) for chunk in section_chunks)

    print(f"\n{'=' * 50}")
    print(f"章节块: {len(section_chunks)}")
    print(
        "section 大小: "
        f"min={min(section_sizes)}, max={max(section_sizes)}, "
        f"avg={sum(section_sizes) // len(section_sizes)}, "
        f"median={sorted(section_sizes)[len(section_sizes) // 2]}"
    )
    print(f"检索块: {len(retrieval_chunks)}")
    print(
        "retrieval 大小: "
        f"min={min(retrieval_sizes)}, max={max(retrieval_sizes)}, "
        f"avg={sum(retrieval_sizes) // len(retrieval_sizes)}, "
        f"median={sorted(retrieval_sizes)[len(retrieval_sizes) // 2]}"
    )
    print(f"总 PIC: {total_pics}")
    print(
        f"\n输出: {OUTPUT_DIR / CATALOG_FILE}, "
        f"{OUTPUT_DIR / SECTION_FILE}, "
        f"{OUTPUT_DIR / RETRIEVAL_FILE}, "
        f"{OUTPUT_DIR / LEGACY_FILE}"
    )


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    all_section_chunks = []
    all_retrieval_chunks = []
    catalog = {}

    manual_files = sorted(MANUAL_DIR.glob("*.md"))
    if not manual_files:
        raise FileNotFoundError(f"未找到 markdown 手册目录: {MANUAL_DIR}")

    for path in manual_files:
        product, lang, section_chunks, pics, total_chars = parse_markdown_manual(path)
        validate_pic_alignment(product, section_chunks, pics)

        base_chunk_id = len(all_retrieval_chunks)
        retrieval_chunks = build_retrieval_chunks(section_chunks)
        for chunk in retrieval_chunks:
            chunk["chunk_id"] += base_chunk_id

        catalog[product] = catalog_entry(section_chunks, retrieval_chunks, total_chars, pics, lang)
        all_section_chunks.extend(section_chunks)
        all_retrieval_chunks.extend(retrieval_chunks)

        print(
            f"  {product}: "
            f"{len(section_chunks)} sections, {len(retrieval_chunks)} retrieval chunks, {len(pics)} pics"
        )

    with open(OUTPUT_DIR / SECTION_FILE, "w", encoding="utf-8") as f:
        json.dump(all_section_chunks, f, ensure_ascii=False, indent=2)
    with open(OUTPUT_DIR / RETRIEVAL_FILE, "w", encoding="utf-8") as f:
        json.dump(all_retrieval_chunks, f, ensure_ascii=False, indent=2)
    with open(OUTPUT_DIR / LEGACY_FILE, "w", encoding="utf-8") as f:
        json.dump(all_retrieval_chunks, f, ensure_ascii=False, indent=2)
    with open(OUTPUT_DIR / CATALOG_FILE, "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)

    print_stats(all_section_chunks, all_retrieval_chunks)


if __name__ == "__main__":
    main()
