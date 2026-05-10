from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import re
import shutil
import uuid
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import httpx
from sqlalchemy.orm import Session, joinedload

from app.core.config import settings
from app.core.prompt_manager import (
    DEFAULT_RAG_TARGET_REQUEST_BODY_TEMPLATE,
    DEFAULT_TARGET_ANSWER_ONLY_PROMPT,
    DEFAULT_TARGET_CONTEXT_PROMPT,
    QUESTION_GENERATION_PROMPT,
    QUESTION_VALIDATION_PROMPT,
    render_template_value,
)
from app.models.dataset import Dataset, DatasetRow
from app.models.llm_config import LLMConfig
from app.models.rag_dataset_job import (
    RagDatasetChunk,
    RagDatasetDocument,
    RagDatasetJob,
    RagDatasetSample,
)

logger = logging.getLogger(__name__)

RAG_JOB_TIMEOUT_SECONDS = 15 * 60
RAG_QUESTION_GENERATION_TIMEOUT_SECONDS = 120
RAG_TARGET_REQUEST_TIMEOUT_SECONDS = 180
RAG_TARGET_REQUEST_RETRIES = 3
RAG_TARGET_REQUEST_CONCURRENCY = 3

_PAGE_LINE_RE = re.compile(r"^\s*第?\s*\d+\s*页(?:\s*/\s*共?\s*\d+\s*页)?\s*$")
_DATE_RE = re.compile(r"\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日")
_SECTION_NUMBER_RE = re.compile(r"^\d+(?:\.\d+){1,4}$")
_REVISION_META_KEYWORDS = (
    "编制",
    "文件编号",
    "版本号",
    "更改页码",
    "更改条款号",
    "新建/修订",
    "审核人",
    "批准",
    "修订日期",
    "备注",
)
_FORM_SECTION_MARKERS = (
    "回执",
    "回条",
    "家长签名",
    "学生姓名",
    "签字",
    "签名",
    "报名缴费",
    "由此进入",
    "班主任",
    "不需要据下",
    "我已认真阅读",
    "我的孩子将参加",
    "原因不能参加",
    "在家学习",
    "到校学习",
    "安全管理",
)
_OCR_GARBLED_RE = re.compile(r"[A-Za-z]{2,}\s*[A-Za-z]{2,}")
_STRUCTURED_HEADING_RE = re.compile(r"^\s*(\d+(?:\.\d+){1,5})(?:[\.、．])?\s*(.+?)\s*$")
_STRUCTURED_HEADING_NUMBER_RE = re.compile(r"^\s*(\d+(?:\.\d+){1,5})\s*$")


def _normalize_pdf_line(line: str) -> str:
    return re.sub(r"\s+", " ", (line or "").strip())


def _looks_like_header_footer(text: str) -> bool:
    normalized = _normalize_pdf_line(text)
    compact = re.sub(r"\s+", "", normalized)
    if not normalized:
        return True
    if _PAGE_LINE_RE.match(normalized):
        return True
    if "第" in normalized and "页" in normalized and _DATE_RE.search(normalized):
        return True
    if _SECTION_NUMBER_RE.match(compact):
        return True
    if _DATE_RE.search(normalized) and len(normalized) <= 40:
        return True
    if len(compact) <= 12 and not re.search(r"[\u4e00-\u9fffA-Za-z]{4,}", compact):
        return True
    return False


def _looks_like_revision_metadata(text: str) -> bool:
    normalized = _normalize_pdf_line(text)
    keyword_hits = sum(1 for keyword in _REVISION_META_KEYWORDS if keyword in normalized)
    if len(normalized) <= 180 and keyword_hits >= 2:
        return True
    if len(normalized) <= 180 and normalized.startswith("注：1.") and "修订" in normalized:
        return True
    if len(normalized) <= 180 and normalized.startswith("注：") and ("修订" in normalized or "审核" in normalized or "批准" in normalized):
        return True
    if len(normalized) <= 180 and re.match(r"^\d+\.", normalized) and ("修订" in normalized or "审核" in normalized or "批准" in normalized):
        return True
    return False


def _looks_like_form_marker(text: str) -> bool:
    normalized = _normalize_pdf_line(text)
    return any(marker in normalized for marker in _FORM_SECTION_MARKERS)


def _looks_like_ocr_garble(text: str) -> bool:
    normalized = _normalize_pdf_line(text)
    if not normalized:
        return True
    compact = re.sub(r"\s+", "", normalized)
    cjk_chars = len(re.findall(r"[\u4e00-\u9fff]", compact))
    ascii_chars = len(re.findall(r"[A-Za-z]", compact))
    digits = len(re.findall(r"\d", compact))
    punct = len(re.findall(r"[^A-Za-z0-9\u4e00-\u9fff]", compact))
    if ascii_chars >= 6 and cjk_chars <= 4 and digits <= 2:
        return True
    if _OCR_GARBLED_RE.search(normalized) and cjk_chars <= 6:
        return True
    if punct >= max(4, len(compact) * 0.25) and cjk_chars <= 6:
        return True
    return False


def _prune_form_tail(lines: list[str]) -> list[str]:
    if not lines:
        return lines
    marker_index: int | None = None
    for idx, line in enumerate(lines):
        if idx < max(3, int(len(lines) * 0.4)):
            continue
        if _looks_like_form_marker(line):
            marker_index = idx
            break
    if marker_index is None:
        return lines
    return lines[:marker_index]


def evaluate_chunk_quality(text: str) -> dict[str, Any]:
    normalized_text = _normalize_pdf_line(text)
    compact = re.sub(r"\s+", "", normalized_text)
    cjk_chars = len(re.findall(r"[\u4e00-\u9fff]", compact))
    word_chars = len(re.findall(r"[A-Za-z]", compact))
    digits = len(re.findall(r"\d", compact))
    sentence_markers = len(re.findall(r"[。！？.!?；;:：]", text))
    reasons: list[str] = []
    score = 1.0

    if _looks_like_header_footer(normalized_text):
        return {"score": 0.0, "label": "filtered", "reasons": ["页眉页脚/页码噪声"]}
    if _looks_like_revision_metadata(normalized_text):
        return {"score": 0.05, "label": "filtered", "reasons": ["制度修订记录或版本元信息"]}
    if _looks_like_form_marker(normalized_text):
        return {"score": 0.05, "label": "filtered", "reasons": ["表单/回执字段"]}
    if _looks_like_ocr_garble(normalized_text):
        return {"score": 0.15, "label": "filtered", "reasons": ["OCR 乱码或低可读文本"]}

    if len(compact) < 40:
        score -= 0.45
        reasons.append("文本过短")
    elif len(compact) < 90:
        score -= 0.15
        reasons.append("信息量偏少")

    if cjk_chars < 12 and word_chars < 24:
        score -= 0.25
        reasons.append("正文特征较弱")

    if digits >= max(8, int(len(compact) * 0.28)) and sentence_markers == 0:
        score -= 0.2
        reasons.append("编号/数字占比偏高")

    if sentence_markers == 0 and len(compact) < 140:
        score -= 0.15
        reasons.append("缺少完整句子")

    if len(compact) < 220 and re.search(r"(因|为|及|等|并|或|由|按)$", compact):
        score -= 0.2
        reasons.append("疑似半句尾段")

    if re.search(r"(步骤|流程|注意事项|条件|规则|要求|说明|如下|适用|负责|应当|可以|不得)", normalized_text):
        score += 0.12
        reasons.append("包含规则/流程型正文")

    if len(compact) >= 220:
        score += 0.08
        reasons.append("正文信息较充足")

    score = max(0.0, min(1.0, round(score, 2)))
    if score >= 0.75:
        label = "good"
    elif score >= 0.45:
        label = "medium"
    elif score >= 0.2:
        label = "low"
    else:
        label = "filtered"

    return {"score": score, "label": label, "reasons": reasons or ["普通正文"]}


def _is_valid_structured_heading(number: str, title: str) -> bool:
    title_text = _normalize_pdf_line(title)
    compact = re.sub(r"\s+", "", title_text)
    if not compact:
        return False
    if _looks_like_revision_metadata(f"{number} {title_text}"):
        return False
    if _looks_like_form_marker(title_text):
        return False
    if len(compact) <= 2 and not re.search(r"[\u4e00-\u9fffA-Za-z]{2,}", compact):
        return False
    return True


def _make_section_node(number: str, heading_line: str) -> dict[str, Any]:
    return {
        "number": number,
        "heading_line": heading_line.strip(),
        "body_lines": [],
        "children": [],
    }


def _parse_structured_sections(text: str) -> dict[str, Any]:
    root = _make_section_node("", "")
    current = root
    stack: list[tuple[int, dict[str, Any]]] = [(0, root)]
    pending_heading_number: str | None = None

    for raw_line in text.split("\n"):
        line = _normalize_pdf_line(raw_line)
        if not line:
            continue

        only_number = _STRUCTURED_HEADING_NUMBER_RE.match(line)
        if only_number:
            pending_heading_number = only_number.group(1)
            continue

        if pending_heading_number:
            if _is_valid_structured_heading(pending_heading_number, line):
                number = pending_heading_number
                level = number.count(".") + 1
                heading_line = f"{number} {line}".strip()
                node = _make_section_node(number, heading_line)
                while stack and stack[-1][0] >= level:
                    stack.pop()
                parent = stack[-1][1] if stack else root
                parent["children"].append(node)
                stack.append((level, node))
                current = node
                pending_heading_number = None
                continue
            current["body_lines"].append(pending_heading_number)
            pending_heading_number = None

        matched = _STRUCTURED_HEADING_RE.match(line)
        if matched and _is_valid_structured_heading(matched.group(1), matched.group(2)):
            number = matched.group(1)
            title = matched.group(2).strip()
            level = number.count(".") + 1
            node = _make_section_node(number, f"{number} {title}".strip())
            while stack and stack[-1][0] >= level:
                stack.pop()
            parent = stack[-1][1] if stack else root
            parent["children"].append(node)
            stack.append((level, node))
            current = node
            continue

        current["body_lines"].append(line)

    if pending_heading_number:
        current["body_lines"].append(pending_heading_number)

    return root


def _node_text(node: dict[str, Any]) -> str:
    lines: list[str] = []
    if node.get("heading_line"):
        lines.append(node["heading_line"])
    lines.extend(node.get("body_lines") or [])
    return "\n".join(line for line in lines if line).strip()


def _node_is_substantive(node: dict[str, Any]) -> bool:
    text = _node_text(node)
    compact = re.sub(r"\s+", "", text)
    return bool(text) and len(compact) >= 24 and not _is_low_information_text(text)


def _collect_structured_section_entries(
    node: dict[str, Any],
    context_headings: tuple[str, ...],
    entries: list[dict[str, Any]],
) -> None:
    node_text = _node_text(node)
    if node.get("number"):
        parent_key = node["number"].rsplit(".", 1)[0] if "." in node["number"] else node["number"]
    else:
        parent_key = "intro"

    if node.get("heading_line") and not node.get("children"):
        lines = [*context_headings[-2:], node_text]
        entries.append(
            {
                "number": node.get("number") or "intro",
                "parent_key": parent_key,
                "text": "\n".join(line for line in lines if line).strip(),
            }
        )
        return

    next_context = context_headings
    if node.get("heading_line"):
        next_context = (*context_headings, node["heading_line"])
        if _node_is_substantive(node):
            lines = [*context_headings[-2:], node_text]
            entries.append(
                {
                    "number": node.get("number") or "intro",
                    "parent_key": parent_key,
                    "text": "\n".join(line for line in lines if line).strip(),
                }
            )

    for child in node.get("children") or []:
        _collect_structured_section_entries(child, next_context, entries)


def _length_chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    cleaned = text.replace("\r\n", "\n").strip()
    if not cleaned:
        return []
    if len(cleaned) <= chunk_size:
        return [cleaned]

    paragraphs = [part.strip() for part in cleaned.split("\n") if part.strip()]
    chunks: list[str] = []
    current = ""

    def _flush() -> None:
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
        current = ""

    for paragraph in paragraphs:
        candidate = f"{current}\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            _flush()

        remaining = paragraph
        while len(remaining) > chunk_size:
            cut = chunk_size
            for sep in ["\n", "。", "；", ".", ";", "，", ",", " "]:
                pos = remaining[:chunk_size].rfind(sep)
                if pos > int(chunk_size * 0.45):
                    cut = pos + 1
                    break
            piece = remaining[:cut].strip()
            if piece:
                chunks.append(piece)
            remaining = remaining[max(0, cut - overlap):].strip()
        current = remaining

    _flush()
    return [chunk for chunk in chunks if chunk]


def _merge_structured_entry_text(current_text: str, next_text: str) -> str:
    current_lines = [line.strip() for line in current_text.split("\n") if line.strip()]
    next_lines = [line.strip() for line in next_text.split("\n") if line.strip()]
    if not current_lines:
        return "\n".join(next_lines).strip()
    if not next_lines:
        return "\n".join(current_lines).strip()

    merged_lines = list(current_lines)
    start_index = 0
    if (
        next_lines
        and _STRUCTURED_HEADING_RE.match(next_lines[0])
        and next_lines[0] in merged_lines
    ):
        start_index = 1
    while (
        start_index < len(next_lines)
        and start_index < len(current_lines)
        and next_lines[start_index] == current_lines[start_index]
    ):
        start_index += 1
    merged_lines.extend(next_lines[start_index:])
    return "\n".join(merged_lines).strip()


def _split_structured_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    root = _parse_structured_sections(text)
    entries: list[dict[str, Any]] = []

    intro_text = "\n".join(root.get("body_lines") or []).strip()
    if intro_text and not _is_low_information_text(intro_text):
        entries.append({"number": "intro", "parent_key": "intro", "text": intro_text})

    for child in root.get("children") or []:
        _collect_structured_section_entries(child, (), entries)

    entries = [entry for entry in entries if entry.get("text")]
    if len(entries) < 2:
        return []

    if (
        len(entries) >= 2
        and entries[0].get("number") == "intro"
        and len(re.sub(r"\s+", "", entries[0]["text"])) < max(80, int(chunk_size * 0.18))
    ):
        entries[1]["text"] = f"{entries[0]['text'].rstrip()}\n{entries[1]['text'].lstrip()}".strip()
        entries = entries[1:]

    merged: list[str] = []
    current = entries[0].copy()
    for entry in entries[1:]:
        current_len = len(current["text"])
        next_len = len(entry["text"])
        same_parent = current.get("parent_key") == entry.get("parent_key")
        should_merge = (
            same_parent
            and (
                current_len < max(320, int(chunk_size * 0.38))
                or next_len < max(180, int(chunk_size * 0.22))
            )
            and current_len + next_len <= int(chunk_size * 1.35)
        )
        if should_merge:
            current["text"] = _merge_structured_entry_text(current["text"], entry["text"])
        else:
            merged.append(current["text"])
            current = entry.copy()
    merged.append(current["text"])

    final_chunks: list[str] = []
    for entry_text in merged:
        if len(entry_text) <= int(chunk_size * 1.2):
            final_chunks.append(entry_text.strip())
        else:
            final_chunks.extend(_length_chunk_text(entry_text, chunk_size, overlap))
    return [chunk for chunk in final_chunks if chunk]


def _is_pdf_noise_line(line: str, repeated_lines: set[str]) -> bool:
    normalized = _normalize_pdf_line(line)
    if not normalized:
        return True
    if _looks_like_header_footer(normalized):
        return True
    if _looks_like_revision_metadata(normalized):
        return True
    if normalized in repeated_lines and len(normalized) <= 120:
        return True
    compact = re.sub(r"\s+", "", normalized)
    if len(compact) <= 8 and not re.search(r"[\u4e00-\u9fffA-Za-z]{4,}", compact):
        return True
    return False


def _extract_pdf_with_fitz(content: bytes) -> str:
    try:
        import fitz  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ValueError("当前环境未安装 PyMuPDF 解析依赖，无法执行版面感知 PDF 提取") from exc

    doc = fitz.open(stream=content, filetype="pdf")
    try:
        repeated_blocks: dict[str, int] = {}
        page_blocks: list[list[tuple[str, float]]] = []
        for page in doc:
            page_height = float(page.rect.height or 0)
            top_band = page_height * 0.12
            bottom_band = page_height * 0.88
            raw_blocks = page.get_text("blocks", sort=True) or []
            kept_blocks: list[tuple[str, float]] = []
            for block in raw_blocks:
                if len(block) < 5:
                    continue
                x0, y0, x1, y1, text = block[:5]
                normalized = _normalize_pdf_line(str(text))
                if not normalized:
                    continue
                block_height = max(1.0, float(y1) - float(y0))
                block_center = (float(y0) + float(y1)) / 2
                in_margin = block_center <= top_band or block_center >= bottom_band
                if in_margin and _looks_like_header_footer(normalized):
                    continue
                if _looks_like_header_footer(normalized) and len(normalized) <= 100:
                    continue
                density = len(re.sub(r"\s+", "", normalized)) / max(1.0, float(x1) - float(x0))
                if density < 0.015 and len(normalized) < 24:
                    continue
                kept_blocks.append((normalized, block_height))
                repeated_blocks[normalized] = repeated_blocks.get(normalized, 0) + 1
            page_blocks.append(kept_blocks)

        repeated_texts = {
            text
            for text, count in repeated_blocks.items()
            if count >= 2 and (count / max(1, len(doc))) >= 0.5 and len(text) <= 120
        }

        cleaned_pages: list[str] = []
        for blocks in page_blocks:
            page_lines: list[str] = []
            for normalized, _block_height in blocks:
                if normalized in repeated_texts and _looks_like_header_footer(normalized):
                    continue
                if _is_low_information_text(normalized):
                    continue
                page_lines.append(normalized)
            page_lines = _prune_form_tail(page_lines)
            page_text = "\n".join(page_lines).strip()
            if page_text:
                cleaned_pages.append(page_text)

        return "\n\n".join(cleaned_pages).strip()
    finally:
        doc.close()


def _ocr_pdf_with_fitz(content: bytes) -> str:
    try:
        import fitz  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ValueError("当前环境未安装 PyMuPDF，无法执行 OCR") from exc

    if shutil.which("tesseract") is None:
        raise ValueError("当前环境未安装 tesseract，无法对扫描 PDF 执行 OCR")

    doc = fitz.open(stream=content, filetype="pdf")
    try:
        page_texts: list[str] = []
        for page in doc:
            try:
                textpage = page.get_textpage_ocr(
                    flags=0,
                    language="chi_sim+eng",
                    dpi=300,
                    full=True,
                )
                text = page.get_text("text", textpage=textpage)
            except Exception:
                text = ""
            if text.strip():
                page_texts.append(text)
        return _clean_pdf_text(page_texts)
    finally:
        doc.close()


def _clean_pdf_text(page_texts: list[str]) -> str:
    if not page_texts:
        return ""

    per_page_lines: list[list[str]] = []
    frequency: dict[str, int] = {}
    for text in page_texts:
        lines = [_normalize_pdf_line(line) for line in (text or "").splitlines()]
        lines = [line for line in lines if line]
        per_page_lines.append(lines)
        for line in set(lines):
            frequency[line] = frequency.get(line, 0) + 1

    repeated_lines = {
        line
        for line, count in frequency.items()
        if count >= 2 and (count / max(1, len(page_texts))) >= 0.5
    }

    cleaned_pages: list[str] = []
    for lines in per_page_lines:
        filtered = [
            line
            for line in lines
            if not _is_pdf_noise_line(line, repeated_lines)
            and not _looks_like_form_marker(line)
            and not _looks_like_ocr_garble(line)
        ]
        filtered = _prune_form_tail(filtered)
        page_text = "\n".join(filtered).strip()
        if page_text:
            cleaned_pages.append(page_text)
    return "\n\n".join(cleaned_pages).strip()


def _is_low_information_text(text: str) -> bool:
    quality = evaluate_chunk_quality(text)
    return quality["score"] < 0.35


def parse_uploaded_file(filename: str, content: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix in {".txt", ".md", ".markdown", ".csv", ".json"}:
        for encoding in ("utf-8", "utf-8-sig", "gbk"):
            try:
                return content.decode(encoding)
            except UnicodeDecodeError:
                continue
        return content.decode("utf-8", errors="ignore")

    if suffix == ".docx":
        try:
            from docx import Document  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ValueError("当前环境未安装 docx 解析依赖，无法读取 .docx 文件") from exc
        from io import BytesIO

        doc = Document(BytesIO(content))
        return "\n".join(paragraph.text for paragraph in doc.paragraphs if paragraph.text)

    if suffix == ".pdf":
        fitz_exc: Exception | None = None
        try:
            cleaned = _extract_pdf_with_fitz(content)
            if cleaned:
                return cleaned
        except Exception as exc:  # pragma: no cover - fallback path
            fitz_exc = exc

        try:
            cleaned = _ocr_pdf_with_fitz(content)
            if cleaned:
                return cleaned
        except Exception:
            pass

        try:
            import pdfplumber  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            if fitz_exc is not None:
                raise ValueError("当前环境未安装可用的 PDF 解析依赖，无法读取 .pdf 文件") from fitz_exc
            raise ValueError("当前环境未安装 PDF 解析依赖，无法读取 .pdf 文件") from exc
        from io import BytesIO

        texts: list[str] = []
        with pdfplumber.open(BytesIO(content)) as pdf:
            for page in pdf.pages:
                texts.append(page.extract_text() or "")
        cleaned = _clean_pdf_text(texts)
        if cleaned:
            return cleaned
        return "\n".join(part for part in texts if part)

    raise ValueError(f"暂不支持的文件类型: {suffix or 'unknown'}")


def split_into_chunks(text: str, chunk_size: int = 900, overlap: int = 120) -> list[str]:
    cleaned = text.replace("\r\n", "\n").strip()
    if not cleaned:
        return []
    structured_chunks = _split_structured_chunks(cleaned, chunk_size, overlap)
    if structured_chunks:
        return structured_chunks
    if len(cleaned) <= chunk_size:
        return [cleaned]
    return _length_chunk_text(cleaned, chunk_size, overlap)


def suggest_question_count_for_chunk(content: str) -> int:
    text = content.strip()
    if not text:
        return 0
    quality = evaluate_chunk_quality(text)
    if quality["score"] < 0.35:
        return 0

    length_score = min(2.2, len(text) / 550)
    sentence_count = max(1, len(re.findall(r"[。！？.!?;\n]", text)))
    structure_bonus = 0.35 if re.search(r"(步骤|流程|注意事项|条件|规则|要求|说明|例如)", text) else 0.0
    density = length_score + min(0.8, sentence_count / 12) + structure_bonus + quality["score"] * 0.35

    if density >= 2.1:
        return 2
    return 1


def allocate_question_counts(
    chunks: Iterable[RagDatasetChunk],
    mode: str,
    requested_count: int | None,
) -> tuple[dict[int, int], int]:
    chunk_list = [chunk for chunk in chunks if (chunk.content or "").strip()]
    if not chunk_list:
        return {}, 0

    weights = {
        chunk.id: float(suggest_question_count_for_chunk(chunk.content))
        for chunk in chunk_list
    }
    allocations = {chunk.id: 0 for chunk in chunk_list}
    eligible_chunks = [chunk for chunk in chunk_list if weights[chunk.id] > 0]
    if not eligible_chunks:
        return allocations, 0

    suggested_total = max(1, int(round(sum(weights[chunk.id] for chunk in eligible_chunks))))
    target_total = suggested_total if mode == "auto" or not requested_count else max(1, requested_count)
    remaining = target_total
    ordered = sorted(eligible_chunks, key=lambda item: (weights[item.id], item.char_count), reverse=True)

    for chunk in ordered:
        if remaining <= 0:
            break
        allocations[chunk.id] += 1
        remaining -= 1

    if remaining > 0:
        total_weight = sum(weights.values()) or 1.0
        fractional: list[tuple[float, int]] = []
        for chunk in ordered:
            share = target_total * weights[chunk.id] / total_weight
            extra = max(0.0, share - allocations[chunk.id])
            fractional.append((extra, chunk.id))
        fractional.sort(reverse=True)
        idx = 0
        while remaining > 0 and fractional:
            chunk_id = fractional[idx % len(fractional)][1]
            allocations[chunk_id] += 1
            remaining -= 1
            idx += 1

    return allocations, suggested_total


def _extract_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    fenced = re.search(r"```json\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    try:
        loaded = json.loads(text)
        return loaded if isinstance(loaded, dict) else None
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            loaded = json.loads(text[start : end + 1])
            return loaded if isinstance(loaded, dict) else None
        except json.JSONDecodeError:
            pass
    return None


def _coerce_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_json_text(value: str | None, default: dict[str, Any] | None = None) -> dict[str, Any]:
    raw = (value or "").strip()
    if not raw:
        return deepcopy(default or {})
    loaded = json.loads(raw)
    if not isinstance(loaded, dict):
        raise ValueError("JSON 配置必须是对象")
    return loaded


def _extract_from_dict(payload: dict[str, Any], candidates: list[str]) -> Any:
    for path in candidates:
        current: Any = payload
        ok = True
        for part in path.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
                current = current[int(part)]
            else:
                ok = False
                break
        if ok and current not in (None, ""):
            return current
    return None


def _normalize_event_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        maybe = _extract_json_object(payload)
        if maybe:
            return maybe
        return {"text": payload}
    return {}


def _collect_stream_message(state: dict[str, Any], payload: dict[str, Any]) -> None:
    final_answer = _extract_from_dict(
        payload,
        [
            "answer",
            "data.answer",
            "choices.0.message.content",
        ],
    )
    if isinstance(final_answer, str) and final_answer.strip():
        state["final_answer"] = final_answer.strip()

    delta_answer = _extract_from_dict(
        payload,
        [
            "content",
            "text",
            "delta",
            "data.content",
            "data.text",
            "data.delta",
            "choices.0.delta.content",
        ],
    )
    if isinstance(delta_answer, str) and delta_answer.strip():
        state["answer_parts"].append(delta_answer.strip())

    contexts = _extract_from_dict(
        payload,
        ["retrieved_contexts", "data.retrieved_contexts", "sources", "data.sources"],
    )
    if isinstance(contexts, list):
        normalized_contexts: list[str] = []
        normalized_ids: list[str] = []
        for item in contexts:
            if isinstance(item, dict):
                content = str(item.get("content") or item.get("text") or "").strip()
                item_id = str(item.get("chunk_id") or item.get("id") or item.get("doc_id") or "").strip()
                if content:
                    normalized_contexts.append(content)
                if item_id:
                    normalized_ids.append(item_id)
            else:
                text = str(item).strip()
                if text:
                    normalized_contexts.append(text)
        if normalized_contexts:
            state["retrieved_contexts"] = normalized_contexts
        if normalized_ids:
            state["retrieved_context_ids"] = normalized_ids

    direct_contexts = _extract_from_dict(payload, ["retrieved_contexts", "data.retrieved_contexts"])
    if isinstance(direct_contexts, list):
        normalized_contexts = [str(item).strip() for item in direct_contexts if str(item).strip()]
        if normalized_contexts:
            state["retrieved_contexts"] = normalized_contexts

    direct_ids = _extract_from_dict(payload, ["retrieved_context_ids", "data.retrieved_context_ids"])
    if isinstance(direct_ids, list):
        normalized_ids = [str(item).strip() for item in direct_ids if str(item).strip()]
        if normalized_ids:
            state["retrieved_context_ids"] = normalized_ids


class OpenAICompatibleClient:
    def __init__(self, llm_config: LLMConfig):
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(
            base_url=llm_config.api_base_url,
            api_key=llm_config.api_key,
            timeout=180,
            max_retries=2,
        )
        self.model = llm_config.model_name
        self.temperature = llm_config.temperature if llm_config.temperature is not None else 0.2
        self.max_tokens = min(int(llm_config.max_tokens or 2048), 2000)

    async def chat(self, messages: list[dict[str, str]], as_json: bool = False) -> str:
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"} if as_json else None,
            )
        except Exception:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        return response.choices[0].message.content or ""


class TargetEndpointClient:
    def __init__(self, job: RagDatasetJob):
        self.url = (job.target_endpoint_url or "").strip()
        self.transport_mode = (job.target_transport_mode or "sse").strip() or "sse"
        self.authorization = (job.target_authorization or "").strip()
        self.extra_headers = job.target_extra_headers or ""
        self.body_template = job.target_request_body_template or DEFAULT_RAG_TARGET_REQUEST_BODY_TEMPLATE
        self.timeout = 180
        self.max_retries = RAG_TARGET_REQUEST_RETRIES

    def build_headers(self) -> dict[str, str]:
        headers = {
            "content-type": "application/json",
            "accept": "text/event-stream" if self.transport_mode == "sse" else "application/json",
            "connection": "close",
        }
        if self.authorization:
            headers["authorization"] = self.authorization
        extra = _parse_json_text(self.extra_headers, default={})
        for key, value in extra.items():
            if value is None:
                continue
            headers[str(key)] = str(value)
        return headers

    def build_body(self, question: str) -> dict[str, Any]:
        template = _parse_json_text(self.body_template, default=json.loads(DEFAULT_RAG_TARGET_REQUEST_BODY_TEMPLATE))
        rendered = render_template_value(template, {"question": question})
        if not isinstance(rendered, dict):
            raise ValueError("目标接口请求体模板渲染后必须是 JSON 对象")
        return rendered

    async def request(self, question: str) -> dict[str, Any]:
        if not self.url:
            raise ValueError("目标 chat 接口 URL 为空")
        headers = self.build_headers()
        body = self.build_body(question)
        last_exc: Exception | None = None
        limits = httpx.Limits(max_keepalive_connections=0, max_connections=10)
        for attempt in range(1, self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout, limits=limits, http2=False) as client:
                    if self.transport_mode == "sse":
                        return await self._request_sse(client, headers, body)
                    return await self._request_json(client, headers, body)
            except Exception as exc:
                last_exc = exc
                if not self._is_retryable_exception(exc) or attempt >= self.max_retries:
                    raise
                backoff_seconds = min(0.8 * attempt, 2.5)
                logger.warning(
                    "RAG target request retrying (%s/%s): %s",
                    attempt,
                    self.max_retries,
                    self._describe_retryable_error(exc),
                )
                await asyncio.sleep(backoff_seconds)
        assert last_exc is not None
        raise last_exc

    def _is_retryable_exception(self, exc: Exception) -> bool:
        if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
            return True
        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response.status_code in {408, 409, 425, 429, 500, 502, 503, 504}
        message = str(exc).lower()
        retryable_patterns = (
            "peer closed connection without sending complete message body",
            "incomplete chunked read",
            "server disconnected without sending a response",
            "connection reset by peer",
            "broken pipe",
        )
        return any(pattern in message for pattern in retryable_patterns)

    def _describe_retryable_error(self, exc: Exception) -> str:
        if isinstance(exc, httpx.HTTPStatusError):
            return f"http {exc.response.status_code}"
        return f"{type(exc).__name__}: {exc}"

    async def _request_json(
        self, client: httpx.AsyncClient, headers: dict[str, str], body: dict[str, Any]
    ) -> dict[str, Any]:
        response = await client.post(self.url, headers=headers, json=body)
        response.raise_for_status()
        content_type = (response.headers.get("content-type") or "").lower()
        if "application/json" in content_type:
            payload = response.json()
            if isinstance(payload, dict):
                return payload
        return {"answer": response.text.strip()}

    async def _request_sse(
        self, client: httpx.AsyncClient, headers: dict[str, str], body: dict[str, Any]
    ) -> dict[str, Any]:
        state: dict[str, Any] = {
            "answer_parts": [],
            "final_answer": None,
            "retrieved_contexts": None,
            "retrieved_context_ids": None,
        }
        async with client.stream("POST", self.url, headers=headers, json=body) as response:
            response.raise_for_status()
            data_lines: list[str] = []
            async for raw_line in response.aiter_lines():
                line = raw_line.strip()
                if not line:
                    if data_lines:
                        payload_text = "\n".join(data_lines).strip()
                        data_lines = []
                        if payload_text == "[DONE]":
                            continue
                        payload = _normalize_event_payload(payload_text)
                        _collect_stream_message(state, payload)
                    continue
                if line.startswith("data:"):
                    data_lines.append(line[5:].strip())

            if data_lines:
                payload_text = "\n".join(data_lines).strip()
                if payload_text and payload_text != "[DONE]":
                    payload = _normalize_event_payload(payload_text)
                    _collect_stream_message(state, payload)

        answer = str(state.get("final_answer") or "").strip() or "".join(state["answer_parts"]).strip()
        result: dict[str, Any] = {"answer": answer}
        if state.get("retrieved_contexts") is not None:
            result["retrieved_contexts"] = state["retrieved_contexts"]
        if state.get("retrieved_context_ids") is not None:
            result["retrieved_context_ids"] = state["retrieved_context_ids"]
        return result


async def generate_chunk_samples(
    question_client: OpenAICompatibleClient,
    chunk: RagDatasetChunk,
    filename: str,
    question_count: int,
) -> list[dict[str, str]]:
    request_count = question_count + min(3, max(1, question_count))
    prompt = QUESTION_GENERATION_PROMPT.format(
        filename=filename,
        chunk_key=chunk.chunk_key,
        content=chunk.content[:12000],
        n=request_count,
    )
    content = await question_client.chat([{"role": "user", "content": prompt}], as_json=True)
    payload = _extract_json_object(content)
    items = payload.get("items") if payload else None
    if not isinstance(items, list):
        raise ValueError("问题生成模型未返回 items 数组")

    samples: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question") or "").strip()
        reference = str(item.get("reference") or "").strip()
        if question and reference:
            samples.append({"question": question, "reference": reference})
    if not samples:
        raise ValueError("问题生成结果为空")
    validated = await validate_generated_samples(question_client, chunk.content, samples)
    if not validated:
        raise ValueError("候选问题经校验后全部无效")
    return validated[:question_count]


async def validate_generated_samples(
    question_client: OpenAICompatibleClient,
    chunk_content: str,
    samples: list[dict[str, str]],
) -> list[dict[str, str]]:
    deduped: list[dict[str, str]] = []
    seen_questions: set[str] = set()
    for item in samples:
        normalized_question = re.sub(r"\s+", "", (item.get("question") or "")).lower()
        if not normalized_question or normalized_question in seen_questions:
            continue
        if "文档" in normalized_question and ("提到了什么" in normalized_question or "说了什么" in normalized_question):
            continue
        seen_questions.add(normalized_question)
        deduped.append(item)

    if not deduped:
        return []

    prompt = QUESTION_VALIDATION_PROMPT.format(
        content=chunk_content[:12000],
        items_json=json.dumps(deduped, ensure_ascii=False, separators=(",", ":")),
    )
    content = await question_client.chat([{"role": "user", "content": prompt}], as_json=True)
    payload = _extract_json_object(content)
    verdict_items = payload.get("items") if payload else None
    if not isinstance(verdict_items, list):
        return deduped

    verdict_by_question: dict[str, tuple[bool, float]] = {}
    for item in verdict_items:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question") or "").strip()
        keep = bool(item.get("keep"))
        score = _coerce_float(item.get("score")) or 0.0
        if question:
            verdict_by_question[question] = (keep, score)

    kept = [
        (item, verdict_by_question.get(item["question"], (True, 0.5))[1])
        for item in deduped
        if verdict_by_question.get(item["question"], (True, 0.5))[0]
    ]
    kept.sort(key=lambda pair: pair[1], reverse=True)
    return [item for item, _score in kept]


async def fetch_target_answer(
    target_client: TargetEndpointClient,
    question: str,
    response_mode: str,
    target_system_prompt: str | None = None,
) -> dict[str, Any]:
    del target_system_prompt
    payload = await target_client.request(question)

    if response_mode == "answer_only":
        answer = ""
        if isinstance(payload, dict):
            extracted = _extract_from_dict(payload, ["answer", "data.answer", "content", "data.content", "text", "data.text"])
            answer = str(extracted or "").strip()
        if not answer:
            raise ValueError("目标 chat 接口未返回 answer")
        return {"answer": answer, "retrieved_contexts": None, "retrieved_context_ids": None}

    if not isinstance(payload, dict):
        raise ValueError("目标 chat 接口未返回 JSON 对象")

    answer = str(
        _extract_from_dict(payload, ["answer", "data.answer", "content", "data.content", "text", "data.text"])
        or ""
    ).strip()
    contexts = _extract_from_dict(payload, ["retrieved_contexts", "data.retrieved_contexts", "sources", "data.sources"])
    context_ids = _extract_from_dict(payload, ["retrieved_context_ids", "data.retrieved_context_ids"])
    if not answer:
        raise ValueError("目标 chat 接口返回的 answer 为空")
    if isinstance(contexts, list) and contexts and isinstance(contexts[0], dict) and not isinstance(context_ids, list):
        derived_contexts: list[str] = []
        derived_ids: list[str] = []
        for item in contexts:
            if not isinstance(item, dict):
                continue
            content = str(item.get("content") or item.get("text") or "").strip()
            item_id = str(item.get("chunk_id") or item.get("id") or item.get("doc_id") or "").strip()
            if content:
                derived_contexts.append(content)
            if item_id:
                derived_ids.append(item_id)
        contexts = derived_contexts
        context_ids = derived_ids
    if not isinstance(contexts, list) or not isinstance(context_ids, list):
        raise ValueError("目标 chat 接口未返回 retrieved_contexts / retrieved_context_ids 数组")
    normalized_contexts = [str(item).strip() for item in contexts if str(item).strip()]
    normalized_ids = [str(item).strip() for item in context_ids if str(item).strip()]
    if not normalized_contexts or not normalized_ids:
        raise ValueError("retrieved_contexts / retrieved_context_ids 不能为空")
    return {
        "answer": answer,
        "retrieved_contexts": normalized_contexts,
        "retrieved_context_ids": normalized_ids,
    }


def build_rag_dataset_field_schema(response_mode: str) -> list[dict[str, Any]]:
    schema = [
        {"name": "user_input", "type": "text", "required": True, "description": "自动生成的问题"},
        {"name": "response", "type": "text", "required": True, "description": "目标 RAG chat 接口的真实回答"},
        {"name": "reference", "type": "text", "required": True, "description": "基于知识库分片生成的标准答案"},
        {
            "name": "reference_context_ids",
            "type": "text_list",
            "required": True,
            "description": "标准证据分片 ID，用于 HitRate@K / MRR 等检索指标",
        },
        {
            "name": "source_chunk_ids",
            "type": "text_list",
            "required": True,
            "description": "该题来源的知识库分片 ID，便于回溯和 chunk 预览",
        },
        {
            "name": "source_document_names",
            "type": "text_list",
            "required": False,
            "description": "该题对应的源文档名称",
        },
        {
            "name": "generation_meta",
            "type": "json",
            "required": False,
            "description": "生成任务、文档、分片等元信息",
        },
    ]
    if response_mode == "answer_with_contexts":
        schema.extend(
            [
                {
                    "name": "retrieved_contexts",
                    "type": "text_list",
                    "required": False,
                    "description": "目标 RAG chat 接口真实返回的检索上下文",
                },
                {
                    "name": "retrieved_context_ids",
                    "type": "text_list",
                    "required": False,
                    "description": "目标 RAG chat 接口真实返回的检索分片 ID",
                },
            ]
        )
    return schema


def summarize_supported_metrics(response_mode: str) -> tuple[list[str], list[str], list[str]]:
    full = [
        "faithfulness",
        "context_recall",
        "context_precision",
        "contextual_relevancy",
        "answer_relevancy",
        "factual_correctness",
        "answer_completeness",
        "retrieval_hit_rate",
        "retrieval_mrr",
    ]
    answer_only = ["answer_relevancy", "factual_correctness", "answer_completeness"]
    if response_mode == "answer_with_contexts":
        return full, [], []
    unsupported = [
        "faithfulness",
        "context_recall",
        "context_precision",
        "contextual_relevancy",
        "retrieval_hit_rate",
        "retrieval_mrr",
    ]
    notes = [
        "当前目标 chat 接口只返回 answer，未返回真实 retrieved_contexts / retrieved_context_ids。",
        "因此本次自动生成的数据集只适合回答质量评测，不应拿来评真实检索质量。",
    ]
    return answer_only, unsupported, notes


def build_dataset_row_payload(sample: RagDatasetSample) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "user_input": sample.question,
        "response": sample.response,
        "reference": sample.reference,
        "reference_context_ids": sample.reference_context_ids or [],
        "source_chunk_ids": sample.source_chunk_ids or [],
        "source_document_names": [sample.document.filename] if sample.document else [],
        "generation_meta": {
            "sample_id": sample.id,
            "job_id": sample.job_id,
            "chunk_id": sample.chunk_id,
        },
    }
    if sample.retrieved_contexts is not None:
        payload["retrieved_contexts"] = sample.retrieved_contexts
    if sample.retrieved_context_ids is not None:
        payload["retrieved_context_ids"] = sample.retrieved_context_ids
    return payload


def _load_job(db: Session, job_id: int) -> RagDatasetJob | None:
    return (
        db.query(RagDatasetJob)
        .options(
            joinedload(RagDatasetJob.documents).joinedload(RagDatasetDocument.chunks),
            joinedload(RagDatasetJob.samples).joinedload(RagDatasetSample.document),
            joinedload(RagDatasetJob.question_llm_config),
            joinedload(RagDatasetJob.target_llm_config),
            joinedload(RagDatasetJob.dataset),
        )
        .filter(RagDatasetJob.id == job_id)
        .first()
    )


def _log(job: RagDatasetJob, message: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    job.logs = (job.logs or "") + f"[{ts}] {message}\n"
    logger.info("[RAG Builder %s] %s", job.id, message)


def _recount_job(job: RagDatasetJob) -> None:
    selected_samples = [sample for sample in job.samples if sample.selected]
    job.total_samples = len(selected_samples)
    job.completed_samples = sum(1 for sample in selected_samples if sample.status == "completed")
    job.failed_samples = sum(1 for sample in selected_samples if sample.status == "failed")
    job.total_documents = len(job.documents)
    job.total_chunks = sum(len(document.chunks) for document in job.documents)


def _coerce_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _remaining_timeout(deadline: datetime, cap_seconds: float) -> float:
    remaining = (deadline - datetime.now(timezone.utc)).total_seconds()
    if remaining <= 0:
        raise TimeoutError(f"任务执行超时（超过 {RAG_JOB_TIMEOUT_SECONDS // 60} 分钟）")
    return max(0.001, min(cap_seconds, remaining))


def sync_generated_dataset(db: Session, job: RagDatasetJob) -> None:
    supported_metrics, unsupported_metrics, notes = summarize_supported_metrics(job.target_response_mode)
    completed_samples = [
        sample for sample in job.samples if sample.selected and sample.status == "completed"
    ]
    completed_samples.sort(key=lambda item: item.id)

    dataset_name = job.name.strip() or f"RAG 自动数据集 #{job.id}"
    description_lines = [
        "自动生成的 RAG 评测数据集。",
        f"支持指标: {', '.join(supported_metrics)}" if supported_metrics else "支持指标: 无",
    ]
    if unsupported_metrics:
        description_lines.append(f"当前不支持的指标: {', '.join(unsupported_metrics)}")
    description_lines.extend(notes)
    description = "\n".join(description_lines)

    if job.dataset is None:
        dataset = Dataset(
            name=dataset_name,
            description=description,
            sample_type="single_turn",
            field_schema=build_rag_dataset_field_schema(job.target_response_mode),
            row_count=0,
        )
        db.add(dataset)
        db.flush()
        job.dataset = dataset
        job.dataset_id = dataset.id
    else:
        dataset = job.dataset
        dataset.name = dataset_name
        dataset.description = description
        dataset.sample_type = "single_turn"
        dataset.field_schema = build_rag_dataset_field_schema(job.target_response_mode)

    next_row_index = (
        db.query(DatasetRow.row_index)
        .filter(DatasetRow.dataset_id == dataset.id)
        .order_by(DatasetRow.row_index.desc())
        .first()
    )
    next_row_index = (next_row_index[0] + 1) if next_row_index else 0

    for sample in completed_samples:
        payload = build_dataset_row_payload(sample)
        if sample.dataset_row_id:
            row = db.query(DatasetRow).filter(DatasetRow.id == sample.dataset_row_id).first()
            if row is not None:
                row.data = payload
                continue
        row = DatasetRow(
            dataset_id=dataset.id,
            row_index=next_row_index,
            data=payload,
        )
        db.add(row)
        db.flush()
        sample.dataset_row_id = row.id
        next_row_index += 1

    dataset.row_count = (
        db.query(DatasetRow).filter(DatasetRow.dataset_id == dataset.id).count()
    )


def recover_stale_rag_dataset_job(
    db: Session,
    job_id: int,
    timeout_seconds: int = RAG_JOB_TIMEOUT_SECONDS,
) -> RagDatasetJob | None:
    job = _load_job(db, job_id)
    if job is None or job.status != "running":
        return job

    reference_time = _coerce_utc(job.started_at) or _coerce_utc(job.updated_at) or _coerce_utc(job.created_at)
    if reference_time is None:
        return job

    age_seconds = (datetime.now(timezone.utc) - reference_time).total_seconds()
    if age_seconds < timeout_seconds:
        return job

    timeout_message = f"任务执行超时（超过 {max(1, timeout_seconds // 60)} 分钟），已自动标记为失败"
    for document in job.documents or []:
        for chunk in document.chunks or []:
            if chunk.generation_status == "running":
                chunk.generation_status = "failed"
                chunk.generation_error = timeout_message
    for sample in job.samples or []:
        if sample.status == "running":
            sample.status = "failed"
            sample.error_message = timeout_message
            sample.retry_count = (sample.retry_count or 0) + 1

    if any(sample.selected and sample.status == "completed" for sample in job.samples) or job.dataset is not None:
        sync_generated_dataset(db, job)
    _recount_job(job)
    job.status = "failed" if job.completed_samples == 0 else "partial"
    job.error_message = timeout_message
    job.finished_at = datetime.now(timezone.utc)
    _log(job, f"✗ 自动回收陈旧运行任务：{timeout_message}")
    db.commit()
    db.refresh(job)
    return _load_job(db, job_id)


async def _process_rag_sample_request(
    session_factory,
    job_id: int,
    sample_id: int,
    sample_question: str,
    target_client: TargetEndpointClient,
    response_mode: str,
    target_system_prompt: str | None,
    deadline: datetime,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    async with semaphore:
        try:
            result = await asyncio.wait_for(
                fetch_target_answer(
                    target_client,
                    sample_question,
                    response_mode,
                    target_system_prompt,
                ),
                timeout=_remaining_timeout(deadline, RAG_TARGET_REQUEST_TIMEOUT_SECONDS),
            )
        except Exception as exc:
            db: Session = session_factory()
            try:
                sample = db.query(RagDatasetSample).filter(RagDatasetSample.id == sample_id).first()
                if sample is not None:
                    sample.status = "failed"
                    sample.error_message = str(exc)[:1000]
                    sample.retry_count = (sample.retry_count or 0) + 1
                    db.commit()
            finally:
                db.close()
            return {
                "sample_id": sample_id,
                "status": "failed",
                "error_message": str(exc)[:1000],
            }

        db = session_factory()
        try:
            sample = db.query(RagDatasetSample).filter(RagDatasetSample.id == sample_id).first()
            if sample is not None:
                sample.response = result["answer"]
                sample.retrieved_contexts = result["retrieved_contexts"]
                sample.retrieved_context_ids = result["retrieved_context_ids"]
                sample.status = "completed"
                sample.error_message = None
                db.commit()
        finally:
            db.close()
        return {
            "sample_id": sample_id,
            "status": "completed",
        }


async def run_rag_dataset_job(
    job_id: int,
    session_factory,
    scope: str = "full",
    sample_ids: list[int] | None = None,
) -> None:
    db: Session = session_factory()
    try:
        deadline = datetime.now(timezone.utc) + timedelta(seconds=RAG_JOB_TIMEOUT_SECONDS)
        job = _load_job(db, job_id)
        if job is None:
            logger.error("RAG dataset job %s not found", job_id)
            return

        if job.status == "running":
            _log(job, "已有相同任务在运行，忽略本次触发。")
            db.commit()
            return

        if not job.documents:
            job.status = "failed"
            job.error_message = "请先上传知识库文档"
            _log(job, "✗ 没有可用文档")
            db.commit()
            return

        question_client = OpenAICompatibleClient(job.question_llm_config)
        target_client = TargetEndpointClient(job)

        job.status = "running"
        job.error_message = None
        job.started_at = datetime.now(timezone.utc)
        job.finished_at = None
        _log(job, f"========== 开始 RAG 数据集生成（scope={scope}） ==========")

        all_chunks = [chunk for document in job.documents for chunk in document.chunks]
        allocations, suggested_total = allocate_question_counts(
            all_chunks, job.question_count_mode, job.requested_question_count
        )
        job.suggested_question_count = suggested_total
        for chunk in all_chunks:
            chunk.suggested_question_count = suggest_question_count_for_chunk(chunk.content)
            chunk.allocated_question_count = allocations.get(chunk.id, 0)
        db.commit()

        targeted_samples = set(sample_ids or [])
        for document in job.documents:
            for chunk in document.chunks:
                allocated_count = chunk.allocated_question_count or 0
                existing_samples = [
                    sample for sample in job.samples if sample.chunk_id == chunk.id and sample.selected
                ]
                should_generate = False
                if scope == "full":
                    should_generate = allocated_count > 0 and not existing_samples
                elif scope == "retry_failed":
                    should_generate = allocated_count > 0 and (
                        chunk.generation_status == "failed" or not existing_samples
                    )

                if not should_generate:
                    continue

                chunk.generation_status = "running"
                chunk.generation_error = None
                db.commit()
                _log(job, f"生成问题：{document.filename} / {chunk.chunk_key}（{allocated_count} 题）")
                try:
                    generated = await asyncio.wait_for(
                        generate_chunk_samples(
                            question_client,
                            chunk,
                            document.filename,
                            allocated_count,
                        ),
                        timeout=_remaining_timeout(deadline, RAG_QUESTION_GENERATION_TIMEOUT_SECONDS),
                    )
                    if not generated:
                        raise ValueError("未生成任何问题")
                    for item in generated:
                        sample = RagDatasetSample(
                            job_id=job.id,
                            document_id=document.id,
                            chunk_id=chunk.id,
                            question=item["question"],
                            reference=item["reference"],
                            reference_context_ids=[chunk.chunk_key],
                            source_chunk_ids=[chunk.chunk_key],
                            status="pending",
                        )
                        db.add(sample)
                    chunk.generation_status = "completed"
                    _log(job, f"✓ 已生成 {len(generated)} 条候选问题")
                except TimeoutError as exc:
                    chunk.generation_status = "failed"
                    chunk.generation_error = str(exc)[:1000]
                    _log(job, f"✗ 问题生成超时：{exc}")
                    db.commit()
                    raise
                except Exception as exc:
                    chunk.generation_status = "failed"
                    chunk.generation_error = str(exc)[:1000]
                    _log(job, f"✗ 问题生成失败：{exc}")
                db.commit()

        db.expire_all()
        job = _load_job(db, job_id)
        if job is None:
            return

        samples_to_process: list[RagDatasetSample] = []
        if scope == "rerun_samples" and targeted_samples:
            samples_to_process = [sample for sample in job.samples if sample.id in targeted_samples]
        elif scope == "retry_failed":
            samples_to_process = [sample for sample in job.samples if sample.selected and sample.status == "failed"]
        else:
            samples_to_process = [
                sample
                for sample in job.samples
                if sample.selected and sample.status in {"pending", "failed"}
            ]

        if samples_to_process:
            _log(
                job,
                f"开始并发调用目标 chat：{len(samples_to_process)} 条样本，最大并发 {RAG_TARGET_REQUEST_CONCURRENCY}",
            )
        for sample in samples_to_process:
            _remaining_timeout(deadline, RAG_TARGET_REQUEST_TIMEOUT_SECONDS)
            _log(job, f"调用目标 chat：样本 #{sample.id} / {sample.question[:40]}")
            sample.status = "running"
            sample.error_message = None
        db.commit()

        semaphore = asyncio.Semaphore(RAG_TARGET_REQUEST_CONCURRENCY)
        sample_results = await asyncio.gather(
            *[
                _process_rag_sample_request(
                    session_factory=session_factory,
                    job_id=job.id,
                    sample_id=sample.id,
                    sample_question=sample.question,
                    target_client=target_client,
                    response_mode=job.target_response_mode,
                    target_system_prompt=job.target_system_prompt,
                    deadline=deadline,
                    semaphore=semaphore,
                )
                for sample in samples_to_process
            ]
        )

        db.expire_all()
        job = _load_job(db, job_id)
        if job is None:
            return
        for result in sample_results:
            if result["status"] == "completed":
                _log(job, f"✓ 样本 #{result['sample_id']} 已完成")
            else:
                _log(job, f"✗ 样本 #{result['sample_id']} 失败：{result['error_message']}")
        db.commit()

        db.expire_all()
        job = _load_job(db, job_id)
        if job is None:
            return

        sync_generated_dataset(db, job)
        _recount_job(job)

        if job.total_samples == 0:
            job.status = "failed"
            job.error_message = "没有成功生成任何样本"
            _log(job, "✗ 未生成可用样本")
        elif job.completed_samples > 0 and job.failed_samples == 0:
            job.status = "completed"
            _log(job, f"✓ 全部完成，共 {job.completed_samples} 条样本")
        elif job.completed_samples > 0:
            job.status = "partial"
            _log(job, f"⚠ 部分完成：成功 {job.completed_samples}，失败 {job.failed_samples}")
        else:
            job.status = "failed"
            job.error_message = "所有样本都生成失败"
            _log(job, "✗ 所有样本均失败")
        job.finished_at = datetime.now(timezone.utc)
        _log(job, "========== RAG 数据集生成结束 ==========")
        db.commit()
    except Exception as exc:
        logger.exception("RAG dataset job %s failed", job_id)
        db.rollback()
        job = _load_job(db, job_id)
        if job is not None:
            error_message = str(exc).strip()
            if isinstance(exc, TimeoutError) and not error_message:
                error_message = f"任务执行超时（超过 {RAG_JOB_TIMEOUT_SECONDS // 60} 分钟）"
            job.status = "failed"
            job.error_message = error_message[:2000]
            job.finished_at = datetime.now(timezone.utc)
            _log(job, f"✗✗✗ 任务异常终止：{error_message}")
            db.commit()
    finally:
        db.close()


def persist_uploaded_document(
    db: Session,
    job: RagDatasetJob,
    filename: str,
    raw_bytes: bytes,
    raw_text: str,
) -> RagDatasetDocument:
    upload_root = Path(settings.UPLOAD_DIR).resolve() / "rag_jobs" / str(job.id)
    upload_root.mkdir(parents=True, exist_ok=True)
    stored_name = f"{uuid.uuid4().hex[:8]}_{Path(filename).name}"
    stored_path = upload_root / stored_name
    stored_path.write_bytes(raw_bytes)

    content_hash = hashlib.sha256(raw_bytes).hexdigest()
    document = RagDatasetDocument(
        job_id=job.id,
        filename=filename,
        file_path=str(stored_path),
        content_hash=content_hash,
        raw_content=raw_text,
        char_count=len(raw_text),
    )
    db.add(document)
    db.flush()

    chunks = split_into_chunks(raw_text)
    for index, content in enumerate(chunks):
        chunk = RagDatasetChunk(
            document_id=document.id,
            chunk_index=index,
            chunk_key=f"doc-{document.id}-chunk-{index}",
            content=content,
            char_count=len(content),
            suggested_question_count=suggest_question_count_for_chunk(content),
            allocated_question_count=0,
            generation_status="pending",
        )
        db.add(chunk)

    document.chunk_count = len(chunks)
    return document
