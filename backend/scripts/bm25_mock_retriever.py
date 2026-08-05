"""零依赖中文 BM25 mock 检索引擎（字符 bigram tokenize + BM25 打分）。

用途：
1. 在没有真实检索引擎（ES/Milvus/Chroma）的环境下演示端到端
   "文档 → 索引 → 检索 → 评测"链路；
2. 为检索侧评测生成"真实检索出来的" retrieved_contexts/ids，
   而不是把源 chunk 直接贴上去（解决合成数据集"完美检索"的天花板效应）。

不依赖 jieba / rank_bm25 等第三方库：中文按连续字符 bigram 分词，
英文按字母数字词，BM25 用经典 Robertson 公式。

用法：
    python -m scripts.bm25_mock_retriever --docs <文件或目录> --query "7天无理由退货的条件" --top-k 3
"""

from __future__ import annotations

import argparse
import logging
import math
import re
import typing as t
from collections import Counter
from pathlib import Path

logger = logging.getLogger("bm25_mock_retriever")

_CJK_RE = re.compile(r"[一-鿿]+")
_ASCII_WORD_RE = re.compile(r"[a-z0-9]+")


def bigram_tokens(text: str) -> list[str]:
    """中文连续字符 bigram + 英文词，作为 BM25 词项。"""
    text = str(text).lower()
    tokens: list[str] = []
    for word in _ASCII_WORD_RE.findall(text):
        tokens.append(word)
    for run in _CJK_RE.findall(text):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return tokens


class BM25Index:
    """经典 BM25（k1=1.5, b=0.75），文档粒度为 chunk。"""

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        if not documents:
            raise ValueError("BM25 索引需要至少一篇文档")
        self.documents = documents
        self.doc_terms = [bigram_tokens(doc) for doc in documents]
        self.doc_lens = [len(terms) for terms in self.doc_terms]
        self.avg_doc_len = sum(self.doc_lens) / len(self.doc_lens)
        self.doc_count = len(documents)
        self.k1 = k1
        self.b = b
        df: Counter[str] = Counter()
        for terms in self.doc_terms:
            df.update(set(terms))
        self.idf: dict[str, float] = {
            term: math.log(1.0 + (self.doc_count - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def score(self, query_terms: list[str], doc_idx: int) -> float:
        term_freq = Counter(self.doc_terms[doc_idx])
        doc_len = self.doc_lens[doc_idx]
        score = 0.0
        for term in set(query_terms):
            idf = self.idf.get(term)
            if idf is None:
                continue
            tf = term_freq.get(term, 0)
            if tf == 0:
                continue
            norm = 1.0 - self.b + self.b * doc_len / self.avg_doc_len
            score += idf * (tf * (self.k1 + 1.0)) / (tf + self.k1 * norm)
        return score

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        query_terms = bigram_tokens(query)
        if not query_terms:
            return []
        scored = [
            (idx, self.score(query_terms, idx)) for idx in range(self.doc_count)
        ]
        ranked = sorted(scored, key=lambda item: item[1], reverse=True)
        return [(idx, score) for idx, score in ranked[:top_k] if score > 0]

    def search_with_ids(self, query: str, chunk_ids: list[str], top_k: int = 5) -> list[tuple[str, float]]:
        if len(chunk_ids) != self.doc_count:
            raise ValueError("chunk_ids 数量必须与索引文档一致")
        return [(chunk_ids[idx], score) for idx, score in self.search(query, top_k)]


def load_documents(source: str) -> tuple[list[str], list[str]]:
    """从单个 .md/.txt 文件或目录加载文档，返回 (contents, chunk_ids)。"""
    source_path = Path(source)
    files: list[Path] = []
    if source_path.is_dir():
        files = sorted(source_path.glob("*.md")) + sorted(source_path.glob("*.txt"))
    elif source_path.is_file():
        files = [source_path]
    if not files:
        raise SystemExit(f"未找到文档: {source}")

    documents: list[str] = []
    chunk_ids: list[str] = []
    for file_path in files:
        content = file_path.read_text(encoding="utf-8", errors="ignore").strip()
        if not content:
            continue
        documents.append(content)
        chunk_ids.append(file_path.stem)
    return documents, chunk_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="零依赖中文 BM25 mock 检索引擎")
    parser.add_argument("--docs", required=True, help="文档文件或目录")
    parser.add_argument("--query", required=True, help="检索查询")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    documents, chunk_ids = load_documents(args.docs)
    index = BM25Index(documents)
    print(f"索引就绪: {len(documents)} 个 chunk, 平均长度 {index.avg_doc_len:.0f} 词项")
    print(f"查询: {args.query}")
    print("=" * 60)
    for chunk_id, score in index.search_with_ids(args.query, chunk_ids, args.top_k):
        idx = chunk_ids.index(chunk_id)
        print(f"  [{chunk_id}] BM25={score:.4f}")
        print(f"    {documents[idx][:120].replace(chr(10), ' ')}...")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
