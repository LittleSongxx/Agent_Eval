"""BM25 mock 检索引擎的单元测试（零依赖实现，字符 bigram 分词）。"""

from __future__ import annotations

import pytest

from scripts.bm25_mock_retriever import BM25Index, bigram_tokens, load_documents


class TestBigramTokens:
    def test_cjk_bigram(self):
        tokens = bigram_tokens("云购商城支持退货")
        assert "云购" in tokens
        assert "购商" in tokens
        assert "商城" in tokens

    def test_ascii_word_lowercased(self):
        tokens = bigram_tokens("BM25 Index v1.5")
        assert "bm25" in tokens
        assert "index" in tokens
        assert "v1" in tokens
        assert "5" in tokens

    def test_single_cjk_char_kept(self):
        assert bigram_tokens("年") == ["年"]

    def test_empty_and_symbols(self):
        assert bigram_tokens("") == []
        assert bigram_tokens("??? ") == []


class TestBM25Index:
    def test_relevant_doc_ranks_first(self):
        docs = [
            "手机采用六轴防抖的主摄模组，等效焦距 24mm。",
            "云购商城支持7天无理由退货、15天换货、30天维修的三级售后保障，自签收次日零时起算。",
        ]
        index = BM25Index(docs)
        ranked = index.search("7天无理由退货的起算时间", top_k=5)
        assert ranked, "查询与文档有词项重叠，不应返回空"
        assert ranked[0][0] == 1, "售后政策文档应排第一"

    def test_unrelated_query_returns_empty(self):
        index = BM25Index(["云购商城支持7天无理由退货。"])
        assert index.search("手机夜景模式曝光时长", top_k=5) == []

    def test_no_token_query_returns_empty(self):
        index = BM25Index(["云购商城支持退货。"])
        assert index.search("???", top_k=5) == []

    def test_top_k_respected(self):
        docs = [f"chunk 第{i}段 售后政策内容" for i in range(10)]
        index = BM25Index(docs)
        ranked = index.search("售后政策内容", top_k=3)
        assert len(ranked) == 3

    def test_search_with_ids_returns_ids(self):
        docs = ["云购商城支持7天无理由退货。", "手机防抖模组。"]
        ids = ["doc-a", "doc-b"]
        index = BM25Index(docs)
        results = index.search_with_ids("无理由退货", ids, top_k=1)
        assert results == [("doc-a", results[0][1])]

    def test_search_with_ids_rejects_length_mismatch(self):
        index = BM25Index(["内容"])
        with pytest.raises(ValueError):
            index.search_with_ids("查询", ["a", "b"], top_k=5)


class TestLoadDocuments:
    def test_load_file(self, tmp_path):
        path = tmp_path / "policy.md"
        path.write_text("云购商城售后政策内容", encoding="utf-8")
        docs, ids = load_documents(str(path))
        assert docs == ["云购商城售后政策内容"]
        assert ids == ["policy"]

    def test_load_directory_skips_non_md(self, tmp_path):
        (tmp_path / "a.md").write_text("第一章内容", encoding="utf-8")
        (tmp_path / "b.txt").write_text("第二章内容", encoding="utf-8")
        (tmp_path / "c.bin").write_text("不该读", encoding="utf-8")
        docs, ids = load_documents(str(tmp_path))
        assert set(ids) == {"a", "b"}

    def test_missing_source_raises(self):
        with pytest.raises(SystemExit):
            load_documents("/nonexistent/path.md")
