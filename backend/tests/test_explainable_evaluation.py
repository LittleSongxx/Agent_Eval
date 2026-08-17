"""
Unit tests for Explainable Evaluation Results Module.
"""

import pytest

from app.core.explainable_evaluation import (
    ExplainableEvaluationAnalyzer,
    FailureCategory,
    analyze_evaluation_results,
)


def test_classify_hallucination():
    """Test classification of hallucination failures."""
    analyzer = ExplainableEvaluationAnalyzer()

    analysis = analyzer.classify_failure(
        sample_id="test_1",
        metric_name="faithfulness",
        score=0.3,
        reasoning="回答编造了上下文中不存在的事实，无法找到支撑证据",
    )

    assert analysis.category == FailureCategory.HALLUCINATION
    assert analysis.confidence > 0.5
    assert len(analysis.matched_keywords) > 0
    assert "编造" in analysis.matched_keywords or "不存在" in analysis.matched_keywords
    assert len(analysis.suggestions) > 0
    assert analysis.priority == "high"


def test_classify_irrelevant():
    """Test classification of irrelevant answer failures."""
    analyzer = ExplainableEvaluationAnalyzer()

    analysis = analyzer.classify_failure(
        sample_id="test_2",
        metric_name="relevance",
        score=0.4,
        reasoning="回答偏离了问题核心，答非所问，没有解决用户的实际需求",
    )

    assert analysis.category == FailureCategory.IRRELEVANT
    assert analysis.confidence > 0.5
    assert len(analysis.matched_keywords) > 0
    assert len(analysis.suggestions) > 0
    assert analysis.priority == "medium"


def test_classify_incomplete():
    """Test classification of incomplete answer failures."""
    analyzer = ExplainableEvaluationAnalyzer()

    analysis = analyzer.classify_failure(
        sample_id="test_3",
        metric_name="completeness",
        score=0.5,
        reasoning="回答不完整，只回答了部分要点，遗漏了关键信息",
    )

    assert analysis.category == FailureCategory.INCOMPLETE
    assert analysis.confidence > 0.5
    assert len(analysis.suggestions) > 0


def test_classify_context_misuse():
    """Test classification of context misuse failures."""
    analyzer = ExplainableEvaluationAnalyzer()

    analysis = analyzer.classify_failure(
        sample_id="test_4",
        metric_name="context_utilization",
        score=0.3,
        reasoning="上下文中有相关信息但未使用，忽略了重要证据",
    )

    assert analysis.category == FailureCategory.CONTEXT_MISUSE
    assert analysis.confidence > 0.5
    assert analysis.priority == "high"


def test_classify_unknown():
    """Test classification when no keywords match."""
    analyzer = ExplainableEvaluationAnalyzer()

    analysis = analyzer.classify_failure(
        sample_id="test_5",
        metric_name="custom_metric",
        score=0.4,
        reasoning="这是一个完全没有关键词的描述",
    )

    assert analysis.category == FailureCategory.UNKNOWN
    assert analysis.confidence == 0.0
    assert len(analysis.matched_keywords) == 0


def test_cluster_failures():
    """Test failure clustering by category."""
    analyzer = ExplainableEvaluationAnalyzer()

    # Add multiple failures of different categories
    analyzer.classify_failure("1", "faithfulness", 0.3, "回答编造了事实")
    analyzer.classify_failure("2", "faithfulness", 0.4, "上下文中不存在该信息")
    analyzer.classify_failure("3", "relevance", 0.5, "回答偏离了问题")
    analyzer.classify_failure("4", "completeness", 0.6, "回答不完整")

    clusters = analyzer.cluster_failures()

    assert len(clusters) > 0
    # Most samples should be in hallucination cluster (2 samples)
    assert clusters[0].sample_count >= 2
    assert clusters[0].category == FailureCategory.HALLUCINATION
    assert len(clusters[0].sample_ids) == clusters[0].sample_count
    assert len(clusters[0].suggestions) > 0


def test_generate_optimization_report():
    """Test comprehensive optimization report generation."""
    analyzer = ExplainableEvaluationAnalyzer()

    # Add various failures
    analyzer.classify_failure("1", "faithfulness", 0.3, "回答编造了事实")
    analyzer.classify_failure("2", "faithfulness", 0.4, "无法找到支撑证据")
    analyzer.classify_failure("3", "relevance", 0.5, "回答偏离问题")

    report = analyzer.generate_optimization_report(total_samples=10)

    assert report.total_samples == 10
    assert report.failed_samples == 3
    assert report.failure_rate == 0.3
    assert len(report.category_distribution) > 0
    assert len(report.clusters) > 0
    assert len(report.prioritized_suggestions) > 0

    # Check priority counts
    assert report.high_priority_count >= 0
    assert report.medium_priority_count >= 0
    assert report.low_priority_count >= 0


def test_prioritized_suggestions_order():
    """Test that suggestions are properly prioritized."""
    analyzer = ExplainableEvaluationAnalyzer()

    # Add high priority failure (hallucination)
    analyzer.classify_failure("1", "faithfulness", 0.2, "编造了不存在的事实")
    # Add medium priority failure (irrelevant)
    analyzer.classify_failure("2", "relevance", 0.4, "回答偏离了问题")
    # Add low priority failure (unknown)
    analyzer.classify_failure("3", "custom", 0.5, "无法分类的问题")

    report = analyzer.generate_optimization_report(total_samples=3)

    # First suggestion should be high priority
    assert report.prioritized_suggestions[0]["priority"] == "high"
    # Last suggestion should be lower priority
    assert report.prioritized_suggestions[-1]["priority"] in ["medium", "low"]


def test_get_top_suggestions():
    """Test getting top N suggestions."""
    analyzer = ExplainableEvaluationAnalyzer()

    # Add multiple failures
    analyzer.classify_failure("1", "faithfulness", 0.3, "编造事实")
    analyzer.classify_failure("2", "faithfulness", 0.4, "不存在的信息")
    analyzer.classify_failure("3", "relevance", 0.5, "偏离问题")

    top_suggestions = analyzer.get_top_suggestions(top_n=2)

    assert len(top_suggestions) <= 2
    assert all(isinstance(s, str) for s in top_suggestions)
    # Should contain category and impact percentage
    assert any("%" in s for s in top_suggestions)


def test_to_dict_serialization():
    """Test dictionary serialization for JSON export."""
    analyzer = ExplainableEvaluationAnalyzer()

    analyzer.classify_failure("1", "faithfulness", 0.3, "编造事实")
    analyzer.classify_failure("2", "relevance", 0.5, "偏离问题")

    result = analyzer.to_dict()

    assert "failure_analyses" in result
    assert len(result["failure_analyses"]) == 2
    assert all(isinstance(a, dict) for a in result["failure_analyses"])
    assert all("category" in a for a in result["failure_analyses"])
    assert all("suggestions" in a for a in result["failure_analyses"])


def test_analyze_evaluation_results_convenience():
    """Test the convenience function for analyzing results."""
    eval_details = [
        {
            "sample_id": "1",
            "metric_name": "faithfulness",
            "score": 0.3,
            "reasoning": "回答编造了不存在的事实",
        },
        {
            "sample_id": "2",
            "metric_name": "relevance",
            "score": 0.5,
            "reasoning": "回答偏离了问题核心",
        },
        {
            "sample_id": "3",
            "metric_name": "completeness",
            "score": 0.8,
            "reasoning": "回答完整准确",  # This should pass
        },
    ]

    report = analyze_evaluation_results(eval_details, pass_threshold=0.7)

    assert report.total_samples == 3
    assert report.failed_samples == 2  # Only 2 below threshold
    assert report.failure_rate == pytest.approx(0.667, abs=0.01)
    assert len(report.category_distribution) > 0


def test_empty_analyzer():
    """Test analyzer behavior with no failures."""
    analyzer = ExplainableEvaluationAnalyzer()

    clusters = analyzer.cluster_failures()
    assert len(clusters) == 0

    top_suggestions = analyzer.get_top_suggestions()
    assert len(top_suggestions) == 0

    result = analyzer.to_dict()
    assert len(result["failure_analyses"]) == 0


def test_multiple_keywords_increase_confidence():
    """Test that more matched keywords increase confidence."""
    analyzer = ExplainableEvaluationAnalyzer()

    # Single keyword
    analysis1 = analyzer.classify_failure(
        "1", "faithfulness", 0.3, "回答编造了事实"
    )

    # Multiple keywords
    analysis2 = analyzer.classify_failure(
        "2",
        "faithfulness",
        0.3,
        "回答编造了上下文中不存在的虚构信息，无法找到支撑证据",
    )

    # More keywords should lead to higher confidence
    assert analysis2.confidence >= analysis1.confidence


def test_category_distribution_counts():
    """Test that category distribution accurately counts failures."""
    analyzer = ExplainableEvaluationAnalyzer()

    # Add 3 hallucination failures
    for i in range(3):
        analyzer.classify_failure(f"h_{i}", "faithfulness", 0.3, "编造事实")

    # Add 2 irrelevant failures
    for i in range(2):
        analyzer.classify_failure(f"i_{i}", "relevance", 0.5, "偏离问题")

    report = analyzer.generate_optimization_report(total_samples=5)

    assert report.category_distribution[FailureCategory.HALLUCINATION.value] == 3
    assert report.category_distribution[FailureCategory.IRRELEVANT.value] == 2


def test_cluster_common_patterns():
    """Test that common patterns are correctly identified."""
    analyzer = ExplainableEvaluationAnalyzer()

    # Add failures with common keywords
    analyzer.classify_failure("1", "faithfulness", 0.3, "回答编造了事实，上下文中不存在")
    analyzer.classify_failure("2", "faithfulness", 0.4, "编造了虚构的信息")
    analyzer.classify_failure("3", "faithfulness", 0.2, "上下文中不存在该说法")

    clusters = analyzer.cluster_failures()

    hallucination_cluster = next(
        c for c in clusters if c.category == FailureCategory.HALLUCINATION
    )

    # Should identify "编造" and "不存在" as common patterns
    assert len(hallucination_cluster.common_patterns) > 0
    assert any("编造" in p for p in hallucination_cluster.common_patterns)


def test_impact_score_calculation():
    """Test that impact scores are correctly calculated."""
    analyzer = ExplainableEvaluationAnalyzer()

    # Add 7 hallucination failures and 3 irrelevant failures
    for i in range(7):
        analyzer.classify_failure(f"h_{i}", "faithfulness", 0.3, "编造事实")
    for i in range(3):
        analyzer.classify_failure(f"i_{i}", "relevance", 0.5, "偏离问题")

    report = analyzer.generate_optimization_report(total_samples=10)

    # Hallucination should have higher impact (70%)
    hallucination_suggestion = next(
        s
        for s in report.prioritized_suggestions
        if s["category"] == FailureCategory.HALLUCINATION.value
    )
    assert hallucination_suggestion["impact_score"] == pytest.approx(0.7, abs=0.01)

    # Irrelevant should have lower impact (30%)
    irrelevant_suggestion = next(
        s
        for s in report.prioritized_suggestions
        if s["category"] == FailureCategory.IRRELEVANT.value
    )
    assert irrelevant_suggestion["impact_score"] == pytest.approx(0.3, abs=0.01)
