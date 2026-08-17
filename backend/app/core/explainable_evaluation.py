"""
Explainable Evaluation Results Module.

This module provides automatic failure reason classification and actionable
improvement suggestions for evaluation results, forming a closed-loop optimization cycle.
"""

from __future__ import annotations

import json
import logging
import re
import typing as t
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class FailureCategory(str, Enum):
    """Categories of evaluation failures for classification."""

    HALLUCINATION = "hallucination"  # 回答编造了上下文中不存在的事实
    IRRELEVANT = "irrelevant"  # 回答偏离问题核心
    INCOMPLETE = "incomplete"  # 回答不完整，可能被截断
    CONTEXT_MISUSE = "context_misuse"  # 有相关上下文但未正确使用
    FACTUAL_ERROR = "factual_error"  # 事实性错误
    FORMATTING_ERROR = "formatting_error"  # 格式错误（JSON、结构化输出）
    TOOL_ERROR = "tool_error"  # Agent工具调用错误
    UNKNOWN = "unknown"  # 无法分类


# Failure category descriptions and improvement suggestions
FAILURE_PATTERNS = {
    FailureCategory.HALLUCINATION: {
        "description": "模型编造了上下文中不存在的事实",
        "keywords": [
            "编造",
            "虚构",
            "不存在",
            "上下文中没有",
            "无法找到",
            "没有提及",
            "捏造",
            "不被支持",
        ],
        "suggestions": [
            "增强上下文约束Prompt，添加'只基于给定信息回答'的硬约束",
            "在系统Prompt中明确禁止编造信息",
            "提高检索质量，确保召回的上下文覆盖问题",
            "考虑增加事实性校验层（Factuality Check）",
            "降低模型temperature参数，减少随机性",
        ],
        "priority": "high",
    },
    FailureCategory.IRRELEVANT: {
        "description": "回答偏离问题核心，答非所问",
        "keywords": [
            "偏离",
            "无关",
            "不相关",
            "答非所问",
            "没有回答",
            "跑题",
            "不符合问题",
            "未解决",
        ],
        "suggestions": [
            "优化Prompt中的问题理解部分，明确回答要点",
            "检查检索是否召回了错误文档（相关性低）",
            "增加Query Rewrite步骤，消解歧义",
            "考虑使用Few-shot示例引导模型聚焦问题",
            "检查是否是问题本身表述不清",
        ],
        "priority": "medium",
    },
    FailureCategory.INCOMPLETE: {
        "description": "回答不完整，可能被截断或遗漏关键信息",
        "keywords": [
            "不完整",
            "截断",
            "遗漏",
            "部分",
            "缺少",
            "只回答了",
            "没有说明",
            "未提及",
        ],
        "suggestions": [
            "检查max_tokens设置，可能回答被截断",
            "优化上下文压缩策略，保留关键信息",
            "在Prompt中明确要求'完整回答所有要点'",
            "检查问题是否过于复杂，需要拆分",
            "考虑增加回答完整性检查机制",
        ],
        "priority": "medium",
    },
    FailureCategory.CONTEXT_MISUSE: {
        "description": "有相关上下文但未正确使用或理解错误",
        "keywords": [
            "未使用",
            "没有引用",
            "理解错误",
            "误解",
            "错误解读",
            "忽略了",
            "没有利用",
        ],
        "suggestions": [
            "检查上下文注入位置，确保模型能看到",
            "增强Rerank召回质量，把最相关的放在前面",
            "在Prompt中明确要求'基于以下上下文回答'",
            "考虑上下文是否过长导致Lost in the Middle",
            "检查上下文格式是否清晰（标题、段落分隔）",
        ],
        "priority": "high",
    },
    FailureCategory.FACTUAL_ERROR: {
        "description": "回答包含事实性错误",
        "keywords": [
            "错误",
            "不准确",
            "不正确",
            "与事实不符",
            "数据错误",
            "时间错误",
            "名称错误",
        ],
        "suggestions": [
            "检查检索到的上下文是否包含错误信息",
            "增加事实校验层，对关键信息做二次验证",
            "考虑使用知识图谱或结构化数据源",
            "检查模型是否依赖了过时的训练数据",
            "对数字、日期、专有名词做特殊处理",
        ],
        "priority": "high",
    },
    FailureCategory.FORMATTING_ERROR: {
        "description": "输出格式错误，如JSON格式不正确、结构化输出缺失",
        "keywords": [
            "格式错误",
            "JSON",
            "结构",
            "缺少字段",
            "解析失败",
            "格式不符",
        ],
        "suggestions": [
            "使用response_format={'type': 'json_object'}强制JSON输出",
            "在Prompt中提供明确的输出格式示例",
            "增加输出校验和重试机制",
            "考虑使用JSON Schema约束输出格式",
            "检查模型是否支持结构化输出",
        ],
        "priority": "medium",
    },
    FailureCategory.TOOL_ERROR: {
        "description": "Agent工具调用错误，包括参数错误、工具选择错误",
        "keywords": [
            "工具",
            "调用失败",
            "参数错误",
            "选择错误",
            "function calling",
        ],
        "suggestions": [
            "优化工具描述，让模型更容易理解工具用途",
            "简化工具参数，减少复杂度",
            "增加工具调用示例（Few-shot）",
            "检查工具schema是否清晰",
            "考虑增加工具调用前的参数校验",
        ],
        "priority": "high",
    },
    FailureCategory.UNKNOWN: {
        "description": "无法明确分类的失败",
        "keywords": [],
        "suggestions": [
            "人工复核该样本，分析具体失败原因",
            "补充到失败分类规则中",
            "考虑是否是评测标准本身的问题",
        ],
        "priority": "low",
    },
}


@dataclass
class FailureAnalysis:
    """Analysis result for a single failed evaluation."""

    sample_id: str
    metric_name: str
    score: float
    reasoning: str
    category: FailureCategory
    confidence: float  # 分类置信度
    matched_keywords: list[str]  # 匹配到的关键词
    suggestions: list[str]  # 改进建议
    priority: str  # high/medium/low


@dataclass
class ClusterAnalysis:
    """Analysis result for a cluster of similar failures."""

    category: FailureCategory
    sample_count: int
    avg_score: float
    common_patterns: list[str]  # 共同模式
    suggestions: list[str]  # 针对该聚类的改进建议
    sample_ids: list[str]  # 该聚类中的样本ID


@dataclass
class OptimizationReport:
    """Complete optimization report with prioritized suggestions."""

    total_samples: int
    failed_samples: int
    failure_rate: float
    category_distribution: dict[str, int]  # 失败类型分布
    clusters: list[ClusterAnalysis]  # 聚类分析结果
    prioritized_suggestions: list[dict[str, t.Any]]  # 按优先级排序的建议
    high_priority_count: int
    medium_priority_count: int
    low_priority_count: int


class ExplainableEvaluationAnalyzer:
    """
    Analyzer for providing explainable evaluation results with automatic
    failure classification and actionable improvement suggestions.

    Features:
    - Automatic failure reason classification (8 categories)
    - Keyword-based pattern matching
    - Failure clustering analysis
    - Prioritized improvement suggestions
    - Optimization report generation
    """

    def __init__(self, confidence_threshold: float = 0.6):
        """
        Initialize the analyzer.

        Args:
            confidence_threshold: Minimum confidence for classification (default 0.6)
        """
        self.confidence_threshold = confidence_threshold
        self.failure_analyses: list[FailureAnalysis] = []

    def classify_failure(
        self, sample_id: str, metric_name: str, score: float, reasoning: str
    ) -> FailureAnalysis:
        """
        Classify a single failure and provide suggestions.

        Args:
            sample_id: Unique identifier for the sample
            metric_name: Name of the metric that failed
            score: The evaluation score (typically < threshold)
            reasoning: The judge's reasoning for the score

        Returns:
            FailureAnalysis with category, confidence, and suggestions
        """
        # Convert reasoning to lowercase for matching
        reasoning_lower = reasoning.lower()

        # Try to match each category
        category_scores = {}
        matched_keywords_per_category = {}

        for category, pattern in FAILURE_PATTERNS.items():
            keywords = pattern["keywords"]
            if not keywords:  # UNKNOWN category has no keywords
                continue

            # Count matched keywords
            matched = [kw for kw in keywords if kw in reasoning_lower]
            matched_keywords_per_category[category] = matched

            # Calculate confidence based on keyword matches
            if matched:
                confidence = min(len(matched) / 3, 1.0)  # Normalize to 0-1
                category_scores[category] = confidence

        # Determine the best category
        if category_scores:
            best_category = max(category_scores, key=category_scores.get)
            confidence = category_scores[best_category]
            matched_keywords = matched_keywords_per_category[best_category]
        else:
            # No keywords matched, classify as UNKNOWN
            best_category = FailureCategory.UNKNOWN
            confidence = 0.0
            matched_keywords = []

        # Get suggestions for the category
        suggestions = FAILURE_PATTERNS[best_category]["suggestions"]
        priority = FAILURE_PATTERNS[best_category]["priority"]

        analysis = FailureAnalysis(
            sample_id=sample_id,
            metric_name=metric_name,
            score=score,
            reasoning=reasoning,
            category=best_category,
            confidence=confidence,
            matched_keywords=matched_keywords,
            suggestions=suggestions,
            priority=priority,
        )

        self.failure_analyses.append(analysis)
        return analysis

    def cluster_failures(self) -> list[ClusterAnalysis]:
        """
        Cluster failures by category and identify common patterns.

        Returns:
            List of ClusterAnalysis, one per category with failed samples
        """
        if not self.failure_analyses:
            return []

        # Group by category
        category_groups = defaultdict(list)
        for analysis in self.failure_analyses:
            category_groups[analysis.category].append(analysis)

        clusters = []
        for category, analyses in category_groups.items():
            # Calculate statistics
            sample_count = len(analyses)
            avg_score = sum(a.score for a in analyses) / sample_count
            sample_ids = [a.sample_id for a in analyses]

            # Find common patterns (most frequent keywords)
            all_keywords = []
            for a in analyses:
                all_keywords.extend(a.matched_keywords)

            keyword_counts = Counter(all_keywords)
            common_patterns = [kw for kw, count in keyword_counts.most_common(5)]

            # Get suggestions (same for all samples in a category)
            suggestions = FAILURE_PATTERNS[category]["suggestions"]

            cluster = ClusterAnalysis(
                category=category,
                sample_count=sample_count,
                avg_score=round(avg_score, 3),
                common_patterns=common_patterns,
                suggestions=suggestions,
                sample_ids=sample_ids,
            )
            clusters.append(cluster)

        # Sort by sample count (descending)
        clusters.sort(key=lambda c: c.sample_count, reverse=True)
        return clusters

    def generate_optimization_report(
        self, total_samples: int, pass_threshold: float = 0.7
    ) -> OptimizationReport:
        """
        Generate a comprehensive optimization report with prioritized suggestions.

        Args:
            total_samples: Total number of samples evaluated
            pass_threshold: Score threshold for pass/fail (default 0.7)

        Returns:
            OptimizationReport with clusters, suggestions, and statistics
        """
        failed_samples = len(self.failure_analyses)
        failure_rate = failed_samples / total_samples if total_samples > 0 else 0

        # Get category distribution
        category_counts = Counter(a.category for a in self.failure_analyses)
        category_distribution = {
            category.value: count for category, count in category_counts.items()
        }

        # Cluster failures
        clusters = self.cluster_failures()

        # Generate prioritized suggestions
        prioritized_suggestions = []
        high_priority_count = 0
        medium_priority_count = 0
        low_priority_count = 0

        for cluster in clusters:
            priority = FAILURE_PATTERNS[cluster.category]["priority"]
            impact_score = cluster.sample_count / failed_samples if failed_samples > 0 else 0

            suggestion_entry = {
                "category": cluster.category.value,
                "priority": priority,
                "impact_score": round(impact_score, 3),
                "affected_samples": cluster.sample_count,
                "suggestions": cluster.suggestions,
                "common_patterns": cluster.common_patterns,
            }
            prioritized_suggestions.append(suggestion_entry)

            if priority == "high":
                high_priority_count += 1
            elif priority == "medium":
                medium_priority_count += 1
            else:
                low_priority_count += 1

        # Sort by priority (high > medium > low) and then by impact
        priority_order = {"high": 0, "medium": 1, "low": 2}
        prioritized_suggestions.sort(
            key=lambda s: (priority_order[s["priority"]], -s["impact_score"])
        )

        report = OptimizationReport(
            total_samples=total_samples,
            failed_samples=failed_samples,
            failure_rate=round(failure_rate, 3),
            category_distribution=category_distribution,
            clusters=clusters,
            prioritized_suggestions=prioritized_suggestions,
            high_priority_count=high_priority_count,
            medium_priority_count=medium_priority_count,
            low_priority_count=low_priority_count,
        )

        return report

    def get_top_suggestions(self, top_n: int = 3) -> list[str]:
        """
        Get top N most impactful suggestions.

        Args:
            top_n: Number of suggestions to return

        Returns:
            List of suggestion strings
        """
        if not self.failure_analyses:
            return []

        # Cluster and generate report
        total_samples = len(self.failure_analyses)
        report = self.generate_optimization_report(total_samples)

        # Extract top suggestions
        top_suggestions = []
        for entry in report.prioritized_suggestions[:top_n]:
            category = entry["category"]
            impact = entry["impact_score"] * 100
            suggestions = entry["suggestions"]

            suggestion_text = (
                f"[{category.upper()}] ({impact:.1f}%影响) - "
                f"{suggestions[0] if suggestions else '人工复核'}"
            )
            top_suggestions.append(suggestion_text)

        return top_suggestions

    def to_dict(self) -> dict[str, t.Any]:
        """Export all analyses to dictionary for JSON serialization."""
        return {
            "failure_analyses": [
                {
                    "sample_id": a.sample_id,
                    "metric_name": a.metric_name,
                    "score": a.score,
                    "reasoning": a.reasoning,
                    "category": a.category.value,
                    "confidence": a.confidence,
                    "matched_keywords": a.matched_keywords,
                    "suggestions": a.suggestions,
                    "priority": a.priority,
                }
                for a in self.failure_analyses
            ]
        }


def analyze_evaluation_results(
    eval_details: list[dict[str, t.Any]], pass_threshold: float = 0.7
) -> OptimizationReport:
    """
    Convenience function to analyze evaluation results and generate report.

    Args:
        eval_details: List of evaluation detail records with sample_id, metric_name,
                     score, and reasoning
        pass_threshold: Score threshold for considering a sample as failed

    Returns:
        OptimizationReport with comprehensive analysis and suggestions

    Example:
        >>> eval_details = [
        ...     {"sample_id": "1", "metric_name": "faithfulness", "score": 0.3,
        ...      "reasoning": "回答编造了上下文中不存在的事实"},
        ...     {"sample_id": "2", "metric_name": "relevance", "score": 0.5,
        ...      "reasoning": "回答偏离了问题核心"},
        ... ]
        >>> report = analyze_evaluation_results(eval_details)
        >>> print(f"失败率: {report.failure_rate}")
        >>> print(f"主要失败类型: {report.category_distribution}")
    """
    analyzer = ExplainableEvaluationAnalyzer()

    # Classify all failed samples
    for detail in eval_details:
        score = detail.get("score", 0)
        if score < pass_threshold:
            analyzer.classify_failure(
                sample_id=detail.get("sample_id", "unknown"),
                metric_name=detail.get("metric_name", "unknown"),
                score=score,
                reasoning=detail.get("reasoning", ""),
            )

    # Generate comprehensive report
    total_samples = len(eval_details)
    report = analyzer.generate_optimization_report(total_samples, pass_threshold)

    return report
