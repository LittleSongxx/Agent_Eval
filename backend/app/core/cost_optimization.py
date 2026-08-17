"""
Cost Optimization Module for Evaluation System.

This module provides tiered model routing, batch inference, and caching
strategies to optimize evaluation costs while maintaining quality.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import typing as t
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ModelTier(str, Enum):
    """Model tiers for cost optimization."""

    TURBO = "turbo"  # Cheapest, for simple metrics
    PLUS = "plus"  # Medium cost, for standard metrics
    MAX = "max"  # Most expensive, for complex metrics


class MetricComplexity(str, Enum):
    """Complexity levels for metrics."""

    LOW = "low"  # Simple metrics: BM25, HitRate, format validation
    MEDIUM = "medium"  # Standard metrics: relevance, completeness
    HIGH = "high"  # Complex metrics: faithfulness, Agent trajectory


# Model configuration with pricing (example values in CNY per 1k tokens)
MODEL_CONFIGS = {
    ModelTier.TURBO: {
        "model_name": "qwen-turbo",
        "cost_per_1k_tokens": 0.001,  # 10x cheaper than plus
        "suitable_for": [MetricComplexity.LOW],
    },
    ModelTier.PLUS: {
        "model_name": "qwen-plus",
        "cost_per_1k_tokens": 0.004,
        "suitable_for": [MetricComplexity.LOW, MetricComplexity.MEDIUM],
    },
    ModelTier.MAX: {
        "model_name": "qwen-max",
        "cost_per_1k_tokens": 0.012,  # 3x more expensive than plus
        "suitable_for": [MetricComplexity.LOW, MetricComplexity.MEDIUM, MetricComplexity.HIGH],
    },
}


@dataclass
class CostEstimate:
    """Cost estimation result."""

    total_tokens: int
    estimated_cost_cny: float
    model_tier: ModelTier
    samples_count: int
    cost_per_sample: float


@dataclass
class OptimizationResult:
    """Result of cost optimization comparison."""

    baseline_cost: float
    optimized_cost: float
    savings_amount: float
    savings_percentage: float
    baseline_tokens: int
    optimized_tokens: int
    quality_degradation: float  # Estimated quality loss (0-1)


class TieredModelRouter:
    """
    Router that assigns appropriate model tier based on metric complexity.

    Features:
    - Automatic complexity detection
    - Cost-quality trade-off optimization
    - Quality degradation estimation
    """

    def __init__(self, default_tier: ModelTier = ModelTier.PLUS):
        """
        Initialize the router.

        Args:
            default_tier: Default tier when complexity cannot be determined
        """
        self.default_tier = default_tier

        # Metric complexity mapping (can be configured per deployment)
        self.metric_complexity_map = {
            # Low complexity (simple rule-based or format checks)
            "bm25": MetricComplexity.LOW,
            "hit_rate": MetricComplexity.LOW,
            "mrr": MetricComplexity.LOW,
            "json_validity": MetricComplexity.LOW,
            # Medium complexity (standard LLM judgment)
            "relevance": MetricComplexity.MEDIUM,
            "completeness": MetricComplexity.MEDIUM,
            "context_precision": MetricComplexity.MEDIUM,
            "answer_relevance": MetricComplexity.MEDIUM,
            # High complexity (requires deep reasoning)
            "faithfulness": MetricComplexity.HIGH,
            "context_recall": MetricComplexity.HIGH,
            "agent_trajectory": MetricComplexity.HIGH,
            "tool_selection_accuracy": MetricComplexity.HIGH,
        }

    def get_metric_complexity(self, metric_name: str) -> MetricComplexity:
        """
        Determine the complexity of a metric.

        Args:
            metric_name: Name of the metric

        Returns:
            MetricComplexity level
        """
        # Normalize metric name
        normalized = metric_name.lower().replace("_", "").replace("-", "")

        # Check exact match first
        if metric_name in self.metric_complexity_map:
            return self.metric_complexity_map[metric_name]

        # Check partial match
        for key, complexity in self.metric_complexity_map.items():
            if key in normalized or normalized in key:
                return complexity

        # Default to medium if unknown
        logger.warning(f"Unknown metric complexity for {metric_name}, using MEDIUM")
        return MetricComplexity.MEDIUM

    def route_to_tier(
        self, metric_name: str, force_tier: ModelTier | None = None
    ) -> ModelTier:
        """
        Route a metric to appropriate model tier.

        Args:
            metric_name: Name of the metric
            force_tier: Optional tier override

        Returns:
            ModelTier to use
        """
        if force_tier:
            return force_tier

        complexity = self.get_metric_complexity(metric_name)

        # Route based on complexity
        if complexity == MetricComplexity.LOW:
            return ModelTier.TURBO
        elif complexity == MetricComplexity.MEDIUM:
            return ModelTier.PLUS
        else:
            return ModelTier.MAX

    def estimate_quality_degradation(
        self, metric_name: str, from_tier: ModelTier, to_tier: ModelTier
    ) -> float:
        """
        Estimate quality degradation when downgrading model tier.

        Args:
            metric_name: Name of the metric
            from_tier: Original tier
            to_tier: Target tier

        Returns:
            Estimated quality degradation (0-1, 0 means no degradation)
        """
        if from_tier == to_tier:
            return 0.0

        complexity = self.get_metric_complexity(metric_name)

        # Define tier order
        tier_order = {ModelTier.TURBO: 0, ModelTier.PLUS: 1, ModelTier.MAX: 2}

        # If upgrading, no degradation
        if tier_order[to_tier] > tier_order[from_tier]:
            return 0.0

        # Calculate degradation based on complexity and tier gap
        tier_gap = tier_order[from_tier] - tier_order[to_tier]

        # Low complexity metrics tolerate downgrade well
        if complexity == MetricComplexity.LOW:
            return tier_gap * 0.02  # 2% per tier
        # Medium complexity metrics have moderate degradation
        elif complexity == MetricComplexity.MEDIUM:
            return tier_gap * 0.05  # 5% per tier
        # High complexity metrics suffer more from downgrade
        else:
            return tier_gap * 0.10  # 10% per tier


class EvaluationCostOptimizer:
    """
    Comprehensive cost optimizer combining tiered routing, caching, and batch inference.

    Features:
    - Tiered model routing
    - Result caching
    - Batch inference support
    - Cost-quality trade-off analysis
    """

    def __init__(
        self,
        enable_tiered_routing: bool = True,
        enable_caching: bool = True,
        cache_ttl_seconds: int = 86400,  # 24 hours
    ):
        """
        Initialize the optimizer.

        Args:
            enable_tiered_routing: Enable automatic tier selection
            enable_caching: Enable result caching
            cache_ttl_seconds: Cache TTL in seconds
        """
        self.enable_tiered_routing = enable_tiered_routing
        self.enable_caching = enable_caching
        self.cache_ttl_seconds = cache_ttl_seconds

        self.router = TieredModelRouter()
        self.cache: dict[str, tuple[t.Any, float]] = {}  # key -> (result, timestamp)

    def _compute_cache_key(
        self, question: str, answer: str, context: str, metric_name: str, criteria: str
    ) -> str:
        """
        Compute cache key for a sample.

        Args:
            question: User question
            answer: Model answer
            context: Retrieved context
            metric_name: Metric being evaluated
            criteria: Evaluation criteria

        Returns:
            Cache key (hash)
        """
        content = f"{question}|{answer}|{context}|{metric_name}|{criteria}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _is_cache_valid(self, timestamp: float) -> bool:
        """Check if cached result is still valid."""
        return (time.time() - timestamp) < self.cache_ttl_seconds

    def estimate_cost(
        self,
        num_samples: int,
        avg_tokens_per_sample: int,
        model_tier: ModelTier,
    ) -> CostEstimate:
        """
        Estimate cost for evaluation.

        Args:
            num_samples: Number of samples
            avg_tokens_per_sample: Average tokens per sample
            model_tier: Model tier to use

        Returns:
            CostEstimate with details
        """
        total_tokens = num_samples * avg_tokens_per_sample
        cost_per_1k = MODEL_CONFIGS[model_tier]["cost_per_1k_tokens"]
        estimated_cost = (total_tokens / 1000) * cost_per_1k

        return CostEstimate(
            total_tokens=total_tokens,
            estimated_cost_cny=round(estimated_cost, 2),
            model_tier=model_tier,
            samples_count=num_samples,
            cost_per_sample=round(estimated_cost / num_samples, 4),
        )

    def compare_optimization_strategies(
        self,
        num_samples: int,
        metrics: list[str],
        avg_tokens_per_sample: int = 600,
    ) -> dict[str, t.Any]:
        """
        Compare different optimization strategies.

        Args:
            num_samples: Number of samples to evaluate
            metrics: List of metric names
            avg_tokens_per_sample: Average tokens per sample

        Returns:
            Comparison report with costs and quality trade-offs
        """
        # Baseline: all metrics use PLUS
        baseline_cost = 0.0
        baseline_tokens = 0
        for metric in metrics:
            estimate = self.estimate_cost(num_samples, avg_tokens_per_sample, ModelTier.PLUS)
            baseline_cost += estimate.estimated_cost_cny
            baseline_tokens += estimate.total_tokens

        # Strategy 1: Tiered routing
        tiered_cost = 0.0
        tiered_tokens = 0
        tier_distribution = {tier: 0 for tier in ModelTier}
        quality_degradations = []

        for metric in metrics:
            tier = self.router.route_to_tier(metric)
            tier_distribution[tier] += 1

            estimate = self.estimate_cost(num_samples, avg_tokens_per_sample, tier)
            tiered_cost += estimate.estimated_cost_cny
            tiered_tokens += estimate.total_tokens

            # Estimate quality degradation from PLUS to assigned tier
            degradation = self.router.estimate_quality_degradation(metric, ModelTier.PLUS, tier)
            quality_degradations.append(degradation)

        avg_quality_degradation = (
            sum(quality_degradations) / len(quality_degradations) if quality_degradations else 0
        )

        # Strategy 2: Aggressive downgrade (all TURBO)
        aggressive_cost = 0.0
        aggressive_tokens = 0
        aggressive_degradations = []

        for metric in metrics:
            estimate = self.estimate_cost(num_samples, avg_tokens_per_sample, ModelTier.TURBO)
            aggressive_cost += estimate.estimated_cost_cny
            aggressive_tokens += estimate.total_tokens

            degradation = self.router.estimate_quality_degradation(
                metric, ModelTier.PLUS, ModelTier.TURBO
            )
            aggressive_degradations.append(degradation)

        avg_aggressive_degradation = (
            sum(aggressive_degradations) / len(aggressive_degradations) if aggressive_degradations else 0
        )

        # Strategy 3: With caching (assume 30% cache hit rate)
        cache_hit_rate = 0.3
        cached_samples = int(num_samples * cache_hit_rate)
        uncached_samples = num_samples - cached_samples

        cached_cost = (tiered_cost / num_samples) * uncached_samples
        cached_tokens = (tiered_tokens / num_samples) * uncached_samples

        return {
            "baseline": {
                "strategy": "All metrics use PLUS tier",
                "cost_cny": round(baseline_cost, 2),
                "tokens": baseline_tokens,
                "quality_degradation": 0.0,
            },
            "tiered_routing": {
                "strategy": "Automatic tier selection by complexity",
                "cost_cny": round(tiered_cost, 2),
                "tokens": tiered_tokens,
                "savings_amount": round(baseline_cost - tiered_cost, 2),
                "savings_percentage": round((1 - tiered_cost / baseline_cost) * 100, 1)
                if baseline_cost > 0
                else 0,
                "quality_degradation": round(avg_quality_degradation, 3),
                "tier_distribution": {tier.value: count for tier, count in tier_distribution.items()},
            },
            "aggressive_downgrade": {
                "strategy": "All metrics use TURBO tier",
                "cost_cny": round(aggressive_cost, 2),
                "tokens": aggressive_tokens,
                "savings_amount": round(baseline_cost - aggressive_cost, 2),
                "savings_percentage": round((1 - aggressive_cost / baseline_cost) * 100, 1)
                if baseline_cost > 0
                else 0,
                "quality_degradation": round(avg_aggressive_degradation, 3),
            },
            "tiered_with_caching": {
                "strategy": "Tiered routing + 30% cache hit rate",
                "cost_cny": round(cached_cost, 2),
                "tokens": int(cached_tokens),
                "savings_amount": round(baseline_cost - cached_cost, 2),
                "savings_percentage": round((1 - cached_cost / baseline_cost) * 100, 1)
                if baseline_cost > 0
                else 0,
                "quality_degradation": round(avg_quality_degradation, 3),
                "cache_hit_rate": cache_hit_rate,
                "cached_samples": cached_samples,
            },
            "summary": {
                "num_samples": num_samples,
                "num_metrics": len(metrics),
                "baseline_cost": round(baseline_cost, 2),
                "recommended_strategy": "tiered_with_caching",
                "best_cost": round(cached_cost, 2),
                "best_savings": round(baseline_cost - cached_cost, 2),
                "best_savings_percentage": round((1 - cached_cost / baseline_cost) * 100, 1)
                if baseline_cost > 0
                else 0,
            },
        }

    def get_optimization_recommendations(
        self, num_samples: int, budget_cny: float, metrics: list[str]
    ) -> list[str]:
        """
        Get optimization recommendations based on budget.

        Args:
            num_samples: Number of samples
            budget_cny: Budget in CNY
            metrics: List of metrics

        Returns:
            List of recommendation strings
        """
        comparison = self.compare_optimization_strategies(num_samples, metrics)

        baseline_cost = comparison["baseline"]["cost_cny"]
        tiered_cost = comparison["tiered_routing"]["cost_cny"]
        cached_cost = comparison["tiered_with_caching"]["cost_cny"]

        recommendations = []

        if budget_cny >= baseline_cost:
            recommendations.append(
                f"✅ 预算充足（¥{budget_cny} >= ¥{baseline_cost}），可使用基线配置（全PLUS）"
            )
        elif budget_cny >= tiered_cost:
            recommendations.append(
                f"💡 建议使用分层路由策略，成本¥{tiered_cost}，"
                f"节省{comparison['tiered_routing']['savings_percentage']}%，"
                f"质量下降仅{comparison['tiered_routing']['quality_degradation']*100:.1f}%"
            )
        elif budget_cny >= cached_cost:
            recommendations.append(
                f"💡 建议使用分层路由+缓存策略，成本¥{cached_cost}，"
                f"节省{comparison['tiered_with_caching']['savings_percentage']}%"
            )
        else:
            shortfall = cached_cost - budget_cny
            recommendations.append(
                f"⚠️ 预算不足（差¥{shortfall:.2f}），建议：\n"
                f"  1. 减少样本数量\n"
                f"  2. 减少评测指标\n"
                f"  3. 使用采样评测（评测部分样本）"
            )

        # Always recommend caching if available
        if self.enable_caching:
            cache_savings = tiered_cost - cached_cost
            recommendations.append(
                f"💾 启用缓存可额外节省¥{cache_savings:.2f}（假设30%缓存命中率）"
            )

        return recommendations


def estimate_batch_inference_cost(
    num_samples: int,
    model_tier: ModelTier = ModelTier.PLUS,
    use_batch_api: bool = False,
) -> dict[str, t.Any]:
    """
    Estimate cost comparison between regular and batch inference.

    OpenAI Batch API provides 50% discount but 24-hour latency.

    Args:
        num_samples: Number of samples
        model_tier: Model tier to use
        use_batch_api: Whether to use batch API

    Returns:
        Cost comparison
    """
    avg_tokens = 600  # Typical tokens per sample
    total_tokens = num_samples * avg_tokens
    cost_per_1k = MODEL_CONFIGS[model_tier]["cost_per_1k_tokens"]

    regular_cost = (total_tokens / 1000) * cost_per_1k

    if use_batch_api:
        # 50% discount for batch API
        batch_cost = regular_cost * 0.5
        savings = regular_cost - batch_cost
        savings_percentage = 50.0
    else:
        batch_cost = regular_cost
        savings = 0.0
        savings_percentage = 0.0

    return {
        "num_samples": num_samples,
        "total_tokens": total_tokens,
        "model_tier": model_tier.value,
        "regular_cost_cny": round(regular_cost, 2),
        "batch_cost_cny": round(batch_cost, 2),
        "savings_cny": round(savings, 2),
        "savings_percentage": savings_percentage,
        "batch_latency_hours": 24 if use_batch_api else 0,
        "recommendation": (
            "建议使用Batch API，节省50%成本（24小时内完成）"
            if num_samples >= 100 and use_batch_api
            else "样本数较少，建议使用实时API"
        ),
    }
