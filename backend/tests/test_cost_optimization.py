"""
Unit tests for Cost Optimization Module.
"""

import pytest

from app.core.cost_optimization import (
    EvaluationCostOptimizer,
    MetricComplexity,
    ModelTier,
    TieredModelRouter,
    estimate_batch_inference_cost,
)


def test_tiered_router_low_complexity():
    """Test routing of low complexity metrics."""
    router = TieredModelRouter()

    # Low complexity metrics should route to TURBO
    assert router.route_to_tier("bm25") == ModelTier.TURBO
    assert router.route_to_tier("hit_rate") == ModelTier.TURBO
    assert router.route_to_tier("json_validity") == ModelTier.TURBO


def test_tiered_router_medium_complexity():
    """Test routing of medium complexity metrics."""
    router = TieredModelRouter()

    # Medium complexity metrics should route to PLUS
    assert router.route_to_tier("relevance") == ModelTier.PLUS
    assert router.route_to_tier("completeness") == ModelTier.PLUS
    assert router.route_to_tier("answer_relevance") == ModelTier.PLUS


def test_tiered_router_high_complexity():
    """Test routing of high complexity metrics."""
    router = TieredModelRouter()

    # High complexity metrics should route to MAX
    assert router.route_to_tier("faithfulness") == ModelTier.MAX
    assert router.route_to_tier("context_recall") == ModelTier.MAX
    assert router.route_to_tier("agent_trajectory") == ModelTier.MAX


def test_tiered_router_force_tier():
    """Test forcing a specific tier."""
    router = TieredModelRouter()

    # Force tier should override automatic routing
    assert router.route_to_tier("bm25", force_tier=ModelTier.MAX) == ModelTier.MAX
    assert router.route_to_tier("faithfulness", force_tier=ModelTier.TURBO) == ModelTier.TURBO


def test_tiered_router_unknown_metric():
    """Test routing of unknown metric defaults to MEDIUM."""
    router = TieredModelRouter()

    # Unknown metric should route to PLUS (medium complexity default)
    tier = router.route_to_tier("unknown_custom_metric")
    assert tier == ModelTier.PLUS


def test_get_metric_complexity():
    """Test metric complexity detection."""
    router = TieredModelRouter()

    assert router.get_metric_complexity("bm25") == MetricComplexity.LOW
    assert router.get_metric_complexity("relevance") == MetricComplexity.MEDIUM
    assert router.get_metric_complexity("faithfulness") == MetricComplexity.HIGH


def test_quality_degradation_no_change():
    """Test quality degradation when tier doesn't change."""
    router = TieredModelRouter()

    degradation = router.estimate_quality_degradation(
        "relevance", from_tier=ModelTier.PLUS, to_tier=ModelTier.PLUS
    )

    assert degradation == 0.0


def test_quality_degradation_upgrade():
    """Test quality degradation when upgrading tier (should be 0)."""
    router = TieredModelRouter()

    degradation = router.estimate_quality_degradation(
        "relevance", from_tier=ModelTier.TURBO, to_tier=ModelTier.PLUS
    )

    assert degradation == 0.0


def test_quality_degradation_downgrade_low_complexity():
    """Test quality degradation for low complexity metric downgrade."""
    router = TieredModelRouter()

    degradation = router.estimate_quality_degradation(
        "bm25",  # Low complexity
        from_tier=ModelTier.PLUS,
        to_tier=ModelTier.TURBO,
    )

    # Low complexity metrics tolerate downgrade well (2% per tier)
    assert degradation == pytest.approx(0.02, abs=0.001)


def test_quality_degradation_downgrade_high_complexity():
    """Test quality degradation for high complexity metric downgrade."""
    router = TieredModelRouter()

    degradation = router.estimate_quality_degradation(
        "faithfulness",  # High complexity
        from_tier=ModelTier.MAX,
        to_tier=ModelTier.TURBO,
    )

    # High complexity metrics suffer more (10% per tier, 2 tiers = 20%)
    assert degradation == pytest.approx(0.20, abs=0.001)


def test_cost_estimate():
    """Test cost estimation."""
    optimizer = EvaluationCostOptimizer()

    estimate = optimizer.estimate_cost(
        num_samples=1000, avg_tokens_per_sample=600, model_tier=ModelTier.PLUS
    )

    assert estimate.samples_count == 1000
    assert estimate.total_tokens == 600000
    assert estimate.model_tier == ModelTier.PLUS
    assert estimate.estimated_cost_cny > 0
    assert estimate.cost_per_sample > 0


def test_cost_estimate_different_tiers():
    """Test that different tiers have different costs."""
    optimizer = EvaluationCostOptimizer()

    turbo = optimizer.estimate_cost(1000, 600, ModelTier.TURBO)
    plus = optimizer.estimate_cost(1000, 600, ModelTier.PLUS)
    max_tier = optimizer.estimate_cost(1000, 600, ModelTier.MAX)

    # Costs should be: TURBO < PLUS < MAX
    assert turbo.estimated_cost_cny < plus.estimated_cost_cny
    assert plus.estimated_cost_cny < max_tier.estimated_cost_cny


def test_compare_optimization_strategies():
    """Test comparison of different optimization strategies."""
    optimizer = EvaluationCostOptimizer()

    # Use metrics without high complexity to ensure tiered routing saves money
    metrics = ["bm25", "relevance", "completeness", "answer_relevance"]
    comparison = optimizer.compare_optimization_strategies(
        num_samples=1000, metrics=metrics
    )

    # Check all strategies are present
    assert "baseline" in comparison
    assert "tiered_routing" in comparison
    assert "aggressive_downgrade" in comparison
    assert "tiered_with_caching" in comparison
    assert "summary" in comparison

    # Check costs are present
    baseline_cost = comparison["baseline"]["cost_cny"]
    tiered_cost = comparison["tiered_routing"]["cost_cny"]
    cached_cost = comparison["tiered_with_caching"]["cost_cny"]

    assert baseline_cost > 0
    assert tiered_cost > 0
    assert cached_cost > 0

    # Caching should always save money
    assert cached_cost <= tiered_cost


def test_tiered_routing_reduces_cost():
    """Test that tiered routing reduces cost when using simple metrics."""
    optimizer = EvaluationCostOptimizer()

    # Use only low and medium complexity metrics
    metrics = ["bm25", "hit_rate", "relevance", "completeness"]
    comparison = optimizer.compare_optimization_strategies(
        num_samples=1000, metrics=metrics
    )

    baseline_cost = comparison["baseline"]["cost_cny"]
    tiered_cost = comparison["tiered_routing"]["cost_cny"]

    # Tiered routing should save money for low/medium complexity metrics
    assert tiered_cost < baseline_cost
    assert comparison["tiered_routing"]["savings_percentage"] > 0


def test_caching_further_reduces_cost():
    """Test that caching provides additional savings."""
    optimizer = EvaluationCostOptimizer(enable_caching=True)

    metrics = ["relevance", "faithfulness"]
    comparison = optimizer.compare_optimization_strategies(
        num_samples=1000, metrics=metrics
    )

    tiered_cost = comparison["tiered_routing"]["cost_cny"]
    cached_cost = comparison["tiered_with_caching"]["cost_cny"]

    # Caching should save additional money
    assert cached_cost < tiered_cost


def test_tier_distribution_in_routing():
    """Test that tier distribution is correctly calculated."""
    optimizer = EvaluationCostOptimizer()

    metrics = [
        "bm25",  # TURBO
        "hit_rate",  # TURBO
        "relevance",  # PLUS
        "faithfulness",  # MAX
    ]

    comparison = optimizer.compare_optimization_strategies(
        num_samples=100, metrics=metrics
    )

    tier_dist = comparison["tiered_routing"]["tier_distribution"]

    # Should have 2 TURBO, 1 PLUS, 1 MAX
    assert tier_dist["turbo"] == 2
    assert tier_dist["plus"] == 1
    assert tier_dist["max"] == 1


def test_quality_degradation_in_comparison():
    """Test that quality degradation is estimated in comparison."""
    optimizer = EvaluationCostOptimizer()

    metrics = ["faithfulness"]  # High complexity
    comparison = optimizer.compare_optimization_strategies(
        num_samples=100, metrics=metrics
    )

    # Aggressive downgrade should have higher degradation
    aggressive_degradation = comparison["aggressive_downgrade"]["quality_degradation"]
    tiered_degradation = comparison["tiered_routing"]["quality_degradation"]

    # Aggressive (all TURBO) should degrade more than tiered (uses MAX for faithfulness)
    assert aggressive_degradation > tiered_degradation


def test_get_optimization_recommendations_sufficient_budget():
    """Test recommendations when budget is sufficient."""
    optimizer = EvaluationCostOptimizer()

    metrics = ["relevance", "faithfulness"]
    comparison = optimizer.compare_optimization_strategies(
        num_samples=100, metrics=metrics
    )

    baseline_cost = comparison["baseline"]["cost_cny"]
    recommendations = optimizer.get_optimization_recommendations(
        num_samples=100, budget_cny=baseline_cost * 2, metrics=metrics
    )

    # Should recommend baseline when budget is sufficient
    assert len(recommendations) > 0
    assert any("预算充足" in rec for rec in recommendations)


def test_get_optimization_recommendations_limited_budget():
    """Test recommendations when budget is limited."""
    optimizer = EvaluationCostOptimizer()

    metrics = ["relevance", "faithfulness"]
    comparison = optimizer.compare_optimization_strategies(
        num_samples=1000, metrics=metrics
    )

    tiered_cost = comparison["tiered_routing"]["cost_cny"]
    recommendations = optimizer.get_optimization_recommendations(
        num_samples=1000, budget_cny=tiered_cost * 0.9, metrics=metrics
    )

    # Should recommend optimization strategies
    assert len(recommendations) > 0
    assert any("分层路由" in rec or "缓存" in rec for rec in recommendations)


def test_get_optimization_recommendations_insufficient_budget():
    """Test recommendations when budget is insufficient."""
    optimizer = EvaluationCostOptimizer()

    metrics = ["relevance", "faithfulness"]
    recommendations = optimizer.get_optimization_recommendations(
        num_samples=10000, budget_cny=1.0, metrics=metrics  # Very small budget
    )

    # Should warn about insufficient budget
    assert len(recommendations) > 0
    assert any("预算不足" in rec for rec in recommendations)


def test_batch_inference_cost_regular():
    """Test batch inference cost estimation without batch API."""
    result = estimate_batch_inference_cost(
        num_samples=1000, model_tier=ModelTier.PLUS, use_batch_api=False
    )

    assert result["num_samples"] == 1000
    assert result["regular_cost_cny"] == result["batch_cost_cny"]
    assert result["savings_cny"] == 0.0
    assert result["savings_percentage"] == 0.0
    assert result["batch_latency_hours"] == 0


def test_batch_inference_cost_with_batch_api():
    """Test batch inference cost estimation with batch API."""
    result = estimate_batch_inference_cost(
        num_samples=1000, model_tier=ModelTier.PLUS, use_batch_api=True
    )

    assert result["num_samples"] == 1000
    assert result["batch_cost_cny"] == result["regular_cost_cny"] * 0.5
    assert result["savings_percentage"] == 50.0
    assert result["batch_latency_hours"] == 24


def test_batch_inference_recommendation():
    """Test batch inference recommendation logic."""
    # Large sample count should recommend batch API
    result_large = estimate_batch_inference_cost(
        num_samples=1000, use_batch_api=True
    )
    assert "Batch API" in result_large["recommendation"]

    # Small sample count should recommend real-time
    result_small = estimate_batch_inference_cost(
        num_samples=10, use_batch_api=False
    )
    assert "实时API" in result_small["recommendation"]


def test_cache_key_computation():
    """Test cache key computation is consistent."""
    optimizer = EvaluationCostOptimizer(enable_caching=True)

    key1 = optimizer._compute_cache_key("q1", "a1", "c1", "m1", "cr1")
    key2 = optimizer._compute_cache_key("q1", "a1", "c1", "m1", "cr1")
    key3 = optimizer._compute_cache_key("q2", "a1", "c1", "m1", "cr1")

    # Same input should produce same key
    assert key1 == key2
    # Different input should produce different key
    assert key1 != key3


def test_optimizer_initialization():
    """Test optimizer initialization with different settings."""
    # Default settings
    opt1 = EvaluationCostOptimizer()
    assert opt1.enable_tiered_routing is True
    assert opt1.enable_caching is True

    # Custom settings
    opt2 = EvaluationCostOptimizer(
        enable_tiered_routing=False, enable_caching=False
    )
    assert opt2.enable_tiered_routing is False
    assert opt2.enable_caching is False
