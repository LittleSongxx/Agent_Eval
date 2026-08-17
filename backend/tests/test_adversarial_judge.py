"""
Unit tests for the Adversarial Judge System.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.adversarial_judge import (
    AdversarialJudgeSystem,
    AdversarialEvaluationResult,
    AdversarialAgreementAnalyzer,
)


@pytest.fixture
def mock_llm_config():
    """Mock LLM configuration."""
    config = MagicMock()
    config.model_name = "qwen-plus"
    config.api_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    config.api_key = "test-key"
    config.temperature = 0.01
    config.max_tokens = 1024
    return config


@pytest.fixture
def mock_primary_config(mock_llm_config):
    """Mock primary judge configuration."""
    config = MagicMock()
    config.model_name = "qwen-plus"
    config.api_base_url = mock_llm_config.api_base_url
    config.api_key = mock_llm_config.api_key
    config.temperature = 0.01
    config.max_tokens = 1024
    return config


@pytest.fixture
def mock_adversary_config(mock_llm_config):
    """Mock adversary judge configuration."""
    config = MagicMock()
    config.model_name = "gpt-4o"
    config.api_base_url = "https://api.openai.com/v1"
    config.api_key = "test-key"
    config.temperature = 0.01
    config.max_tokens = 1024
    return config


@pytest.mark.asyncio
async def test_adversarial_evaluation_high_agreement(mock_primary_config, mock_adversary_config):
    """Test adversarial evaluation with high agreement between judges."""
    # Mock OpenAIJudgeClient initialization to avoid openai import
    with patch("app.core.adversarial_judge.OpenAIJudgeClient") as MockJudgeClient:
        mock_primary_judge = MagicMock()
        mock_adversary_judge = MagicMock()
        MockJudgeClient.side_effect = [mock_primary_judge, mock_adversary_judge]

        system = AdversarialJudgeSystem(
            primary_llm_config=mock_primary_config,
            adversary_llm_config=mock_adversary_config,
            disagreement_threshold=0.3,
        )

    # Mock the judge methods
    with patch.object(
        system.primary_judge, "judge_json", new_callable=AsyncMock
    ) as mock_primary_initial, patch.object(
        system.adversary_judge, "chat_json", new_callable=AsyncMock
    ) as mock_adversary_critique, patch.object(
        system.primary_judge, "chat_json", new_callable=AsyncMock
    ) as mock_consensus, patch.object(
        system.primary_judge, "take_row_usage", return_value={"total_tokens": 100}
    ), patch.object(
        system.adversary_judge, "take_row_usage", return_value={"total_tokens": 150}
    ):
        # Primary judge gives 0.8
        mock_primary_initial.return_value = {"score": 0.8, "reason": "回答相关且有证据支持"}

        # Adversary judge gives 0.75 (high agreement)
        mock_adversary_critique.return_value = {
            "critique": "初评合理，但部分细节可以更严格",
            "counter_examples": ["某些表述略显模糊"],
            "suggested_score": 0.75,
            "confidence": 0.9,
            "key_concerns": ["表述精确性"],
        }

        # Consensus slightly adjusts
        mock_consensus.return_value = {
            "final_score": 0.78,
            "final_reasoning": "综合考虑质疑后，微调分数",
            "critique_validity": 0.7,
            "adjusted_reasoning": "认同部分质疑",
        }

        sample = {
            "user_input": "什么是RAG？",
            "response": "RAG是检索增强生成...",
            "retrieved_contexts": ["RAG定义文档"],
        }

        result = await system.evaluate_with_adversary(
            sample=sample, metric_name="faithfulness", criteria="判断回答是否忠实于上下文", score_instruction="0-1分数"
        )

        # Assertions
        assert result.primary_score == 0.8
        assert result.adversary_score == 0.75
        assert result.final_score == 0.78
        assert result.disagreement == pytest.approx(0.05, abs=0.01)
        assert result.needs_human_review is False
        assert result.agreement_category == "high"
        assert result.total_tokens > 0


@pytest.mark.asyncio
async def test_adversarial_evaluation_high_disagreement(mock_primary_config, mock_adversary_config):
    """Test adversarial evaluation with high disagreement (needs human review)."""
    # Mock OpenAIJudgeClient initialization
    with patch("app.core.adversarial_judge.OpenAIJudgeClient") as MockJudgeClient:
        mock_primary_judge = MagicMock()
        mock_adversary_judge = MagicMock()
        MockJudgeClient.side_effect = [mock_primary_judge, mock_adversary_judge]

        system = AdversarialJudgeSystem(
            primary_llm_config=mock_primary_config,
            adversary_llm_config=mock_adversary_config,
            disagreement_threshold=0.3,
        )

    with patch.object(
        system.primary_judge, "judge_json", new_callable=AsyncMock
    ) as mock_primary_initial, patch.object(
        system.adversary_judge, "chat_json", new_callable=AsyncMock
    ) as mock_adversary_critique, patch.object(
        system.primary_judge, "chat_json", new_callable=AsyncMock
    ) as mock_consensus, patch.object(
        system.primary_judge, "take_row_usage", return_value={"total_tokens": 100}
    ), patch.object(
        system.adversary_judge, "take_row_usage", return_value={"total_tokens": 150}
    ):
        # Primary judge gives 0.8
        mock_primary_initial.return_value = {"score": 0.8, "reason": "回答完整"}

        # Adversary judge gives 0.4 (high disagreement)
        mock_adversary_critique.return_value = {
            "critique": "回答存在明显事实错误",
            "counter_examples": ["与上下文冲突的表述"],
            "suggested_score": 0.4,
            "confidence": 0.85,
            "key_concerns": ["事实准确性", "上下文一致性"],
        }

        # Consensus adjusts significantly
        mock_consensus.return_value = {
            "final_score": 0.55,
            "final_reasoning": "认真考虑质疑后，发现确实存在问题",
            "critique_validity": 0.8,
            "adjusted_reasoning": "承认初评过于宽松",
        }

        sample = {"user_input": "测试问题", "response": "测试回答"}

        result = await system.evaluate_with_adversary(
            sample=sample, metric_name="test_metric", criteria="测试标准", score_instruction="0-1分数"
        )

        # Assertions
        assert result.primary_score == 0.8
        assert result.adversary_score == 0.4
        assert result.disagreement == pytest.approx(0.4, abs=0.01)
        assert result.needs_human_review is True
        assert result.agreement_category == "low"


@pytest.mark.asyncio
async def test_adversarial_evaluation_no_consensus(mock_primary_config, mock_adversary_config):
    """Test adversarial evaluation with consensus round disabled."""
    # Mock OpenAIJudgeClient initialization
    with patch("app.core.adversarial_judge.OpenAIJudgeClient") as MockJudgeClient:
        mock_primary_judge = MagicMock()
        mock_adversary_judge = MagicMock()
        MockJudgeClient.side_effect = [mock_primary_judge, mock_adversary_judge]

        system = AdversarialJudgeSystem(
            primary_llm_config=mock_primary_config,
            adversary_llm_config=mock_adversary_config,
            disagreement_threshold=0.3,
            enable_consensus=False,  # Disable consensus round
        )

    with patch.object(
        system.primary_judge, "judge_json", new_callable=AsyncMock
    ) as mock_primary_initial, patch.object(
        system.adversary_judge, "chat_json", new_callable=AsyncMock
    ) as mock_adversary_critique, patch.object(
        system.primary_judge, "take_row_usage", return_value={"total_tokens": 100}
    ), patch.object(
        system.adversary_judge, "take_row_usage", return_value={"total_tokens": 150}
    ):
        mock_primary_initial.return_value = {"score": 0.8, "reason": "初评理由"}
        mock_adversary_critique.return_value = {
            "critique": "质疑理由",
            "counter_examples": [],
            "suggested_score": 0.6,
            "confidence": 0.8,
            "key_concerns": [],
        }

        sample = {"user_input": "测试"}

        result = await system.evaluate_with_adversary(
            sample=sample, metric_name="test", criteria="测试", score_instruction="0-1"
        )

        # With no consensus, should use weighted average: 0.8 * 0.7 + 0.6 * 0.3 = 0.74
        assert result.final_score == pytest.approx(0.74, abs=0.01)
        assert result.consensus_tokens == 0


def test_agreement_analyzer_basic():
    """Test basic agreement metrics calculation."""
    analyzer = AdversarialAgreementAnalyzer()

    # Add some mock results
    result1 = AdversarialEvaluationResult(
        primary_score=0.8,
        primary_reasoning="理由1",
        adversary_score=0.75,
        adversary_critique="质疑1",
        adversary_counter_examples=[],
        adversary_confidence=0.8,
        adversary_concerns=[],
        final_score=0.78,
        final_reasoning="终评1",
        critique_validity=0.7,
        adjusted_reasoning="调整1",
        disagreement=0.05,
        needs_human_review=False,
        agreement_category="high",
        total_tokens=300,
        primary_tokens=100,
        adversary_tokens=150,
        consensus_tokens=50,
    )

    result2 = AdversarialEvaluationResult(
        primary_score=0.9,
        primary_reasoning="理由2",
        adversary_score=0.5,
        adversary_critique="质疑2",
        adversary_counter_examples=[],
        adversary_confidence=0.85,
        adversary_concerns=[],
        final_score=0.65,
        final_reasoning="终评2",
        critique_validity=0.8,
        adjusted_reasoning="调整2",
        disagreement=0.4,
        needs_human_review=True,
        agreement_category="low",
        total_tokens=300,
        primary_tokens=100,
        adversary_tokens=150,
        consensus_tokens=50,
    )

    analyzer.add_result(result1)
    analyzer.add_result(result2)

    metrics = analyzer.calculate_agreement_metrics()

    assert metrics["total_samples"] == 2
    assert metrics["mean_disagreement"] == pytest.approx(0.225, abs=0.01)  # (0.05 + 0.4) / 2
    assert metrics["high_disagreement_rate"] == 50.0  # 1 out of 2
    assert "cohens_kappa" in metrics
    assert "interpretation" in metrics


def test_agreement_analyzer_perfect_agreement():
    """Test analyzer with perfect agreement."""
    analyzer = AdversarialAgreementAnalyzer()

    # Both judges give same scores
    for i in range(5):
        result = AdversarialEvaluationResult(
            primary_score=0.8,
            primary_reasoning="理由",
            adversary_score=0.8,
            adversary_critique="无质疑",
            adversary_counter_examples=[],
            adversary_confidence=1.0,
            adversary_concerns=[],
            final_score=0.8,
            final_reasoning="终评",
            critique_validity=1.0,
            adjusted_reasoning="无调整",
            disagreement=0.0,
            needs_human_review=False,
            agreement_category="high",
            total_tokens=300,
            primary_tokens=100,
            adversary_tokens=150,
            consensus_tokens=50,
        )
        analyzer.add_result(result)

    metrics = analyzer.calculate_agreement_metrics()

    assert metrics["mean_disagreement"] == 0.0
    assert metrics["high_disagreement_rate"] == 0.0
    assert metrics["percent_agreement"] == 100.0


def test_agreement_analyzer_by_category():
    """Test agreement analysis grouped by category."""
    analyzer = AdversarialAgreementAnalyzer()

    # Add results with different categories
    categories_data = [
        ("high", 0.05),
        ("high", 0.08),
        ("medium", 0.25),
        ("medium", 0.28),
        ("low", 0.35),
        ("low", 0.45),
    ]

    for category, disagreement in categories_data:
        result = AdversarialEvaluationResult(
            primary_score=0.8,
            primary_reasoning="理由",
            adversary_score=0.8 - disagreement,
            adversary_critique="质疑",
            adversary_counter_examples=[],
            adversary_confidence=0.8,
            adversary_concerns=[],
            final_score=0.75,
            final_reasoning="终评",
            critique_validity=0.7,
            adjusted_reasoning="调整",
            disagreement=disagreement,
            needs_human_review=disagreement > 0.3,
            agreement_category=category,
            total_tokens=300,
            primary_tokens=100,
            adversary_tokens=150,
            consensus_tokens=50,
        )
        analyzer.add_result(result)

    category_metrics = analyzer.calculate_agreement_by_category()

    assert "high" in category_metrics
    assert "medium" in category_metrics
    assert "low" in category_metrics

    # High category should have lower mean disagreement
    assert category_metrics["high"]["mean_disagreement"] < category_metrics["low"]["mean_disagreement"]


def test_adversarial_result_to_dict():
    """Test conversion of AdversarialEvaluationResult to dictionary."""
    result = AdversarialEvaluationResult(
        primary_score=0.8,
        primary_reasoning="初评理由",
        adversary_score=0.6,
        adversary_critique="质疑理由",
        adversary_counter_examples=["反例1"],
        adversary_confidence=0.85,
        adversary_concerns=["问题1"],
        final_score=0.7,
        final_reasoning="终评理由",
        critique_validity=0.75,
        adjusted_reasoning="调整说明",
        disagreement=0.2,
        needs_human_review=False,
        agreement_category="medium",
        total_tokens=300,
        primary_tokens=100,
        adversary_tokens=150,
        consensus_tokens=50,
    )

    result_dict = result.to_dict()

    assert result_dict["primary_score"] == 0.8
    assert result_dict["adversary_score"] == 0.6
    assert result_dict["final_score"] == 0.7
    assert result_dict["disagreement"] == 0.2
    assert result_dict["needs_human_review"] is False
    assert result_dict["agreement_category"] == "medium"
    assert result_dict["total_tokens"] == 300
