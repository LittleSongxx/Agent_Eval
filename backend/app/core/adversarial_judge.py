"""
Adversarial Judge System for Enhanced Evaluation Reliability.

This module implements a multi-judge adversarial validation approach to reduce
LLM judge bias and improve evaluation reliability. The system uses two judges:
- Primary Judge: Makes initial scoring decision
- Adversary Judge: Critiques the primary decision and proposes alternatives

When disagreement exceeds threshold, samples are flagged for human review.
"""

from __future__ import annotations

import asyncio
import json
import logging
import typing as t
from dataclasses import dataclass

from app.core.evaluation_engine import OpenAIJudgeClient

logger = logging.getLogger(__name__)


# System prompt for the adversary judge (critique mode)
ADVERSARY_CRITIQUE_SYSTEM_PROMPT = """你是一个批判性的评审专家，任务是质疑另一个评委的初步评分。

你需要：
1. 仔细审查初步评分是否合理
2. 找出可能的误判、遗漏或过于主观的判断
3. 提出反例或相反观点
4. 给出你建议的分数和置信度

返回严格JSON格式：
{
    "critique": "你的质疑理由，指出初评可能存在的问题",
    "counter_examples": ["具体反例1", "具体反例2"],
    "suggested_score": 0.6,
    "confidence": 0.8,
    "key_concerns": ["关键问题1", "关键问题2"]
}

注意：
- critique必须具体指出初评的问题，不要泛泛而谈
- counter_examples要从样本中找到具体证据
- suggested_score是你认为更合理的分数（0-1之间）
- confidence是你对自己判断的置信度（0-1之间）
"""


# System prompt for final consensus (after receiving critique)
CONSENSUS_SYSTEM_PROMPT = """你是一个公正的评审专家，需要基于对抗性质疑重新审视你的初步评分。

你收到了另一位评委的质疑意见，现在需要：
1. 认真考虑质疑中的合理部分
2. 重新审视你的初评是否存在偏差
3. 给出更客观的终评分数和理由

返回严格JSON格式：
{
    "final_score": 0.7,
    "final_reasoning": "综合考虑质疑后的最终判断理由",
    "critique_validity": 0.6,
    "adjusted_reasoning": "说明你如何调整了初评"
}

注意：
- 如果质疑有道理，应当调整分数
- 如果质疑无道理，维持原分数并说明理由
- final_reasoning要综合初评和质疑的合理部分
- critique_validity表示你认为质疑的合理程度（0-1）
"""


@dataclass
class AdversarialEvaluationResult:
    """Result of adversarial evaluation with multiple judge perspectives."""

    # Primary judge results
    primary_score: float
    primary_reasoning: str

    # Adversary judge results
    adversary_score: float
    adversary_critique: str
    adversary_counter_examples: list[str]
    adversary_confidence: float
    adversary_concerns: list[str]

    # Final consensus results
    final_score: float
    final_reasoning: str
    critique_validity: float
    adjusted_reasoning: str

    # Disagreement metrics
    disagreement: float  # abs(primary_score - adversary_score)
    needs_human_review: bool
    agreement_category: str  # "high", "medium", "low"

    # Token usage tracking
    total_tokens: int
    primary_tokens: int
    adversary_tokens: int
    consensus_tokens: int

    def to_dict(self) -> dict[str, t.Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "primary_score": self.primary_score,
            "primary_reasoning": self.primary_reasoning,
            "adversary_score": self.adversary_score,
            "adversary_critique": self.adversary_critique,
            "adversary_counter_examples": self.adversary_counter_examples,
            "adversary_confidence": self.adversary_confidence,
            "adversary_concerns": self.adversary_concerns,
            "final_score": self.final_score,
            "final_reasoning": self.final_reasoning,
            "critique_validity": self.critique_validity,
            "adjusted_reasoning": self.adjusted_reasoning,
            "disagreement": self.disagreement,
            "needs_human_review": self.needs_human_review,
            "agreement_category": self.agreement_category,
            "total_tokens": self.total_tokens,
            "primary_tokens": self.primary_tokens,
            "adversary_tokens": self.adversary_tokens,
            "consensus_tokens": self.consensus_tokens,
        }


class AdversarialJudgeSystem:
    """
    Adversarial validation system using multiple LLM judges.

    Workflow:
    1. Primary judge makes initial scoring decision
    2. Adversary judge critiques the primary decision
    3. Primary judge reconsiders based on critique and makes final decision
    4. System calculates disagreement and flags samples for human review

    Args:
        primary_llm_config: Configuration for the primary judge (e.g., qwen-plus)
        adversary_llm_config: Configuration for the adversary judge (e.g., gpt-4o)
        disagreement_threshold: Threshold for flagging human review (default 0.3)
        enable_consensus: Whether to enable final consensus round (default True)
    """

    def __init__(
        self,
        primary_llm_config,
        adversary_llm_config,
        disagreement_threshold: float = 0.3,
        enable_consensus: bool = True,
    ):
        self.primary_judge = OpenAIJudgeClient(primary_llm_config)
        self.adversary_judge = OpenAIJudgeClient(adversary_llm_config)
        self.disagreement_threshold = disagreement_threshold
        self.enable_consensus = enable_consensus

        logger.info(
            "AdversarialJudgeSystem initialized: primary=%s, adversary=%s, threshold=%.2f",
            primary_llm_config.model_name,
            adversary_llm_config.model_name,
            disagreement_threshold,
        )

    async def evaluate_with_adversary(
        self,
        sample: dict[str, t.Any],
        metric_name: str,
        criteria: str,
        score_instruction: str,
    ) -> AdversarialEvaluationResult:
        """
        Execute adversarial evaluation with three rounds:
        1. Primary judge initial scoring
        2. Adversary judge critique
        3. Primary judge final consensus (if enabled)

        Args:
            sample: The evaluation sample data
            metric_name: Name of the metric being evaluated
            criteria: Evaluation criteria/rubric
            score_instruction: Instruction for scoring format

        Returns:
            AdversarialEvaluationResult with all judge perspectives and disagreement metrics
        """
        logger.info("Starting adversarial evaluation for metric: %s", metric_name)

        # Round 1: Primary judge initial scoring
        primary_result = await self._primary_judge_initial(sample, metric_name, criteria, score_instruction)
        primary_score = primary_result["score"]
        primary_reasoning = primary_result["reason"]
        primary_tokens = self.primary_judge.take_row_usage()["total_tokens"]

        logger.info("Primary judge score: %.2f, reasoning: %s", primary_score, primary_reasoning[:100])

        # Round 2: Adversary judge critique
        adversary_result = await self._adversary_judge_critique(
            sample, metric_name, criteria, primary_score, primary_reasoning
        )
        adversary_score = adversary_result.get("suggested_score", primary_score)
        adversary_critique = adversary_result.get("critique", "")
        adversary_counter_examples = adversary_result.get("counter_examples", [])
        adversary_confidence = adversary_result.get("confidence", 0.5)
        adversary_concerns = adversary_result.get("key_concerns", [])
        adversary_tokens = self.adversary_judge.take_row_usage()["total_tokens"]

        logger.info(
            "Adversary judge score: %.2f, confidence: %.2f, critique: %s",
            adversary_score,
            adversary_confidence,
            adversary_critique[:100],
        )

        # Calculate disagreement
        disagreement = abs(primary_score - adversary_score)
        needs_human_review = disagreement > self.disagreement_threshold

        if disagreement < 0.2:
            agreement_category = "high"
        elif disagreement < 0.3:
            agreement_category = "medium"
        else:
            agreement_category = "low"

        logger.info(
            "Disagreement: %.2f, category: %s, needs_human_review: %s",
            disagreement,
            agreement_category,
            needs_human_review,
        )

        # Round 3: Final consensus (if enabled)
        if self.enable_consensus:
            consensus_result = await self._primary_judge_consensus(
                sample, metric_name, criteria, primary_score, primary_reasoning, adversary_result
            )
            final_score = consensus_result.get("final_score", primary_score)
            final_reasoning = consensus_result.get("final_reasoning", primary_reasoning)
            critique_validity = consensus_result.get("critique_validity", 0.5)
            adjusted_reasoning = consensus_result.get("adjusted_reasoning", "")
            consensus_tokens = self.primary_judge.take_row_usage()["total_tokens"]

            logger.info("Final consensus score: %.2f, critique_validity: %.2f", final_score, critique_validity)
        else:
            # No consensus round: use weighted average based on confidence
            weight_primary = 0.7  # Primary judge has higher weight by default
            weight_adversary = 0.3
            final_score = primary_score * weight_primary + adversary_score * weight_adversary
            final_reasoning = f"综合评分（主评分 {primary_score:.2f} × 0.7 + 对抗评分 {adversary_score:.2f} × 0.3）"
            critique_validity = adversary_confidence
            adjusted_reasoning = "未启用共识轮，使用加权平均"
            consensus_tokens = 0

        total_tokens = primary_tokens + adversary_tokens + consensus_tokens

        return AdversarialEvaluationResult(
            primary_score=primary_score,
            primary_reasoning=primary_reasoning,
            adversary_score=adversary_score,
            adversary_critique=adversary_critique,
            adversary_counter_examples=adversary_counter_examples,
            adversary_confidence=adversary_confidence,
            adversary_concerns=adversary_concerns,
            final_score=final_score,
            final_reasoning=final_reasoning,
            critique_validity=critique_validity,
            adjusted_reasoning=adjusted_reasoning,
            disagreement=disagreement,
            needs_human_review=needs_human_review,
            agreement_category=agreement_category,
            total_tokens=total_tokens,
            primary_tokens=primary_tokens,
            adversary_tokens=adversary_tokens,
            consensus_tokens=consensus_tokens,
        )

    async def _primary_judge_initial(
        self,
        sample: dict[str, t.Any],
        metric_name: str,
        criteria: str,
        score_instruction: str,
    ) -> dict[str, t.Any]:
        """Round 1: Primary judge makes initial scoring decision."""
        payload = {
            "metric": metric_name,
            "criteria": criteria,
            "instruction": score_instruction,
            "sample": sample,
        }
        return await self.primary_judge.judge_json(payload)

    async def _adversary_judge_critique(
        self,
        sample: dict[str, t.Any],
        metric_name: str,
        criteria: str,
        primary_score: float,
        primary_reasoning: str,
    ) -> dict[str, t.Any]:
        """Round 2: Adversary judge critiques the primary decision."""
        user_prompt = json.dumps(
            {
                "metric": metric_name,
                "criteria": criteria,
                "sample": sample,
                "primary_evaluation": {
                    "score": primary_score,
                    "reasoning": primary_reasoning,
                },
                "task": "请质疑上述primary_evaluation，找出可能的误判或遗漏",
            },
            ensure_ascii=False,
            indent=2,
        )

        return await self.adversary_judge.chat_json(ADVERSARY_CRITIQUE_SYSTEM_PROMPT, user_prompt)

    async def _primary_judge_consensus(
        self,
        sample: dict[str, t.Any],
        metric_name: str,
        criteria: str,
        primary_score: float,
        primary_reasoning: str,
        adversary_result: dict[str, t.Any],
    ) -> dict[str, t.Any]:
        """Round 3: Primary judge reconsiders based on adversary critique."""
        user_prompt = json.dumps(
            {
                "metric": metric_name,
                "criteria": criteria,
                "sample": sample,
                "your_initial_evaluation": {
                    "score": primary_score,
                    "reasoning": primary_reasoning,
                },
                "adversary_critique": {
                    "suggested_score": adversary_result.get("suggested_score"),
                    "critique": adversary_result.get("critique"),
                    "counter_examples": adversary_result.get("counter_examples", []),
                    "key_concerns": adversary_result.get("key_concerns", []),
                },
                "task": "请基于adversary_critique重新审视你的初评，给出最终评分",
            },
            ensure_ascii=False,
            indent=2,
        )

        return await self.primary_judge.chat_json(CONSENSUS_SYSTEM_PROMPT, user_prompt)


class AdversarialAgreementAnalyzer:
    """
    Analyzer for measuring inter-judge agreement in adversarial evaluation.

    Calculates Cohen's Kappa and other agreement metrics to quantify
    reliability of the evaluation system.
    """

    def __init__(self):
        self.results: list[AdversarialEvaluationResult] = []

    def add_result(self, result: AdversarialEvaluationResult) -> None:
        """Add a result to the analyzer."""
        self.results.append(result)

    def calculate_agreement_metrics(self) -> dict[str, t.Any]:
        """
        Calculate inter-judge agreement metrics.

        Returns:
            Dictionary containing:
            - cohens_kappa: Cohen's Kappa coefficient
            - percent_agreement: Percentage of exact agreement
            - mean_disagreement: Average disagreement magnitude
            - high_disagreement_rate: Percentage needing human review
        """
        if not self.results:
            return {
                "cohens_kappa": 0.0,
                "percent_agreement": 0.0,
                "mean_disagreement": 0.0,
                "high_disagreement_rate": 0.0,
            }

        # Calculate basic agreement metrics
        disagreements = [r.disagreement for r in self.results]
        mean_disagreement = sum(disagreements) / len(disagreements)

        high_disagreement_count = sum(1 for r in self.results if r.needs_human_review)
        high_disagreement_rate = high_disagreement_count / len(self.results)

        # Calculate Cohen's Kappa (simplified binary classification)
        # Convert scores to binary: pass (>= 0.7) or fail (< 0.7)
        threshold = 0.7
        primary_binary = [1 if r.primary_score >= threshold else 0 for r in self.results]
        adversary_binary = [1 if r.adversary_score >= threshold else 0 for r in self.results]

        # Calculate observed agreement
        agreements = sum(1 for p, a in zip(primary_binary, adversary_binary) if p == a)
        p_o = agreements / len(self.results)

        # Calculate expected agreement by chance
        primary_pass_rate = sum(primary_binary) / len(primary_binary)
        adversary_pass_rate = sum(adversary_binary) / len(adversary_binary)
        p_e = primary_pass_rate * adversary_pass_rate + (1 - primary_pass_rate) * (1 - adversary_pass_rate)

        # Cohen's Kappa
        cohens_kappa = (p_o - p_e) / (1 - p_e) if p_e < 1 else 1.0

        return {
            "cohens_kappa": round(cohens_kappa, 3),
            "percent_agreement": round(p_o * 100, 2),
            "mean_disagreement": round(mean_disagreement, 3),
            "high_disagreement_rate": round(high_disagreement_rate * 100, 2),
            "total_samples": len(self.results),
            "interpretation": self._interpret_kappa(cohens_kappa),
        }

    def calculate_agreement_by_category(self) -> dict[str, dict[str, t.Any]]:
        """
        Calculate agreement metrics grouped by agreement category.

        Returns:
            Dictionary mapping agreement_category to metrics
        """
        categories = {}
        for category in ["high", "medium", "low"]:
            category_results = [r for r in self.results if r.agreement_category == category]
            if category_results:
                analyzer = AdversarialAgreementAnalyzer()
                analyzer.results = category_results
                categories[category] = analyzer.calculate_agreement_metrics()
        return categories

    @staticmethod
    def _interpret_kappa(kappa: float) -> str:
        """Interpret Cohen's Kappa value."""
        if kappa < 0:
            return "差（比随机还差）"
        elif kappa < 0.2:
            return "轻微一致"
        elif kappa < 0.4:
            return "一般一致"
        elif kappa < 0.6:
            return "中等一致"
        elif kappa < 0.8:
            return "较高一致"
        else:
            return "几乎完全一致"
