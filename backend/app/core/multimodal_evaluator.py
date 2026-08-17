"""
多模态评测器
支持图像OCR、图文匹配度、视觉问答等评测任务
"""
import logging
from typing import Optional, List
from difflib import SequenceMatcher
import re

from app.core.multimodal_client import QwenVLClient
from app.schemas.multimodal import (
    MultimodalSample,
    OCREvaluationResult,
    ImageTextMatchResult,
    VisualQAEvaluationResult,
    MultimodalEvaluationResult,
    MultimodalEvaluationSummary,
    MultimodalTaskType,
    ImageInput,
    ImageInputType
)

logger = logging.getLogger(__name__)


class MultimodalEvaluator:
    """
    多模态评测器基类
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "qwen-vl-plus"
    ):
        """
        初始化评测器

        Args:
            api_key: API密钥
            model: 模型名称
        """
        self.client = QwenVLClient(api_key=api_key, model=model)
        self.model = model
        logger.info(f"Initialized MultimodalEvaluator with model: {model}")

    def _get_image_input_value(self, image: ImageInput) -> str:
        """
        获取图像输入值（处理不同类型）

        Args:
            image: 图像输入对象

        Returns:
            可用于API调用的图像输入字符串
        """
        if image.type == ImageInputType.FILE_PATH:
            # 需要转换为Base64
            from app.core.multimodal_client import encode_image_to_base64
            return encode_image_to_base64(image.value)
        else:
            # URL或Base64直接返回
            return image.value


class ImageOCREvaluator(MultimodalEvaluator):
    """
    图像OCR评测器
    评测AI对图像中文字的识别准确率
    """

    def evaluate(self, sample: MultimodalSample) -> OCREvaluationResult:
        """
        评测OCR准确率

        Args:
            sample: 评测样本

        Returns:
            OCR评测结果
        """
        if sample.task_type != MultimodalTaskType.OCR:
            raise ValueError(f"Sample task type must be OCR, got {sample.task_type}")

        if not sample.ground_truth_text:
            raise ValueError("ground_truth_text is required for OCR evaluation")

        # 调用OCR识别
        image_input = self._get_image_input_value(sample.image)
        predicted_text = self.client.ocr(image_input)

        # 计算准确率指标
        metrics = self._calculate_ocr_metrics(
            predicted=predicted_text,
            ground_truth=sample.ground_truth_text
        )

        # 生成评测理由
        reasoning = self._generate_ocr_reasoning(
            predicted=predicted_text,
            ground_truth=sample.ground_truth_text,
            metrics=metrics
        )

        # 估算token使用
        tokens_used = self.client.estimate_image_tokens() + self.client.count_tokens(predicted_text)

        return OCREvaluationResult(
            sample_id=sample.sample_id,
            predicted_text=predicted_text,
            ground_truth_text=sample.ground_truth_text,
            char_accuracy=metrics["char_accuracy"],
            word_accuracy=metrics["word_accuracy"],
            edit_distance=metrics["edit_distance"],
            correct_chars=metrics["correct_chars"],
            total_chars=metrics["total_chars"],
            reasoning=reasoning,
            tokens_used=tokens_used
        )

    def _calculate_ocr_metrics(
        self,
        predicted: str,
        ground_truth: str
    ) -> dict:
        """
        计算OCR指标

        Args:
            predicted: 预测文本
            ground_truth: 标准文本

        Returns:
            指标字典
        """
        # 字符级准确率
        matcher = SequenceMatcher(None, ground_truth, predicted)
        char_accuracy = matcher.ratio()

        # 编辑距离（Levenshtein距离）
        edit_distance = self._levenshtein_distance(ground_truth, predicted)

        # 正确字符数
        correct_chars = int(char_accuracy * len(ground_truth))
        total_chars = len(ground_truth)

        # 词级准确率（按空格或标点分词）
        gt_words = self._tokenize_text(ground_truth)
        pred_words = self._tokenize_text(predicted)
        word_matcher = SequenceMatcher(None, gt_words, pred_words)
        word_accuracy = word_matcher.ratio()

        return {
            "char_accuracy": round(char_accuracy, 4),
            "word_accuracy": round(word_accuracy, 4),
            "edit_distance": edit_distance,
            "correct_chars": correct_chars,
            "total_chars": total_chars
        }

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """
        计算Levenshtein编辑距离

        Args:
            s1: 字符串1
            s2: 字符串2

        Returns:
            编辑距离
        """
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                # 插入、删除、替换的成本
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def _tokenize_text(self, text: str) -> List[str]:
        """
        文本分词（简单版）

        Args:
            text: 输入文本

        Returns:
            词列表
        """
        # 按空格和标点分割
        tokens = re.findall(r'\w+|[^\w\s]', text, re.UNICODE)
        return [t for t in tokens if t.strip()]

    def _generate_ocr_reasoning(
        self,
        predicted: str,
        ground_truth: str,
        metrics: dict
    ) -> str:
        """
        生成OCR评测理由

        Args:
            predicted: 预测文本
            ground_truth: 标准文本
            metrics: 指标字典

        Returns:
            评测理由
        """
        char_acc = metrics["char_accuracy"]
        word_acc = metrics["word_accuracy"]
        edit_dist = metrics["edit_distance"]

        if char_acc >= 0.95:
            quality = "优秀"
        elif char_acc >= 0.85:
            quality = "良好"
        elif char_acc >= 0.70:
            quality = "中等"
        else:
            quality = "较差"

        reasoning = f"OCR识别质量：{quality}。"
        reasoning += f"字符级准确率为{char_acc:.2%}（{metrics['correct_chars']}/{metrics['total_chars']}），"
        reasoning += f"词级准确率为{word_acc:.2%}，"
        reasoning += f"编辑距离为{edit_dist}。"

        # 分析主要错误
        if char_acc < 1.0:
            if len(predicted) < len(ground_truth) * 0.8:
                reasoning += " 存在漏识别问题。"
            elif len(predicted) > len(ground_truth) * 1.2:
                reasoning += " 存在误识别或重复识别问题。"
            else:
                reasoning += " 主要是字符替换错误。"

        return reasoning


class ImageTextMatchEvaluator(MultimodalEvaluator):
    """
    图文匹配度评测器
    评测AI生成的描述与图像内容的一致性
    """

    def evaluate(self, sample: MultimodalSample) -> ImageTextMatchResult:
        """
        评测图文匹配度

        Args:
            sample: 评测样本

        Returns:
            图文匹配评测结果
        """
        if sample.task_type != MultimodalTaskType.IMAGE_TEXT_MATCH:
            raise ValueError(f"Sample task type must be IMAGE_TEXT_MATCH, got {sample.task_type}")

        if not sample.generated_description:
            raise ValueError("generated_description is required for image-text match evaluation")

        # 调用图文匹配度评测
        image_input = self._get_image_input_value(sample.image)
        match_result = self.client.evaluate_image_text_match(
            image_input=image_input,
            text_description=sample.generated_description
        )

        # 获取图像真实内容摘要（用于分析）
        true_content = self.client.describe_image(image_input, detailed=False)

        # 估算token使用
        tokens_used = (
            self.client.estimate_image_tokens() * 2 +  # 两次图像理解调用
            self.client.count_tokens(sample.generated_description) +
            self.client.count_tokens(match_result.get("reasoning", ""))
        )

        return ImageTextMatchResult(
            sample_id=sample.sample_id,
            generated_description=sample.generated_description,
            content_consistency=match_result["content_consistency"],
            detail_accuracy=match_result["detail_accuracy"],
            completeness=match_result["completeness"],
            overall_score=match_result["overall_score"],
            reasoning=match_result["reasoning"],
            true_content_summary=true_content,
            tokens_used=tokens_used
        )


class VisualQAEvaluator(MultimodalEvaluator):
    """
    视觉问答评测器
    评测AI对图像内容的理解和推理能力
    """

    def evaluate(self, sample: MultimodalSample) -> VisualQAEvaluationResult:
        """
        评测视觉问答准确性

        Args:
            sample: 评测样本

        Returns:
            视觉问答评测结果
        """
        if sample.task_type != MultimodalTaskType.VISUAL_QA:
            raise ValueError(f"Sample task type must be VISUAL_QA, got {sample.task_type}")

        if not sample.question or not sample.ground_truth_answer or not sample.predicted_answer:
            raise ValueError("question, ground_truth_answer, and predicted_answer are required for VQA evaluation")

        # 获取参考答案（模型独立回答）
        image_input = self._get_image_input_value(sample.image)
        reference_answer = self.client.answer_question(
            image_input=image_input,
            question=sample.question
        )

        # 判断答案正确性
        is_correct, score, error_type, reasoning = self._evaluate_answer(
            predicted=sample.predicted_answer,
            ground_truth=sample.ground_truth_answer,
            reference=reference_answer,
            question=sample.question
        )

        # 估算token使用
        tokens_used = (
            self.client.estimate_image_tokens() +
            self.client.count_tokens(sample.question) +
            self.client.count_tokens(reference_answer)
        )

        return VisualQAEvaluationResult(
            sample_id=sample.sample_id,
            question=sample.question,
            predicted_answer=sample.predicted_answer,
            ground_truth_answer=sample.ground_truth_answer,
            is_correct=is_correct,
            score=score,
            reference_answer=reference_answer,
            error_type=error_type,
            reasoning=reasoning,
            tokens_used=tokens_used
        )

    def _evaluate_answer(
        self,
        predicted: str,
        ground_truth: str,
        reference: str,
        question: str
    ) -> tuple:
        """
        评估答案正确性

        Args:
            predicted: 预测答案
            ground_truth: 标准答案
            reference: 参考答案（模型给出）
            question: 问题

        Returns:
            (is_correct, score, error_type, reasoning)
        """
        # 简化处理：转小写、去空格
        pred_normalized = predicted.lower().strip()
        gt_normalized = ground_truth.lower().strip()
        ref_normalized = reference.lower().strip()

        # 精确匹配
        if pred_normalized == gt_normalized:
            return True, 1.0, None, f"答案完全正确：'{predicted}' 与标准答案一致。"

        # 包含关系
        if gt_normalized in pred_normalized or pred_normalized in gt_normalized:
            return True, 0.9, None, f"答案基本正确：'{predicted}' 与标准答案'{ground_truth}'语义一致。"

        # 与参考答案对比
        if pred_normalized == ref_normalized:
            return True, 0.85, None, f"答案正确：与模型参考答案一致（'{reference}'），虽然表述与标准答案略有不同。"

        if ref_normalized in pred_normalized or pred_normalized in ref_normalized:
            return True, 0.8, None, f"答案基本正确：与模型参考答案语义相近。"

        # 数字答案特殊处理
        pred_numbers = re.findall(r'\d+', predicted)
        gt_numbers = re.findall(r'\d+', ground_truth)
        if pred_numbers and gt_numbers:
            if pred_numbers == gt_numbers:
                return True, 0.9, None, f"答案正确：关键数字'{pred_numbers[0]}'与标准答案一致。"
            else:
                error_type = "counting_error" if "多少" in question or "几" in question else "factual_error"
                return False, 0.0, error_type, f"答案错误：预测数字为{pred_numbers}，标准答案为{gt_numbers}。"

        # 相似度检查
        similarity = SequenceMatcher(None, pred_normalized, gt_normalized).ratio()
        if similarity >= 0.7:
            return True, round(similarity, 2), None, f"答案部分正确：与标准答案相似度为{similarity:.2%}。"

        # 答案错误
        error_type = self._classify_error_type(predicted, ground_truth, reference, question)
        reasoning = f"答案错误：预测为'{predicted}'，标准答案为'{ground_truth}'，模型参考答案为'{reference}'。"

        return False, 0.0, error_type, reasoning

    def _classify_error_type(
        self,
        predicted: str,
        ground_truth: str,
        reference: str,
        question: str
    ) -> str:
        """
        分类错误类型

        Args:
            predicted: 预测答案
            ground_truth: 标准答案
            reference: 参考答案
            question: 问题

        Returns:
            错误类型
        """
        # 简单的规则分类
        if "多少" in question or "几" in question or "数" in question:
            return "counting_error"
        elif "颜色" in question or "什么色" in question:
            return "attribute_error"
        elif "位置" in question or "哪里" in question or "在哪" in question:
            return "location_error"
        elif "是否" in question or "有没有" in question:
            return "judgment_error"
        else:
            return "understanding_error"


class MultimodalEvaluationEngine:
    """
    多模态评测引擎
    统一管理各类多模态评测任务
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "qwen-vl-plus"
    ):
        """
        初始化评测引擎

        Args:
            api_key: API密钥
            model: 模型名称
        """
        self.ocr_evaluator = ImageOCREvaluator(api_key=api_key, model=model)
        self.match_evaluator = ImageTextMatchEvaluator(api_key=api_key, model=model)
        self.qa_evaluator = VisualQAEvaluator(api_key=api_key, model=model)

        logger.info("Initialized MultimodalEvaluationEngine")

    def evaluate(self, sample: MultimodalSample) -> MultimodalEvaluationResult:
        """
        评测单个样本

        Args:
            sample: 评测样本

        Returns:
            统一格式的评测结果
        """
        if sample.task_type == MultimodalTaskType.OCR:
            ocr_result = self.ocr_evaluator.evaluate(sample)
            return MultimodalEvaluationResult(
                sample_id=sample.sample_id,
                task_type=sample.task_type,
                score=ocr_result.char_accuracy * 5,  # 转换为1-5分
                reasoning=ocr_result.reasoning,
                ocr_result=ocr_result,
                tokens_used=ocr_result.tokens_used or 0
            )

        elif sample.task_type == MultimodalTaskType.IMAGE_TEXT_MATCH:
            match_result = self.match_evaluator.evaluate(sample)
            return MultimodalEvaluationResult(
                sample_id=sample.sample_id,
                task_type=sample.task_type,
                score=match_result.overall_score,
                reasoning=match_result.reasoning,
                match_result=match_result,
                tokens_used=match_result.tokens_used or 0
            )

        elif sample.task_type == MultimodalTaskType.VISUAL_QA:
            qa_result = self.qa_evaluator.evaluate(sample)
            return MultimodalEvaluationResult(
                sample_id=sample.sample_id,
                task_type=sample.task_type,
                score=qa_result.score * 5,  # 转换为1-5分
                reasoning=qa_result.reasoning,
                qa_result=qa_result,
                tokens_used=qa_result.tokens_used or 0
            )

        else:
            raise ValueError(f"Unsupported task type: {sample.task_type}")

    def evaluate_batch(self, samples: List[MultimodalSample]) -> MultimodalEvaluationSummary:
        """
        批量评测

        Args:
            samples: 评测样本列表

        Returns:
            评测汇总结果
        """
        if not samples:
            raise ValueError("Samples list is empty")

        # 检查任务类型一致性
        task_type = samples[0].task_type
        if not all(s.task_type == task_type for s in samples):
            raise ValueError("All samples must have the same task type")

        # 逐个评测
        results = []
        for sample in samples:
            try:
                result = self.evaluate(sample)
                results.append(result)
            except Exception as e:
                logger.error(f"Error evaluating sample {sample.sample_id}: {str(e)}")
                # 继续处理其他样本

        if not results:
            raise ValueError("No samples were successfully evaluated")

        # 计算汇总统计
        scores = [r.score for r in results]
        total_tokens = sum(r.tokens_used for r in results)

        summary = MultimodalEvaluationSummary(
            total_samples=len(results),
            task_type=task_type,
            average_score=sum(scores) / len(scores),
            min_score=min(scores),
            max_score=max(scores),
            total_tokens_used=total_tokens,
            average_tokens_per_sample=total_tokens / len(results),
            results=results
        )

        # 任务特定指标
        if task_type == MultimodalTaskType.OCR:
            ocr_results = [r.ocr_result for r in results if r.ocr_result]
            if ocr_results:
                summary.average_char_accuracy = sum(r.char_accuracy for r in ocr_results) / len(ocr_results)
                summary.average_word_accuracy = sum(r.word_accuracy for r in ocr_results) / len(ocr_results)

        elif task_type == MultimodalTaskType.VISUAL_QA:
            qa_results = [r.qa_result for r in results if r.qa_result]
            if qa_results:
                correct_count = sum(1 for r in qa_results if r.is_correct)
                summary.correct_count = correct_count
                summary.accuracy = correct_count / len(qa_results)

        return summary
