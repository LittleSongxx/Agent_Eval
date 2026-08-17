"""
多模态评测器测试
"""
import pytest
from unittest.mock import Mock, patch, MagicMock

from app.core.multimodal_evaluator import (
    ImageOCREvaluator,
    ImageTextMatchEvaluator,
    VisualQAEvaluator,
    MultimodalEvaluationEngine
)
from app.schemas.multimodal import (
    MultimodalSample,
    MultimodalTaskType,
    ImageInput,
    ImageInputType
)


class TestImageOCREvaluator:
    """测试OCR评测器"""

    @patch("app.core.multimodal_evaluator.QwenVLClient")
    def test_evaluate_ocr_perfect_match(self, mock_client_class):
        """测试完美匹配的OCR评测"""
        # Mock客户端
        mock_client = Mock()
        mock_client.ocr.return_value = "发票号码: 12345678"
        mock_client.estimate_image_tokens.return_value = 1500
        mock_client.count_tokens.return_value = 10
        mock_client_class.return_value = mock_client

        # 创建评测器
        evaluator = ImageOCREvaluator(api_key="test-key")

        # 创建样本
        sample = MultimodalSample(
            sample_id="test-001",
            task_type=MultimodalTaskType.OCR,
            image=ImageInput(type=ImageInputType.URL, value="https://example.com/invoice.jpg"),
            ground_truth_text="发票号码: 12345678"
        )

        # 评测
        result = evaluator.evaluate(sample)

        assert result.sample_id == "test-001"
        assert result.char_accuracy == 1.0
        assert result.word_accuracy == 1.0
        assert result.edit_distance == 0
        assert result.correct_chars == len("发票号码: 12345678")
        assert "优秀" in result.reasoning

    @patch("app.core.multimodal_evaluator.QwenVLClient")
    def test_evaluate_ocr_partial_match(self, mock_client_class):
        """测试部分匹配的OCR评测"""
        mock_client = Mock()
        mock_client.ocr.return_value = "发票号码: 12345679"  # 最后一位错误
        mock_client.estimate_image_tokens.return_value = 1500
        mock_client.count_tokens.return_value = 10
        mock_client_class.return_value = mock_client

        evaluator = ImageOCREvaluator(api_key="test-key")

        sample = MultimodalSample(
            sample_id="test-002",
            task_type=MultimodalTaskType.OCR,
            image=ImageInput(type=ImageInputType.URL, value="https://example.com/invoice.jpg"),
            ground_truth_text="发票号码: 12345678"
        )

        result = evaluator.evaluate(sample)

        assert result.sample_id == "test-002"
        assert result.char_accuracy < 1.0
        assert result.char_accuracy > 0.9  # 只有一个字符错误
        assert result.edit_distance == 1
        assert "优秀" in result.reasoning or "良好" in result.reasoning

    @patch("app.core.multimodal_evaluator.QwenVLClient")
    def test_evaluate_ocr_poor_match(self, mock_client_class):
        """测试差匹配的OCR评测"""
        mock_client = Mock()
        mock_client.ocr.return_value = "完全不同的文本"
        mock_client.estimate_image_tokens.return_value = 1500
        mock_client.count_tokens.return_value = 10
        mock_client_class.return_value = mock_client

        evaluator = ImageOCREvaluator(api_key="test-key")

        sample = MultimodalSample(
            sample_id="test-003",
            task_type=MultimodalTaskType.OCR,
            image=ImageInput(type=ImageInputType.URL, value="https://example.com/invoice.jpg"),
            ground_truth_text="发票号码: 12345678"
        )

        result = evaluator.evaluate(sample)

        assert result.sample_id == "test-003"
        assert result.char_accuracy < 0.5
        assert "较差" in result.reasoning or "中等" in result.reasoning

    @patch("app.core.multimodal_evaluator.QwenVLClient")
    def test_evaluate_ocr_missing_text(self, mock_client_class):
        """测试漏识别的OCR评测"""
        mock_client = Mock()
        mock_client.ocr.return_value = "发票号码"  # 漏掉数字部分
        mock_client.estimate_image_tokens.return_value = 1500
        mock_client.count_tokens.return_value = 10
        mock_client_class.return_value = mock_client

        evaluator = ImageOCREvaluator(api_key="test-key")

        sample = MultimodalSample(
            sample_id="test-004",
            task_type=MultimodalTaskType.OCR,
            image=ImageInput(type=ImageInputType.URL, value="https://example.com/invoice.jpg"),
            ground_truth_text="发票号码: 12345678"
        )

        result = evaluator.evaluate(sample)

        assert result.sample_id == "test-004"
        assert "漏识别" in result.reasoning

    @patch("app.core.multimodal_evaluator.QwenVLClient")
    def test_evaluate_ocr_invalid_task_type(self, mock_client_class):
        """测试错误的任务类型"""
        evaluator = ImageOCREvaluator(api_key="test-key")

        sample = MultimodalSample(
            sample_id="test-005",
            task_type=MultimodalTaskType.VISUAL_QA,  # 错误的类型
            image=ImageInput(type=ImageInputType.URL, value="https://example.com/test.jpg"),
            question="测试",
            ground_truth_answer="答案",
            predicted_answer="预测"
        )

        with pytest.raises(ValueError, match="Sample task type must be OCR"):
            evaluator.evaluate(sample)

    @patch("app.core.multimodal_evaluator.QwenVLClient")
    def test_evaluate_ocr_missing_ground_truth(self, mock_client_class):
        """测试缺少标准答案"""
        evaluator = ImageOCREvaluator(api_key="test-key")

        sample = MultimodalSample(
            sample_id="test-006",
            task_type=MultimodalTaskType.OCR,
            image=ImageInput(type=ImageInputType.URL, value="https://example.com/test.jpg")
            # 缺少ground_truth_text
        )

        with pytest.raises(ValueError, match="ground_truth_text is required"):
            evaluator.evaluate(sample)

    def test_levenshtein_distance(self):
        """测试编辑距离计算"""
        with patch("app.core.multimodal_evaluator.QwenVLClient"):
            evaluator = ImageOCREvaluator(api_key="test-key")

            # 相同字符串
            assert evaluator._levenshtein_distance("hello", "hello") == 0

            # 一个字符差异
            assert evaluator._levenshtein_distance("hello", "hallo") == 1

            # 插入操作
            assert evaluator._levenshtein_distance("hello", "helllo") == 1

            # 删除操作
            assert evaluator._levenshtein_distance("hello", "helo") == 1

            # 完全不同
            assert evaluator._levenshtein_distance("abc", "xyz") == 3

    def test_tokenize_text(self):
        """测试文本分词"""
        with patch("app.core.multimodal_evaluator.QwenVLClient"):
            evaluator = ImageOCREvaluator(api_key="test-key")

            # 中文分词
            tokens = evaluator._tokenize_text("发票号码: 12345678")
            assert "发票号码" in tokens
            assert "12345678" in tokens

            # 英文分词
            tokens = evaluator._tokenize_text("Invoice No: 12345")
            assert "Invoice" in tokens
            assert "No" in tokens
            assert "12345" in tokens


class TestImageTextMatchEvaluator:
    """测试图文匹配度评测器"""

    @patch("app.core.multimodal_evaluator.QwenVLClient")
    def test_evaluate_match_high_score(self, mock_client_class):
        """测试高分匹配"""
        mock_client = Mock()
        mock_client.evaluate_image_text_match.return_value = {
            "content_consistency": 5,
            "detail_accuracy": 4,
            "completeness": 5,
            "overall_score": 4.7,
            "reasoning": "描述非常准确，细节完整"
        }
        mock_client.describe_image.return_value = "一只橘色的猫坐在窗台上"
        mock_client.estimate_image_tokens.return_value = 1500
        mock_client.count_tokens.return_value = 20
        mock_client_class.return_value = mock_client

        evaluator = ImageTextMatchEvaluator(api_key="test-key")

        sample = MultimodalSample(
            sample_id="test-101",
            task_type=MultimodalTaskType.IMAGE_TEXT_MATCH,
            image=ImageInput(type=ImageInputType.URL, value="https://example.com/cat.jpg"),
            generated_description="一只橘色的猫坐在窗台上，阳光洒在它的身上"
        )

        result = evaluator.evaluate(sample)

        assert result.sample_id == "test-101"
        assert result.content_consistency == 5
        assert result.detail_accuracy == 4
        assert result.completeness == 5
        assert result.overall_score == 4.7
        assert "准确" in result.reasoning

    @patch("app.core.multimodal_evaluator.QwenVLClient")
    def test_evaluate_match_low_score(self, mock_client_class):
        """测试低分匹配"""
        mock_client = Mock()
        mock_client.evaluate_image_text_match.return_value = {
            "content_consistency": 2,
            "detail_accuracy": 1,
            "completeness": 2,
            "overall_score": 1.7,
            "reasoning": "描述与图像内容不符，存在幻觉"
        }
        mock_client.describe_image.return_value = "一只猫"
        mock_client.estimate_image_tokens.return_value = 1500
        mock_client.count_tokens.return_value = 20
        mock_client_class.return_value = mock_client

        evaluator = ImageTextMatchEvaluator(api_key="test-key")

        sample = MultimodalSample(
            sample_id="test-102",
            task_type=MultimodalTaskType.IMAGE_TEXT_MATCH,
            image=ImageInput(type=ImageInputType.URL, value="https://example.com/cat.jpg"),
            generated_description="一只大象站在草原上"  # 完全错误
        )

        result = evaluator.evaluate(sample)

        assert result.sample_id == "test-102"
        assert result.overall_score < 3.0
        assert "幻觉" in result.reasoning or "不符" in result.reasoning

    @patch("app.core.multimodal_evaluator.QwenVLClient")
    def test_evaluate_match_invalid_task_type(self, mock_client_class):
        """测试错误的任务类型"""
        evaluator = ImageTextMatchEvaluator(api_key="test-key")

        sample = MultimodalSample(
            sample_id="test-103",
            task_type=MultimodalTaskType.OCR,
            image=ImageInput(type=ImageInputType.URL, value="https://example.com/test.jpg"),
            ground_truth_text="测试"
        )

        with pytest.raises(ValueError, match="Sample task type must be IMAGE_TEXT_MATCH"):
            evaluator.evaluate(sample)

    @patch("app.core.multimodal_evaluator.QwenVLClient")
    def test_evaluate_match_missing_description(self, mock_client_class):
        """测试缺少描述"""
        evaluator = ImageTextMatchEvaluator(api_key="test-key")

        sample = MultimodalSample(
            sample_id="test-104",
            task_type=MultimodalTaskType.IMAGE_TEXT_MATCH,
            image=ImageInput(type=ImageInputType.URL, value="https://example.com/test.jpg")
            # 缺少generated_description
        )

        with pytest.raises(ValueError, match="generated_description is required"):
            evaluator.evaluate(sample)


class TestVisualQAEvaluator:
    """测试视觉问答评测器"""

    @patch("app.core.multimodal_evaluator.QwenVLClient")
    def test_evaluate_qa_correct_exact_match(self, mock_client_class):
        """测试完全正确的答案"""
        mock_client = Mock()
        mock_client.answer_question.return_value = "3辆"
        mock_client.estimate_image_tokens.return_value = 1500
        mock_client.count_tokens.return_value = 10
        mock_client_class.return_value = mock_client

        evaluator = VisualQAEvaluator(api_key="test-key")

        sample = MultimodalSample(
            sample_id="test-201",
            task_type=MultimodalTaskType.VISUAL_QA,
            image=ImageInput(type=ImageInputType.URL, value="https://example.com/traffic.jpg"),
            question="图中有几辆红色汽车？",
            ground_truth_answer="3辆",
            predicted_answer="3辆"
        )

        result = evaluator.evaluate(sample)

        assert result.sample_id == "test-201"
        assert result.is_correct is True
        assert result.score == 1.0
        assert result.error_type is None
        assert "完全正确" in result.reasoning

    @patch("app.core.multimodal_evaluator.QwenVLClient")
    def test_evaluate_qa_correct_semantic_match(self, mock_client_class):
        """测试语义匹配的答案"""
        mock_client = Mock()
        mock_client.answer_question.return_value = "有三辆红色的车"
        mock_client.estimate_image_tokens.return_value = 1500
        mock_client.count_tokens.return_value = 10
        mock_client_class.return_value = mock_client

        evaluator = VisualQAEvaluator(api_key="test-key")

        sample = MultimodalSample(
            sample_id="test-202",
            task_type=MultimodalTaskType.VISUAL_QA,
            image=ImageInput(type=ImageInputType.URL, value="https://example.com/traffic.jpg"),
            question="图中有几辆红色汽车？",
            ground_truth_answer="3辆",
            predicted_answer="有三辆红色的车"
        )

        result = evaluator.evaluate(sample)

        assert result.sample_id == "test-202"
        assert result.is_correct is True
        assert result.score >= 0.8  # 高分但不是满分

    @patch("app.core.multimodal_evaluator.QwenVLClient")
    def test_evaluate_qa_incorrect_counting(self, mock_client_class):
        """测试计数错误"""
        mock_client = Mock()
        mock_client.answer_question.return_value = "3辆"
        mock_client.estimate_image_tokens.return_value = 1500
        mock_client.count_tokens.return_value = 10
        mock_client_class.return_value = mock_client

        evaluator = VisualQAEvaluator(api_key="test-key")

        sample = MultimodalSample(
            sample_id="test-203",
            task_type=MultimodalTaskType.VISUAL_QA,
            image=ImageInput(type=ImageInputType.URL, value="https://example.com/traffic.jpg"),
            question="图中有几辆红色汽车？",
            ground_truth_answer="3辆",
            predicted_answer="2辆"  # 错误
        )

        result = evaluator.evaluate(sample)

        assert result.sample_id == "test-203"
        assert result.is_correct is False
        assert result.score == 0.0
        assert result.error_type == "counting_error"

    @patch("app.core.multimodal_evaluator.QwenVLClient")
    def test_evaluate_qa_number_extraction(self, mock_client_class):
        """测试数字提取匹配"""
        mock_client = Mock()
        mock_client.answer_question.return_value = "我看到5个苹果"
        mock_client.estimate_image_tokens.return_value = 1500
        mock_client.count_tokens.return_value = 10
        mock_client_class.return_value = mock_client

        evaluator = VisualQAEvaluator(api_key="test-key")

        sample = MultimodalSample(
            sample_id="test-204",
            task_type=MultimodalTaskType.VISUAL_QA,
            image=ImageInput(type=ImageInputType.URL, value="https://example.com/fruits.jpg"),
            question="有多少个苹果？",
            ground_truth_answer="5",
            predicted_answer="我数了一下，有5个苹果"
        )

        result = evaluator.evaluate(sample)

        assert result.is_correct is True
        assert result.score >= 0.9

    @patch("app.core.multimodal_evaluator.QwenVLClient")
    def test_classify_error_type_counting(self, mock_client_class):
        """测试计数错误分类"""
        evaluator = VisualQAEvaluator(api_key="test-key")

        error_type = evaluator._classify_error_type(
            predicted="2个",
            ground_truth="3个",
            reference="3个",
            question="有多少个苹果？"
        )

        assert error_type == "counting_error"

    @patch("app.core.multimodal_evaluator.QwenVLClient")
    def test_classify_error_type_color(self, mock_client_class):
        """测试颜色错误分类"""
        evaluator = VisualQAEvaluator(api_key="test-key")

        error_type = evaluator._classify_error_type(
            predicted="蓝色",
            ground_truth="红色",
            reference="红色",
            question="这辆车是什么颜色？"
        )

        assert error_type == "attribute_error"

    @patch("app.core.multimodal_evaluator.QwenVLClient")
    def test_classify_error_type_location(self, mock_client_class):
        """测试位置错误分类"""
        evaluator = VisualQAEvaluator(api_key="test-key")

        error_type = evaluator._classify_error_type(
            predicted="左边",
            ground_truth="右边",
            reference="右边",
            question="猫在哪里？"
        )

        assert error_type == "location_error"


class TestMultimodalEvaluationEngine:
    """测试多模态评测引擎"""

    @patch("app.core.multimodal_evaluator.QwenVLClient")
    def test_engine_evaluate_ocr(self, mock_client_class):
        """测试评测引擎处理OCR任务"""
        mock_client = Mock()
        mock_client.ocr.return_value = "测试文本"
        mock_client.estimate_image_tokens.return_value = 1500
        mock_client.count_tokens.return_value = 10
        mock_client_class.return_value = mock_client

        engine = MultimodalEvaluationEngine(api_key="test-key")

        sample = MultimodalSample(
            sample_id="test-301",
            task_type=MultimodalTaskType.OCR,
            image=ImageInput(type=ImageInputType.URL, value="https://example.com/test.jpg"),
            ground_truth_text="测试文本"
        )

        result = engine.evaluate(sample)

        assert result.sample_id == "test-301"
        assert result.task_type == MultimodalTaskType.OCR
        assert result.ocr_result is not None
        assert result.match_result is None
        assert result.qa_result is None

    @patch("app.core.multimodal_evaluator.QwenVLClient")
    def test_engine_evaluate_match(self, mock_client_class):
        """测试评测引擎处理图文匹配任务"""
        mock_client = Mock()
        mock_client.evaluate_image_text_match.return_value = {
            "content_consistency": 4,
            "detail_accuracy": 4,
            "completeness": 4,
            "overall_score": 4.0,
            "reasoning": "测试"
        }
        mock_client.describe_image.return_value = "真实内容"
        mock_client.estimate_image_tokens.return_value = 1500
        mock_client.count_tokens.return_value = 10
        mock_client_class.return_value = mock_client

        engine = MultimodalEvaluationEngine(api_key="test-key")

        sample = MultimodalSample(
            sample_id="test-302",
            task_type=MultimodalTaskType.IMAGE_TEXT_MATCH,
            image=ImageInput(type=ImageInputType.URL, value="https://example.com/test.jpg"),
            generated_description="生成的描述"
        )

        result = engine.evaluate(sample)

        assert result.sample_id == "test-302"
        assert result.task_type == MultimodalTaskType.IMAGE_TEXT_MATCH
        assert result.ocr_result is None
        assert result.match_result is not None
        assert result.qa_result is None

    @patch("app.core.multimodal_evaluator.QwenVLClient")
    def test_engine_evaluate_qa(self, mock_client_class):
        """测试评测引擎处理视觉问答任务"""
        mock_client = Mock()
        mock_client.answer_question.return_value = "答案"
        mock_client.estimate_image_tokens.return_value = 1500
        mock_client.count_tokens.return_value = 10
        mock_client_class.return_value = mock_client

        engine = MultimodalEvaluationEngine(api_key="test-key")

        sample = MultimodalSample(
            sample_id="test-303",
            task_type=MultimodalTaskType.VISUAL_QA,
            image=ImageInput(type=ImageInputType.URL, value="https://example.com/test.jpg"),
            question="问题？",
            ground_truth_answer="答案",
            predicted_answer="答案"
        )

        result = engine.evaluate(sample)

        assert result.sample_id == "test-303"
        assert result.task_type == MultimodalTaskType.VISUAL_QA
        assert result.ocr_result is None
        assert result.match_result is None
        assert result.qa_result is not None

    @patch("app.core.multimodal_evaluator.QwenVLClient")
    def test_engine_evaluate_batch_ocr(self, mock_client_class):
        """测试批量评测OCR"""
        mock_client = Mock()
        mock_client.ocr.side_effect = ["文本1", "文本2", "文本3"]
        mock_client.estimate_image_tokens.return_value = 1500
        mock_client.count_tokens.return_value = 10
        mock_client_class.return_value = mock_client

        engine = MultimodalEvaluationEngine(api_key="test-key")

        samples = [
            MultimodalSample(
                sample_id=f"test-{i}",
                task_type=MultimodalTaskType.OCR,
                image=ImageInput(type=ImageInputType.URL, value=f"https://example.com/{i}.jpg"),
                ground_truth_text=f"文本{i}"
            )
            for i in range(1, 4)
        ]

        summary = engine.evaluate_batch(samples)

        assert summary.total_samples == 3
        assert summary.task_type == MultimodalTaskType.OCR
        assert summary.average_char_accuracy is not None
        assert summary.average_word_accuracy is not None
        assert len(summary.results) == 3

    @patch("app.core.multimodal_evaluator.QwenVLClient")
    def test_engine_evaluate_batch_qa(self, mock_client_class):
        """测试批量评测视觉问答"""
        mock_client = Mock()
        mock_client.answer_question.side_effect = ["正确", "错误", "正确"]
        mock_client.estimate_image_tokens.return_value = 1500
        mock_client.count_tokens.return_value = 10
        mock_client_class.return_value = mock_client

        engine = MultimodalEvaluationEngine(api_key="test-key")

        samples = [
            MultimodalSample(
                sample_id=f"test-{i}",
                task_type=MultimodalTaskType.VISUAL_QA,
                image=ImageInput(type=ImageInputType.URL, value=f"https://example.com/{i}.jpg"),
                question="问题",
                ground_truth_answer="正确" if i != 2 else "错误",
                predicted_answer="正确" if i != 2 else "错误"
            )
            for i in range(1, 4)
        ]

        summary = engine.evaluate_batch(samples)

        assert summary.total_samples == 3
        assert summary.task_type == MultimodalTaskType.VISUAL_QA
        assert summary.accuracy is not None
        assert summary.correct_count is not None
        assert len(summary.results) == 3

    @patch("app.core.multimodal_evaluator.QwenVLClient")
    def test_engine_evaluate_batch_mixed_types(self, mock_client_class):
        """测试批量评测混合类型（应该失败）"""
        engine = MultimodalEvaluationEngine(api_key="test-key")

        samples = [
            MultimodalSample(
                sample_id="test-1",
                task_type=MultimodalTaskType.OCR,
                image=ImageInput(type=ImageInputType.URL, value="https://example.com/1.jpg"),
                ground_truth_text="文本"
            ),
            MultimodalSample(
                sample_id="test-2",
                task_type=MultimodalTaskType.VISUAL_QA,
                image=ImageInput(type=ImageInputType.URL, value="https://example.com/2.jpg"),
                question="问题",
                ground_truth_answer="答案",
                predicted_answer="答案"
            )
        ]

        with pytest.raises(ValueError, match="All samples must have the same task type"):
            engine.evaluate_batch(samples)

    @patch("app.core.multimodal_evaluator.QwenVLClient")
    def test_engine_evaluate_batch_empty(self, mock_client_class):
        """测试空样本列表"""
        engine = MultimodalEvaluationEngine(api_key="test-key")

        with pytest.raises(ValueError, match="Samples list is empty"):
            engine.evaluate_batch([])
