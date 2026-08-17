"""
多模态客户端测试
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from app.core.multimodal_client import QwenVLClient, encode_image_to_base64


class TestQwenVLClient:
    """测试QwenVLClient"""

    @patch("app.core.multimodal_client.OpenAI")
    def test_init_with_api_key(self, mock_openai):
        """测试使用API Key初始化"""
        client = QwenVLClient(api_key="test-key")
        assert client.api_key == "test-key"
        assert client.model == "qwen-vl-plus"
        mock_openai.assert_called_once()

    @patch("app.core.multimodal_client.OpenAI")
    @patch.dict("os.environ", {"LLM_API_KEY": "env-key"})
    def test_init_with_env_key(self, mock_openai):
        """测试从环境变量读取API Key"""
        client = QwenVLClient()
        assert client.api_key == "env-key"

    @patch("app.core.multimodal_client.OpenAI")
    @patch.dict("os.environ", {}, clear=True)
    def test_init_without_api_key(self, mock_openai):
        """测试没有API Key时抛出异常"""
        with pytest.raises(ValueError, match="API key is required"):
            QwenVLClient()

    @patch("app.core.multimodal_client.OpenAI")
    def test_prepare_image_message_url(self, mock_openai):
        """测试准备URL格式的图像消息"""
        client = QwenVLClient(api_key="test-key")
        messages = client._prepare_image_message(
            image_input="https://example.com/image.jpg",
            text="Describe this image"
        )

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert len(messages[0]["content"]) == 2
        assert messages[0]["content"][0]["type"] == "text"
        assert messages[0]["content"][1]["type"] == "image_url"
        assert messages[0]["content"][1]["image_url"]["url"] == "https://example.com/image.jpg"

    @patch("app.core.multimodal_client.OpenAI")
    def test_prepare_image_message_base64(self, mock_openai):
        """测试准备Base64格式的图像消息"""
        client = QwenVLClient(api_key="test-key")
        base64_str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        messages = client._prepare_image_message(
            image_input=base64_str,
            text="Analyze"
        )

        assert messages[0]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")

    @patch("app.core.multimodal_client.OpenAI")
    def test_prepare_image_message_data_url(self, mock_openai):
        """测试准备Data URL格式的图像消息"""
        client = QwenVLClient(api_key="test-key")
        data_url = "data:image/png;base64,iVBORw0KGgo="
        messages = client._prepare_image_message(
            image_input=data_url,
            text="Test"
        )

        assert messages[0]["content"][1]["image_url"]["url"] == data_url

    @patch("app.core.multimodal_client.OpenAI")
    def test_understand_image(self, mock_openai):
        """测试理解图像"""
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "This is a cat"
        mock_response.usage = Mock(prompt_tokens=100, completion_tokens=20, total_tokens=120)

        mock_client_instance = Mock()
        mock_client_instance.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client_instance

        client = QwenVLClient(api_key="test-key")
        result = client.understand_image(
            image_input="https://example.com/cat.jpg",
            prompt="What's in this image?"
        )

        assert result == "This is a cat"
        mock_client_instance.chat.completions.create.assert_called_once()

    @patch("app.core.multimodal_client.OpenAI")
    def test_ocr(self, mock_openai):
        """测试OCR功能"""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "发票号码: 12345678"
        mock_response.usage = Mock(prompt_tokens=1500, completion_tokens=50, total_tokens=1550)

        mock_client_instance = Mock()
        mock_client_instance.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client_instance

        client = QwenVLClient(api_key="test-key")
        result = client.ocr(image_input="https://example.com/invoice.jpg")

        assert result == "发票号码: 12345678"
        # 验证temperature=0.0（确定性输出）
        call_kwargs = mock_client_instance.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.0

    @patch("app.core.multimodal_client.OpenAI")
    def test_describe_image_detailed(self, mock_openai):
        """测试详细图像描述"""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "A beautiful sunset over the ocean"
        mock_response.usage = Mock(prompt_tokens=1500, completion_tokens=100, total_tokens=1600)

        mock_client_instance = Mock()
        mock_client_instance.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client_instance

        client = QwenVLClient(api_key="test-key")
        result = client.describe_image(
            image_input="https://example.com/sunset.jpg",
            detailed=True
        )

        assert result == "A beautiful sunset over the ocean"

    @patch("app.core.multimodal_client.OpenAI")
    def test_answer_question(self, mock_openai):
        """测试视觉问答"""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "3辆红色汽车"
        mock_response.usage = Mock(prompt_tokens=1600, completion_tokens=30, total_tokens=1630)

        mock_client_instance = Mock()
        mock_client_instance.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client_instance

        client = QwenVLClient(api_key="test-key")
        result = client.answer_question(
            image_input="https://example.com/traffic.jpg",
            question="图中有几辆红色汽车？"
        )

        assert result == "3辆红色汽车"

    @patch("app.core.multimodal_client.OpenAI")
    def test_evaluate_image_text_match(self, mock_openai):
        """测试图文匹配度评估"""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '''```json
{
  "content_consistency": 4,
  "detail_accuracy": 5,
  "completeness": 4,
  "overall_score": 4.3,
  "reasoning": "描述准确，细节完整"
}
```'''
        mock_response.usage = Mock(prompt_tokens=1700, completion_tokens=80, total_tokens=1780)

        mock_client_instance = Mock()
        mock_client_instance.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client_instance

        client = QwenVLClient(api_key="test-key")
        result = client.evaluate_image_text_match(
            image_input="https://example.com/scene.jpg",
            text_description="一只猫坐在窗台上"
        )

        assert result["content_consistency"] == 4
        assert result["detail_accuracy"] == 5
        assert result["completeness"] == 4
        assert result["overall_score"] == 4.3
        assert "描述准确" in result["reasoning"]

    @patch("app.core.multimodal_client.OpenAI")
    def test_evaluate_image_text_match_json_parse_error(self, mock_openai):
        """测试图文匹配度评估JSON解析失败"""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Invalid JSON response"
        mock_response.usage = Mock(prompt_tokens=1700, completion_tokens=10, total_tokens=1710)

        mock_client_instance = Mock()
        mock_client_instance.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client_instance

        client = QwenVLClient(api_key="test-key")
        result = client.evaluate_image_text_match(
            image_input="https://example.com/scene.jpg",
            text_description="测试"
        )

        # 应该返回默认结果
        assert result["content_consistency"] == 3
        assert result["detail_accuracy"] == 3
        assert result["completeness"] == 3
        assert result["overall_score"] == 3.0
        assert "解析评测结果失败" in result["reasoning"]

    @patch("app.core.multimodal_client.OpenAI")
    def test_understand_image_error(self, mock_openai):
        """测试API调用失败"""
        mock_client_instance = Mock()
        mock_client_instance.chat.completions.create.side_effect = Exception("API Error")
        mock_openai.return_value = mock_client_instance

        client = QwenVLClient(api_key="test-key")

        with pytest.raises(Exception, match="API Error"):
            client.understand_image(
                image_input="https://example.com/test.jpg",
                prompt="Test"
            )

    def test_count_tokens(self):
        """测试token计数"""
        with patch("app.core.multimodal_client.OpenAI"):
            client = QwenVLClient(api_key="test-key")

            # 简单估算：2字符/token
            assert client.count_tokens("Hello World") == 5  # 11字符 / 2
            assert client.count_tokens("你好世界") == 2  # 4字符 / 2

    def test_estimate_image_tokens(self):
        """测试图像token估算"""
        with patch("app.core.multimodal_client.OpenAI"):
            client = QwenVLClient(api_key="test-key")

            # 大图像
            assert client.estimate_image_tokens(2 * 1024 * 1024) == 2000

            # 中等图像
            assert client.estimate_image_tokens(600 * 1024) == 1500

            # 小图像
            assert client.estimate_image_tokens(300 * 1024) == 1000

            # 默认
            assert client.estimate_image_tokens() == 1500


class TestEncodeImageToBase64:
    """测试图像Base64编码"""

    def test_encode_png(self, tmp_path):
        """测试编码PNG图像"""
        # 创建临时PNG文件
        image_file = tmp_path / "test.png"
        image_file.write_bytes(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR')

        result = encode_image_to_base64(str(image_file))

        assert result.startswith("data:image/png;base64,")
        assert len(result) > 30

    def test_encode_jpg(self, tmp_path):
        """测试编码JPG图像"""
        image_file = tmp_path / "test.jpg"
        image_file.write_bytes(b'\xff\xd8\xff\xe0\x00\x10JFIF')

        result = encode_image_to_base64(str(image_file))

        assert result.startswith("data:image/jpeg;base64,")

    def test_encode_jpeg(self, tmp_path):
        """测试编码JPEG图像"""
        image_file = tmp_path / "test.jpeg"
        image_file.write_bytes(b'\xff\xd8\xff\xe0')

        result = encode_image_to_base64(str(image_file))

        assert result.startswith("data:image/jpeg;base64,")

    def test_encode_unknown_format(self, tmp_path):
        """测试编码未知格式（默认为JPEG）"""
        image_file = tmp_path / "test.bin"
        image_file.write_bytes(b'binary data')

        result = encode_image_to_base64(str(image_file))

        assert result.startswith("data:image/jpeg;base64,")
