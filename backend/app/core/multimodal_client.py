"""
多模态模型客户端
支持通义千问视觉语言模型（qwen-vl-plus）
"""
import os
import base64
import logging
from typing import Optional, Dict, Any, List
from openai import OpenAI

logger = logging.getLogger(__name__)


class QwenVLClient:
    """
    通义千问视觉语言模型客户端
    支持图像理解、OCR、视觉问答等任务
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "qwen-vl-plus",
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ):
        """
        初始化客户端

        Args:
            api_key: DashScope API Key，如果为None则从环境变量读取
            model: 模型名称，默认qwen-vl-plus
            base_url: API端点，默认DashScope兼容模式
        """
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        if not self.api_key:
            raise ValueError("API key is required. Set LLM_API_KEY environment variable.")

        self.model = model
        self.base_url = base_url

        # 初始化OpenAI客户端（兼容模式）
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

        logger.info(f"Initialized QwenVLClient with model: {self.model}")

    def _prepare_image_message(
        self,
        image_input: str,
        text: str
    ) -> List[Dict[str, Any]]:
        """
        准备图像消息

        Args:
            image_input: 图像输入，支持URL或Base64编码
            text: 文本提示

        Returns:
            消息列表
        """
        # 判断是URL还是Base64
        if image_input.startswith("http://") or image_input.startswith("https://"):
            # URL格式
            image_content = {
                "type": "image_url",
                "image_url": {"url": image_input}
            }
        elif image_input.startswith("data:image"):
            # Data URL格式（已包含data:image/jpeg;base64,前缀）
            image_content = {
                "type": "image_url",
                "image_url": {"url": image_input}
            }
        else:
            # 假设是纯Base64编码，添加前缀
            image_content = {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_input}"}
            }

        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    image_content
                ]
            }
        ]

    def understand_image(
        self,
        image_input: str,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2000
    ) -> str:
        """
        理解图像内容

        Args:
            image_input: 图像输入（URL或Base64）
            prompt: 提示文本
            temperature: 温度参数
            max_tokens: 最大token数

        Returns:
            模型响应文本
        """
        try:
            messages = self._prepare_image_message(image_input, prompt)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )

            result = response.choices[0].message.content

            # 记录token使用情况
            if hasattr(response, "usage"):
                logger.info(
                    f"Token usage - prompt: {response.usage.prompt_tokens}, "
                    f"completion: {response.usage.completion_tokens}, "
                    f"total: {response.usage.total_tokens}"
                )

            return result

        except Exception as e:
            logger.error(f"Error in understand_image: {str(e)}")
            raise

    def ocr(
        self,
        image_input: str,
        language: str = "ch"
    ) -> str:
        """
        OCR文字识别

        Args:
            image_input: 图像输入（URL或Base64）
            language: 语言，ch（中文）或en（英文）

        Returns:
            识别的文本
        """
        prompt = """请识别图像中的所有文字，并按原文输出。

要求：
1. 保持原有的换行和格式
2. 如果有表格，尽量保持表格结构
3. 只输出识别的文字，不要添加任何解释

识别结果："""

        return self.understand_image(
            image_input=image_input,
            prompt=prompt,
            temperature=0.0,  # OCR需要确定性输出
            max_tokens=4000
        )

    def describe_image(
        self,
        image_input: str,
        detailed: bool = True
    ) -> str:
        """
        描述图像内容

        Args:
            image_input: 图像输入（URL或Base64）
            detailed: 是否需要详细描述

        Returns:
            图像描述文本
        """
        if detailed:
            prompt = """请详细描述这张图像的内容。

要求：
1. 描述主要物体和场景
2. 描述物体的位置关系
3. 描述颜色、形状等细节
4. 描述图像的整体氛围或风格

描述："""
        else:
            prompt = "请简要描述这张图像的主要内容："

        return self.understand_image(
            image_input=image_input,
            prompt=prompt,
            temperature=0.3,
            max_tokens=1000 if detailed else 300
        )

    def answer_question(
        self,
        image_input: str,
        question: str
    ) -> str:
        """
        视觉问答

        Args:
            image_input: 图像输入（URL或Base64）
            question: 问题

        Returns:
            答案
        """
        prompt = f"""请根据图像内容回答以下问题。

问题：{question}

要求：
1. 答案要准确，基于图像中的实际内容
2. 如果无法确定答案，请说明原因
3. 答案要简洁明了

答案："""

        return self.understand_image(
            image_input=image_input,
            prompt=prompt,
            temperature=0.1,
            max_tokens=500
        )

    def evaluate_image_text_match(
        self,
        image_input: str,
        text_description: str
    ) -> Dict[str, Any]:
        """
        评估图文匹配度

        Args:
            image_input: 图像输入（URL或Base64）
            text_description: 文本描述

        Returns:
            评估结果（包含评分和理由）
        """
        prompt = f"""你是一个图文匹配度评测专家。请评估以下文本描述与图像内容的匹配程度。

文本描述：
{text_description}

请从以下3个维度进行评分（1-5分）：
1. 内容一致性：描述是否与图像内容一致
2. 细节准确性：关键细节是否准确
3. 完整性：是否覆盖了重要信息

请按以下JSON格式输出（只输出JSON，不要其他内容）：
{{
  "content_consistency": <1-5分数>,
  "detail_accuracy": <1-5分数>,
  "completeness": <1-5分数>,
  "overall_score": <平均分>,
  "reasoning": "<详细评测理由，说明为什么给出这些分数>"
}}"""

        response = self.understand_image(
            image_input=image_input,
            prompt=prompt,
            temperature=0.2,
            max_tokens=1000
        )

        # 解析JSON响应
        import json
        try:
            # 尝试提取JSON部分
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]

            result = json.loads(response.strip())
            return result
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {response}")
            # 返回默认结果
            return {
                "content_consistency": 3,
                "detail_accuracy": 3,
                "completeness": 3,
                "overall_score": 3.0,
                "reasoning": f"解析评测结果失败: {str(e)}"
            }

    def count_tokens(self, text: str) -> int:
        """
        估算token数量（简单估算）

        Args:
            text: 文本内容

        Returns:
            估算的token数
        """
        # 简单估算：中文约1.5字符/token，英文约4字符/token
        # 这里简化为2字符/token
        return len(text) // 2

    def estimate_image_tokens(self, image_size_bytes: Optional[int] = None) -> int:
        """
        估算图像token消耗

        Args:
            image_size_bytes: 图像大小（字节）

        Returns:
            估算的token数
        """
        # 根据qwen-vl的文档，一张图像大约消耗1000-2000个token
        # 这里简化处理
        if image_size_bytes:
            # 大图像消耗更多token
            if image_size_bytes > 1024 * 1024:  # >1MB
                return 2000
            elif image_size_bytes > 512 * 1024:  # >512KB
                return 1500
            else:
                return 1000
        else:
            # 默认估算
            return 1500


def encode_image_to_base64(image_path: str) -> str:
    """
    将本地图像编码为Base64

    Args:
        image_path: 图像文件路径

    Returns:
        Base64编码的字符串（包含data URL前缀）
    """
    with open(image_path, "rb") as f:
        image_data = f.read()

    base64_str = base64.b64encode(image_data).decode("utf-8")

    # 根据文件扩展名确定MIME类型
    if image_path.lower().endswith(".png"):
        mime_type = "image/png"
    elif image_path.lower().endswith(".jpg") or image_path.lower().endswith(".jpeg"):
        mime_type = "image/jpeg"
    elif image_path.lower().endswith(".gif"):
        mime_type = "image/gif"
    else:
        mime_type = "image/jpeg"  # 默认

    return f"data:{mime_type};base64,{base64_str}"
