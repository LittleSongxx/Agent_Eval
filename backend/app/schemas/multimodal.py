"""
多模态评测数据模型
"""
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum


class ImageInputType(str, Enum):
    """图像输入类型"""
    URL = "url"
    BASE64 = "base64"
    FILE_PATH = "file_path"


class ImageInput(BaseModel):
    """图像输入"""
    model_config = ConfigDict(use_enum_values=True)

    type: ImageInputType = Field(..., description="图像输入类型")
    value: str = Field(..., description="图像数据（URL、Base64或文件路径）")
    size_bytes: Optional[int] = Field(None, description="图像大小（字节）")


class MultimodalTaskType(str, Enum):
    """多模态任务类型"""
    OCR = "ocr"  # 文字识别
    IMAGE_TEXT_MATCH = "image_text_match"  # 图文匹配度
    VISUAL_QA = "visual_qa"  # 视觉问答
    IMAGE_DESCRIPTION = "image_description"  # 图像描述生成


class MultimodalSample(BaseModel):
    """多模态评测样本"""
    model_config = ConfigDict(use_enum_values=True)

    sample_id: str = Field(..., description="样本ID")
    task_type: MultimodalTaskType = Field(..., description="任务类型")
    image: ImageInput = Field(..., description="图像输入")

    # OCR任务字段
    ground_truth_text: Optional[str] = Field(None, description="标准答案文本（OCR任务）")

    # 图文匹配任务字段
    generated_description: Optional[str] = Field(None, description="AI生成的描述（图文匹配任务）")

    # 视觉问答任务字段
    question: Optional[str] = Field(None, description="问题（视觉问答任务）")
    ground_truth_answer: Optional[str] = Field(None, description="标准答案（视觉问答任务）")
    predicted_answer: Optional[str] = Field(None, description="AI预测答案（视觉问答任务）")

    # 通用字段
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="元数据")


class OCREvaluationResult(BaseModel):
    """OCR评测结果"""
    sample_id: str = Field(..., description="样本ID")
    predicted_text: str = Field(..., description="识别的文本")
    ground_truth_text: str = Field(..., description="标准文本")

    # 准确率指标
    char_accuracy: float = Field(..., description="字符级准确率", ge=0.0, le=1.0)
    word_accuracy: float = Field(..., description="词级准确率", ge=0.0, le=1.0)
    edit_distance: int = Field(..., description="编辑距离")

    # 详细分析
    correct_chars: int = Field(..., description="正确字符数")
    total_chars: int = Field(..., description="总字符数")
    reasoning: str = Field(..., description="评测理由")

    # Token使用
    tokens_used: Optional[int] = Field(None, description="使用的token数")


class ImageTextMatchResult(BaseModel):
    """图文匹配度评测结果"""
    sample_id: str = Field(..., description="样本ID")
    generated_description: str = Field(..., description="生成的描述")

    # 评分维度
    content_consistency: float = Field(..., description="内容一致性", ge=1.0, le=5.0)
    detail_accuracy: float = Field(..., description="细节准确性", ge=1.0, le=5.0)
    completeness: float = Field(..., description="完整性", ge=1.0, le=5.0)
    overall_score: float = Field(..., description="综合得分", ge=1.0, le=5.0)

    # 详细分析
    reasoning: str = Field(..., description="评测理由")
    true_content_summary: Optional[str] = Field(None, description="真实内容摘要")

    # Token使用
    tokens_used: Optional[int] = Field(None, description="使用的token数")


class VisualQAEvaluationResult(BaseModel):
    """视觉问答评测结果"""
    sample_id: str = Field(..., description="样本ID")
    question: str = Field(..., description="问题")
    predicted_answer: str = Field(..., description="预测答案")
    ground_truth_answer: str = Field(..., description="标准答案")

    # 评测结果
    is_correct: bool = Field(..., description="答案是否正确")
    score: float = Field(..., description="得分", ge=0.0, le=1.0)
    reference_answer: Optional[str] = Field(None, description="模型参考答案")

    # 错误分析
    error_type: Optional[str] = Field(None, description="错误类型")
    reasoning: str = Field(..., description="评测理由")

    # Token使用
    tokens_used: Optional[int] = Field(None, description="使用的token数")


class MultimodalEvaluationResult(BaseModel):
    """多模态评测结果（统一格式）"""
    model_config = ConfigDict(use_enum_values=True)

    sample_id: str = Field(..., description="样本ID")
    task_type: MultimodalTaskType = Field(..., description="任务类型")
    score: float = Field(..., description="得分", ge=0.0, le=5.0)
    reasoning: str = Field(..., description="评测理由")

    # 详细结果（根据任务类型选择性填充）
    ocr_result: Optional[OCREvaluationResult] = Field(None, description="OCR评测结果")
    match_result: Optional[ImageTextMatchResult] = Field(None, description="图文匹配结果")
    qa_result: Optional[VisualQAEvaluationResult] = Field(None, description="视觉问答结果")

    # 通用字段
    tokens_used: int = Field(..., description="使用的token数")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="元数据")


class MultimodalEvaluationSummary(BaseModel):
    """多模态评测汇总"""
    model_config = ConfigDict(use_enum_values=True)

    total_samples: int = Field(..., description="总样本数")
    task_type: MultimodalTaskType = Field(..., description="任务类型")

    # 汇总指标
    average_score: float = Field(..., description="平均得分")
    min_score: float = Field(..., description="最低得分")
    max_score: float = Field(..., description="最高得分")

    # OCR专用指标
    average_char_accuracy: Optional[float] = Field(None, description="平均字符准确率")
    average_word_accuracy: Optional[float] = Field(None, description="平均词准确率")

    # 视觉问答专用指标
    accuracy: Optional[float] = Field(None, description="准确率（正确率）")
    correct_count: Optional[int] = Field(None, description="正确数量")

    # 成本统计
    total_tokens_used: int = Field(..., description="总token使用量")
    average_tokens_per_sample: float = Field(..., description="平均每样本token数")

    # 详细结果列表
    results: List[MultimodalEvaluationResult] = Field(default_factory=list, description="详细结果")
