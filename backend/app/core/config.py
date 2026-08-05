from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    DATABASE_URL: str = "sqlite:///./eval_platform.db"
    UPLOAD_DIR: str = "./uploads"
    CORS_ORIGINS: list[str] = ["http://localhost:5173"]
    SEED_ON_STARTUP: bool = True
    RUN_EVAL_ON_CREATE: bool = True

    DEFAULT_LLM_NAME: str = "Qwen Plus (通义千问)"
    LLM_PROVIDER: str = "openai"
    LLM_ENDPOINT: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    LLM_MODEL: str = "qwen-plus"
    LLM_API_KEY: str = ""
    LLM_TEMPERATURE: float = 0.01
    LLM_MAX_TOKENS: int = 1024

    # Judge 可靠性：每行每指标的独立采样次数（>1 开启多采样，用标准差量化评分稳定性）
    EVAL_JUDGE_SAMPLES: int = 1
    # 行级采样标准差超过该阈值的样本标记为低置信度
    JUDGE_STD_THRESHOLD: float = 0.1
    # Judge 调用成本估算单价（元 / 1K tokens，按 DashScope qwen-plus 参考价）
    LLM_INPUT_PRICE_PER_1K: float = 0.0008
    LLM_OUTPUT_PRICE_PER_1K: float = 0.002

    model_config = SettingsConfigDict(
        env_file=(".env", "backend/.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
