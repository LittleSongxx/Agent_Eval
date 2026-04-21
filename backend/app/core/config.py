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

    model_config = SettingsConfigDict(
        env_file=(".env", "backend/.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
