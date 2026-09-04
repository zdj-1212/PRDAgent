"""全局配置：从环境变量 / .env 读取，遵循 12-factor 风格。"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- LLM（OpenAI 兼容接口）----
    # provider: openai | mock
    #   - openai：调用任意 OpenAI 兼容的 Chat Completions 服务
    #     （OpenAI / DeepSeek / 通义千问(DashScope兼容模式) / vLLM / Ollama 等）
    #   - mock：不调用真实模型，返回固定结构化结果，用于离线验证工作流与 API
    llm_provider: str = "openai"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.2
    llm_timeout: float = 120.0
    llm_max_retries: int = 2

    # ---- Embedding（Milvus 检索用，OpenAI 兼容接口）----
    embedding_base_url: str = ""
    embedding_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    # ---- Milvus ----
    # 支持: http://host:port (Milvus 2.x 服务)、https://...、file:./xxx.db (Milvus Lite)
    milvus_uri: str = "http://localhost:19530"
    milvus_collection: str = "prd_examples"
    milvus_top_k: int = 3
    # 检索不到 Milvus 时是否允许静默降级（true: 跳过检索继续生成）
    milvus_fail_soft: bool = True

    # ---- 流程控制 ----
    # 需求校验迭代的最大轮数（设计文档要求最多 2 轮，避免死循环）
    default_max_iterations: int = 2

    @property
    def eff_embedding_base_url(self) -> str:
        return self.embedding_base_url or self.llm_base_url

    @property
    def eff_embedding_api_key(self) -> str:
        return self.embedding_api_key or self.llm_api_key


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
