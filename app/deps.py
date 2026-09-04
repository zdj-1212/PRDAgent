"""全局依赖：LLM 客户端与 Milvus 存储的单例。"""
from app.config import settings
from app.llm import LLMClient
from app.milvus_store import MilvusStore

llm = LLMClient(settings)
milvus = MilvusStore(settings, llm)
