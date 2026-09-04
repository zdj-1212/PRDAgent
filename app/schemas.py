"""API 层 Pydantic 模型：请求 / 响应 / 中断信息。"""
from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------- 请求 ----------
class GenerateRequest(BaseModel):
    """生成 PRD 的请求体。

    - mode="auto"（默认，程序调用）：信息不足时自动补全假设，一次性输出完整 PRD。
    - mode="interactive"（推荐给人用）：信息不足时先反问用户，补齐后再继续。
    """

    raw_input: str = Field(
        ..., min_length=1, description="用户原始想法/一句话需求/草稿文本"
    )
    mode: Literal["auto", "interactive"] = "auto"
    product_name: str = ""
    business_background: str = ""
    target_user: str = ""
    constraints: list[str] = Field(default_factory=list)
    max_iterations: int = Field(default=2, ge=0, le=5)
    use_milvus: bool = True


class ResumeRequest(BaseModel):
    """交互模式下，用户回答完反问问题后，续跑图。"""

    thread_id: str = Field(..., description="generate 返回的会话 id")
    answers: dict[str, Any] = Field(
        ..., description="问题字段 -> 用户回答的映射，字段与 questions 中的 field 对应"
    )


# ---------- 响应 ----------
class QuestionItem(BaseModel):
    field: str = Field(..., description="缺失信息对应的字段名，用于 answers 回填")
    question: str = Field(..., description="面向用户的提问文本")


class GenerateResponse(BaseModel):
    status: Literal["need_more_info", "done", "error"]
    thread_id: str = ""
    mode: str = "auto"
    # need_more_info 时返回
    questions: list[QuestionItem] = Field(default_factory=list)
    # done 时返回
    product_name: str = ""
    prd_markdown: str = ""
    prd_json: dict[str, Any] = Field(default_factory=dict)
    assumptions: list[str] = Field(default_factory=list)
    iterations_used: int = 0
    # 检索信息
    retrieved_examples: list[str] = Field(default_factory=list)
    milvus_available: bool = False
    # error 时返回
    detail: str = ""
