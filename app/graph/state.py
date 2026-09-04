"""LangGraph 状态定义。"""
from typing import Any, TypedDict


class PRDState(TypedDict):
    # ---- 输入 ----
    raw_input: str
    mode: str  # auto | interactive
    product_name: str
    business_background: str
    target_user: str
    constraints: list[str]
    max_iterations: int
    use_milvus: bool

    # ---- 节点1 输出 ----
    parsed: dict[str, Any]  # 结构化解析结果
    is_info_enough: bool
    questions: list[dict[str, str]]  # 交互模式反问列表
    assumptions: list[str]

    # ---- Milvus 检索 ----
    retrieved: list[str]

    # ---- 节点2 输出 ----
    req_material: dict[str, Any]  # 中间结构化需求素材

    # ---- 节点3 输出 ----
    review: dict[str, Any]  # {need_fix, comments}

    # ---- 迭代 ----
    iteration: int
    iterations_used: int

    # ---- 节点4 输出 ----
    prd_markdown: str
    prd_json: dict[str, Any]

    # ---- 错误 ----
    error: str | None
