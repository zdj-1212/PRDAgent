"""LangGraph 图构建与运行封装。

工作流（与设计文档第 7 节一致）：
    用户输入
      ↓
    信息解析补全(parse) → [交互模式: 信息不足 → interrupt 反问用户 → resume 回填]
      ↓
    Milvus 检索(可选, 失败降级)
      ↓
    业务拆解(decompose) ──┐
      ↓                    │ 未达标且未到最大轮次(≤2)
    需求校验评审(review) ──┘
      ↓ 达标
    PRD文档组装(assemble) → 输出 markdown/json
"""
from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from app.graph.nodes import (
    assemble_node,
    bump_iteration,
    decompose_node,
    parse_node,
    retrieve_node,
    review_node,
    route_review,
)
from app.graph.state import PRDState
from app.schemas import GenerateRequest, GenerateResponse, QuestionItem

# 进程内共享的检查点存储器：以 thread_id 区分会话，支持交互模式挂起/恢复
_memory = MemorySaver()


def build_graph() -> Any:
    g = StateGraph(PRDState)
    g.add_node("parse", parse_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("decompose", decompose_node)
    g.add_node("bump_iteration", bump_iteration)
    g.add_node("review", review_node)
    g.add_node("assemble", assemble_node)

    g.add_edge(START, "parse")
    g.add_edge("parse", "retrieve")
    g.add_edge("retrieve", "decompose")
    g.add_edge("decompose", "review")
    g.add_conditional_edges(
        "review",
        route_review,
        {"fix": "bump_iteration", "done": "assemble"},
    )
    g.add_edge("bump_iteration", "decompose")
    g.add_edge("assemble", END)

    return g.compile(checkpointer=_memory)


def _initial_state(req: GenerateRequest) -> dict[str, Any]:
    return {
        "raw_input": req.raw_input,
        "mode": req.mode,
        "product_name": req.product_name,
        "business_background": req.business_background,
        "target_user": req.target_user,
        "constraints": list(req.constraints),
        "max_iterations": req.max_iterations,
        "use_milvus": req.use_milvus,
        "parsed": {},
        "is_info_enough": True,
        "questions": [],
        "assumptions": [],
        "retrieved": [],
        "req_material": {},
        "review": {},
        "iteration": 0,
        "iterations_used": 0,
        "prd_markdown": "",
        "prd_json": {},
        "error": None,
    }


def _run_until_halt(app: Any, config: dict, inputs: dict) -> tuple[dict, list, bool]:
    """运行图直到结束或触发 interrupt。

    返回 (最新完整状态, 中断值列表, 是否发生中断)。
    未中断时用 checkpointer 拉取合并后的完整状态，避免只拿到末节点局部更新。
    """
    latest: dict = {}
    interrupts: list = []
    halted = False
    for chunk in app.stream(inputs, config=config, stream_mode="updates"):
        for key, data in chunk.items():
            # 中断时 LangGraph 以 "__interrupt__" 作为 chunk 键（而非节点名）
            if key == "__interrupt__":
                halted = True
                interrupts = [
                    i.value for i in data if hasattr(i, "value")
                ]
            else:
                latest = data
    if not halted:
        latest = app.get_state(config).values
    return latest, interrupts, halted


def _to_response(
    req: GenerateRequest, state: dict, thread_id: str, milvus_available: bool
) -> GenerateResponse:
    return GenerateResponse(
        status="done",
        thread_id=thread_id,
        mode=req.mode,
        product_name=state.get("product_name", ""),
        prd_markdown=state.get("prd_markdown", ""),
        prd_json=state.get("prd_json", {}),
        assumptions=state.get("assumptions", []),
        iterations_used=state.get("iterations_used", 0),
        retrieved_examples=state.get("retrieved", []),
        milvus_available=milvus_available,
    )


def _questions_from_interrupts(interrupts: list) -> list[QuestionItem]:
    questions: list[QuestionItem] = []
    for it in interrupts:
        if isinstance(it, dict) and it.get("type") == "ask_questions":
            for q in it.get("questions", []):
                questions.append(
                    QuestionItem(field=q.get("field", ""), question=q.get("question", ""))
                )
    return questions


class PRDGraphRunner:
    """FastAPI 层使用的图运行器：封装 start / resume 两种入口。"""

    def __init__(self) -> None:
        self.app = build_graph()

    # ---------------- 发起生成 ----------------
    def start(self, req: GenerateRequest, thread_id: str, milvus_available: bool) -> GenerateResponse:
        config = {"configurable": {"thread_id": thread_id}}
        state, interrupts, halted = _run_until_halt(
            self.app, config, _initial_state(req)
        )
        if halted:
            return GenerateResponse(
                status="need_more_info",
                thread_id=thread_id,
                mode=req.mode,
                questions=_questions_from_interrupts(interrupts),
                milvus_available=milvus_available,
            )
        return _to_response(req, state, thread_id, milvus_available)

    # ---------------- 交互模式续跑 ----------------
    def resume(self, thread_id: str, answers: dict, mode: str, milvus_available: bool) -> GenerateResponse:
        config = {"configurable": {"thread_id": thread_id}}
        state, interrupts, halted = _run_until_halt(
            self.app, config, Command(resume=answers)
        )
        if halted:
            # 理论上交互模式至多反问一轮；若仍不足则继续返回提问
            return GenerateResponse(
                status="need_more_info",
                thread_id=thread_id,
                mode=mode,
                questions=_questions_from_interrupts(interrupts),
                milvus_available=milvus_available,
            )
        # 以检查点中保存的 mode 为准（兼容由 generate 创建的会话）
        saved_mode = state.get("mode") or mode
        req = GenerateRequest(raw_input=state.get("raw_input", ""), mode=saved_mode)
        return _to_response(req, state, thread_id, milvus_available)
