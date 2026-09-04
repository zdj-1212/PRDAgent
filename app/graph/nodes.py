"""LangGraph 节点函数：解析 → 检索 → 拆解 → 评审 → 组装。"""
from __future__ import annotations

from typing import Any

from langgraph.types import interrupt

from app.deps import llm, milvus
from app.graph.state import PRDState
from app.prompts import (
    DECOMPOSE_SYSTEM,
    PARSE_SYSTEM,
    REVIEW_SYSTEM,
    build_decompose_user,
    build_parse_user,
    build_review_user,
)
from app.render import render_prd


# ============================================================
# 节点 1：信息解析 & 补全
# ============================================================
def parse_node(state: PRDState) -> dict[str, Any]:
    parsed = llm.chat_json(
        PARSE_SYSTEM,
        build_parse_user(
            raw_input=state["raw_input"],
            mode=state["mode"],
            product_name=state.get("product_name", ""),
            business_background=state.get("business_background", ""),
            target_user=state.get("target_user", ""),
            constraints=state.get("constraints", []),
        ),
        purpose="parse",
    )
    assumptions = list(state.get("assumptions") or [])
    assumptions += list(parsed.get("assumptions") or [])

    # 交互模式：信息不足则反问用户（LangGraph interrupt 挂起图）
    if (
        state["mode"] == "interactive"
        and not parsed.get("is_info_enough", False)
        and parsed.get("questions")
    ):
        questions = parsed["questions"]
        answers = interrupt(
            {"type": "ask_questions", "questions": questions}
        )
        parsed, filled = _apply_answers(parsed, questions, answers)
        assumptions += [
            f"【假设】{q.get('question', '')} —— 用户补充：{answers.get(q.get('field', ''), '')}"
            for q in questions
            if q.get("field") in filled
        ]
        parsed["is_info_enough"] = True

    return {
        "parsed": parsed,
        "is_info_enough": parsed.get("is_info_enough", True),
        "questions": parsed.get("questions", []),
        "assumptions": assumptions,
    }


def _apply_answers(
    parsed: dict, questions: list[dict], answers: dict
) -> tuple[dict, set[str]]:
    """把用户回答按 field 回填到解析结果中。返回 (parsed, 已回填的field集合)。"""
    filled: set[str] = set()
    for q in questions:
        field = q.get("field")
        if not field or field not in answers:
            continue
        val = answers[field]
        if val is None or (isinstance(val, str) and not val.strip()):
            continue
        if field == "user_roles":
            if isinstance(val, str):
                normalized = val.replace("、", "，").replace(",", "，")
                val = [x.strip() for x in normalized.split("，") if x.strip()]
            parsed["user_roles"] = val
        elif field == "constraints":
            existing = list(parsed.get("constraints") or [])
            if isinstance(val, list):
                existing.extend(val)
            else:
                existing.append(str(val))
            parsed["constraints"] = existing
        else:
            parsed[field] = val
        filled.add(field)
    # 从缺失项中移除已回填字段
    if "missing_info" in parsed and filled:
        parsed["missing_info"] = [
            m for m in parsed["missing_info"] if m not in filled
        ]
    return parsed, filled


# ============================================================
# 节点 1.5：Milvus 检索同类 PRD 范例（可选增强，失败降级）
# ============================================================
def retrieve_node(state: PRDState) -> dict[str, Any]:
    if not state.get("use_milvus", True) or not milvus.available:
        return {"retrieved": []}
    parsed = state.get("parsed") or {}
    query = " ".join(
        filter(
            None,
            [
                str(parsed.get("product_name") or ""),
                str(parsed.get("business_goal") or ""),
                state.get("raw_input", ""),
            ],
        )
    )
    if not query.strip():
        return {"retrieved": []}
    return {"retrieved": milvus.search(query)}


# ============================================================
# 节点 2：业务拆解（核心）
# ============================================================
def decompose_node(state: PRDState) -> dict[str, Any]:
    review = state.get("review") or {}
    suggestions = review.get("comments") if review.get("need_fix") else None
    material = llm.chat_json(
        DECOMPOSE_SYSTEM,
        build_decompose_user(
            parsed=state["parsed"],
            review_suggestions=suggestions,
            iteration=state.get("iteration", 0),
            retrieved=state.get("retrieved") or [],
        ),
        purpose="decompose",
    )
    assumptions = list(state.get("assumptions") or [])
    assumptions += list(material.get("assumptions") or [])
    return {"req_material": material, "assumptions": assumptions}


# ============================================================
# 节点 3：需求校验评审（独立 LLM 调用）
# ============================================================
def review_node(state: PRDState) -> dict[str, Any]:
    review = llm.chat_json(
        REVIEW_SYSTEM,
        build_review_user(state["req_material"]),
        purpose="review",
    )
    return {"review": review}


def bump_iteration(state: PRDState) -> dict[str, Any]:
    return {"iteration": state.get("iteration", 0) + 1}


def route_review(state: PRDState) -> str:
    """评审后判断：需要修改且未到最大轮次 -> 回拆解；否则 -> 组装。"""
    review = state.get("review") or {}
    if (
        review.get("need_fix")
        and state.get("iteration", 0) < state.get("max_iterations", 2)
    ):
        return "fix"
    return "done"


# ============================================================
# 节点 4：文档组装（确定性渲染，保证模板结构稳定）
# ============================================================
def assemble_node(state: PRDState) -> dict[str, Any]:
    parsed = state.get("parsed") or {}
    product_name = (
        parsed.get("product_name")
        or state.get("product_name")
        or "未命名产品"
    )
    material = state["req_material"]
    md = render_prd(product_name, material, state.get("assumptions") or [])
    return {
        "prd_markdown": md,
        "prd_json": {
            "product_name": product_name,
            "meta": {
                "mode": state["mode"],
                "iterations_used": state.get("iteration", 0),
                "milvus_retrieved": state.get("retrieved") or [],
                "assumptions": state.get("assumptions") or [],
            },
            "material": material,
        },
        "iterations_used": state.get("iteration", 0),
        "product_name": product_name,
    }
