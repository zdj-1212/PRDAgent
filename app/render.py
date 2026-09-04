"""确定性 PRD 文档渲染器：把结构化需求素材渲染为标准 Markdown PRD。

模板结构遵循设计文档第 5 节，保证标题层级稳定；
所有 AI 推断的【假设】统一高亮，并在第 8 节集中列出。
"""
from __future__ import annotations

from typing import Any


def _fmt_list(items: Any) -> str:
    if not items:
        return "- 无"
    if isinstance(items, str):
        items = [items]
    return "\n".join(f"- {i}" for i in items)


def render_prd(product_name: str, material: dict[str, Any], assumptions: list[str]) -> str:
    title = product_name or material.get("product_name") or "未命名产品"
    lines: list[str] = []
    a = lines.append

    a(f"# {title} 产品需求文档 PRD")
    a("")
    a("## 1 文档概述")
    a("")
    a("### 1.1 文档目的")
    a("")
    a("本文档定义产品的业务背景、用户角色、功能与非功能需求、接口概要、风险与验收标准，作为研发、测试与验收的依据。")
    a("")
    a("### 1.2 范围")
    a("")
    a("本文档覆盖产品从 v1.0 起的首期需求范围，包含核心业务流程与基础支撑能力。")
    a("")
    a("### 1.3 修订记录")
    a("")
    a("| 版本 | 日期 | 修订说明 | 作者 |")
    a("| --- | --- | --- | --- |")
    a("| v1.0 | 待填写 | 初稿 | PRD Agent |")
    a("")

    # 2 业务背景与目标
    a("## 2 业务背景与目标")
    a("")
    a(material.get("business_background") or "（待补充）")
    a("")
    a(f"- **业务目标**：{material.get('business_goal') or '（待补充）'}")
    a("")

    # 3 用户角色
    a("## 3 用户角色说明")
    a("")
    roles = material.get("user_roles") or []
    if not roles:
        a("- 无")
    else:
        for r in roles:
            if isinstance(r, str):
                a(f"- **{r}**")
                continue
            a(f"- **{r.get('role', '')}**：{r.get('description', '')}")
            for s in r.get("stories") or []:
                a(f"    - {s}")
    a("")

    # 4 业务流程图
    a("## 4 业务流程图（mermaid）")
    a("")
    a("```mermaid")
    flow = material.get("business_flow_mermaid") or "graph TD\n  A[开始] --> B[结束]"
    # 兼容 LLM 偶发返回字面 "\n" 的情况
    flow = str(flow).replace("\\n", "\n")
    a(flow)
    a("```")
    a("")

    # 5 功能需求
    a("## 5 功能需求")
    a("")
    mods = material.get("functional_requirements") or []
    if not mods:
        a("（待补充）")
    for mi, mod in enumerate(mods, start=1):
        mod_name = mod.get("module") if isinstance(mod, dict) else str(mod)
        a(f"### 5.{mi} {mod_name}")
        a("")
        features = mod.get("features", []) if isinstance(mod, dict) else []
        for fi, f in enumerate(features, start=1):
            a(f"#### 5.{mi}.{fi} {f.get('name', f.get('id', ''))}")
            a("")
            a(f"- **编号**：{f.get('id', '')}")
            a(f"- **简述**：{f.get('summary', '')}")
            a(f"- **用户故事**：{f.get('user_story', '')}")
            a(f"- **前置条件**：")
            a(_fmt_list(f.get("preconditions")))
            a(f"- **输入**：")
            a(_fmt_list(f.get("input")))
            a(f"- **输出**：{f.get('output', '')}")
            a(f"- **操作流程**：")
            a(_fmt_list(f.get("steps")))
            a(f"- **异常流程**：")
            a(_fmt_list(f.get("exception_flow")) or "- 无")
            a(f"- **验收标准**：")
            a(_fmt_list(f.get("acceptance_criteria")) or "- 无")
            a("")

    # 6 非功能需求
    a("## 6 非功能需求")
    a("")
    nf = material.get("non_functional") or {}
    nf_labels = {
        "performance": "性能",
        "security": "安全",
        "compatibility": "兼容性",
        "compliance": "合规",
        "usability": "可用性",
    }
    for key, label in nf_labels.items():
        a(f"### 6.{list(nf_labels).index(key) + 1} {label}")
        a("")
        a(_fmt_list(nf.get(key)))
        a("")

    # 7 外部依赖与接口概要
    a("## 7 外部依赖与接口概要")
    a("")
    a("### 7.1 接口概要")
    a("")
    interfaces = material.get("interfaces") or []
    if not interfaces:
        a("- 无")
    else:
        for i in interfaces:
            if isinstance(i, str):
                a(f"- {i}")
                continue
            a(f"- **{i.get('name', '')}**（{i.get('direction', '')}）：{i.get('description', '')}")
    a("")
    a("### 7.2 外部依赖")
    a("")
    deps = material.get("dependencies") or []
    a(_fmt_list(deps) if deps else "- 无")
    a("")

    # 8 风险点与假设条件
    a("## 8 风险点与假设条件")
    a("")
    a("### 8.1 风险点")
    a("")
    risks = material.get("risks") or []
    if not risks:
        a("- 无")
    else:
        for r in risks:
            if isinstance(r, str):
                a(f"- {r}")
                continue
            a(
                f"- **{r.get('risk', '')}**（影响：{r.get('impact', '')}）"
                f" 缓解措施：{r.get('mitigation', '')}"
            )
    a("")
    a("### 8.2 假设条件")
    a("")
    a("> 以下为 AI 基于行业常识推断、用户未明确说明的信息（【假设】），请在评审时逐条确认：")
    a("")
    a(_fmt_list(assumptions) if assumptions else "- 无")
    a("")

    # 9 附录
    a("## 9 附录")
    a("")
    a("- 本文档由 PRD 生成 Agent 自动生成，正文中标注【假设】的内容需人工确认后进入评审。")
    a("")
    return "\n".join(lines)
