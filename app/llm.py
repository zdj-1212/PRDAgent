"""LLM 客户端封装。

- provider=openai：调用任意 OpenAI 兼容 Chat Completions / Embeddings 接口
  （OpenAI、DeepSeek、通义千问 DashScope 兼容模式、vLLM、Ollama 等）
- provider=mock：不调用外部模型，返回确定性结构化结果，用于离线验证工作流与 API。
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

from app.config import Settings


def parse_json(content: str) -> dict[str, Any]:
    """健壮解析 LLM 输出的 JSON：去除 ```json 围栏、截取首个平衡的 {...}。"""
    if not content or not content.strip():
        raise ValueError("LLM 返回内容为空")
    text = content.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 去除 markdown 代码围栏
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 截取第一个 { 到最后一个 } 之间的平衡 JSON
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"无法从 LLM 输出中解析 JSON: {content[:200]!r}")
    return json.loads(text[start : end + 1])


class LLMClient:
    """统一 LLM 接口：chat_json 强制 JSON 输出，embed 提供向量。"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.is_mock = settings.llm_provider.lower() == "mock"
        self._client: Any = None
        self._embed_client: Any = None
        if not self.is_mock:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=settings.llm_base_url,
                api_key=settings.llm_api_key,
                timeout=settings.llm_timeout,
                max_retries=settings.llm_max_retries,
            )

    # ---------------- 对话 / 结构化输出 ----------------
    def chat_json(
        self,
        system: str,
        user: str,
        purpose: str = "",
    ) -> dict[str, Any]:
        """调用模型并要求输出 JSON 对象。purpose 仅 mock 模式用于选择固定返回。"""
        if self.is_mock:
            return _mock_json(purpose, user)
        resp = self._client.chat.completions.create(
            model=self.settings.llm_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self.settings.llm_temperature,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content or ""
        return parse_json(content)

    # ---------------- Embedding ----------------
    def embed(self, texts: list[str]) -> list[list[float]]:
        if self.is_mock:
            return [_mock_embedding(t, self.settings.embedding_dim) for t in texts]
        if self._embed_client is None:
            from openai import OpenAI

            self._embed_client = OpenAI(
                base_url=self.settings.eff_embedding_base_url,
                api_key=self.settings.eff_embedding_api_key,
                timeout=self.settings.llm_timeout,
                max_retries=self.settings.llm_max_retries,
            )
        resp = self._embed_client.embeddings.create(
            model=self.settings.embedding_model, input=texts
        )
        # 按输入顺序返回向量
        rows = sorted(resp.data, key=lambda d: d.index)
        return [r.embedding for r in rows]


# ============================================================
# Mock 实现：确定性输出，仅用于离线验证工作流拓扑与 API 契约
# ============================================================
def _mock_embedding(text: str, dim: int) -> list[float]:
    """确定性哈希向量：同一文本得到同一向量，保证检索可复现。"""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    vec = []
    for i in range(dim):
        b = h[i % len(h)]
        # 用文本中字符构造伪随机相位
        phase = (b + sum(text.encode("utf-8")) + i * 7919) % (2 * math.pi * 100) / 100
        vec.append(math.sin(phase))
    return vec


def _mock_json(purpose: str, user: str) -> dict[str, Any]:
    """按节点用途返回固定结构化 JSON。"""
    if purpose == "parse":
        return _mock_parse(user)
    if purpose == "decompose":
        return _mock_decompose(user)
    if purpose == "review":
        return _mock_review(user)
    raise ValueError(f"mock 模式不支持 purpose={purpose!r}")


def _mock_parse(user: str) -> dict[str, Any]:
    # 模板里固定包含"用户原始输入"字样，必须先从模板中抽取真实的 raw_input 再判断
    raw = _extract_raw(user)
    # 判定信息是否充足：出现 "用户" 或 "目标" 视为充足，否则反问
    info_enough = ("用户" in raw) or ("目标" in raw)
    base = {
        "product_name": _extract_name(raw),
        "business_goal": "用一句业务目标（mock 生成，仅供验证流程）",
        "business_background": "业务背景（mock 生成）",
        "user_roles": ["普通用户", "管理员"],
        "constraints": ["工期 2 个月（mock）"],
        "missing_info": [],
        "questions": [],
        "assumptions": ["产品名称/目标为 mock 假设，需人工确认"],
        "is_info_enough": True,
    }
    if not info_enough:
        base.update(
            {
                "missing_info": ["business_goal", "user_roles"],
                "questions": [
                    {"field": "business_goal", "question": "这个产品的核心业务目标是什么？"},
                    {"field": "user_roles", "question": "主要使用用户角色有哪些？"},
                ],
                "is_info_enough": False,
            }
        )
    return base


def _extract_raw(user: str) -> str:
    """从 build_parse_user 模板中抽出「用户原始输入」原文。"""
    m = re.search(r"用户原始输入:\n(.*?)\n补充字段", user, re.S)
    return m.group(1).strip() if m else user


def _extract_name(user: str) -> str:
    m = re.search(r"(?:做一个|做个|开发|搭建|实现)\s*([^，。, \n\r]{1,20})", user)
    return m.group(1) if m else "未命名产品"


def _mock_decompose(user: str) -> dict[str, Any]:
    return {
        "business_background": "业务背景：为提升内部效率而构建（mock 素材，供验证流程）",
        "business_goal": "实现核心业务流程线上化，减少人工操作（mock）",
        "user_roles": [
            {
                "role": "普通用户",
                "description": "日常使用该产品的人",
                "stories": ["作为普通用户，我希望快速完成核心操作，以便节省时间"],
            },
            {
                "role": "管理员",
                "description": "负责配置与审核",
                "stories": ["作为管理员，我希望查看并审核所有记录，以便控制风险"],
            },
        ],
        "business_flow_mermaid": "graph TD\n  A[发起] --> B[审核]\n  B -->|通过| C[完成]\n  B -->|驳回| A",
        "functional_requirements": [
            {
                "module": "核心流程模块",
                "features": [
                    {
                        "id": "FR-1",
                        "name": "发起申请",
                        "summary": "用户填写信息并发起申请",
                        "user_story": "作为普通用户，我希望发起申请，以便完成业务",
                        "preconditions": ["用户已登录"],
                        "input": ["申请类型", "金额", "说明"],
                        "output": "生成一条待审核申请记录",
                        "steps": ["填写表单", "提交", "系统生成记录"],
                        "exception_flow": ["提交失败提示重试", "金额超出限额拦截"],
                        "acceptance_criteria": [
                            "提交成功后状态为待审核",
                            "金额超限时系统拦截并给出提示",
                        ],
                    }
                ],
            }
        ],
        "non_functional": {
            "performance": ["常规操作响应时间 < 2 秒"],
            "security": ["敏感数据加密存储，权限分级"],
            "compatibility": ["支持主流浏览器最新两个版本"],
            "compliance": ["符合公司数据安全规范"],
            "usability": ["核心流程 3 步内完成"],
        },
        "interfaces": [
            {"name": "用户认证接口", "description": "登录鉴权，概要设计", "direction": "入站"}
        ],
        "risks": [
            {"risk": "边界场景考虑不全", "impact": "高", "mitigation": "评审迭代补全异常流程"}
        ],
        "dependencies": ["企业微信集成（如涉及）"],
        "assumptions": ["验收指标为 mock 假设，需业务确认"],
    }


def _mock_review(user: str) -> dict[str, Any]:
    # 第一次评审总是通过（need_fix=False），保证演示时走完整流程；
    # 若用户消息里显式带有 “强制修正” 标记，则返回 need_fix=True 以演示迭代。
    if "强制修正" in user:
        return {
            "need_fix": True,
            "comments": [
                {
                    "severity": "high",
                    "issue": "mock 评审：要求补全异常分支",
                    "suggestion": "为每个功能点补充网络异常与权限越界场景",
                }
            ],
        }
    return {
        "need_fix": False,
        "comments": [
            {
                "severity": "low",
                "issue": "mock 评审：整体结构完整，可进入文档组装",
                "suggestion": "正式环境建议再人工复核假设项",
            }
        ],
    }
