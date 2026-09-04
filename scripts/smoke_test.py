"""端到端验证脚本：用 FastAPI TestClient + mock LLM + Milvus Lite 跑通全流程。

覆盖：
1. 健康检查（Milvus 可用）
2. 自动模式生成（含 Milvus 检索 + 假设项）
3. 交互模式：信息不足 -> 反问 -> resume 续跑
4. 自动模式：信息不足时自动补全假设
5. 评审迭代循环（monkeypatch 强制第一轮评审 need_fix）
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from main import app


def section(title: str) -> None:
    print(f"\n{'=' * 20} {title} {'=' * 20}")


def main() -> None:
    with TestClient(app) as client:  # 触发 lifespan，连接 Milvus
        # 1. 健康检查
        section("1. GET /api/health")
        r = client.get("/api/health")
        print(r.status_code, r.json())
        assert r.json()["milvus_available"] is True, "Milvus 应在 lifespan 中连接成功"

        # 2. 自动模式
        section("2. POST /api/prd/generate (auto)")
        r = client.post(
            "/api/prd/generate",
            json={
                "raw_input": "做一个内部报销系统，目标是员工线上提交报销、财务审核，减少纸质单据；必须对接企业微信",
                "mode": "auto",
                "constraints": ["工期2个月", "预算有限"],
            },
        )
        body = r.json()
        print("status =", body["status"])
        print("product_name =", body["product_name"])
        print("milvus_available =", body["milvus_available"])
        print("retrieved_examples count =", len(body["retrieved_examples"]))
        print("assumptions =", body["assumptions"])
        print("iterations_used =", body["iterations_used"])
        assert body["status"] == "done"
        assert body["product_name"] == "内部报销系统"
        assert "## 5 功能需求" in body["prd_markdown"]
        print("PRD markdown 长度 =", len(body["prd_markdown"]))
        if body["retrieved_examples"]:
            print("检索到范例: ", body["retrieved_examples"][0][:60], "...")

        # 3. 交互模式（信息不足 -> 反问）
        section("3. POST /api/prd/generate (interactive, 信息不足)")
        r = client.post(
            "/api/prd/generate",
            json={"raw_input": "做一个报销系统", "mode": "interactive"},
        )
        body = r.json()
        print("status =", body["status"])
        assert body["status"] == "need_more_info", f"应反问，实际 {body['status']}"
        print("thread_id =", body["thread_id"])
        for q in body["questions"]:
            print(f"  Q[{q['field']}] {q['question']}")
        tid = body["thread_id"]

        # 4. 交互续跑
        section("4. POST /api/prd/resume (交互续跑)")
        r = client.post(
            "/api/prd/resume",
            json={
                "thread_id": tid,
                "answers": {
                    "business_goal": "员工线上提交报销并完成财务审核",
                    "user_roles": "普通员工,财务审核员,管理员",
                },
            },
        )
        body = r.json()
        print("status =", body["status"])
        print("product_name =", body["product_name"])
        assert body["status"] == "done"
        assert "补充字段" not in body["prd_markdown"], "产品名不应混入模板文字"
        print("PRD markdown 长度 =", len(body["prd_markdown"]))

        # 5. 自动模式（信息不足自动假设）
        section("5. POST /api/prd/generate (auto, 信息不足自动假设)")
        r = client.post(
            "/api/prd/generate",
            json={"raw_input": "做一个内部工具系统", "mode": "auto"},
        )
        body = r.json()
        print("status =", body["status"])
        print("assumptions =", body["assumptions"])
        assert body["status"] == "done"

    # 6. 评审迭代循环（单独跑，monkeypatch 强制第一轮评审 need_fix=True）
    section("6. 评审迭代循环（强制第一轮 need_fix）")
    import app.graph.nodes as nodes_mod

    real_chat_json = nodes_mod.llm.chat_json
    counts = {"decompose": 0}
    forced = {"done": False}

    def fake(system: str, user: str, purpose: str = ""):
        if purpose == "decompose":
            counts["decompose"] += 1
        out = real_chat_json(system, user, purpose=purpose)
        if purpose == "review" and not forced["done"]:
            forced["done"] = True
            out = {
                "need_fix": True,
                "comments": [
                    {
                        "severity": "high",
                        "issue": "测试：强制要求补全异常分支",
                        "suggestion": "为功能点补充权限越界与网络异常场景",
                    }
                ],
            }
        return out

    try:
        nodes_mod.llm.chat_json = fake
        r = client.post(
            "/api/prd/generate",
            json={"raw_input": "做一个订单管理系统，目标是自动同步多平台订单", "mode": "auto"},
        )
        body = r.json()
        print("decompose 调用次数 =", counts["decompose"])
        print("iterations_used =", body["iterations_used"])
        assert counts["decompose"] >= 2, "need_fix 后应回到 decompose 再拆解一次"
        assert body["status"] == "done"
    finally:
        nodes_mod.llm.chat_json = real_chat_json

    print("\n全部通过 ✔")


if __name__ == "__main__":
    main()
