# PRD 生成 Agent

基于 **LangGraph + FastAPI** 实现的多节点 PRD（产品需求文档）生成服务，可选 **Milvus** 向量库做同类 PRD 范例检索增强。

## 设计来源

实现对齐项目内《PRD生成Agent整体设计思路.md》：

```
用户输入 → 信息解析补全 → 业务拆解 → 需求校验评审(可迭代≤2轮) → 文档组装 → 输出 PRD
```

四个节点分别对应四个独立的 LLM 调用（评审与拆解分离，效果更好），并落实设计文档的 Prompt 技巧：
- 强制 JSON 结构化输出，禁止自由文本；
- AI 推断的信息一律标记【假设】，文档第 8 节集中列出供人工确认；
- 每个功能点强制携带：用户故事（作为[角色]…以便…）、前置条件、输入、输出、操作流程、**异常流程**、**验收标准**；
- 主业务流程以 Mermaid 输出。

## 目录结构

```
PRDAgent/
├── main.py                 # FastAPI 入口（uvicorn main:app）
├── app/
│   ├── config.py           # 环境变量配置（pydantic-settings）
│   ├── schemas.py          # API 请求/响应模型
│   ├── deps.py             # LLM 与 Milvus 全局单例
│   ├── llm.py              # OpenAI 兼容客户端 + mock 模式（离线验证）
│   ├── prompts.py          # 四节点提示词
│   ├── milvus_store.py     # Milvus 向量库（失败自动降级）
│   ├── render.py           # 确定性 PRD Markdown 渲染器
│   ├── graph/
│   │   ├── state.py        # LangGraph 状态
│   │   ├── nodes.py        # 节点函数（parse/retrieve/decompose/review/assemble）
│   │   └── build.py        # 图构建 + 交互中断/续跑封装
│   └── api/routes.py       # FastAPI 路由
├── scripts/seed_milvus.py  # 向量库种子脚本
├── seed/prd_examples.md    # 示例 PRD 库
└── .env.example            # 环境变量模板
```

## 安装

```powershell
uv sync            # 或 uv add ... 安装依赖
Copy-Item .env.example .env   # 然后按需编辑 .env
```

## 快速体验（零配置）

仓库已内置 `.env`（`LLM_PROVIDER=mock` + `MILVUS_URI=file:./milvus_lite.db`），无需任何外部服务即可跑通：

```powershell
uv run python -m scripts.seed_milvus      # 灌入示例 PRD（首次）
uv run python -m scripts.smoke_test        # 端到端冒烟测试（可选）
uv run uvicorn main:app --reload --port 8000
```

示例输出见 `examples/sample_prd.md`。接入真实模型时，把 `.env` 中 `LLM_PROVIDER` 改为 `openai` 并填写 `LLM_*` 即可。

## 配置（.env）

| 变量 | 说明 | 示例 |
| --- | --- | --- |
| `LLM_PROVIDER` | `openai` 或 `mock` | `openai` |
| `LLM_BASE_URL` | OpenAI 兼容服务地址 | `https://api.deepseek.com/v1` |
| `LLM_API_KEY` | API Key | |
| `LLM_MODEL` | 模型名 | `gpt-4o-mini` |
| `EMBEDDING_MODEL/DIM` | Milvus 检索用 embedding | `text-embedding-3-small` / `1536` |
| `MILVUS_URI` | Milvus 服务 或 Milvus Lite 本地文件 | `http://localhost:19530` 或 `file:./milvus_lite.db` |
| `MILVUS_FAIL_SOFT` | Milvus 不可用时是否静默降级 | `true` |
| `DEFAULT_MAX_ITERATIONS` | 评审迭代最大轮数 | `2` |

> 没有 LLM Key / Milvus 时：`LLM_PROVIDER=mock` + `MILVUS_FAIL_SOFT=true` 即可离线跑通全流程。

## 运行

```powershell
# 1) 启动 Milvus（本地调试可直接用 Milvus Lite URI）
# 2) 灌入示例 PRD（可选，仅当需要检索增强）
uv run python -m scripts.seed_milvus
# 3) 启动服务
uv run uvicorn main:app --reload --port 8000
```

交互式文档：http://127.0.0.1:8000/docs

## API

### POST /api/prd/generate —— 发起生成

```json
{
  "raw_input": "做一个内部报销系统，员工线上提交报销、财务审核，减少纸质单据；必须对接企业微信",
  "mode": "auto",
  "constraints": ["工期2个月", "预算有限"]
}
```

- `mode=auto`（默认）：信息不足自动补全假设，一次性返回完整 PRD。
- `mode=interactive`：信息不足时返回 `status=need_more_info` + `questions` + `thread_id`。

### POST /api/prd/resume —— 交互续跑

```json
{
  "thread_id": "<generate 返回的 thread_id>",
  "answers": { "business_goal": "员工线上提交报销并完成财务审核", "user_roles": "普通员工,财务审核员,管理员" }
}
```

### 响应字段

- `status`：`done` / `need_more_info` / `error`
- `prd_markdown`：标准 PRD Markdown（主输出）
- `prd_json`：结构化需求 JSON（供下游：转测试用例 Agent / 开发任务拆解 Agent）
- `assumptions`：AI 推断的【假设】项列表
- `retrieved_examples`：Milvus 检索到的同类 PRD 范例
- `milvus_available`：向量库是否可用

### GET /api/health —— 健康检查

## 两种运行模式（设计文档 6.3）

1. **对话交互模式（推荐给人用）**：`mode=interactive`，信息不全时通过 LangGraph `interrupt` 挂起图、反问用户，`resume` 回填后续跑。
2. **全自动模式（程序调用）**：`mode=auto`，不反问，全部自动假设并标记【假设】，直接输出完整文档，适合嵌入其他系统。

## 后续可延伸链路（设计文档 10）

```
PRD Agent → 测试用例生成 Agent → 开发任务拆解 Agent
```
