"""PRD 生成 Agent —— FastAPI 入口。

启动：
    uv run uvicorn main:app --reload --port 8000
    # 或
    .\\.venv\\Scripts\\python.exe -m uvicorn main:app --reload --port 8000

交互式 API 文档：http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.config import settings
from app.deps import milvus

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("prdagent")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时尽力连接 Milvus（失败自动降级为无检索模式）
    if settings.milvus_uri:
        if milvus.connect():
            logger.info("Milvus 就绪，范例数=%s", milvus.count())
        else:
            logger.warning("Milvus 不可用，已降级为无检索模式（不影响 PRD 生成）")
    yield


app = FastAPI(
    title="PRD 生成 Agent",
    description="基于 LangGraph + FastAPI 的多节点 PRD 生成服务，支持交互/自动两种模式，Milvus 检索同类范例增强。",
    version="0.1.0",
    lifespan=lifespan,
)
app.include_router(router)


@app.get("/")
def root() -> dict:
    return {
        "service": "PRD 生成 Agent",
        "docs": "/docs",
        "health": "/api/health",
        "generate": "/api/prd/generate",
        "resume": "/api/prd/resume",
    }
