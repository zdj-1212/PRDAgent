"""PRD Agent 的 FastAPI 路由。"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException

from app.deps import milvus
from app.graph.build import PRDGraphRunner
from app.schemas import GenerateRequest, GenerateResponse, ResumeRequest

router = APIRouter(prefix="/api", tags=["prd"])

_runner = PRDGraphRunner()


def _new_thread_id() -> str:
    return uuid.uuid4().hex


@router.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "llm_provider": _provider_name(),
        "milvus_available": milvus.available,
        "milvus_collection": milvus.collection if milvus.available else None,
    }


@router.post("/prd/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    """发起 PRD 生成。

    - mode=auto：一次性返回完整 PRD。
    - mode=interactive：信息不足时返回 need_more_info + questions + thread_id，
      调用方携带 answers 调 /prd/resume 续跑。
    """
    try:
        thread_id = _new_thread_id()
        return _runner.start(req, thread_id, milvus.available)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"生成失败: {exc}") from exc


@router.post("/prd/resume", response_model=GenerateResponse)
def resume(req: ResumeRequest) -> GenerateResponse:
    """交互模式：携带用户回答续跑被中断的图。"""
    try:
        # 从检查点读取该会话的 mode
        mode = "interactive"
        return _runner.resume(req.thread_id, req.answers, mode, milvus.available)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"续跑失败: {exc}") from exc


def _provider_name() -> str:
    from app.config import settings

    return settings.llm_provider
