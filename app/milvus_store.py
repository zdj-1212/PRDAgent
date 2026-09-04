"""Milvus 向量库封装：存储 / 检索同类 PRD 范例，作为业务拆解节点的知识增强。

设计要点：
- 双后端（自动识别 uri）：
    http://host:19530  / https://...   → pymilvus MilvusClient（连接 Milvus 服务）
    file:./x.db 或 ./x.db（无协议头）    → milvus_lite.MilvusLite（本地文件，免部署）
- 失败自动降级：不可用时 available=False，检索返回空列表，不影响主流程。
- 所有操作惰性初始化 + 幂等，避免重复建集合。
"""
from __future__ import annotations

import logging
from typing import Any

from app.config import Settings
from app.llm import LLMClient

logger = logging.getLogger("prdagent.milvus")

_TEXT_MAX_LEN = 4096


class MilvusStore:
    def __init__(self, settings: Settings, llm: LLMClient):
        self.settings = settings
        self.llm = llm
        self.collection = settings.milvus_collection
        self.dim = settings.embedding_dim
        self._client: Any = None  # MilvusClient 或 MilvusLite
        self._col: Any = None  # 集合句柄（MilvusLite 模式）
        self._mode: str = ""  # "server" | "lite"
        self._init_error: str | None = None

    # ---------------- 生命周期 ----------------
    @property
    def available(self) -> bool:
        return self._client is not None

    def _is_lite(self) -> bool:
        uri = self.settings.milvus_uri
        return uri.startswith("file:") or "://" not in uri

    def connect(self) -> bool:
        if self._client is not None:
            return True
        try:
            if self._is_lite():
                self._connect_lite()
            else:
                self._connect_server()
            return True
        except Exception as exc:  # noqa: BLE001 - 降级路径需要吞掉所有异常
            self._client = None
            self._col = None
            self._init_error = str(exc)
            if self.settings.milvus_fail_soft:
                logger.warning(
                    "Milvus 连接失败，已降级为无检索模式（fail_soft）。原因: %s", exc
                )
            else:
                raise
            return False

    def _connect_server(self) -> None:
        from pymilvus import MilvusClient

        self._mode = "server"
        client = MilvusClient(uri=self.settings.milvus_uri)
        if not client.has_collection(self.collection):
            client.create_collection(
                collection_name=self.collection,
                dimension=self.dim,
                auto_id=True,
                metric_type="COSINE",
            )
            logger.info("已创建 Milvus 集合 %s (dim=%s)", self.collection, self.dim)
        self._client = client
        logger.info("Milvus 服务连接成功: %s", self.settings.milvus_uri)

    def _connect_lite(self) -> None:
        from milvus_lite import CollectionSchema, DataType, FieldSchema, MilvusLite

        self._mode = "lite"
        data_dir = self.settings.milvus_uri
        if data_dir.startswith("file:"):
            data_dir = data_dir[len("file:"):]
        db = MilvusLite(data_dir)
        if not db.has_collection(self.collection):
            schema = CollectionSchema(
                fields=[
                    FieldSchema("id", DataType.INT64, is_primary=True, auto_id=True),
                    FieldSchema("vector", DataType.FLOAT_VECTOR, dim=self.dim),
                    FieldSchema("text", DataType.VARCHAR, max_length=_TEXT_MAX_LEN),
                ]
            )
            db.create_collection(self.collection, schema)
            logger.info("已创建 Milvus Lite 集合 %s (dim=%s)", self.collection, self.dim)
        self._col = db.get_collection(self.collection)
        self._client = db
        logger.info("Milvus Lite 连接成功: %s", data_dir)

    # ---------------- 写入 ----------------
    def add_texts(self, texts: list[str]) -> int:
        if not self.connect() or not texts:
            return 0
        try:
            vectors = self.llm.embed(texts)
            rows = [
                {"vector": vec, "text": text}
                for vec, text in zip(vectors, texts)
            ]
            if self._mode == "lite":
                self._col.insert(rows)
                n = len(rows)
            else:
                res = self._client.insert(
                    collection_name=self.collection, data=rows
                )
                n = res.get("insert_count", len(rows)) if isinstance(res, dict) else len(rows)
            logger.info("已写入 %s 条 PRD 范例", n)
            return int(n)
        except Exception as exc:  # noqa: BLE001
            logger.warning("写入 Milvus 失败（跳过）: %s", exc)
            return 0

    # ---------------- 检索 ----------------
    def search(self, query: str, top_k: int | None = None) -> list[str]:
        if not self.connect():
            return []
        top_k = top_k or self.settings.milvus_top_k
        try:
            vector = self.llm.embed([query])[0]
            if self._mode == "lite":
                hits = self._col.search(
                    query_vectors=[vector],
                    top_k=top_k,
                    metric_type="COSINE",
                    output_fields=["text"],
                )
                results: list[str] = []
                for hit in hits[0]:
                    entity = hit.get("entity") or {}
                    text = entity.get("text") or ""
                    if text:
                        results.append(text[:800])
                return results
            hits = self._client.search(
                collection_name=self.collection,
                data=[vector],
                limit=top_k,
                output_fields=["text"],
                search_params={"metric_type": "COSINE"},
            )
            results = []
            for hit in hits[0]:
                entity = hit.get("entity") or {}
                text = entity.get("text") or ""
                if text:
                    results.append(text[:800])
            return results
        except Exception as exc:  # noqa: BLE001
            logger.warning("Milvus 检索失败（跳过）: %s", exc)
            return []

    def count(self) -> int:
        if not self.connect():
            return 0
        try:
            if self._mode == "lite":
                stats = self._client.get_collection_stats(self.collection)
                return int(stats.get("row_count", 0))
            stats = self._client.get_collection_stats(self.collection)
            return int(stats["row_count"])
        except Exception:  # noqa: BLE001
            return 0
