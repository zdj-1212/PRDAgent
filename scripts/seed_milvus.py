"""Milvus 种子脚本：把 seed/prd_examples.md 中的 PRD 范例向量化写入 Milvus。

用法（在项目根目录）：
    uv run python -m scripts.seed_milvus
    # 或
    .\\.venv\\Scripts\\python.exe -m scripts.seed_milvus
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from app.deps import llm, milvus
from app.config import settings

SEED_FILE = Path(__file__).resolve().parent.parent / "seed" / "prd_examples.md"


def load_chunks(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    # 按 "## " 二级标题切块，去掉标题行本身与首部说明
    parts = re.split(r"\n##\s+", text)
    chunks = []
    for p in parts:
        lines = [ln.strip() for ln in p.splitlines() if ln.strip()]
        if len(lines) < 3:  # 跳过说明段
            continue
        body = "\n".join(lines[1:])
        chunks.append(f"{lines[0]}\n{body}")
    return chunks


def main() -> int:
    if not milvus.connect():
        print(f"[FAIL] Milvus 不可用（uri={settings.milvus_uri}）", file=sys.stderr)
        return 1
    chunks = load_chunks(SEED_FILE)
    print(f"读取到 {len(chunks)} 条 PRD 范例，开始向量化写入...")
    n = milvus.add_texts(chunks)
    print(f"写入完成，集合 {milvus.collection} 现有 {milvus.count()} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
