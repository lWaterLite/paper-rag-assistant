"""索引 manifest。

manifest 用来记录一次索引构建的关键配置和统计信息。
它的目标是让索引可以被解释、复现和比较。
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class IndexManifest:
    """索引构建清单。"""

    index_id: str
    source_dir: str
    created_at: str
    chunker: str
    chunk_size: int
    chunk_overlap: int
    embedding_provider: str
    embedding_model: str
    embedding_dimension: int
    document_count: int
    chunk_count: int
    vector_count: int
    document_versions: dict[str, str] = field(default_factory=dict)

    @classmethod
    def build(
            cls,
            *,
            source_dir: Path,
            chunker: str,
            chunk_size: int,
            chunk_overlap: int,
            embedding_provider: str,
            embedding_model: str,
            embedding_dimension: int,
            document_count: int,
            chunk_count: int,
            vector_count: int,
            document_versions: dict[str, str],
    ) -> "IndexManifest":
        """根据索引构建参数创建 manifest。"""

        index_id = _build_index_id(
            source_dir=source_dir,
            chunker=chunker,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_dimension=embedding_dimension,
        )
        return cls(
            index_id=index_id,
            source_dir=source_dir.as_posix(),
            created_at=datetime.now(UTC).isoformat(),
            chunker=chunker,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_dimension=embedding_dimension,
            document_count=document_count,
            chunk_count=chunk_count,
            vector_count=vector_count,
            document_versions=document_versions,
        )

    def to_dict(self) -> dict[str, Any]:
        """转换为可 JSON 序列化的 dict。"""

        return asdict(self)


def _build_index_id(
        *,
        source_dir: Path,
        chunker: str,
        chunk_size: int,
        chunk_overlap: int,
        embedding_provider: str,
        embedding_model: str,
        embedding_dimension: int,
) -> str:
    """生成稳定 index_id。

    这里暂时不把 document_versions 放进 index_id，避免一篇文档内容变化就让 ID 过长。
    后续做持久化索引时，可以把文档版本摘要也加入 index_id。
    """

    payload = "|".join(
        [
            source_dir.as_posix(),
            chunker,
            str(chunk_size),
            str(chunk_overlap),
            embedding_provider,
            embedding_model,
            str(embedding_dimension),
        ]
    )
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return f"idx_{digest}"
