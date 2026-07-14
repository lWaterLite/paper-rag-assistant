"""Embedding 输出校验。"""

from __future__ import annotations

import math

from app.core.errors import AppError, ErrorCode


def validate_embedding_vectors(
    *,
    expected_count: int,
    vectors: list[list[float]],
    expected_dimension: int,
    context: str,
) -> None:
    """校验向量数量、维度与数值合法性。"""

    if len(vectors) != expected_count:
        raise AppError(
            ErrorCode.INDEX_FAILED,
            f"{context}数量不一致：期望 {expected_count} 个向量，实际 {len(vectors)} 个",
        )
    for index, vector in enumerate(vectors):
        if len(vector) != expected_dimension:
            raise AppError(
                ErrorCode.INDEX_FAILED,
                f"{context}维度不一致：第 {index} 个向量为 {len(vector)} 维，期望 {expected_dimension} 维",
            )
        if any(not math.isfinite(value) for value in vector):
            raise AppError(
                ErrorCode.INDEX_FAILED,
                f"{context}包含非法数值：第 {index} 个向量存在 NaN 或 Infinity",
            )
