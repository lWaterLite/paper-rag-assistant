"""应用配置。

当前子模块只使用标准库读取环境变量，不直接修改你的环境。
后续如果你需要接入真实 LLM、embedding 或向量数据库，可以在 README 中按说明自行配置。
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """RAG pipeline 的基础配置。

    这里故意只保留子模块 1 需要理解的配置项，避免一开始就陷入外部模型和数据库细节。
    """

    chunk_size: int = 500
    chunk_overlap: int = 80
    top_k: int = 3
    max_context_chars: int = 1800
    mock_embedding_dimension: int = 16
    require_citation: bool = True

    @classmethod
    def from_env(cls) -> "Settings":
        """从环境变量读取配置。

        TODO 练习 1：
        当前实现只做了最简单的 int 转换。
        请你补充更健壮的配置校验，例如：
        1. chunk_size 必须大于 0。
        2. chunk_overlap 必须小于 chunk_size。
        3. top_k 必须大于 0。
        4. max_context_chars 必须大于 0。
        """
        return cls(
            chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "500")),
            chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "80")),
            top_k=int(os.getenv("RAG_TOP_K", "3")),
            max_context_chars=int(os.getenv("RAG_MAX_CONTEXT_CHARS", "1800")),
            mock_embedding_dimension=int(os.getenv("RAG_MOCK_EMBEDDING_DIMENSION", "16")),
            require_citation=os.getenv("RAG_REQUIRE_CITATION", "true").lower() == "true",
        )

