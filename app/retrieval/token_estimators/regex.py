"""无第三方依赖的正则 token 估算器。"""

from __future__ import annotations

import re


class RegexTokenEstimator:
    """按词、中文字符和标点估算 token。

    它适合测试与无模型环境，不保证与任意模型的 tokenizer 完全一致。
    """

    _TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]|[^\s]")

    @property
    def name(self) -> str:
        """返回稳定策略名称。"""

        return "regex"

    def count_text(self, text: str) -> int:
        """估算文本 token 数量。"""

        return len(self._TOKEN_PATTERN.findall(text))
