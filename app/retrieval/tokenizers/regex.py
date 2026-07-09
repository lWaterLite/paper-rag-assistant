"""基于正则表达式的内置分词器。"""

from __future__ import annotations

import re


class RegexTokenizer:
    """无第三方依赖的轻量分词器。

    英文、数字和下划线按连续词提取，中文按单字提取。
    """

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """使用固定正则规则切分文本。"""

        return re.findall(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]", text.lower())
