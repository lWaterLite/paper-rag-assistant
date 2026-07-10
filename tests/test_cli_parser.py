"""命令行参数解析测试。"""

from __future__ import annotations

import unittest

from app.main import build_parser


class CliParserTest(unittest.TestCase):
    """验证 CLI 入口不会把可扩展策略提前写死。"""

    def test_search_accepts_hybrid_retriever(self) -> None:
        parser = build_parser()

        args = parser.parse_args(
            [
                "search",
                "faithfulness evaluation",
                "--use-existing-index",
                "--retriever",
                "hybrid",
            ]
        )

        self.assertEqual(args.command, "search")
        self.assertEqual(args.retriever, "hybrid")

    def test_search_accepts_external_registered_retriever_name(self) -> None:
        parser = build_parser()

        args = parser.parse_args(
            [
                "search",
                "RAG",
                "--use-existing-index",
                "--retriever",
                "external_dense_v2",
            ]
        )

        self.assertEqual(args.retriever, "external_dense_v2")


if __name__ == "__main__":
    unittest.main()
