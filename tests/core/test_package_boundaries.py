"""包级公共边界与惰性导入测试。"""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class PackageBoundaryTest(unittest.TestCase):
    """验证包根只暴露稳定契约，且不会预加载默认实现。"""

    def test_package_exports_are_limited_to_stable_contracts(self) -> None:
        from app.factory import __all__ as factory_exports
        from app.factory.configs import __all__ as config_factory_exports
        from app.indexing.embeddings import __all__ as embedding_exports
        from app.retrieval.context import __all__ as context_exports
        from app.retrieval.retrievers import __all__ as retriever_exports

        self.assertEqual(factory_exports, ["ApplicationFactory"])
        self.assertEqual(config_factory_exports, ["ConfigFactory"])
        self.assertEqual(
            embedding_exports,
            ["EmbeddingClient", "EmbeddingClientRegistry"],
        )
        self.assertEqual(
            context_exports,
            [
                "ContextPackRequest",
                "ContextPacker",
                "ContextPackerConfig",
                "PackedContext",
            ],
        )
        self.assertEqual(retriever_exports, ["Retriever", "RetrieverRegistry"])

    def test_importing_contract_packages_does_not_load_default_implementations(self) -> None:
        script = """
import sys
import app.indexing.embeddings
import app.retrieval.context
import app.retrieval.tokenizers
assert 'app.indexing.embeddings.mock' not in sys.modules
assert 'app.retrieval.context.token_estimators.regex' not in sys.modules
assert 'app.retrieval.tokenizers.regex' not in sys.modules
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stderr,
        )


if __name__ == "__main__":
    unittest.main()
