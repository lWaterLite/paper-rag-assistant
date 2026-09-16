"""JSONL 黄金评测集 Repository 测试。"""

from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from uuid import uuid4

from app.evaluation.datasets import JsonlEvaluationDatasetRepository


class JsonlEvaluationDatasetRepositoryTest(unittest.TestCase):
    """验证评测集读取、结构校验和样例身份约束。"""

    def setUp(self) -> None:
        self._temp_dir = Path(".tmp_tests") / f"evaluation_dataset_{uuid4().hex}"
        self._temp_dir.mkdir(parents=True)
        self._repository = JsonlEvaluationDatasetRepository()

    def tearDown(self) -> None:
        shutil.rmtree(self._temp_dir, ignore_errors=True)

    def test_loads_valid_jsonl_cases_and_ignores_comments(self) -> None:
        dataset_path = self._write_dataset(
            """# 评测集说明
{"id": "fact-001", "question": "论文的主要结论是什么？", "question_type": "fact", "expected_doc_ids": ["paper-a"], "tags": ["core"]}

{"id": "abstain-001", "question": "文档没有提到的结论是什么？", "question_type": "abstention", "expected_abstention": true}
"""
        )

        cases = self._repository.load(dataset_path)

        self.assertEqual([case.case_id for case in cases], ["fact-001", "abstain-001"])
        self.assertEqual(cases[0].expected_doc_ids, ("paper-a",))
        self.assertTrue(cases[1].expected_abstention)

    def test_rejects_non_abstention_case_without_expected_evidence(self) -> None:
        dataset_path = self._write_dataset(
            '{"id": "fact-001", "question": "结论是什么？", "question_type": "fact"}\n'
        )

        with self.assertRaisesRegex(ValueError, "取值无效"):
            self._repository.load(dataset_path)

    def test_rejects_duplicate_case_id(self) -> None:
        dataset_path = self._write_dataset(
            """{"id": "fact-001", "question": "问题一", "question_type": "fact", "expected_doc_ids": ["paper-a"]}
{"id": "fact-001", "question": "问题二", "question_type": "fact", "expected_doc_ids": ["paper-b"]}
"""
        )

        with self.assertRaisesRegex(ValueError, "重复 id"):
            self._repository.load(dataset_path)

    def test_rejects_wrong_field_type_with_type_error(self) -> None:
        dataset_path = self._write_dataset(
            '{"id": "fact-001", "question": "问题", "question_type": "fact", "expected_doc_ids": "paper-a"}\n'
        )

        with self.assertRaisesRegex(TypeError, "类型无效"):
            self._repository.load(dataset_path)

    def _write_dataset(self, content: str) -> Path:
        path = self._temp_dir / "dataset.jsonl"
        path.write_text(content, encoding="utf-8")
        return path


if __name__ == "__main__":
    unittest.main()
