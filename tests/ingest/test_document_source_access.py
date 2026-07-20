"""文档导入目录访问策略测试。"""

from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from app.core.errors import AppError, ErrorCode
from app.core.settings import (
    DocumentSourceAccessSettings,
    IngestionSettings,
    ProjectSettings,
)
from app.factory import ApplicationFactory
from app.ingest.loading import (
    DocumentSourceAccessConfig,
    DocumentSourceAccessService,
)


class DocumentSourceAccessServiceTest(unittest.TestCase):
    """验证受限导入入口只能访问显式允许的目录。"""

    def setUp(self) -> None:
        self._root = Path(".tmp_tests") / f"source_access_{uuid.uuid4().hex}"
        self._allowed_dir = self._root / "allowed"
        self._nested_dir = self._allowed_dir / "papers"
        self._outside_dir = self._root / "outside"
        self._nested_dir.mkdir(parents=True)
        self._outside_dir.mkdir(parents=True)
        self._file_path = self._allowed_dir / "not-a-directory.txt"
        self._file_path.write_text("test", encoding="utf-8")
        self._service = DocumentSourceAccessService(
            DocumentSourceAccessConfig(allowed_source_dirs=(self._allowed_dir,))
        )

    def tearDown(self) -> None:
        shutil.rmtree(self._root, ignore_errors=True)

    def test_resolve_directory_accepts_nested_allowed_directory(self) -> None:
        resolved = self._service.resolve_directory(self._nested_dir)

        self.assertEqual(resolved, self._nested_dir.resolve())

    def test_resolve_directory_rejects_path_outside_allowed_root(self) -> None:
        with self.assertRaises(AppError) as context:
            self._service.resolve_directory(self._outside_dir)

        self.assertEqual(context.exception.code, ErrorCode.DOCUMENT_LOAD_FAILED)
        self.assertIn("不在允许", context.exception.message)

    def test_resolve_directory_rejects_parent_path_escape(self) -> None:
        escaped_path = self._nested_dir / ".." / ".." / "outside"

        with self.assertRaises(AppError) as context:
            self._service.resolve_directory(escaped_path)

        self.assertEqual(context.exception.code, ErrorCode.DOCUMENT_LOAD_FAILED)
        self.assertIn("不在允许", context.exception.message)

    def test_resolve_directory_rejects_file_and_missing_directory(self) -> None:
        with self.assertRaises(AppError) as file_context:
            self._service.resolve_directory(self._file_path)
        with self.assertRaises(AppError) as missing_context:
            self._service.resolve_directory(self._allowed_dir / "missing")

        self.assertIn("不是目录", file_context.exception.message)
        self.assertIn("不存在", missing_context.exception.message)

    def test_resolve_directory_rejects_unreadable_directory(self) -> None:
        with patch("app.ingest.loading.access.os.scandir", side_effect=OSError("denied")):
            with self.assertRaises(AppError) as context:
                self._service.resolve_directory(self._nested_dir)

        self.assertEqual(context.exception.code, ErrorCode.DOCUMENT_LOAD_FAILED)
        self.assertIn("不可读取", context.exception.message)

    def test_factory_builds_access_service_from_project_settings(self) -> None:
        factory = ApplicationFactory(
            project_settings=ProjectSettings(
                ingestion=IngestionSettings(
                    access=DocumentSourceAccessSettings(
                        allowed_source_dirs=(self._allowed_dir,)
                    )
                )
            )
        )

        resolved = factory.ingestion.build_document_source_access_service().resolve_directory(
            self._nested_dir
        )

        self.assertEqual(resolved, self._nested_dir.resolve())


if __name__ == "__main__":
    unittest.main()
