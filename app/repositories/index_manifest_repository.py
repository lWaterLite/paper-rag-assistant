"""索引 manifest 持久化 Repository。"""

from __future__ import annotations

import json
from pathlib import Path

from app.indexing.configs import IndexBuilderConfig
from app.indexing.manifest import IndexManifest


class IndexManifestRepository:
    """负责 IndexManifest 与本地 JSON 文件之间的读写。"""

    def __init__(self, index_dir: Path, config: IndexBuilderConfig) -> None:
        self._index_dir = index_dir
        self._config = config

    @property
    def manifest_path(self) -> Path:
        """manifest 文件路径。"""

        return self._index_dir / self._config.manifest_filename

    def write(self, manifest: IndexManifest) -> Path:
        """写入 manifest 并返回文件路径。"""

        self.manifest_path.write_text(
            json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return self.manifest_path

    def read(self) -> IndexManifest:
        """读取 manifest。"""

        data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return IndexManifest.from_dict(data)

    def exists(self) -> bool:
        """判断 manifest 是否存在。"""

        return self.manifest_path.exists()
