"""索引构建、索引恢复与索引运行时软件包。

调用方应从具体职责子包导入对象，例如 `app.indexing.pipeline` 或
`app.indexing.collections`。本包不聚合导出实现，避免导入 Collection 或
Repository 时触发 Builder、Manifest 等无关模块的初始化。
"""
