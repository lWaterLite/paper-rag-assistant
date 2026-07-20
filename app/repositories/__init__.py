"""持久化 Repository 软件包。

Repository 实现必须从具体模块导入，例如
`app.repositories.vector`。本包不聚合导出实现，避免持久化层在
导入阶段反向初始化 indexing、manifest 等业务模块。
"""
