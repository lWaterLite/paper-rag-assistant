# 测试结构说明

`tests` 按被测系统边界组织，而不是按功能实现的先后顺序组织。这样在修改某个模块时，可以直接定位对应测试；跨越多个模块的行为则进入 `integration`。

```text
tests/
  core/          核心模型、元数据、Settings 与 Config
  ingest/        文档加载、解析、清洗、chunking
  indexing/      embedding、collection、repository、manifest、索引加载与构建
  retrieval/     vector、BM25、hybrid、pipeline、report、compare search
  interfaces/    CLI 与 API schema / handler 契约
  generation/    prompt 与回答生成器
  integration/   跨子系统的 RAG 工作流
```

## 放置规则

1. 测试只验证一个子系统的行为时，放入对应目录。
2. 测试通过公开接口组合两个及以上子系统时，放入 `integration`。
3. 不为了共享少量测试辅助函数创建通用 `utils` 目录；只有当多个测试包稳定复用同一套 fixture 或 fake 时，才新增明确命名的 `tests/support` 包。
4. 测试文件继续使用 `test_<被测能力>.py` 命名，不按实现类的内部私有方法命名。

## 运行方式

在项目根目录运行全部测试：

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -t .
```

运行一个测试子系统：

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests/retrieval -t .
```

运行一个测试文件：

```powershell
.\.venv\Scripts\python.exe -B -m unittest tests.retrieval.test_search_service
```
