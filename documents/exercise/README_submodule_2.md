# 子模块 2 练习：文档加载、解析与清洗

本练习目标：把子模块 1 中偏 demo 的 `LocalTextLoader + PlainTextParser`，升级为更接近真实工程的 ingestion 子系统。

当前代码已经实现了完整基础框架和主要功能，你需要在此基础上完成留下的代码题，并使用真实论文、Markdown、HTML 文件验证解析质量。

---

## 1. 本次工程改造内容

### 1.1 新增和改造的核心能力

本次已经完成：

- 统一本地文档加载器：
  - PDF
  - Markdown
  - HTML
  - TXT
- 统一文档身份：
  - `doc_id`
  - `content_hash`
  - `version_id`
- 结构化解析结果：
  - `ParsedBlock`
  - `ParseIssue`
  - `ParsedDocument.blocks`
  - `ParsedDocument.parse_issues`
- 文档清洗器：
  - 通用文本清洗
  - PDF 错误断行修复
  - PDF 重复页眉页脚基础检测
  - HTML 空白规范化
- 真实格式解析器：
  - `MarkdownParser`
  - `HtmlDocumentParser`
  - `PdfDocumentParser`
  - `PlainTextParser`
- ingestion pipeline：
  - 单个文件失败不会中断整个目录摄取
  - 成功文档和失败文件分开记录
  - 输出 `IngestionResult`
- 索引构建流程接入 ingestion pipeline。
- 新增 HTML 样例文件。
- 新增 ingestion 自检测试。

---

## 2. 当前重要文件

### 2.1 核心模型

位置：

```text
app/core/models.py
```

新增：

- `ParseIssue`
- `ParsedBlock`
- `BlockType`

改造：

- `RawDocument`
  - 新增 `raw_bytes`
  - 新增 `source_uri`
- `ParsedDocument`
  - 新增 `blocks`
  - 新增 `parse_issues`

注意：

- `RawDocument.raw_text` 不再总是有内容。
- PDF 这种二进制文件在 loading 阶段主要保存在 `raw_bytes` 中。
- parser 才负责把 PDF 字节转换成文本和 block。

### 2.2 Loader

位置：

```text
app/ingest/loaders.py
```

主要类：

- `DocumentSource`
- `DocumentLoader`
- `DocumentIdentityBuilder`
- `LocalDocumentLoader`
- `LocalTextLoader`

说明：

- 新代码优先使用 `LocalDocumentLoader`。
- `LocalTextLoader` 只是为了兼容子模块 1 的旧测试和旧代码。
- `LocalDocumentLoader` 会读取真实文件字节，并根据后缀判断 `file_type`。
- `LocalDocumentLoader` 必须显式接收 `LocalDocumentLoaderConfig`，不要在类内部私自创建配置。

### 2.2.1 Factory 统一组装

位置：

```text
app/factory.py
```

说明：

- `factory.py` 是当前项目的对象组装入口。
- `EnvSettings` 和 `ProjectSettings` 到各模块 `Config` 对象的转换在 factory 中完成。
- 生产路径不要直接裸写 `LocalDocumentLoader()`、`IndexBuilder(...)` 或 `RagPipeline(...)`。
- 底层类只声明自己需要哪些依赖，不负责猜默认依赖。
- 后续真实 embedding、真实 LLM、持久化向量库也应该从 factory 接入。

### 2.3 Cleaner

位置：

```text
app/ingest/cleaners.py
```

主要类：

- `CleanedText`
- `DocumentCleaner`
- `BasicTextCleaner`
- `PdfTextCleaner`
- `HtmlTextCleaner`

说明：

- 清洗器会返回文本、清洗 metadata 和 parse issues。
- PDF 清洗目前实现了保守规则，避免过度删除有用内容。

### 2.4 Parser

位置：

```text
app/ingest/parsers.py
```

主要类：

- `DocumentParser`
- `ParserRegistry`
- `PlainTextParser`
- `MarkdownParser`
- `HtmlDocumentParser`
- `PdfDocumentParser`

说明：

- `ParserRegistry` 根据 `RawDocument.file_type` 选择解析器。
- Markdown 会提取简单 frontmatter。
- HTML 会去掉 `nav`、`footer`、`script`、`style` 等噪声。
- PDF 会优先使用 PyMuPDF；如果没有，再尝试 pypdf。

### 2.5 Ingestion Pipeline

位置：

```text
app/ingest/pipeline.py
```

主要类：

- `IngestionFailure`
- `IngestedDocument`
- `IngestionResult`
- `IngestionPipeline`

说明：

- 目录不存在属于系统性错误，会抛出 `AppError`。
- 单个文件加载或解析失败会记录到 `failures`，不会让整批任务中断。

### 2.6 IndexBuilder 接入

位置：

```text
app/indexing/index_builder.py
```

变化：

- `IndexBuilder` 现在通过 `IngestionPipeline` 获取 `raw_documents` 和 `parsed_documents`。
- `IndexBuildResult` 新增 `ingestion_failures`。

---

## 3. 可选依赖说明

本练习没有替你安装任何依赖。

如果你要解析真实 PDF，建议安装：

```powershell
uv add "pymupdf>=1.24"
```

或：

```powershell
pip install "pymupdf>=1.24"
```

备用 PDF 解析库：

```powershell
uv add "pypdf>=4.2"
```

或：

```powershell
pip install "pypdf>=4.2"
```

项目中已经在 `pyproject.toml` 增加了可选依赖组：

```powershell
uv sync --extra ingest
```

是否执行由你自己决定。

---

## 4. 真实数据准备

建议目录结构：

```text
data/raw/papers/
  rag_intro_note.md
  rag_evaluation_note.md
  paper_project_page.html
  pdf/
    你的真实论文1.pdf
    你的真实论文2.pdf
    你的真实论文3.pdf
```

注意：

- 本练习已经提供 Markdown 和 HTML 样例。
- PDF 请你自行放入公开论文或本地论文文件。
- 不建议继续用随便生成的 mock 文本验证 ingestion。
- 每次新增真实文件后，都要运行索引和测试，观察解析质量。

---

## 5. 运行方式

### 5.1 运行测试

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

当前应通过：

```text
Ran 75 tests
OK
```

### 5.2 构建索引

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -m app.main index --source data/raw/papers
```

当前样例数据下，应该能看到 Markdown 和 HTML 都进入索引。

### 5.3 执行 mock RAG 问答

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe -m app.main ask "RAG evaluation 为什么需要 citations？" --source data/raw/papers
```

注意观察：

- 回答是否包含 citation。
- citation 来自哪个文件。
- HTML 样例是否可能被检索到。
- trace_id 是否输出。

---

## 6. 本次留下的代码题

只列代码任务，不再列思考题。

### TODO 子模块2-练习1：目录扫描策略

位置：

```text
app/ingest/loaders.py
```

任务：

- 改造 `LocalDocumentLoader.iter_supported_files()`。
- 支持调用方配置是否递归扫描子目录。当前这部分已经完成，配置从 `settings.toml` 进入 `ProjectSettings`，再由 `factory.py` 转换为 `LocalDocumentLoaderConfig`。
- 默认跳过隐藏目录和工程产物目录，例如：
  - `.git`
  - `.tmp_tests`
  - `__pycache__`
  - `data/indexes`
- 跳过常见临时文件，例如：
  - 以 `~$` 开头的 Office 临时文件。
  - 以 `.tmp` 结尾的临时文件。
- 保持扫描输出顺序稳定，避免同一批文档每次构建索引顺序不同。
- 不要把“是否支持某个文件类型”的判断塞进 parser，文件类型过滤仍然属于 loader 的发现阶段。

注意：

- `load_file()` 仍然只负责加载单个文件，不要让它承担目录扫描策略。
- 目录扫描策略应该放在 loader 初始化参数或独立配置对象中，而不是写死在循环内部。
- 不要让 `LocalDocumentLoader` import `EnvSettings`。
- 不要写 `config or LocalDocumentLoaderConfig()` 这类隐式默认配置。
- 测试代码和案例由我补充，你只需要专注功能实现。

### TODO 子模块2-练习2：PDF 页眉页脚检测配置化

位置：

```text
app/ingest/cleaners.py
```

任务：

- 把 `PdfTextCleaner._detect_repeated_edge_lines()` 改成可配置策略。
- 例如：
  - `edge_line_count`
  - `min_repeat_ratio`
  - `min_line_length`
- `max_line_length`

注意：

- 不要把整页中高频出现的普通术语误删。
- 只检测页面顶部和底部区域更安全。
- 测试代码和案例由我补充，你只需要专注功能实现。

### TODO 子模块2-练习3：IngestionReportWriter

位置：

```text
app/ingest/pipeline.py
```

任务：

- 实现 `IngestionReportWriter`。
- 将 `IngestionResult` 保存成 JSON。

建议字段：

```json
{
  "trace_id": "trace_xxx",
  "source_dir": "data/raw/papers",
  "succeeded": 3,
  "failed": 1,
  "documents": [
    {
      "doc_id": "doc_xxx",
      "version_id": "v_xxx",
      "title": "RAG Evaluation",
      "source_path": "data/raw/papers/example.pdf",
      "block_count": 42,
      "issue_count": 2
    }
  ],
  "failures": [
    {
      "source_path": "bad.pdf",
      "stage": "parsing",
      "error_code": "DOCUMENT_PARSE_FAILED",
      "error_message": "..."
    }
  ]
}
```

注意：

- 写文件属于副作用，不要放到 `EnvSettings`。
- 可以由 CLI 或索引构建入口显式调用。

### TODO 子模块2-练习4：真实 PDF 质量检查

位置：

```text
tests/
documents/exercise/
```

任务：

- 找至少 2 篇真实论文 PDF 放入 `data/raw/papers/pdf/`。
- 运行 ingestion 或 index。
- 记录每篇 PDF 的解析质量问题。

建议记录：

- 是否能提取正文。
- 页数。
- 文本长度。
- 是否有明显页眉页脚污染。
- 是否有双栏顺序问题。
- 是否有表格或公式乱码。
- citation 是否能追溯页码。

注意：

- 不要把大体积 PDF 提交到仓库，除非你明确希望这样做。
- 可以只在本地保留真实论文，用报告记录解析现象。

### TODO 子模块2-练习5：Markdown frontmatter 增强

位置：

```text
app/ingest/parsers.py
```

任务：

- 当前 frontmatter 只支持简单 `key: value`。
- 请增强对列表字段的支持，例如：

```yaml
authors:
  - Alice
  - Bob
tags:
  - rag
  - evaluation
```

注意：

- 可以继续手写轻量解析。
- 如果你希望更真实，可以引入 `PyYAML`，但需要先在依赖中声明并自行安装。

---

## 7. 验收标准

完成本子模块后，应达到：

- 至少支持 Markdown 和 PDF。
- HTML 支持作为增强项，目前代码已支持本地 HTML。
- 每个解析后的文档都有稳定 `doc_id`、`content_hash`、`version_id`。
- 每个解析结果尽量保留 block、section、page 信息。
- 单个坏文件不会导致整个目录 ingestion 失败。
- 至少有 5 个解析/加载/清洗相关测试。
- 能用真实论文 PDF 验证解析质量。
- 能说清 PDF 解析失败如何影响后续 RAG 检索和 citation。

---

## 8. 当前实现的边界

当前已经比子模块 1 的 demo 更接近真实工程，但仍有边界：

- PDF 解析依赖外部库，当前没有自动安装。
- 没有 OCR，因此扫描版 PDF 只能记录无法提取文本。
- HTML 正文抽取使用标准库实现，复杂网页可以后续接入 `BeautifulSoup` 或 `trafilatura`。
- Markdown frontmatter 暂时只支持简单键值。
- 表格和公式目前只保留文本，不做结构化解析。
- ingestion report 还没有持久化，需要你完成 TODO。

这些不是偷懒，而是刻意把工程分层打清楚：先让真实文档能稳定进入系统，再逐步提高解析质量。
