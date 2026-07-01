# 子模块 3 练习说明：Chunking 策略与 Metadata 设计

本练习把子模块 3 的 chunking 能力接入到真实 RAG 索引构建流程中，而不是单独写一个 demo 脚本。你会看到文档从 loading、parsing、cleaning 进入 chunking，再进入 embedding 和 vector index 的完整链路。

## 本次已经实现的内容

1. 新增可配置 chunking 子系统：
   - `CharacterChunker`：字符窗口 baseline。
   - `FixedTokenChunker`：基于轻量 token span 的固定 token 窗口切分。
   - `SectionAwareChunker`：优先使用解析阶段保留的 `ParsedBlock.section`、`page_start`、`page_end` 等结构化信息。
2. 新增运行时配置对象：
   - `ChunkerConfig` 是功能类真正接收的 Config。
   - `ChunkingSettings` 是从 `settings.toml` 读取的 Settings。
   - 二者在 `app/factory.py` 中完成转换，保持 Settings 与 Config 分离。
3. 新增 `settings.toml` 配置：
   - `[chunking]` 控制策略、窗口大小、重叠长度和 tokenizer。
   - `[chunking_report]` 控制 chunking 报告输出目录。
4. 新增 chunking 质量报告：
   - 默认输出到 `logs/chunking_report.json`。
   - 记录 chunk 数量、token 统计、缺失页码、缺失 section 等质量指标。
5. 已接入主索引构建流程：
   - `IndexBuilder` 会使用 factory 统一创建 chunker。
   - 构建索引时会写入 ingestion 报告和 chunking 报告。

## 主要代码位置

1. `app/ingest/chunkers.py`
   - chunking 策略、chunker 抽象、chunk metadata 构造。
2. `app/ingest/chunking_report.py`
   - chunking 质量报告生成与写入。
3. `app/core/settings.py`
   - TOML 结构化配置模型。
4. `app/factory.py`
   - Settings 到 Config 的转换，以及对象统一组装。
5. `app/indexing/index_builder.py`
   - chunking 接入索引构建主流程。
6. `tests/test_chunking.py`
   - chunking 子系统自检。
7. `tests/test_section_aware_chunker.py`
   - section-aware chunker 的行为自检。

## 运行方式

在项目根目录执行：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

如果你的环境还没有安装项目依赖，请根据项目依赖文件自行安装。这里不需要为了本练习额外添加新的第三方库。

## 配置示例

当前 `settings.toml` 中的默认配置如下：

```toml
[chunking]
strategy = "section_aware"
chunk_size = 600
chunk_overlap = 100
tokenizer = "char_approx"

[chunking_report]
output_dir = "logs"
```

可选策略：

1. `character`：适合做 baseline，对 metadata 利用较少。
2. `fixed_token`：适合观察 token 窗口和 overlap 对检索颗粒度的影响。
3. `section_aware`：适合论文、报告、Markdown 等有章节结构的文档，是当前推荐默认值。

## 练习任务

所有练习都已经以 `TODO 子模块3-练习X` 的形式写在代码中。练习只要求你改局部代码，不要求你重写完整功能。

### 练习 1：改进轻量 tokenizer

位置：`app/ingest/chunkers.py` 的 `_regex_token_spans`

目标：

1. 改进当前正则规则。
2. 让 `U.S.A.`、`3.14`、`2024-06` 这类文本尽量作为更合理的 token。
3. 不要改动 chunker 主流程。

建议自检：

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_chunking
```

### 练习 2：扩展 Markdown fallback section 识别

位置：`app/ingest/chunkers.py` 的 `_split_markdown_sections`

目标：

1. 当前 fallback 只识别 `# Introduction` 这种 Markdown 标题。
2. 请扩展为可以识别常见论文标题，例如 `1. Introduction`、`2 Related Work`。
3. 不要修改 `ParsedBlock` 主路径。

建议自检：

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_section_aware_chunker
```

### 练习 3：归一化参考文献 section

位置：`app/ingest/chunkers.py` 的 `_build_section_group`

目标：

1. 当一个 section group 中出现 `block_type == "reference"` 的块时，把该 group 的 section 统一设置为 `References`。
2. 只修改 section 决策，不要修改 `ParsedBlock` 数据模型。
3. 可以新增一条针对 `ParsedBlock` 的测试。

建议自检：

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_section_aware_chunker
```

### 练习 4：补充 section 级别报告统计

位置：`app/ingest/chunking_report.py` 的 `build_report`

目标：

1. 在 chunking 报告中新增 section 级别统计字段。
2. 可以从 `section_count` 开始，再考虑 `top_sections`。
3. 只扩展报告结构和对应测试，不要改动 chunker 主流程。

建议自检：

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_chunking
```

## 完成本练习后的验收标准

1. 可以通过 `settings.toml` 切换 chunking 策略。
2. 索引构建流程能够输出 `logs/chunking_report.json`。
3. chunk 中保留必要 metadata：
   - `doc_id`
   - `version_id`
   - `source_path`
   - `section`
   - `page_start`
   - `page_end`
   - `char_start`
   - `char_end`
4. 改动 TODO 后，相关测试仍然通过。
5. 你能根据报告判断 chunk 是否过长、过短、缺失页码或缺失 section。

