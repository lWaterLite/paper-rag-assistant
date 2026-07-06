# 子模块 3 练习说明：Chunking 策略与 Metadata 设计

本练习对应模块 2 的子模块 3，主题是 RAG 系统中的 chunking 策略与 metadata 设计。

这份文档的定位已经调整：它不是只列出 TODO，也不是要求你脱离代码去写架构说明。它会先仔细讲解本次生成的工程代码，让你从代码中学习结构、分层、范式和实现方式；然后再给出模块级代码练习，让你练习“添加一个工程部件”或“改造一个架构点”，而不是陷入某个正则或某个字段统计的细节。

## 学习目标

完成本子模块后，你应该能理解：

1. RAG 系统为什么需要 chunking，以及 chunking 对检索质量、引用质量和索引成本的影响。
2. 一个 chunking 子系统应该如何分层：策略、配置、工厂、报告、索引构建流程分别负责什么。
3. 为什么功能类接收 `Config`，而不是直接读取 `settings.toml` 或持有整个 `ProjectSettings`。
4. 为什么 chunk metadata 是 RAG 系统的工程契约，而不是随手塞进去的附加信息。
5. 如何为 chunking 策略扩展、质量检查、实验比较预留工程结构。

## 当前代码全局结构

本次生成的代码不是单脚本 demo，而是直接接入已有 RAG 工程主流程。

核心文件如下：

1. `app/ingest/chunking/strategies.py`
   - 定义 chunking 策略、chunker 抽象、具体 chunker 实现、chunk metadata 构造逻辑。
2. `app/core/metadata.py`
   - 定义跨领域复用的结构化 metadata 基类。
3. `app/ingest/chunking/metadata.py`
   - 定义 chunking 领域的 metadata 契约和构造器。
4. `app/ingest/chunking/registry.py`
   - 定义 chunker 策略注册表和默认内置策略注册入口。
5. `app/ingest/chunking/report.py`
   - 定义 chunking 质量报告 writer，把 chunking 结果转成可检查的 JSON 报告。
6. `app/ingest/chunking/quality.py`
   - 定义 chunking 质量检查器，把 chunking 结果转换成质量验收结果。
7. `app/ingest/chunking/__init__.py`
   - 定义 chunking 子系统的公开导出入口。
8. `app/core/settings.py`
   - 定义从 `settings.toml` 读取的结构化 Settings。
9. `settings.toml`
   - 保存非敏感、结构化、工程运行相关配置。
10. `app/factory.py`
   - 作为 composition root，把 Settings 转换成各功能模块真正接收的 Config，并统一组装对象。
11. `app/indexing/index_builder.py`
   - 离线索引构建主流程，负责串联 ingestion、chunking、embedding、vector store、manifest 和 report。
12. `tests/test_chunking.py`
   - 测试 chunking 策略选择、token 切分行为和 report 输出。
13. `tests/test_metadata.py`
   - 测试通用 metadata 基类和 chunk metadata 构造器。
14. `tests/test_section_aware_chunker.py`
   - 测试 section-aware chunker 如何保留 section、page、char offset 等 metadata。

整体数据流如下：

```text
RawDocument
  -> Parser / Cleaner
  -> ParsedDocument
  -> Chunker
  -> DocumentChunk
  -> EmbeddingClient
  -> VectorCollection
  -> Retriever / Citation / Answer
```

chunking 位于 parsing/cleaning 之后、embedding 之前。它不应该负责读取文件、不应该负责调用 embedding、不应该负责写 vector collection 或 repository。它只负责把 `ParsedDocument` 转成 `DocumentChunk`，并尽量保留对检索和引用有价值的 metadata。

## 配置结构讲解

### `settings.toml`

当前 chunking 配置位于：

```toml
[chunking]
strategy = "section_aware"
chunk_size = 600
chunk_overlap = 100
tokenizer = "char_approx"

[chunking_report]
output_dir = "logs"
```

这里选择放在 `settings.toml`，而不是 `.env`，原因是这些配置不是密钥，也不是强环境相关的部署参数。它们更像项目运行策略，适合用结构化配置文件统一管理。

### `ChunkingSettings`

`ChunkingSettings` 位于 `app/core/settings.py`，它的职责是描述 TOML 文件中的配置形状，并完成配置校验。

它校验了几个关键约束：

1. `strategy` 只能是允许的策略名称。
2. `chunk_size` 必须大于 0。
3. `chunk_overlap` 必须大于等于 0。
4. `chunk_overlap` 必须小于 `chunk_size`。

这属于 Settings 层职责：负责外部配置读取和校验。

### `ChunkerConfig`

`ChunkerConfig` 位于 `app/ingest/chunking/strategies.py`，它是 chunker 真正接收的运行时配置。

这里保留 Settings 和 Config 的分离，是一个很重要的工程范式：

1. Settings 面向外部配置文件。
2. Config 面向功能模块运行。
3. factory 负责把 Settings 转换为 Config。
4. 功能类不直接依赖外部配置系统。

这样做的好处是：

1. chunker 可以在测试中直接使用 `ChunkerConfig(...)` 构造，不需要真实 TOML 文件。
2. 如果以后配置来源从 TOML 改成数据库、远程配置中心或 API 参数，chunker 不需要改。
3. 功能类不会拿到整个 `ProjectSettings`，避免配置依赖扩散。

## `app/ingest/chunking/strategies.py` 代码讲解

### 类型别名

```python
ChunkingStrategy = str
TokenizerName = Literal["char_approx", "simple_regex"]
```

`ChunkingStrategy` 当前是开放字符串。这样外部模块可以注册新的 chunker 策略名，例如 `semantic`、`layout_aware`。

策略名称是否合法不再由 `Literal` 静态限制，而是交给 `ChunkerRegistry` 检查。`TokenizerName` 仍然是封闭 `Literal`，因为当前 tokenizer 仍然只有内置选项。

### `ChunkerConfig`

`ChunkerConfig` 是不可变 dataclass：

```python
@dataclass(frozen=True)
class ChunkerConfig:
    strategy: ChunkingStrategy = "section_aware"
    chunk_size: int = 600
    chunk_overlap: int = 100
    tokenizer: TokenizerName = "char_approx"
```

它使用 `__post_init__` 做跨字段校验。这里的重点不是校验本身，而是让功能模块拿到的配置一定是合法的。

这和 `ChunkingSettings` 的校验有一点重复，但职责不同：

1. `ChunkingSettings` 保护外部配置入口。
2. `ChunkerConfig` 保护功能模块入口。

在真实工程中，越底层、越核心的模块越应该保护自己的输入边界。

### `TextWindow`

`TextWindow` 表示一次窗口切分结果：

```python
@dataclass(frozen=True)
class TextWindow:
    text: str
    char_start: int
    char_end: int
```

它不只是返回字符串，还保留了 `char_start` 和 `char_end`。这是为了让 chunk 能追溯到原始文档位置。

如果只返回字符串，后续 citation、调试和质量检查都会变弱。

### `SectionGroup`

`SectionGroup` 是 section-aware chunker 的中间结构。

它把多个 `ParsedBlock` 聚合成一个 section 级别的文本组，同时保留：

1. section 名称。
2. section 文本。
3. 起止页码。
4. 起止字符偏移。
5. block 数量。

这体现了一个重要工程思路：复杂流程不要直接在一个循环里把所有事做完。先引入中间模型，把“按 section 聚合”和“在 section 内切分”拆开，代码会更容易测试和扩展。

### `_build_chunk_id`

`_build_chunk_id` 使用 `version_id`、`chunk_index` 和 `chunk_text` 生成稳定 ID。

稳定 chunk ID 很重要，因为它会影响：

1. embedding cache 是否能命中。
2. vector store 是否能判断 chunk 已经存在。
3. citation 是否能稳定引用同一个 chunk。
4. 重新构建索引时是否能复用旧结果。

这里没有使用随机 ID，因为 chunk 是索引构建产物，不是一次请求的临时事件。

### `estimate_token_count`

当前有两种 token 估算方式：

1. `char_approx`：直接用字符数近似。
2. `simple_regex`：用轻量正则切分 token。

真实工程中，最终应该使用 embedding provider 对应的 tokenizer。但在当前学习阶段，先保留无额外依赖的 fallback，可以让 chunking 流程完整运行。

### `_regex_token_spans`

这个函数是轻量 tokenizer fallback。它会优先匹配：

1. `U.S.A.` 这类英文缩写。
2. `3.14`、`2024-06` 这类数字组合。
3. 普通英文词。
4. 普通整数。
5. CJK 单字。
6. 其他非空白符号。

工程重点不是正则本身，而是规则顺序：更特殊的模式放前面，兜底规则放最后。

### `_split_text_windows_by_chars`

这是字符窗口切分的通用 helper。它负责：

1. 根据 `chunk_size` 和 `chunk_overlap` 计算窗口。
2. 去掉窗口前后的空白。
3. 保留窗口在原始文本中的字符偏移。

把这个逻辑抽成 helper，是因为 `CharacterChunker` 和 `SectionAwareChunker` 都需要字符窗口切分。复用 helper 比让 `SectionAwareChunker` 调用 `CharacterChunker` 的 protected 方法更干净。

### `Chunker` 抽象基类

`Chunker` 是所有 chunker 的抽象基类。

它做两件事：

1. 规定所有 chunker 都必须实现 `split(document)`。
2. 提供 `_build_chunk(...)`，统一构造 `DocumentChunk`。

`_build_chunk` 是一个非常关键的设计点。它避免每个 chunker 自己拼 metadata，从而保证不同策略产出的 chunk 在结构上保持一致。

如果没有这个统一入口，不同 chunker 很容易出现字段缺失、字段命名不一致、metadata 结构漂移等问题。

当前 `_build_chunk(...)` 不再直接手写 metadata 字典，而是通过 `ChunkMetadata` 和 `ChunkMetadataBuilder` 完成结构化构造。这样 `Chunker` 仍然负责创建 `DocumentChunk`，但 metadata 字段契约被拆到了独立模块中。

### `CharacterChunker`

`CharacterChunker` 是 baseline 策略。

它直接对整篇文档按字符窗口切分，优点是简单、稳定、容易理解；缺点是不了解章节、页码和语义边界。

在工程中，baseline 很重要。它不一定是最终策略，但它可以作为对照组，用于判断更复杂策略是否真的带来收益。

### `FixedTokenChunker`

`FixedTokenChunker` 先用 `_regex_token_spans` 得到 token span，再按 token 数量切分。

它比字符切分更接近 embedding 模型的输入限制，因为很多模型的限制是 token 数，而不是字符数。

它的主流程是：

1. 把文本切成 token span。
2. 根据 `chunk_size` 和 `chunk_overlap` 计算 token 窗口。
3. 根据 token span 反推字符区间。
4. 构造 `DocumentChunk`。
5. 在 metadata 中记录 `token_start` 和 `token_end`。

这里体现了一个重要原则：即使策略内部按 token 工作，对外仍然输出统一的 `DocumentChunk`。

### `SectionAwareChunker`

`SectionAwareChunker` 是当前推荐默认策略。

它优先使用解析阶段提供的 `ParsedBlock`，因为 `ParsedBlock` 里已经包含 section、页码、block 类型、字符位置等结构化信息。

它的主流程是：

1. 如果文档没有文本，直接返回空列表。
2. 如果文档有 `blocks`，按 section 聚合成 `SectionGroup`。
3. 如果文档没有 `blocks`，退回到 Markdown 标题切分。
4. 在每个 section 内部再做字符窗口切分。
5. 构造 chunk 时写入 section、page、char offset 和 block count。

这类策略非常适合论文 RAG，因为论文问答往往需要稳定引用页码和章节。

## `app/ingest/chunking/registry.py` 代码讲解

### `ChunkerRegistry`

`ChunkerRegistry` 根据配置创建具体 chunker。

它维护“策略名称 -> chunker 类”的映射。内置策略通过 `build_default_chunker_registry()` 统一注册。

当前创建过程是：

```python
registry = build_default_chunker_registry()
chunker = registry.create(config)
```

在 `factory.py` 中，registry 不是固定写死在流程里，而是可以通过参数注入：

```python
chunker = build_configured_chunker(
    project_settings,
    chunker_registry=custom_registry,
)
```

如果不传 `chunker_registry`，系统会使用 `build_default_chunker_registry()` 创建内置 registry。如果外部模块已经注册了自定义 chunker，可以把那份 registry 传进来。

这里的工程重点是：`IndexBuilder` 不知道也不关心具体 chunker 类，它只接收一个实现了 `Chunker` 接口的对象。

## `app/core/metadata.py` 代码讲解

`metadata.py` 放在 `app/core` 中，因为 `BaseMetadata` 不是 chunking 专属能力。未来 loader、parser、retrieval、generation、evaluation 都可能需要结构化 metadata。

### `BaseMetadata`

`BaseMetadata` 是一个很薄的基类，只提供通用序列化能力：

```python
metadata.to_dict()
```

默认会过滤值为 `None` 的字段。这样上层可以用 dataclass 明确声明 metadata 契约，同时最终仍然输出普通 `dict[str, Any]`，兼容现有 `DocumentChunk.metadata`。

这个基类刻意不包含 chunking、retrieval、citation 等业务逻辑。它只知道如何把结构化字段转成字典。

## `app/ingest/chunking/metadata.py` 代码讲解

`metadata.py` 是 chunking 领域的 metadata 契约层。它依赖 `app/core/metadata.py`，但 `core` 不依赖它。

这个依赖方向很重要：

```text
app/core/metadata.py
  <- app/ingest/chunking/metadata.py
  <- app/ingest/chunking/strategies.py
```

### `ChunkMetadata`

`ChunkMetadata` 继承 `BaseMetadata`，用于声明 `DocumentChunk.metadata` 中由 chunking 子系统维护的标准字段：

1. `chunker`
2. `chunking_strategy`
3. `chunk_size`
4. `chunk_overlap`
5. `tokenizer`
6. `char_start`
7. `char_end`
8. `section_title`
9. `token_start`
10. `token_end`
11. `section_block_count`

这样做的意义是把 metadata 字段契约从 `_build_chunk(...)` 的手写 dict 中提取出来。以后字段发生变化，可以先看 `ChunkMetadata`，而不是到多个 chunker 里翻找。

### `ChunkMetadataBuilder`

`ChunkMetadataBuilder` 负责组合三类 metadata：

1. 原始文档 metadata，例如 `filename`、`suffix`。
2. 标准 chunk metadata，例如 `chunker`、`chunking_strategy`。
3. 策略扩展 metadata，例如 `token_start`、`token_end`、`section_block_count`。

合并顺序是：

```text
document metadata -> standard chunk metadata -> extra metadata
```

这表示越靠后的 metadata 优先级越高。比如原始文档里如果刚好有一个 `chunker` 字段，标准 chunk metadata 会覆盖它，避免核心契约被上游 metadata 污染。

`ChunkMetadataBuilder` 当前是无状态 builder，所以 `build(...)` 是静态方法。它仍然保留 builder 类，是为了表达“metadata 合并策略”这个工程职责。

## `app/ingest/chunking/report.py` 代码讲解

### `ChunkingReportConfig`

`ChunkingReportConfig` 只保存 report writer 运行所需配置：输出目录。

它和 `ChunkingReportSettings` 的关系也遵循 Settings/Config 分离：

1. `ChunkingReportSettings` 来自 TOML。
2. `ChunkingReportConfig` 给 report writer 使用。
3. factory 负责二者转换。

### `ChunkingReportWriter`

`ChunkingReportWriter` 的职责是把 chunking 结果转换成 JSON 报告并写入文件。

它不负责创建目录。目录创建放在 `IndexBuilder._prepare_chunking_report_output()` 中。

这个边界很重要：

1. writer 只关心“写什么”。
2. pipeline 关心“写到哪里、运行产物目录是否存在”。
3. 配置系统关心“输出目录配置是什么”。

不要把这些职责混在一个类里。

### `build_report`

`build_report` 会统计：

1. 使用的 chunker。
2. strategy、chunk_size、chunk_overlap、tokenizer。
3. 文档数量和 chunk 数量。
4. 平均、最小、最大 token 数。
5. 空 chunk 数。
6. 缺失 doc_id、source_path、page、section 的数量。
7. 每篇文档的 chunk 摘要。

这份报告的作用不是给用户展示漂亮数据，而是帮助工程验收：策略是否切得太碎、metadata 是否丢失、PDF 页码是否保留下来。

## `app/ingest/chunking/quality.py` 代码讲解

`quality.py` 是 chunking 质量检查模块。它和 `report.py` 是一组相邻但职责不同的工程部件：

1. `ChunkingReportWriter` 负责统计和输出事实。
2. `ChunkingQualityChecker` 负责根据规则判断这些事实是否可接受。

它不读取文件、不写 JSON、不创建目录、不读取 `settings.toml`，也不调用 chunker 或 embedding。

### `ChunkingQualityConfig`

`ChunkingQualityConfig` 表示质量检查规则。

它包含：

1. `allow_empty_chunks`
   - 是否允许空 chunk。
2. `require_doc_id`
   - 是否要求每个 chunk 都有 `doc_id`。
3. `require_source_path`
   - 是否要求每个 chunk 都有 `source_path`。
4. `min_avg_token_count`
   - 平均 token 数过低时，说明 chunk 可能切得太碎。
5. `max_avg_token_count`
   - 平均 token 数过高时，说明 chunk 可能过长。
6. `max_missing_pdf_page_ratio`
   - PDF chunk 缺失页码的最大允许比例。
7. `max_missing_section_ratio`
   - chunk 缺失 section 的最大允许比例。
8. `avg_token_issue_severity`
   - token 均值问题默认是 warning。
9. `missing_section_issue_severity`
   - section 缺失问题默认是 warning。

这里的设计重点是：质量规则通过 Config 传入，而不是写死在 checker 内部。不同项目、不同文档类型可以使用不同质量门槛。

### `ChunkingQualityIssue`

`ChunkingQualityIssue` 表示一条结构化质量问题。

它包含：

1. `code`
   - 面向程序判断，例如 `empty_chunk_found`。
2. `message`
   - 面向人阅读的说明。
3. `severity`
   - `info`、`warning` 或 `error`。
4. `value`
   - 实际检查值。
5. `threshold`
   - 规则阈值。
6. `metadata`
   - 额外上下文，例如 PDF chunk 总数和缺失页码数量。

使用结构化 issue，而不是只返回字符串，是为了后续能方便地接入 CLI、API、CI、日志和报告。

### `ChunkingQualityCheckResult`

`ChunkingQualityCheckResult` 表示一次检查的整体结果。

它包含：

1. `issues`
   - 所有质量问题。
2. `checked_document_count`
   - 本次检查覆盖的文档数。
3. `checked_chunk_count`
   - 本次检查覆盖的 chunk 数。
4. `passed`
   - 派生属性，只要没有 `error` 级别 issue，就认为通过。
5. `error_count`
   - error 数量。
6. `warning_count`
   - warning 数量。

这里故意让 warning 不阻断通过。因为有些问题属于质量提示，例如 section 缺失比例偏高，不一定所有场景都必须中止流程。

### `ChunkingQualityChecker`

`ChunkingQualityChecker` 是执行检查的服务类。

它的公开入口是：

```python
result = ChunkingQualityChecker().check(
    documents=documents,
    chunks=chunks,
    config=config,
)
```

内部规则被拆成多个私有方法：

1. `_check_empty_result`
   - 有解析后文档，但没有产生任何 chunk。
2. `_check_required_identity_fields`
   - 检查 `doc_id` 和 `source_path`。
3. `_check_empty_chunks`
   - 检查空 chunk。
4. `_check_avg_token_count`
   - 检查平均 token 数是否过低或过高。
5. `_check_pdf_page_ratio`
   - 检查 PDF chunk 页码缺失比例。
6. `_check_section_ratio`
   - 检查 section 缺失比例。

这种拆分方式的重点是让规则可读、可测试、可扩展。以后如果新增规则，例如“过短 chunk 比例”“重复 chunk 比例”“跨页 chunk 比例”，可以继续添加私有检查方法，而不需要把一个大函数越写越长。

### 为什么不接入主流程

当前 checker 暂时没有接入 `IndexBuilder`，这是刻意保留的边界。

原因是质量检查接入主流程后还需要决定策略：

1. 发现 warning 是否继续构建索引。
2. 发现 error 是否中止构建。
3. 是否把 quality result 写入 report。
4. 是否允许通过配置关闭某些规则。

这些属于更高一层的 pipeline policy。当前阶段先把 checker 做成独立模块，更方便你学习它和 report writer 的职责区别。

## `app/factory.py` 代码讲解

`factory.py` 是项目的 composition root。它的职责不是做业务逻辑，而是统一组装对象。

本子模块新增了几个函数：

1. `build_chunker_config(project_settings)`
   - 把 `ProjectSettings.chunking` 转成 `ChunkerConfig`。
2. `build_chunking_report_config(project_settings)`
   - 把 `ProjectSettings.chunking_report` 转成 `ChunkingReportConfig`。
3. `build_configured_chunker(project_settings)`
   - 根据配置创建具体 chunker。

`build_index_builder(...)` 中也接入了：

1. chunker。
2. chunking report writer。
3. chunking report config。

这样做的好处是所有依赖都在上层统一组装，底层类不会偷偷创建自己的默认依赖。

这里还有一个重要扩展点：`build_configured_chunker(project_settings, chunker_registry=None)` 支持注入已经注册过的 registry。

默认路径：

```python
chunker = build_configured_chunker(project_settings)
```

使用内置 registry。

扩展路径：

```python
registry = build_default_chunker_registry()
registry.register("custom_strategy", CustomChunker)
```

这样外部策略可以进入 composition root，而不是被 `build_default_chunker_registry()` 固定死。

当前配置层的 `strategy` 已经放宽为字符串。真正使用 `"custom_strategy"` 时，配置层会接受这个名字，随后由 `ChunkerRegistry` 判断它是否已经注册。

## `app/indexing/index_builder.py` 代码讲解

`IndexBuilder` 是离线索引构建主流程。

当前流程是：

1. ingestion
   - 加载、解析、清洗文档。
   - 保存 raw 和 parsed document。
   - 输出 ingestion report。
2. chunking
   - 调用 chunker，把 `ParsedDocument` 转成 `DocumentChunk`。
   - 保存 chunks。
   - 输出 chunking report。
3. indexing
   - 对 chunks 生成 embedding。
   - 写入 vector store。
   - 使用 embedding cache 避免重复生成。
4. manifest
   - 记录本次索引构建的可复现信息。

这里的关键工程范式是 pipeline stage。每个阶段都有明确输入、输出、报告和 trace。

chunking report 接在 chunking 阶段之后，而不是放在 chunker 内部。原因是 report 需要看到全局 chunks 和 documents，而单个 chunker 只应该关心单篇文档的切分。

## 测试结构讲解

### `tests/test_chunking.py`

这个测试文件关注 chunking 子系统的公共行为：

1. `ChunkerRegistry` 是否按配置创建正确策略。
2. `FixedTokenChunker` 是否按 token 窗口切分。
3. `simple_regex` tokenizer 是否保留常见缩写、小数和日期 token。
4. `ChunkingReportWriter` 是否能输出质量报告。

这些测试不是为了覆盖每一行代码，而是保护工程契约。

### `tests/test_section_aware_chunker.py`

这个测试文件关注 section-aware 策略：

1. Markdown fallback 是否能识别标题。
2. 长 section 是否会二次切分。
3. chunk 是否保留文档版本字段。
4. char offset 是否能对应回原始文本。
5. ParsedBlock 中的 page metadata 是否能保留到 chunk。

这类测试保护的是 RAG 引用链路的基础。

## 本子模块体现的工程范式

### Strategy Pattern

`CharacterChunker`、`FixedTokenChunker`、`SectionAwareChunker` 是不同策略。它们都遵循 `Chunker.split(document)` 接口。

调用方不需要知道具体策略，只需要知道它拿到的是一个 chunker。

### Composition Root

`factory.py` 统一组装对象，避免底层类私自读取配置或创建依赖。

这是避免大型项目配置混乱的关键。

### Settings / Config 分离

`Settings` 负责外部配置文件。

`Config` 负责功能模块运行。

factory 负责转换。

这让代码更容易测试，也更容易替换配置来源。

### Pipeline Artifact

ingestion report 和 chunking report 都是 pipeline artifact。

它们不是业务核心数据，但它们对调试、验收、复现实验非常重要。

### Metadata Contract

`DocumentChunk` 的 metadata 不是随便加字段，而是检索、引用、评测和排障共同依赖的契约。

字段越核心，越应该放在 dataclass 顶层；字段越偏策略内部，越适合放在 metadata 中。

## 模块级练习

下面的练习不要求你修改某个小函数，而是要求你围绕当前代码添加或改造一个工程部件。每个练习都有明确代码入口，但不会把注意力放在局部算法细节上。

### 练习 1：把 chunker factory 升级为策略注册表

本练习已经作为示例代码完成。旧的 `build_chunker(config)` 入口已经移除，当前代码改为使用 `ChunkerRegistry`。

你应该重点阅读：

1. `app/ingest/chunking/registry.py` 中的 `ChunkerRegistry`。
2. `app/ingest/chunking/registry.py` 中的 `build_default_chunker_registry()`。
3. `app/factory.py` 中的 `build_configured_chunker(...)`。
4. `tests/test_chunking.py` 中的 registry 测试。

当前实现的核心结构是：

```python
registry = chunker_registry if chunker_registry is not None else build_default_chunker_registry()
chunker = registry.create(config)
```

内置策略在 `build_default_chunker_registry()` 中注册：

```python
registry.register("character", CharacterChunker)
registry.register("fixed_token", FixedTokenChunker)
registry.register("section_aware", SectionAwareChunker)
```

这表示：

1. registry 声明系统支持哪些策略。
2. `ChunkerConfig.strategy` 决定本次运行选择哪个策略。
3. factory 负责把配置转换成真实 chunker 对象。
4. `IndexBuilder` 只依赖 `Chunker` 抽象，不知道具体实现类。

本次练习中还处理了一个关键问题：如果 `build_configured_chunker()` 内部永远直接调用 `build_default_chunker_registry()`，那么外部注册的 chunker 没有机会进入系统。

当前采用的方案是让 `build_configured_chunker()` 和 `build_index_builder()` 都支持接收 `chunker_registry` 参数。这样外部模块可以先完成注册，再把 registry 传入 factory。

当前测试注入链路时，可以使用一份 fresh registry 替换默认策略映射：

```python
registry = ChunkerRegistry()
registry.register("section_aware", CustomSectionAwareChunker)

chunker = build_configured_chunker(
    project_settings,
    chunker_registry=registry,
)
```

注意：上面的示例用于说明“registry 注入链路”，不是推荐在真实业务中覆盖内置策略。真实扩展新策略时，应注册新的策略名。

当前已经完成的外部策略支持方案：

1. `ChunkingSettings.strategy` 和 `ChunkerConfig.strategy` 已经放宽为 `str`。
2. 配置层只校验策略名是非空字符串，并做 `strip()` 标准化。
3. 策略名称是否合法由 `ChunkerRegistry.validate_strategy(...)` 判断。
4. `ChunkerRegistry.create(config)` 会先调用校验器，再创建 chunker。
5. `ChunkerRegistry.register(...)` 会检查注册对象必须是 `Chunker` 子类。
6. 如果 registry 构建过程未来变复杂，可以进一步引入 `ChunkerRegistryProvider` 或应用级 dependency container。
7. 当前项目暂时使用“直接注入 registry 对象”，因为语义清楚、实现简单、测试容易。

你要学习的重点：

1. 策略扩展如何不污染主流程。
2. 为什么 `register(...)` 的参数来自代码层，而不是来自 `settings.toml`。
3. 为什么 registry 对象应该能从 composition root 外部注入。
4. 什么时候简单 `if` 足够，什么时候 registry 更合适。

验收标准：

1. `build_configured_chunker(project_settings)` 通过 registry 创建 chunker。
2. 新增策略时不需要修改 `IndexBuilder`。
3. 旧的 `build_chunker` 入口不再存在。
4. `build_configured_chunker(project_settings, chunker_registry=registry)` 能使用外部传入的 registry。
5. 测试仍然通过。

### 练习 2：新增 chunking 质量检查模块

本练习已经作为示例代码完成。当前已经新增 `app/ingest/chunking/quality.py`，用于判断 chunking 结果是否满足质量规则。

你应该重点阅读：

1. `app/ingest/chunking/quality.py` 中的 `ChunkingQualityConfig`。
2. `app/ingest/chunking/quality.py` 中的 `ChunkingQualityIssue`。
3. `app/ingest/chunking/quality.py` 中的 `ChunkingQualityCheckResult`。
4. `app/ingest/chunking/quality.py` 中的 `ChunkingQualityChecker`。
5. `tests/test_chunking_quality.py` 中的质量检查测试。

当前实现的核心结构是：

```python
result = ChunkingQualityChecker().check(
    documents=documents,
    chunks=chunks,
    config=ChunkingQualityConfig(),
)
```

它先作为独立模块存在，暂时不接入 `IndexBuilder`。这样可以保持职责清楚：checker 负责判断质量，pipeline 负责决定是否中止流程。

你要学习的重点：

1. report 和 quality checker 的职责区别。
2. 为什么质量规则不应该硬塞进 writer。
3. 如何为工程验收建立独立模块。
4. 为什么 error 可以阻断通过，而 warning 只作为质量提示。

验收标准：

1. checker 不负责写文件。
2. checker 不负责创建目录。
3. checker 不读取 `settings.toml`。
4. 测试覆盖通过案例、失败案例、warning 案例和配置校验案例。

### 练习 3：实现 chunking 实验运行器

真实项目中不会只凭感觉切换 chunking 策略。你需要能对同一批文档运行多种策略，并比较它们的结果。

你的任务是实现一个实验运行器，用来比较多个 chunking 配置。

建议新增文件：

```text
app/ingest/chunking/experiment.py
```

建议包含的结构：

1. `ChunkingExperimentCase`
   - 保存一个实验名称和一个 `ChunkerConfig`。
2. `ChunkingExperimentResult`
   - 保存策略名称、chunk 数量、平均 token、缺失 section 数、缺失 page 数等摘要。
3. `ChunkingExperimentRunner`
   - 接收 parsed documents 和多个 case，返回结果列表。

这个模块可以复用已有 `build_default_chunker_registry().create(config)` 和 `ChunkingReportWriter.build_report(...)`。

你要学习的重点：

1. 如何复用已有组件，而不是复制 chunking 逻辑。
2. 如何把实验流程做成独立模块。
3. 如何让策略比较成为工程流程的一部分。

验收标准：

1. runner 不读取原始文件，只接收 `ParsedDocument`。
2. runner 不调用 embedding。
3. runner 不写 vector store。
4. 同一批 documents 可以跑多个 chunking config。

### 练习 4：整理 chunk metadata 契约到代码层

本练习已经作为示例代码完成。当前已经将 chunk metadata 从 `_build_chunk(...)` 中拆出，形成通用基类和 chunking 专属 metadata 构造层。

你应该重点阅读：

1. `app/core/metadata.py` 中的 `BaseMetadata`。
2. `app/ingest/chunking/metadata.py` 中的 `ChunkMetadata`。
3. `app/ingest/chunking/metadata.py` 中的 `ChunkMetadataBuilder`。
4. `app/ingest/chunking/strategies.py` 中 `_build_chunk(...)` 如何使用 builder。
5. `tests/test_metadata.py` 中对 metadata 契约的测试。
6. `tests/test_chunking.py` 中对 chunker 输出 metadata 的契约测试。

当前实现采用两层结构：

```text
app/core/metadata.py
  BaseMetadata

app/ingest/chunking/metadata.py
  ChunkMetadata
  ChunkMetadataBuilder
```

这样分层的原因是：

1. `BaseMetadata` 是跨领域通用能力，未来 parser、retrieval、generation metadata 都可能复用。
2. `ChunkMetadata` 是 chunking 领域契约，应该放在 ingest/chunking 相关模块中。
3. `ChunkMetadataBuilder` 表达 metadata 合并策略，避免 `_build_chunk(...)` 继续膨胀。

当前核心结构是：

```python
metadata = ChunkMetadataBuilder().build(
    document_metadata=document.metadata,
    chunk_metadata=ChunkMetadata(...),
    extra_metadata=extra_metadata,
)
```

其中合并优先级是：

```text
document metadata -> standard chunk metadata -> extra metadata
```

你提出的判断也记录在这里：如果一个 metadata 基类未来可能被多个领域复用，它不应该放在 chunk 专属文件中，而应该放在 `app/core` 这样的通用层。

你要学习的重点：

1. 通用抽象应该放在更稳定、更底层的位置。
2. 领域契约应该放在对应领域模块中。
3. 抽象应该服务于字段契约和变化点，而不是为了看起来更高级。
4. metadata 契约需要被测试保护。

验收标准：

1. chunk 必须保留身份字段和来源字段。
2. 不同 chunker 输出的基础 metadata 形状一致。
3. `BaseMetadata` 不包含 chunking 业务逻辑。
4. 新增或调整字段时，测试能发现破坏性变更。

## 建议练习顺序

推荐顺序：

1. 先读懂当前代码和本 README 的代码讲解。
2. 做练习 2：新增质量检查模块。这个练习最接近子模块 2 的 ingest report 学习方式，能训练你新增一个完整工程部件。
3. 再做练习 3：实现实验运行器。它会帮助你理解 chunking 策略比较为什么需要工程化。
4. 最后再考虑练习 1 和练习 4，它们更偏结构重构，需要你对现有代码更熟。

不建议从改正则、调窗口参数开始。那些可以交给 AI 或后续实验来处理。当前阶段更重要的是理解：一个 RAG 工程为什么要把 chunking、report、quality、experiment 拆成不同部件。

## 运行方式

在项目根目录执行：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

如果只想检查 chunking 相关测试：

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_chunking tests.test_section_aware_chunker
```

## 完成本子模块后的验收标准

1. 你能解释 chunking 在 RAG pipeline 中的位置。
2. 你能说清 `Settings -> Config -> factory -> feature class` 的依赖方向。
3. 你能看懂并解释三个 chunker 的设计差异。
4. 你能说明为什么 chunking report 是 pipeline artifact。
5. 你能新增一个模块级工程部件，而不是只修改某个函数内部细节。
6. 你能判断什么时候应该加抽象，什么时候保持简单实现。
