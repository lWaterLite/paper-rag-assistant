# 子模块 6 练习说明：重排序、检索后处理与上下文预算

本练习在既有 `vector`、`bm25`、`hybrid` 检索能力之上，完成“候选结果如何成为可靠模型上下文”的后半段流程。

本次不是重新实现检索器。`HybridRetriever` 仍只负责组合多个召回源，`RetrievalPipeline` 负责候选与最终结果的编排，`RerankStage` 是独立的后处理阶段，`TokenAwareContextPacker` 负责在模型 token 预算内选择并追溯证据。

## 1. 本子模块完成的能力

```text
query
  -> RetrieverRegistry.resolve(strategy)
  -> VectorRetriever / BM25Retriever / HybridRetriever
  -> candidate_limit 候选集合
  -> ChunkIdDeduplicationStage
  -> RerankStage（可关闭）
  -> TopKLimitStage
  -> RetrievedChunk[]
  -> TokenAwareContextPacker
  -> PackedContext / ContextSegment[]
  -> RagPipeline / AnswerGenerator
```

当前新增或升级的能力：

1. `Reranker` Protocol 和 `RerankerRegistry`。
2. 无外部依赖的 `LexicalReranker` baseline。
3. `candidate_limit` 与最终 `top_k` 的职责分离。
4. `fail_open` 与 `fail_closed` 两种 rerank 失败策略。
5. `TokenEstimator` Protocol 和 `TokenEstimatorRegistry`。
6. `TokenAwareContextPacker`：token 预算、文档配额、去重、相邻合并、截断与 dropped reason。
7. `ContextSegment`：合并或截断后仍保存完整 `source_chunk_ids`、页码范围与章节信息。
8. 检索报告记录候选上限、rerank 阶段、策略、降级状态和最终结果。
9. API 返回保留原始 retrieval score，同时可返回 `rerank_signal`，避免混淆不同分数的含义。

## 2. 代码结构

```text
app/retrieval/
  rerankers/
    base.py
      Reranker Protocol、RerankedCandidate
    config.py
      RerankingConfig
    lexical.py
      LexicalReranker
    registry.py
      RerankerRegistry、内置策略注册
    stage.py
      RerankStage

  token_estimators/
    base.py
      TokenEstimator Protocol
    config.py
      TokenEstimatorConfig
    regex.py
      RegexTokenEstimator
    registry.py
      TokenEstimatorRegistry

  postprocessing/
    config.py
      PostProcessingConfig
    validator.py
      PostProcessingConfigValidator
    profile.py
      PostProcessingProfile

  pipeline_types.py
    RetrievalPipelineContext、RetrievalStageResult

  pipeline.py
    候选召回、后处理阶段、rerank、报告、compare search

  context_packer.py
    ContextPackRequest、ContextSegment、TokenAwareContextPacker

app/factory/
  configs.py
    Settings -> Config
  retrieval.py
    reranker、token estimator、context packer、search service 组装
  application.py
    可选外部 registry 的组合根注入
  pipelines.py
    RAG pipeline 组装
```

相关测试位于：

```text
tests/retrieval/test_reranking.py
tests/retrieval/test_token_estimators.py
tests/retrieval/test_context_packer.py
tests/retrieval/test_retrieval_reporting.py
tests/interfaces/test_api_schemas.py
```

## 3. 核心架构与依赖方向

### 3.1 候选召回与重排序分层

```text
Retriever
  负责从索引中召回候选。
  不知道 reranker，也不负责最终上下文选择。

Reranker
  只接收 query 和已召回候选。
  不访问 VectorCollection、ChunkRepository 或 settings.toml。

RerankStage
  是 RetrievalPipeline 的一个阶段。
  负责调用已注入的 Reranker、验证输出契约、处理降级策略。

TopKLimitStage
  在 rerank 之后选择最终返回数。
```

因此，关闭 rerank 时，pipeline 仍保持此前行为：

```text
retrieve(top_k)
  -> 去重
  -> 最终截断
```

启用 rerank 时，才会扩大候选集合：

```text
retrieve(candidate_limit)
  -> 去重
  -> rerank
  -> 最终 top_k
```

`candidate_limit` 是 reranker 可以观察的候选宽度；`top_k` 是最终交给 ContextPacker、API 和生成流程的结果数量。两者不能混用。

### 3.2 为什么 `RerankStage` 不在 `HybridRetriever` 中

`HybridRetriever` 处理的是“多个召回源如何融合”的问题，当前使用加权 RRF。Rerank 处理的是“给定候选集合，哪些更直接回答 query”的问题。

若把 rerank 放进 `HybridRetriever`：

1. vector-only、BM25-only 无法复用 rerank。
2. compare search 无法区分融合排序与重排序。
3. hybrid 的职责会从“召回策略”膨胀为“完整检索流程”。

现在 rerank 位于 pipeline 后处理链，因此它可以平等地作用于 vector、BM25、hybrid 和未来外部 retriever。

### 3.3 `LexicalReranker` 的定位

`LexicalReranker` 基于 query token 覆盖率、token 密度与短语命中评分。它是一个确定性 baseline，用于验证完整工程链路：

```text
Settings
  -> RerankingConfig
  -> RerankerRegistry
  -> LexicalReranker
  -> RerankStage
  -> trace / report / API
```

它不等价于 cross-encoder。真实 cross-encoder 更适合捕捉语义、否定、比较关系和长距离依赖，但需要额外模型依赖和推理资源。项目默认不安装、不下载任何 rerank 模型。

### 3.4 Reranker 的输出契约

`RerankStage` 会验证 reranker 输出：

1. 不能返回候选集合之外的 chunk。
2. 不能返回重复 chunk。
3. 不能遗漏候选；rerank 只排序，不承担过滤职责。

这条约束很重要。若未来某个外部服务返回了未知 chunk 或静默丢弃候选，系统会抛出明确的 `RERANK_FAILED`，或按配置走 `fail_open`，而不是悄悄污染结果。

### 3.5 fail-open 与 fail-closed

`[retrieval.reranking].failure_mode` 支持：

```text
fail_open
  reranker 出错后保留原始候选排序。
  trace 和报告记录 degraded=true。

fail_closed
  reranker 出错后终止检索请求。
  对外抛出 RERANK_FAILED，并保留 trace。
```

交互式问答通常优先 `fail_open`，因为基础检索结果仍有价值；将 rerank 作为硬质量或合规要求的场景才适合 `fail_closed`。

## 4. Token-aware Context Packing

### 4.1 为什么要替换字符预算

旧 `SimpleContextPacker` 以字符数限制上下文。字符数无法准确描述模型窗口：中文、英文、标点、引用格式和代码的 token 比例不同，系统 prompt 与输出预留也不包含在字符预算中。

新的 `ContextPackerConfig` 使用：

```text
model_context_window
reserved_prompt_tokens
reserved_output_tokens
safety_margin_tokens
max_context_tokens
max_chunks_per_document
```

实际资料预算为：

```text
min(
  max_context_tokens,
  model_context_window
    - question_tokens
    - reserved_prompt_tokens
    - reserved_output_tokens
    - safety_margin_tokens
)
```

这表示 `max_context_tokens` 是资料的业务上限，模型窗口推导出的值则是硬上限。两者取更小者。

### 4.2 `Tokenizer` 与 `TokenEstimator` 的边界

```text
Tokenizer
  为 BM25 分词服务。
  决定词频、文档频率和 query term。

TokenEstimator
  为生成模型上下文预算服务。
  决定文本是否还能放进模型窗口。
```

当前 `RegexTokenEstimator` 是不依赖第三方库的稳定近似实现。它可用于测试和 mock 环境，但并不保证与具体 LLM 的 tokenizer 完全一致。

### 4.3 `ContextSegment` 为什么存在

当两个相邻 chunk 合并时：

```text
chunk_12 + chunk_13
  -> 一个上下文段
```

单个 `Citation` 仍为了兼容既有回答格式而引用首个 chunk，但 `ContextSegment.source_chunk_ids` 会保留完整来源列表。子模块 7 的 citation 校验可以据此判断回答引用是否真正覆盖了对应证据段。

`PackedContext` 同时提供：

```text
context_text
citations
used_chunks
dropped_chunks
segments
token_usage
```

其中 `dropped_chunks` 不只是调试列表。它能说明证据为何被舍弃：重复、文档配额、token 预算不足等。

## 5. 配置与对象创建流程

### 5.1 TOML 配置

`settings.toml` 新增以下配置树：

```toml
[retrieval.reranking]
enabled = false
strategy = "lexical"
candidate_limit = 12
batch_size = 8
failure_mode = "fail_open"

[retrieval.context_packing]
model_context_window = 4096
max_context_tokens = 1800
reserved_prompt_tokens = 200
reserved_output_tokens = 512
safety_margin_tokens = 64
max_chunks_per_document = 2

[retrieval.context_packing.token_estimator]
strategy = "regex"
```

`.env` 不新增这些行为配置；它继续只放 API key 等敏感信息。

### 5.2 Settings 到运行时对象

```text
settings.toml
  -> RerankingSettings / ContextPackingSettings / TokenEstimatorSettings
  -> ConfigFactory
  -> RerankingConfig / ContextPackerConfig / TokenEstimatorConfig
  -> PostProcessingConfig
  -> RetrievalFactory
  -> PostProcessingConfigValidator / PostProcessingProfile
  -> RerankerRegistry / TokenEstimatorRegistry
  -> SearchService / CompareSearchService / TokenAwareContextPacker
```

`RetrievalFactory` 会先验证 `PostProcessingConfig`，再构建 `SearchService`、reporter 或 `TokenAwareContextPacker`。`PipelineFactory` 则把同一份已校验 Config 同时传给 SearchService 与 ContextPacker，保证一次 `RagPipeline` 组装不会混用不同配置快照。

`RerankStage`、`TokenAwareContextPacker` 和具体策略对象不会读取 `ProjectSettings`，也不会自己选择默认实现。所有对象由 Factory 组装。

### 5.3 外部策略的注册方式

可以在组合根创建 `ApplicationFactory` 时注入外部 registry。例如真实 cross-encoder 策略应在工厂层注册 provider，再由 `RerankerRegistry` 根据 `RerankingConfig.strategy` 解析。业务 pipeline 不需要新增 `if strategy == ...`。

当前默认 registry 只注册 `lexical`，并将已有 BM25 tokenizer 注入该策略。这样 lexical baseline 在索引查询与 rerank 的词项理解上保持一致；真实 cross-encoder 则可以采用自身模型 tokenizer，不需要复用 BM25 tokenizer。

## 6. 报告、trace 与 API

每个单策略检索报告现在包含：

```text
request.resolved_candidate_limit
stages[].status
stages[].detail.reranker
stages[].detail.degraded
results[].rerank_signal
runtime.config.postprocessing
```

`runtime.config.postprocessing` 是 `PostProcessingProfile` 的稳定快照。它会记录 rerank 是否启用、candidate limit 当前的实际语义、默认候选宽度、单文档上限、模型窗口、预留 token、静态可用预算和有效上下文上限。单次请求再扣除 question token 后的实际预算仍保留在 `context_packing` trace 中。

`RetrievedChunk.score` 仍表示原检索器给出的 score；rerank 的 score 放在独立的 `rerank_signal` 中。这样不会把 BM25、向量、RRF 与 rerank 分数误当作同一量纲。

`/search` 和 `/search/compare` 的响应中的每个结果也可携带 `rerank_signal`。compare search 仍是并列观察工具，不会把多个策略的 score 直接相加。

`RagPipeline` 的 `context_packing` trace 增加了：

```text
context_tokens
available_context_tokens
dropped_chunk_count
```

这属于领域流程的局部 trace；HTTP 请求级日志、SSE 和通用可观测性仍留在子模块 9。

## 7. 如何运行

项目默认关闭 rerank，因此可直接沿用现有检索命令：

```powershell
.\.venv\Scripts\python.exe -B -m app.main search "RAG 中 rerank 的作用" --use-existing-index --retriever hybrid
```

要启用内置 lexical reranker，修改 `settings.toml`：

```toml
[retrieval.reranking]
enabled = true
strategy = "lexical"
candidate_limit = 12
batch_size = 8
failure_mode = "fail_open"
```

之后同一个 `search` 或 `ask` 命令会自动走：

```text
候选召回 -> 去重 -> lexical rerank -> 最终 top-k
```

报告默认写入 `logs/retrieval`。单策略检索报告会记录 rerank 阶段；compare search 的父级聚合报告仍记录各策略执行情况及其子报告路径。

## 8. 测试与自检

运行全量测试：

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -t .
```

运行子模块 6 的核心测试：

```powershell
.\.venv\Scripts\python.exe -B -m unittest `
  tests.retrieval.test_reranking `
  tests.retrieval.test_token_estimators `
  tests.retrieval.test_context_packer `
  tests.retrieval.test_postprocessing `
  tests.retrieval.test_retrieval_reporting
```

现有测试由项目直接维护，覆盖：

1. lexical rerank 的排序变化。
2. candidate limit 大于最终 top-k。
3. fail-open 与 fail-closed。
4. token estimator registry。
5. token 预算、文档配额、去重、截断与 segment provenance。
6. rerank 阶段写入 retrieval report。
7. API 映射保留原始 retrieval score 并暴露 rerank signal。
8. 后处理配置组合校验、Factory 的提前失败与 Profile 报告快照。

## 9. 工程练习

本次练习不要求你接入真实模型、下载模型文件或新增第三方推理依赖，也不要求补测试；新增或改动功能时，对应测试由项目维护。

练习关注的是已经具备真实实现后的下一层问题：如何让后处理流程的配置组合能够被清晰校验与说明，以及如何在不破坏来源追溯与报告的前提下扩展证据处理阶段。这两项能力比替换某个具体 reranker 或 tokenizer 更值得在当前阶段掌握。

### 练习 1：定义后处理流程配置的有效组合（已实现）

当前系统的后处理行为由 reranking、candidate limit、context packing 等多组配置共同决定。它们分散存在是合理的，但组合本身也有领域规则，例如：

```text
reranking.enabled = false 时，candidate_limit 不应被误解为 rerank 候选宽度。
max_chunks_per_document 会直接影响单篇文档可以占用的上下文上限。
上下文 token 预算必须大于各类预留预算推导出的最低可用值。
```

本项目已增加一个面向“后处理流程”的配置校验与说明层，而不是把互相依赖的检查散落到每个 Config 的 `__post_init__` 中：

```text
app/retrieval/postprocessing/
  config.py
    PostProcessingConfig
  validator.py
    PostProcessingConfigValidator
  profile.py
    PostProcessingProfile
```

推荐分工：

```text
RerankingConfig / ContextPackerConfig
  各自验证自身字段是否合法。

PostProcessingConfigValidator
  验证跨配置组合是否有意义，抛出带字段路径的 AppError 或 ValueError。

PostProcessingProfile
  生成不可变的运行时摘要，供 retrieval report、CLI 或未来 API 使用。
  它不构建 retriever、reranker 或 context packer。
```

当前实现的规则与接入位置：

1. `ConfigFactory.build_postprocessing_config()` 聚合既有 Config，不新增 TOML 或新的 Settings 类。
2. `PostProcessingConfigValidator` 拒绝启用 rerank 但 `candidate_limit < top_k` 的组合；同时拒绝预留 token 后没有资料预算、或 `max_context_tokens` 超出静态可用预算的组合。
3. `RetrievalFactory` 在创建 SearchService、CompareSearchService 或 ContextPacker 前执行校验，并将错误统一转换为 `AppError(INVALID_CONFIG)`。
4. `PostProcessingProfile` 不可变，记录启用状态、默认候选宽度及其来源、失败策略、文档配额和 token 预算；它被写入 `runtime.config.postprocessing`。
5. `PipelineFactory` 使用同一份已验证的 `PostProcessingConfig` 组装 SearchService 与 ContextPacker，避免一次 RAG 请求混用不同快照。

验收重点是：错误配置能在应用组装时尽早暴露；一份检索报告能够解释本次结果使用了哪一种完整的后处理方案。

### 练习 2：设计证据变换阶段的边界

未来可能需要对候选证据进行去模板、相邻段合并、抽取式压缩或格式规整，但这些步骤不能混入 `AnswerGenerator`，也不能丢失来源。请为“证据变换”设计一个可插拔阶段边界，而不是立刻接入 LLM 压缩。

建议结构：

```text
app/retrieval/evidence_transformers/
  base.py
    EvidenceTransformer Protocol
    EvidenceTransformRequest / EvidenceTransformResult
  passthrough.py
    PassthroughEvidenceTransformer
  stage.py
    EvidenceTransformStage
```

建议在 `TopKLimitStage` 之后、`TokenAwareContextPacker` 的合并与预算逻辑之前插入该阶段：

```text
retrieve -> dedup -> rerank -> top-k
  -> evidence transform
  -> token-aware context packing
```

实现要求：

1. 先只实现 `PassthroughEvidenceTransformer`，以验证完整组装、阶段 trace、错误处理和 provenance 契约，不实现细粒度的文本压缩算法。
2. 输入输出必须能明确关联原始 `chunk_id`；若一个输出片段来自多个 chunk，必须保留完整 source id 集合和原文范围。
3. 将阶段状态、输入/输出数量、是否降级及转换器名称写入已有 report；不要另起一套日志格式。
4. future transformer 若产生无法追溯到原文的新表述，必须被接口契约禁止；LLM 摘要式压缩仍留到子模块 7 的 grounded generation 与 citation 校验之后。

验收重点是：你可以在不修改 Reranker、Retriever、AnswerGenerator 的情况下新增任意证据变换器，并且最终 `ContextSegment` 仍能解释每一段上下文来自哪里。

## 10. 本子模块边界

本子模块不会提前完成以下内容：

1. Query rewrite、multi-query、HyDE、真实回答生成和回答级 citation 校验属于子模块 7。
2. Golden dataset、Recall@k、MRR、Ragas、实验 runner 与实验报告属于子模块 8。
3. FastAPI、SSE streaming、应用状态与请求级结构化日志属于子模块 9。
4. 权限、prompt injection、限流、Docker、CI 与成本治理属于子模块 10。

完成本子模块后，你应该能够解释：为什么 rerank 不替代 retriever、为什么 candidate limit 与 top-k 必须分开、为什么 token estimator 不能复用 BM25 tokenizer，以及为什么上下文段必须保存完整的来源关系。
