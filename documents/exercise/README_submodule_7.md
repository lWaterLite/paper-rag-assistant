# 子模块 7 练习说明：Query 改写、回答生成与引用校验

本子模块将此前完成的可检索证据链路闭合为一个可用的论文问答流程：用户问题先经过查询规划，再进入统一检索和上下文组织；回答生成器只消费本次 `PackedContext` 中的证据，模型输出必须通过 citation 校验，最终返回带来源、状态与 trace 的 `RagAnswer`。

这不是把 `LLM SDK` 直接塞进 `RagPipeline` 的实现。代码把供应商适配、查询规划、prompt、生成、citation 校验、Settings/Config、Factory 与 Runtime 分开，使真实模型、本地模型和离线 mock 能在不改变业务流程的前提下替换。

## 1. 已完成的能力

```text
用户问题
  -> QueryPlanningStage
      -> QueryPlan
         original_query / primary_query / additional_queries / keywords / HyDE metadata
  -> SearchService.search_queries(...)
      -> 多 query 候选合并
      -> 统一 dedup -> rerank -> top-k
  -> EvidenceTransformStage
  -> TokenAwareContextPacker
      -> PackedContext（context_text + ContextCitation + token usage）
  -> RagAnswerPromptBuilder
  -> PromptBudgetValidator
  -> LlmClient
  -> GeneratedAnswerPayload（受限 JSON）
  -> CitationValidator
  -> RagAnswer（answer + citations + status + trace_id）
```

本次完成的关键能力：

1. `QueryPlan` 保留原始问题、主检索 query、额外 query、关键词、可选 HyDE 文本和回退状态。
2. `QueryPlannerRegistry` 支持 `passthrough`、`rule_based`、`llm` 三种查询规划策略；外部策略可在组合根注册。
3. 多 query 会先合并候选，再统一执行 dedup、rerank 和最终 top-k，不会分别截断后粗暴拼接结果。
4. `LlmClient` 是供应商无关 Protocol；默认 `MockLlmClient` 可离线运行，`OpenAiCompatibleLlmClient` 使用标准库 HTTP 支持显式配置的兼容服务。
5. `RagAnswerPromptBuilder` 将论文正文明确标记为资料数据，并要求模型只输出 JSON。
6. `GroundedAnswerGenerator` 在调用模型前执行完整 prompt token 预算校验；无证据时直接返回业务性拒答，不调用模型。
7. `CitationValidator` 校验格式、来源存在性、正文 citation 与结构化字段一致性、以及拒答语义。
8. 最终的 `Citation` 由 `ContextCitation` 映射生成，模型不能自由编造论文标题、页码、路径或 chunk id。
9. `RagTrace` 记录 `query_planning`、`retrieval`、`evidence_transformation`、`context_packing`、`generation` 五个阶段；generation 阶段只写入安全诊断摘要。
10. `/ask` 的框架无关 handler 已接入 `AskRequest`、`AskResponse` 与可选 trace、检索片段返回。

## 2. 代码总览

```text
app/
  llm/
    models.py                 LlmMessage、LlmRequest、LlmResponse、LlmUsage
    base.py                   LlmClient Protocol
    config.py                 LlmClientConfig
    registry.py               LlmClientRegistry 与内置 provider 注册
    mock.py                   离线、确定性的 mock provider
    openai_compatible.py      无 SDK 依赖的 Chat Completions HTTP 适配器
    retrying.py               可重试基础设施失败的装饰器

  retrieval/query/
    models.py                 QueryPlan
    base.py                   QueryPlanner Protocol
    config.py                 QueryPlanningConfig
    passthrough.py            原始 query 基线
    rule_based.py             受控中英文论文术语补充
    llm.py                    LLM JSON 输出到 QueryPlan 的解析器
    registry.py               QueryPlannerRegistry
    stage.py                  fail-open / fail-closed 查询规划边界

  generation/
    models.py                 Citation、GenerationDiagnostics、RagAnswer
    configuration.py          GenerationConfig、CitationValidationConfig
    generated.py              未校验的 GeneratedAnswerPayload
    prompts/                  版本化回答 prompt 构造
    citations/                CitationValidator 与校验结果
    answering/                AnswerGenerator Protocol、预算校验、GroundedAnswerGenerator

  factory/
    llm.py                    LLM Client 生命周期内缓存与敏感密钥注入
    query.py                  QueryPlanner 的注册表解析与阶段创建
    generation.py             生成器、prompt、citation、token estimator 组合
    configs/generation.py     Generation Settings -> Config 适配
    pipelines.py              将查询、检索、上下文和生成组装为 RagPipeline

  core/settings/generation.py
                              GenerationSettings 及其子 Settings
  pipeline.py                 顶层 RAG 流程编排与跨阶段 trace
  api/handlers/retrieval.py   handle_ask_request
```

新增或重点修改的测试：

```text
tests/generation/test_answer_generation.py
tests/retrieval/test_query_planning.py
tests/retrieval/test_retrieval_pipeline.py
tests/integration/test_rag_pipeline.py
tests/core/test_config_factory.py
tests/interfaces/test_api_schemas.py
```

## 3. 整体架构与依赖方向

### 3.1 查询规划属于检索入口，不属于回答生成

`QueryPlanner` 可能使用 LLM，但它的输出直接决定检索输入，因此代码位于 `app/retrieval/query/`。它不依赖 `RagAnswer`、Citation 或 prompt；它只产生 `QueryPlan`。

```text
retrieval.query
  依赖 LlmClient Protocol
  产出 QueryPlan

generation
  依赖 PackedContext、ContextCitation 与 LlmClient Protocol
  产出 RagAnswer

retrieval 不反向依赖 generation。
```

这样划分的直接收益是：未来关闭 query rewrite、切换到词典改写、替换为 LLM 改写，或增加领域缩写扩展时，都不会触碰回答生成器。反过来，修改 citation 规则也不会改变检索 query。

### 3.2 `llm` 是基础设施包，不是业务领域包

`app/llm/` 只定义一次模型调用所需的稳定数据契约和 provider adapter。它不知道：

- 什么是论文 chunk；
- 什么是 `PackedContext`；
- 什么是 citation；
- 当前调用是在改写 query 还是生成答案。

调用目的通过上层的 prompt 和 `metadata` 描述；`metadata` 只供离线 mock 使用，真实 provider 不依赖它。业务层只看 `LlmResponse`，不会泄漏第三方 SDK 的 request、response 或异常对象。

### 3.3 `generation` 负责最终回答，而不是重新检索

`GroundedAnswerGenerator` 接收：

```text
question
PackedContext
RetrievedChunk[]
RagTrace
```

其中 `PackedContext` 是不可跨越的证据边界。生成器不能自行调用 `ChunkRepository`、`VectorRepository` 或 `Retriever` 再找资料；这样既能保证本次 citation 和本次上下文一致，也避免回答阶段绕过 token packing 和来源审计。

### 3.4 为什么 Prompt、模型调用和校验要分开

```text
RagAnswerPromptBuilder
  只定义“给模型什么输入”。

LlmClient
  只负责“如何向一个模型供应商发请求”。

GeneratedAnswerPayload
  只描述“模型声称生成了什么”。

CitationValidator
  只判断“这个回答是否符合当前来源契约”。

GroundedAnswerGenerator
  只负责编排以上对象并产生 RagAnswer。
```

如果把这些职责都放进一个类，未来替换模型、审阅 prompt、切换 JSON schema、处理 citation 错误或记录 token 用量时都会互相干扰。现在的分层让这些变化分别发生在对应包内。

## 4. 数据流动路线

### 4.1 查询到候选证据

```text
RagPipeline.ask(question)
  -> QueryPlanningStage.plan(question)
  -> QueryPlan
  -> SearchService.search_queries(
       original_query,
       retrieval_queries=primary_query + additional_queries,
     )
  -> RetrievalPipeline.search_queries(...)
  -> 每个 query 调用同一个 Retriever
  -> 合并候选
  -> ChunkIdDeduplicationStage
  -> RerankStage（若启用）
  -> TopKLimitStage
  -> RetrievalPipelineResult.results
```

这里的 `query` 与 `retrieval_queries` 有意分开：报告、用户可见 trace 和后续回答仍保留 `original_query`；实际召回使用改写后的表达。多 query 的候选会在 rerank 前合并，因此同一证据不会因不同措辞分别占据最终 top-k。

`QueryPlan.hyde_document` 会被保留在计划与 trace 摘要中，但默认流程不会把它当作普通文本 query 交给 BM25 或 hybrid。HyDE 是 dense retrieval 的专用辅助物；在当前统一 `SearchService` 里强行把它发给所有检索器会混淆语义。它保留了正确的数据契约，待未来拆分 dense query 分支时再接入向量查询。

### 4.2 候选到模型上下文

```text
RetrievedChunk[]
  -> EvidenceTransformStage
  -> EvidenceCandidate[]
  -> TokenAwareContextPacker
  -> PackedContext
     context_text
     citations: ContextCitation[]
     used_chunks
     dropped_chunks
     segments
     token_usage
```

这部分沿用子模块 6 的职责：它选择并组织原文证据，但不写答案、不解释事实。`ContextCitation` 中已经保存 `doc_id`、`version_id`、章节和页码等来源信息。

### 4.3 上下文到最终回答

```text
PackedContext
  -> RagAnswerPromptBuilder.build(...)
  -> RagAnswerPrompt(system_prompt, user_prompt, version)
  -> PromptBudgetValidator.validate(...)
  -> LlmClient.complete(...)
  -> GeneratedAnswerPayload.from_json(...)
  -> CitationValidator.validate(...)
  -> Citation.from_context_citation(...)
  -> RagAnswer
```

若 `PackedContext.citations` 为空，`GroundedAnswerGenerator` 不会调用 LLM，而是直接返回：

```text
status = "abstained"
citations = []
abstention_reason = "当前知识库中没有检索到足够相关的资料"
```

这是一种成功完成的业务结果，不是 `GENERATION_FAILED`。

## 5. 配置与对象创建流程

### 5.1 TOML、Settings 与 Config

非敏感行为配置放在 `settings.toml`：

```toml
[generation.llm]
provider = "mock"
model = "mock-grounded-json"
base_url = ""
timeout_seconds = 30.0
max_retries = 1

[generation.query_planning]
enabled = true
strategy = "rule_based"
multi_query_enabled = false
max_additional_queries = 2
hyde_enabled = false
failure_mode = "fail_open"

[generation.answering]
temperature = 0.0
prompt_version = "rag_answer_v2"
default_language = "中文"
invalid_output_mode = "fail_closed"
```

`core/settings/generation.py` 读取它们为 `GenerationSettings`。`factory/configs/generation.py` 再生成：

```text
LlmClientConfig
QueryPlanningConfig
GenerationConfig
CitationValidationConfig
```

`GenerationConfig.max_output_tokens` 没有再单独配置，而是从 `retrieval.context_packing.reserved_output_tokens` 适配而来。原因是模型最大输出既属于回答生成行为，也必须是 ContextPacker 的硬预留；保留单一配置源可以避免“packing 预留 512、模型实际允许 1024”这种潜在超窗错误。

### 5.2 敏感配置

当 provider 为 `openai_compatible` 时，认证密钥只从环境变量或 `.env` 读取：

```text
RAG_LLM_API_KEY=<你的密钥>
```

它由 `EnvSettings.llm_api_key` 读取，仅在 `LlmFactory` 构建 provider adapter 时取出。密钥不会进入：

- `LlmClientConfig`；
- `GenerationConfig`；
- prompt；
- `RagTrace`；
- retrieval/generation 报告；
- 对外 API 响应。

### 5.3 Factory 与 Runtime 的组装路线

```text
ApplicationFactory
  -> ConfigFactory
  -> RetrievalFactory
  -> LlmFactory
      -> LlmClientRegistry.create(...)
      -> RetryingLlmClient
  -> QueryFactory
      -> QueryPlannerRegistry.create(...)
      -> QueryPlanningStage
  -> GenerationFactory
      -> RagAnswerPromptBuilder
      -> CitationValidator
      -> PromptBudgetValidator
      -> GroundedAnswerGenerator
  -> PipelineFactory
      -> RagPipeline
  -> ApplicationRuntime
      -> 缓存 index、SearchService、CompareSearchService、RagPipeline
```

`LlmFactory` 在一个 `ApplicationFactory` 生命周期内只构建一个 client，`QueryFactory` 和 `GenerationFactory` 复用它。业务类没有 `client or SomeProviderClient()` 这样的隐式构造路径，也不会私自读取 `settings.toml` 或 `.env`。

`ApplicationRuntime` 的职责没有扩大：它仍在应用启动时加载一次索引并缓存在线服务。真实 HTTP client 的连接池或会话若需要关闭，应在未来 provider adapter 增加显式 `close()` 协议后接入 Runtime 的 `shutdown()`；当前标准库 adapter 不持有长期连接。

## 6. 注册表与扩展方式

### 6.1 LLM provider 注册表

默认 `LlmClientRegistry` 注册：

```text
mock
  -> MockLlmClient

openai_compatible
  -> OpenAiCompatibleLlmClient
```

`mock` 是离线确定性实现：它用于测试 Settings、Factory、Prompt、JSON 解析和 citation 契约，不声称具备真实问答推理能力。`openai_compatible` 不引入供应商 SDK，要求 `base_url` 是完整的 Chat Completions endpoint；不同供应商响应若不兼容，需要添加独立 adapter，而不是在业务生成器内判断 provider 名称。

`RetryingLlmClient` 只对 429、5xx、连接错误等标记为可重试的基础设施错误做有限指数退避；认证、协议解析和 citation 校验错误不会重试，避免重复扣费或掩盖配置问题。

### 6.2 QueryPlanner 注册表

```text
passthrough
  -> 原始问题直接检索

rule_based
  -> 使用受控中英文论文术语扩展 query

llm
  -> 请求模型返回 JSON QueryPlan
```

`QueryPlanningStage` 承担失败策略：`fail_open` 时使用原始 query 并记录 `fallback_used=true`；`fail_closed` 时抛出 `QUERY_REWRITE_FAILED`。具体 planner 不应自行吞掉异常或擅自决定回退。

## 7. Citation 与拒答的实际语义

`CitationValidator` 当前完成确定性校验：

1. `citation_ids` 必须是字符串列表。
2. 正文中的 `[C1]` 等 citation id 必须与结构化 `citation_ids` 一致。
3. 每个 id 必须在本次 `PackedContext.citations` 中存在。
4. 有可用证据但非拒答时必须至少引用一个 id。
5. 拒答不能同时宣称有 citation；开启配置时还必须说明资料不足原因。

校验成功后，系统才从 `ContextCitation` 构造 `Citation`。这保证返回的标题、页码、章节、路径、文档版本和 snippet 都来自本次检索上下文。

当前实现不会宣称已经完成“citation 是否在语义上支撑相邻句子”的完全自动验证。这需要更细粒度的 claim 切分、NLI/LLM judge 和子模块 8 的评测集。确定性来源校验是第一道必须可靠的边界，而不是语义评测的替代物。

## 8. Trace、错误与 API

`RagPipeline` 的 trace 阶段顺序为：

```text
query_planning
  -> retrieval
  -> evidence_transformation
  -> context_packing
  -> generation
```

generation 的 trace 只写入安全摘要：provider 名称、模型标识、prompt token、输出 token、citation 校验状态和回答状态。它不记录完整 prompt、论文正文、API key 或认证 header。

错误语义应区分：

| 情况 | 对外/领域语义 |
| --- | --- |
| query planner fail-open | 正常检索原始问题，trace 记录回退 |
| query planner fail-closed | `QUERY_REWRITE_FAILED` |
| 无证据 | `RagAnswer(status="abstained")` |
| LLM 连接、限流或响应故障 | `GENERATION_FAILED` |
| JSON、citation id 或拒答契约无效 | `CITATION_VALIDATION_FAILED` |

`handle_ask_request()` 已使用 `AskRequest` 调用 `RagPipeline.ask()`，并将 `RagAnswer` 转为 `AskResponse`。当 `debug_trace=true` 时，响应携带 trace；当 `include_retrieved_chunks=true` 时，响应携带本次检索结果。实际 FastAPI route、SSE 和 HTTP 中间件仍在子模块 9 实现。

## 9. 运行与配置真实模型

### 9.1 离线 mock 模式

默认配置无需任何密钥：

```toml
[generation.llm]
provider = "mock"
```

它可以验证完整工程链路，但回答会明确说明自己不生成新的事实性归纳。它不是用于评估 RAG 效果的真实模型。

在已有依赖环境中，可使用：

```powershell
.\.venv\Scripts\python.exe -B -m app.main ask "RAG 为什么需要引用？" --use-existing-index
```

### 9.2 OpenAI-compatible 模式

本项目没有新增第三方 SDK 依赖。若你的服务兼容 Chat Completions 协议，在 `settings.toml` 修改：

```toml
[generation.llm]
provider = "openai_compatible"
model = "你的模型名"
base_url = "https://你的服务地址/v1/chat/completions"
timeout_seconds = 30.0
max_retries = 1
```

并在 shell、`.env` 或部署平台 Secret 中提供：

```text
RAG_LLM_API_KEY=<你的密钥>
```

不要把密钥提交到 `settings.toml`、报告或测试样例。不同供应商若不支持 `model`、`messages`、`temperature`、`max_tokens` 和 `choices[0].message.content` 这一协议，应新增 provider adapter 并注册，而不是修改 `GroundedAnswerGenerator`。

## 10. 测试与自检

不需要为本次练习自行编写测试；项目已补充核心覆盖。

运行全量测试：

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -t .
```

运行子模块 7 核心测试：

```powershell
.\.venv\Scripts\python.exe -B -m unittest `
  tests.generation.test_answer_generation `
  tests.retrieval.test_query_planning `
  tests.retrieval.test_retrieval_pipeline `
  tests.integration.test_rag_pipeline
```

重点检查：

1. 模型输出未知 `[C9]` 时会被拒绝，而不是自动附上所有来源。
2. 无 `PackedContext.citations` 时返回 `abstained`，且不调用 LLM。
3. 多 query 的候选在 rerank 前汇合，最终结果仍按一个 `top_k` 约束。
4. `fail_open` 的 query planner 会保留原始问题并写入回退信息。
5. 实际 prompt 与最大输出、安全余量的 token 总和超过模型窗口时，生成请求会在网络调用前失败。
6. `ApplicationFactory` 内的 QueryPlanner 与 AnswerGenerator 复用同一个 LLM Client，而不是各自新建连接或私自读取配置。

## 11. 推荐阅读顺序

1. `app/retrieval/query/models.py` 和 `stage.py`：先理解原始 query 为什么必须保留，以及回退语义为何集中在 Stage。
2. `app/retrieval/pipeline.py::search_queries()`：理解多 query 必须先合并候选，再做一次后处理。
3. `app/llm/models.py`、`base.py`、`registry.py`：理解业务代码如何摆脱具体 SDK。
4. `app/generation/prompts/answer.py`：审阅输入边界与 JSON 输出契约。
5. `app/generation/answering/grounded.py`：观察“无证据拒答 -> prompt 预算 -> 调用模型 -> 解析 -> citation 校验”的控制流。
6. `app/generation/citations/validator.py`：理解确定性校验能做什么、不能做什么。
7. `app/factory/llm.py`、`query.py`、`generation.py`、`pipelines.py`：沿对象创建路线理解依赖注入。
8. `app/pipeline.py`、`app/api/handlers/retrieval.py`：最后看完整流程如何进入 trace 与 API 契约。

## 12. 后续工程练习

以下练习都围绕完整工程能力，不要求补孤立函数或专门编写测试；对应测试应由项目维护。

### 练习 1：接入一个新的查询规划策略

为论文领域设计一个新的 `QueryPlanner`，例如“缩写与术语词典策略”或“实验条件保留策略”。要求它通过 `QueryPlannerRegistry` 注册，由 `GenerationSettings.query_planning.strategy` 选择，并能在 `QueryPlan` 与 trace 中说明使用了哪些变换。不要在 `RagPipeline` 中新增策略名判断。

完成后应能解释：为什么原始问题不能被覆盖、哪些改写失败应该 fail-open、以及多 query 会如何影响候选宽度、延迟和 rerank 成本。

### 练习 2：接入一个真实或本地模型 Provider

为一个你实际可用的模型服务实现独立 `LlmClient` adapter，并通过 `LlmClientRegistry` 注册。它必须把供应商响应转换成 `LlmResponse`，区分可重试与不可重试错误，不把密钥或完整 prompt 写入错误文本，并保持 `GroundedAnswerGenerator` 无须修改。

完成后应能解释：为什么 provider adapter 属于 `app/llm`、为什么 API key 只能在 Factory 注入、以及 Runtime 关闭时若 provider 有连接池该如何释放资源。

### 练习 3：实现面向 dense retrieval 的 HyDE 分支

当前 `QueryPlan` 已保留 `hyde_document`，但故意没有把它送入统一 BM25/hybrid 查询。为 vector 检索新增一个明确的 dense-query 分支：HyDE 文本只用于构造向量查询，永远不进入 `PackedContext`、不参与 citation，也不出现在最终回答。将该分支建模为清晰的检索策略或阶段，而不是在 `RagPipeline` 添加临时条件。

完成后应能说明：HyDE 为什么是检索辅助物、为什么它的幻觉不会自动污染回答、以及如何在 trace 中审计它是否启用。

### 练习 4：扩展回答级质量策略

在不让 `CitationValidator` 访问 Repository 的前提下，新增一个回答质量策略，例如“冲突资料显式披露”或“比较型问题必须至少引用两个不同来源”。它应以 Config、独立策略和组合校验的方式进入生成链路，并明确什么属于确定性校验、什么需要留给子模块 8 的评测。

完成后应能解释：为什么不能只用正则判断语义真实性，为什么 citation id 的来源映射必须来自 `PackedContext`，以及拒答为何是正常业务结果。

## 13. 本子模块边界

本子模块已经建立生成阶段的领域链路，但不会提前完成：

1. Golden dataset、回答指标、Ragas、实验矩阵和失败样例管理，它们属于子模块 8。
2. FastAPI 应用、SSE Streaming、HTTP 中间件、请求日志和应用状态注入，它们属于子模块 9。
3. 权限过滤、提示注入全链路防护、配额、限流、成本监控、部署与 CI，它们属于子模块 10。

完成本子模块后，你应该能从工程角度解释：为什么“生成文本”必须被视为待校验输出，为什么 query rewrite 与回答生成需要不同边界，为什么来源映射和 token 总预算必须贯穿整个问答链路，以及为何 Factory/Runtime 比业务类自行创建模型客户端更适合真实系统。
