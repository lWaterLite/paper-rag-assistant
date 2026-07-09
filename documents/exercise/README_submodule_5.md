# 子模块 5 练习说明：Baseline 检索与 BM25 检索

本练习对应模块 2 的子模块 5，主题是把前面已经构建好的索引真正用于在线检索。

这次代码生成遵循前面确认过的约束：功能实现尽量完整，代码结构尽量贴近真实工程；练习不聚焦零碎正则或局部函数，而是让你从整体结构、分层方式和后续扩展中学习。

---

## 1. 本子模块完成了什么

子模块 4 已经完成：

```text
DocumentChunk
  -> Embedding
  -> VectorCollection
  -> Repository
  -> Manifest
```

子模块 5 在此基础上补齐在线检索层：

```text
User Query
  -> SearchService
  -> RetrievalPipeline
  -> Retriever
  -> Result Stages
  -> RetrievedChunk[]
  -> /search
  -> 后续 /ask 与 context packing
```

本次实现的重点不是“写一个能跑的 BM25 函数”，而是建立一套后续可以扩展到 hybrid retrieval、rerank、query rewrite 和 evaluation 的检索子系统。

---

## 2. 本次生成和调整的代码结构

核心新增或调整文件：

```text
app/retrieval/configs.py
app/retrieval/pipeline.py
app/retrieval/service.py
app/retrieval/retrievers/__init__.py
app/retrieval/retrievers/base.py
app/retrieval/retrievers/vector.py
app/retrieval/retrievers/bm25.py
app/retrieval/retrievers/result_builder.py
app/api/handlers.py
app/core/settings.py
app/factory/__init__.py
app/factory/application.py
app/factory/configs.py
app/factory/ingestion.py
app/factory/indexing.py
app/factory/retrieval.py
app/factory/pipelines.py
app/main.py
settings.toml
tests/test_retrieval_pipeline.py
tests/test_search_service.py
tests/test_bm25_retriever.py
tests/test_config_settings.py
tests/test_rag_pipeline.py
```

整体分层如下：

```text
retrieval/configs.py
  检索运行时 Config

retrieval/pipeline.py
  RetrievalPipeline
  ChunkIdDeduplicationStage
  TopKLimitStage

retrieval/service.py
  SearchService

retrieval/retrievers/base.py
  检索器协议

retrieval/retrievers/vector.py
  VectorRetriever

retrieval/retrievers/bm25.py
  BM25Index
  BM25Retriever

retrieval/retrievers/result_builder.py
  RetrievedChunkBuilder

api/handlers.py
  SearchRequest -> SearchService -> SearchResponse

factory/
  使用 ApplicationFactory 统一管理 settings 和依赖组装

main.py
  CLI search 命令
```

---

## 3. 为什么要拆分 retrieval 包

旧结构中，`VectorRetriever`、`BM25Retriever`、去重函数和结果组装函数散落在 `retrieval` 包下。

这在很小的 demo 里可以接受，但实际工程会很快膨胀：

```text
VectorRetriever
BM25Retriever
HybridRetriever
Reranker
QueryRewriter
SearchService
RetrievalConfig
去重策略
分数归一化
评测辅助
```

如果全部放在一个文件里，后续会很难维护。

所以本次改成：

```text
一个文件表达一个清晰职责
retrieval/retrievers/
  只放检索器协议和具体检索器实现

retrieval/pipeline.py
  收束一次完整检索流程

retrieval/service.py
  保留为 API 和 CLI 面向的应用服务入口
```

调用方可以从稳定聚合入口导入检索器：

```python
from app.retrieval.retrievers import BM25Retriever
```

内部实现可以更精确地导入：

```python
from app.retrieval.retrievers.bm25 import BM25Retriever
from app.retrieval.retrievers.vector import VectorRetriever
from app.retrieval.service import SearchService
```

---

## 4. `retrievers/base.py`：检索器协议

`Retriever` 协议定义了检索器的最小公共接口：

```python
def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
    ...
```

不管底层实现是：

```text
vector
bm25
hybrid
rerank
```

都应该返回统一的 `RetrievedChunk`。

这就是依赖倒置：

```text
RagPipeline
  -> Retriever 协议
  -> VectorRetriever / BM25Retriever / 未来 HybridRetriever
```

`RagPipeline` 不应该知道 BM25 公式，也不应该知道向量库怎么搜索。它只需要一个能返回 `RetrievedChunk[]` 的检索器。

---

## 5. `configs.py`：检索运行时配置

本次新增：

```python
BM25Config
RetrievalConfig
```

### `BM25Config`

字段：

```text
k1
b
```

`k1` 控制词频增长的饱和速度。

`b` 控制文档长度归一化强度。

默认值：

```text
k1 = 1.5
b = 0.75
```

### `RetrievalConfig`

字段：

```text
strategy
top_k
bm25
deduplicate_by_chunk_id
```

它表示一次检索服务运行时需要的配置。

注意这里继续遵守前面的命名约束：

```text
Settings：外部配置文件结构
Config：功能类真正接收的运行时配置
```

所以：

```text
ProjectSettings.retrieval
  -> ApplicationFactory.configs
  -> RetrievalConfig
  -> SearchService / BM25Retriever
```

---

## 6. `vector.py`：VectorRetriever

`VectorRetriever` 的职责是：

```text
query
  -> embedding_client.embed_text(query)
  -> vector_collection.search(query_vector, top_k)
  -> chunk_collection.get_by_id(chunk_id)
  -> RetrievedChunk
```

它依赖三个对象：

```text
EmbeddingClient
VectorCollection
ChunkCollection
```

这很重要。

`VectorRetriever` 不直接读 JSON，不知道向量如何持久化，也不负责构建索引。

它只在在线检索阶段工作。

如果 `VectorCollection` 命中了一个 `chunk_id`，但 `ChunkCollection` 中找不到对应 chunk，系统会抛出清晰错误。

这说明索引产物已经不一致，例如：

- vector_collection.json 损坏。
- chunk_collection.json 丢失。
- 构建流程或加载流程有 bug。

这种错误不能静默跳过。

---

## 7. `retrievers/bm25.py`：BM25Index 与 BM25Retriever

本次把 BM25 拆成两层：

```text
BM25Index
  负责保存 chunks、token、词频、文档频率、平均长度等统计信息

BM25Retriever
  负责调用 BM25Index，并转换成统一 RetrievedChunk
```

这样做比把所有逻辑都写在 `BM25Retriever` 里更清晰。

### `BM25Index`

构建时会计算：

```text
_chunks
_tokenized_chunks
_term_frequencies
_document_frequencies
_average_document_length
```

查询时：

```text
query
  -> index 持有的 Tokenizer.tokenize(query)
  -> 对每个 chunk 计算 BM25 score
  -> 过滤 score <= 0 的 chunk
  -> 按 score 降序
  -> 返回 BM25SearchHit
```

### `BM25Retriever`

`BM25Retriever` 只接收已经构建完成的：

```python
BM25Index
```

```python
index = BM25Index.from_chunks(
    chunks,
    config=config,
    tokenizer=tokenizer,
)
retriever = BM25Retriever(index)
```

这种边界确保 BM25Retriever 不负责索引构建，也避免它自行创建配置或 tokenizer。
后续实现“BM25 index 持久化”或“启动时构建、请求时复用”时，检索器接口无需变化。

---

## 8. Tokenizer 策略

当前内置的 `RegexTokenizer` 是轻量正则分词器：

```text
英文、数字、下划线：按连续词提取
中文：按单字提取
```

示例：

```text
"RAG evaluation"
  -> ["rag", "evaluation"]

"检索增强生成"
  -> ["检", "索", "增", "强", "生", "成"]
```

它的优点是无第三方依赖，便于学习 BM25 主流程。

它的局限是中文词组会被拆散。

项目通过 `Tokenizer` Protocol 和 `TokenizerRegistry` 支持替换策略：

```text
RegexTokenizer
JiebaTokenizer
OpenSearchAnalyzerTokenizer
```

`RetrievalFactory` 根据 `settings.toml` 中的
`retrieval.tokenizer.strategy` 选择实现，并把同一个 tokenizer 注入
`BM25Index`。索引文本与查询文本因此始终采用一致的分词规则。

---

## 9. `retrievers/result_builder.py`：统一组装 RetrievedChunk

`VectorRetriever` 和 `BM25Retriever` 最终都要生成 `RetrievedChunk`。

如果每个检索器自己手写一遍字段映射，很容易出现不一致：

```text
vector 保留 metadata
bm25 忘记 metadata

vector 保留 page_start
bm25 忘记 page_start
```

所以本次抽出：

```python
RetrievedChunkBuilder.from_chunk(...)
```

它统一把 `DocumentChunk` 转成 `RetrievedChunk`。

不同检索器只需要传入：

```text
score
rank
retriever
```

---

## 10. `pipeline.py`：RetrievalPipeline

`RetrievalPipeline` 用来封装一次完整 retrieval 流程。

它负责：

```text
1. 清洗和校验 query
2. 解析 top_k
3. 根据策略选择 Retriever
4. 调用 retriever.retrieve
5. 执行结果后处理阶段
6. 记录 trace
7. 返回 RetrievalPipelineResult
```

它不负责：

- 构建索引。
- 解析 PDF。
- 生成回答。
- 直接处理 HTTP。
- 持久化 vector 或 chunk。

这样做的意义是：API、CLI、测试和后续 `RagPipeline` 都不需要各自拼装检索细节。

### 后处理阶段

当前默认后处理阶段包括：

```text
ChunkIdDeduplicationStage
  按 chunk_id 去重，并重新分配 rank

TopKLimitStage
  截断最终返回数量
```

这比把一个 `deduplicate_by_chunk_id(...)` 函数单独放在文件里更清晰。

因为去重不是孤立工具函数，它是 retrieval pipeline 的一个阶段。

后续要接入更多检索处理能力时，可以继续增加阶段：

```text
ScoreNormalizeStage
RerankStage
CitationBoostStage
MetadataFilterStage
```

### 为什么去重是 pipeline 阶段

虽然当前 vector 和 BM25 是单检索器检索，但后续 hybrid retrieval 很容易出现：

```text
VectorRetriever 命中 chunk_a
BM25Retriever 也命中 chunk_a
```

如果不去重，后续 context packing 会浪费上下文窗口。

把去重放在 pipeline 中，意味着无论请求来自 API、CLI 还是后续问答流程，都会共享同一套后处理规则。

---

## 11. `service.py`：SearchService

`SearchService` 现在是一个很薄的应用服务入口。

它不再自己处理去重、trace 和 top_k 截断，而是委托给 `RetrievalPipeline`：

```text
SearchService.search
  -> RetrievalPipeline.search
  -> RetrievalPipelineResult
```

保留 `SearchService` 的原因是：它表达的是应用层语义，也就是“对外提供 search 能力”。

而 `RetrievalPipeline` 表达的是领域流程语义，也就是“如何完成一次 retrieval”。

这两个概念不要混在一起：

```text
API handler
  -> SearchService
  -> RetrievalPipeline
  -> Retriever
```

---

### 为什么需要 SearchService

如果没有 SearchService，检索逻辑很容易散落在：

```text
CLI
API route
RagPipeline
测试代码
```

每个地方都自己选择 retriever、处理 top-k、记录 trace，就会重复且容易不一致。

现在这句话可以拆得更准确：

```text
SearchService
  收束“对外提供 search 能力”的应用语义

RetrievalPipeline
  收束“完成一次 retrieval”的领域流程
```

---

## 12. `api/handlers.py`：/search 处理函数

当前项目暂时不引入 FastAPI，但已经可以把 API 边界稳定下来。

本次新增：

```python
handle_search_request(request: SearchRequest, search_service: SearchService) -> SearchResponse
```

它做的事情很简单：

```text
SearchRequest
  -> SearchService.search
  -> SearchResponse
```

这样后续接入 FastAPI 时，route 只需要：

```python
@router.post("/search")
def search(request: SearchRequest):
    return handle_search_request(request, search_service)
```

API 层不应该直接知道 BM25 公式，也不应该直接访问 vector collection。

---

## 13. `factory/`：统一组装检索依赖

`factory` 现在是一个软件包，而不是一个巨大的无状态函数文件。

核心入口是：

```python
factory = ApplicationFactory(env_settings=env_settings, project_settings=project_settings)
```

这个对象持有同一组：

```text
EnvSettings
ProjectSettings
ChunkerRegistry
```

并把对象组装拆到几个更小的工厂中：

```text
ConfigFactory
  Settings -> Config

IngestionFactory
  loader / parser / cleaner / chunker

IndexingFactory
  embedding / repository / collection / index builder

RetrievalFactory
  vector retriever / BM25 retriever / search service

PipelineFactory
  RAG pipeline
```

这样做的重点不是“多写几个类”，而是避免整个项目到处传递 settings。

主入口现在可以只创建一次组合根：

```python
factory = ApplicationFactory(env_settings=env_settings, project_settings=project_settings)
```

然后继续创建不同对象：

```python
index_builder = factory.build_index_builder()
search_service = factory.build_search_service(index)
rag_pipeline = factory.build_rag_pipeline(index)
```

现在默认问答 pipeline 不再固定使用向量检索，而是根据：

```text
EnvSettings.retrieval_strategy
```

选择：

```text
vector
bm25
```

如果选择 `hybrid`，当前会清晰报错：

```text
hybrid 检索将在后续子模块实现
```

这比静默 fallback 到 vector 更好。

### 配置流向

配置流向如下：

```text
.env
  -> EnvSettings.retrieval_strategy
  -> RetrievalConfig.strategy

settings.toml
  -> ProjectSettings.retrieval
  -> BM25Config
```

也就是说：

```text
检索策略、top_k
  偏运行时请求/环境行为，放在 EnvSettings

BM25 k1/b、去重策略
  偏工程结构化配置，放在 settings.toml
```

---

## 14. `main.py`：search CLI

本次新增 CLI：

```bash
python -m app.main search "RAG 为什么需要引用？" --source data/raw/papers
```

也可以加载已有索引：

```bash
python -m app.main search "faithfulness evaluation" --use-existing-index --retriever bm25 --top-k 5
```

参数：

```text
query
--source
--use-existing-index
--top-k
--retriever vector|bm25
```

这个命令只展示检索结果，不生成回答。

它的价值是排查：

- BM25 是否命中精确术语。
- vector 是否命中语义相近内容。
- top-k 是否合适。
- source、section、metadata 是否完整。

---

## 15. 测试覆盖

本次新增和更新的测试包括：

```text
tests/test_search_service.py
tests/test_bm25_retriever.py
tests/test_config_settings.py
tests/test_rag_pipeline.py
tests/test_api_schemas.py
```

覆盖点包括：

1. BM25 返回排序后的 RetrievedChunk。
2. BM25 保留 metadata。
3. BM25 支持 top-k 边界。
4. SearchService 能按默认策略检索。
5. SearchService 支持请求级 retriever override。
6. SearchService 能按 chunk_id 去重。
7. SearchService 对不支持的策略清晰失败。
8. API handler 能把 SearchRequest 映射成 SearchResponse。
9. factory 能让 pipeline 使用配置指定的 BM25 retriever。
10. TOML 中 `[retrieval]` 配置能被读取。

---

## 16. 如何运行

构建索引：

```bash
python -m app.main index --source data/raw/papers
```

执行向量检索：

```bash
python -m app.main search "RAG 为什么需要引用？" --retriever vector --top-k 5
```

执行 BM25 检索：

```bash
python -m app.main search "faithfulness evaluation" --retriever bm25 --top-k 5
```

加载已有索引后检索：

```bash
python -m app.main search "faithfulness evaluation" --use-existing-index --retriever bm25 --top-k 5
```

切换问答 pipeline 默认检索器：

```bash
RAG_RETRIEVAL_STRATEGY=bm25 python -m app.main ask "faithfulness evaluation" --source data/raw/papers
```

Windows PowerShell 可以使用：

```powershell
$env:RAG_RETRIEVAL_STRATEGY = "bm25"
python -m app.main ask "faithfulness evaluation" --source data/raw/papers
```

---

## 17. 如何运行测试

运行子模块 5 相关测试：

```bash
python -B -m unittest tests.test_bm25_retriever tests.test_search_service tests.test_api_schemas tests.test_config_settings tests.test_rag_pipeline
```

运行全量测试：

```bash
python -B -m unittest discover -s tests
```

---

## 18. 当前实现的边界

当前子模块 5 仍然保持 baseline 定位。

已经完成：

```text
VectorRetriever
BM25Retriever
SearchService
/search handler
search CLI
配置接入
测试覆盖
```

暂未实现：

```text
hybrid retrieval
score normalization
rerank
query rewrite
BM25 index 持久化
专业中文分词器
真实 HTTP 服务
```

这些内容会在后续子模块继续推进。

---

## 19. 练习 1：设计 Tokenizer 策略

当前 BM25 使用可注入的 `Tokenizer` 协议。

这个实现适合教学，但真实工程里 tokenizer 往往需要可替换。

本练习已经实现以下 tokenizer 策略结构：

```text
Tokenizer Protocol
RegexTokenizer
TokenizerConfig
TokenizerRegistry
未来 JiebaTokenizer
未来 OpenSearchAnalyzerTokenizer
```

要求：

1. 不要让 `BM25Index` 直接依赖某个具体 tokenizer 函数。
2. factory 可以根据配置选择 tokenizer。
3. 默认 tokenizer 仍然不需要第三方依赖。
4. 结构要允许后续引入中文分词器。

这个练习重点是工程结构设计，不是写复杂分词算法。

---

## 20. 练习 2：设计 Hybrid Retrieval 的接口位置

本次代码还没有实现 hybrid retrieval，但已经为它留下位置：

```text
Retriever 协议
SearchService retrievers registry
dedup.py
RetrievalConfig.strategy = "hybrid"
```

请你设计 `HybridRetriever` 应该放在哪里，以及它应该依赖什么。

建议思考：

1. 它是否应该同时依赖 `VectorRetriever` 和 `BM25Retriever`？
2. 它是否应该先分别取 top-k，再做合并？
3. 分数不能直接比较时，应使用什么融合策略？
4. 去重应该在 HybridRetriever 内部做，还是交给 SearchService？
5. 它返回的 `retriever` 字段应该是 `hybrid`，还是保留原始来源？

这个练习的重点是检索策略组合，不是立刻实现复杂排序算法。

---

## 21. 练习 3：设计 Search Debug Report

`SearchService` 当前已经返回 trace，但真实排查检索问题时可能还需要更详细的 debug report。

你可以设计一个：

```text
SearchDebugReport
```

它可以记录：

```text
query
resolved_top_k
retriever
candidate_count
deduplicated_count
returned_count
index_id
embedding_model
bm25_k1
bm25_b
latency_ms
```

要求：

1. 不要把 report 写入 retriever。
2. 不要让 API schema 直接依赖内部 report 数据结构。
3. 思考 report 应该由 SearchService 生成，还是由单独 writer 生成。

这个练习重点是排障能力设计。

---

## 22. 练习 4：让 `/search` 支持比较模式

现在 `/search` 一次只使用一个 retriever。

你可以设计一个比较模式：

```json
{
  "query": "faithfulness evaluation",
  "top_k": 5,
  "compare": ["vector", "bm25"]
}
```

响应中分别返回两个检索器的结果。

要求：

1. 不要破坏现有 SearchRequest。
2. 思考是否应该新增 CompareSearchRequest。
3. 返回结构中要能清楚区分不同 retriever。
4. 不要直接比较 vector score 和 BM25 score。

这个练习重点是 API 契约设计。

---

## 23. 子模块 5 验收标准

完成本子模块后，你应该能做到：

1. 解释 `Retriever` 协议为什么存在。
2. 解释 `VectorRetriever` 的在线检索流程。
3. 解释 `BM25Index` 和 `BM25Retriever` 为什么拆开。
4. 解释 BM25 的 TF、IDF、长度归一化。
5. 说明 vector 和 BM25 的分数为什么不能直接比较。
6. 使用 CLI 分别运行 vector 与 BM25 检索。
7. 使用 SearchService 获得只检索不生成的结果。
8. 说明 `/search` 和 `/ask` 的职责差异。
9. 理解当前 baseline 与后续 hybrid/rerank 的关系。
10. 能提出一个合理的 tokenizer 或 hybrid retrieval 扩展方案。
