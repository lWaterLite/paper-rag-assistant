# 子模块 4 练习说明：Embedding、向量索引与持久化存储

本练习对应模块 2 的子模块 4，主题是把前面已经解析、清洗、切分好的 `DocumentChunk` 转换成可检索、可复现、可持久化的向量索引。

这份文档延续子模块 3 的方式：先讲清楚本次生成的工程代码，再给出模块级练习。练习重点是理解“索引子系统应该如何分层、如何配置、如何持久化、如何支持后续替换真实向量库”，而不是让你把精力放在零散算法细节上。

## 学习目标

完成本子模块后，你应该能理解：

1. 为什么 RAG 系统需要把 chunk 转成 embedding，再写入向量运行时集合。
2. 为什么 embedding client、vector collection、repository、embedding cache、manifest 都应该有独立接口。
3. 为什么离线索引构建必须支持配置化、幂等、缓存、持久化和构建报告。
4. 为什么 mock embedding 只能验证工程流程，不能验证真实检索质量。
5. 如何设计一个后续可以替换为 FAISS、Chroma、Qdrant、pgvector 的向量检索与持久化适配层。
6. 如何用 manifest 记录索引版本和关键配置，避免实验结果不可追溯。

## 本次生成的代码结构

核心文件如下：

1. `app/indexing/configs.py`
   - 定义 indexing 子系统的运行时 `Config`。
   - 包括 `EmbeddingConfig`、`VectorRepositoryConfig`、`IndexBuilderConfig`。
2. `app/indexing/embeddings.py`
   - 定义 `EmbeddingClient` 协议。
   - 实现 `MockEmbeddingClient`。
   - 预留并实现可选的 `OpenAIEmbeddingClient`。
   - 提供 embedding 结果维度和数值校验。
3. `app/indexing/embedding_cache.py`
   - 定义 `EmbeddingCache` 协议。
   - 实现 `InMemoryEmbeddingCache` 和 `FileEmbeddingCache`。
4. `app/indexing/vector_collection.py`
   - 定义 `VectorCollection` 协议。
   - 实现 `InMemoryVectorCollection`。
   - 只负责内存向量管理和余弦相似度搜索，不负责文件读写。
5. `app/indexing/manifest.py`
   - 定义 `IndexManifest`。
   - 提供 manifest 与当前配置的兼容性校验。
6. `app/repositories/manifest.py`
   - 定义 `IndexManifestRepository`，负责 manifest JSON 文件读写。
7. `app/indexing/report.py`
   - 定义 `IndexBuildReportWriter`。
   - 把一次索引构建结果写成稳定 JSON 报告。
8. `app/indexing/index_builder.py`
   - 离线索引构建主流程。
   - 串联 ingestion、chunking、embedding cache、vector collection、repository、manifest 和 build report。
9. `app/factory/`
   - 项目的 composition root。
   - 把 `ProjectSettings` 转换成各功能类接收的 `Config`。
   - 根据配置选择 mock/openai embedding，并通过 Registry 创建 `local_json` 向量 Repository。
10. `app/core/settings.py`
   - 增加 `EmbeddingSettings`、`VectorRepositorySettings`、`IndexBuilderSettings`。
11. `settings.toml`
   - 增加 `[indexing.embedding]`、`[indexing.vector_repository]`、`[indexing.builder]` 配置段。
12. `tests/test_embedding_clients.py`
   - 测试 mock embedding、维度校验和 OpenAI provider 缺 key 的错误。
13. `tests/test_embedding_cache.py`
   - 测试内存 cache 和文件 cache。
14. `tests/test_vector_collection.py`
   - 测试内存向量集合和本地 JSON 向量 Repository。
15. `tests/test_index_manifest.py`
   - 测试 manifest 构建、读写和兼容性校验。
16. `tests/test_index_builder_embedding_cache.py`
   - 测试 IndexBuilder 的 cache 复用、跳过已有 chunk、报告写入和本地索引持久化。

## 整体数据流

子模块 4 接在子模块 3 后面：

```text
RawDocument
  -> Parser / Cleaner
  -> ParsedDocument
  -> Chunker
  -> DocumentChunk
  -> EmbeddingClient
  -> EmbeddingCache
  -> VectorCollection
  -> VectorRepository
  -> IndexManifest
  -> IndexBuildReport
```

在线检索阶段会使用子模块 4 的产物：

```text
User Query
  -> EmbeddingClient.embed_text
  -> VectorCollection.search
  -> ChunkCollection.get_by_id
  -> RetrievedChunk
```

注意离线和在线的区别：

1. 离线索引构建会处理大量文档和 chunks，重点是可恢复、可缓存、可复现。
2. 在线检索只处理用户 query，重点是低延迟和稳定返回结果。

## 配置结构讲解

### `settings.toml`

子模块 4 新增三段配置：

```toml
[indexing.embedding]
provider = "mock"
model = "mock-hash-embedding"
dimension = 16
batch_size = 32
timeout_seconds = 30.0
max_retries = 2

[indexing.vector_repository]
type = "local_json"
index_dir = "data/indexes"
collection_name = "papers_baseline"
distance_metric = "cosine"

[indexing.builder]
manifest_filename = "manifest.json"
build_report_filename = "index_build_report.json"
skip_existing = true
fail_on_empty_chunk = true
```

这里仍然遵守之前讨论过的配置原则：

1. `.env` 只保存 API key、令牌等敏感项。
2. `settings.toml` 保存结构化、非敏感、会影响工程行为的配置。
3. 功能类不直接读取 `.env` 或 TOML。
4. factory 负责把外部 `Settings` 转换成功能类使用的 `Config`。

### `EmbeddingSettings` 与 `EmbeddingConfig`

`EmbeddingSettings` 位于 `app/core/settings.py`，代表 TOML 中 `[indexing.embedding]` 的形状。

`EmbeddingConfig` 位于 `app/indexing/configs.py`，代表 embedding client 真正接收的配置。

这两个对象分开，是为了避免功能模块直接依赖外部配置系统。

### `VectorRepositorySettings` 与 `VectorRepositoryConfig`

`VectorRepositorySettings` 负责从 TOML 读取：

```text
type
index_dir
collection_name
distance_metric
```

`VectorRepositoryConfig` 则提供运行时需要的路径推导：

```text
collection_dir
vector_collection_path
chunk_collection_path
document_collection_path
embedding_cache_path
```

例如：

```text
data/indexes/papers_baseline/
  vector_collection.json
  chunk_collection.json
  document_collection.json
  embedding_cache.json
  manifest.json
  index_build_report.json
```

### 为什么默认值也是 local_json

向量索引的运行时数据由 `InMemoryVectorCollection` 管理；只要需要跨进程复用，持久化边界就必须是 Repository。因此 `VectorRepositorySettings` 的默认值与 `settings.toml` 一样，均为 `local_json`。

真实项目运行时会调用：

```python
ProjectSettings.from_toml()
```

它会读取项目根目录下的 `settings.toml`，从而启用 `local_json` 持久化。

这是一种常见工程取舍：

1. 类默认值尽量安全、轻量、无副作用。
2. 项目配置文件定义真实运行策略。
3. 测试中需要持久化时显式传入临时目录。

## `app/indexing/configs.py` 代码讲解

这个文件只放功能类运行时配置，不放读取 TOML 的逻辑。

### `EmbeddingConfig`

字段包括：

```text
provider
model
dimension
batch_size
timeout_seconds
max_retries
```

它会校验：

1. `model` 不能为空。
2. `dimension` 必须大于 0。
3. `batch_size` 必须大于 0。
4. `timeout_seconds` 必须大于 0。
5. `max_retries` 必须大于等于 0。
API key 不属于 `EmbeddingConfig`，而是由 `EnvSettings` 读取后在 factory
组装阶段显式注入 `OpenAIEmbeddingClient`。

### `VectorRepositoryConfig`

字段包括：

```text
repository_type
index_dir
collection_name
distance_metric
```

它额外提供三个路径属性：

```python
collection_dir
vector_collection_path
chunk_collection_path
document_collection_path
embedding_cache_path
```

这样路径拼接集中在配置对象里，业务流程不用到处手写：

```python
index_dir / collection_name / "xxx.json"
```

### `IndexBuilderConfig`

字段包括：

```text
manifest_filename
build_report_filename
skip_existing
fail_on_empty_chunk
```

`skip_existing` 控制重复构建时是否跳过已经写入 `VectorCollection` 的 chunk。

`fail_on_empty_chunk` 控制发现空 chunk 时是否直接失败。当前默认严格失败，因为空 chunk 往往说明前面的 parser、cleaner 或 chunker 有问题。

## `app/indexing/embeddings.py` 代码讲解

### `EmbeddingClient`

`EmbeddingClient` 是协议接口：

```python
class EmbeddingClient(Protocol):
    provider: str
    model_name: str
    dimension: int
    def embed_text(self, text: str) -> list[float]: ...
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...
```

`IndexBuilder` 只依赖这个协议，不关心底层是 mock、OpenAI、本地模型还是企业内部服务。

这就是依赖倒置：

```text
IndexBuilder -> EmbeddingClient 协议
MockEmbeddingClient -> 实现协议
OpenAIEmbeddingClient -> 实现协议
未来 BGEEmbeddingClient -> 实现协议
```

### `MockEmbeddingClient`

mock embedding 用文本 hash 生成稳定向量：

```text
同一 model + 同一 text -> 同一 vector
```

这让测试可复现。

本次修正了一个容易忽略的问题：`hashlib.blake2b` 单次 digest 最多 64 字节。现在 mock 实现会按轮次扩展 hash，因此可以支持 128、384、768 等更大的 mock 维度。

但它仍然不理解真实语义。

它能验证：

1. pipeline 是否跑通。
2. cache 是否命中。
3. vector collection 是否能写入和搜索。
4. manifest 是否记录正确。

它不能验证：

1. RAG 检索质量。
2. 中文 query 检索英文论文的效果。
3. 语义相近但关键词不同的召回能力。

### `OpenAIEmbeddingClient`

`OpenAIEmbeddingClient` 是真实 provider 的接口层。

它有几个工程特点：

1. 只有配置选择 `provider = "openai"` 时才会被构造。
2. 只有构造它时才懒加载 `openai` SDK。
3. API key 由 factory 从 `EnvSettings.openai_api_key` 显式注入。
4. 不会把 API key 写入 TOML、manifest、report 或日志。
5. 会校验返回向量数量和维度。

如果你希望使用它，需要自行添加依赖：

```bash
pip install openai
```

然后在环境中设置：

```bash
OPENAI_API_KEY=你的真实 key
```

再把 `settings.toml` 改为：

```toml
[indexing.embedding]
provider = "openai"
model = "text-embedding-3-small"
dimension = 1536
```

OpenAI 官方文档说明，`text-embedding-3-small` 默认维度是 1536，`text-embedding-3-large` 默认维度是 3072，也支持通过 `dimensions` 参数缩短维度。参考：[OpenAI Embeddings Guide](https://developers.openai.com/api/docs/guides/embeddings)。

### `validate_embedding_vectors`

这个函数校验：

1. 返回向量数量是否等于输入文本数量。
2. 每个向量维度是否等于配置维度。
3. 向量里是否包含 `NaN` 或 `Infinity`。

维度校验非常重要。不同 embedding model 的向量不能混在同一个索引里。

## `app/indexing/embedding_cache.py` 代码讲解

### `EmbeddingCacheKey`

cache key 包含：

```text
provider
model_name
dimension
text_hash
```

这能避免错误复用旧 embedding。

例如：

```text
mock / mock-hash-embedding / 16 / hash(text)
openai / text-embedding-3-small / 1536 / hash(text)
```

即使文本相同，只要模型或维度不同，缓存就不会命中。

### `InMemoryEmbeddingCache`

内存 cache 用于：

1. 快速测试。
2. 单进程实验。
3. 不希望写文件的场景。

程序结束后会丢失。

### `FileEmbeddingCache`

文件 cache 用于本地持久化。

它写入：

```text
data/indexes/papers_baseline/embedding_cache.json
```

注意一个工程细节：

> 目录创建不放在 cache 的 `persist` 方法里，而由 `IndexBuilder` 在流程准备阶段统一完成。

这是延续你之前提到的职责边界：writer/cache/store 只负责写文件，目录准备属于流程编排。

## `app/indexing/vector_collection.py` 代码讲解

### `VectorCollection`

协议定义：

```python
add(record)
search(query_vector, top_k)
count()
contains_chunk(chunk_id)
iter_records()
dimension
```

`VectorCollection` 是运行时对象，只管理已经进入内存的向量记录。它不读写 JSON，也不知道数据来自本地文件、SQLite、Qdrant 还是 pgvector。

`iter_records()` 返回的是 `Iterable[VectorRecord]`，而不是 `list[VectorRecord]`。
这表示调用方只应该依赖“可以遍历记录”，不应该假设 collection 会一次性复制全部数据。
后续替换真实向量库或处理更大规模集合时，这个接口更容易改成分页、游标或流式遍历。

### `VectorRecord`

`VectorRecord` 是向量集合中的轻量记录：

```text
chunk_id
vector
metadata
```

这里不再保存完整 `DocumentChunk`。完整文本、页码、章节等引用信息由 `ChunkCollection` 管理，向量集合只保存检索所需的最小数据。

### `InMemoryVectorCollection`

当前实现是精确余弦相似度检索，负责：

1. 保存 `chunk_id -> VectorRecord`。
2. 首次写入时确定向量维度。
3. 后续写入和查询时校验维度一致。
4. 检索时使用 cosine similarity 排序。
5. 返回 `VectorSearchResult`，而不是直接返回完整 chunk。

它不是生产级向量库，但它是一个完整、可运行、可测试、无第三方依赖的 baseline。

## `app/repositories/vector.py` 代码讲解

### `VectorRepository`

协议定义：

```python
load() -> VectorCollection
save(collection: VectorCollection) -> None
```

Repository 只处理持久化边界，不做相似度搜索。

### `LocalJsonVectorRepository`

本地 JSON Repository 用于小规模持久化 baseline。

它保存：

```text
dimension
records:
  chunk_id
  vector
  metadata
```

加载时它会校验 JSON 中声明的 `dimension` 是否与每条 record 的实际 vector 长度一致。
这属于 Repository 的防腐层职责：外部文件可能被手动修改、写坏或来自旧版本程序，不能直接信任。

这样重新启动后可以加载已有向量集合，并继续搜索。

它的限制也很明确：

1. JSON 不适合非常大的向量集合。
2. 并发写入没有事务保护。
3. 搜索仍然发生在 `InMemoryVectorCollection` 中，数据量大时需要换成专业向量库。

但它的好处是：

1. 无需安装第三方库。
2. 结构透明，方便学习。
3. 运行时集合与持久化边界已经拆开，后续更容易替换实现。

## `app/indexing/manifest.py` 代码讲解

### `IndexManifest`

manifest 记录索引如何生成：

```text
index_id
schema_version
status
parent_index_id
source_dir
created_at
chunker
chunk_size
chunk_overlap
embedding_provider
embedding_model
embedding_dimension
embedding_batch_size
vector_repository_type
vector_collection_name
distance_metric
document_count
chunk_count
vector_count
config_hash
document_set_hash
document_versions
```

这比只保存一个 `index_id` 更有价值，因为你能追溯：

1. 用了哪个 embedding model。
2. 用了什么 chunking 配置。
3. 写入了多少文档和 chunk。
4. 用了什么 vector repository。
5. 文档版本是否变化。

### `config_hash`

`config_hash` 只由索引构建配置生成，不包含 `document_versions`。

其中 `source_dir` 会先规范化为绝对 POSIX 路径，再进入 `config_hash`。这样 `data/raw/papers` 和它对应的绝对路径不会被误判成两个不同索引配置。

它解决的问题是：

```text
同一个 collection 名称下，构建配置是否发生变化？
```

如果文档内容变化但 chunker、embedding、vector repository 等配置不变，`config_hash` 仍然应该保持不变。

### `document_set_hash`

`document_set_hash` 由 `document_versions` 生成。

它解决的问题是：

```text
同一个 collection 名称下，输入文档集合是否发生变化？
```

把它和 `config_hash` 拆开后，我们可以判断索引变化到底来自配置变化，还是来自语料变化。

后续做实验比较时，manifest 是非常重要的证据。

### `IndexManifestRepository`

负责 manifest 与本地 JSON 文件之间的读写。

它只写文件，不创建目录。目录由 `IndexBuilder` 准备。

### `validate_manifest_compatible`

校验已有 manifest 是否和当前配置兼容。

当前检查：

1. embedding provider。
2. embedding model。
3. embedding dimension。
4. vector repository type。
5. distance metric。

如果不兼容，应拒绝加载旧索引，避免把不同语义空间的向量混在一起。

## `app/indexing/report.py` 代码讲解

`IndexBuildReportWriter` 负责把一次构建结果写成 JSON。

报告包含：

```text
index_id
status
document_count
chunk_count
vector_count
embedding_cache_hits
embedding_cache_misses
skipped_existing_chunks
empty_chunk_count
manifest_path
ingestion_report_path
chunking_report_path
manifest
trace
```

它和 manifest 的区别：

1. manifest 描述索引本身是什么。
2. build report 描述这一次构建过程发生了什么。

manifest 偏“索引说明书”，build report 偏“构建日志摘要”。

## `app/indexing/index_builder.py` 代码讲解

`IndexBuilder` 是子模块 4 的主流程编排者。

它接收的依赖包括：

```text
IndexBuilderConfig
EmbeddingConfig
VectorRepositoryConfig
IngestionPipeline
Chunker
EmbeddingClient
EmbeddingCache
VectorCollection
DocumentCollection
ChunkCollection
VectorRepository
DocumentRepository
ChunkRepository
IndexManifestRepository
IndexBuildReportWriter
IngestionReportWriter
ChunkingReportWriter
```

这看起来依赖很多，但这是有意设计的。

原因是：

1. IndexBuilder 不应该自己偷偷 new 一个默认 embedding client。
2. IndexBuilder 不应该自己偷偷选择 vector collection 或 repository。
3. IndexBuilder 不应该读取 TOML。
4. IndexBuilder 只负责流程编排，依赖由 factory 统一注入。

这延续了你前面确认过的工厂管理范式。

### 构建流程

`build_from_directory` 的主流程：

```text
1. 准备输出目录
2. ingestion
3. 保存 raw 和 parsed document 到 repository
4. 写 ingestion report
5. chunking
6. 保存 chunks 到 repository
7. 写 chunking report
8. 过滤空 chunk
9. 根据 skip_existing 判断要写入哪些 chunk
10. 使用 embedding cache 生成向量
11. 写入 vector collection
12. 持久化 cache、vector collection、document collection 和 chunk collection
13. 构建并写入 manifest
14. 写入 index build report
15. 返回 RagIndex 和 IndexBuildResult
```

### `skip_existing`

如果 `skip_existing = true`，重复构建时：

```text
已经在 vector collection 中的 chunk 不再重新 embedding
```

这就是幂等构建的一部分。

### `fail_on_empty_chunk`

如果发现空 chunk，默认直接失败。

原因是空 chunk 往往说明前面流程出现问题：

1. parser 解析出空文本。
2. cleaner 把内容清空了。
3. chunker 产生了空片段。

严格失败可以让问题尽早暴露。

### `RagIndex`

`RagIndex` 是构建完成后的在线检索入口对象：

```text
vector_collection
document_collection
chunk_collection
embedding_client
manifest
```

后续子模块 5 的 `VectorRetriever` 会使用：

```text
embedding_client + vector_collection + chunk_collection
```

## `app/factory/` 代码讲解

factory 软件包新增了几组工厂方法：

```python
ConfigFactory.build_embedding_config
ConfigFactory.build_vector_repository_config
ConfigFactory.build_index_builder_config
IndexingFactory.build_embedding_client
IndexingFactory.build_embedding_cache
IndexingFactory.build_vector_collection
IndexingFactory.build_vector_repository
IndexingFactory.build_document_collection
IndexingFactory.build_chunk_collection
IndexingFactory.build_document_repository
IndexingFactory.build_chunk_repository
```

这使配置流向非常清晰：

```text
settings.toml
  -> ProjectSettings
  -> factory
  -> Config
  -> 功能类
```

例如：

```text
[indexing.vector_repository].type = "local_json"
  -> VectorRepositorySettings
  -> VectorRepositoryConfig
  -> InMemoryVectorCollection
  -> LocalJsonVectorRepository
```

如果以后换 Chroma，可以增加：

```python
if config.repository_type == "qdrant":
    return QdrantVectorRepository(...)
```

而 `IndexBuilder` 和 `VectorRetriever` 不需要知道 Chroma 的存在。

## 如何运行

不需要直接安装新依赖即可运行 mock + local_json baseline。

在项目根目录执行：

```bash
python -m app.main index --source data/raw/papers
```

如果使用当前 `settings.toml`，会生成：

```text
data/indexes/papers_baseline/
  vector_collection.json
  chunk_collection.json
  document_collection.json
  embedding_cache.json
  manifest.json
  index_build_report.json
```

你也可以继续运行：

```bash
python -m app.main ask "RAG 为什么需要引用？" --source data/raw/papers
```

注意：当前 `ask` 命令仍然会先构建索引再问答。后续可以在练习中把“构建索引”和“加载已有索引”拆开，这会更接近真实服务。

## 如何运行测试

```bash
python -m unittest discover -s tests
```

本次新增和更新的重点测试：

```bash
python -m unittest tests.test_embedding_clients
python -m unittest tests.test_embedding_cache
python -m unittest tests.test_vector_collection
python -m unittest tests.test_index_manifest
python -m unittest tests.test_index_builder_embedding_cache
```

## 本次实现的工程取舍

### 为什么没有直接使用 FAISS 或 Chroma

你之前明确要求不要直接改变环境，不要替你安装依赖。

所以本次实现采用：

```text
InMemoryVectorCollection + LocalJsonVectorRepository + exact cosine search
```

它不是最终生产方案，但它比单纯 in-memory demo 更进一步：

1. 可以持久化。
2. 可以重新加载。
3. 可以把向量、chunk、document 分别持久化。
4. 可以跑通真实论文目录。
5. 可以被统一接口替换。

后续接 FAISS、Chroma、Qdrant 或 pgvector 时，重点不是重写 `IndexBuilder`，而是新增符合当前分层的 collection/repository adapter。

### 为什么要把向量和 chunk 拆开

概念文档中提到，真实系统通常会拆成：

```text
VectorCollection
  vector + chunk_id + filter metadata

ChunkCollection / DocumentCollection
  full text + citation metadata
```

当前代码已经完成这个拆分：

1. `VectorCollection` 只保存 `chunk_id`、vector 和轻量 metadata。
2. `ChunkCollection` 保存完整 `DocumentChunk`。
3. `DocumentCollection` 保存 raw document 和 parsed document。
4. `LocalJsonVectorRepository`、`LocalJsonChunkRepository`、`LocalJsonDocumentRepository` 分别负责持久化。
5. 检索时先查 vector collection，再根据 `chunk_id` 从 chunk collection 补全内容。

### 为什么 manifest 和 build report 始终写入索引目录

`VectorCollection` 只负责进程内向量管理，`VectorRepository` 则是持久化边界。一次成功的索引构建必须同时写入向量、文档、Chunk、Manifest 与构建报告，才能成为可恢复的索引版本。

测试应显式提供临时 `index_dir`，而不是依赖不存在的 memory Repository 或关闭持久化来规避文件输出。

## 练习 1：实现索引加载入口

当前 `app.main ask` 仍然会先构建索引，再执行问答。

这对学习流程足够，但真实系统不应该每次请求都重新构建索引。

请你设计并实现一个“加载已有索引”的入口，例如：

```python
ApplicationFactory(project_settings=project_settings).build_rag_index_from_storage() -> RagIndex
```

建议实现位置：

```text
app/factory/indexing.py
app/indexing/index_loader.py
```

目标：

1. 从 `settings.toml` 得到 `VectorRepositoryConfig`。
2. 加载 `manifest.json`。
3. 校验 manifest 与当前 embedding/vector repository 配置兼容。
4. 通过 `LocalJsonVectorRepository` 加载 `VectorCollection`。
5. 通过 `LocalJsonChunkRepository` 加载 `ChunkCollection`。
6. 通过 `LocalJsonDocumentRepository` 加载 `DocumentCollection`。
7. 构造与索引一致的 `EmbeddingClient`。
8. 返回 `RagIndex`。

完成后，可以考虑新增 CLI：

```bash
python -m app.main ask --use-existing-index "RAG 为什么需要引用？"
```

这个练习的重点是理解：

```text
离线 build
在线 load
```

二者应该分离。

## 练习 2：拆分 VectorCollection 与 Repository

当前练习已经完成：旧的“向量存储直接保存完整 chunk”的做法已经拆成运行时集合和持久化 Repository。

当前结构是：

```text
VectorCollection
  chunk_id
  vector
  metadata for filter

ChunkCollection
  full DocumentChunk

DocumentCollection
  RawDocument
  ParsedDocument

VectorRepository
  load/save VectorCollection

ChunkRepository
  load/save ChunkCollection

DocumentRepository
  load/save DocumentCollection
```

对应代码位置：

```text
app/indexing/vector_collection.py
app/ingest/chunking/collection.py
app/ingest/document_collection.py
app/repositories/vector.py
app/repositories/chunk.py
app/repositories/document.py
```

构建阶段会生成：

```text
vector_collection.json
chunk_collection.json
document_collection.json
```

在线检索阶段：

1. `VectorRetriever` 调用 `VectorCollection.search()` 得到 `VectorSearchResult`。
2. `VectorRetriever` 根据 `chunk_id` 从 `ChunkCollection` 补全 `DocumentChunk`。
3. 最终仍然返回统一的 `RetrievedChunk`。

这个练习的重点是学习真实 RAG 系统的数据分层。

## 练习 3：新增一个专业向量库 adapter

在不改变 `IndexBuilder` 主流程的前提下，新增一个真实向量库实现。

可以选择：

```text
FAISS
Chroma
Qdrant
pgvector
```

建议优先尝试 Chroma 或 FAISS。

要求：

1. 新实现必须满足当前 collection/repository 分层。
2. factory 根据 `settings.toml` 的 `[indexing.vector_repository].type` 选择实现。
3. 不要让业务代码直接依赖具体 SDK。
4. manifest 中记录新的 vector repository 类型。

你需要自行添加依赖，不要把依赖安装写进代码。

这个练习的重点不是某个 SDK 的 API，而是 adapter 设计。

## 练习 4：设计更完整的索引版本策略

当前代码已经把 index version 落到了 `IndexManifest` 中。

这次实现的核心思想是：

```text
index version = schema_version + config_hash + document_set_hash + status
```

其中：

```text
index_id
schema_version
status
parent_index_id
config_hash
document_set_hash
document_versions
```

### 字段含义

1. `schema_version`
   - 表示 manifest 和索引产物的内部结构版本。
   - 当前值为 `3`。
   - 如果后续 JSON 结构、字段含义或持久化布局发生破坏性变化，就应该提升它。
   - 加载已有索引时，如果 schema version 不匹配，系统会拒绝加载。

2. `status`
   - 表示当前索引版本状态。
   - 当前支持：`building`、`ready`、`failed`、`deprecated`。
   - 在线加载只接受 `ready` 状态。
   - 这为后续“构建中索引不对外服务”“失败索引保留诊断信息”“旧索引下线”预留了工程入口。
   - 当前构建流程会在 ingestion 成功、已经拿到 `document_versions` 后先写入 `building`。
   - 如果后续 chunking、embedding 或持久化全部成功，会覆盖为 `ready`。
   - 如果后续阶段失败，会尽量覆盖为 `failed`，并继续抛出原始异常。

3. `parent_index_id`
   - 表示当前索引版本来源于哪个旧索引版本。
   - 当前构建流程默认是 `None`。
   - 后续做增量索引、实验分支、回滚链路时，可以用它建立版本关系。

4. `config_hash`
   - 只由构建配置生成。
   - 包括 source directory、chunker、chunk size、chunk overlap、embedding provider、embedding model、embedding dimension、embedding batch size、vector repository type、collection name、distance metric。
   - source directory 会先规范化为绝对 POSIX 路径，再参与 hash。
   - 它不包含文档版本。
   - 所以当文档内容变化但构建配置不变时，`config_hash` 不变。

5. `document_set_hash`
   - 只由 `document_versions` 生成。
   - 它表示输入文档集合的版本指纹。
   - 所以当文档内容、路径或解析出的版本 ID 变化时，`document_set_hash` 会变化。

6. `index_id`
   - 是最终索引版本 ID。
   - 由 `schema_version`、`config_hash`、`document_set_hash` 一起生成。
   - 只要索引结构、构建配置或输入文档集合任意一项发生变化，就会得到新的 `index_id`。

### 为什么要拆成两个 hash

旧设计中 `config_hash` 同时包含构建配置和 `document_versions`。这能判断“索引是否变化”，但无法解释“为什么变化”。

现在拆成：

```text
config_hash
document_set_hash
```

好处是：

1. 同一批文档、同一配置会生成同一个稳定 `index_id`。
2. 文档内容变化后，只有 `document_set_hash` 和 `index_id` 变化。
3. chunker 或 embedding 配置变化后，只有 `config_hash` 和 `index_id` 变化。
4. 评测结果可以引用具体 `index_id`，同时解释实验差异来自配置还是语料。
5. 后续做增量索引时，可以先比较 `config_hash`，再比较 `document_set_hash`，判断是全量重建还是只处理文档变化。

### 当前代码位置

相关实现位于：

```text
app/indexing/manifest.py
tests/test_index_manifest.py
tests/test_index_builder_embedding_cache.py
```

构建流程会在 `manifest_building` 和 `manifest_ready` 阶段记录：

```text
index_id
schema_version
status
config_hash
document_set_hash
```

加载已有索引时，`validate_manifest_compatible` 会拒绝：

1. schema version 不匹配的索引。
2. status 不是 `ready` 的索引。
3. embedding provider/model/dimension/batch size 不匹配的索引。
4. vector repository type、collection name、distance metric 不匹配的索引。

这个练习的重点是实验可追溯性和生产环境的索引安全加载。

## 练习 5：把 OpenAI embedding 接入一次真实索引

这个练习需要你自行准备环境和 API key。

步骤建议：

1. 自行添加 `openai` 依赖。
2. 设置环境变量：

```bash
OPENAI_API_KEY=你的真实 key
```

3. 修改 `settings.toml`：

```toml
[indexing.embedding]
provider = "openai"
model = "text-embedding-3-small"
dimension = 1536
```

4. 执行：

```bash
python -m app.main index --source data/raw/papers
```

5. 检查：

```text
manifest.json
embedding_cache.json
vector_collection.json
chunk_collection.json
document_collection.json
index_build_report.json
```

观察：

1. 首次构建 cache miss 数量。
2. 第二次构建是否跳过已有 chunk。
3. manifest 中 embedding model 和 dimension 是否正确。
4. mock embedding 和真实 embedding 的检索结果有什么差异。

这个练习的重点是从工程流程正确走向真实检索质量。

## 子模块 4 验收标准

完成本子模块后，你应该能做到：

1. 用 mock embedding 跑通索引构建，不依赖外部服务。
2. 通过配置切换 embedding provider。
3. 通过配置和 Registry 创建可替换的向量 Repository；当前内置 `local_json` 实现。
4. 索引构建能复用 embedding cache。
5. 重复构建不会无意义重复写入同一批 chunk。
6. manifest 能说明索引由哪些配置生成。
7. build report 能说明一次构建过程发生了什么。
8. 向量维度不一致时系统会清晰失败。
9. 后续可以用 adapter 方式替换真实向量库。
10. 能解释为什么离线索引和在线检索应该分离。
