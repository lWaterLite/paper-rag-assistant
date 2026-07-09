# 子模块 4：Embedding、向量索引与持久化存储概念教学

对应学习路线：模块 2《RAG 知识库与检索增强生成》子模块 4  
核心项目：`paper-rag-assistant`  
学习定位：把已经解析、清洗、切分好的 chunks 转换成可检索、可复现、可持久化的向量索引

---

## 1. 子模块 4 要解决什么问题

前三个子模块已经完成了 RAG 离线流程的前半段：

```text
原始文档
  -> 文档加载
  -> 文档解析
  -> 文档清洗
  -> chunking
  -> DocumentChunk
```

到这里为止，系统已经拿到了很多结构化文本片段，例如：

```text
chunk_id: paper_xxx_chunk_0001
doc_id: paper_xxx
text: "Retrieval-Augmented Generation combines parametric memory..."
section: "Introduction"
page_start: 1
page_end: 2
metadata: {...}
```

但是这些 chunk 还不能高效支持语义检索。

如果用户问：

```text
RAG 为什么能减少幻觉？
```

而论文 chunk 里写的是：

```text
Retrieval-Augmented Generation grounds the model output in external non-parametric knowledge.
```

这两个句子没有完全相同的关键词，但语义高度相关。传统字符串匹配或关键词检索可能很难稳定命中。Embedding 和向量检索要解决的就是这个问题：

> 把文本转换成向量，让语义相近的文本在向量空间里距离更近，从而支持“按语义找内容”。

子模块 4 的核心任务不是简单调用一个 embedding API，而是建立一套真实工程需要的索引系统：

- 如何把 chunk 文本批量转换成 embedding 向量。
- 如何校验 embedding 维度、模型名称、输入长度和返回结果。
- 如何把向量写入本地或远程向量库。
- 如何保存 chunk 的 metadata，支持引用、过滤、调试和评测。
- 如何缓存 embedding，避免重复花钱、重复耗时。
- 如何记录 index manifest，保证索引可以复现。
- 如何设计索引版本，让不同配置的实验结果不会混在一起。
- 如何让 mock embedding 和真实 embedding 可以通过同一套接口替换。

这一层完成后，后续子模块 5 才能实现真正的 `VectorRetriever`：

```text
用户问题
  -> query embedding
  -> vector search
  -> top-k chunks
```

---

## 2. 从文本检索到向量检索

### 2.1 关键词检索的局限

关键词检索通常根据词项是否出现、出现频率、词项稀有程度来判断相关性。后续会学习的 BM25 就是一种经典关键词检索算法。

例如用户问：

```text
什么是检索增强生成？
```

如果 chunk 中包含：

```text
检索增强生成是一种结合外部知识库和语言模型的技术。
```

关键词检索很容易命中，因为“检索”“增强”“生成”都出现了。

但如果 chunk 是英文论文：

```text
Retrieval-Augmented Generation combines a retriever with a generator.
```

中文 query 和英文 chunk 没有共享词项。关键词检索就可能失败。

再比如：

```text
用户问题：RAG 如何减少模型胡编？
论文原文：RAG improves factuality by grounding generation in retrieved evidence.
```

“胡编”和 “hallucination / factuality / grounding” 之间是语义关系，不是简单词面匹配。

这就是向量检索的价值。

### 2.2 向量检索的基本思想

Embedding model 会把文本映射成一个固定长度的数字列表：

```text
"RAG 如何减少幻觉？"
  -> [0.013, -0.27, 0.81, ..., 0.04]
```

这个数字列表就是 embedding，也可以叫向量表示。

一个 embedding 向量通常有几百到几千维。例如：

```text
384 维
768 维
1024 维
1536 维
3072 维
```

不同模型的维度不同。维度不是随便改的，它由模型决定。

向量检索的流程可以理解成：

```text
离线索引阶段：
  chunk text -> embedding vector -> 写入 vector store

在线检索阶段：
  query text -> query vector -> 与索引中的 chunk vector 比较相似度 -> 返回 top-k
```

这里的关键是：

> query 和 chunk 都必须使用同一个 embedding model 转成向量，否则它们不在同一个语义空间里，比较结果没有意义。

---

## 3. 什么是 Embedding

### 3.1 Embedding 的定义

Embedding 是一种把离散对象转换成连续向量的表示方式。

在 RAG 中，最常见的是文本 embedding：

```text
文本 -> 数字向量
```

例如：

```python
text = "RAG combines retrieval and generation."
vector = [0.12, -0.03, 0.78, ...]
```

这个向量不是人工设计出来的，而是 embedding model 根据大量训练数据学习到的表示。

它的目标不是让人直接读懂每一维数字的含义，而是让机器可以通过数学距离比较文本的语义相似度。

### 3.2 Embedding 的作用

Embedding 在 RAG 中主要有五个作用。

第一，支持语义检索。

用户问题和文档片段即使没有完全相同的关键词，只要语义接近，也可能被召回。

第二，支持跨语言或近似跨语言检索。

一些 embedding model 能把中文问题和英文论文映射到相近空间，让中文 query 检索英文 chunk 成为可能。

第三，支持聚类和去重。

相似 chunk 的向量距离通常较近，可以用来发现重复段落、相似主题或重复论文版本。

第四，支持推荐和关联。

如果某个 chunk 讲 “context packing”，系统可以找出语义接近的其他 chunk，例如 “prompt context construction”。

第五，支持后续 hybrid retrieval。

真实 RAG 系统通常不会只靠向量检索，而是结合 BM25、rerank、过滤条件和业务规则。Embedding 是其中的 dense retrieval 基础。

### 3.3 Embedding 不是什么

Embedding 不是摘要。

一个 embedding 向量不能直接还原成原文，也不能替代 chunk text。向量只是为了比较相似度，最终回答仍然需要原文 chunk。

Embedding 不是知识库本身。

把文本 embedding 后写入向量库，不代表系统已经理解了文档。它只是建立了一个可检索索引。

Embedding 不是永远正确的语义判断。

Embedding 可能召回语义相似但事实不相关的 chunk，也可能漏掉包含精确术语但语义表达不同的 chunk。这就是后续需要 BM25、hybrid retrieval 和 rerank 的原因。

Embedding 不是跨模型通用的。

不同 embedding model 生成的向量不能混在同一个索引里直接比较。

---

## 4. Embedding Model 的关键概念

### 4.1 输入文本限制

Embedding model 通常有最大输入长度限制。

例如某个模型可能最多接受：

```text
8192 tokens
```

如果 chunk 太长，可能出现三种情况：

- API 直接报错。
- SDK 自动截断，导致信息丢失但你不一定知道。
- 成本和延迟显著上升。

所以 chunking 和 embedding 是强相关的：

```text
chunk size 过大
  -> embedding 输入可能超限
  -> 向量表达变粗
  -> 检索不够精确

chunk size 过小
  -> embedding 成本增加
  -> 上下文不足
  -> 召回结果碎片化
```

### 4.2 向量维度

向量维度是 embedding model 输出向量的长度。

例如：

```text
model_a -> 768 维
model_b -> 1536 维
model_c -> 3072 维
```

向量库通常要求同一个 collection 或 index 里的所有向量维度一致。

如果索引已经用 1536 维模型构建，再写入 768 维向量，就应该立即失败，而不是默默写入。

因此工程上必须记录：

```text
embedding_model
embedding_dimension
```

并在写入时校验。

### 4.3 成本

真实 embedding 服务通常按 token 或请求量计费。

如果有：

```text
10 篇论文
850 个 chunks
每个 chunk 平均 600 tokens
```

那么一次完整索引可能需要处理几十万 tokens。

随着论文数量增长，embedding 成本会成为实际问题。

所以真实项目中必须考虑：

- batch embedding
- embedding cache
- 增量索引
- 重复 chunk 跳过
- 失败重试
- 速率限制
- 成本统计

### 4.4 批处理

不要对每个 chunk 单独调用一次 embedding API。

低效方式：

```text
for chunk in chunks:
    embed_text(chunk.text)
```

更合理方式：

```text
把 chunks 分批
  -> embed_batch(texts)
  -> 批量写入 vector store
```

batch 的好处：

- 减少网络请求次数。
- 提升吞吐。
- 更容易统一处理速率限制。
- 更容易记录批次级别日志。

但 batch 也不能无限大：

- API 可能限制单批文本数量。
- API 可能限制单批 token 总量。
- 单批失败时重试成本更高。
- 内存占用更高。

所以需要配置：

```text
embedding_batch_size
max_batch_tokens
request_timeout_seconds
max_retries
```

---

## 5. EmbeddingClient 抽象接口

### 5.1 为什么需要 EmbeddingClient

在真实工程中，不应该在索引构建逻辑里直接调用某个具体厂商的 SDK。

不推荐：

```python
client = OpenAI(...)
response = client.embeddings.create(...)
```

如果业务代码直接依赖具体 SDK，会带来几个问题：

- 测试时必须访问真实外部服务。
- API key 容易散落在业务逻辑里。
- 将来换模型或换供应商需要大范围改代码。
- mock embedding 和真实 embedding 难以切换。
- 失败重试、日志、限流逻辑容易重复。

更合理的方式是定义抽象接口：

```python
class EmbeddingClient(Protocol):
    def embed_text(self, text: str) -> list[float]:
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        ...
```

业务层只依赖这个接口：

```text
IndexBuilder -> EmbeddingClient
```

而不是依赖具体实现：

```text
IndexBuilder -> OpenAI SDK
IndexBuilder -> SentenceTransformers
IndexBuilder -> MockEmbedding
```

### 5.2 EmbeddingClient 的职责

EmbeddingClient 应该负责：

- 接收文本输入。
- 调用具体 embedding model。
- 返回向量。
- 确保返回数量和输入数量一致。
- 处理模型维度信息。
- 封装 provider 的错误。
- 记录必要日志。
- 支持 mock 和真实实现。

EmbeddingClient 不应该负责：

- 决定 chunk 如何切分。
- 决定索引版本号。
- 决定向量写入哪里。
- 决定检索 top-k。
- 保存业务 metadata。

这能保持职责清晰。

### 5.3 MockEmbeddingClient

Mock embedding 是学习和测试阶段非常重要的工具。

它的目标不是模拟真实语义质量，而是让系统可以在不依赖外部 API 的情况下跑通：

```text
chunks -> embedding -> vector store -> search
```

Mock embedding 可以做到：

- 输出固定维度向量。
- 同样输入稳定返回同样向量。
- 不需要 API key。
- 运行速度快。
- 测试结果可复现。

但是要注意：

> mock embedding 只能证明工程流程正确，不能证明检索质量真实有效。

真实检索效果必须用真实 embedding model 和真实文档评测。

### 5.4 真实 EmbeddingClient

真实实现可能来自：

- OpenAI embeddings
- Azure OpenAI embeddings
- Hugging Face sentence-transformers
- BGE / E5 / GTE 等本地模型
- 企业内部 embedding 服务

真实实现需要考虑：

- API key 从环境变量或 `.env` 读取。
- 不在日志里打印 API key。
- 超时和重试。
- rate limit。
- batch size。
- 输入 token 限制。
- 返回维度校验。
- provider 错误转换成项目内部错误。
- 模型名称和维度写入 manifest。
- 成本统计。

---

## 6. 相似度搜索的数学直觉

向量检索的本质是：

> 在很多 chunk 向量中，找到和 query 向量最相似的 top-k 个。

常见相似度或距离指标有三种。

### 6.1 Cosine Similarity

Cosine similarity 比较两个向量方向是否接近。

直觉上：

```text
方向越相似，语义越接近。
```

取值通常在：

```text
-1 到 1
```

在很多 embedding 检索中，cosine similarity 是非常常见的选择。

特点：

- 关注方向，不太关注向量长度。
- 很适合语义相似度比较。
- 向量归一化后，cosine 和 dot product 有密切关系。

### 6.2 Dot Product

Dot product 是两个向量对应维度相乘再求和。

如果向量已经归一化，dot product 排序结果通常和 cosine similarity 等价。

特点：

- 计算快。
- 许多向量库和模型默认使用。
- 是否需要归一化取决于模型和向量库配置。

### 6.3 L2 Distance

L2 distance 是欧式距离。

直觉上：

```text
两个点在空间中越近，距离越小。
```

特点：

- 返回的是距离，不是相似度。
- 分数方向和 cosine 不同：距离越小越相似。
- 在某些向量库里默认使用。

### 6.4 分数不能乱比

一个非常重要的工程原则：

> 不同检索器、不同模型、不同距离算法的分数不能直接比较。

例如：

```text
FAISS cosine score: 0.82
BM25 score: 12.4
reranker score: 0.63
```

这些分数没有共同尺度。后续做 hybrid retrieval 时需要 score normalization 或 rank fusion。

---

## 7. 什么是向量索引

### 7.1 向量库不是简单 list

如果只有 10 个 chunk，可以把所有向量放在列表里，逐个计算相似度。

但真实系统可能有：

```text
10 万 chunks
100 万 chunks
甚至更多
```

逐个比较会非常慢。

向量索引的目标是：

> 用专门的数据结构加速相似向量搜索。

这个过程也叫 ANN search：

```text
Approximate Nearest Neighbor Search
近似最近邻搜索
```

为什么是近似？

因为在大规模向量中，完全精确搜索成本很高。很多向量库会用近似算法换取速度。

### 7.2 Index 与 Collection

不同向量库术语不同。

常见概念：

```text
index
collection
namespace
table
partition
```

在本项目里可以统一理解为：

> 一个 index 是一组使用同一套配置构建出来的向量集合。

同一个 index 应该至少共享：

- embedding model
- embedding dimension
- distance metric
- chunking strategy
- chunk size
- chunk overlap
- 文档集合范围
- metadata schema

如果这些配置变了，就应该生成新的 index version。

### 7.3 写入向量库的内容

写入向量库的不只是向量。

通常至少需要：

```text
id: chunk_id
vector: embedding
metadata: 可过滤、可回溯的字段
document: 可选，chunk 原文
```

例如：

```json
{
  "id": "paper_001_chunk_0032",
  "vector": [0.12, -0.03, 0.78],
  "metadata": {
    "doc_id": "paper_001",
    "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
    "section": "Introduction",
    "page_start": 1,
    "page_end": 2,
    "content_hash": "abc123"
  }
}
```

是否把完整 chunk text 存在向量库中，要看向量库能力和项目设计。

很多真实系统会拆成：

```text
Vector Store
  保存向量、chunk_id、少量过滤 metadata

Metadata Store / Document Store
  保存完整 chunk text、引用信息、解析来源、调试信息
```

这种拆分更灵活。

---

## 8. 向量库的选择

子模块 4 要求了解 FAISS、Chroma、Qdrant、Milvus、pgvector 的差异。它们都能做向量检索，但定位不同。

### 8.1 FAISS

FAISS 是 Meta 开源的高性能向量检索库。

适合：

- 本地实验。
- 高性能向量搜索。
- 自己管理 metadata。
- 对工程控制力要求高的场景。

特点：

- 很强的向量搜索能力。
- 本身更偏底层库，不是完整数据库。
- metadata 管理需要自己补。
- 持久化、并发、权限、多租户等能力需要自己设计。

在学习项目中，FAISS 适合作为 baseline，因为它能让你清楚理解向量索引本身。

### 8.2 Chroma

Chroma 是面向 AI 应用的本地/轻量向量数据库。

适合：

- 快速构建 RAG 原型。
- 本地持久化。
- 同时保存 document、embedding、metadata。
- 学习阶段降低工程复杂度。

特点：

- 上手简单。
- API 更接近 RAG 应用。
- 适合中小规模项目。
- 生产级能力取决于部署和规模要求。

如果你的目标是尽快跑通完整 RAG，Chroma 是友好的选择。

### 8.3 Qdrant

Qdrant 是一个向量数据库服务，常用于生产环境。

适合：

- 服务化部署。
- payload metadata 过滤。
- 中大型项目。
- 需要 REST/gRPC API 的场景。

特点：

- metadata filter 能力较强。
- 工程化程度高。
- 适合独立部署为检索服务。
- 比 FAISS 更像完整数据库。

### 8.4 Milvus

Milvus 是面向大规模向量检索的向量数据库。

适合：

- 大规模数据。
- 分布式部署。
- 企业级向量检索。
- 高吞吐场景。

特点：

- 能力强，但运维复杂度更高。
- 学习成本比 Chroma / FAISS 高。
- 对小型学习项目可能偏重。

### 8.5 pgvector

pgvector 是 PostgreSQL 的向量扩展。

适合：

- 已经使用 PostgreSQL 的项目。
- 希望关系数据和向量数据放在同一个数据库。
- 需要 SQL、事务、权限、业务表关联的场景。

特点：

- 与传统业务数据库融合好。
- metadata 和业务字段管理方便。
- 检索性能和扩展性取决于数据规模、索引类型和数据库配置。

### 8.6 学习阶段怎么选

建议路线：

```text
学习和本地 baseline：
  FAISS 或 Chroma

如果希望更像真实服务：
  Qdrant

如果已有 PostgreSQL 背景：
  pgvector

如果面向大规模生产：
  Milvus
```

本项目可以先选择一个本地向量库作为 baseline，但接口层必须保持可替换：

```text
IndexBuilder -> VectorStore interface -> FAISSVectorStore / ChromaVectorStore / ...
```

---

## 9. VectorStore 抽象接口

### 9.1 为什么需要 VectorStore

和 EmbeddingClient 一样，索引构建流程不应该直接依赖某个具体向量库 SDK。

更合理的结构：

```text
IndexBuilder
  -> EmbeddingClient
  -> VectorStore
  -> MetadataStore
  -> IndexManifestWriter
```

VectorStore 负责保存和查询向量。

常见接口包括：

```python
class VectorStore(Protocol):
    def add(
        self,
        items: list[VectorItem],
    ) -> None:
        ...

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        filters: dict[str, object] | None = None,
    ) -> list[VectorSearchResult]:
        ...

    def persist(self) -> None:
        ...

    def load(self) -> None:
        ...
```

### 9.2 VectorItem

写入向量库时，最好不要到处传散乱参数，而是定义结构化对象：

```text
VectorItem
  id
  vector
  metadata
```

其中：

- `id` 通常使用 `chunk_id`。
- `vector` 是 embedding。
- `metadata` 是可过滤、可调试的轻量字段。

完整 chunk text 是否写入 `VectorItem`，取决于具体 vector store 方案。

### 9.3 VectorSearchResult

搜索结果也应该结构化：

```text
VectorSearchResult
  id
  score
  metadata
```

注意它通常还不是最终的 `RetrievedChunk`。

因为向量库返回的可能只有 chunk id 和 metadata。真正的 chunk text、source_path、page 信息，可能需要从 DocumentStore 或 ChunkRepository 中补齐。

后续检索层可能是：

```text
VectorStore.search
  -> VectorSearchResult
  -> ChunkRepository.get_by_ids
  -> RetrievedChunk
```

这样可以避免把所有业务数据都塞进向量库。

---

## 10. Metadata Store 与 Vector Store 的关系

### 10.1 为什么不能只保存向量

如果只保存向量和 chunk_id，系统只能知道：

```text
这个 query 命中了 chunk_0032
```

但不知道：

- chunk 原文是什么。
- 来自哪篇论文。
- 第几页。
- 哪一节。
- 当时使用哪个 parser 解析。
- 当时使用哪个 chunker 切分。
- 是否属于某个标签。
- 是否允许当前用户访问。

这些信息对 RAG 至关重要。

所以 metadata 不是附属品，而是 RAG 工程可用性的基础。

### 10.2 metadata 的用途

metadata 至少支持五类能力。

第一，引用。

回答中需要给出：

```text
[C1] Paper Title, page 3, section "Method"
```

这依赖 chunk metadata。

第二，过滤。

例如只检索：

```text
年份 >= 2023
主题 = RAG
文档类型 = paper
可见性 = public
```

第三，调试。

当检索失败时，需要查看：

```text
召回的是哪篇文档？
哪一节？
chunk 是否太短？
解析是否乱码？
```

第四，评测。

评测指标可能需要判断：

```text
expected_doc_ids 是否出现在 top-k 中
expected_section 是否被召回
```

第五，增量构建。

通过 `content_hash` 可以判断 chunk 内容是否变化，避免重复 embedding。

### 10.3 metadata 放在哪里

有三种常见方式。

第一，全部放向量库。

优点：

- 实现简单。
- 查询时一次返回。

缺点：

- 向量库 metadata 能力有限时会受限制。
- 大字段会让索引臃肿。
- 修改 metadata 可能不方便。

第二，向量库只放轻量 metadata，完整数据放独立 store。

优点：

- 职责清晰。
- 可扩展性好。
- 便于后续接数据库。

缺点：

- 查询流程多一步。
- 需要维护 id 一致性。

第三，全部放关系数据库，向量也存在数据库中，例如 pgvector。

优点：

- 数据一致性强。
- SQL 查询方便。
- 业务系统集成好。

缺点：

- 需要数据库配置。
- 向量检索性能受具体实现影响。

学习项目建议采用第二种思路：

```text
VectorStore:
  chunk_id
  vector
  filter metadata

ChunkRepository / MetadataStore:
  chunk_id
  full text
  citation metadata
  parsing/chunking metadata
```

---

## 11. Embedding Cache

### 11.1 为什么需要 cache

Embedding 是高成本环节。

如果每次运行索引构建都重新 embedding 所有 chunk，会浪费：

- API 费用。
- 时间。
- provider 额度。
- 本地计算资源。

Embedding cache 的目标是：

> 对同一个模型、同一段文本、同一套关键配置，复用已经生成过的向量。

### 11.2 cache key 怎么设计

cache key 不能只用文本。

因为同一段文本使用不同模型会得到不同向量。

更合理的 cache key 应该包含：

```text
embedding_model
embedding_dimension
normalized_text_hash 或 content_hash
embedding_options_hash
```

例如：

```text
text-embedding-xxx:1536:chunk_content_hash
```

如果后续模型变了，cache 必须失效。

### 11.3 cache value 保存什么

cache value 至少保存：

```text
vector
model
dimension
created_at
text_hash
```

也可以保存：

```text
provider
token_count
request_id
```

### 11.4 cache 的位置

常见选择：

- 本地 JSONL。
- SQLite。
- DuckDB。
- Redis。
- 专门的 embedding cache 表。

学习阶段可以先用文件或 SQLite。真实项目更推荐 SQLite 或数据库，因为：

- 查询快。
- 不容易因为文件过大变慢。
- 更适合增量更新。
- 更容易保证结构。

### 11.5 cache 的风险

cache 虽然有用，但也会引入风险：

- cache key 设计不完整，导致误用旧向量。
- 文本清洗逻辑变了，但 content_hash 没变。
- embedding model 版本变了，但模型名看起来没变。
- cache 文件损坏。
- 并发写入导致数据不一致。

所以 cache 不是简单“有就用”，必须配合 manifest 和版本管理。

---

## 12. Index Version

### 12.1 为什么需要 index version

RAG 实验经常会调整配置：

```text
chunk_size: 300 -> 600
chunk_overlap: 50 -> 100
embedding_model: model_a -> model_b
distance_metric: cosine -> dot
document_set: 5 篇 -> 10 篇
```

这些变化都会影响检索结果。

如果所有结果都写到同一个 index 里，就会出现：

- 不知道当前索引用什么配置生成。
- 不知道评测结果对应哪个索引。
- 无法复现实验。
- 新旧 chunk 混在一起。
- debug 时无法定位问题。

所以每次关键配置变化，都应该形成新的 index version。

### 12.2 什么配置影响 index version

通常包括：

```text
embedding provider
embedding model
embedding dimension
distance metric
chunking strategy
chunk size
chunk overlap
chunk metadata schema version
document set hash
parser version
cleaner version
index builder version
```

不是所有配置都必须手工写入版本号，但 manifest 里必须记录。

### 12.3 index_id 如何设计

可以采用可读名称：

```text
papers_section_aware_600_100_bge_v1
```

也可以采用配置 hash：

```text
idx_20260701_abc123
```

更真实的方案通常是两者结合：

```text
index_id: papers_medium_v1
config_hash: sha256(...)
created_at: 2026-07-01T12:00:00Z
```

`index_id` 方便人读，`config_hash` 方便机器判断是否一致。

---

## 13. Index Manifest

### 13.1 Manifest 的定义

Index manifest 是索引的说明书。

它记录：

> 这个索引是谁、什么时候、用什么配置、基于哪些文档、生成了多少 chunk、写到了哪里。

示例：

```json
{
  "index_id": "papers_medium_v1",
  "created_at": "2026-07-01T12:00:00Z",
  "embedding_provider": "mock",
  "embedding_model": "mock-hash-embedding",
  "embedding_dimension": 128,
  "vector_store_type": "faiss",
  "distance_metric": "cosine",
  "chunking_strategy": "section_aware",
  "chunk_size": 600,
  "chunk_overlap": 100,
  "document_count": 10,
  "chunk_count": 850,
  "config_hash": "abc123",
  "source_manifest_hash": "def456"
}
```

### 13.2 Manifest 的作用

第一，复现实验。

看到评测结果时，可以知道它基于哪个索引。

第二，避免错误加载。

如果检索配置要求 `embedding_dimension=1536`，但 manifest 显示索引是 768 维，系统应该拒绝加载。

第三，支持增量构建。

manifest 可以记录已经索引过的文档和 chunk 数量。

第四，支持排查问题。

如果某个索引效果很差，可以回看：

```text
是否用了错误 chunk size？
是否用了 mock embedding？
是否文档数量太少？
是否 parser 版本变化？
```

第五，支持作品展示。

RAG 项目不是“看起来能回答”就够了。manifest 能证明系统有工程可追溯性。

### 13.3 Manifest 不应该包含什么

manifest 不应该包含：

- API key。
- 大量 chunk 原文。
- 用户隐私数据。
- 无法公开的内部路径。

可以记录相对路径、hash 和统计信息，但敏感信息要避免写入。

---

## 14. 可重复构建、可缓存、可恢复

子模块 4 的学习目标中特别强调：

```text
建立可重复构建、可缓存、可恢复的索引流程
```

这三个词很重要。

### 14.1 可重复构建

可重复构建是指：

> 同一批输入文档、同一套配置、同一套代码版本，应该生成可解释且一致的索引结果。

不一定要求向量值字节级完全一致，因为某些远程服务可能有版本变化。但系统至少要能记录配置，并尽量保证结果稳定。

可重复构建依赖：

- 稳定 doc_id。
- 稳定 chunk_id。
- 稳定 content_hash。
- 明确 chunker config。
- 明确 embedding model。
- 明确 index manifest。
- 确定性排序。

### 14.2 可缓存

可缓存是指：

> 已经 embedding 过的 chunk 不需要无意义重复计算。

它依赖：

- content_hash。
- embedding cache。
- 模型和维度校验。
- chunk 级别去重。

### 14.3 可恢复

可恢复是指：

> 索引构建过程中某一步失败后，系统能定位失败位置，并尽量从已有结果继续，而不是全部推倒重来。

真实索引构建可能因为以下原因失败：

- embedding API 超时。
- 网络中断。
- 某个 chunk 为空或过长。
- 向量库写入失败。
- metadata 缺字段。
- 程序中途停止。

可恢复设计通常包括：

- 批次级别日志。
- 已处理 chunk 记录。
- cache。
- 幂等写入。
- manifest 草稿或构建状态文件。
- 失败报告。

---

## 15. 什么是幂等索引构建

幂等是工程中非常重要的概念。

在索引构建里，幂等意味着：

> 同一个构建任务重复运行多次，不应该产生重复数据或不可预测结果。

例如：

```text
第一次运行：
  写入 850 个 chunk

第二次运行同样配置：
  应该识别这些 chunk 已存在
  不应该变成 1700 个 chunk
```

实现幂等的常用手段：

- 使用稳定 `chunk_id` 作为主键。
- 写入前检查是否存在。
- 使用 upsert 而不是 insert。
- 根据 `content_hash` 判断内容是否变化。
- manifest 记录构建完成状态。

向量库和 metadata store 都需要考虑幂等。

---

## 16. 索引构建流程

子模块 4 的核心工程流程可以设计成：

```text
读取 chunks
  -> 校验 chunk
  -> 过滤空 chunk
  -> 根据 content_hash 查询 embedding cache
  -> 对 cache miss 的文本批量 embedding
  -> 校验向量数量和维度
  -> 写入 embedding cache
  -> 组装 VectorItem
  -> 写入 vector store
  -> 写入 metadata store
  -> 保存 index manifest
  -> 输出 index build report
```

更详细一点：

```text
IndexBuilder
  1. 加载构建配置
  2. 加载待索引 chunks
  3. 运行 chunk 级校验
  4. 计算 index_id / config_hash
  5. 初始化 vector store
  6. 初始化 metadata store
  7. 初始化 embedding cache
  8. 按 batch 遍历 chunks
  9. 对 batch 做 cache lookup
  10. 对缺失项调用 EmbeddingClient.embed_batch
  11. 校验 embedding 结果
  12. 写入 cache
  13. 写入 vector store
  14. 写入 metadata store
  15. 持久化 vector index
  16. 写入 manifest
  17. 返回 build result
```

注意：

> manifest 应该在构建成功后标记为完成。如果中途失败，最好记录失败状态，而不是伪装成可用索引。

---

## 17. 关键数据结构建议

### 17.1 EmbeddingVector

可以用简单类型：

```python
list[float]
```

但真实项目里建议至少在边界处校验：

- 是否为空。
- 是否全是数字。
- 是否包含 NaN 或 Infinity。
- 维度是否等于配置维度。

### 17.2 EmbeddedChunk

表示 chunk 和 embedding 的绑定结果：

```text
EmbeddedChunk
  chunk_id
  doc_id
  vector
  embedding_model
  embedding_dimension
  content_hash
  metadata
```

它可以作为索引构建中间产物。

### 17.3 VectorItem

表示准备写入向量库的数据：

```text
VectorItem
  id
  vector
  metadata
```

### 17.4 IndexManifest

表示索引说明书：

```text
IndexManifest
  index_id
  created_at
  embedding_model
  embedding_dimension
  chunking_config
  vector_store_config
  document_count
  chunk_count
  config_hash
```

### 17.5 IndexBuildResult

表示一次构建结果：

```text
IndexBuildResult
  index_id
  status
  total_chunks
  indexed_chunks
  skipped_chunks
  failed_chunks
  cache_hits
  cache_misses
  manifest_path
  errors
```

这个结构对于 CLI、API、日志和测试都很有用。

---

## 18. 配置管理

子模块 4 会引入很多配置项。

建议分成几类。

### 18.1 EmbeddingSettings

来自外部配置文件或环境变量：

```text
provider
model
dimension
batch_size
timeout_seconds
max_retries
```

API key 不写入 TOML 或 `EmbeddingConfig`。它由 `EnvSettings` 从 `.env` 或系统
环境变量读取，再由 factory 显式注入真实 embedding client。

### 18.2 EmbeddingConfig

传给功能类使用的内部配置：

```text
provider
model
dimension
batch_size
timeout_seconds
max_retries
```

保持你前面已经建立的命名规范：

```text
Settings: 外部配置文件直接加载出来的配置
Config: 转换后传给功能类使用的配置
```

### 18.3 VectorStoreSettings

```text
type
index_dir
collection_name
distance_metric
persist
```

### 18.4 IndexSettings

```text
index_id
version
manifest_dir
rebuild
skip_existing
fail_on_error
```

### 18.5 为什么不要把配置散落在业务代码

如果在代码里到处写：

```python
dimension = 1536
batch_size = 32
metric = "cosine"
```

后续会很难回答：

- 当前索引到底用什么配置生成？
- 为什么本地和别人机器结果不同？
- 为什么测试和真实运行不一致？
- 为什么换模型后索引加载失败？

所以配置应该集中加载，在 factory 层组装对象，再注入到业务流程。

---

## 19. Factory 与依赖注入

你前面已经在 loader、chunker 等部分使用了 factory 管理对象构建。子模块 4 更应该延续这个范式。

推荐结构：

```text
factory
  -> build_embedding_client
  -> build_vector_store
  -> build_embedding_cache
  -> build_index_manifest_writer
  -> build_index_builder
```

业务流程不应该自己私自构造默认对象。

不推荐：

```python
class IndexBuilder:
    def __init__(self, embedding_client=None):
        self.embedding_client = embedding_client or MockEmbeddingClient()
```

推荐：

```python
class IndexBuilder:
    def __init__(
        self,
        embedding_client: EmbeddingClient,
        vector_store: VectorStore,
        metadata_store: ChunkRepository,
        cache: EmbeddingCache,
    ) -> None:
        ...
```

这样做的好处：

- 对象依赖清晰。
- 配置统一从 factory 进入。
- 不会出现一部分对象用显式配置，一部分对象用默认配置。
- 测试可以直接注入 mock。
- 将来替换 Chroma/Qdrant 不影响 IndexBuilder 主流程。

---

## 20. 维度校验

维度校验是子模块 4 最重要的健壮性检查之一。

应该在多个层次做：

### 20.1 EmbeddingClient 返回后校验

```text
输入 texts 数量 = N
返回 vectors 数量必须 = N
每个 vector 长度必须 = expected_dimension
```

否则应该抛出清晰错误。

### 20.2 写入 VectorStore 前校验

VectorStore 接收到向量时也可以校验维度，避免上游绕过检查。

### 20.3 加载已有索引时校验

如果 manifest 记录：

```text
embedding_dimension = 1536
```

而当前配置是：

```text
embedding_dimension = 768
```

系统应该拒绝加载该索引。

### 20.4 为什么不能自动兼容

不能把 768 维向量和 1536 维向量补零后混用。

因为它们来自不同语义空间，维度不只是长度问题，而是模型表示空间问题。

---

## 21. 重复 chunk 与增量索引

### 21.1 什么是重复 chunk

重复 chunk 可能来自：

- 同一文档重复导入。
- PDF 页眉页脚没有清理干净。
- 论文不同版本内容高度重复。
- chunk overlap 导致相邻 chunk 大量重复。

### 21.2 如何识别

常见方式：

```text
content_hash
doc_id + chunk_index
chunk_id
normalized_text_hash
```

其中：

- `chunk_id` 适合做主键。
- `content_hash` 适合判断内容是否变化。
- `normalized_text_hash` 适合跨 chunk 检测重复文本。

### 21.3 如何处理

索引构建时可以采用：

- 同一 index 内相同 `chunk_id` 执行 upsert。
- 相同 `content_hash` 可以复用 embedding。
- 相同文本但不同来源，可以复用 vector，但 metadata 仍然要保留不同引用来源。

注意最后一点：

> 内容重复不代表来源可以丢弃。

同一段话出现在两篇论文或两个版本里，引用信息仍然不同。

---

## 22. 错误处理

索引构建中常见错误包括：

```text
EmptyChunkError
EmbeddingInputTooLongError
EmbeddingProviderError
EmbeddingDimensionMismatchError
VectorStoreWriteError
MetadataMissingError
IndexManifestError
IndexLoadError
```

真实工程中需要区分：

- 单个 chunk 失败。
- 一个 batch 失败。
- 整个索引失败。

有些错误可以跳过并记录：

```text
空 chunk
metadata 不完整但不影响基础索引
某个文件解析质量较差
```

有些错误应该立刻停止：

```text
embedding 维度不一致
vector store 无法加载
manifest 与当前配置冲突
API key 缺失
```

这取决于配置：

```text
fail_on_error = true / false
```

学习阶段建议对关键一致性错误严格失败，这能帮助你更早发现问题。

---

## 23. 持久化存储

### 23.1 为什么要持久化

如果索引只存在内存中，那么每次启动都要重新：

```text
读取 chunks
embedding
构建向量索引
```

这在真实 RAG 中不可接受。

持久化的目标是：

> 构建一次，多次加载使用。

### 23.2 需要持久化哪些内容

至少包括：

```text
vector index
metadata store
embedding cache
index manifest
build report
```

推荐目录结构：

```text
data/
  indexes/
    papers_medium_v1/
      manifest.json
      vector.index
      metadata.jsonl
      embedding_cache.sqlite
      build_report.json
```

具体文件名取决于选用的向量库。

### 23.3 加载索引时要检查什么

加载已有索引时，不能只看文件存在。

应该检查：

- manifest 是否存在。
- vector index 是否存在。
- metadata store 是否存在。
- manifest 中的维度是否匹配当前配置。
- manifest 中的 vector store 类型是否匹配。
- manifest 中的 index_id 是否符合要求。
- 必要文件是否损坏或为空。

---

## 24. Mock 模式与真实模式

本项目应该同时支持：

```text
mock embedding
真实 embedding
```

### 24.1 Mock 模式用途

mock 模式用于：

- 本地开发。
- 自动化测试。
- 无 API key 环境。
- 验证索引构建流程。
- 验证异常处理。

### 24.2 真实模式用途

真实模式用于：

- 构建真实检索索引。
- 评测实际召回质量。
- 与 BM25 和 rerank 对比。
- 最终作品展示。

### 24.3 切换方式

应该通过配置切换：

```toml
[embedding]
provider = "mock"
model = "mock-hash-embedding"
dimension = 128
```

或：

```toml
[embedding]
provider = "openai"
model = "text-embedding-3-small"
dimension = 1536
```

factory 根据 provider 构建不同实现。

---

## 25. 与前面子模块的关系

子模块 4 不是独立存在的，它强依赖前面几个阶段的输出。

### 25.1 依赖子模块 1 的项目骨架

需要：

- 配置管理。
- factory。
- 核心数据模型。
- 错误类型。
- 日志结构。

### 25.2 依赖子模块 2 的解析质量

如果 PDF 解析出乱码，embedding 会把乱码也转成向量。

向量库不会知道文本质量差，它只负责保存向量。

所以解析质量问题会直接污染索引。

### 25.3 依赖子模块 3 的 chunking 质量

chunk 太大，embedding 粗糙。

chunk 太小，语义不完整。

metadata 缺失，后续无法引用。

所以子模块 4 的输入质量由子模块 3 决定。

### 25.4 支撑子模块 5 的检索

子模块 5 会用到：

```text
query embedding
vector store search
RetrievedChunk
top-k score
metadata
```

这些都需要子模块 4 提供。

---

## 26. 真实 RAG 工程中的常见坑

### 26.1 把不同 embedding model 的向量混进同一个索引

这是严重错误。

表现：

- 检索结果混乱。
- 分数不可解释。
- 评测结果不稳定。

解决：

- manifest 记录 model 和 dimension。
- 加载索引时校验。
- index version 随模型变化。

### 26.2 只保存向量，不保存 metadata

表现：

- 检索到了 chunk，但无法引用。
- 不知道来自哪篇论文。
- 无法做权限过滤。
- 无法评测命中率。

解决：

- 设计 ChunkRepository / MetadataStore。
- 保留 doc_id、chunk_id、page、section、source_path。

### 26.3 每次运行都重新 embedding

表现：

- 运行慢。
- 成本高。
- 开发体验差。

解决：

- embedding cache。
- content_hash。
- skip_existing。

### 26.4 index_id 随手命名

表现：

- 不知道哪个索引对应哪个实验。
- 评测结果不可复现。

解决：

- index manifest。
- config hash。
- 版本命名规范。

### 26.5 忽略 batch 失败

表现：

- 少量 chunk 没有进入索引，但系统没报告。
- 后续某些问题永远检索不到。

解决：

- build report。
- failed_chunks。
- fail_on_error 配置。

### 26.6 在 API 请求里实时构建索引

表现：

- 请求极慢。
- 多用户并发时系统不可用。
- 每次请求重复加载或构建资源。

解决：

- 离线构建索引。
- 应用启动时加载索引。
- 在线请求只做 query embedding 和 search。

---

## 27. 子模块 4 推荐工程结构

可以考虑增加如下目录：

```text
app/
  indexing/
    __init__.py
    embeddings.py
    embedding_cache.py
    vector_store.py
    manifest.py
    builder.py
    report.py
    errors.py
```

各文件职责：

```text
embeddings.py
  EmbeddingClient 协议
  MockEmbeddingClient
  真实 EmbeddingClient 接口层

embedding_cache.py
  EmbeddingCache 协议
  本地 cache 实现

vector_store.py
  VectorStore 协议
  本地 baseline vector store 实现

manifest.py
  IndexManifest
  IndexManifestWriter
  manifest 校验

builder.py
  IndexBuilder
  离线索引主流程

report.py
  IndexBuildResult
  构建统计和失败记录

errors.py
  索引阶段错误类型
```

是否一开始就拆这么细，要看代码量。但职责边界应该从一开始就清晰。

---

## 28. 学习时应该重点掌握什么

这一子模块新概念很多，但你不需要一开始就把所有向量数据库细节都学完。重点是掌握下面这条主线：

```text
DocumentChunk
  -> EmbeddingClient
  -> embedding vectors
  -> VectorStore
  -> MetadataStore
  -> IndexManifest
  -> persisted index
```

你应该能回答：

1. 为什么 RAG 需要 embedding。
2. embedding model 的维度为什么必须一致。
3. 为什么 query 和 chunk 要使用同一个 embedding model。
4. cosine、dot product、L2 的直觉区别是什么。
5. 为什么向量库不等于完整知识库。
6. metadata 为什么必须和向量一起设计。
7. embedding cache 的 key 应该包含哪些因素。
8. index version 为什么会影响实验可复现。
9. manifest 应该记录哪些信息。
10. mock embedding 能验证什么，不能验证什么。
11. 为什么索引构建应该离线完成。
12. 为什么索引构建要支持幂等和恢复。

---

## 29. 子模块 4 验收标准解释

学习路线中的验收标准如下。

### 29.1 可以用 mock embedding 跑通测试

含义：

```text
没有 API key
没有外部网络
没有真实 embedding 服务
```

系统仍然可以跑通：

```text
chunk -> embedding -> vector store -> manifest
```

这证明工程流程是可测试的。

### 29.2 可以用真实 embedding 构建索引，但不把 API key 写死进代码

含义：

真实模式通过配置打开，敏感信息来自环境变量。

代码中不能出现：

```python
api_key = "sk-..."
```

文档中也不应该出现真实 key。

### 29.3 索引构建支持重复运行，不会无意义重复写入同一批数据

含义：

重复运行同一构建任务时：

- 不重复插入同一个 chunk。
- 不重复 embedding 未变化文本。
- 不生成无法解释的重复索引。

这依赖：

- chunk_id
- content_hash
- embedding cache
- vector upsert
- manifest

### 29.4 index manifest 能说明索引由哪些配置生成

含义：

别人拿到 manifest 后，应该能知道：

- 使用什么 embedding model。
- 使用什么 chunker。
- 使用什么 chunk size。
- 索引多少文档。
- 索引多少 chunks。
- 向量维度是多少。
- 向量库类型是什么。
- 索引什么时候生成。

### 29.5 至少有 5 个索引相关测试

学习路线提到测试覆盖：

- 空 chunk。
- 重复 chunk。
- embedding 维度不一致。
- metadata 缺失。
- 索引加载失败。

你之前已经说明后续练习不要让你把主要精力放在测试实现上，所以后续代码练习中我会直接补好关键测试，把你的练习重点放在工程结构理解和设计取舍上。

---

## 30. 建议的实现顺序

子模块 4 可以按这个顺序推进：

1. 定义 indexing 包结构。
2. 定义 `EmbeddingClient` 协议。
3. 实现 `MockEmbeddingClient`。
4. 定义 `VectorStore` 协议。
5. 选择并实现一个本地 baseline vector store。
6. 定义 `EmbeddingCache`。
7. 定义 `IndexManifest`。
8. 实现 `IndexBuilder` 主流程。
9. 接入 factory。
10. 增加 CLI 或脚本入口。
11. 编写构建报告。
12. 补齐异常处理和测试。

不要一开始就纠结选哪个最强向量库。学习主线应该是：

> 先把索引构建流程设计正确，再替换更强的存储实现。

---

## 31. 一个完整流程示例

假设当前有 2 篇论文，共 100 个 chunks。

索引构建可能是：

```text
1. 读取 chunks
   输入：100 个 DocumentChunk

2. 校验 chunks
   跳过空文本
   检查 doc_id、chunk_id、metadata

3. 查询 embedding cache
   70 个 cache hit
   30 个 cache miss

4. 调用 embed_batch
   对 30 个新 chunk 生成向量

5. 校验向量
   返回数量必须是 30
   每个向量维度必须是 1536

6. 写入 cache
   保存 30 个新向量

7. 写入 vector store
   upsert 100 个 vector items

8. 写入 metadata store
   保存 100 个 chunk 的引用信息

9. 持久化 vector index
   保存到 data/indexes/papers_medium_v1/

10. 写入 manifest
    记录模型、维度、chunk 配置、文档数量、chunk 数量

11. 返回 build result
    indexed_chunks = 100
    cache_hits = 70
    cache_misses = 30
```

这就是一个可以复用、可以解释、可以调试的索引构建流程。

---

## 32. 与后续子模块的衔接

子模块 4 完成后，系统应该具备：

```text
已有 chunks
已有 embedding vectors
已有 vector index
已有 metadata store
已有 manifest
```

子模块 5 会在此基础上实现：

```text
VectorRetriever
BM25Retriever
统一 RetrievedChunk
/search API
```

子模块 6 会继续实现：

```text
hybrid retrieval
rerank
context packing
```

子模块 7 会实现：

```text
query rewrite
answer generation
citation validation
```

所以子模块 4 的质量会直接影响后面所有阶段。

如果索引层设计混乱，后面会出现典型问题：

- 检索结果没有原文。
- 检索结果没有引用信息。
- 分数不可解释。
- 索引无法复现。
- 改了模型但不知道旧索引是否还能用。
- 评测结果无法归因。

---

## 33. 本子模块的核心结论

子模块 4 的本质是：

> 把文本知识片段转换成可检索的向量索引，同时保留足够的工程信息，让索引可以复现、可以调试、可以替换、可以评测。

你应该把它理解成 RAG 系统中的“知识存储层”和“语义检索基础设施”，而不是一次简单的 API 调用。

最重要的工程原则是：

- embedding client 要抽象。
- vector store 要抽象。
- mock 和真实实现要可切换。
- 向量维度必须校验。
- embedding 要缓存。
- 索引要有版本。
- manifest 要完整。
- metadata 不能丢。
- 构建流程要幂等。
- 离线索引和在线检索要分离。

完成这一部分后，你就真正具备了从“文档已经切好”走向“系统可以按语义检索知识”的能力。
