# 子模块 5：Baseline 检索与 BM25 检索概念教学

对应学习路线：模块 2《RAG 知识库与检索增强生成》子模块 5  
核心项目：`paper-rag-assistant`  
学习定位：把已经构建好的索引真正用于检索，并建立可以解释、测试和比较的 retrieval baseline

---

## 1. 子模块 5 要解决什么问题

前四个子模块已经完成了 RAG 的离线索引主链路：

```text
PDF / Markdown / HTML / TXT
  -> RawDocument
  -> ParsedDocument
  -> DocumentChunk
  -> Embedding
  -> VectorCollection
  -> Manifest / Repository
```

到这里，系统已经拥有了可以被检索的索引产物。但“索引存在”不等于“检索系统可用”。

用户真正提问时，系统还需要完成在线 retrieval：

```text
User Query
  -> Retriever
  -> top-k RetrievedChunk
  -> 后续 context packing
  -> 后续 answer generation
```

子模块 5 的核心问题是：

> 给定一个用户问题，系统应该如何从知识库中找出最相关的 chunks，并且让这些检索结果可解释、可测试、可比较？

这一层非常关键，因为后面的生成质量很大程度取决于检索质量。如果检索阶段没有找到正确证据，即使后面的 LLM prompt 写得再漂亮，也很容易出现无依据回答、引用错误或幻觉。

---

## 2. Retrieval 在 RAG 中的位置

RAG 可以拆成两个大流程：

```text
离线索引流程：
Document -> Parse -> Clean -> Chunk -> Embed -> Store

在线问答流程：
Query -> Retrieve -> Pack Context -> Generate Answer -> Cite Sources
```

子模块 5 位于在线问答流程的起点：

```text
Query
  -> Retrieval
  -> RetrievedChunk[]
```

Retrieval 只回答一个问题：

```text
哪些 chunk 可能和用户问题有关？
```

它不负责：

- 组织最终 prompt。
- 生成自然语言回答。
- 判断最终答案是否充分。
- 编造缺失信息。

这条边界很重要。真实工程中，`/search` 和 `/ask` 应该分开：

- `/search`：只返回检索结果、分数、metadata、来源信息。
- `/ask`：在检索结果基础上组织上下文并生成回答。

如果一开始就把检索和生成混在一起，调试会非常困难。用户看到错误回答时，你很难判断问题来自：

- 文档解析失败。
- chunking 粒度不合适。
- embedding 质量差。
- 检索 top-k 没命中。
- rerank 排错了。
- prompt 没组织好。
- LLM 生成时过度推断。

所以子模块 5 的第一目标不是“让系统会回答”，而是“让系统能透明地展示它找到了什么”。

---

## 3. 什么是 Baseline 检索

Baseline 是一个可以稳定运行、可以被后续优化比较的基准实现。

在 RAG 检索中，baseline 的价值是：

1. 先跑通完整链路。
2. 让系统有一个可复现的最低能力。
3. 为后续 hybrid retrieval、rerank、query rewrite、evaluation 提供比较对象。
4. 避免一上来就引入复杂技术，却不知道它到底有没有提升效果。

本项目的 baseline 应至少包括两类检索器：

```text
VectorRetriever
  基于 embedding 向量相似度

BM25Retriever
  基于关键词匹配和词频统计
```

这两类检索器的优势不同。

向量检索适合：

- 语义近似问题。
- 同义表达。
- 中英文语义关联。
- 用户问题和原文不共享明显关键词的场景。

BM25 适合：

- 精确术语。
- 缩写。
- 论文方法名。
- 指标名。
- 代码名。
- 专有名词。
- 用户明确指定某个短语的场景。

真实 RAG 系统通常不会只依赖其中一种，而是逐步走向：

```text
Vector Retrieval
BM25 Retrieval
Hybrid Retrieval
Rerank
```

子模块 5 先学习前两者。

---

## 4. 统一检索结果：RetrievedChunk

不管底层是向量检索、BM25、混合检索还是 rerank，系统最终都应该返回统一结构。

当前项目中的核心模型是：

```python
RetrievedChunk
```

它包含：

```text
chunk_id
doc_id
content_hash
version_id
text
score
rank
retriever
source_path
chunk_index
title
section
page_start
page_end
metadata
```

这里每个字段都有实际用途。

### chunk_id

`chunk_id` 是检索命中的最小单位 ID。

后续 citation、context packing、去重、评测都要依赖它。

如果两个检索器都命中了同一个 chunk，就可以通过 `chunk_id` 判断它们其实指向同一份证据。

### doc_id

`doc_id` 表示 chunk 所属文档。

它用于：

- 聚合同一篇论文的结果。
- 展示来源论文。
- 统计 hit rate。
- 做文档级过滤。

### content_hash 与 version_id

这两个字段用于追踪内容版本。

如果文档内容变了，旧的检索结果就不应该和新索引混在一起比较。

### text

`text` 是真正被检索出来的证据内容。

`/search` 必须返回它，否则无法人工判断检索质量。

### score

`score` 表示该检索器内部认为这个 chunk 与 query 的相关程度。

注意：

> 不同检索器的 score 不能直接比较。

向量检索的 score 可能是 cosine similarity：

```text
0.83
0.72
0.61
```

BM25 的 score 可能是词频、IDF 和长度归一化后的非负数：

```text
12.3
7.8
1.4
```

它们不在同一个数值空间里。

所以：

- 可以比较同一个检索器内部的 rank。
- 不应该直接说 BM25 的 `12.3` 一定比向量检索的 `0.83` 更相关。
- 做 hybrid retrieval 时通常需要归一化、rank fusion 或 rerank。

### rank

`rank` 表示当前检索器内部的排序位置。

例如：

```text
rank=1
rank=2
rank=3
```

评测指标如 MRR、Recall@k、HitRate@k 都会依赖 rank。

### retriever

`retriever` 用来标记结果来自哪个检索器：

```text
vector
bm25
hybrid
rerank
```

这对调试非常重要。

当一个 query 回答失败时，你需要知道是：

- vector 没召回。
- BM25 没召回。
- 两者都召回了但后续融合排错。
- 召回了正确 chunk 但生成阶段没用好。

### source_path、section、page_start、page_end

这些字段让检索结果可以被引用和人工审查。

RAG 系统不是只要回答“像真的”就够了，它应该能说明：

```text
答案来自哪篇文档？
来自哪一页？
来自哪个章节？
具体证据文本是什么？
```

这也是 RAG 比普通聊天模型更适合知识库问答的原因。

---

## 5. Top-k 检索

Top-k 表示每次检索返回前 k 个最相关结果。

例如：

```text
top_k = 5
```

表示返回排序最高的 5 个 chunks。

Top-k 是 RAG 中非常重要的参数。

如果 top-k 太小：

- 正确证据可能没有被召回。
- 回答缺少关键上下文。
- 复杂问题容易漏掉多个方面。

如果 top-k 太大：

- 无关 chunk 会进入上下文。
- LLM 更容易被噪声干扰。
- prompt 变长，成本和延迟上升。
- 引用可能变得混乱。

所以 top-k 不是越大越好。它应该通过评测确定。

常见实验方式是比较：

```text
top_k = 3
top_k = 5
top_k = 8
```

观察：

- HitRate@k 是否提升。
- Recall@k 是否提升。
- 回答是否更忠实。
- 上下文是否引入更多噪声。
- 平均检索耗时是否可接受。

---

## 6. Dense Retrieval：向量检索

Dense retrieval 指基于 dense embedding 的检索方式。

所谓 dense embedding，是指每个文本被表示为一个固定长度的密集向量：

```text
"RAG 如何减少幻觉？"
  -> [0.013, -0.27, 0.81, ...]
```

在当前项目中，子模块 4 已经完成了离线索引：

```text
DocumentChunk.text
  -> EmbeddingClient
  -> VectorRecord
  -> VectorCollection
```

子模块 5 的 `VectorRetriever` 做在线 query 检索：

```text
query
  -> EmbeddingClient.embed_text(query)
  -> VectorCollection.search(query_vector, top_k)
  -> chunk_id
  -> ChunkCollection.get_by_id(chunk_id)
  -> RetrievedChunk
```

### 为什么 query 也要 embedding

向量检索比较的是向量和向量。

文档 chunk 已经在离线阶段转成了向量。用户 query 在在线阶段也必须转成同一 embedding model 的向量。

如果 chunk 使用模型 A，query 使用模型 B，就会出现语义空间不一致。

所以 manifest 中需要记录：

```text
embedding_provider
embedding_model
embedding_dimension
embedding_batch_size
```

加载旧索引时也要校验这些配置。

### 向量相似度

当前项目的 `InMemoryVectorCollection` 使用 cosine similarity。

cosine similarity 衡量两个向量方向是否接近：

```text
cosine(a, b) = dot(a, b) / (||a|| * ||b||)
```

直觉上：

- 越接近 1，方向越相似。
- 越接近 0，相关性越弱。
- 如果允许负值，负数表示方向相反。

在 RAG 检索中，我们通常关心排序，而不是单个分数绝对值。

### VectorRetriever 的工程职责

`VectorRetriever` 不应该知道向量如何持久化，也不应该直接读 JSON。

它只依赖：

```text
EmbeddingClient
VectorCollection
ChunkCollection
```

职责是：

1. 把 query 转成 query vector。
2. 调用 vector collection 做 top-k 搜索。
3. 根据命中的 `chunk_id` 从 chunk collection 补全完整 chunk。
4. 转换成统一的 `RetrievedChunk`。

如果 vector collection 命中了一个不存在的 `chunk_id`，应该清晰失败。

因为这说明：

- vector_collection 和 chunk_collection 不一致。
- 索引文件损坏。
- 构建或加载流程有 bug。

这属于 retrieval 阶段错误，不应该悄悄跳过。

---

## 7. Sparse Retrieval：关键词检索

Sparse retrieval 指基于稀疏词项特征的检索方式。

它不把文本转成 dense embedding，而是关注：

- query 中有哪些词。
- chunk 中有哪些词。
- 词出现了几次。
- 词是否稀有。
- chunk 长度是否影响得分。

BM25 是最经典的 sparse retrieval 算法之一。

---

## 8. BM25 的基本思想

BM25 可以理解为对关键词匹配的工程化打分。

它主要考虑三件事：

1. Query 词是否出现在 chunk 中。
2. 这个词在 chunk 中出现得多不多。
3. 这个词在整个语料中是否稀有。

如果用户问：

```text
faithfulness evaluation
```

某个 chunk 中多次出现：

```text
faithfulness
evaluation
```

它应该得分更高。

但如果一个词在所有 chunk 中都出现，例如：

```text
method
paper
model
```

这个词区分度就不强，权重应该低一些。

这就是 IDF 的作用。

---

## 9. BM25 公式直觉

常见 BM25 形式如下：

```text
score(D, Q) = sum(
  IDF(q_i) * (tf(q_i, D) * (k1 + 1))
             / (tf(q_i, D) + k1 * (1 - b + b * |D| / avgdl))
)
```

其中：

```text
D      当前 chunk
Q      query
q_i    query 中的某个词
tf     词在当前 chunk 中出现次数
IDF    词在整个语料中的稀有程度
|D|    当前 chunk 的长度
avgdl  平均 chunk 长度
k1     控制词频增长的饱和速度
b      控制文档长度归一化强度
```

不需要一开始死记公式，先理解几个直觉。

### tf：词频

如果 query 词在 chunk 中出现，说明相关性增加。

但出现次数不是无限有用。

一个词出现 10 次不应该简单等于出现 1 次的 10 倍重要。

BM25 会让词频收益逐渐饱和。

### IDF：逆文档频率

越稀有的词越有区分度。

例如在 RAG 论文知识库中：

```text
retrieval
generation
model
```

可能很常见。

而：

```text
FiD
Self-RAG
RAGAS
faithfulness
```

可能更有区分度。

BM25 会给稀有词更高权重。

### 长度归一化

长 chunk 天然包含更多词，如果不归一化，长 chunk 更容易匹配 query。

BM25 用 `b` 参数修正这个问题。

如果 `b = 0`：

```text
不考虑 chunk 长度
```

如果 `b = 1`：

```text
完全按 chunk 长度归一化
```

常见默认值是：

```text
k1 = 1.5
b = 0.75
```

当前项目教学版 BM25 也采用类似默认参数。

---

## 10. 中文与英文检索

英文 BM25 通常按单词切分：

```text
"retrieval augmented generation"
  -> ["retrieval", "augmented", "generation"]
```

中文没有天然空格，简单做法可以按单字切分：

```text
"检索增强生成"
  -> ["检", "索", "增", "强", "生", "成"]
```

这种方法可以作为教学 baseline，但不够理想。

问题是：

- “检索”被拆成“检”和“索”，语义变弱。
- “大语言模型”应该是一个词，但单字切分会丢失词组信息。
- 中英文混合文本会更复杂。

真实工程中，可以考虑：

- `jieba`
- `pkuseg`
- HanLP
- Elasticsearch / OpenSearch analyzer
- PostgreSQL text search 配置
- 专门面向中文的 tokenizer

但当前阶段先用轻量 tokenizer 有价值，因为它让你能先理解 BM25 的工程结构，不被分词器细节淹没。

---

## 11. Dense Retrieval 与 BM25 的互补关系

向量检索和 BM25 的差异可以这样理解：

```text
Vector Retrieval:
  更像“按意思找”

BM25:
  更像“按词面找”
```

### 向量检索更擅长的场景

用户问：

```text
RAG 为什么可以减少模型幻觉？
```

chunk 写的是：

```text
Grounding generation in retrieved evidence improves factuality.
```

两者词面差异较大，但语义接近。

向量检索更可能命中。

### BM25 更擅长的场景

用户问：

```text
Self-RAG 的 reflection token 是什么？
```

chunk 中出现：

```text
Self-RAG uses reflection tokens...
```

这里 `Self-RAG` 和 `reflection token` 是精确术语。

BM25 往往很强。

### 两者都会失败的场景

如果文档解析阶段丢掉了关键段落，检索器再好也找不到。

如果 chunking 把一个概念拆碎，BM25 和向量检索都可能只命中半截证据。

如果 query 很模糊，例如：

```text
这个方法有什么问题？
```

没有上下文时，任何检索器都很难知道“这个方法”指什么。

所以 retrieval 不是孤立模块，它依赖前面的 parsing、chunking、metadata，也影响后面的 generation 和 evaluation。

---

## 12. 检索结果去重

真实系统中，多个检索器可能命中同一个 chunk：

```text
VectorRetriever -> chunk_001
BM25Retriever   -> chunk_001
```

也可能命中同一篇文档中相邻或高度重复的 chunks：

```text
chunk_001
chunk_002
chunk_003
```

如果不去重，后续 context packing 会浪费上下文窗口。

去重可以按不同层级做：

### chunk 级去重

按 `chunk_id` 去重：

```text
同一个 chunk 只保留一次
```

### 文本 hash 去重

按 `content_hash` 或 chunk text hash 去重：

```text
内容完全相同的 chunk 只保留一次
```

### 文档窗口去重

如果同一文档中多个相邻 chunk 都命中，可以保留一个主 chunk，再在 context packing 阶段扩展邻近窗口。

这一点后续和 context packing 关系很大。

---

## 13. `/search` API 的意义

`/search` API 应该只做检索。

请求可以类似：

```json
{
  "query": "RAG 如何减少幻觉？",
  "top_k": 5,
  "retriever": "vector"
}
```

响应应该包含：

```json
{
  "query": "RAG 如何减少幻觉？",
  "retriever": "vector",
  "top_k": 5,
  "results": [
    {
      "chunk_id": "...",
      "doc_id": "...",
      "text": "...",
      "score": 0.8231,
      "rank": 1,
      "source_path": "...",
      "section": "Introduction",
      "page_start": 1,
      "page_end": 2,
      "metadata": {}
    }
  ]
}
```

`/search` 的价值是调试：

- query 有没有命中相关 chunk。
- top-k 是否太小。
- 分数排序是否合理。
- metadata 是否保留完整。
- 中文 query 是否能工作。
- BM25 和 vector 哪个更适合当前问题。

在真实 RAG 工程中，`/search` 往往是最重要的排障接口之一。

---

## 14. 检索测试应该覆盖什么

子模块 5 的测试不应该只验证“返回了结果”。

至少要覆盖：

1. `top_k <= 0` 时返回空结果。
2. 空 query 返回空结果或清晰错误。
3. 无匹配结果时返回空列表。
4. vector 命中后能从 `ChunkCollection` 补全完整 chunk。
5. vector 命中不存在的 `chunk_id` 时清晰失败。
6. BM25 能命中英文关键词。
7. BM25 能处理基本中文 query。
8. 检索结果保留 `doc_id/source_path/section/page_start/page_end/metadata`。
9. 分数排序稳定。
10. 重复 chunk 不会污染最终结果。

这些测试的目标不是证明检索质量“很好”，而是保证 retrieval 的工程行为可控。

真正的质量评估要等后续 evaluation 子模块，用人工问题集和指标来做。

---

## 15. 常见错误与排查方式

### 错误 1：向量检索结果看起来随机

可能原因：

- 当前使用 mock embedding，它不理解真实语义。
- query embedding 和 chunk embedding 的模型不一致。
- 向量维度配置不一致。
- chunk 文本太短或太碎。
- 解析结果质量差。

排查方式：

- 查看 manifest 中的 embedding model 和 dimension。
- 查看 chunk 文本是否正常。
- 使用 `/search` 直接观察 top-k。
- 对比 BM25 是否能命中关键词。

### 错误 2：BM25 命不中语义相近问题

这是正常现象。

BM25 依赖词面重叠。如果 query 和 chunk 没有共享关键词，它很难命中。

解决方向：

- 使用向量检索。
- 做 query rewrite。
- 使用 hybrid retrieval。
- 引入同义词扩展。

### 错误 3：BM25 被高频通用词干扰

可能原因：

- tokenizer 太粗糙。
- 没有停用词过滤。
- chunk 太长。
- 语料太小，IDF 不稳定。

解决方向：

- 加停用词表。
- 改进 tokenizer。
- 调整 chunk size。
- 用评测集观察真实影响。

### 错误 4：top-k 增大后回答变差

可能原因：

- 召回了更多无关 chunks。
- context packing 没有去重。
- LLM 被弱相关证据干扰。
- 分数没有归一化或 rerank。

解决方向：

- 分析 `/search` 返回结果。
- 比较 top-k=3、5、8。
- 加 rerank。
- 做 context packing 去重和预算控制。

---

## 16. 与后续子模块的关系

子模块 5 的输出是：

```text
RetrievedChunk[]
```

后续模块会继续处理它。

### 与 context packing 的关系

Context packing 会决定：

- 哪些 RetrievedChunk 进入 prompt。
- 每个 chunk 截取多少内容。
- 是否合并相邻 chunk。
- 是否去重。
- 是否保留 citation marker。

如果 retrieval 返回结果混乱，context packing 会很难做好。

### 与 answer generation 的关系

Answer generator 不应该直接访问 vector collection 或 BM25 index。

它应该只看到：

```text
question
retrieved_chunks
packed_context
```

这样生成层和检索层可以独立测试。

### 与 evaluation 的关系

检索评测会关注：

- HitRate@k
- Recall@k
- MRR
- nDCG
- context precision

这些指标都依赖子模块 5 的检索结果结构。

---

## 17. 当前项目中的工程落点

当前项目已经有这些相关对象：

```text
app/retrieval/models.py
  RetrievedChunk

app/retrieval/retrievers.py
  Retriever
  VectorRetriever
  BM25Retriever

app/indexing/vector_collection.py
  VectorCollection
  VectorSearchResult

app/ingest/chunking/collection.py
  ChunkCollection

app/api/schemas.py
  SearchRequest
  SearchResponse
  RetrievedChunkResponse
```

子模块 5 的后续工程练习可以围绕这些内容继续完善：

- 让 `VectorRetriever`、`BM25Retriever` 的结构更标准。
- 把检索策略接入 factory。
- 让 `/search` 真正调用当前配置的 retriever。
- 为 BM25 建立更明确的 collection 或 index 对象。
- 增加检索结果去重。
- 补齐中文、英文、无结果、top-k 边界等测试。

---

## 18. 本子模块你需要真正掌握什么

学完子模块 5 后，你应该能清楚解释：

1. Retrieval 和 generation 为什么要分层。
2. `/search` 为什么是 RAG 系统重要的调试入口。
3. `RetrievedChunk` 为什么要保留 score、rank、retriever 和 metadata。
4. Vector retrieval 的在线流程是什么。
5. Query embedding 为什么必须和 index embedding 使用同一模型空间。
6. BM25 如何利用 TF、IDF 和长度归一化打分。
7. 为什么 BM25 和向量检索分数不能直接比较。
8. 为什么 top-k 不是越大越好。
9. 为什么 BM25 擅长精确术语，向量检索擅长语义近似。
10. 为什么检索结果去重和 metadata 保留会影响后续引用与回答质量。

---

## 19. 子模块 5 的验收标准

概念层面，你应该能做到：

1. 画出 query 到 top-k chunks 的完整 retrieval 流程。
2. 解释 dense retrieval 和 sparse retrieval 的差异。
3. 解释 BM25 公式中 TF、IDF、chunk length normalization 的作用。
4. 说明为什么不同 retriever 的 score 不能直接比较。
5. 说明 `/search` 和 `/ask` 的职责差异。
6. 能举例说明向量检索失败、BM25 检索失败分别可能是什么原因。

工程层面，后续练习应能做到：

1. `VectorRetriever` 和 `BM25Retriever` 都能独立运行。
2. 检索结果统一返回 `RetrievedChunk`。
3. `/search` 返回 chunk 内容、分数、来源和 metadata。
4. 对同一个 query 可以比较 vector 与 BM25 的结果差异。
5. 至少覆盖无结果、top-k 边界、metadata 保留、中文 query、英文 query 等测试场景。

---

## 20. 建议学习顺序

建议按以下顺序学习：

1. 先理解 `RetrievedChunk` 为什么是检索层统一输出。
2. 再理解 `VectorRetriever` 如何把 query embedding 和 vector search 串起来。
3. 再理解 BM25 的 TF、IDF、长度归一化。
4. 再比较 vector 和 BM25 的优缺点。
5. 再看 `/search` API 应该如何暴露检索结果。
6. 最后再考虑去重、hybrid retrieval 和 rerank。

这一顺序能避免一开始就被 hybrid 或 rerank 搅乱。先把“单检索器 baseline”打稳，后续优化才有比较对象。
