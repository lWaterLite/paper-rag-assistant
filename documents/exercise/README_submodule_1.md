# 子模块 1 练习：RAG 系统全链路与项目骨架

本练习对应模块 2 的子模块 1。目标不是马上接入真实 PDF、真实 embedding 或真实向量数据库，而是先搭出一个可运行、可观察、可继续扩展的 RAG 工程骨架。

---

## 1. 本练习你会得到什么

当前代码已经搭好了一个最小 mock RAG 项目骨架：

```text
app/
  api/                # API 层占位，后续接 FastAPI
  core/               # 配置、错误、核心数据模型
  ingest/             # 文档加载、解析、切分
  indexing/           # mock embedding、内存向量库、索引构建
  retrieval/          # 向量检索、上下文组织
  generation/         # mock 回答生成
  storage/            # 内存仓储
  pipeline.py         # 在线 RAG 问答流程
  main.py             # CLI 入口
data/raw/papers/      # 练习用 mock 文档
tests/                # unittest 自检
```

这个骨架已经包含两条核心流程：

1. 离线索引流程：

```text
LocalTextLoader
  -> PlainTextParser
  -> CharacterChunker
  -> MockEmbeddingClient
  -> InMemoryVectorCollection
```

2. 在线问答流程：

```text
用户问题
  -> VectorRetriever
  -> SimpleContextPacker
  -> MockAnswerGenerator
  -> RagAnswer
```

---

## 2. 当前没有直接修改的环境配置

我没有为你安装依赖，也没有修改虚拟环境。

当前练习已经使用 `pydantic` 和 `pydantic-EnvSettings` 来管理项目配置。它们已经写入 `pyproject.toml`，但你需要自行同步或安装依赖。

如果你使用 uv，可以运行：

```powershell
uv sync
```

如果你使用 pip，可以运行：

```powershell
python -m pip install "pydantic>=2.7" "pydantic-EnvSettings>=2.2"
```

这两个库的作用：

- `pydantic`：负责类型转换、字段约束和跨字段校验。
- `pydantic-EnvSettings`：负责从环境变量、`.env` 文件等来源读取配置。

如果你后续要把它升级成真实工程，可以自行考虑这些配置：

```text
RAG_CHUNK_SIZE=500
RAG_CHUNK_OVERLAP=80
RAG_TOP_K=3
RAG_MAX_CONTEXT_CHARS=1800
RAG_MOCK_EMBEDDING_DIMENSION=16
RAG_REQUIRE_CITATION=true
```

如果后续接入真实能力，可能还需要：

```text
LLM_PROVIDER=openai-compatible
LLM_MODEL=...
LLM_API_KEY=...
EMBEDDING_PROVIDER=openai-compatible
EMBEDDING_MODEL=...
EMBEDDING_API_KEY=...
VECTOR_STORE=faiss 或 chroma 或 qdrant
```

这些配置本次不会自动写入 `.env`，你可以根据需要自行创建。

---

## 3. 如何运行

在 `paper-rag-assistant` 根目录下运行。

构建练习用内存索引：

```powershell
python -m app.main index --source data/raw/papers
```

执行一次 mock RAG 问答：

```powershell
python -m app.main ask "RAG 为什么需要引用？" --source data/raw/papers
```

运行自检：

```powershell
python -m unittest discover -s tests
```

如果你的终端默认没有激活虚拟环境，可以使用你自己的 Python 解释器路径运行。这里不要求安装任何新依赖。

---

## 4. 当前代码的重要文件

### `app/core/models.py`

定义 RAG 系统的核心数据模型：

- `RawDocument`
- `ParsedDocument`
- `DocumentChunk`
- `RetrievedChunk`
- `Citation`
- `RagAnswer`
- `RagTrace`

学习重点：

- 每个模型属于 pipeline 的哪个阶段。
- 每个模型为什么需要 metadata。
- 为什么最终回答不能只是字符串。

### `app/indexing/index_builder.py`

定义离线索引流程。

学习重点：

- loading、parsing、chunking、indexing 如何串联。
- 为什么索引构建应该和用户问答分开。
- trace 如何记录每个阶段。

### `app/pipeline.py`

定义在线问答流程。

学习重点：

- retrieval、context packing、generation 如何串联。
- 为什么检索结果和引用要随最终回答一起返回。
- 失败时应该如何记录 trace。

### `app/indexing/embeddings.py`

定义 mock embedding。

学习重点：

- embedding client 为什么应该抽象成接口。
- mock embedding 能测试 pipeline，但不能代表真实语义检索。

---

## 5. TODO 练习清单

### 练习 1：配置校验

位置：`app/core/ProjectSettings.py`

当前已升级为 `pydantic-EnvSettings` 方案。你需要重点阅读：

- `SettingsConfigDict`
- `Field(default=..., gt=..., ge=...)`
- `model_validator(mode="after")`
- `EnvSettings.from_env()`
- `ValidationError` 和项目统一 `AppError` 的区别

追加任务：

- 增加 `retrieval_strategy` 配置，只允许 `vector`、`bm25`、`hybrid`。
- 增加 `index_storage_path` 配置，并思考它应该是字符串还是 `Path`。
- 增加 `debug_trace` 配置，用来控制响应中是否返回完整 trace。
- 为新增配置补充测试。

思考：

- 配置错误应该在应用启动时暴露，还是等到请求执行时暴露？
- 为什么实际工程里通常不建议在业务代码中到处调用 `os.getenv()`？
- 为什么跨字段校验不适合只用 `Field` 完成？

### 练习 2：doc_id 设计

位置：`app/ingest/loaders.py`

任务：

- 当前 `doc_id` 只基于路径生成。
- 你可以尝试加入文件内容 hash。
- 思考路径变化、内容变化、版本变化时 doc_id 应该如何处理。

思考：

- 如果一篇论文内容更新了，它应该是同一个 doc_id 的新版本，还是一个新的 doc_id？

### 练习 3：文本清洗

位置：`app/ingest/parsers.py`

任务：

- 合并连续空行。
- 去除行尾多余空格。
- 保留 Markdown 标题。
- 为清洗前后文本长度做记录。

思考：

- 清洗是否可能误删有价值信息？

### 练习 4：SectionAwareChunker

位置：`app/ingest/chunkers.py`

任务：

- 实现按 Markdown 标题优先切分。
- 在 metadata 中记录 section。
- 对过长 section 再做二次切分。

思考：

- 对论文来说，章节边界为什么比纯字符长度更有意义？

### 练习 5：真实 EmbeddingClient 设计

位置：`app/indexing/embeddings.py`

任务：

- 不需要直接接入真实服务。
- 先设计真实 embedding client 需要哪些参数。
- 思考 batch、timeout、重试、cache。

思考：

- 为什么 embedding 成本通常需要缓存？

### 练习 6：向量维度校验

位置：`app/indexing/vector_collection.py`

任务：

- 搜索时检查 query vector 和 record vector 维度。
- add 时也可以检查向量维度是否一致。
- 维度错误时抛出清晰异常。

思考：

- 如果换了 embedding 模型，旧索引还能继续使用吗？

### 练习 7：索引 manifest

位置：`app/indexing/manifest.py`

任务：

- 设计一个 JSON manifest。
- 记录 index_id、chunk_size、chunk_overlap、embedding model、document_count、chunk_count。
- 暂时可以只写设计，不一定实现持久化。

思考：

- 为什么没有 manifest 的实验结果很难复现？

### 练习 8：避免重复 embedding

位置：`app/indexing/index_builder.py`

任务：

- 设计 chunk hash。
- 思考 embedding cache。
- 思考内容没有变化时如何跳过重复 embedding。

思考：

- embedding cache 应该按 chunk_id 缓存，还是按 text hash 缓存？

### 练习 9：BM25Retriever 接口预留

位置：`app/retrieval/retrievers.py`

任务：

- 定义 `BM25Retriever` 的基本接口。
- 保证它也返回 `RetrievedChunk`。
- 思考 BM25 分数和向量相似度分数如何融合。

思考：

- 为什么关键词检索和向量检索经常互补？

### 练习 10：Context Packing 改进

位置：`app/retrieval/context_packer.py`

任务：

- 去重。
- 合并相邻 chunk。
- 记录被丢弃的 chunk 和原因。
- 保证 citation id 不丢失。

思考：

- 检索到了正确 chunk，但 context packing 没放进去，最终回答会发生什么？

### 练习 11：真实回答生成 Prompt 设计

位置：`app/generation/answer_generator.py`

任务：

- 设计真实 LLM prompt。
- 要求只基于 context 回答。
- 信息不足时拒绝编造。
- 回答必须带 citation id。

思考：

- prompt 能否完全阻止模型幻觉？

### 练习 12：失败 trace

位置：`app/pipeline.py`

任务：

- 为 retrieval、context packing、generation 增加异常捕获。
- 失败时记录 `final_status` 和 `failure_type`。
- 返回或抛出带 trace_id 的错误。

思考：

- 如果没有 trace，一次错误回答应该如何定位？

### 练习 13：API schema 设计

位置：`app/api/routes.py`

任务：

- 设计 `/ask` 请求 JSON。
- 设计 `/ask` 响应 JSON。
- 设计 `/search` 响应 JSON。
- 暂时不需要引入 FastAPI。

思考：

- `retrieved_chunks` 应该默认返回给用户，还是只在 debug 模式返回？

### 练习 14：补充自检测试

位置：

- `tests/test_config_settings.py`
- `tests/test_document_identity.py`
- `tests/test_rag_pipeline.py`

任务：

- 测试目录不存在。
- 测试非法 chunk 配置。
- 测试空文档。
- 测试 context budget 很小时的行为。

思考：

- 哪些测试属于单元测试，哪些属于集成测试？

---

## 6. 本练习的验收标准

完成本练习后，你应该能够：

- 解释离线索引流程和在线问答流程的区别。
- 说明每个核心数据模型在 pipeline 中的位置。
- 运行 CLI 构建 mock 索引。
- 运行 CLI 执行一次 mock RAG 问答。
- 查看回答中的 citations 和 trace_id。
- 至少完成 5 个 TODO。
- 至少补充 3 个自检测试。
- 能说明当前 mock embedding 和真实 embedding 的区别。

---

## 7. 当前练习的刻意限制

当前代码有意没有实现：

- 真实 PDF 解析。
- 真实 embedding。
- 真实向量数据库。
- BM25。
- rerank。
- FastAPI。
- LLM 回答生成。
- 持久化索引。

这些不是遗漏，而是为了让你先掌握 RAG 工程结构。

子模块 2 会逐步进入文档解析和清洗，子模块 3 会深入 chunking，后续再接 embedding、向量库、hybrid retrieval、rerank 和 evaluation。
