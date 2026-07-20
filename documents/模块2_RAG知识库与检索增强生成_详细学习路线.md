# 模块 2：RAG 知识库与检索增强生成详细学习路线

生成日期：2026-06-02  
对应主文档：`documents/AI_Agent_学习路径_硕士一年级到求职.md` 的“阶段 B：RAG 与知识库”和“模块 2：RAG 知识库与检索增强生成”  
阶段定位：模块 2，围绕论文知识库构建可评测、可解释、可工程化的 RAG 系统  
核心项目：`paper-rag-assistant`

---

## 1. 阶段总目标

模块 1 已经完成了最小工具调用 Agent、FastAPI 服务、配置管理、测试、日志、Streaming 和真实 LLM 接入。模块 2 的目标不是再做一个简单“把文档塞进向量库然后问答”的 demo，而是构建一个可以被评测、可以定位失败原因、可以逐步优化检索质量的论文 RAG 系统。

完成本阶段后，你应该能够：

1. 解释 RAG 的完整工程链路：loading、parsing、chunking、indexing、storing、retrieval、rerank、context packing、generation、citation、evaluation。
2. 独立完成 PDF、Markdown、HTML 等多种文档的解析、清洗、切分和 metadata 设计。
3. 构建 dense retrieval、BM25、hybrid retrieval、rerank 等多种检索策略，并能用实验数据比较效果。
4. 设计带引用溯源的回答生成流程，让系统能够说明“答案来自哪篇论文、哪一节、哪个片段”。
5. 建立人工评测集，评估 hit rate、Recall@k、MRR、context precision、answer relevance、faithfulness 等指标。
6. 能分析一次错误回答属于解析失败、切分失败、检索失败、重排失败、上下文组织失败、引用失败，还是模型生成失败。
7. 将 RAG pipeline 服务化，提供可测试的 API、日志、配置和 README。
8. 能用一份 `EVALUATION.md` 证明某个优化确实提升了系统指标。

阶段完成物：

- 一个独立工程项目：`paper-rag-assistant`
- 一套论文知识库数据目录
- 一套可持久化的索引和检索服务
- 一个支持中文问答与引用来源的 RAG API
- 一组检索与回答生成测试
- 一份评测集，例如 `data/eval/questions.jsonl`
- 一份实验报告：`EVALUATION.md`
- 一份项目 README

---

## 2. 核心项目范围

项目名：`paper-rag-assistant`

项目目标：

- 以 AI Agent、RAG、LLM Security、LLM Evaluation 等论文作为知识库。
- 先用 5 到 10 篇论文建立 baseline，再扩展到 30 篇以上。
- 支持中文提问，回答中必须包含引用来源。
- 能返回相关论文列表、相关片段、相似度分数、metadata。
- 必须包含 eval 脚本，用于比较 chunk size、top-k、是否 rerank、检索策略等配置。

建议项目结构：

```text
paper-rag-assistant/
  app/
    main.py
    api/
      routes.py
      schemas.py
    core/
      config.py
      errors.py
      logging.py
    ingest/
      loaders.py
      parsers.py
      cleaners.py
      chunkers.py
      metadata.py
    indexing/
      embeddings.py
      vector_collection.py
      bm25_store.py
      index_builder.py
    repositories/
      vector.py
      chunk.py
      document.py
    retrieval/
      retrievers.py
      hybrid.py
      rerankers.py
      context_packer.py
    generation/
      prompts.py
      llm_client.py
      answer_generator.py
      citations.py
    evaluation/
      datasets.py
      metrics.py
      runner.py
      reports.py
    storage/
      documents.py
      repositories.py
  data/
    raw/
      papers/
      markdown/
      html/
    parsed/
    indexes/
    eval/
      questions.jsonl
      runs/
  documents/
    notes/
  tests/
    test_chunkers.py
    test_retrievers.py
    test_rag_pipeline.py
    test_evaluation.py
  README.md
  EVALUATION.md
  .env.example
  pyproject.toml
```

工程原则：

- RAG pipeline 要可拆分，不要把解析、检索、生成都写在一个函数里。
- 每个阶段都要有清晰输入输出模型，便于测试和定位问题。
- 外部服务，例如 LLM、embedding model、reranker、vector DB，要通过接口抽象，不要在业务逻辑里写死。
- 初期可以使用本地向量库和 mock embedding，但结构要允许替换真实模型和真实数据库。
- 回答生成不能只返回自然语言，还要返回 `answer`、`citations`、`retrieved_chunks`、`trace_id`、`latency_ms` 等结构化字段。

---

## 3. 本阶段能力拆解

### 子模块 1：RAG 系统全链路与项目骨架

学习目标：

- 从工程视角理解 RAG 不是一个“检索 + 生成”的单函数，而是一条可观测、可评测、可替换的 pipeline。
- 明确论文 RAG 项目的输入、输出、状态、配置和边界。
- 建立项目骨架，为后续每个子模块预留清晰接口。

需要掌握：

- RAG 的五个核心阶段：loading、indexing、storing、querying、evaluation。
- 离线索引流程和在线问答流程的区别。
- 文档状态管理：raw、parsed、chunked、indexed。
- pipeline trace：一次请求经过了哪些阶段，每个阶段耗时和产物是什么。
- 配置分层：模型配置、索引配置、检索配置、生成配置、评测配置。

实践任务：

1. 创建 `paper-rag-assistant` 项目骨架。
2. 定义核心数据模型：
   - `RawDocument`
   - `ParsedDocument`
   - `DocumentChunk`
   - `RetrievedChunk`
   - `Citation`
   - `RagAnswer`
   - `RagTrace`
3. 编写 `.env.example`，至少包含：
   - LLM provider 配置
   - embedding provider 配置
   - vector store 类型
   - top-k
   - chunk size
   - chunk overlap
   - rerank 开关
4. 设计离线索引命令，例如：
   - `python -m app.indexing.index_builder --source data/raw/papers`
5. 设计在线问答 API，例如：
   - `POST /ask`
   - `POST /search`
   - `GET /documents`
   - `GET /health`

验收标准：

- 能画出或说明离线索引流程和在线问答流程的差异。
- 项目结构中解析、索引、检索、生成、评测相互独立。
- 任意一个模块替换实现时，不需要大范围修改其他模块。
- README 中能说明项目如何启动、如何导入论文、如何提问、如何评测。

思考题：

1. 为什么 RAG 系统需要把“建索引”和“用户问答”拆成两个流程？
2. 为什么“文档已经进入向量库”不等于“RAG 系统已经做好”？
3. 如果一次回答错误，你需要哪些 trace 信息才能定位原因？

---

### 子模块 2：文档加载、解析与清洗

学习目标：

- 掌握论文、Markdown、HTML 等文档的加载方式。
- 理解 PDF 文本提取中的常见问题，例如断行、页眉页脚、参考文献、公式、表格、双栏排版。
- 为后续 chunking 和 citation 保留足够 metadata。

需要掌握：

- PDF loader、Markdown loader、HTML loader。
- 文档解析和文档清洗的区别。
- 原始文本、规范化文本、结构化段落之间的关系。
- title、authors、year、section、page、source_path、url、doi 等 metadata。
- 解析质量对检索效果的影响。

实践任务：

1. 实现 `PDFLoader`、`MarkdownLoader`、`HTMLLoader` 的统一接口。
2. 将每份文档解析成统一的 `ParsedDocument`。
3. 对 PDF 文本做基础清洗：
   - 合并错误断行。
   - 去除重复页眉页脚。
   - 保留页码信息。
   - 尽量保留章节标题。
4. 建立 metadata 提取逻辑：
   - 文件名
   - 文档标题
   - 来源路径
   - 页码范围
   - 文档类型
   - 解析时间
5. 选择 5 篇论文作为 baseline 数据，记录解析质量问题。

验收标准：

- 至少支持 PDF 和 Markdown 两种文档。
- 每个解析后的文档都有稳定的 `doc_id`。
- 每段文本都能追溯到原始文档和页码或章节。
- 至少有 5 个解析器测试，包括空文件、损坏文件、中文内容、英文论文、缺失 metadata。
- 能列出 3 个 PDF 解析失败会导致 RAG 错误的例子。

工程注意事项：

- 不要在解析阶段丢掉来源信息，否则后续无法引用。
- 不要为了“文本更短”过度清洗，参考文献、标题、表格说明可能对检索有价值。
- 对解析失败的文件要记录错误状态，而不是让整个索引流程中断。

---

### 子模块 3：Chunking 策略与 Metadata 设计

学习目标：

- 理解 chunk size、overlap、语义边界、章节边界对检索质量的影响。
- 学会设计能支持引用、过滤、聚合和评测的 chunk metadata。
- 让 chunking 结果可复现、可测试、可比较。

需要掌握：

- 固定长度切分。
- 递归字符切分。
- 按段落、标题、章节切分。
- token-based chunking。
- overlap 的作用和副作用。
- parent-child chunk、section-aware chunk、sentence window retrieval。

实践任务：

1. 实现至少两种 chunker：
   - `FixedTokenChunker`
   - `SectionAwareChunker`
2. 为每个 chunk 生成：
   - `chunk_id`
   - `doc_id`
   - `text`
   - `section`
   - `page_start`
   - `page_end`
   - `token_count`
   - `chunk_index`
3. 为 chunker 编写质量检查脚本：
   - chunk 是否为空。
   - chunk 是否过长。
   - chunk 是否丢失 doc_id。
   - overlap 是否符合配置。
4. 对同一批论文生成多组 chunk 配置：
   - small：约 300 tokens
   - medium：约 600 tokens
   - large：约 1000 tokens
5. 保存 chunk 统计报告，例如：
   - chunk 总数
   - 平均 token 数
   - 最大 token 数
   - 每篇文档 chunk 数量

验收标准：

- 至少两种 chunking 策略可通过配置切换。
- chunk 结果可持久化，便于后续复现实验。
- 每个 chunk 都能追溯到文档、章节和页码。
- 至少有 8 个 chunker 单元测试。
- 能说明“chunk 越大”和“chunk 越小”分别会带来什么问题。

工程注意事项：

- chunk 不是越小越好。太小会缺上下文，太大则会降低检索精度并浪费上下文窗口。
- overlap 能缓解边界断裂，但会增加索引体积和重复上下文。
- 对论文类文档，章节信息通常比纯字符长度更重要。

---

### 子模块 4：Embedding、向量索引与持久化存储

学习目标：

- 理解 embedding 如何把文本转成向量，以及相似度搜索的基本原理。
- 设计可替换的 embedding client 和 vector store。
- 建立可重复构建、可缓存、可恢复的索引流程。

需要掌握：

- embedding model 的输入限制、维度、成本和批处理。
- cosine similarity、dot product、L2 distance。
- FAISS、Chroma、Qdrant、Milvus、pgvector 的差异。
- embedding cache。
- index version。
- 向量库和文档 metadata 存储的关系。

实践任务：

1. 定义 `EmbeddingClient` 抽象接口：
   - `embed_text(text: str) -> list[float]`
   - `embed_batch(texts: list[str]) -> list[list[float]]`
2. 实现 mock embedding，保证本地测试不依赖真实外部服务。
3. 实现真实 embedding client 的接口层，但 API key 只通过 `.env` 或环境变量读取。
4. 选择一个本地向量库作为 baseline，例如 FAISS 或 Chroma。
5. 实现索引构建流程：
   - 读取 chunks
   - 批量 embedding
   - 写入 vector store
   - 写入 metadata store
   - 保存 index manifest
6. 实现索引 manifest，例如：

```json
{
  "index_id": "papers_medium_v1",
  "embedding_model": "text-embedding-xxx",
  "chunker": "section_aware",
  "chunk_size": 600,
  "chunk_overlap": 100,
  "document_count": 10,
  "chunk_count": 850
}
```

验收标准：

- 可以用 mock embedding 跑通测试。
- 可以用真实 embedding 构建索引，但不把 API key 写死进代码。
- 索引构建支持重复运行，不会无意义重复写入同一批数据。
- 索引 manifest 能说明索引由哪些配置生成。
- 至少有 5 个索引相关测试，包括空 chunk、重复 chunk、embedding 维度不一致、metadata 缺失、索引加载失败。

工程注意事项：

- embedding 是高成本环节，要优先考虑 batch 和 cache。
- 索引配置变更后要生成新的 index version，否则实验结果不可追溯。
- 不要只把文本放进向量库，还要保存能支撑引用和调试的 metadata。

---

### 子模块 5：多策略检索、Hybrid 融合与检索可观测性

实际完成情况：

本子模块已从最初的 vector/BM25 baseline 扩展为完整的候选检索子系统。它完成了
`VectorRetriever`、`BM25Retriever`、加权 RRF `HybridRetriever`、`RetrieverRegistry`、
`TokenizerRegistry`、`SearchService`、检索报告和 compare search。基础的字符预算版
`SimpleContextPacker` 也已接入在线 RAG 调用链，负责为后续生成阶段提供去重、相邻合并和
citation 标记后的上下文。

这一阶段的定位是“找什么、如何组合候选、如何看见检索过程”。它不负责把候选做模型级
重排序，也不负责正式检索指标评测；两部分分别留给子模块 6 与子模块 8。

学习目标：

- 建立 dense、sparse 与 hybrid 三种可独立运行的候选检索策略。
- 理解 BM25、向量相似度和 RRF 的分数及排名语义差异。
- 让检索结果具有稳定数据契约、来源信息、运行 trace 和可持久化报告。
- 建立单策略 search 与多策略 compare search 的清晰边界。

需要掌握：

- query embedding、top-k、向量相似度与关键词召回。
- BM25 的 TF、IDF、长度归一化和 tokenizer 一致性。
- dense retrieval 与 sparse retrieval 的互补关系。
- `RetrievedChunk`、`RetrievalSignal`、rank 与原始 score 的数据边界。
- `Retriever` Protocol、策略注册表、惰性 provider 与依赖注入。
- reciprocal rank fusion（RRF）、候选集扩张、加权融合与 chunk-id 聚合。
- 检索结果去重、最终 top-k 截断与 pipeline 后处理阶段。
- retrieval trace、执行报告、失败报告、compare search 与 overlap 分析。
- 基础 context packing：去重、相邻 chunk 合并、字符预算与 citation 标记。

实践任务：

1. 实现并统一三种候选检索器：
   - `VectorRetriever` 使用 query embedding 和 `VectorCollection` 搜索。
   - `BM25Index` 与 `BM25Retriever` 分离索引统计和在线检索职责。
   - `HybridRetriever` 通过多个 `Retriever` 召回源和 RRF 输出统一 `RetrievedChunk`。
2. 设计统一检索结果与运行时证据：
   - 保留 chunk、文档、章节、页码、来源路径和 metadata。
   - 将每个召回源的 score/rank 保存在运行时 `RetrievalSignal`，不写入文档 metadata。
3. 使用 registry 与 factory 组装策略：
   - 由 `RetrieverRegistry` 校验策略名、惰性构建并缓存实例。
   - 由 `TokenizerRegistry` 确保 BM25 建索引和 query 使用同一分词规则。
   - `settings.toml -> Settings -> ConfigFactory -> Config -> Factory` 保持单向配置流。
4. 实现检索流程与对外能力：
   - `SearchService -> RetrievalPipeline -> Retriever -> ResultStage[]`。
   - `/search` handler 返回结果、来源、rank、score 和可选 trace。
   - `CompareSearchService` 并列执行多个策略，返回独立结果、失败信息和 chunk overlap。
5. 完成检索领域报告：
   - 单策略请求记录候选数、去重数、最终返回数、阶段耗时、索引与配置快照。
   - compare search 写入父级聚合报告，并关联各子策略报告和 trace。
   - 成功与失败请求都保留可追溯记录。
6. 实现基础上下文组织：
   - 去除重复 chunk 与重复文本。
   - 合并同一文档的相邻 chunk。
   - 在字符预算内生成包含 citation id 的 `PackedContext`。

验收标准：

- vector、BM25、hybrid 三种策略均通过同一个 `Retriever` 协议与 registry 运行。
- BM25 索引文本与查询文本使用同一个可注入 tokenizer，且中英文基础检索可运行。
- hybrid 不直接相加 BM25 与向量原始分数，而是使用加权 RRF；每个结果保留多路检索证据。
- `/search` 的每条结果都包含文本、rank、score、来源路径、文档/章节/页码 metadata。
- compare search 能在同一 query 下展示多个策略的独立结果、失败状态和 overlap，不把它误当作融合排序。
- retrieval pipeline 的去重和 top-k 处理对 API、CLI 与 RAG pipeline 一致生效。
- 检索成功、检索失败和比较检索都能写入结构化报告并关联 trace id。
- 基础 context packing 能去重、合并相邻 chunk、记录 dropped reason，并保留用于后续生成的 citation 信息。
- 已有检索测试覆盖无结果、top-k、重复结果、metadata、tokenizer、registry、hybrid、报告和比较模式。
- 能解释为什么向量检索不一定命中精确术语、BM25 不一定命中语义近似问题，以及为什么不同策略的原始 score 不能直接比较。

工程注意事项：

- 检索阶段先暴露“找到了什么”，不能把回答生成混入 `SearchService` 或 `RetrievalPipeline`。
- `HybridRetriever` 的职责是融合候选召回，不承担 rerank；rerank 将在子模块 6 以独立后处理阶段接入。
- compare search 是观察与诊断工具，不是分数融合器，也不是正式评测系统。
- 当前 `SimpleContextPacker` 仍使用字符预算且合并段只以基础 citation 表达来源；token-aware 预算和完整段级 provenance 将在子模块 6 完成。
- 本子模块的 compare/report 用于工程诊断，不替代 golden dataset、Recall@k、MRR 和实验管理；正式评测基础设施仍属于子模块 8。

---

### 子模块 6：重排序、检索后处理与上下文预算

调整说明：

子模块 5 已经提前完成了 `HybridRetriever`、加权 RRF 融合、基础去重、字符预算版
`ContextPacker`、检索比较接口和检索报告。因此子模块 6 不再重复实现 hybrid retrieval，
而是以现有 `vector`、`bm25`、`hybrid` 为候选召回基线，专注完成“候选结果如何成为
可靠 LLM 上下文”的后半段流程。

本次调整不改变后续模块边界：query rewrite、回答生成和回答级 citation 校验仍属于子模块
7；正式 golden dataset、指标计算和实验管理仍属于子模块 8；HTTP 服务、Streaming 和通用
请求可观测性仍属于子模块 9；生产安全、权限、成本和部署仍属于子模块 10。

学习目标：

- 理解两阶段检索：先扩大候选召回，再用 rerank 选择真正值得进入上下文的 chunk。
- 将 rerank 设计成可替换、可配置、可降级的检索后处理阶段，而不是耦合在某个 retriever 中。
- 将现有字符预算版 context packing 升级为面向模型上下文窗口的 token 预算与来源追溯流程。
- 保持现有 hybrid、report 和 compare search 的稳定语义，并为子模块 7 的生成与引用校验提供可靠输入。

需要掌握：

- candidate retrieval、final selection 与 two-stage retrieval。
- reranker、bi-encoder 与 cross-encoder 的职责差异。
- pointwise、pairwise、listwise rerank 的基本思想。
- rerank score、原始 retrieval score 与 rank 的语义边界。
- `Reranker` Protocol、`RerankStage`、策略注册表与故障降级。
- candidate limit、final top-k 与 rerank batch 的配置关系。
- token budget、token estimator、模型上下文窗口和输出预留。
- context packing、文档多样性、相邻 chunk 合并、来源追溯与安全的 extractive compression。
- rerank 与 context packing 的阶段统计和领域级报告。

实践任务：

1. 重构 retrieval pipeline 的候选与最终结果边界：
   - 明确候选数量 `candidate_limit` 与最终返回数量 `top_k`。
   - 保持 `Retriever` 负责召回，后处理阶段负责选择，不让具体 retriever 感知 rerank。
   - 保证关闭 rerank 时，现有 vector、BM25、hybrid 的行为可复现。
2. 设计并接入 `Reranker` 抽象：
   - 使用 Protocol 定义批量重排序输入、输出和错误边界。
   - 实现一个确定性的本地 baseline reranker，作为不依赖模型下载的可测试实现。
   - 为 cross-encoder 或远程 reranker 预留 client adapter 与 registry 接入点，不在业务 pipeline 中直接导入第三方 SDK。
   - 配置 rerank 的启用状态、策略名、候选上限、batch size、失败策略与最终 top-k。
3. 将 context packing 升级为 token-aware 流程：
   - 将 BM25 tokenizer 与模型 token 估算职责分离。
   - 引入可替换 `TokenEstimator`，计算提示词开销、问题、引用前缀、上下文和预留输出所占预算。
   - 保留去重、相邻 chunk 合并，同时增加每篇文档上限和 dropped reason。
   - 为合并或截断后的上下文段保存完整 source chunk 列表，避免来源信息只剩第一个 chunk。
4. 设计安全的压缩边界：
   - 仅允许不改变事实归属的 extractive 压缩进入本子模块实现。
   - 明确 generative compression 必须在后续回答生成与 citation 校验能力具备后再接入。
5. 扩展 retrieval 领域报告：
   - 记录候选数、去重后数量、rerank 前后顺序、最终选择数、token 使用量和丢弃原因。
   - 记录 reranker 的耗时、策略名、降级状态和子阶段 trace。
6. 使用确定性 fixture 验证关键工作流：
   - rerank 开关不会改变配置外的行为。
   - rerank 失败按配置执行 fail-open 或 fail-closed。
   - token 预算不会超限，且所有上下文段仍能追溯到原始 chunk。

验收标准：

- `hybrid` 继续作为普通 `Retriever` 在 registry 中工作，未被 rerank 逻辑侵入。
- rerank 可以通过 `settings.toml` 开关、策略名和 Config 配置，且关闭时保留当前检索排序。
- `candidate_limit`、rerank 输入数和最终 `top_k` 的职责明确，报告可以看出每一步数量变化。
- reranker 的失败路径有明确错误码、trace 和可配置降级策略，不会静默吞掉排序失败。
- context packing 使用 token 预算而不是字符数近似值，并为每个使用或丢弃的 chunk 保留可解释原因。
- 相邻 chunk 合并、截断或 extractive 压缩后仍保留完整来源 chunk 列表和位置范围。
- 已有 retrieval report、compare search、`/search` 和 `/ask` 主调用链继续通过回归测试。
- 能解释 rerank 提升的对象、额外延迟与成本，以及为何正式指标评测仍留到子模块 8。

工程注意事项：

- Rerank 不是新的召回器；它只能在已有候选集合内重新排序，召回阶段遗漏的证据无法被它找回。
- 不要把不同模型或不同 reranker 的原始 score 当作同一量纲；对外稳定展示 rank，原始分数仅用于本次运行诊断。
- ContextPacker 不应擅自调用 LLM 或修改事实含义；其职责是选择、组织和追溯证据。
- token 预算必须预留系统提示词、问题、citation 格式和输出空间，不能只计算 chunk 正文。
- 不在本子模块建立 golden dataset、Recall@k/MRR 运行器或配置实验矩阵；这些属于子模块 8，避免过早把评测基础设施和检索实现混在一起。

---

### 子模块 7：Query Rewrite、Answer Generation 与 Citation

学习目标：

- 让用户自然语言问题被改写成更适合检索的 query。
- 生成回答时严格基于检索上下文，并给出引用来源。
- 明确 RAG 系统对“无法回答”的处理方式。

需要掌握：

- query rewrite。
- multi-query retrieval。
- HyDE 的基本思想。
- grounded answer。
- citation format。
- hallucination guard。
- answer abstention。
- 中文回答与英文论文引用的转换。

实践任务：

1. 实现 `QueryRewriter`：
   - 原始 query。
   - 改写 query。
   - 可选英文关键词 query。
2. 设计回答生成 prompt：
   - 只能基于给定 context 回答。
   - 引用必须使用 `[C1]`、`[C2]` 这样的 citation id。
   - 信息不足时明确说明不能确定。
3. 实现 `AnswerGenerator`：
   - 输入问题和 packed context。
   - 输出 `RagAnswer`。
4. 实现引用校验：
   - 回答中出现的 citation id 必须来自 retrieved chunks。
   - 每个 citation id 能映射到文档标题、页码、章节。
5. 支持中文问答：
   - 用户用中文提问。
   - 系统可以检索英文论文。
   - 回答使用中文，但引用保留英文论文信息。

验收标准：

- `/ask` 返回 answer、citations、retrieved_chunks、trace_id。
- 引用来源可追溯到论文、章节和页码。
- 当检索不到足够信息时，系统不会编造答案。
- 至少有 10 个回答生成测试，包括无上下文、无效 citation、引用缺失、中文问题、比较型问题。
- 能解释 query rewrite 在什么情况下有用，什么情况下可能引入错误。

工程注意事项：

- 生成阶段不应该隐藏检索结果，否则无法调试。
- 引用不是装饰，引用必须和实际 chunk 对齐。
- 如果上下文中有冲突信息，回答应该指出冲突，而不是强行合并成一个结论。

---

### 子模块 8：RAG Evaluation 与实验管理

学习目标：

- 从“主观感觉效果不错”升级到“用指标和样例证明效果变化”。
- 构建人工评测集，并用脚本比较不同 RAG 配置。
- 学会通过失败样例诊断系统问题。

需要掌握：

- golden dataset。
- hit rate。
- Recall@k。
- MRR。
- context precision。
- context recall。
- answer relevance。
- faithfulness。
- groundedness。
- Ragas 或 LlamaIndex eval 的基本用法。
- 实验配置和结果记录。

评测集建议格式：

```jsonl
{"id":"q001","question":"RAG 系统中 rerank 的作用是什么？","type":"fact","expected_doc_ids":["paper_001"],"expected_keywords":["rerank","retrieval"],"answer_notes":"应说明 rerank 用于重新排序候选 chunk"}
{"id":"q002","question":"论文 A 和论文 B 对 agent memory 的观点有什么不同？","type":"compare","expected_doc_ids":["paper_002","paper_003"],"expected_keywords":["memory","agent"],"answer_notes":"应比较两个来源"}
```

实践任务：

1. 构造 80 到 120 条测试问题，分为：
   - 事实型
   - 比较型
   - 综述型
   - 引用定位型
2. 编写 `evaluation/runner.py`：
   - 读取评测集。
   - 运行指定 RAG 配置。
   - 保存检索结果、回答、引用、耗时。
3. 实现基础指标：
   - HitRate@k
   - Recall@k
   - MRR
   - citation coverage
   - answer has citation ratio
4. 增加人工评审字段：
   - answer relevance
   - faithfulness
   - failure_type
5. 对比至少 3 组配置：
   - chunk size：300、600、1000
   - top-k：3、5、8
   - rerank：on、off
6. 编写 `EVALUATION.md`：
   - 数据集说明
   - 实验配置
   - 指标表格
   - 失败案例
   - 优化结论

验收标准：

- 至少有 80 条评测问题。
- 至少完成 3 组配置对比。
- `EVALUATION.md` 中有指标表格和失败案例。
- 能明确说明某个优化提升了哪个指标，同时牺牲了什么。
- 能把失败回答归类到具体阶段，而不是只说“模型没答好”。

工程注意事项：

- 评测集要覆盖真实使用场景，不要只写容易命中的关键词问题。
- 指标不能只看回答文本，必须单独评估检索质量。
- 任何优化都要记录配置，否则无法复现实验。

---

### 子模块 9：API 服务化、Streaming 与可观测性

学习目标：

- 将 RAG pipeline 包装成真实可用的服务接口。
- 建立结构化日志和 trace，支持排查线上问题。
- 在回答生成阶段支持 streaming，同时保留最终引用和检索结果。

需要掌握：

- FastAPI 服务结构。
- dependency injection。
- application state。
- SSE streaming。
- request trace。
- structured logging。
- API schema。
- 错误处理。

实践任务：

1. 实现 API：
   - `GET /health`
   - `POST /documents/ingest`
   - `GET /documents`
   - `POST /search`
   - `POST /ask`
   - `POST /ask/stream`
   - `POST /eval/run`
2. 为 RAG pipeline 注入依赖：
   - settings
   - embedding client
   - vector store
   - retriever
   - reranker
   - answer generator
3. 记录结构化日志：
   - request_started
   - query_rewritten
   - retrieval_finished
   - rerank_finished
   - context_packed
   - answer_generated
   - request_finished
   - request_failed
4. Streaming 输出建议事件：
   - `status`
   - `token`
   - `citation`
   - `final`
   - `error`
5. 为 API 编写测试：
   - 健康检查
   - search 正常返回
   - ask 正常返回引用
   - stream 正常输出 final
   - 检索失败返回清晰错误

验收标准：

- API 可以在 mock LLM 和真实 LLM 两种模式下运行。
- `/ask` 和 `/search` 都有明确响应 schema。
- `/ask/stream` 最终事件包含完整 answer、citations、trace_id。
- 错误响应包含 code、message、trace_id。
- 日志能复盘一次完整 RAG 请求。

工程注意事项：

- Streaming 阶段不要只输出 token，最后必须输出结构化 final 事件。
- API 层不要直接拼 prompt，应调用 generation 层。
- 依赖对象应在应用启动时初始化，避免每个请求重复加载索引。

---

### 子模块 10：安全边界、生产化与作品集整理

学习目标：

- 理解 RAG 系统在真实场景中的安全、隐私、成本和稳定性问题。
- 让项目从“课程练习”变成“可以展示给面试官和协作者看的工程作品”。
- 明确 RAG 的能力边界。

需要掌握：

- prompt injection in retrieved documents。
- 数据泄露与访问控制。
- 文档权限过滤。
- rate limit。
- embedding 成本控制。
- index rebuild。
- cache。
- Docker。
- CI。
- README 和实验报告写法。

实践任务：

1. 增加基础安全规则：
   - 不允许读取项目外路径。
   - 不在日志里输出 API key。
   - 不把用户隐私数据写进评测报告。
2. 增加 prompt injection 防护 prompt：
   - 检索文本只能作为资料。
   - 文档中的指令不能覆盖 system prompt。
3. 增加文档权限字段：
   - `visibility`
   - `owner`
   - `tags`
4. 增加 Dockerfile 和 docker-compose 配置。
5. 增加 CI 测试建议。
6. 完成 README：
   - 项目简介
   - 架构图
   - 安装方式
   - 配置说明
   - 数据导入
   - API 使用
   - 评测方法
   - 已知限制
7. 完成 `EVALUATION.md`：
   - 评测集
   - 指标
   - 实验
   - 失败案例
   - 结论

验收标准：

- README 能让别人复现最小运行流程。
- `EVALUATION.md` 能说明系统如何被评测和优化。
- 项目支持 mock 模式，不依赖真实 API 也能运行测试。
- 至少有 30 个自动化测试，覆盖解析、chunking、检索、生成、API、评测。
- 能清楚说明 RAG 不能解决的边界，例如知识库没有相关内容、文档解析错误、上下文冲突、模型过度推断、引用不完整。

工程注意事项：

- 不要把安全边界完全交给 LLM 判断。
- 文档权限过滤应发生在检索阶段或检索前，而不是生成回答之后。
- README 和 EVALUATION 是作品集的一部分，不是附属品。

---

## 4. 推荐实现顺序

本模块不按时间安排推进，而按工程依赖关系推进：

1. 先搭项目骨架和核心数据模型。
2. 再做文档解析和 chunking。
3. 再做 embedding、索引和 vector retrieval。
4. 再补 BM25、hybrid retrieval、rerank。
5. 再做 answer generation 和 citation。
6. 再做 evaluation。
7. 最后服务化、加日志、补安全边界和 README。

每完成一个子模块，都应该保证：

- 代码可以运行。
- 有测试覆盖关键路径。
- 有文档记录当前设计。
- 当前模块的输出能被下一个模块直接使用。

---

## 5. 最小可行版本要求

如果先做 MVP，最低要求如下：

1. 支持 Markdown 或 PDF 中至少一种文档。
2. 支持固定长度 chunking。
3. 支持 mock embedding 或本地 embedding。
4. 支持一个本地 vector store。
5. 支持 `/search` 返回 top-k chunk。
6. 支持 `/ask` 返回中文回答和引用来源。
7. 有 20 条评测问题。
8. 能比较 top-k=3、5、8 的检索效果。
9. 有 README 和基础测试。

MVP 不应该包含：

- 多 agent。
- 复杂 UI。
- 长期记忆。
- 自动论文下载。
- 未评测的高级 rerank。
- 和主项目无关的花哨功能。

---

## 6. 完整版本验收清单

### 数据与解析

- [ ] 至少导入 30 篇 AI Agent、RAG 或 LLM Security 论文。
- [ ] 每篇论文有稳定 `doc_id`。
- [ ] 每个 chunk 有 `chunk_id`、`doc_id`、页码、章节、来源路径。
- [ ] 解析失败会被记录，不会让整个索引流程崩溃。

### 检索与生成

- [ ] 支持 vector retrieval。
- [ ] 支持 BM25 retrieval。
- [ ] 支持 hybrid retrieval。
- [ ] 支持 rerank 开关。
- [ ] 支持 query rewrite 开关。
- [ ] 回答必须包含引用。
- [ ] 引用能映射回具体 chunk。
- [ ] 信息不足时系统会拒绝编造。

### 评测与实验

- [ ] 至少有 80 条评测问题。
- [ ] 问题类型包括事实型、比较型、综述型、引用定位型。
- [ ] 至少比较 3 组配置。
- [ ] `EVALUATION.md` 有指标表格。
- [ ] `EVALUATION.md` 有失败案例分析。
- [ ] 能说明某个优化是否真的提升了指标。

### 工程质量

- [ ] 有 `.env.example`。
- [ ] 有 README。
- [ ] 有结构化日志。
- [ ] 有 API 测试。
- [ ] 有检索测试。
- [ ] 有评测脚本测试。
- [ ] 支持 mock 模式。
- [ ] 不把 API key 写死在代码或文档中。
- [ ] 项目可以被别人复现启动。

---

## 7. 关键实验建议

### 实验 1：chunk size 对检索效果的影响

比较配置：

- chunk size 300，overlap 50
- chunk size 600，overlap 100
- chunk size 1000，overlap 150

观察指标：

- HitRate@5
- Recall@5
- MRR
- 平均检索耗时
- answer has citation ratio

需要回答的问题：

1. 哪个 chunk size 对事实型问题最好？
2. 哪个 chunk size 对综述型问题最好？
3. 是否存在“检索命中但回答变差”的情况？

### 实验 2：top-k 对上下文质量的影响

比较配置：

- top-k=3
- top-k=5
- top-k=8

观察指标：

- context precision
- context recall
- 回答长度
- 引用数量
- 无关 chunk 比例

需要回答的问题：

1. top-k 增大后，召回是否真的提高？
2. top-k 增大后，是否引入更多噪声？
3. 生成模型是否会被无关 chunk 干扰？

### 实验 3：是否启用 rerank

比较配置：

- vector only
- vector + rerank
- hybrid + rerank

观察指标：

- HitRate@5
- MRR
- context precision
- 平均延迟
- 失败案例数量

需要回答的问题：

1. rerank 是否改善了排序？
2. rerank 是否值得额外延迟和成本？
3. rerank 对哪类问题最有效？

---

## 8. 常见失败类型与定位方法

### 解析失败

表现：

- 引用页码错误。
- 章节标题缺失。
- chunk 中出现大量乱码。
- 表格和公式被破坏。

定位方式：

- 查看 parsed 文本。
- 查看 chunk 原文。
- 对比原始 PDF 页码。

### 切分失败

表现：

- 关键定义被切到两个 chunk 中。
- chunk 过短，缺少上下文。
- chunk 过长，检索结果不精确。

定位方式：

- 查看 chunk 统计报告。
- 查看命中 chunk 前后相邻内容。
- 比较不同 chunk size 的结果。

### 检索失败

表现：

- top-k 中没有相关 chunk。
- 关键词问题 BM25 能找到，向量检索找不到。
- 语义问题向量检索能找到，BM25 找不到。

定位方式：

- 查看 query rewrite。
- 查看 vector top-k 和 BM25 top-k。
- 查看检索分数和 metadata。

### 上下文组织失败

表现：

- 检索到了相关 chunk，但没有进入最终 prompt。
- 上下文里重复内容太多。
- 引用 id 和 chunk 对不上。

定位方式：

- 查看 context packing 结果。
- 查看 token budget。
- 查看 citation map。

### 生成失败

表现：

- 上下文中没有的信息被模型编造。
- 回答没有引用。
- 引用与句子不匹配。
- 对冲突来源没有说明。

定位方式：

- 查看最终 prompt。
- 查看模型输出。
- 检查 citation validator。
- 用同一 context 重新生成并对比。

---

## 9. 与模块 1 的衔接

模块 1 的 `mini-tool-agent` 提供了以下可迁移经验：

- FastAPI 服务结构。
- `.env` 配置管理。
- mock 与真实 LLM 双模式。
- structured output。
- dependency injection。
- application state。
- streaming。
- structured logging。
- pytest 测试。
- README 和工程说明文档。

模块 2 不建议直接依赖模块 1 的练习代码，但可以复用其工程思想：

- 把 RAG 检索封装成工具或服务，而不是散落在 API 中。
- 用 mock embedding 和 mock LLM 保证测试稳定。
- 用结构化日志记录每个阶段。
- 用清晰错误类型区分解析错误、索引错误、检索错误、生成错误。
- 用配置项控制策略，而不是在代码里写死 top-k、chunk size、rerank 开关。

后续进入模块 3 时，`paper-rag-assistant` 应该可以作为一个 RAG 工具或 RAG 节点接入 Agent Workflow。

---

## 10. 模块完成后的自检问题

完成模块 2 后，你应该能回答：

1. RAG 的 loading、indexing、storing、querying、evaluation 分别解决什么问题？
2. 为什么“把文档放进向量库”不是完整 RAG？
3. chunk size 和 overlap 如何影响检索质量？
4. metadata 在 citation、过滤、调试和评测中分别有什么作用？
5. BM25 和向量检索各自适合什么场景？
6. hybrid retrieval 如何融合不同检索器的结果？
7. rerank 为什么通常发生在初次召回之后？
8. query rewrite 什么时候有帮助，什么时候会伤害结果？
9. context packing 为什么会影响最终回答质量？
10. 如何判断一次错误回答是检索问题还是生成问题？
11. faithfulness 和 answer relevance 有什么区别？
12. 为什么 RAG evaluation 必须同时评估检索和生成？
13. 如何证明某个优化真的提升了系统，而不是只改善了几个样例？
14. RAG 无法解决哪些问题？
15. 如果要把 RAG 接入 Agent，应该作为工具、节点，还是独立服务？

---

## 11. 模块 2 最终交付标准

当满足以下条件时，可以认为模块 2 学习完成：

1. `paper-rag-assistant` 可以导入论文并建立索引。
2. 用户可以用中文提问，并获得带引用的回答。
3. `/search` 可以展示检索 chunk、分数、metadata。
4. `/ask` 可以返回 answer、citations、retrieved_chunks、trace_id。
5. 至少有 80 条人工评测问题。
6. 至少完成 3 组检索或生成配置对比。
7. `EVALUATION.md` 能展示指标、失败案例和优化结论。
8. README 能让其他人复现项目运行。
9. 测试覆盖核心 pipeline。
10. 你能独立讲清一次 RAG 请求从用户问题到最终回答的完整路径。
