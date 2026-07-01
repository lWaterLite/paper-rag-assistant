# 子模块 3：Chunking 策略与 Metadata 设计概念教学

对应学习路线：模块 2《RAG 知识库与检索增强生成》子模块 3  
核心项目：`paper-rag-assistant`  
学习定位：把已经解析好的文档切成可检索、可引用、可评测的知识片段

---

## 1. 子模块 3 要解决什么问题

子模块 2 已经完成了真实文档 ingestion：

```text
PDF / Markdown / HTML / TXT
  -> RawDocument
  -> ParsedDocument
  -> ParsedBlock
  -> IngestionReport
```

这一步让真实文档稳定进入系统。但 RAG 检索通常不是直接检索整篇论文，也不是直接把整篇论文交给 LLM。

原因很简单：

- 一篇论文太长，无法直接放进模型上下文。
- 整篇文档 embedding 太粗，用户问一个具体概念时很难精确命中。
- 检索结果必须能引用到具体页码、章节或片段。
- 后续评测需要知道“到底哪个片段被召回了”。

所以子模块 3 的核心问题是：

> 如何把解析后的文档切成大小合适、语义尽量完整、metadata 足够丰富、结果可复现的 chunks。

Chunking 看起来只是“切文本”，但它实际上决定了 RAG 系统的检索颗粒度。切得不好，后面的 embedding、向量库、BM25、rerank、prompt 都会被拖累。

---

## 2. 什么是 Chunk

Chunk 是 RAG 检索的基本单位。

在当前项目中，chunk 对应核心模型：

```python
DocumentChunk
```

它至少包含：

```text
chunk_id
doc_id
content_hash
version_id
text
source_path
chunk_index
token_count
title
section
page_start
page_end
metadata
```

可以把文档层级理解成：

```text
RawDocument
  原始文件读取结果

ParsedDocument
  解析和清洗后的整篇文档

ParsedBlock
  解析阶段识别出的段落、标题、表格、代码块、页码信息等结构块

DocumentChunk
  检索阶段真正进入 embedding、BM25、向量库和 citation 的片段
```

Chunk 不是随便截出来的一段字符串。它应该是一个带身份、来源、位置、配置痕迹的工程对象。

---

## 3. 为什么不能直接用整篇文档

很多初学 RAG 的实现会这样做：

```text
一篇文档 -> 一个 embedding -> 存入向量库
```

这对非常短的 FAQ 也许能用，但对论文 RAG 基本不够。

### 3.1 召回太粗

用户问：

```text
RAGAS 里 faithfulness 是怎么计算的？
```

如果一整篇 RAGAS 论文只有一个向量，那么检索器只能判断“这篇论文整体是否相关”，无法定位具体方法段落。

结果可能是：

- 检索能找到论文。
- 但上下文太长，放不进 prompt。
- 或者放进去后噪声太多，模型找不到关键段落。

### 3.2 引用不够精确

如果检索单位是整篇论文，最终引用只能是：

```text
来自 ragas.pdf
```

这不够。真实 RAG 系统更希望引用：

```text
来自 ragas.pdf，第 3 页，Evaluation Metrics 小节
```

这要求 chunk 保留页码、章节、源路径等 metadata。

### 3.3 评测不可诊断

如果回答错了，你需要判断：

- 是解析没有拿到正确文本？
- 是 chunking 把定义切断了？
- 是检索没有召回正确 chunk？
- 是上下文组织时丢掉了正确 chunk？
- 是生成阶段没有忠实使用上下文？

没有 chunk 级别的结构，就很难定位失败阶段。

---

## 4. Chunk Size 是什么

Chunk size 表示每个 chunk 的目标长度。

它可以用不同单位衡量：

- 字符数。
- 单词数。
- token 数。
- 句子数。
- 段落数。
- block 数。

在真实 LLM/RAG 工程中，最推荐使用 token 数，因为模型上下文窗口和 embedding 模型输入限制通常都按 token 计费和限制。

但在早期练习中，字符数切分更容易理解，也更容易测试。当前项目的 `CharacterChunker` 就是字符级 chunker。

### 4.1 小 chunk

小 chunk 的优点：

- 检索更精确。
- 每个 chunk 噪声少。
- 更容易定位引用。
- 适合事实型问题。

小 chunk 的缺点：

- 容易丢上下文。
- 一个定义可能被切成两半。
- embedding 只能看到局部文字，语义表达不足。
- 需要更多向量，索引变大。

例如：

```text
Reranking improves retrieval quality by reordering candidate passages
according to their relevance to the query.
```

如果切成：

```text
Reranking improves retrieval quality by
```

和：

```text
reordering candidate passages according to...
```

那么两个 chunk 都不完整。

### 4.2 大 chunk

大 chunk 的优点：

- 上下文更完整。
- 适合综述型、比较型问题。
- 不容易把定义或论证链切断。

大 chunk 的缺点：

- 检索不够精确。
- chunk 内部可能混入多个主题。
- top-k 中每个结果占用更多上下文。
- 进入 LLM 的噪声更多。

例如一个 chunk 同时包含：

```text
Introduction
Related Work
Evaluation
Limitations
```

用户只问 evaluation，但 chunk 中混有大量不相关内容，生成阶段更容易被干扰。

### 4.3 没有绝对最佳 chunk size

Chunk size 不是越大越好，也不是越小越好。

它取决于：

- 文档类型。
- 用户问题类型。
- embedding 模型能力。
- 检索策略。
- 是否有 rerank。
- LLM 上下文窗口。
- 是否需要精确 citation。

论文 RAG 中常见的实验范围是：

```text
small:  约 300 tokens
medium: 约 600 tokens
large:  约 1000 tokens
```

子模块 3 的实践目标不是猜一个“神奇参数”，而是建立可以比较不同配置的工程能力。

---

## 5. Chunk Overlap 是什么

Overlap 表示相邻 chunk 之间重叠的内容长度。

例如 chunk size = 100，overlap = 20：

```text
chunk 1: 0   -> 100
chunk 2: 80  -> 180
chunk 3: 160 -> 260
```

### 5.1 overlap 的作用

Overlap 主要用来缓解边界断裂。

如果一句关键定义刚好跨过两个 chunk 的边界，没有 overlap 时两个 chunk 都可能语义不完整。有 overlap 后，至少有一个 chunk 可能包含完整上下文。

例如：

```text
... Retrieval-Augmented Generation combines parametric knowledge
with non-parametric memory retrieved from an external corpus ...
```

如果切分边界落在 `combines parametric knowledge` 后面，没有 overlap 时定义会断裂。Overlap 可以让后一段重新看到前面的关键短语。

### 5.2 overlap 的副作用

Overlap 不是免费午餐。

副作用包括：

- chunk 数量增加。
- embedding 成本增加。
- 向量库体积增加。
- 检索结果更容易重复。
- context packing 时可能浪费上下文窗口。

如果 overlap 太大，很多 chunk 会高度相似。检索 top-k 可能返回同一段附近的多个重复 chunk，导致上下文里信息密度下降。

### 5.3 overlap 应该可配置

Overlap 不应该写死在代码里。

它应该属于 chunker 配置，例如：

```toml
[chunking]
strategy = "section_aware"
chunk_size = 600
chunk_overlap = 100
```

配置进入 `Settings`，再由 factory 转换成 chunker 实际使用的 `Config`，这和我们前面 loader、PDF cleaner、ingestion report 的做法一致。

---

## 6. 语义边界为什么重要

固定长度切分最大的问题是：它不理解文档结构。

它可能切断：

- 一个句子。
- 一个公式解释。
- 一个列表。
- 一个表格。
- 一个 Markdown 小节。
- 一段实验结论。

但真实文档往往天然有语义边界：

- 标题。
- 段落。
- 列表。
- 表格。
- 代码块。
- PDF 页。
- 章节。

好的 chunker 应该尽量尊重这些边界。

### 6.1 段落边界

段落通常是最基础的语义单位。

如果一个 chunk 由若干完整段落组成，通常比硬切字符更自然。

风险是：有些 PDF 解析后的段落可能并不可靠，尤其是双栏论文和表格区域。

### 6.2 标题和章节边界

论文和技术文档中，章节信息很重要。

例如：

```text
3. Retrieval
4. Reranking
5. Evaluation
```

如果 chunk 能保留 `section="Evaluation"`，后续检索和引用会更清楚。

对于问题：

```text
这篇论文怎么评估 RAG 的 faithfulness？
```

带有 `Evaluation` section 的 chunk 往往比来自 `Introduction` 的 chunk 更有价值。

### 6.3 页码边界

PDF 中页码是 citation 的关键。

如果一个 chunk 跨了多页，需要记录：

```text
page_start = 3
page_end = 4
```

如果 chunk 来自单页：

```text
page_start = 3
page_end = 3
```

不要在 chunking 阶段丢掉解析阶段保留下来的页码。

---

## 7. 常见 Chunking 策略

### 7.1 固定字符切分

固定字符切分按照字符长度切：

```text
每 500 字符切一个 chunk，重叠 80 字符。
```

优点：

- 实现简单。
- 行为稳定。
- 测试容易。
- 适合建立 baseline。

缺点：

- 不理解 token。
- 不理解语义边界。
- 可能切断句子、段落和章节。

当前项目已有的 `CharacterChunker` 就是这个思路。

### 7.2 固定 token 切分

固定 token 切分按 tokenizer 计算长度。

优点：

- 更贴近 embedding 和 LLM 的真实限制。
- 可以更准确控制上下文预算。

缺点：

- 需要引入 tokenizer。
- 不同模型 tokenizer 不同。
- token 统计会带来额外依赖和配置。

真实工程中常见做法是为不同 provider 配置 tokenizer，例如：

```text
OpenAI tokenizer
HuggingFace tokenizer
自定义近似 tokenizer
```

### 7.3 递归字符切分

递归切分会按优先级尝试分隔符：

```text
先按标题切
再按段落切
再按句子切
再按空格切
最后按字符硬切
```

它比固定字符切分更自然。

典型优先级可以是：

```text
"\n# "     Markdown 一级标题
"\n## "    Markdown 二级标题
"\n\n"     段落
"。" / "." 句子
" "        单词
""         字符
```

这种策略适合 Markdown、HTML 正文和清洗后的纯文本。

### 7.4 Section-aware chunking

Section-aware chunking 优先按章节切分。

流程通常是：

```text
ParsedDocument / ParsedBlock
  -> 识别 section
  -> 按 section 聚合内容
  -> 每个 section 内部再按 chunk size 切分
  -> chunk metadata 记录 section
```

优点：

- 适合论文和技术文档。
- citation 更清楚。
- 后续可以按章节过滤。
- 对比较型、综述型问题更友好。

缺点：

- 依赖解析阶段是否准确识别 section。
- PDF heading 识别差时效果会打折。
- 一个 section 过长时仍需要二次切分。

当前项目已有早期 `SectionAwareChunker`，但它主要针对 Markdown 标题。子模块 3 需要把它升级成更贴近 `ParsedBlock` 的实现。

### 7.5 Sentence window retrieval

Sentence window 不是简单切 chunk，而是一种检索和上下文扩展策略。

基本思想：

```text
索引时用较小单位，例如句子。
检索时命中某一句。
进入上下文时带上前后若干句。
```

例如命中第 10 句，则最终上下文使用：

```text
第 8 句 + 第 9 句 + 第 10 句 + 第 11 句 + 第 12 句
```

优点：

- 检索粒度细。
- 生成上下文相对完整。

缺点：

- 实现更复杂。
- 需要保存句子位置和相邻关系。
- context packing 更复杂。

### 7.6 Parent-child chunk

Parent-child chunk 也是一种常见工程策略。

基本思想：

```text
child chunk：小片段，用于精确检索。
parent chunk：大片段，用于生成上下文。
```

例如：

```text
parent = 一个章节
child = 章节里的若干段
```

检索时查 child；命中后，把对应 parent 或 parent 的局部窗口送给 LLM。

优点：

- 检索精确。
- 上下文完整。

缺点：

- 需要维护 parent-child 关系。
- 去重、合并和引用更复杂。

这个策略适合后续高级 RAG，不一定在子模块 3 立刻完整实现，但需要理解。

---

## 8. Metadata 为什么和 Chunking 一样重要

Chunk 的 `text` 只是内容。Metadata 决定它能不能被追溯、过滤、聚合、评测和引用。

一个真实可用的 chunk 至少应该回答这些问题：

```text
它来自哪篇文档？
它来自文档哪个版本？
它是第几个 chunk？
它来自哪一页？
它来自哪个章节？
它对应原文的哪个范围？
它由什么 chunker 配置生成？
```

### 8.1 支撑 citation

回答生成时，citation 需要映射回 chunk。

如果 chunk 没有：

```text
source_path
page_start
page_end
section
title
```

最终回答只能给出含糊来源。

### 8.2 支撑过滤

后续检索可能需要过滤：

```text
只检索某一年后的论文
只检索某个作者
只检索 Evaluation 章节
只检索 PDF，不检索 HTML
只检索公开文档
```

这些都依赖 metadata。

### 8.3 支撑去重和聚合

检索结果可能命中同一篇文档的相邻 chunk。

如果 metadata 中有：

```text
doc_id
chunk_index
section
page_start
page_end
```

context packing 就可以合并相邻 chunk、去除重复、保留引用顺序。

### 8.4 支撑评测

评测时你需要知道：

- 问题预期命中哪篇论文。
- top-k 是否命中正确 doc_id。
- 命中 chunk 是否来自正确章节。
- citation 是否覆盖正确页码。

没有 metadata，就只能粗略评估“答案看起来对不对”，很难定位系统问题。

---

## 9. 稳定 ID 与可复现性

Chunking 结果必须可复现。

同一份文档、同一份内容、同一套 chunking 配置，多次构建索引应该生成稳定 chunk。

当前项目的 chunk_id 思路是：

```text
chunk_id = hash(version_id + chunk_index + chunk_text)
```

这有几个好处：

- 文档内容不变时，chunk_id 稳定。
- 文档内容变化时，version_id 变化，chunk_id 随之变化。
- chunk_text 变化时，chunk_id 变化。
- embedding cache 可以根据稳定内容复用结果。

但还要注意一个问题：如果 chunking 策略或 chunk size 改了，即使文档没变，chunk 结果也可能变化。

因此真实工程中，索引 manifest 需要记录：

```text
chunker_name
chunk_size
chunk_overlap
tokenizer
normalization_version
```

否则你无法解释某个索引到底是怎么构建出来的。

---

## 10. Token Count 为什么重要

当前 `DocumentChunk.token_count` 在早期实现中可能只是字符长度近似值。

真实工程里，token_count 有三个用途：

1. 控制 embedding 输入长度。
2. 控制 LLM context packing。
3. 做 chunk 质量统计。

例如：

```text
embedding model 最大输入 8192 tokens
生成模型上下文预算 128k tokens
单次 prompt 只允许检索上下文占 6000 tokens
```

如果没有可靠 token_count，context packing 就只能靠字符数猜，很容易超限或浪费。

### 10.1 token 和字符不同

英文中，一个 token 大约接近 0.75 个单词，但不是固定比例。

中文中，一个汉字可能接近一个 token，也可能因 tokenizer 不同而变化。

代码、公式、URL、表格会让 token 数更难估计。

所以真实工程中，token_count 最好由明确 tokenizer 计算，而不是随便用 `len(text)`。

### 10.2 什么时候可以先近似

学习阶段可以先用字符数近似，但要在 metadata 或配置中清楚记录：

```text
tokenizer = "char_approx"
```

这样后续切换到真实 tokenizer 时，索引版本和评测结果不会混淆。

---

## 11. Chunking 质量检查

Chunking 完成后，不应该只看“代码能跑”。

应该生成质量报告。

常见检查包括：

```text
chunk_count
empty_chunk_count
avg_token_count
max_token_count
min_token_count
overlap_valid
missing_doc_id_count
missing_source_path_count
missing_page_count
missing_section_count
per_document_chunk_count
```

### 11.1 空 chunk

空 chunk 没有检索价值，还会污染索引。

应该确保：

```text
chunk.text.strip() != ""
```

### 11.2 过长 chunk

过长 chunk 可能导致：

- embedding 输入超限。
- 检索不精确。
- context packing 浪费空间。

质量检查应该标出超过阈值的 chunk。

### 11.3 metadata 缺失

如果 chunk 缺少 `doc_id`、`source_path`、`version_id`，它就很难被追溯。

PDF chunk 如果缺少页码，citation 质量会下降。

### 11.4 overlap 是否符合配置

对固定窗口切分来说，可以检查相邻 chunk 的字符范围是否满足 overlap。

对 section-aware chunking 来说，overlap 可能只发生在过长 section 内部，而不是跨 section 强行 overlap。

---

## 12. Chunking Report

子模块 2 已经有 ingestion report。子模块 3 可以继续建立 chunking report。

建议报告结构：

```json
{
  "trace_id": "trace_xxx",
  "chunker": "SectionAwareChunker",
  "chunk_size": 600,
  "chunk_overlap": 100,
  "tokenizer": "char_approx",
  "document_count": 8,
  "chunk_count": 530,
  "avg_token_count": 482.3,
  "max_token_count": 612,
  "empty_chunk_count": 0,
  "missing_page_count": 12,
  "documents": [
    {
      "doc_id": "doc_xxx",
      "title": "RAGAS",
      "source_path": "data/raw/papers/pdf/ragas.pdf",
      "chunk_count": 42,
      "avg_token_count": 510,
      "missing_page_count": 0
    }
  ]
}
```

这类报告的价值是：

- 快速发现切分异常。
- 比较不同 chunk size。
- 为后续 evaluation 记录实验配置。
- 帮助解释检索结果变化。

---

## 13. Chunking 与后续模块的关系

### 13.1 对 Embedding 的影响

Embedding 是把 chunk text 转成向量。

如果 chunk 太短：

```text
retrieval augmented
```

embedding 语义不完整。

如果 chunk 太长：

```text
Introduction + Related Work + Evaluation + References
```

embedding 会变成多个主题的混合平均。

好的 chunking 会让 embedding 表达更聚焦。

### 13.2 对 BM25 的影响

BM25 依赖词频和文档长度。

chunk 太长会稀释关键词分数。

chunk 太短则可能缺少必要关键词。

例如用户问：

```text
Dense Passage Retrieval 使用什么训练数据？
```

如果 chunk 中只有 `training data`，没有 `Dense Passage Retrieval`，BM25 可能无法命中。

### 13.3 对 Rerank 的影响

Rerank 通常在初次召回后重新排序。

如果初次 chunking 太差，正确内容没有进入候选集，rerank 也救不了。

如果 chunk 内容完整但有噪声，rerank 可能帮助把更相关的 chunk 排到前面。

### 13.4 对 Context Packing 的影响

Context packing 要在有限 token budget 内选择材料。

如果 chunk 太大，一个 chunk 就占满上下文。

如果 chunk 太碎，context packing 需要合并大量相邻 chunk。

因此 chunking 和 context packing 是一对联动设计。

### 13.5 对 Citation 的影响

Citation 需要引用 chunk。

chunk metadata 越完整，citation 越可信。

如果 chunk 跨页，citation 应该显示页码范围。

如果 chunk 来自 section，citation 可以显示章节。

---

## 14. 论文 RAG 中的特殊问题

### 14.1 Abstract 是否应该单独成 chunk

Abstract 很重要，通常适合作为单独或优先保留的 chunk。

用户问论文贡献、方法概览、研究目标时，Abstract 很可能被检索到。

### 14.2 References 是否应该索引

References 不应该无脑删除。

它可能有价值：

- 用户问“这篇论文引用了哪些 RAG 工作？”
- 用户问“某方法和 DPR 有什么关系？”

但 references 也容易污染普通问题检索。

工程上可以：

- 保留 references。
- metadata 标记 `section="References"`。
- 检索时按问题类型决定是否过滤。

### 14.3 Tables 和 Figures 怎么办

表格和图注对论文问答很重要。

子模块 3 阶段可以先不完整结构化表格，但应该尽量：

- 不要把表格内容静默丢掉。
- 在 metadata 中标记疑似 table/caption。
- 后续为表格问答预留扩展空间。

### 14.4 PDF 页码和逻辑页码

PDF 的物理页码不一定等于论文显示页码。

例如 arXiv PDF 的第一页可能显示页码为空，正文页码从 1 开始。

当前阶段先记录物理页码即可。后续如果需要严格 citation，可以增加 logical page number。

---

## 15. 当前项目中已有的基础

当前 `paper-rag-assistant` 已经具备：

- `ParsedDocument`
- `ParsedBlock`
- `DocumentChunk`
- `CharacterChunker`
- `SectionAwareChunker`
- 稳定 `chunk_id`
- `doc_id`、`content_hash`、`version_id`
- 基础 `chunk_size`、`chunk_overlap` 配置
- PDF block 的 `page_start`、`page_end`
- Markdown section 信息

但子模块 3 还需要继续升级：

- 把 chunking 配置从 `.env` 或旧 `EnvSettings` 中进一步结构化。
- 支持至少两种 chunking 策略可配置切换。
- 更充分利用 `ParsedBlock`，而不是只面对 `ParsedDocument.text`。
- 保存 chunking 质量报告。
- 让 chunk 结果为后续 embedding、BM25、retrieval evaluation 做好准备。

---

## 16. 子模块 3 的推荐工程设计

建议把 chunking 子系统拆成几个部分：

```text
app/ingest/chunkers.py
  Chunker
  CharacterChunker
  FixedTokenChunker
  SectionAwareChunker

app/ingest/chunking_report.py
  ChunkingReportWriter
  ChunkingQualityReport

app/core/settings.py
  ChunkingSettings

settings.toml
  [chunking]
  strategy = "section_aware"
  chunk_size = 600
  chunk_overlap = 100
  tokenizer = "char_approx"

app/factory.py
  build_chunker_config()
  build_chunker()
```

仍然保持我们前面建立的命名规则：

- 从外部配置文件直接生成的类叫 `Settings`。
- 转换后给功能类实际使用的类叫 `Config`。

例如：

```text
ChunkingSettings
  来自 settings.toml

ChunkerConfig
  给 CharacterChunker / SectionAwareChunker 使用
```

---

## 17. 子模块 3 的验收理解

学习路线中的验收标准可以拆成下面几类。

### 17.1 至少两种 chunking 策略可切换

你应该能通过配置选择：

```text
character
section_aware
```

后续可以扩展：

```text
fixed_token
recursive
sentence_window
parent_child
```

### 17.2 chunk 结果可持久化

当前项目有内存 repository 和 vector store。子模块 3 可以先生成 chunking report，后续再做真正 chunk 持久化。

重要的是：chunking 结果不能只是一次运行中的临时变量。

### 17.3 每个 chunk 可追溯

每个 chunk 都应该尽量包含：

```text
doc_id
version_id
source_path
chunk_index
page_start
page_end
section
char_start / char_end
```

PDF 重点是页码。

Markdown 重点是 section。

HTML 重点是 title、canonical_url、section 或 DOM 来源。

### 17.4 至少 8 个 chunker 测试

测试应该覆盖：

- 空文档。
- 短文档。
- 超长文档。
- overlap。
- section 保留。
- page 信息保留。
- chunk_id 稳定性。
- metadata 不丢失。
- 配置非法值。

测试代码由我来补，你不需要把学习重点放在写测试细节上。但你需要理解每个测试在保护什么行为。

### 17.5 能说明 chunk 大小的权衡

你需要能清楚说出：

- 小 chunk：精确但缺上下文。
- 大 chunk：完整但噪声多。
- overlap：缓解边界断裂，但增加成本和重复。
- section-aware：更符合论文结构，但依赖解析质量。

---

## 18. 常见错误

### 18.1 只保存 text，不保存 metadata

这是最常见错误。

如果只保存：

```json
{"text": "..."}
```

后续无法引用、无法过滤、无法调试。

### 18.2 chunk_id 不稳定

如果 chunk_id 每次运行都随机生成，embedding cache 和增量索引都会失效。

应该让 chunk_id 依赖稳定输入。

### 18.3 overlap 跨章节硬重叠

如果从 `Introduction` 的末尾 overlap 到 `Methods` 的开头，有时会制造奇怪上下文。

Section-aware chunker 可以选择只在同一 section 内 overlap。

### 18.4 chunking 阶段丢掉页码

解析阶段辛苦保留了 `page_start`，chunking 阶段如果没有继承，就会破坏 citation。

### 18.5 过早追求复杂策略

不要一上来就做 parent-child、sentence window、semantic chunking、LLM chunking 全家桶。

正确顺序是：

```text
先建立 baseline
再加 section-aware
再做质量报告
再用评测证明改进
```

---

## 19. 子模块 3 学完后应该能回答的问题

1. Chunk 和 ParsedBlock 有什么区别？
2. 为什么 RAG 不能直接检索整篇论文？
3. chunk size 太小会有什么问题？
4. chunk size 太大会有什么问题？
5. overlap 解决什么问题，又带来什么成本？
6. 为什么 token-based chunking 比 character-based chunking 更接近真实工程？
7. section-aware chunking 依赖解析阶段的哪些信息？
8. parent-child chunk 和 sentence window retrieval 分别解决什么问题？
9. 一个合格的 chunk metadata 应该包含哪些字段？
10. 为什么 chunk_id 必须稳定？
11. chunking report 应该记录哪些指标？
12. chunking 如何影响 embedding、BM25、rerank 和 citation？
13. 为什么 references 不应该无脑删除？
14. 为什么不同 chunk 配置需要对应不同 index version？
15. 如果检索结果命中错误，你如何判断是不是 chunking 的问题？

---

## 20. 下一步练习方向

进入练习时，我们会基于当前代码继续工程化升级，而不是重写 demo：

1. 将 chunking 配置迁移到 `settings.toml`。
2. 建立 `ChunkingSettings -> ChunkerConfig` 转换。
3. 改造 factory，让 chunker 由配置统一组装。
4. 实现更真实的 `SectionAwareChunker`，优先使用 `ParsedBlock` 的 section/page 信息。
5. 增加 chunking quality report。
6. 对同一批真实论文生成不同 chunk 配置的统计报告。

本子模块完成后，后续子模块 4 的 embedding 和索引就会有更可靠的输入。

