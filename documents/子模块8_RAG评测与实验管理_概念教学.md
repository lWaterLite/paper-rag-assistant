# 子模块 8：RAG 评测与实验管理概念教学

## 1. 学习定位

前七个子模块已经完成了一个可运行的论文 RAG 主链：真实文档被加载、解析、切分、索引；在线请求经过 query planning、检索、重排序、证据变换、context packing 与回答生成，最终返回带来源的 `RagAnswer`。

```text
论文资料
  -> ingest / chunking / indexing
  -> BM25 / vector / hybrid retrieval
  -> rerank / evidence transformation / context packing
  -> query planning / grounded answer / citation validation
  -> 带 trace、citation、retrieved_chunks 的回答
```

到这里，系统已经能回答问题，但还不能可靠地回答下面这些更重要的问题：

1. 将 chunk size 从 500 改为 800，检索效果真的更好吗？
2. 开启 hybrid 或 rerank 后，提升的是召回、排序、上下文质量，还是只是个别样例看起来更好？
3. 一条错误答案的根因是文档解析、chunking、检索、packing、生成，还是 citation 规则？
4. 某次实验的结果，能否在同一索引版本、同一配置与同一数据集上复现？
5. 当系统拒答时，它是在正确避免编造，还是本应能找到资料却遗漏了证据？

**子模块 8 的目标，是把“我感觉这次回答不错”变成“我能用数据、样例和可复现实验说明系统在哪些维度变好了，又付出了什么代价”。**

它不会提前实现 Web 服务、Streaming、鉴权、限流或部署；这些仍属于后续子模块。它也不把评测逻辑塞进在线 `RagPipeline`。评测应作为独立的离线应用能力，复用主系统的稳定服务边界，并产生可审计的实验产物。

---

## 2. 为什么 RAG 不能只评估最终答案

RAG 是一个多阶段系统。用户最终看到的是回答，但一条回答的好坏可能由完全不同的阶段决定。

```text
用户问题
  -> QueryPlan
  -> RetrievedChunk[]
  -> PackedContext
  -> LLM 输出
  -> CitationValidator
  -> RagAnswer
```

例如，用户问“为什么 rerank 通常放在初次召回之后？”。

- 若相关论文没有被召回，问题在 retrieval，生成模型即使能力很强也无从回答。
- 若相关 chunk 被召回但在 token 预算下被丢弃，问题在 context packing。
- 若上下文已经包含完整说明，但答案遗漏关键对比，问题在 generation 或 prompt。
- 若回答内容正确但引用了错误的来源，问题在 citation correctness。
- 若系统没有资料时明确拒答，这可能是正确行为，不能简单计作“回答失败”。

因此，真实工程中的 RAG evaluation 至少分为三层：

```text
Retrieval evaluation
  评估“相关证据是否被找回、是否排在前面”。

Context evaluation
  评估“被找回的证据是否真正进入模型上下文，且上下文是否干净、充分”。

Answer evaluation
  评估“回答是否相关、受证据支持、正确引用，并在资料不足时正确拒答”。
```

只看最终答案会掩盖检索问题；只看 Recall@k 又会忽略模型是否依据资料作答。子模块 8 的核心习惯是：**每个指标都必须说明它衡量哪个阶段、依赖什么标注、不能说明什么。**

---

## 3. 评测的基本单位：Evaluation Case

### 3.1 什么是评测样例

一个 `EvaluationCase` 不是只有一个问题字符串。它是一份关于“什么算成功”的可审计契约，通常包含：

```text
问题
  + 问题类型
  + 期望证据
  + 可接受答案范围或人工标注说明
  + 对拒答的预期
  + 可选难度、标签和来源版本
```

对于当前论文知识库，最小 JSONL 记录可表达为：

```json
{
  "id": "q_rerank_001",
  "question": "RAG 系统中 rerank 的作用是什么？",
  "question_type": "fact",
  "expected_doc_ids": ["paper_rag_survey"],
  "expected_chunk_ids": ["chunk_rag_survey_014"],
  "expected_keywords": ["rerank", "reorder", "candidate"],
  "answer_notes": "应说明 rerank 对初次召回候选重新排序，通常用于提高前排证据相关性。",
  "expected_abstention": false,
  "tags": ["retrieval", "reranking"]
}
```

这里的字段有不同用途：

| 字段 | 用途 | 不应被误用为 |
| --- | --- | --- |
| `expected_doc_ids` | 文档级检索相关性标注 | 唯一正确答案的证明 |
| `expected_chunk_ids` | 精细检索、定位和 citation 覆盖评估 | 强迫模型只引用一个 chunk |
| `expected_keywords` | 轻量辅助检查或人工审阅提示 | 语义正确性的唯一判断依据 |
| `answer_notes` | 人工评审的评分依据与可接受结论范围 | 直接拼进生成 prompt 的泄漏信息 |
| `expected_abstention` | 评估资料不足时的正确行为 | 默认把拒答视为错误 |
| `question_type` | 分组统计和失败诊断 | 影响线上用户请求的业务字段 |

### 3.2 为什么优先使用 JSONL

**JSONL（JSON Lines）**指每行存放一条独立 JSON 对象的文件格式。它适合评测集，因为：

1. 单条样例可独立读取、追加与定位。
2. Git diff 通常比一个巨大 JSON 数组更容易审查。
3. runner 可以流式处理大数据集，不必一次加载全部。
4. 一条坏记录不会让人工定位变得困难。

JSONL 只是持久化格式，不是领域模型。进入业务逻辑后，记录应被解析为结构化 `EvaluationCase`，避免让裸 `dict` 在 runner、指标计算和报告层之间流动。

### 3.3 评测问题的类型

路线要求至少覆盖四类问题。它们分别给系统施加不同压力：

| 类型 | 例子 | 主要评估什么 |
| --- | --- | --- |
| 事实型 `fact` | “某论文提出的 rerank 机制是什么？” | 精确召回、定义性回答、引用定位 |
| 比较型 `compare` | “论文 A 和 B 对 agent memory 的观点有何不同？” | 多来源召回、证据平衡、比较性回答 |
| 综述型 `summary` | “当前 RAG 中常见的检索优化有哪些？” | 覆盖广度、上下文组织、避免过度概括 |
| 引用定位型 `citation_lookup` | “哪篇论文讨论了某个术语？位于哪一节？” | metadata 完整性、页码/章节、citation 可追溯性 |

建议额外增加两类真实系统常见样例：

| 类型 | 例子 | 主要评估什么 |
| --- | --- | --- |
| 资料不足型 `abstention` | “知识库中未收录的论文提出了什么？” | 正确拒答，而非生成常识性幻觉 |
| 对抗歧义型 `ambiguous` | “它的效果为什么更好？” | query rewrite 是否擅自补全指代、系统是否请求澄清或保守回答 |

### 3.4 Golden Dataset 与普通测试数据的区别

**Golden dataset（黄金评测集）**是经过人工审阅、版本管理且长期稳定使用的评测样例集合。它不是随手写的 demo 问题，也不是单元测试的替代品。

```text
单元测试
  验证程序契约，例如未知 citation id 必须被拒绝。

Golden dataset
  验证系统能力，例如 hybrid + rerank 是否提高比较型问题的证据覆盖。
```

黄金评测集应具备：

1. 问题来自真实用户任务、真实论文内容或真实失败案例。
2. 每条证据标注有来源，能回到稳定的 `doc_id`、`version_id` 和可选 `chunk_id`。
3. 样例有版本号和变更记录；修改标注也必须被审阅。
4. 不只包含容易命中关键词的题目，也包含同义表达、跨文档比较、低频术语和资料不足场景。
5. 评测集与 prompt、索引和实验结果之间不存在信息泄漏。

---

## 4. 数据集设计、标注与版本控制

### 4.1 相关性不是二元的，但起点可以是二元标注

最简单的标注是“某文档/某 chunk 是否相关”。这足以计算 HitRate、Recall 和 MRR，但在真实论文问答中，相关性常有层次：

```text
0：无关
1：提到相关术语，但不能支撑回答
2：部分相关，可作为补充证据
3：核心证据，直接支撑预期结论
```

第一阶段可采用文档或 chunk 的二元相关性标注，优先建立稳定流程。后续若需要判断排序质量，再引入分级相关性并计算 nDCG 等指标。不要一开始追求复杂标注体系，却无法稳定维护 80 条基础样例。

### 4.2 文档级、chunk 级与答案级标注

三类标注不可混淆：

```text
文档级标注：哪篇论文相关。
chunk 级标注：哪一个片段直接支持结论。
答案级标注：回答需要包含哪些结论、比较维度或拒答行为。
```

文档级标注较容易维护，适合早期 `HitRate@k`。但一个文档被找回，不代表正确页面或章节被找回。对于 citation lookup 和事实型问题，应逐步补充 chunk 级标注。

答案级不必一开始给出唯一“标准答案”。论文问答常存在合理表述差异，更适合使用 `answer_notes`、关键结论、需比较维度和人工 rubric。把自然语言参考答案当成唯一真值，会错误惩罚表达不同但证据充分的回答。

### 4.3 资料不足样例必须显式标注

RAG 评测若只包含“知识库必有答案”的问题，会鼓励模型永远尝试作答。真实系统还需要知道何时不该回答。

`expected_abstention=true` 的样例应明确说明：

1. 预期知识不在当前索引中。
2. 合理系统应拒答或明确资料不足。
3. 若系统声称具体事实且给出不相关 citation，应被判为严重错误。

这类样例用于评价 **abstention precision**：系统发生拒答时，拒答是否合理；以及 **abstention recall**：面对确实无证据的问题，系统是否成功拒答。两者往往存在取舍：过于保守会提高前者但降低有资料问题的回答率。

### 4.4 数据泄漏

数据泄漏是指本不应在评测时可见的信息，以某种方式影响了系统结果。例如：

1. 将 `answer_notes` 放入 query rewrite 或 answer prompt。
2. 根据评测集问题专门手写规则词典，却没有在真实请求中通用化。
3. 用同一批问题反复调参，最后又用它们宣布“效果提升”。
4. 索引重建后文档版本变化，却继续把旧 `expected_chunk_ids` 当作真值。

工程上应至少区分：

```text
development：用于设计与调试，允许频繁查看。
validation：用于选择配置，但应控制调参次数。
holdout：只在阶段性结论时使用，避免反复针对它优化。
```

对于当前 80 到 120 条的小规模评测集，可以先用简单的 `split` 字段管理；重点不是复杂统计流程，而是避免同一题既是“调参依据”又被当作完全独立的最终证明。

### 4.5 评测集版本必须绑定索引版本

当前项目已有 `IndexManifest`、文档版本与 chunk metadata。评测集中的期望 `chunk_id` 只能在对应索引版本下解释。

当发生以下变化时，旧结果不可直接横向比较：

- 原始论文内容变更；
- parser、清洗逻辑或 chunking 策略变更；
- embedding 模型、维度或距离度量变更；
- 评测集标注本身变更。

每次实验至少应记录：

```text
evaluation_dataset_id + dataset_version
index_id + artifact_definition_hash + document_set_hash
experiment_config_snapshot
application_version
prompt_version
执行时间与随机性控制信息
```

否则“指标从 0.62 变到 0.71”没有可解释性：它可能来自 retriever 改进，也可能只是换了语料、换了问题或换了不同的 chunk identity。

---

## 5. 检索评测指标

设某个评测问题的相关证据集合为 `R`，系统按顺序返回的前 `k` 个结果为 `L@k`。这里的元素可以是文档 id，也可以是 chunk id，但一次指标计算必须使用同一粒度。

### 5.1 HitRate@k

**HitRate@k** 关心前 `k` 个结果中是否至少命中一条相关证据：

```text
Hit@k = 1，若 R ∩ L@k 不为空
Hit@k = 0，若 R ∩ L@k 为空

HitRate@k = 所有问题的 Hit@k 平均值
```

它回答：“系统至少找到了一个有用来源的比例是多少？”

优点：直观、适合早期判断检索是否完全漏召回。

局限：命中一个相关结果和命中全部关键结果都会得到 1。比较型问题通常需要多个来源，单独使用 HitRate 容易掩盖证据不完整。

### 5.2 Recall@k

**Recall@k** 衡量前 `k` 个结果覆盖了多少已标注相关证据：

```text
Recall@k = |R ∩ L@k| / |R|
```

例如，比较型问题需要论文 A、B 的两个核心 chunk；top-5 只找回 A，则 Recall@5 为 `1 / 2`。

它回答：“系统把需要的证据找回得有多完整？”

注意：若标注只列出一个“代表性相关 chunk”，Recall 会被高估或低估。因此，评测集要明确 `expected_chunk_ids` 是“所有必须证据”还是“任意可接受证据之一”。后者应使用可接受集合或分组表达，不能简单拿一个 id 做唯一真值。

### 5.3 MRR

**MRR（Mean Reciprocal Rank，平均倒数排名）**强调第一个相关结果出现得是否足够靠前。

对单个问题：

```text
RR = 1 / 第一条相关结果的 rank
```

总体：

```text
MRR = 所有问题 RR 的平均值
```

若首条相关结果排第 1，RR 为 1；排第 5，RR 为 0.2；完全未命中则为 0。

它尤其适合衡量 reranker：rerank 可能不改变 Recall@k，却把已有相关证据从第 8 位提升到第 1 位，此时 MRR 会提高，而用户体验通常也会更好。

### 5.4 Precision@k 与为什么路线没有把它作为首要指标

```text
Precision@k = |R ∩ L@k| / k
```

它衡量前 `k` 个结果中相关证据的比例。RAG 中它很有价值，因为无关 chunk 会挤占 token budget、干扰模型并增加错误引用风险。

但早期论文知识库常难以完整标注所有“部分相关”的 chunk，导致 Precision 对标注质量非常敏感。因此路线优先要求 HitRate、Recall、MRR，并在 context 阶段补充 precision 思维。后续标注成熟后，应加入 Precision@k 与 nDCG。

### 5.5 nDCG：分级相关性成熟后的排序指标

**nDCG（Normalized Discounted Cumulative Gain）**适合相关性有等级时使用。它会让排在前面的高价值证据贡献更大，并将实际排序与理想排序归一化比较。

它解决的问题是：两个结果列表都找回 3 个相关 chunk，但一个把核心证据放在第 1 位，另一个把核心证据排在第 10 位，二者不应被视为同样好。

当前子模块不要求立刻实现 nDCG。先建立可靠的二元标注和 MRR，后续再引入分级标注更稳妥。

---

## 6. Context 评测：检索到不等于模型看到了

`RetrievedChunk[]` 经 evidence transformation 和 context packing 后，才成为 `PackedContext`。因此必须区分：

```text
RetrievedChunk[]
  检索阶段找回的候选。

PackedContext.used_chunks / citations
  在预算、去重和变换后真正允许模型使用的证据。
```

### 6.1 Context Recall

**Context Recall** 关注完成回答所需的相关证据，有多少真正进入了上下文。

```text
Context Recall = 已进入 PackedContext 的相关证据数 / 期望相关证据数
```

若 Retrieval Recall 很高、Context Recall 很低，常见原因是：

- top-k 太大，相关 chunk 被更多噪声挤出 token budget；
- chunk 过长，少量 chunk 已耗尽预算；
- evidence transformer 合并或压缩时丢失来源；
- ContextPacker 的排序规则与回答任务不匹配。

### 6.2 Context Precision

**Context Precision** 关注进入 prompt 的证据中，有多少与问题相关。

```text
Context Precision = 上下文中相关证据数 / 上下文使用证据总数
```

它不是单纯追求“上下文越短越好”。过短可能漏掉比较问题所需的第二个来源；过长则可能增加模型注意力分散和 token 成本。应结合问题类型观察：事实型往往偏好高 precision，综述型通常需要更高 recall。

### 6.3 Context 的工程价值

当前 `PackedContext` 已有 `used_chunks`、`dropped_chunks`、`citations`、segments 与 token usage。这些结构不是只服务于 prompt，它们也是子模块 8 能准确诊断问题的前提。

评测 runner 应记录“召回结果”和“实际上下文”两份快照摘要，才能回答：

```text
相关资料是没被找回？
还是被找回后在 packing 时被放弃？
```

---

## 7. 回答与引用评测指标

### 7.1 Answer Relevance

**Answer relevance（回答相关性）**衡量回答是否真正回应了用户问题，而不是只复述相关术语。

例如用户问“rerank 的代价是什么”，回答只解释“rerank 会重排候选”可能有 citation，也可能 faithful，但仍未回答“代价”。

回答相关性通常需要人工 rubric 或 LLM judge 辅助判断。评审时应关注：

1. 是否回答了问题的核心谓词，例如“比较”“原因”“限制”“代价”。
2. 是否遗漏必要比较维度或条件。
3. 是否引入无关背景，把真正结论埋没。
4. 是否使用用户要求的语言和可理解的表述。

它不能从是否包含关键词、是否拥有 citation 或回答长度直接推导。

### 7.2 Faithfulness 与 Groundedness

这两个词经常被混用。项目中建议采用下列工作定义：

```text
Faithfulness
  回答中的事实主张是否能由给定上下文支持。

Groundedness
  回答是否整体以检索证据为依据，并避免使用上下文之外无法验证的事实。
```

二者都关注“不要编造”，但 faithfulness 更适合逐条 claim 对证据的支持关系；groundedness 更适合评价整段回答的受证据约束程度。

当前 `CitationValidator` 能确定性保证 citation id 来自本次 `PackedContext`，但不能证明 `[C1]` 真的支持紧邻的一句话。这正是回答级人工审阅或 LLM judge 的价值所在。

### 7.3 Citation Correctness、Coverage 与 Answer Has Citation Ratio

应至少区分以下指标：

| 指标 | 问题 | 当前项目可如何计算 |
| --- | --- | --- |
| `answer_has_citation_ratio` | 非拒答回答是否至少携带一个合法引用？ | `RagAnswer.citations` 非空的比例 |
| citation validity | citation id 是否来自本次上下文？ | 已由 `CitationValidator` 确定性保障 |
| citation coverage | 期望来源中有多少最终被回答引用？ | 与 `expected_doc_ids` 或 `expected_chunk_ids` 比较 |
| citation correctness | 一个 citation 是否实际支撑其关联主张？ | 人工或 judge 审阅，不能只靠 id 匹配 |
| citation completeness | 回答中的重要可验证主张是否都给出足够来源？ | 人工或 claim 级审阅 |

`answer_has_citation_ratio` 很容易计算，但它只是最低卫生指标。一个回答即使每句都附 `[C1]`，仍可能错误引用同一个无关 chunk。

### 7.4 拒答质量

拒答不是简单的 0 分。需要分别记录：

```text
正确拒答
  知识库确实没有足够证据，系统保守拒答。

错误拒答
  相关证据已存在，却因检索、packing 或生成失败而放弃回答。

危险作答
  无充分证据仍声称具体事实，或提供无关 citation。
```

对子模块 8 而言，资料不足样例与有答案样例都应包含。否则系统可能通过“永远拒答”获得表面上的低幻觉率，却失去实用价值。

---

## 8. 人工评审、LLM Judge 与自动指标

### 8.1 三种评估方法各自的边界

```text
确定性自动指标
  例如 HitRate@k、MRR、citation id 合法性、延迟、token 数。
  优点是稳定、低成本、可复现；缺点是难判断语义质量。

人工评审
  例如回答是否真正比较了两篇论文、citation 是否支撑主张。
  优点是语义可靠；缺点是成本高、主观性与一致性需要管理。

LLM Judge
  用另一个受控 prompt 的模型按 rubric 打分。
  优点是可规模化；缺点是可能偏好冗长文本、受模型版本影响，也会产生额外成本。
```

真实工程通常采用组合策略：硬性结构与检索指标自动化，关键样例人工复核，规模化语义趋势使用 judge 辅助。

### 8.2 LLM-as-a-Judge 不是客观真相机

LLM judge 可以回答“给定 question、context、answer 和 rubric，这个回答是否覆盖主要结论”，但它也可能：

- 偏好长度更长、语言更流畅的答案；
- 被答案中的自信语气影响；
- 漏掉论文中的细微限定条件；
- 因 judge 模型、prompt、temperature 变化而改变分数；
- 读取到不应看到的参考答案，从而掩盖真实系统缺陷。

因此 judge 必须保存模型标识、prompt 版本、temperature、rubric 和原始评分理由摘要。对重要结论，应抽样进行人工复审，而不是把 judge 分数当作唯一真值。

### 8.3 Ragas 与 LlamaIndex Evaluation 的位置

**Ragas** 与 **LlamaIndex evaluation** 是常见 RAG 评测框架或工具集。它们可提供 context relevance、faithfulness、answer relevance 等评测流程，并常借助 embedding 或 LLM judge。

它们适合解决：

- 批量调用 judge 的编排；
- 常见 RAG 指标的标准化实现；
- 结果导出与实验比较；
- 快速建立第一版语义评估基线。

它们不能替代：

- 你对论文任务的证据标注；
- 当前项目的 `doc_id`、`chunk_id`、索引版本与 citation 契约；
- 可解释的失败分类；
- 人工确定“某条论文证据是否真的支持某主张”。

本项目下一步应先建立自身的评测领域模型、数据集格式和运行记录，再决定是否通过独立 adapter 接入 Ragas 或 LlamaIndex。不要让第三方框架的数据模型侵入 `RagPipeline`、`generation` 或 `retrieval` 的核心业务模型。

如需引入相关库，应在后续练习说明中记录库名、用途、安装方式、版本锁定策略、模型成本和数据外发风险；本概念阶段不直接安装依赖或修改环境。

---

## 9. 实验管理：让结果可复现、可比较

### 9.1 实验不是“换个配置再跑一次”

一次可比较的实验应回答一个明确假设，例如：

```text
假设：在相同索引版本、相同问题集与相同 top-k 下，开启 lexical rerank
可以提升比较型问题的 MRR 与 Context Recall，但会增加 P95 延迟。
```

为了检验它，应保持其他变量不变，只改变 rerank 开关。若同时更换 embedding 模型、chunk size、top-k 和 prompt，就无法知道变化来自哪里。

### 9.2 Baseline、变量与实验矩阵

**Baseline（基线）**是当前已知可用、可复现的配置。没有基线，就没有“提升”的参照物。

一个小而可信的实验矩阵可以是：

| 实验 | Retriever | Rerank | top-k | 主要回答的问题 |
| --- | --- | --- | --- | --- |
| `baseline_vector_k5` | vector | off | 5 | 向量召回的基础表现 |
| `vector_rerank_k5` | vector | on | 5 | rerank 是否改善排序，延迟代价如何 |
| `hybrid_rerank_k5` | hybrid | on | 5 | 混合召回是否提升关键词与比较型覆盖 |

路线要求至少三组配置对比。开始时不要同时扫描 chunk size、top-k、retriever、reranker、prompt、模型温度等全部组合；组合爆炸会让数据集与运行成本失控。先围绕一个假设建立小矩阵，再扩展。

### 9.3 Config Snapshot

实验记录不能只写“hybrid + rerank”。至少要持久化完整、可序列化的 **config snapshot**：

```text
retrieval strategy
reranking strategy 与开关
top_k / candidate_limit
context packing 预算
query planning strategy 与开关
generation provider、model、temperature、prompt version
index manifest identity
```

注意密钥永远不进入 snapshot。模型 provider、模型名和公开的运行参数是可记录配置；API key、Authorization header 与完整敏感 prompt 不是。

### 9.4 运行级记录与样例级记录

一次实验有两种粒度的产物：

```text
EvaluationRun
  一个数据集、一个索引版本、一个配置快照的一次整体执行。

EvaluationCaseResult
  一个评测问题在该次运行中的事实记录。
```

样例级记录应保存足够的诊断事实：

```text
case_id
question 与 question_type
resolved configuration identifier
trace_id
retrieved chunk identities、rank、score
packed citation identities、used/dropped chunk 摘要
answer status、citation identities、abstention reason
latency、token usage、错误码
自动指标与人工评审字段
```

汇总指标必须能够回溯到样例级事实。只保存一个平均分会让你无法解释“为什么 Recall@5 降低”或“哪类问题受影响”。

### 9.5 延迟、成本与稳定性也是结果

一个配置的质量提升不等于适合上线。实验应至少记录：

- 总耗时、平均延迟、P50/P95 延迟；
- 各阶段耗时，例如 query planning、retrieval、rerank、packing、generation；
- LLM 输入/输出 token；
- 成功率、拒答率、超时率与重试次数；
- 每次运行的成本估算，若 provider 提供可用计费信息。

例如，hybrid + rerank 让 MRR 从 0.54 提高到 0.60，但 P95 延迟从 0.8 秒升到 3.6 秒。这个结论不是“新配置更好”，而是“它在特定质量目标下值得或不值得该延迟”。

---

## 10. 当前项目中评测子系统的推荐边界

子模块 8 将新增一个独立 `evaluation` 领域包，而不是让 `retrieval`、`generation` 或 API handler 负责批量运行和指标汇总。推荐职责如下：

```text
app/evaluation/
  models.py
    EvaluationCase、EvaluationRun、EvaluationCaseResult、ReviewAnnotation

  datasets/
    JSONL Dataset Repository、schema 校验、版本读取

  metrics/
    纯指标计算：HitRate、Recall、MRR、citation coverage 等

  reviewing/
    人工评审字段、未来 LLM judge adapter 的边界

  runner/
    EvaluationRunner：逐条调用稳定的 RAG 应用服务，收集事实并调用指标模块

  reporting/
    结果持久化、摘要表、失败案例和 EVALUATION.md 所需的数据
```

这只是目标分层，不表示现在必须一次创建全部文件。工程边界比目录数量更重要：

| 组件 | 应负责 | 不应负责 |
| --- | --- | --- |
| `EvaluationDatasetRepository` | JSONL 的读取、写入、schema 边界 | 运行 RAG 请求或计算 MRR |
| `EvaluationRunner` | 运行一组 case、收集输出、协调指标 | 私自构造 Retriever、LLM Client 或 Runtime |
| metric 函数/类 | 从结构化结果计算单一指标 | 读取文件、调用 LLM、写 Markdown |
| reporter/writer | 写 JSONL、Markdown 或表格结果 | 改变指标语义或创建输出目录 |
| review 模块 | 保存人工/LLM 评审事实 | 代替检索指标或修改生产回答 |

依赖方向应保持：

```text
evaluation runner
  -> 稳定的应用服务接口（RagPipeline / SearchService 的窄 Protocol）
  -> evaluation models
  -> metric / reporting abstractions

生产 retrieval 与 generation
  不反向依赖 evaluation
```

`EvaluationRunner` 应由 Factory 统一组装，使用现有 `ApplicationRuntime` 提供的索引和服务。它不能在循环中重新构建 `ApplicationFactory`、自行读取 `settings.toml`，或为每题创建新的 LLM client；这会破坏配置一致性、性能和可复现性。

### 10.1 Settings、Config 与实验配置

项目既有约束继续有效：

```text
外部 TOML/环境变量
  -> EvaluationSettings
  -> Factory 适配
  -> EvaluationConfig
  -> EvaluationRunner / Reporter / Review Adapter
```

- `EvaluationSettings` 表示外部文件中的路径、数据集标识、默认输出目录、是否启用 judge 等。
- `EvaluationConfig` 是运行时不可变配置，只提供评测对象实际所需的参数。
- 某次实验覆盖的 retrieval/generation 配置不应临时修改全局 Settings 对象，而应成为显式 `ExperimentConfig` 或快照输入。

`ExperimentConfig` 不是 Settings 的别名。它表示“这一轮实验选择了什么变量”，并且要连同结果保存；Settings 表示“应用从哪里读取默认行为和路径”。

### 10.2 Repository、Collection 与结果持久化

遵守现有命名规则：

```text
EvaluationDatasetRepository
  与 JSONL 或未来数据库交互。

EvaluationResultRepository
  读取、写入运行记录与样例结果。

EvaluationResultCollection
  仅在内存中组织某次已加载/已计算的结果。
```

不要新建模糊的 `EvaluationStore`。持久化、内存集合、运行服务和报告写入的职责应可从名称中直接看出。

---

## 11. 建议的数据流与产物布局

子模块 8 的离线运行流程应接近：

```text
评测数据集 JSONL
  -> EvaluationDatasetRepository.load()
  -> EvaluationCase[]
  -> EvaluationRunner
      -> SearchService / RagPipeline
      -> EvaluationCaseResult[]
  -> Metrics
      -> EvaluationSummary
  -> Result Repository / Writer
      -> case_results.jsonl
      -> summary.json
      -> failures.jsonl
      -> EVALUATION.md 的表格与案例素材
```

推荐将数据与运行产物区分：

```text
evaluation/
  datasets/
    rag_papers_v1.jsonl          # 可审阅、可版本管理的输入

  runs/
    2026-09-15_vector_k5/        # 某次不可变运行产物
      run.json
      case_results.jsonl
      summary.json
      failures.jsonl
```

实际路径最终应通过 Settings/Config 配置，不写死在 runner 内。目录创建属于应用启动或运行准备阶段，不属于 writer 的职责。

### 11.1 为什么不直接复用在线日志

在线 `RagTrace` 用于一次真实请求的可观测性；它可能只保留安全摘要、并受隐私和采样约束。评测结果需要额外保存期望证据、指标、配置快照和人工 review 字段。

二者可以通过 `trace_id` 关联，但不能互相替代：

```text
RagTrace
  描述一次请求经历了哪些运行阶段。

EvaluationCaseResult
  解释这次请求相对“期望证据与评分规则”表现如何。
```

---

## 12. 失败分类与根因诊断

“模型没答好”不是可行动的失败分类。每条失败样例应尽量归入最早发生、且最能解释结果的阶段：

| failure_type | 典型表现 | 优先检查位置 |
| --- | --- | --- |
| `document_loading` | 目标论文根本不在索引输入中 | loader 报告、源文件路径 |
| `parsing_or_cleaning` | 目标定义被解析损坏、页码错误 | ParsedDocument、parser 输出 |
| `chunking` | 结论被切碎或 metadata 缺失 | ChunkingReport、相邻 chunk |
| `embedding_or_indexing` | 文档/向量版本不一致、索引未更新 | Manifest、Repository、完整性校验 |
| `query_planning` | 改写遗漏术语、偏离原始问题 | QueryPlan、fallback 状态 |
| `retrieval` | 相关证据完全未进入 top-k | RetrievedChunk、HitRate/Recall |
| `reranking` | 相关结果被召回却排到后面 | rerank 前后 rank、MRR |
| `context_packing` | 已召回证据未进入上下文 | used/dropped chunks、token usage |
| `generation` | 上下文充分但回答遗漏、误解或表达不相关 | prompt、模型输出、answer relevance |
| `citation` | 回答与来源不一致、缺失或引用错误 | citation ids、validator、人工核验 |
| `abstention` | 不该拒答或应该拒答却继续作答 | expected_abstention、证据覆盖 |
| `infrastructure` | 超时、provider 故障、文件写入失败 | ErrorCode、trace、运行日志 |

一个样例可以有辅助标签，但报告时应指定一个主要根因。否则失败案例无法驱动下一轮实验设计。

---

## 13. 比较实验与统计解释

### 13.1 不只比较平均值

假设两个配置的平均 MRR 分别是 0.61 与 0.63。若只有 20 条问题，或提升只来自一两条极端样例，这个差异未必可靠。

小规模评测集至少应同时查看：

1. 每条 case 的成败变化，而不只看平均数。
2. 分问题类型的指标，例如事实型、比较型、引用定位型。
3. 失败样例是否集中在某个文档、术语或索引版本。
4. 同一问题在两个配置下的配对差异。
5. 延迟、token 和拒答率是否发生不可接受的变化。

### 13.2 置信区间与显著性：先理解，再逐步引入

**置信区间**用于表达指标估计的不确定范围。常见的 bootstrap 方法会对评测样例反复重采样，得到平均 Recall、MRR 等指标可能波动的区间。

**配对比较**指同一批问题在两个配置下分别运行，再比较每个问题的差异。因为问题难度被固定，通常比拿两次独立平均值直接比较更有解释力。

子模块 8 的第一版无需追求完整统计学平台，但不能把 0.01 的微小平均差异夸大为确定提升。报告应使用保守表述：

```text
在当前 80 条数据集上，hybrid + rerank 的 MRR 高于 vector baseline；
提升主要来自比较型问题，且 P95 延迟明显增加。该结论仍需要扩展 holdout 数据集验证。
```

### 13.3 随机性与真实 LLM

检索基线可能近似确定性，但真实 LLM 的回答和 judge 分数通常有随机性。实验记录应包含：

- generation/judge 模型与 provider 标识；
- temperature、seed（若 provider 支持）、prompt version；
- 运行次数；
- 是否复用了 response cache；
- 索引与数据集版本。

对高成本模型，可以先在固定温度下单次跑完整集，再挑选边界配置和关键失败样例进行多次复跑。不要把一次偶然的语言模型输出当作算法结论。

---

## 14. `EVALUATION.md` 应该写什么

`EVALUATION.md` 不是把所有原始结果复制进去的日志，而是一份面向协作者和未来自己的实验结论文档。至少应包含：

1. **数据集说明**：问题数量、类型分布、数据来源、标注粒度、版本与已知局限。
2. **实验目标与假设**：本轮要验证什么，而不是泛泛描述“测试效果”。
3. **配置表**：每个实验的 retriever、rerank、top-k、index id、prompt/model 等关键快照。
4. **指标表**：检索指标、上下文指标、回答/引用指标、延迟和成本。
5. **分组结果**：按问题类型或难度展示差异，避免平均值掩盖问题。
6. **失败案例**：问题、预期证据、实际证据、回答摘要、主失败类型和诊断结论。
7. **结论与取舍**：说明优化提升了哪个指标、牺牲了什么、是否值得采用。
8. **可复现信息**：数据集版本、索引版本、结果目录、命令与限制条件。

一份有价值的结论示例：

```text
在 papers_eval_v1 的 84 条问题上，hybrid + lexical rerank 相比 vector baseline
将比较型问题的 Recall@5 从 0.48 提升到 0.66，但 P95 延迟增加 1.9 秒。
事实型问题的提升较小，且 4 条低频术语问题仍未命中。当前推荐将 hybrid + rerank
用于离线分析与高价值问答；低延迟场景仍使用 vector baseline。
```

这比“hybrid 效果更好”更可执行，也更诚实。

---

## 15. 常见误区

1. **只测几个自己熟悉的问题。** 这会形成确认偏差，无法代表真实用户任务。
2. **把检索命中等同于回答正确。** 证据可能没有进入 context，也可能被模型误解。
3. **把有 citation 等同于 citation 正确。** id 合法、来源存在、来源支持主张是不同层次。
4. **把所有拒答都记为失败。** 无证据时拒答是 RAG 的关键安全能力。
5. **为了提高指标删除难题。** 难题应被分类和诊断，而不是从数据集中悄悄移除。
6. **调参后仍用同一批数据宣布最终提升。** 这会过拟合评测集。
7. **只保存汇总均值。** 没有样例级结果，就无法解释回归、定位失败或复现结论。
8. **不记录索引版本。** chunking 或语料变化会让旧期望 chunk id 和指标失去意义。
9. **让 runner 自己构造生产依赖。** 会产生与线上不同的配置、client 生命周期和索引状态。
10. **把完整论文正文、API key 或敏感 prompt 写进公开评测报告。** 评测产物同样需要安全边界。
11. **直接把第三方评测框架的分数当作结论。** 工具提供测量方法，不提供你的任务定义和证据真相。
12. **一次实验同时修改所有变量。** 结果即使变化，也无法知道原因。

---

## 16. 与当前项目的衔接

子模块 8 会直接复用当前工程已经提供的可观测事实：

| 当前能力 | 评测中如何使用 |
| --- | --- |
| `IndexManifest` | 固定并记录索引身份、文档版本和构建配置 |
| `RetrievedChunk` | 计算文档/chunk 级 HitRate、Recall、MRR 与排序诊断 |
| `RetrievalPipelineResult` | 获得检索策略、trace、候选限制和结果列表 |
| `PackedContext` | 计算 Context Precision/Recall，分析 used/dropped chunks 与 token 预算 |
| `RagAnswer` | 获取 answer、citations、abstention、diagnostics 与 trace id |
| `CitationValidator` | 提供 citation 结构合法性的确定性底线 |
| `RagTrace` | 对齐阶段耗时、错误码和失败根因 |
| `ApplicationFactory` / `ApplicationRuntime` | 保证评测与生产使用同一组装和生命周期规则 |

因此，子模块 8 不应重写检索或生成逻辑。它的价值在于把已经存在的过程事实转化为结构化、可比较、可解释的实验结论。

---

## 17. 学习检查清单

进入子模块 8 工程实践前，你应能清楚回答：

1. 为什么 RAG evaluation 必须同时评估 retrieval、context 和 answer，而不能只比较最终回答？
2. `EvaluationCase` 为什么需要期望证据、问题类型、答案说明和拒答预期，而不只是 `question`？
3. HitRate@k、Recall@k、MRR 分别衡量什么？什么情况下它们会得出不同结论？
4. 为什么文档级相关性、chunk 级相关性和答案级 rubric 不能混为一种标注？
5. Retrieval Recall 高但 Context Recall 低时，最可能应该检查系统的哪个阶段？
6. citation validity、citation correctness 和 citation completeness 分别能说明什么？
7. 为什么无证据场景中的正确拒答应被视为成功，而不是一律计作失败？
8. 人工评审、LLM judge 和确定性自动指标各自适合判断什么？各自有哪些偏差？
9. 为什么一次实验必须记录索引版本、数据集版本、配置快照、模型/prompt 版本和样例级结果？
10. 为什么评测 runner 应复用 Factory/Runtime 组装的应用服务，而不是私自创建 Retriever 或 LLM Client？
11. 一项优化使 MRR 提升但 P95 延迟明显恶化时，报告应该如何表达结论？
12. 如何把“模型没答好”进一步诊断为 parsing、chunking、retrieval、packing、generation、citation 或 abstention 问题？

当你能结合当前的 `IndexManifest`、`RetrievalPipelineResult`、`PackedContext`、`RagAnswer` 和 `RagTrace` 回答这些问题时，就可以进入子模块 8 的工程实践：构建评测集、实现离线 runner、持久化实验结果，并形成第一份 `EVALUATION.md`。
