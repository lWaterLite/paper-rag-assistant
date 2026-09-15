# 子模块 8 工程说明：RAG 评测与实验管理

本子模块在既有的 ingest、indexing、retrieval、context packing 与 generation 流程之外，补上一个独立的离线评测子系统。它不修改在线问答的业务语义，而是复用同一条生产 `RagPipeline`，在固定的索引版本、黄金数据集和实验配置下记录可比较的运行事实。

评测的目标不是只产生一个分数。一次可审阅的 RAG 实验至少要回答：本次评的是哪份资料、哪个索引版本、哪组检索和后处理配置；每道题实际召回了什么、哪些证据进入了上下文、最终引用了什么；失败发生在什么位置；以及两个实验结果是否可以公平比较。

## 1. 已完成的能力

1. 使用 JSONL 维护人工标注的黄金评测集，每行一条问题与期望证据。
2. 支持以 `chunk_id`、`doc_id` 或来源相对路径作为黄金证据身份；优先级为 chunk、文档、来源路径。
3. 直接复用顶层 `RagPipeline.execute()`，取得查询计划、检索结果、最终上下文和回答，不绕过生产流程重新拼装评测结果。
4. 自动计算 HitRate、Recall、MRR、Precision、context recall/precision、citation coverage、回答是否含引用和拒答准确率。
5. 为每个问题写入样例级事实，为一次实验写入运行快照、汇总指标、失败样例和 Markdown 审阅报告。
6. 通过 `settings.toml` 定义多个可比较的 experiment profile；默认包含 `vector_baseline`、`vector_rerank`、`hybrid_rerank`。
7. 同一组评测 profile 共享同一个已加载的 `RagIndex`，但各自使用独立的不可变 Settings/Config 快照和 RAG pipeline，避免比较时原地修改全局配置。
8. 提供 CLI `evaluate` 命令，并配套 JSONL、指标、运行器和配置快照测试。

## 2. 工程目录

```text
app/
  evaluation/
    configuration.py          EvaluationConfig、ExperimentConfig
    models.py                 黄金样例、样例结果、指标、运行与产物领域模型
    runner.py                 单 profile 运行器与多 profile 套件运行器
    reporting.py              EVALUATION.md 报告构造与写入
    repositories.py           运行结果 JSON / JSONL 持久化 Repository
    datasets/
      jsonl.py                黄金集 JSONL Repository
    metrics/
      calculation.py          确定性自动指标和聚合逻辑

  core/settings/
    evaluation.py             EvaluationSettings、EvaluationExperimentSettings
  factory/
    evaluation.py             评测对象组装
    configs/evaluation.py     Evaluation Settings -> Config 适配
  pipeline.py                 RagPipelineExecutionResult 与 execute()
  main.py                     evaluate CLI 命令

evaluation/
  datasets/
    papers_eval_v1.jsonl      基于 data/raw/papers 的人工标注种子评测集
  runs/                       每次离线运行生成的产物目录，不应提交运行结果

tests/evaluation/
  test_dataset_repository.py  数据集结构、类型与唯一性校验
  test_metrics.py             自动指标与跨平台来源路径匹配
  test_runner.py              运行产物和失败样例记录
```

## 3. 系统位置与依赖方向

评测模块位于应用服务之上，但不属于 API、检索器或生成器的实现细节。它的唯一运行依赖是一个拥有 `execute()` 方法的 RAG 服务协议，以及固定的 `RagIndex`。

```text
JSONL 黄金集 + settings.toml
        |
        v
EvaluationRunner
        |
        +---- EvaluationRagService.execute(question, profile)
        |          |
        |          v
        |      RagPipeline.execute()
        |          |
        |          +-> QueryPlan
        |          +-> RetrievalPipelineResult
        |          +-> PackedContext
        |          +-> RagAnswer
        |
        +---- 自动指标计算
        |
        +---- EvaluationResultRepository + EvaluationReportWriter
                   |
                   v
             evaluation/runs/<run_id>/
```

依赖方向是单向的：`evaluation` 可以读取 `pipeline`、`generation`、`retrieval` 和 `indexing` 已公开的领域结果；这些生产模块不反向依赖 `evaluation`。这使在线 API 即使完全不启用离线评测，也不会被评测数据格式或报告逻辑污染。

## 4. 为什么新增 `RagPipeline.execute()`

已有的 `RagPipeline.ask()` 继续只返回 `RagAnswer`，因此 API 和调用方不需要改动。评测却不能只看最终回答：它必须知道正确证据有没有进入候选集、是否被 context packing 丢弃、最终 citation 是否覆盖期望资料。

因此 `app/pipeline.py` 新增了内部观察模型 `RagPipelineExecutionResult`：

```text
RagPipelineExecutionResult
  answer             最终 RagAnswer
  query_plan         查询改写、关键词和回退信息
  retrieval_result   统一后处理后的检索结果与 retrieval trace
  packed_context     实际喂给生成器的上下文、citation 和被丢弃证据
```

`execute()` 仍会经过 query planning、retrieval、evidence transformation、context packing 与 generation 的完整生产链路；它不是给评测模块重新实现一份简化 RAG。这样得到的指标反映真实用户请求实际经历的路径。

## 5. 黄金数据集设计

`evaluation/datasets/papers_eval_v1.jsonl` 是 UTF-8 JSONL：空行和 `#` 开头的说明行会被忽略；其余每一行必须是一个 JSON 对象。当前种子集直接对应 `data/raw/papers` 中的真实 Markdown 与 HTML 内容，并包含一个资料中没有答案的拒答样例。

一个正常样例可写成：

```json
{
  "id": "rag-eval-003",
  "question": "answer relevance 与 faithfulness 有什么区别？",
  "question_type": "compare",
  "expected_source_paths": ["data/raw/papers/rag_evaluation_note.md"],
  "expected_keywords": ["切中问题", "忠实", "检索上下文", "编造"],
  "answer_notes": "回答应区分是否回答问题和是否受证据约束。",
  "tags": ["metrics", "generation"],
  "split": "development"
}
```

字段含义如下。

| 字段 | 用途 | 是否自动参与计算 |
| --- | --- | --- |
| `id` | 样例稳定身份，整个数据集中必须唯一 | 是，用于审计和失败定位 |
| `question` | 原始用户问题 | 是，直接传给生产 pipeline |
| `question_type` | `fact`、`compare`、`summary`、`citation_lookup`、`abstention`、`ambiguous` | 是，用于分类型聚合 |
| `expected_chunk_ids` | 最精细的期望证据标注 | 是，优先级最高 |
| `expected_doc_ids` | 期望文档标注 | 是，未提供 chunk 时使用 |
| `expected_source_paths` | 期望来源的项目相对路径 | 是，未提供 chunk/doc 时使用 |
| `expected_keywords` | 人工审阅回答覆盖度时的提示 | 当前保留，不伪装成语义质量自动分数 |
| `answer_notes` | 人工标注的答题边界与允许表达 | 当前保留，供复盘 |
| `expected_abstention` | 该题是否应拒答 | 是，计算拒答准确率 |
| `tags`、`split` | 数据集筛选、版本演进和实验分组 | 当前作为可追溯元数据 |

非拒答样例必须至少拥有一种期望证据。`JsonlEvaluationDatasetRepository` 会在读入阶段拒绝空 ID、错误字段类型、无法解析的 JSON、重复 ID 和缺少期望证据的普通问题，避免“错误数据也能顺利跑完”的假象。

### 5.1 为什么支持来源路径

既有 `doc_id` 的来源身份中包含本机解析后的路径。它非常适合当前索引内部关联，但把它直接写入随项目提交的黄金集会让基准集与某一台机器的绝对工作目录绑定。

评测模型因而支持 `expected_source_paths`。指标计算时会把运行期 `RetrievedChunk.source_path` 与 JSONL 中的路径都规范化为 POSIX 形式，再比较证据身份。这解决 Windows 反斜杠与 JSONL 正斜杠差异，也让基准集能够伴随项目移动。对一个已经有稳定业务文档 URI 的生产系统，更推荐将文档的规范来源 URI 作为长期黄金身份。

## 6. 自动指标及其边界

### 6.1 检索与上下文指标

`calculate_case_metrics()` 根据期望证据与实际阶段产物计算：

```text
retrieval_hit_rate         top-k 中是否至少出现一条正确证据
retrieval_recall           正确证据集合被候选检索结果覆盖的比例
retrieval_mrr              第一条正确证据的倒数排名
retrieval_precision        返回候选中正确证据的比例
context_recall             正确证据中实际进入 PackedContext 的比例
context_precision          PackedContext 证据中正确证据的比例
citation_coverage          最终 RagAnswer citation 覆盖期望证据的比例
answer_has_citation_ratio  非拒答回答是否带至少一个 citation
abstention_accuracy        是否符合该样例的应答/拒答预期
```

对 `expected_abstention=true` 的样例，检索、上下文和 citation 的“正确证据覆盖”没有数学定义，因此这些字段会保持 `null`，而不是被误写为 0。该样例只参与 `abstention_accuracy`。这是区分“没有可期待的证据”与“存在证据但没有召回”的必要边界。

### 6.2 当前不会自动声称的指标

回答相关性、faithfulness、groundedness 和 citation correctness 都包含语义判断。当前代码为它们预留 `ManualReview` 结构，但不会用关键词匹配或 LLM 调用伪造这些评分。

真实工程中的下一步通常是：定义明确 rubric，抽样双人标注，记录裁决规则，再将独立 judge 的结果作为补充字段写入样例级产物。judge 自身也要被版本化、校准和审计，不能将模型评分当作无误差事实。

## 7. 配置流与实验隔离

评测行为配置位于 `settings.toml` 的 `[evaluation]`。敏感信息不属于评测模块，因此不应写入 `.env`。

```toml
[evaluation]
dataset_path = "evaluation/datasets/papers_eval_v1.jsonl"
dataset_id = "papers_eval"
dataset_version = "v1"
output_dir = "evaluation/runs"
include_answer_text = true

[evaluation.experiments.vector_baseline]
retriever = "vector"
top_k = 5
reranking_enabled = false

[evaluation.experiments.hybrid_rerank]
retriever = "hybrid"
top_k = 5
reranking_enabled = true
```

配置转换路线严格遵循项目约定：

```text
settings.toml
  -> ProjectSettings.evaluation: EvaluationSettings
  -> EvaluationConfigAdapter
  -> EvaluationConfig + ExperimentConfig[]
  -> EvaluationFactory
  -> EvaluationRunner / EvaluationSuiteRunner
```

`EvaluationSettings` 负责外部值的 Pydantic 校验与名称清理。`EvaluationConfig` 和 `ExperimentConfig` 是冻结的运行期快照，业务对象只接受后者，避免在运行中重新读取 TOML 或随意修改配置。

`ApplicationFactory.build_evaluation_suite(index)` 对每个 experiment 深拷贝 `ProjectSettings`，再用该 profile 覆盖 `retrieval.strategy`、`retrieval.top_k` 与 `retrieval.reranking.enabled`，最后新建独立 `ApplicationFactory` 和 `RagPipeline`。索引对象本身共享，因为它是本次比较固定不变的输入；策略配置和运行对象隔离，因为它们是要比较的变量。

## 8. 运行与产物

先准备或加载索引，再运行评测。命令不启动 Web 服务：

```powershell
python -m app.main evaluate --source data/raw/papers
python -m app.main evaluate --use-existing-index
```

第一条命令会先针对当前资料构建索引；第二条命令读取已经持久化的索引。评测配置中的每个 profile 都会生成一个带 UTC 时间和随机后缀的独立目录：

```text
evaluation/runs/<timestamp>_<experiment>_<suffix>/
  run.json                 数据集、实验配置和索引版本快照
  case_results.jsonl       每道题的检索、上下文、回答、citation 和自动指标
  summary.json             全局及按问题类型聚合的指标
  failures.jsonl           仅技术失败的样例结果
  EVALUATION.md            人工审阅友好的报告
```

目录创建属于 `EvaluationRunner.run()` 的工作流准备阶段；`LocalJsonEvaluationResultRepository.write()` 与 `EvaluationReportWriter.write()` 都只负责写入既有路径。这延续项目的职责边界：Repository 和 Writer 不在调用者未明确要求时私自创建目录。

样例执行发生 `AppError`、`OSError`、`RuntimeError`、`TypeError` 或 `ValueError` 时，运行器不会让整批实验中断；它会将该题写为 `status="error"`，并记录 `failure_type`、可用错误码、消息和 trace ID。数据集读取失败则在运行开始前明确失败，因为没有有效基准时继续运行没有意义。

## 9. 工厂与注册表边界

评测模块本身没有新的策略注册表：它复用既有 `RetrieverRegistry`、reranker registry、token estimator registry 和 LLM registry。它只负责把 profile 的策略名交给独立的 `ApplicationFactory`；具体策略是否合法仍由下游 registry 校验。

这避免了评测层复制一份“可用 retriever 列表”。将来注册第三方 retriever 时，只需要在应用组合根注册一次，在线请求与评测 profile 都会走同一入口。

`EvaluationFactory` 只组装 JSONL 数据集 Repository、结果 Repository、报告 Writer 和 Runner；它不构造索引，也不构造生产 RAG pipeline。后者由 `ApplicationFactory` 统一管理，保持 Factory 的职责清楚。

## 10. 自动验证

已新增的验证覆盖：

```powershell
python -m unittest discover -s tests/evaluation -t .
python -m unittest discover -s tests -t .
ruff check app/evaluation tests/evaluation
```

测试关注公共行为而不是某个真实 LLM 的具体措辞：

1. JSONL Repository 能读取真实结构，并拒绝重复 ID、错误类型和缺少期望证据的样例。
2. 指标计算验证正确证据在第二名时 MRR 为 `0.5`，拒答样例不会伪造 retrieval 分数。
3. 运行时来源路径可跨 Windows/POSIX 分隔符匹配。
4. 运行器会写出 JSON、JSONL、汇总与 Markdown 报告；单题运行失败会作为样例级事实被保留，而不是丢失整个批次。
5. `ConfigFactory` 在同一应用工厂中复用同一个 `EvaluationConfig` 快照，在不同应用工厂之间隔离快照。

## 11. 后续扩展边界

1. 按 `split`、`tag`、文档集合或问题难度筛选评测集，并在 `run.json` 记录精确筛选条件。
2. 增加基于 Bootstrap 的置信区间和显著性比较，而不是只比较两个平均值的小数点末位。
3. 将 `ManualReview` 接入版本化的人工评审流程；所有 judge prompt、模型版本和 rubric 也应进入运行快照。
4. 为公开运行结果增加回答脱敏或关闭 `include_answer_text`，避免报告泄露受限语料。
5. 将本地 JSON Repository 替换为对象存储或实验数据库时，保持 `EvaluationResultRepository` 协议不变。
6. 当业务系统拥有稳定的文档 URI 时，将 `expected_source_paths` 渐进迁移为规范 URI，减少本地目录结构对评测数据的影响。

本子模块已经形成“黄金数据 -> 固定索引 -> 多 profile 执行 -> 样例级事实 -> 可审阅报告”的完整闭环。之后接入 LLM judge、人工标注平台或实验追踪平台时，应在这个闭环外追加新的评估维度，而不是回到检索器或生成器内部塞入评测逻辑。
