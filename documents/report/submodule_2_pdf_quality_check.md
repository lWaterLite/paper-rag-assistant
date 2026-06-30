# 子模块 2 练习 4：真实 PDF 解析质量检查

检查日期：2026-06-30

本次检查目标：

- 使用 `data/raw/papers/pdf/` 中的真实论文 PDF 验证当前 ingestion 体系。
- 检查 PDF 是否能提取正文、页数、文本长度、block 数、页码追溯、解析问题记录。
- 观察页眉页脚、双栏顺序、表格、公式、引用页码等真实场景问题。

## 检查方式

执行路径：

```text
LocalDocumentLoader -> PdfDocumentParser -> PdfTextCleaner -> ParsedDocument
```

同时运行了完整索引入口：

```powershell
.\.venv\Scripts\python.exe -m app.main index --source data/raw/papers
```

索引入口可以成功生成：

```text
logs/ingestion_report.json
```

## 总体结论

当前 5 篇真实 PDF 都可以成功解析，并且每篇 PDF 的所有页面都能生成带页码的 blocks。整体已经满足“真实 PDF 能进入 RAG 系统”的基本要求。

但当前实现仍偏基础：

- PDF metadata 中没有可用 title 时，标题退化为文件名。
- 章节 heading 识别效果较弱，当前 5 篇 PDF 的 block 都被识别为 `paragraph`。
- 表格、图、公式只作为普通文本进入系统，没有结构化保留。
- 双栏论文的文本顺序基本可读，但局部会出现图表、页码、栏间文本混排。
- 页眉页脚清洗较保守，本次没有触发实际删除。

## 逐篇检查结果

| 文件 | 页数 | 文本长度 | block 数 | 页码覆盖 | parse issue | 质量结论 |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| `es_2023_ragas_evaluation.pdf` | 8 | 31,815 | 136 | 8/8 | 0 | 正文提取完整，表格内容会被压成普通文本 |
| `gao_2023_rag_survey.pdf` | 21 | 109,799 | 489 | 21/21 | 0 | 长综述解析完整，但双栏、图注和参考文献较密集 |
| `izacard_2020_fusion_in_decoder.pdf` | 6 | 23,228 | 127 | 6/6 | 0 | 正文可读，图表数字序列会混入段落 |
| `karpukhin_2020_dense_passage_retrieval.pdf` | 13 | 55,786 | 252 | 13/13 | 0 | 正文和参考文献可提取，公式符号有少量普通文本化 |
| `lewis_2020_retrieval_augmented_generation.pdf` | 19 | 69,275 | 293 | 19/19 | 0 | 正文提取完整，附录和表格区域顺序需要人工复查 |

## 质量观察

### 1. 正文提取

5 篇论文均有稳定正文输出，没有出现空文本、扫描版不可提取文本、受保护 PDF 无法读取等问题。

每页文本量也正常，没有低于 500 字符的异常空页：

- `es_2023_ragas_evaluation.pdf`：最少页约 2,695 字符。
- `gao_2023_rag_survey.pdf`：最少页约 517 字符。
- `izacard_2020_fusion_in_decoder.pdf`：最少页约 3,459 字符。
- `karpukhin_2020_dense_passage_retrieval.pdf`：最少页约 2,696 字符。
- `lewis_2020_retrieval_augmented_generation.pdf`：最少页约 1,137 字符。

### 2. 页码追溯

当前 `PdfDocumentParser._build_pdf_blocks()` 会为 PDF block 写入：

```text
page_start
page_end
```

本次 5 篇 PDF 的 block 页码覆盖都完整，因此后续 citation 可以追溯到页码级来源。

这是当前实现里最重要的合格点。

### 3. 页眉页脚清洗

当前 `PdfTextCleaner` 没有删除任何重复页眉页脚：

```text
removed_repeated_edge_lines = 0
```

这不一定是错误。因为当前策略比较保守，只检测每页顶部和底部的重复短行，并要求重复比例达到阈值。

但需要注意：

- `gao_2023_rag_survey.pdf` 每页开头有页码，例如 `1`、`11`、`21`。
- `lewis_2020_retrieval_augmented_generation.pdf` 局部存在重复短语，例如 `Knowledge-Intensive NLP Tasks`。
- 当前规则没有删除它们，说明误删风险较低，但噪声清理也不强。

### 4. 章节 heading 识别

当前 5 篇 PDF 的 block 类型统计中，全部都是：

```text
paragraph
```

这说明 heading 识别没有达到预期。

主要原因是 PDF 行合并时会把：

```text
Abstract
正文第一句...
```

合并成一个较长段落，例如：

```text
Abstract Generative models for open domain question answering...
```

合并后长度超过 heading 判断阈值，因此不会被识别为 heading。

后续建议：

- 在合并 PDF 行之前先识别独立标题行。
- 或者在 `_build_pdf_blocks()` 中保留原始行级结构，再构造 blocks。
- 对常见论文标题如 `Abstract`、`Introduction`、`References`、`Appendix` 做更稳定的规则。

### 5. 双栏与图表

多数论文正文顺序基本可读，但图表区域有典型 PDF 解析问题。

例如 `es_2023_ragas_evaluation.pdf` 的中间页出现：

```text
Faith.
Ans. Rel.
Cont. Rel.
Ragas
0.95
0.78
0.70
```

这说明表格被拆成普通行，数值和表头关系丢失。

`gao_2023_rag_survey.pdf` 的图注和双栏正文混合较多，RAG 检索仍可用，但如果后续要做高质量 citation 或表格问答，需要更专门的布局解析。

### 6. 乱码和公式

本次没有发现明显替换乱码字符：

```text
weird_chars = 0
```

但公式和特殊符号会以普通字符混入正文。当前系统还没有公式结构化能力，因此只能作为普通检索文本使用。

## 当前实现是否通过练习 4

通过基础验收。

已经满足：

- 真实 PDF 可以加载。
- 真实 PDF 可以解析出正文。
- 单篇 PDF 解析失败不会影响整体 ingestion。
- 解析结果保留页码。
- 可以生成 ingestion report。
- 可以进入索引流程并生成 chunk、embedding、vector。

仍需后续改进：

- 改进 PDF title 提取。
- 改进 PDF heading/section 识别。
- 把页眉页脚、页码噪声检测结果更明确地记录为 `ParseIssue`。
- 区分普通正文、图注、表格、参考文献。
- 对双栏论文引入更强的布局解析策略。

## 后续建议

短期可以优先做三件事：

1. 在 `PdfDocumentParser` 中从第一页文本推断标题，而不是只依赖 PDF metadata。
2. 在 `PdfDocumentParser._build_pdf_blocks()` 中改进 heading 识别，至少让 `Abstract`、`Introduction`、`References` 能成为 section。
3. 在 ingestion report 的每篇文档摘要中加入 `page_count`，方便快速判断 PDF 解析覆盖情况。

更长期可以考虑：

- 使用 PyMuPDF 的 block/layout 信息，而不是只使用 `page.get_text("text")`。
- 为表格区域接入专门解析器。
- 为扫描版 PDF 增加 OCR fallback。
- 将 PDF 解析质量指标写入单独的 quality report。
