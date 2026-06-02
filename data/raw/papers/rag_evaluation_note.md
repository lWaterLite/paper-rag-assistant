# RAG 评测笔记

RAG evaluation 需要同时关注检索质量和生成质量。

常见检索指标包括 HitRate@k、Recall@k 和 MRR。

HitRate@k 用来判断 top-k 检索结果中是否包含目标文档或目标片段。

MRR 关注正确结果在检索结果中的排名位置。

生成质量可以关注 answer relevance 和 faithfulness。

answer relevance 表示回答是否切中用户问题。

faithfulness 表示回答是否忠实于检索上下文，而不是模型自己编造。

如果一次回答失败，需要判断失败原因来自文档解析、chunking、检索、上下文组织还是最终生成。

