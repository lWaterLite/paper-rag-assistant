"""命令行入口。

运行示例：
python -m app.main index --source data/raw/papers
python -m app.main search "RAG 为什么需要引用？" --use-existing-index
python -m app.main ask "RAG 为什么需要引用？" --source data/raw/papers
"""

from __future__ import annotations

import argparse
from pathlib import Path

from app.core.settings import EnvSettings, ProjectSettings
from app.factory import build_index_builder, build_rag_index_from_storage, build_rag_pipeline, build_search_service


def build_index(source: Path, env_settings: EnvSettings, project_settings: ProjectSettings):
    """构建 RAG 离线索引。"""

    builder = build_index_builder(env_settings, project_settings)
    return builder.build_from_directory(source)


def handle_index(args: argparse.Namespace) -> None:
    env_settings = EnvSettings.from_env()
    project_settings = ProjectSettings.from_toml()
    _, result = build_index(Path(args.source), env_settings, project_settings)
    print("索引构建完成")
    print(f"- 文档数量：{result.document_count}")
    print(f"- chunk 数量：{result.chunk_count}")
    print(f"- 向量数量：{result.vector_count}")
    if result.ingestion_report_path is not None:
        print(f"- ingestion 报告：{result.ingestion_report_path.as_posix()}")
    if result.chunking_report_path is not None:
        print(f"- chunking 报告：{result.chunking_report_path.as_posix()}")
    if result.manifest_path is not None:
        print(f"- index manifest：{result.manifest_path.as_posix()}")
    if result.build_report_path is not None:
        print(f"- index 构建报告：{result.build_report_path.as_posix()}")
    print(f"- trace_id：{result.trace.trace_id}")


def handle_ask(args: argparse.Namespace) -> None:
    env_settings = EnvSettings.from_env()
    project_settings = ProjectSettings.from_toml()
    if args.use_existing_index:
        index = build_rag_index_from_storage(project_settings)
        build_result = None
    else:
        index, build_result = build_index(Path(args.source), env_settings, project_settings)
    pipeline = build_rag_pipeline(env_settings=env_settings, project_settings=project_settings, index=index)
    answer = pipeline.ask(args.question)

    print("回答：")
    print(answer.answer)
    print()
    print("引用：")
    for citation in answer.citations:
        print(f"- [{citation.citation_id}] {citation.title or citation.doc_id} | {citation.source_path}")
        print(f"  {citation.snippet}")
    print()
    print("Trace：")
    if build_result is not None:
        print(f"- index_trace_id：{build_result.trace.trace_id}")
    else:
        print(f"- loaded_index_id：{index.manifest.index_id}")
    print(f"- ask_trace_id：{answer.trace_id}")
    print(f"- latency_ms：{answer.latency_ms}")


def handle_search(args: argparse.Namespace) -> None:
    """执行一次检索，不生成回答。"""

    env_settings = EnvSettings.from_env()
    project_settings = ProjectSettings.from_toml()
    if args.use_existing_index:
        index = build_rag_index_from_storage(project_settings)
        build_result = None
    else:
        index, build_result = build_index(Path(args.source), env_settings, project_settings)

    service = build_search_service(env_settings, project_settings, index)
    result = service.search(
        args.query,
        top_k=args.top_k,
        retriever=args.retriever,
    )

    print("检索结果：")
    for chunk in result.results:
        print(f"- rank={chunk.rank} score={chunk.score} retriever={chunk.retriever}")
        print(f"  chunk_id：{chunk.chunk_id}")
        print(f"  source：{chunk.title or chunk.doc_id} | {chunk.source_path}")
        if chunk.section:
            print(f"  section：{chunk.section}")
        print(f"  text：{chunk.text[:240]}")
    print()
    print("Trace：")
    if build_result is not None:
        print(f"- index_trace_id：{build_result.trace.trace_id}")
    else:
        print(f"- loaded_index_id：{index.manifest.index_id}")
    print(f"- search_trace_id：{result.trace.trace_id}")
    print(f"- retriever：{result.retriever}")
    print(f"- top_k：{result.top_k}")
    print(f"- returned：{len(result.results)}")
    print(f"- latency_ms：{result.trace.latency_ms}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="paper-rag-assistant RAG 工程练习入口")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="构建离线 RAG 索引")
    index_parser.add_argument("--source", default="data/raw/papers", help="文档目录")
    index_parser.set_defaults(handler=handle_index)

    search_parser = subparsers.add_parser("search", help="只执行检索，不生成回答")
    search_parser.add_argument("query", help="检索查询")
    search_parser.add_argument("--source", default="data/raw/papers", help="文档目录")
    search_parser.add_argument("--use-existing-index", action="store_true", help="直接加载已有索引，不重新构建")
    search_parser.add_argument("--top-k", type=int, default=None, help="本次检索返回数量")
    search_parser.add_argument("--retriever", choices=["vector", "bm25"], default=None, help="本次检索策略")
    search_parser.set_defaults(handler=handle_search)

    ask_parser = subparsers.add_parser("ask", help="执行一次 mock RAG 问答")
    ask_parser.add_argument("question", help="用户问题")
    ask_parser.add_argument("--source", default="data/raw/papers", help="文档目录")
    ask_parser.add_argument("--use-existing-index", action="store_true", help="直接加载已有索引，不重新构建")
    ask_parser.set_defaults(handler=handle_ask)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
