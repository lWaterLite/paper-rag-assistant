"""命令行入口。

运行示例：
python -m app.main index --source data/raw/papers
python -m app.main ask "RAG 为什么需要引用？" --source data/raw/papers
"""

from __future__ import annotations

import argparse
from pathlib import Path

from app.core.settings import EnvSettings, ProjectSettings
from app.factory import build_index_builder, build_rag_pipeline


def build_index(source: Path, env_settings: EnvSettings, project_settings: ProjectSettings):
    """构建练习用内存索引。"""

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
    print(f"- trace_id：{result.trace.trace_id}")


def handle_ask(args: argparse.Namespace) -> None:
    env_settings = EnvSettings.from_env()
    project_settings = ProjectSettings.from_toml()
    index, build_result = build_index(Path(args.source), env_settings, project_settings)
    pipeline = build_rag_pipeline(env_settings=env_settings, index=index)
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
    print(f"- index_trace_id：{build_result.trace.trace_id}")
    print(f"- ask_trace_id：{answer.trace_id}")
    print(f"- latency_ms：{answer.latency_ms}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="paper-rag-assistant 子模块 1 练习入口")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="构建练习用内存索引")
    index_parser.add_argument("--source", default="data/raw/papers", help="文档目录")
    index_parser.set_defaults(handler=handle_index)

    ask_parser = subparsers.add_parser("ask", help="执行一次 mock RAG 问答")
    ask_parser.add_argument("question", help="用户问题")
    ask_parser.add_argument("--source", default="data/raw/papers", help="文档目录")
    ask_parser.set_defaults(handler=handle_ask)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
