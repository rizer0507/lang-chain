"""Retriever tool.

中文翻译:
检索工具。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

# Cannot move Callbacks and Document to TYPE_CHECKING as StructuredTool's
# 中文: 无法将回调和文档移动到 TYPE_CHECKING 作为 StructuredTool 的
# func/coroutine parameter annotations are evaluated at runtime.
# 中文: func/协程参数注释在运行时评估。
from langchain_core.callbacks import Callbacks  # noqa: TC001
from langchain_core.documents import Document  # noqa: TC001
from langchain_core.prompts import (
    BasePromptTemplate,
    PromptTemplate,
    aformat_document,
    format_document,
)
from langchain_core.tools.structured import StructuredTool

if TYPE_CHECKING:
    from langchain_core.retrievers import BaseRetriever


class RetrieverInput(BaseModel):
    """Input to the retriever.

    中文翻译:
    输入到检索器。"""

    query: str = Field(description="query to look up in retriever")


def create_retriever_tool(
    retriever: BaseRetriever,
    name: str,
    description: str,
    *,
    document_prompt: BasePromptTemplate | None = None,
    document_separator: str = "\n\n",
    response_format: Literal["content", "content_and_artifact"] = "content",
) -> StructuredTool:
    r"""Create a tool to do retrieval of documents.

    Args:
        retriever: The retriever to use for the retrieval
        name: The name for the tool. This will be passed to the language model,
            so should be unique and somewhat descriptive.
        description: The description for the tool. This will be passed to the language
            model, so should be descriptive.
        document_prompt: The prompt to use for the document.
        document_separator: The separator to use between documents.
        response_format: The tool response format.

            If `"content"` then the output of the tool is interpreted as the contents of
            a `ToolMessage`. If `"content_and_artifact"` then the output is expected to
            be a two-tuple corresponding to the `(content, artifact)` of a `ToolMessage`
            (artifact being a list of documents in this case).

    Returns:
        Tool class to pass to an agent.
    

中文翻译:
创建一个工具来检索文档。
    参数：
        检索器：用于检索的检索器
        name：工具的名称。这将被传递到语言模型，
            所以应该是独特的并且具有一定的描述性。
        描述：工具的描述。这将传递给语言
            模型，所以应该是描述性的。
        document_prompt：用于文档的提示。
        document_separator：文档之间使用的分隔符。
        response_format：工具响应格式。
            如果“内容”，则该工具的输出将被解释为以下内容
            一个“工具消息”。如果“content_and_artifact”则输出预计为
            是对应于“ToolMessage”的“(content,artifact)”的二元组
            （在本例中，工件是文档列表）。
    返回：
        传递给代理的工具类。"""
    document_prompt_ = document_prompt or PromptTemplate.from_template("{page_content}")

    def func(
        query: str, callbacks: Callbacks = None
    ) -> str | tuple[str, list[Document]]:
        docs = retriever.invoke(query, config={"callbacks": callbacks})
        content = document_separator.join(
            format_document(doc, document_prompt_) for doc in docs
        )
        if response_format == "content_and_artifact":
            return (content, docs)
        return content

    async def afunc(
        query: str, callbacks: Callbacks = None
    ) -> str | tuple[str, list[Document]]:
        docs = await retriever.ainvoke(query, config={"callbacks": callbacks})
        content = document_separator.join(
            [await aformat_document(doc, document_prompt_) for doc in docs]
        )
        if response_format == "content_and_artifact":
            return (content, docs)
        return content

    return StructuredTool(
        name=name,
        description=description,
        func=func,
        coroutine=afunc,
        args_schema=RetrieverInput,
        response_format=response_format,
    )
