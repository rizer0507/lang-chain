"""Document compressor.

中文翻译:
文档压缩器。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pydantic import BaseModel

from langchain_core.runnables import run_in_executor

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain_core.callbacks import Callbacks
    from langchain_core.documents import Document


class BaseDocumentCompressor(BaseModel, ABC):
    """Base class for document compressors.

    This abstraction is primarily used for post-processing of retrieved documents.

    `Document` objects matching a given query are first retrieved.

    Then the list of documents can be further processed.

    For example, one could re-rank the retrieved documents using an LLM.

    !!! note
        Users should favor using a `RunnableLambda` instead of sub-classing from this
        interface.

    

    中文翻译:
    文档压缩器的基类。
    该抽象主要用于检索文档的后处理。
    首先检索与给定查询匹配的“文档”对象。
    然后可以进一步处理文档列表。
    例如，可以使用法学硕士对检索到的文档重新排序。
    ！！！注释
        用户应该倾向于使用“RunnableLambda”，而不是从此子类
        界面。"""

    @abstractmethod
    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:
        """Compress retrieved documents given the query context.

        Args:
            documents: The retrieved `Document` objects.
            query: The query context.
            callbacks: Optional `Callbacks` to run during compression.

        Returns:
            The compressed documents.

        

        中文翻译:
        在给定查询上下文的情况下压缩检索到的文档。
        参数：
            文档：检索到的“文档”对象。
            查询：查询上下文。
            回调：在压缩期间运行的可选“回调”。
        返回：
            压缩文档。"""

    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Callbacks | None = None,
    ) -> Sequence[Document]:
        """Async compress retrieved documents given the query context.

        Args:
            documents: The retrieved `Document` objects.
            query: The query context.
            callbacks: Optional `Callbacks` to run during compression.

        Returns:
            The compressed documents.

        

        中文翻译:
        在给定查询上下文的情况下异步压缩检索到的文档。
        参数：
            文档：检索到的“文档”对象。
            查询：查询上下文。
            回调：在压缩期间运行的可选“回调”。
        返回：
            压缩文档。"""
        return await run_in_executor(
            None, self.compress_documents, documents, query, callbacks
        )
