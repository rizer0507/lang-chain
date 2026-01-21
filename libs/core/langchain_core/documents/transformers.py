"""Document transformers.

中文翻译:
文档转换器。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from langchain_core.runnables.config import run_in_executor

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain_core.documents import Document


class BaseDocumentTransformer(ABC):
    """Abstract base class for document transformation.

    A document transformation takes a sequence of `Document` objects and returns a
    sequence of transformed `Document` objects.

    Example:
        ```python
        class EmbeddingsRedundantFilter(BaseDocumentTransformer, BaseModel):
            embeddings: Embeddings
            similarity_fn: Callable = cosine_similarity
            similarity_threshold: float = 0.95

            class Config:
                arbitrary_types_allowed = True

            def transform_documents(
                self, documents: Sequence[Document], **kwargs: Any
            ) -> Sequence[Document]:
                stateful_documents = get_stateful_documents(documents)
                embedded_documents = _get_embeddings_from_stateful_docs(
                    self.embeddings, stateful_documents
                )
                included_idxs = _filter_similar_embeddings(
                    embedded_documents,
                    self.similarity_fn,
                    self.similarity_threshold,
                )
                return [stateful_documents[i] for i in sorted(included_idxs)]

            async def atransform_documents(
                self, documents: Sequence[Document], **kwargs: Any
            ) -> Sequence[Document]:
                raise NotImplementedError
        ```
    

    中文翻译:
    文档转换的抽象基类。
    文档转换采用一系列“Document”对象并返回
    转换后的“Document”对象的序列。
    示例：
        ````蟒蛇
        类 EmbeddingsRedundantFilter（BaseDocumentTransformer，BaseModel）：
            嵌入：嵌入
            相似度_fn：可调用=余弦_相似度
            相似度阈值：浮点数 = 0.95
            类配置：
                任意类型允许 = True
            def 变换文档(
                self，文档：序列[文档]，**kwargs：任何
            ) -> 序列[文档]:
                stateful_documents = get_stateful_documents(文档)
                嵌入的文档 = _get_embeddings_from_stateful_docs(
                    self.embeddings，stateful_documents
                ）
                include_idxs = _filter_similar_embeddings(
                    嵌入式文档，
                    自相似性_fn，
                    自相似度阈值，
                ）
                返回 [stateful_documents[i] for i insorted(included_idxs)]
            异步 def atransform_documents(
                self，文档：序列[文档]，**kwargs：任何
            ) -> 序列[文档]:
                引发未实现错误
        ````"""

    @abstractmethod
    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        """Transform a list of documents.

        Args:
            documents: A sequence of `Document` objects to be transformed.

        Returns:
            A sequence of transformed `Document` objects.
        

        中文翻译:
        转换文档列表。
        参数：
            文档：要转换的一系列“文档”对象。
        返回：
            一系列转换后的“Document”对象。"""

    async def atransform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        """Asynchronously transform a list of documents.

        Args:
            documents: A sequence of `Document` objects to be transformed.

        Returns:
            A sequence of transformed `Document` objects.
        

        中文翻译:
        异步转换文档列表。
        参数：
            文档：要转换的一系列“文档”对象。
        返回：
            一系列转换后的“Document”对象。"""
        return await run_in_executor(
            None, self.transform_documents, documents, **kwargs
        )
