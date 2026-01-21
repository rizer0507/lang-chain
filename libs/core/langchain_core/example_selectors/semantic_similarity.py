"""Example selector that selects examples based on SemanticSimilarity.

中文翻译:
示例选择器根据语义相似性选择示例。"""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from langchain_core.example_selectors.base import BaseExampleSelector
from langchain_core.vectorstores import VectorStore

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings


def sorted_values(values: dict[str, str]) -> list[Any]:
    """Return a list of values in dict sorted by key.

    Args:
        values: A dictionary with keys as input variables
            and values as their values.

    Returns:
        A list of values in dict sorted by key.
    

    中文翻译:
    返回字典中按键排序的值列表。
    参数：
        值：以键作为输入变量的字典
            和价值观作为他们的价值观。
    返回：
        字典中按键排序的值列表。"""
    return [values[val] for val in sorted(values)]


class _VectorStoreExampleSelector(BaseExampleSelector, BaseModel, ABC):
    """Example selector that selects examples based on SemanticSimilarity.

    中文翻译:
    示例选择器根据语义相似性选择示例。"""

    vectorstore: VectorStore
    """VectorStore that contains information about examples.

    中文翻译:
    VectorStore 包含有关示例的信息。"""
    k: int = 4
    """Number of examples to select.

    中文翻译:
    要选择的示例数量。"""
    example_keys: list[str] | None = None
    """Optional keys to filter examples to.

    中文翻译:
    用于过滤示例的可选键。"""
    input_keys: list[str] | None = None
    """Optional keys to filter input to. If provided, the search is based on
    the input variables instead of all variables.

    中文翻译:
    用于过滤输入的可选键。如果提供，则搜索基于
    输入变量而不是所有变量。"""
    vectorstore_kwargs: dict[str, Any] | None = None
    """Extra arguments passed to similarity_search function of the `VectorStore`.

    中文翻译:
    额外的参数传递给 `VectorStore` 的相似度搜索函数。"""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    @staticmethod
    def _example_to_text(example: dict[str, str], input_keys: list[str] | None) -> str:
        if input_keys:
            return " ".join(sorted_values({key: example[key] for key in input_keys}))
        return " ".join(sorted_values(example))

    def _documents_to_examples(self, documents: list[Document]) -> list[dict]:
        # Get the examples from the metadata.
        # 中文: 从元数据中获取示例。
        # This assumes that examples are stored in metadata.
        # 中文: 这假设示例存储在元数据中。
        examples = [dict(e.metadata) for e in documents]
        # If example keys are provided, filter examples to those keys.
        # 中文: 如果提供了示例键，请过滤这些键的示例。
        if self.example_keys:
            examples = [{k: eg[k] for k in self.example_keys} for eg in examples]
        return examples

    def add_example(self, example: dict[str, str]) -> str:
        """Add a new example to vectorstore.

        Args:
            example: A dictionary with keys as input variables
                and values as their values.

        Returns:
            The ID of the added example.
        

        中文翻译:
        向矢量存储添加一个新示例。
        参数：
            示例：以键作为输入变量的字典
                和价值观作为他们的价值观。
        返回：
            添加的示例的 ID。"""
        ids = self.vectorstore.add_texts(
            [self._example_to_text(example, self.input_keys)], metadatas=[example]
        )
        return ids[0]

    async def aadd_example(self, example: dict[str, str]) -> str:
        """Async add new example to vectorstore.

        Args:
            example: A dictionary with keys as input variables
                and values as their values.

        Returns:
            The ID of the added example.
        

        中文翻译:
        异步向矢量存储添加新示例。
        参数：
            示例：以键作为输入变量的字典
                和价值观作为他们的价值观。
        返回：
            添加的示例的 ID。"""
        ids = await self.vectorstore.aadd_texts(
            [self._example_to_text(example, self.input_keys)], metadatas=[example]
        )
        return ids[0]


class SemanticSimilarityExampleSelector(_VectorStoreExampleSelector):
    """Select examples based on semantic similarity.

    中文翻译:
    根据语义相似性选择示例。"""

    def select_examples(self, input_variables: dict[str, str]) -> list[dict]:
        """Select examples based on semantic similarity.

        Args:
            input_variables: The input variables to use for search.

        Returns:
            The selected examples.
        

        中文翻译:
        根据语义相似性选择示例。
        参数：
            input_variables：用于搜索的输入变量。
        返回：
            所选示例。"""
        # Get the docs with the highest similarity.
        # 中文: 获取相似度最高的文档。
        vectorstore_kwargs = self.vectorstore_kwargs or {}
        example_docs = self.vectorstore.similarity_search(
            self._example_to_text(input_variables, self.input_keys),
            k=self.k,
            **vectorstore_kwargs,
        )
        return self._documents_to_examples(example_docs)

    async def aselect_examples(self, input_variables: dict[str, str]) -> list[dict]:
        """Asynchronously select examples based on semantic similarity.

        Args:
            input_variables: The input variables to use for search.

        Returns:
            The selected examples.
        

        中文翻译:
        根据语义相似性异步选择示例。
        参数：
            input_variables：用于搜索的输入变量。
        返回：
            所选示例。"""
        # Get the docs with the highest similarity.
        # 中文: 获取相似度最高的文档。
        vectorstore_kwargs = self.vectorstore_kwargs or {}
        example_docs = await self.vectorstore.asimilarity_search(
            self._example_to_text(input_variables, self.input_keys),
            k=self.k,
            **vectorstore_kwargs,
        )
        return self._documents_to_examples(example_docs)

    @classmethod
    def from_examples(
        cls,
        examples: list[dict],
        embeddings: Embeddings,
        vectorstore_cls: type[VectorStore],
        k: int = 4,
        input_keys: list[str] | None = None,
        *,
        example_keys: list[str] | None = None,
        vectorstore_kwargs: dict | None = None,
        **vectorstore_cls_kwargs: Any,
    ) -> SemanticSimilarityExampleSelector:
        """Create k-shot example selector using example list and embeddings.

        Reshuffles examples dynamically based on query similarity.

        Args:
            examples: List of examples to use in the prompt.
            embeddings: An initialized embedding API interface, e.g. OpenAIEmbeddings().
            vectorstore_cls: A vector store DB interface class, e.g. FAISS.
            k: Number of examples to select.
            input_keys: If provided, the search is based on the input variables
                instead of all variables.
            example_keys: If provided, keys to filter examples to.
            vectorstore_kwargs: Extra arguments passed to similarity_search function
                of the `VectorStore`.
            vectorstore_cls_kwargs: optional kwargs containing url for vector store

        Returns:
            The ExampleSelector instantiated, backed by a vector store.
        

        中文翻译:
        使用示例列表和嵌入创建 k-shot 示例选择器。
        根据查询相似性动态重新排列示例。
        参数：
            示例：提示中使用的示例列表。
            embeddings：初始化的嵌入 API 接口，例如OpenAIEmbeddings()。
            vectorstore_cls：矢量存储数据库接口类，例如费斯。
            k：要选择的示例数。
            input_keys：如果提供，则搜索基于输入变量
                而不是所有变量。
            example_keys：如果提供，则用于过滤示例的键。
            vectorstore_kwargs：传递给相似度搜索函数的额外参数
                “VectorStore”的。
            vectorstore_cls_kwargs：包含矢量存储 url 的可选 kwargs
        返回：
            由向量存储支持的示例选择器实例化。"""
        string_examples = [cls._example_to_text(eg, input_keys) for eg in examples]
        vectorstore = vectorstore_cls.from_texts(
            string_examples, embeddings, metadatas=examples, **vectorstore_cls_kwargs
        )
        return cls(
            vectorstore=vectorstore,
            k=k,
            input_keys=input_keys,
            example_keys=example_keys,
            vectorstore_kwargs=vectorstore_kwargs,
        )

    @classmethod
    async def afrom_examples(
        cls,
        examples: list[dict],
        embeddings: Embeddings,
        vectorstore_cls: type[VectorStore],
        k: int = 4,
        input_keys: list[str] | None = None,
        *,
        example_keys: list[str] | None = None,
        vectorstore_kwargs: dict | None = None,
        **vectorstore_cls_kwargs: Any,
    ) -> SemanticSimilarityExampleSelector:
        """Async create k-shot example selector using example list and embeddings.

        Reshuffles examples dynamically based on query similarity.

        Args:
            examples: List of examples to use in the prompt.
            embeddings: An initialized embedding API interface, e.g. OpenAIEmbeddings().
            vectorstore_cls: A vector store DB interface class, e.g. FAISS.
            k: Number of examples to select.
            input_keys: If provided, the search is based on the input variables
                instead of all variables.
            example_keys: If provided, keys to filter examples to.
            vectorstore_kwargs: Extra arguments passed to similarity_search function
                of the `VectorStore`.
            vectorstore_cls_kwargs: optional kwargs containing url for vector store

        Returns:
            The ExampleSelector instantiated, backed by a vector store.
        

        中文翻译:
        使用示例列表和嵌入异步创建 k-shot 示例选择器。
        根据查询相似性动态重新排列示例。
        参数：
            示例：提示中使用的示例列表。
            embeddings：初始化的嵌入 API 接口，例如OpenAIEmbeddings()。
            vectorstore_cls：矢量存储数据库接口类，例如费斯。
            k：要选择的示例数。
            input_keys：如果提供，则搜索基于输入变量
                而不是所有变量。
            example_keys：如果提供，则用于过滤示例的键。
            vectorstore_kwargs：传递给相似度搜索函数的额外参数
                “VectorStore”的。
            vectorstore_cls_kwargs：包含矢量存储 url 的可选 kwargs
        返回：
            由向量存储支持的示例选择器实例化。"""
        string_examples = [cls._example_to_text(eg, input_keys) for eg in examples]
        vectorstore = await vectorstore_cls.afrom_texts(
            string_examples, embeddings, metadatas=examples, **vectorstore_cls_kwargs
        )
        return cls(
            vectorstore=vectorstore,
            k=k,
            input_keys=input_keys,
            example_keys=example_keys,
            vectorstore_kwargs=vectorstore_kwargs,
        )


class MaxMarginalRelevanceExampleSelector(_VectorStoreExampleSelector):
    """Select examples based on Max Marginal Relevance.

    This was shown to improve performance in this paper:
    https://arxiv.org/pdf/2211.13892.pdf
    

    中文翻译:
    根据最大边际相关性选择示例。
    本文表明这可以提高性能：
    https://arxiv.org/pdf/2211.13892.pdf"""

    fetch_k: int = 20
    """Number of examples to fetch to rerank.

    中文翻译:
    要获取以重新排名的示例数量。"""

    def select_examples(self, input_variables: dict[str, str]) -> list[dict]:
        """Select examples based on Max Marginal Relevance.

        Args:
            input_variables: The input variables to use for search.

        Returns:
            The selected examples.
        

        中文翻译:
        根据最大边际相关性选择示例。
        参数：
            input_variables：用于搜索的输入变量。
        返回：
            所选示例。"""
        example_docs = self.vectorstore.max_marginal_relevance_search(
            self._example_to_text(input_variables, self.input_keys),
            k=self.k,
            fetch_k=self.fetch_k,
        )
        return self._documents_to_examples(example_docs)

    async def aselect_examples(self, input_variables: dict[str, str]) -> list[dict]:
        """Asynchronously select examples based on Max Marginal Relevance.

        Args:
            input_variables: The input variables to use for search.

        Returns:
            The selected examples.
        

        中文翻译:
        Asynchronously select examples based on Max Marginal Relevance.
        Args:
            input_variables: The input variables to use for search.
        Returns:
            The selected examples."""
        example_docs = await self.vectorstore.amax_marginal_relevance_search(
            self._example_to_text(input_variables, self.input_keys),
            k=self.k,
            fetch_k=self.fetch_k,
        )
        return self._documents_to_examples(example_docs)

    @classmethod
    def from_examples(
        cls,
        examples: list[dict],
        embeddings: Embeddings,
        vectorstore_cls: type[VectorStore],
        k: int = 4,
        input_keys: list[str] | None = None,
        fetch_k: int = 20,
        example_keys: list[str] | None = None,
        vectorstore_kwargs: dict | None = None,
        **vectorstore_cls_kwargs: Any,
    ) -> MaxMarginalRelevanceExampleSelector:
        """Create k-shot example selector using example list and embeddings.

        Reshuffles examples dynamically based on Max Marginal Relevance.

        Args:
            examples: List of examples to use in the prompt.
            embeddings: An initialized embedding API interface, e.g. OpenAIEmbeddings().
            vectorstore_cls: A vector store DB interface class, e.g. FAISS.
            k: Number of examples to select.
            fetch_k: Number of `Document` objects to fetch to pass to MMR algorithm.
            input_keys: If provided, the search is based on the input variables
                instead of all variables.
            example_keys: If provided, keys to filter examples to.
            vectorstore_kwargs: Extra arguments passed to similarity_search function
                of the `VectorStore`.
            vectorstore_cls_kwargs: optional kwargs containing url for vector store

        Returns:
            The ExampleSelector instantiated, backed by a vector store.
        

        中文翻译:
        使用示例列表和嵌入创建 k-shot 示例选择器。
        根据最大边际相关性动态重新排列示例。
        参数：
            示例：提示中使用的示例列表。
            embeddings：初始化的嵌入 API 接口，例如OpenAIEmbeddings()。
            vectorstore_cls：矢量存储数据库接口类，例如费斯。
            k：要选择的示例数。
            fetch_k：要获取并传递给 MMR 算法的“文档”对象的数量。
            input_keys：如果提供，则搜索基于输入变量
                而不是所有变量。
            example_keys：如果提供，则用于过滤示例的键。
            vectorstore_kwargs：传递给相似度搜索函数的额外参数
                “VectorStore”的。
            vectorstore_cls_kwargs：包含矢量存储 url 的可选 kwargs
        返回：
            由向量存储支持的示例选择器实例化。"""
        string_examples = [cls._example_to_text(eg, input_keys) for eg in examples]
        vectorstore = vectorstore_cls.from_texts(
            string_examples, embeddings, metadatas=examples, **vectorstore_cls_kwargs
        )
        return cls(
            vectorstore=vectorstore,
            k=k,
            fetch_k=fetch_k,
            input_keys=input_keys,
            example_keys=example_keys,
            vectorstore_kwargs=vectorstore_kwargs,
        )

    @classmethod
    async def afrom_examples(
        cls,
        examples: list[dict],
        embeddings: Embeddings,
        vectorstore_cls: type[VectorStore],
        *,
        k: int = 4,
        input_keys: list[str] | None = None,
        fetch_k: int = 20,
        example_keys: list[str] | None = None,
        vectorstore_kwargs: dict | None = None,
        **vectorstore_cls_kwargs: Any,
    ) -> MaxMarginalRelevanceExampleSelector:
        """Create k-shot example selector using example list and embeddings.

        Reshuffles examples dynamically based on Max Marginal Relevance.

        Args:
            examples: List of examples to use in the prompt.
            embeddings: An initialized embedding API interface, e.g. OpenAIEmbeddings().
            vectorstore_cls: A vector store DB interface class, e.g. FAISS.
            k: Number of examples to select.
            fetch_k: Number of `Document` objects to fetch to pass to MMR algorithm.
            input_keys: If provided, the search is based on the input variables
                instead of all variables.
            example_keys: If provided, keys to filter examples to.
            vectorstore_kwargs: Extra arguments passed to similarity_search function
                of the `VectorStore`.
            vectorstore_cls_kwargs: optional kwargs containing url for vector store

        Returns:
            The ExampleSelector instantiated, backed by a vector store.
        

        中文翻译:
        使用示例列表和嵌入创建 k-shot 示例选择器。
        根据最大边际相关性动态重新排列示例。
        参数：
            示例：提示中使用的示例列表。
            embeddings：初始化的嵌入 API 接口，例如OpenAIEmbeddings()。
            vectorstore_cls：矢量存储数据库接口类，例如费斯。
            k：要选择的示例数。
            fetch_k：要获取并传递给 MMR 算法的“文档”对象的数量。
            input_keys：如果提供，则搜索基于输入变量
                而不是所有变量。
            example_keys：如果提供，则用于过滤示例的键。
            vectorstore_kwargs：传递给相似度搜索函数的额外参数
                “VectorStore”的。
            vectorstore_cls_kwargs：包含矢量存储 url 的可选 kwargs
        返回：
            由向量存储支持的示例选择器实例化。"""
        string_examples = [cls._example_to_text(eg, input_keys) for eg in examples]
        vectorstore = await vectorstore_cls.afrom_texts(
            string_examples, embeddings, metadatas=examples, **vectorstore_cls_kwargs
        )
        return cls(
            vectorstore=vectorstore,
            k=k,
            fetch_k=fetch_k,
            input_keys=input_keys,
            example_keys=example_keys,
            vectorstore_kwargs=vectorstore_kwargs,
        )
