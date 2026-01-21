"""**Embeddings** interface.

中文翻译:
**嵌入**接口。"""

from abc import ABC, abstractmethod

from langchain_core.runnables.config import run_in_executor


class Embeddings(ABC):
    """Interface for embedding models.

    This is an interface meant for implementing text embedding models.

    Text embedding models are used to map text to a vector (a point in n-dimensional
    space).

    Texts that are similar will usually be mapped to points that are close to each
    other in this space. The exact details of what's considered "similar" and how
    "distance" is measured in this space are dependent on the specific embedding model.

    This abstraction contains a method for embedding a list of documents and a method
    for embedding a query text. The embedding of a query text is expected to be a single
    vector, while the embedding of a list of documents is expected to be a list of
    vectors.

    Usually the query embedding is identical to the document embedding, but the
    abstraction allows treating them independently.

    In addition to the synchronous methods, this interface also provides asynchronous
    versions of the methods.

    By default, the asynchronous methods are implemented using the synchronous methods;
    however, implementations may choose to override the asynchronous methods with
    an async native implementation for performance reasons.
    

    中文翻译:
    嵌入模型的接口。
    这是一个用于实现文本嵌入模型的接口。
    文本嵌入模型用于将文本映射到向量（n 维中的点）
    空间）。
    相似的文本通常会被映射到彼此接近的点
    这个空间中的其他人。什么被视为“相似”以及如何被视为“相似”的确切细节
    在这个空间中测量的“距离”取决于具体的嵌入模型。
    该抽象包含一个用于嵌入文档列表的方法和一个方法
    用于嵌入查询文本。查询文本的嵌入预计是单个
    向量，而文档列表的嵌入预计是一个列表
    向量。
    通常查询嵌入与文档嵌入相同，但是
    抽象允许独立地处理它们。
    除了同步方法外，该接口还提供异步方法
    方法的版本。
    默认情况下，异步方法是使用同步方法实现的；
    然而，实现可以选择用以下方式覆盖异步方法
    出于性能原因，采用异步本机实现。"""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed search docs.

        Args:
            texts: List of text to embed.

        Returns:
            List of embeddings.
        

        中文翻译:
        嵌入搜索文档。
        参数：
            texts：要嵌入的文本列表。
        返回：
            嵌入列表。"""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed query text.

        Args:
            text: Text to embed.

        Returns:
            Embedding.
        

        中文翻译:
        嵌入查询文本。
        参数：
            文本：要嵌入的文本。
        返回：
            嵌入。"""

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Asynchronous Embed search docs.

        Args:
            texts: List of text to embed.

        Returns:
            List of embeddings.
        

        中文翻译:
        异步嵌入搜索文档。
        参数：
            texts：要嵌入的文本列表。
        返回：
            嵌入列表。"""
        return await run_in_executor(None, self.embed_documents, texts)

    async def aembed_query(self, text: str) -> list[float]:
        """Asynchronous Embed query text.

        Args:
            text: Text to embed.

        Returns:
            Embedding.
        

        中文翻译:
        异步嵌入查询文本。
        参数：
            文本：要嵌入的文本。
        返回：
            嵌入。"""
        return await run_in_executor(None, self.embed_query, text)
