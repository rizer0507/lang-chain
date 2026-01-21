"""Abstract interface for document loader implementations.

中文翻译:
文档加载器实现的抽象接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from langchain_core.runnables import run_in_executor

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from langchain_text_splitters import TextSplitter

    from langchain_core.documents import Document
    from langchain_core.documents.base import Blob

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    _HAS_TEXT_SPLITTERS = True
except ImportError:
    _HAS_TEXT_SPLITTERS = False


class BaseLoader(ABC):  # noqa: B024
    """Interface for Document Loader.

    Implementations should implement the lazy-loading method using generators
    to avoid loading all documents into memory at once.

    `load` is provided just for user convenience and should not be overridden.
    

    中文翻译:
    文档加载器的接口。
    实现应该使用生成器实现延迟加载方法
    以避免一次将所有文档加载到内存中。
    提供“load”只是为了方便用户，不应被覆盖。"""

    # Sub-classes should not implement this method directly. Instead, they
    # 中文: 子类不应直接实现此方法。相反，他们
    # should implement the lazy load method.
    # 中文: 应该实现延迟加载方法。
    def load(self) -> list[Document]:
        """Load data into `Document` objects.

        Returns:
            The documents.
        

        中文翻译:
        将数据加载到“Doc​​ument”对象中。
        返回：
            文件。"""
        return list(self.lazy_load())

    async def aload(self) -> list[Document]:
        """Load data into `Document` objects.

        Returns:
            The documents.
        

        中文翻译:
        将数据加载到“Doc​​ument”对象中。
        返回：
            文件。"""
        return [document async for document in self.alazy_load()]

    def load_and_split(
        self, text_splitter: TextSplitter | None = None
    ) -> list[Document]:
        """Load `Document` and split into chunks. Chunks are returned as `Document`.

        !!! danger

            Do not override this method. It should be considered to be deprecated!

        Args:
            text_splitter: `TextSplitter` instance to use for splitting documents.
                Defaults to `RecursiveCharacterTextSplitter`.

        Raises:
            ImportError: If `langchain-text-splitters` is not installed
                and no `text_splitter` is provided.

        Returns:
            List of `Document`.
        

        中文翻译:
        加载“文档”并分成块。块作为“文档”返回。
        ！！！危险
            不要重写此方法。应该认为是废弃的！
        参数：
            text_splitter：用于分割文档的“TextSplitter”实例。
                默认为“RecursiveCharacterTextSplitter”。
        加薪：
            ImportError：如果未安装“langchain-text-splitters”
                并且没有提供“text_splitter”。
        返回：
            “文档”列表。"""
        if text_splitter is None:
            if not _HAS_TEXT_SPLITTERS:
                msg = (
                    "Unable to import from langchain_text_splitters. Please specify "
                    "text_splitter or install langchain_text_splitters with "
                    "`pip install -U langchain-text-splitters`."
                )
                raise ImportError(msg)

            text_splitter_: TextSplitter = RecursiveCharacterTextSplitter()
        else:
            text_splitter_ = text_splitter
        docs = self.load()
        return text_splitter_.split_documents(docs)

    # Attention: This method will be upgraded into an abstractmethod once it's
    # 中文: 注意：该方法一旦被调用将会升级为抽象方法
    #            implemented in all the existing subclasses.
    #            中文: 在所有现有子类中实现。
    def lazy_load(self) -> Iterator[Document]:
        """A lazy loader for `Document`.

        Yields:
            The `Document` objects.
        

        中文翻译:
        “文档”的惰性加载器。
        产量：
            “文档”对象。"""
        if type(self).load != BaseLoader.load:
            return iter(self.load())
        msg = f"{self.__class__.__name__} does not implement lazy_load()"
        raise NotImplementedError(msg)

    async def alazy_load(self) -> AsyncIterator[Document]:
        """A lazy loader for `Document`.

        Yields:
            The `Document` objects.
        

        中文翻译:
        “文档”的惰性加载器。
        产量：
            “文档”对象。"""
        iterator = await run_in_executor(None, self.lazy_load)
        done = object()
        while True:
            doc = await run_in_executor(None, next, iterator, done)
            if doc is done:
                break
            yield doc  # type: ignore[misc]


class BaseBlobParser(ABC):
    """Abstract interface for blob parsers.

    A blob parser provides a way to parse raw data stored in a blob into one
    or more `Document` objects.

    The parser can be composed with blob loaders, making it easy to reuse
    a parser independent of how the blob was originally loaded.
    

    中文翻译:
    Blob 解析器的抽象接口。
    Blob 解析器提供了一种将存储在 Blob 中的原始数据解析为一个数据的方法
    或更多“文档”对象。
    解析器可以与blob加载器组合，使其易于重用
    独立于 blob 最初加载方式的解析器。"""

    @abstractmethod
    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazy parsing interface.

        Subclasses are required to implement this method.

        Args:
            blob: `Blob` instance

        Returns:
            Generator of `Document` objects
        

        中文翻译:
        惰性解析接口。
        需要子类来实现此方法。
        参数：
            blob：“Blob”实例
        返回：
            “文档”对象的生成器"""

    def parse(self, blob: Blob) -> list[Document]:
        """Eagerly parse the blob into a `Document` or list of `Document` objects.

        This is a convenience method for interactive development environment.

        Production applications should favor the `lazy_parse` method instead.

        Subclasses should generally not over-ride this parse method.

        Args:
            blob: `Blob` instance

        Returns:
            List of `Document` objects
        

        中文翻译:
        急切地将 blob 解析为“Document”或“Document”对象列表。
        这是交互式开发环境的一种便捷方法。
        生产应用程序应该倾向于使用“lazy_parse”方法。
        子类通常不应覆盖此解析方法。
        参数：
            blob：“Blob”实例
        返回：
            “文档”对象列表"""
        return list(self.lazy_parse(blob))
