"""Schema for Blobs and Blob Loaders.

The goal is to facilitate decoupling of content loading from content parsing code.

In addition, content loading code should provide a lazy loading interface by default.

中文翻译:
Blob 和 Blob 加载器的架构。
目标是促进内容加载与内容解析代码的解耦。
此外，内容加载代码应该默认提供延迟加载接口。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

# Re-export Blob and PathLike for backwards compatibility
# 中文: 重新导出 Blob 和 PathLike 以实现向后兼容性
from langchain_core.documents.base import Blob, PathLike

if TYPE_CHECKING:
    from collections.abc import Iterable


class BlobLoader(ABC):
    """Abstract interface for blob loaders implementation.

    Implementer should be able to load raw content from a storage system according
    to some criteria and return the raw content lazily as a stream of blobs.
    

    中文翻译:
    Blob 加载器实现的抽象接口。
    实施者应该能够根据存储系统加载原始内容
    符合某些标准，并将原始内容作为 blob 流延迟返回。"""

    @abstractmethod
    def yield_blobs(
        self,
    ) -> Iterable[Blob]:
        """A lazy loader for raw data represented by LangChain's `Blob` object.

        Returns:
            A generator over blobs
        

        中文翻译:
        由 LangChain 的“Blob”对象表示的原始数据的惰性加载器。
        返回：
            斑点生成器"""


# Re-export Blob and Pathlike for backwards compatibility
# 中文: 重新导出 Blob 和 Pathlike 以实现向后兼容性
__all__ = ["Blob", "BlobLoader", "PathLike"]
