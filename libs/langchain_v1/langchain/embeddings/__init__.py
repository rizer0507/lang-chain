"""Embeddings models.

!!! warning "Modules moved"
    With the release of `langchain 1.0.0`, several embeddings modules were moved to
    `langchain-classic`, such as `CacheBackedEmbeddings` and all community
    embeddings. See [list](https://github.com/langchain-ai/langchain/blob/bdf1cd383ce36dc18381a3bf3fb0a579337a32b5/libs/langchain/langchain/embeddings/__init__.py)
    of moved modules to inform your migration.

中文翻译:
嵌入模型。
！！！警告“模块已移动”
    随着“langchain 1.0.0”的发布，几个嵌入模块被移至
    “langchain-classic”，例如“CacheBackedEmbeddings”和所有社区
    嵌入。请参阅[列表](https://github.com/langchain-ai/langchain/blob/bdf1cd383ce36dc18381a3bf3fb0a579337a32b5/libs/langchain/langchain/embeddings/__init__.py)
    已移动的模块以通知您的迁移。
"""

from langchain_core.embeddings import Embeddings

from langchain.embeddings.base import init_embeddings

__all__ = [
    "Embeddings",
    "init_embeddings",
]
