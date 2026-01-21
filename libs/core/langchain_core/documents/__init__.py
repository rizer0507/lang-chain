"""Documents module for data retrieval and processing workflows.

This module provides core abstractions for handling data in retrieval-augmented
generation (RAG) pipelines, vector stores, and document processing workflows.

!!! warning "Documents vs. message content"
    This module is distinct from `langchain_core.messages.content`, which provides
    multimodal content blocks for **LLM chat I/O** (text, images, audio, etc. within
    messages).

    **Key distinction:**

    - **Documents** (this module): For **data retrieval and processing workflows**
        - Vector stores, retrievers, RAG pipelines
        - Text chunking, embedding, and semantic search
        - Example: Chunks of a PDF stored in a vector database

    - **Content Blocks** (`messages.content`): For **LLM conversational I/O**
        - Multimodal message content sent to/from models
        - Tool calls, reasoning, citations within chat
        - Example: An image sent to a vision model in a chat message (via
            [`ImageContentBlock`][langchain.messages.ImageContentBlock])

    While both can represent similar data types (text, files), they serve different
    architectural purposes in LangChain applications.

中文翻译:
用于数据检索和处理工作流程的文档模块。
该模块提供了处理检索增强数据的核心抽象
生成 (RAG) 管道、向量存储和文档处理工作流程。
!!!警告“文档与消息内容”
    该模块与“langchain_core.messages.content”不同，后者提供
    **LLM 聊天 I/O** 的多模式内容块（文本、图像、音频等）
    消息）。
    **主要区别：**
    - **文档**（此模块）：用于**数据检索和处理工作流程**
        - 矢量存储、检索器、RAG 管道
        - 文本分块、嵌入和语义搜索
        - 示例：存储在矢量数据库中的 PDF 块
    - **内容块**（`messages.content`）：用于**LLM会话I/O**
        - 发送至模型/从模型发送的多模式消息内容
        - 聊天中的工具调用、推理、引用
        - 示例：在聊天消息中发送到视觉模型的图像（通过
            [`ImageContentBlock`][langchain.messages.ImageContentBlock])
    虽然两者都可以表示相似的数据类型（文本、文件），但它们提供不同的服务
    LangChain 应用程序中的架构目的。
"""

from typing import TYPE_CHECKING

from langchain_core._import_utils import import_attr

if TYPE_CHECKING:
    from .base import Document
    from .compressor import BaseDocumentCompressor
    from .transformers import BaseDocumentTransformer

__all__ = ("BaseDocumentCompressor", "BaseDocumentTransformer", "Document")

_dynamic_imports = {
    "Document": "base",
    "BaseDocumentCompressor": "compressor",
    "BaseDocumentTransformer": "transformers",
}


def __getattr__(attr_name: str) -> object:
    module_name = _dynamic_imports.get(attr_name)
    result = import_attr(attr_name, module_name, __spec__.parent)
    globals()[attr_name] = result
    return result


def __dir__() -> list[str]:
    return list(__all__)
