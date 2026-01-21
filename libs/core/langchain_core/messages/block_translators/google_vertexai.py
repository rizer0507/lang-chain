"""Derivations of standard content blocks from Google (VertexAI) content.

中文翻译:
来自 Google (VertexAI) 内容的标准内容块的派生。"""

from langchain_core.messages.block_translators.google_genai import (
    translate_content,
    translate_content_chunk,
)


def _register_google_vertexai_translator() -> None:
    """Register the Google (VertexAI) translator with the central registry.

    Run automatically when the module is imported.
    

    中文翻译:
    在中央注册表中注册 Google (VertexAI) 翻译器。
    导入模块时自动运行。"""
    from langchain_core.messages.block_translators import (  # noqa: PLC0415
        register_translator,
    )

    register_translator("google_vertexai", translate_content, translate_content_chunk)


_register_google_vertexai_translator()
