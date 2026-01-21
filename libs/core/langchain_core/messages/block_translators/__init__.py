"""Derivations of standard content blocks from provider content.

`AIMessage` will first attempt to use a provider-specific translator if
`model_provider` is set in `response_metadata` on the message. Consequently, each
provider translator must handle all possible content response types from the provider,
including text.

If no provider is set, or if the provider does not have a registered translator,
`AIMessage` will fall back to best-effort parsing of the content into blocks using
the implementation in `BaseMessage`.

中文翻译:
从提供者内容衍生出标准内容块。
如果满足以下条件，“AIMessage”将首先尝试使用特定于提供者的翻译器：
“model_provider”在消息的“response_metadata”中设置。因此，每个
提供者翻译者必须处理来自提供者的所有可能的内容响应类型，
包括文字。
如果未设置提供商，或者提供商没有注册翻译人员，
`AIMessage` 将回退到使用尽力而为的方式将内容解析为块
`BaseMessage` 中的实现。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_core.messages import AIMessage, AIMessageChunk
    from langchain_core.messages import content as types

# Provider to translator mapping
# 中文: 提供者到翻译者的映射
PROVIDER_TRANSLATORS: dict[str, dict[str, Callable[..., list[types.ContentBlock]]]] = {}
"""Map model provider names to translator functions.

The dictionary maps provider names (e.g. `'openai'`, `'anthropic'`) to another
dictionary with two keys:
- `'translate_content'`: Function to translate `AIMessage` content.
- `'translate_content_chunk'`: Function to translate `AIMessageChunk` content.

When calling `content_blocks` on an `AIMessage` or `AIMessageChunk`, if
`model_provider` is set in `response_metadata`, the corresponding translator
functions will be used to parse the content into blocks. Otherwise, best-effort parsing
in `BaseMessage` will be used.

中文翻译:
将模型提供者名称映射到转换器功能。
字典将提供者名称（例如“openai”、“anthropic”）映射到另一个提供者名称
有两个键的字典：
- “translate_content”：翻译“AIMessage”内容的函数。
- “translate_content_chunk”：翻译“AIMessageChunk”内容的函数。
在“AIMessage”或“AIMessageChunk”上调用“content_blocks”时，如果
`model_provider`是在`response_metadata`中设置的，对应的翻译器
函数将用于将内容解析为块。否则，尽力解析
将使用“BaseMessage”中的内容。
"""


def register_translator(
    provider: str,
    translate_content: Callable[[AIMessage], list[types.ContentBlock]],
    translate_content_chunk: Callable[[AIMessageChunk], list[types.ContentBlock]],
) -> None:
    """Register content translators for a provider in `PROVIDER_TRANSLATORS`.

    Args:
        provider: The model provider name (e.g. `'openai'`, `'anthropic'`).
        translate_content: Function to translate `AIMessage` content.
        translate_content_chunk: Function to translate `AIMessageChunk` content.
    

    中文翻译:
    在“PROVIDER_TRANSLATORS”中为提供者注册内容翻译器。
    参数：
        提供者：模型提供者名称（例如“openai”、“anthropic”）。
        translate_content：翻译`AIMessage`内容的函数。
        translate_content_chunk：翻译`AIMessageChunk`内容的函数。"""
    PROVIDER_TRANSLATORS[provider] = {
        "translate_content": translate_content,
        "translate_content_chunk": translate_content_chunk,
    }


def get_translator(
    provider: str,
) -> dict[str, Callable[..., list[types.ContentBlock]]] | None:
    """Get the translator functions for a provider.

    Args:
        provider: The model provider name.

    Returns:
        Dictionary with `'translate_content'` and `'translate_content_chunk'`
        functions, or None if no translator is registered for the provider. In such
        case, best-effort parsing in `BaseMessage` will be used.
    

    中文翻译:
    获取提供商的翻译功能。
    参数：
        提供者：模型提供者名称。
    返回：
        带有“translate_content”和“translate_content_chunk”的字典
        函数，如果没有为提供者注册翻译器，则为“无”。在这样的
        在这种情况下，将使用“BaseMessage”中的尽力解析。"""
    return PROVIDER_TRANSLATORS.get(provider)


def _register_translators() -> None:
    """Register all translators in langchain-core.

    A unit test ensures all modules in `block_translators` are represented here.

    For translators implemented outside langchain-core, they can be registered by
    calling `register_translator` from within the integration package.
    

    中文翻译:
    在 langchain-core 中注册所有翻译器。
    单元测试确保“block_translators”中的所有模块都在这里表示。
    对于在 langchain-core 之外实现的翻译器，可以通过以下方式注册：
    从集成包中调用“register_translator”。"""
    from langchain_core.messages.block_translators.anthropic import (  # noqa: PLC0415
        _register_anthropic_translator,
    )
    from langchain_core.messages.block_translators.bedrock import (  # noqa: PLC0415
        _register_bedrock_translator,
    )
    from langchain_core.messages.block_translators.bedrock_converse import (  # noqa: PLC0415
        _register_bedrock_converse_translator,
    )
    from langchain_core.messages.block_translators.google_genai import (  # noqa: PLC0415
        _register_google_genai_translator,
    )
    from langchain_core.messages.block_translators.google_vertexai import (  # noqa: PLC0415
        _register_google_vertexai_translator,
    )
    from langchain_core.messages.block_translators.groq import (  # noqa: PLC0415
        _register_groq_translator,
    )
    from langchain_core.messages.block_translators.openai import (  # noqa: PLC0415
        _register_openai_translator,
    )

    _register_bedrock_translator()
    _register_bedrock_converse_translator()
    _register_anthropic_translator()
    _register_google_genai_translator()
    _register_google_vertexai_translator()
    _register_groq_translator()
    _register_openai_translator()


_register_translators()
