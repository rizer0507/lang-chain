"""Derivations of standard content blocks from Bedrock content.

中文翻译:
从基岩内容衍生出标准内容块。"""

from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.messages import content as types
from langchain_core.messages.block_translators.anthropic import (
    _convert_to_v1_from_anthropic,
)


def _convert_to_v1_from_bedrock(message: AIMessage) -> list[types.ContentBlock]:
    """Convert bedrock message content to v1 format.

    中文翻译:
    将 bedrock 消息内容转换为 v1 格式。"""
    out = _convert_to_v1_from_anthropic(message)

    content_tool_call_ids = {
        block.get("id")
        for block in out
        if isinstance(block, dict) and block.get("type") == "tool_call"
    }
    for tool_call in message.tool_calls:
        if (id_ := tool_call.get("id")) and id_ not in content_tool_call_ids:
            tool_call_block: types.ToolCall = {
                "type": "tool_call",
                "id": id_,
                "name": tool_call["name"],
                "args": tool_call["args"],
            }
            if "index" in tool_call:
                tool_call_block["index"] = tool_call["index"]  # type: ignore[typeddict-item]
            if "extras" in tool_call:
                tool_call_block["extras"] = tool_call["extras"]  # type: ignore[typeddict-item]
            out.append(tool_call_block)
    return out


def _convert_to_v1_from_bedrock_chunk(
    message: AIMessageChunk,
) -> list[types.ContentBlock]:
    """Convert bedrock message chunk content to v1 format.

    中文翻译:
    将基岩消息块内容转换为 v1 格式。"""
    if (
        message.content == ""
        and not message.additional_kwargs
        and not message.tool_calls
    ):
        # Bedrock outputs multiple chunks containing response metadata
        # 中文: Bedrock 输出多个包含响应元数据的块
        return []

    out = _convert_to_v1_from_anthropic(message)

    if (
        message.tool_call_chunks
        and not message.content
        and message.chunk_position != "last"  # keep tool_calls if aggregated
    ):
        for tool_call_chunk in message.tool_call_chunks:
            tc: types.ToolCallChunk = {
                "type": "tool_call_chunk",
                "id": tool_call_chunk.get("id"),
                "name": tool_call_chunk.get("name"),
                "args": tool_call_chunk.get("args"),
            }
            if (idx := tool_call_chunk.get("index")) is not None:
                tc["index"] = idx
            out.append(tc)
    return out


def translate_content(message: AIMessage) -> list[types.ContentBlock]:
    """Derive standard content blocks from a message with Bedrock content.

    Args:
        message: The message to translate.

    Returns:
        The derived content blocks.
    

    中文翻译:
    从包含基岩内容的消息中派生标准内容块。
    参数：
        message：要翻译的消息。
    返回：
        派生的内容块。"""
    if "claude" not in message.response_metadata.get("model_name", "").lower():
        raise NotImplementedError  # fall back to best-effort parsing
    return _convert_to_v1_from_bedrock(message)


def translate_content_chunk(message: AIMessageChunk) -> list[types.ContentBlock]:
    """Derive standard content blocks from a message chunk with Bedrock content.

    Args:
        message: The message chunk to translate.

    Returns:
        The derived content blocks.
    

    中文翻译:
    从包含基岩内容的消息块中派生标准内容块。
    参数：
        message：要翻译的消息块。
    返回：
        派生的内容块。"""
    # TODO: add model_name to all Bedrock chunks and update core merging logic
    # to not append during aggregation. Then raise NotImplementedError here if
    # 中文: 在聚合期间不追加。然后在此处引发 NotImplementedError 如果
    # not an Anthropic model to fall back to best-effort parsing.
    # 中文: 不是一个可以回退到尽力解析的人择模型。
    return _convert_to_v1_from_bedrock_chunk(message)


def _register_bedrock_translator() -> None:
    """Register the bedrock translator with the central registry.

    Run automatically when the module is imported.
    

    中文翻译:
    在中央注册表中注册基岩翻译器。
    导入模块时自动运行。"""
    from langchain_core.messages.block_translators import (  # noqa: PLC0415
        register_translator,
    )

    register_translator("bedrock", translate_content, translate_content_chunk)


_register_bedrock_translator()
