"""Function Message.

中文翻译:
功能消息。"""

from typing import Any, Literal

from typing_extensions import override

from langchain_core.messages.base import (
    BaseMessage,
    BaseMessageChunk,
    merge_content,
)
from langchain_core.utils._merge import merge_dicts


class FunctionMessage(BaseMessage):
    """Message for passing the result of executing a tool back to a model.

    `FunctionMessage` are an older version of the `ToolMessage` schema, and
    do not contain the `tool_call_id` field.

    The `tool_call_id` field is used to associate the tool call request with the
    tool call response. Useful in situations where a chat model is able
    to request multiple tool calls in parallel.

    

    中文翻译:
    用于将执行工具的结果传递回模型的消息。
    `FunctionMessage` 是 `ToolMessage` 模式的旧版本，并且
    不包含“tool_call_id”字段。
    `tool_call_id`字段用于将工具调用请求与
    工具调用响应。在聊天模型能够的情况下很有用
    请求并行调用多个工具。"""

    name: str
    """The name of the function that was executed.

    中文翻译:
    被执行的函数的名称。"""

    type: Literal["function"] = "function"
    """The type of the message (used for serialization).

    中文翻译:
    消息的类型（用于序列化）。"""


class FunctionMessageChunk(FunctionMessage, BaseMessageChunk):
    """Function Message chunk.

    中文翻译:
    功能消息块。"""

    # Ignoring mypy re-assignment here since we're overriding the value
    # 中文: 此处忽略 mypy 重新分配，因为我们要覆盖该值
    # to make sure that the chunk variant can be discriminated from the
    # 中文: 以确保可以将块变体与
    # non-chunk variant.
    # 中文: 非块变体。
    type: Literal["FunctionMessageChunk"] = "FunctionMessageChunk"  # type: ignore[assignment]
    """The type of the message (used for serialization).

    中文翻译:
    消息的类型（用于序列化）。"""

    @override
    def __add__(self, other: Any) -> BaseMessageChunk:  # type: ignore[override]
        if isinstance(other, FunctionMessageChunk):
            if self.name != other.name:
                msg = "Cannot concatenate FunctionMessageChunks with different names."
                raise ValueError(msg)

            return self.__class__(
                name=self.name,
                content=merge_content(self.content, other.content),
                additional_kwargs=merge_dicts(
                    self.additional_kwargs, other.additional_kwargs
                ),
                response_metadata=merge_dicts(
                    self.response_metadata, other.response_metadata
                ),
                id=self.id,
            )

        return super().__add__(other)
