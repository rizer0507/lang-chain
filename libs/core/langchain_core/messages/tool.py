"""Messages for tools.

中文翻译:
工具消息。"""

import json
from typing import Any, Literal, cast, overload
from uuid import UUID

from pydantic import Field, model_validator
from typing_extensions import NotRequired, TypedDict, override

from langchain_core.messages import content as types
from langchain_core.messages.base import BaseMessage, BaseMessageChunk, merge_content
from langchain_core.messages.content import InvalidToolCall
from langchain_core.utils._merge import merge_dicts, merge_obj


class ToolOutputMixin:
    """Mixin for objects that tools can return directly.

    If a custom BaseTool is invoked with a `ToolCall` and the output of custom code is
    not an instance of `ToolOutputMixin`, the output will automatically be coerced to
    a string and wrapped in a `ToolMessage`.

    

    中文翻译:
    Mixin 用于工具可以直接返回的对象。
    如果使用“ToolCall”调用自定义 BaseTool 并且自定义代码的输出为
    不是`ToolOutputMixin`的实例，输出将自动被强制为
    一个字符串并包装在“ToolMessage”中。"""


class ToolMessage(BaseMessage, ToolOutputMixin):
    """Message for passing the result of executing a tool back to a model.

    `ToolMessage` objects contain the result of a tool invocation. Typically, the result
    is encoded inside the `content` field.

    `tool_call_id` is used to associate the tool call request with the tool call
    response. Useful in situations where a chat model is able to request multiple tool
    calls in parallel.

    Example:
        A `ToolMessage` representing a result of `42` from a tool call with id

        ```python
        from langchain_core.messages import ToolMessage

        ToolMessage(content="42", tool_call_id="call_Jja7J89XsjrOLA5r!MEOW!SL")
        ```

    Example:
        A `ToolMessage` where only part of the tool output is sent to the model
        and the full output is passed in to artifact.

        ```python
        from langchain_core.messages import ToolMessage

        tool_output = {
            "stdout": "From the graph we can see that the correlation between "
            "x and y is ...",
            "stderr": None,
            "artifacts": {"type": "image", "base64_data": "/9j/4gIcSU..."},
        }

        ToolMessage(
            content=tool_output["stdout"],
            artifact=tool_output,
            tool_call_id="call_Jja7J89XsjrOLA5r!MEOW!SL",
        )
        ```
    

    中文翻译:
    用于将执行工具的结果传递回模型的消息。
    “ToolMessage”对象包含工具调用的结果。通常，结果
    被编码在“content”字段内。
    `tool_call_id` 用于将工具调用请求与工具调用关联起来
    回应。在聊天模型能够请求多个工具的情况下很有用
    并行调用。
    示例：
        一个“ToolMessage”，表示来自 id 的工具调用的“42”结果
        ````蟒蛇
        从 langchain_core.messages 导入 ToolMessage
        ToolMessage(content="42", tool_call_id="call_Jja7J89XsjrOLA5r!MEOW!SL")
        ````
    示例：
        “ToolMessage”，仅将工具输出的一部分发送到模型
        并将完整输出传递给工件。
        ````蟒蛇
        从 langchain_core.messages 导入 ToolMessage
        工具输出 = {
            "stdout": "从图中我们可以看出之间的相关性"
            “x 和 y 是...”,
            “标准错误”：无，
            "artifacts": {"type": "image", "base64_data": "/9j/4gIcSU..."},
        }
        工具消息(
            内容=tool_output["stdout"],
            工件=工具输出，
            tool_call_id="call_Jja7J89XsjrOLA5r!MEOW!SL",
        ）
        ````"""

    tool_call_id: str
    """Tool call that this message is responding to.

    中文翻译:
    此消息正在响应的工具调用。"""

    type: Literal["tool"] = "tool"
    """The type of the message (used for serialization).

    中文翻译:
    消息的类型（用于序列化）。"""

    artifact: Any = None
    """Artifact of the Tool execution which is not meant to be sent to the model.

    Should only be specified if it is different from the message content, e.g. if only
    a subset of the full tool output is being passed as message content but the full
    output is needed in other parts of the code.

    

    中文翻译:
    工具执行的工件并不意味着发送到模型。
    仅当与消息内容不同时才应指定，例如如果只是
    完整工具输出的子集作为消息内容传递，但完整的
    代码的其他部分需要输出。"""

    status: Literal["success", "error"] = "success"
    """Status of the tool invocation.

    中文翻译:
    工具调用的状态。"""

    additional_kwargs: dict = Field(default_factory=dict, repr=False)
    """Currently inherited from `BaseMessage`, but not used.

    中文翻译:
    目前继承自`BaseMessage`，但未使用。"""
    response_metadata: dict = Field(default_factory=dict, repr=False)
    """Currently inherited from `BaseMessage`, but not used.

    中文翻译:
    目前继承自`BaseMessage`，但未使用。"""

    @model_validator(mode="before")
    @classmethod
    def coerce_args(cls, values: dict) -> dict:
        """Coerce the model arguments to the correct types.

        Args:
            values: The model arguments.

        

        中文翻译:
        将模型参数强制为正确的类型。
        参数：
            值：模型参数。"""
        content = values["content"]
        if isinstance(content, tuple):
            content = list(content)

        if not isinstance(content, (str, list)):
            try:
                values["content"] = str(content)
            except ValueError as e:
                msg = (
                    "ToolMessage content should be a string or a list of string/dicts. "
                    f"Received:\n\n{content=}\n\n which could not be coerced into a "
                    "string."
                )
                raise ValueError(msg) from e
        elif isinstance(content, list):
            values["content"] = []
            for i, x in enumerate(content):
                if not isinstance(x, (str, dict)):
                    try:
                        values["content"].append(str(x))
                    except ValueError as e:
                        msg = (
                            "ToolMessage content should be a string or a list of "
                            "string/dicts. Received a list but "
                            f"element ToolMessage.content[{i}] is not a dict and could "
                            f"not be coerced to a string.:\n\n{x}"
                        )
                        raise ValueError(msg) from e
                else:
                    values["content"].append(x)

        tool_call_id = values["tool_call_id"]
        if isinstance(tool_call_id, (UUID, int, float)):
            values["tool_call_id"] = str(tool_call_id)
        return values

    @overload
    def __init__(
        self,
        content: str | list[str | dict],
        **kwargs: Any,
    ) -> None: ...

    @overload
    def __init__(
        self,
        content: str | list[str | dict] | None = None,
        content_blocks: list[types.ContentBlock] | None = None,
        **kwargs: Any,
    ) -> None: ...

    def __init__(
        self,
        content: str | list[str | dict] | None = None,
        content_blocks: list[types.ContentBlock] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a `ToolMessage`.

        Specify `content` as positional arg or `content_blocks` for typing.

        Args:
            content: The contents of the message.
            content_blocks: Typed standard content.
            **kwargs: Additional fields.
        

        中文翻译:
        初始化一个“ToolMessage”。
        将 `content` 指定为位置参数或用于输入的 `content_blocks`。
        参数：
            内容：消息的内容。
            content_blocks：键入的标准内容。
            **kwargs：附加字段。"""
        if content_blocks is not None:
            super().__init__(
                content=cast("str | list[str | dict]", content_blocks),
                **kwargs,
            )
        else:
            super().__init__(content=content, **kwargs)


class ToolMessageChunk(ToolMessage, BaseMessageChunk):
    """Tool Message chunk.

    中文翻译:
    工具消息块。"""

    # Ignoring mypy re-assignment here since we're overriding the value
    # 中文: 此处忽略 mypy 重新分配，因为我们要覆盖该值
    # to make sure that the chunk variant can be discriminated from the
    # 中文: 以确保可以将块变体与
    # non-chunk variant.
    # 中文: 非块变体。
    type: Literal["ToolMessageChunk"] = "ToolMessageChunk"  # type: ignore[assignment]

    @override
    def __add__(self, other: Any) -> BaseMessageChunk:  # type: ignore[override]
        if isinstance(other, ToolMessageChunk):
            if self.tool_call_id != other.tool_call_id:
                msg = "Cannot concatenate ToolMessageChunks with different names."
                raise ValueError(msg)

            return self.__class__(
                tool_call_id=self.tool_call_id,
                content=merge_content(self.content, other.content),
                artifact=merge_obj(self.artifact, other.artifact),
                additional_kwargs=merge_dicts(
                    self.additional_kwargs, other.additional_kwargs
                ),
                response_metadata=merge_dicts(
                    self.response_metadata, other.response_metadata
                ),
                id=self.id,
                status=_merge_status(self.status, other.status),
            )

        return super().__add__(other)


class ToolCall(TypedDict):
    """Represents an AI's request to call a tool.

    Example:
        ```python
        {"name": "foo", "args": {"a": 1}, "id": "123"}
        ```

        This represents a request to call the tool named `'foo'` with arguments
        `{"a": 1}` and an identifier of `'123'`.

    

    中文翻译:
    代表AI调用工具的请求。
    示例：
        ````蟒蛇
        {“name”：“foo”，“args”：{“a”：1}，“id”：“123”}
        ````
        这表示使用参数调用名为“foo”的工具的请求
        `{"a": 1}` 和标识符 `'123'`。"""

    name: str
    """The name of the tool to be called.

    中文翻译:
    要调用的工具的名称。"""
    args: dict[str, Any]
    """The arguments to the tool call.

    中文翻译:
    工具调用的参数。"""
    id: str | None
    """An identifier associated with the tool call.

    An identifier is needed to associate a tool call request with a tool
    call result in events when multiple concurrent tool calls are made.

    

    中文翻译:
    与工具调用关联的标识符。
    需要一个标识符来将工具调用请求与工具关联起来
    当进行多个并发工具调用时，调用会导致事件。"""
    type: NotRequired[Literal["tool_call"]]


def tool_call(
    *,
    name: str,
    args: dict[str, Any],
    id: str | None,
) -> ToolCall:
    """Create a tool call.

    Args:
        name: The name of the tool to be called.
        args: The arguments to the tool call.
        id: An identifier associated with the tool call.

    Returns:
        The created tool call.
    

    中文翻译:
    创建工具调用。
    参数：
        name：要调用的工具的名称。
        args：工具调用的参数。
        id：与工具调用关联的标识符。
    返回：
        创建的工具调用。"""
    return ToolCall(name=name, args=args, id=id, type="tool_call")


class ToolCallChunk(TypedDict):
    """A chunk of a tool call (yielded when streaming).

    When merging `ToolCallChunk`s (e.g., via `AIMessageChunk.__add__`),
    all string attributes are concatenated. Chunks are only merged if their
    values of `index` are equal and not None.

    Example:
    ```python
    left_chunks = [ToolCallChunk(name="foo", args='{"a":', index=0)]
    right_chunks = [ToolCallChunk(name=None, args="1}", index=0)]

    (
        AIMessageChunk(content="", tool_call_chunks=left_chunks)
        + AIMessageChunk(content="", tool_call_chunks=right_chunks)
    ).tool_call_chunks == [ToolCallChunk(name="foo", args='{"a":1}', index=0)]
    ```
    

    中文翻译:
    工具调用的一大块（流式传输时产生）。
    合并`ToolCallChunk`时（例如，通过`AIMessageChunk.__add__`），
    所有字符串属性都连接在一起。仅当它们的块被合并时
    `index` 的值相等而不是 None。
    示例：
    ````蟒蛇
    left_chunks = [ToolCallChunk(name="foo", args='{"a":', index=0)]
    right_chunks = [ToolCallChunk(name=None, args="1}", index=0)]
    （
        AIMessageChunk(content="", tool_call_chunks=left_chunks)
        + AIMessageChunk(content="", tool_call_chunks=right_chunks)
    ).tool_call_chunks == [ToolCallChunk(name="foo", args='{"a":1}', index=0)]
    ````"""

    name: str | None
    """The name of the tool to be called.

    中文翻译:
    要调用的工具的名称。"""
    args: str | None
    """The arguments to the tool call.

    中文翻译:
    工具调用的参数。"""
    id: str | None
    """An identifier associated with the tool call.

    中文翻译:
    与工具调用关联的标识符。"""
    index: int | None
    """The index of the tool call in a sequence.

    中文翻译:
    序列中工具调用的索引。"""
    type: NotRequired[Literal["tool_call_chunk"]]


def tool_call_chunk(
    *,
    name: str | None = None,
    args: str | None = None,
    id: str | None = None,
    index: int | None = None,
) -> ToolCallChunk:
    """Create a tool call chunk.

    Args:
        name: The name of the tool to be called.
        args: The arguments to the tool call.
        id: An identifier associated with the tool call.
        index: The index of the tool call in a sequence.

    Returns:
        The created tool call chunk.
    

    中文翻译:
    创建一个工具调用块。
    参数：
        name：要调用的工具的名称。
        args：工具调用的参数。
        id：与工具调用关联的标识符。
        索引：工具调用在序列中的索引。
    返回：
        创建的工具调用 chunk。"""
    return ToolCallChunk(
        name=name, args=args, id=id, index=index, type="tool_call_chunk"
    )


def invalid_tool_call(
    *,
    name: str | None = None,
    args: str | None = None,
    id: str | None = None,
    error: str | None = None,
) -> InvalidToolCall:
    """Create an invalid tool call.

    Args:
        name: The name of the tool to be called.
        args: The arguments to the tool call.
        id: An identifier associated with the tool call.
        error: An error message associated with the tool call.

    Returns:
        The created invalid tool call.
    

    中文翻译:
    创建无效的工具调用。
    参数：
        name：要调用的工具的名称。
        args：工具调用的参数。
        id：与工具调用关联的标识符。
        error：与工具调用相关的错误消息。
    返回：
        创建的工具调用无效。"""
    return InvalidToolCall(
        name=name, args=args, id=id, error=error, type="invalid_tool_call"
    )


def default_tool_parser(
    raw_tool_calls: list[dict],
) -> tuple[list[ToolCall], list[InvalidToolCall]]:
    """Best-effort parsing of tools.

    Args:
        raw_tool_calls: List of raw tool call dicts to parse.

    Returns:
        A list of tool calls and invalid tool calls.
    

    中文翻译:
    尽力解析工具。
    参数：
        raw_tool_calls：要解析的原始工具调用字典列表。
    返回：
        工具调用和无效工具调用的列表。"""
    tool_calls = []
    invalid_tool_calls = []
    for raw_tool_call in raw_tool_calls:
        if "function" not in raw_tool_call:
            continue
        function_name = raw_tool_call["function"]["name"]
        try:
            function_args = json.loads(raw_tool_call["function"]["arguments"])
            parsed = tool_call(
                name=function_name or "",
                args=function_args or {},
                id=raw_tool_call.get("id"),
            )
            tool_calls.append(parsed)
        except json.JSONDecodeError:
            invalid_tool_calls.append(
                invalid_tool_call(
                    name=function_name,
                    args=raw_tool_call["function"]["arguments"],
                    id=raw_tool_call.get("id"),
                    error=None,
                )
            )
    return tool_calls, invalid_tool_calls


def default_tool_chunk_parser(raw_tool_calls: list[dict]) -> list[ToolCallChunk]:
    """Best-effort parsing of tool chunks.

    Args:
        raw_tool_calls: List of raw tool call dicts to parse.

    Returns:
        List of parsed ToolCallChunk objects.
    

    中文翻译:
    尽力解析工具块。
    参数：
        raw_tool_calls：要解析的原始工具调用字典列表。
    返回：
        已解析的 ToolCallChunk 对象的列表。"""
    tool_call_chunks = []
    for tool_call in raw_tool_calls:
        if "function" not in tool_call:
            function_args = None
            function_name = None
        else:
            function_args = tool_call["function"]["arguments"]
            function_name = tool_call["function"]["name"]
        parsed = tool_call_chunk(
            name=function_name,
            args=function_args,
            id=tool_call.get("id"),
            index=tool_call.get("index"),
        )
        tool_call_chunks.append(parsed)
    return tool_call_chunks


def _merge_status(
    left: Literal["success", "error"], right: Literal["success", "error"]
) -> Literal["success", "error"]:
    return "error" if "error" in {left, right} else "success"
