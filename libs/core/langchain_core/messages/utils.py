"""Module contains utility functions for working with messages.

Some examples of what you can do with these functions include:

* Convert messages to strings (serialization)
* Convert messages from dicts to Message objects (deserialization)
* Filter messages from a list of messages based on name, type or id etc.

中文翻译:
模块包含用于处理消息的实用函数。
您可以使用这些函数执行哪些操作的一些示例包括：
* 将消息转换为字符串（序列化）
* 将消息从字典转换为消息对象（反序列化）
* 根据名称、类型或 ID 等从消息列表中过滤消息。
"""

from __future__ import annotations

import base64
import inspect
import json
import logging
import math
from collections.abc import Callable, Iterable, Sequence
from functools import partial, wraps
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Concatenate,
    Literal,
    ParamSpec,
    Protocol,
    TypeVar,
    cast,
    overload,
)

from pydantic import Discriminator, Field, Tag

from langchain_core.exceptions import ErrorCode, create_message
from langchain_core.messages.ai import AIMessage, AIMessageChunk
from langchain_core.messages.base import BaseMessage, BaseMessageChunk
from langchain_core.messages.block_translators.openai import (
    convert_to_openai_data_block,
)
from langchain_core.messages.chat import ChatMessage, ChatMessageChunk
from langchain_core.messages.content import (
    is_data_content_block,
)
from langchain_core.messages.function import FunctionMessage, FunctionMessageChunk
from langchain_core.messages.human import HumanMessage, HumanMessageChunk
from langchain_core.messages.modifier import RemoveMessage
from langchain_core.messages.system import SystemMessage, SystemMessageChunk
from langchain_core.messages.tool import ToolCall, ToolMessage, ToolMessageChunk

if TYPE_CHECKING:
    from langchain_core.language_models import BaseLanguageModel
    from langchain_core.prompt_values import PromptValue
    from langchain_core.runnables.base import Runnable

try:
    from langchain_text_splitters import TextSplitter

    _HAS_LANGCHAIN_TEXT_SPLITTERS = True
except ImportError:
    _HAS_LANGCHAIN_TEXT_SPLITTERS = False

logger = logging.getLogger(__name__)


def _get_type(v: Any) -> str:
    """Get the type associated with the object for serialization purposes.

    中文翻译:
    获取与对象关联的类型以进行序列化。"""
    if isinstance(v, dict) and "type" in v:
        result = v["type"]
    elif hasattr(v, "type"):
        result = v.type
    else:
        msg = (
            f"Expected either a dictionary with a 'type' key or an object "
            f"with a 'type' attribute. Instead got type {type(v)}."
        )
        raise TypeError(msg)
    if not isinstance(result, str):
        msg = f"Expected 'type' to be a str, got {type(result).__name__}"
        raise TypeError(msg)
    return result


AnyMessage = Annotated[
    Annotated[AIMessage, Tag(tag="ai")]
    | Annotated[HumanMessage, Tag(tag="human")]
    | Annotated[ChatMessage, Tag(tag="chat")]
    | Annotated[SystemMessage, Tag(tag="system")]
    | Annotated[FunctionMessage, Tag(tag="function")]
    | Annotated[ToolMessage, Tag(tag="tool")]
    | Annotated[AIMessageChunk, Tag(tag="AIMessageChunk")]
    | Annotated[HumanMessageChunk, Tag(tag="HumanMessageChunk")]
    | Annotated[ChatMessageChunk, Tag(tag="ChatMessageChunk")]
    | Annotated[SystemMessageChunk, Tag(tag="SystemMessageChunk")]
    | Annotated[FunctionMessageChunk, Tag(tag="FunctionMessageChunk")]
    | Annotated[ToolMessageChunk, Tag(tag="ToolMessageChunk")],
    Field(discriminator=Discriminator(_get_type)),
]
"""A type representing any defined `Message` or `MessageChunk` type.

中文翻译:
表示任何定义的“Message”或“MessageChunk”类型的类型。"""


def get_buffer_string(
    messages: Sequence[BaseMessage], human_prefix: str = "Human", ai_prefix: str = "AI"
) -> str:
    r"""Convert a sequence of messages to strings and concatenate them into one string.

    Args:
        messages: Messages to be converted to strings.
        human_prefix: The prefix to prepend to contents of `HumanMessage`s.
        ai_prefix: The prefix to prepend to contents of `AIMessage`.

    Returns:
        A single string concatenation of all input messages.

    Raises:
        ValueError: If an unsupported message type is encountered.

    Note:
        If a message is an `AIMessage` and contains both tool calls under `tool_calls`
        and a function call under `additional_kwargs["function_call"]`, only the tool
        calls will be appended to the string representation.

    Example:
        ```python
        from langchain_core import AIMessage, HumanMessage

        messages = [
            HumanMessage(content="Hi, how are you?"),
            AIMessage(content="Good, how are you?"),
        ]
        get_buffer_string(messages)
        # -> "Human: Hi, how are you?\nAI: Good, how are you?"
        # 中文: ->“人类：嗨，你好吗？\n人工智能：很好，你好吗？”
        ```
    

中文翻译:
将一系列消息转换为字符串并将它们连接成一个字符串。
    参数：
        messages：要转换为字符串的消息。
        human_prefix：添加到“HumanMessage”内容之前的前缀。
        ai_prefix：添加到“AIMessage”内容之前的前缀。
    返回：
        所有输入消息的单个字符串串联。
    加薪：
        ValueError：如果遇到不支持的消息类型。
    注意：
        如果消息是“AIMessage”并且包含“tool_calls”下的两个工具调用
        以及 `additional_kwargs["function_call"]` 下的函数调用，仅工具
        调用将附加到字符串表示形式中。
    示例：
        ````蟒蛇
        从 langchain_core 导入 AIMessage、HumanMessage
        消息 = [
            HumanMessage(content="嗨，你好吗？"),
            AIMessage(content="很好，你好吗？"),
        ]
        获取缓冲区字符串（消息）
        # -> “人类：嗨，你好吗？\nAI：很好，你好吗？”
        ````"""
    string_messages = []
    for m in messages:
        if isinstance(m, HumanMessage):
            role = human_prefix
        elif isinstance(m, AIMessage):
            role = ai_prefix
        elif isinstance(m, SystemMessage):
            role = "System"
        elif isinstance(m, FunctionMessage):
            role = "Function"
        elif isinstance(m, ToolMessage):
            role = "Tool"
        elif isinstance(m, ChatMessage):
            role = m.role
        else:
            msg = f"Got unsupported message type: {m}"
            raise ValueError(msg)  # noqa: TRY004

        message = f"{role}: {m.text}"

        if isinstance(m, AIMessage):
            if m.tool_calls:
                message += f"{m.tool_calls}"
            elif "function_call" in m.additional_kwargs:
                # Legacy behavior assumes only one function call per message
                # 中文: 传统行为假设每条消息仅调用一次函数
                message += f"{m.additional_kwargs['function_call']}"

        string_messages.append(message)

    return "\n".join(string_messages)


def _message_from_dict(message: dict) -> BaseMessage:
    type_ = message["type"]
    if type_ == "human":
        return HumanMessage(**message["data"])
    if type_ == "ai":
        return AIMessage(**message["data"])
    if type_ == "system":
        return SystemMessage(**message["data"])
    if type_ == "chat":
        return ChatMessage(**message["data"])
    if type_ == "function":
        return FunctionMessage(**message["data"])
    if type_ == "tool":
        return ToolMessage(**message["data"])
    if type_ == "remove":
        return RemoveMessage(**message["data"])
    if type_ == "AIMessageChunk":
        return AIMessageChunk(**message["data"])
    if type_ == "HumanMessageChunk":
        return HumanMessageChunk(**message["data"])
    if type_ == "FunctionMessageChunk":
        return FunctionMessageChunk(**message["data"])
    if type_ == "ToolMessageChunk":
        return ToolMessageChunk(**message["data"])
    if type_ == "SystemMessageChunk":
        return SystemMessageChunk(**message["data"])
    if type_ == "ChatMessageChunk":
        return ChatMessageChunk(**message["data"])
    msg = f"Got unexpected message type: {type_}"
    raise ValueError(msg)


def messages_from_dict(messages: Sequence[dict]) -> list[BaseMessage]:
    """Convert a sequence of messages from dicts to `Message` objects.

    Args:
        messages: Sequence of messages (as dicts) to convert.

    Returns:
        list of messages (BaseMessages).

    

    中文翻译:
    将消息序列从字典转换为“Message”对象。
    参数：
        messages：要转换的消息序列（作为字典）。
    返回：
        消息列表 (BaseMessages)。"""
    return [_message_from_dict(m) for m in messages]


def message_chunk_to_message(chunk: BaseMessage) -> BaseMessage:
    """Convert a message chunk to a `Message`.

    Args:
        chunk: Message chunk to convert.

    Returns:
        Message.
    

    中文翻译:
    将消息块转换为“消息”。
    参数：
        chunk：要转换的消息块。
    返回：
        留言。"""
    if not isinstance(chunk, BaseMessageChunk):
        return chunk
    # chunk classes always have the equivalent non-chunk class as their first parent
    # 中文: 块类始终具有等效的非块类作为其第一个父级
    ignore_keys = ["type"]
    if isinstance(chunk, AIMessageChunk):
        ignore_keys.extend(["tool_call_chunks", "chunk_position"])
    return cast(
        "BaseMessage",
        chunk.__class__.__mro__[1](
            **{k: v for k, v in chunk.__dict__.items() if k not in ignore_keys}
        ),
    )


MessageLikeRepresentation = (
    BaseMessage | list[str] | tuple[str, str] | str | dict[str, Any]
)
"""A type representing the various ways a message can be represented.

中文翻译:
表示消息表示的各种方式的类型。"""


def _create_message_from_message_type(
    message_type: str,
    content: str,
    name: str | None = None,
    tool_call_id: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    id: str | None = None,
    **additional_kwargs: Any,
) -> BaseMessage:
    """Create a message from a `Message` type and content string.

    Args:
        message_type: the type of the message (e.g., `'human'`, `'ai'`, etc.).
        content: the content string.
        name: the name of the message.
        tool_call_id: the tool call id.
        tool_calls: the tool calls.
        id: the id of the message.
        additional_kwargs: additional keyword arguments.

    Returns:
        a message of the appropriate type.

    Raises:
        ValueError: if the message type is not one of `'human'`, `'user'`, `'ai'`,
            `'assistant'`, `'function'`, `'tool'`, `'system'`, or
            `'developer'`.
    

    中文翻译:
    从“Message”类型和内容字符串创建消息。
    参数：
        message_type：消息的类型（例如，“人类”、“人工智能”等）。
        内容：内容字符串。
        名称：消息的名称。
        tool_call_id：工具调用 ID。
        tool_calls：工具调用。
        id：消息的id。
        extra_kwargs：附加关键字参数。
    返回：
        适当类型的消息。
    加薪：
        ValueError：如果消息类型不是“人类”、“用户”、“人工智能”之一，
            “助手”、“功能”、“工具”、“系统”或
            “开发商”。"""
    kwargs: dict[str, Any] = {}
    if name is not None:
        kwargs["name"] = name
    if tool_call_id is not None:
        kwargs["tool_call_id"] = tool_call_id
    if additional_kwargs:
        if response_metadata := additional_kwargs.pop("response_metadata", None):
            kwargs["response_metadata"] = response_metadata
        kwargs["additional_kwargs"] = additional_kwargs
        additional_kwargs.update(additional_kwargs.pop("additional_kwargs", {}))
    if id is not None:
        kwargs["id"] = id
    if tool_calls is not None:
        kwargs["tool_calls"] = []
        for tool_call in tool_calls:
            # Convert OpenAI-format tool call to LangChain format.
            # 中文: 将OpenAI格式的工具调用转换为LangChain格式。
            if "function" in tool_call:
                args = tool_call["function"]["arguments"]
                if isinstance(args, str):
                    args = json.loads(args, strict=False)
                kwargs["tool_calls"].append(
                    {
                        "name": tool_call["function"]["name"],
                        "args": args,
                        "id": tool_call["id"],
                        "type": "tool_call",
                    }
                )
            else:
                kwargs["tool_calls"].append(tool_call)
    if message_type in {"human", "user"}:
        if example := kwargs.get("additional_kwargs", {}).pop("example", False):
            kwargs["example"] = example
        message: BaseMessage = HumanMessage(content=content, **kwargs)
    elif message_type in {"ai", "assistant"}:
        if example := kwargs.get("additional_kwargs", {}).pop("example", False):
            kwargs["example"] = example
        message = AIMessage(content=content, **kwargs)
    elif message_type in {"system", "developer"}:
        if message_type == "developer":
            kwargs["additional_kwargs"] = kwargs.get("additional_kwargs") or {}
            kwargs["additional_kwargs"]["__openai_role__"] = "developer"
        message = SystemMessage(content=content, **kwargs)
    elif message_type == "function":
        message = FunctionMessage(content=content, **kwargs)
    elif message_type == "tool":
        artifact = kwargs.get("additional_kwargs", {}).pop("artifact", None)
        status = kwargs.get("additional_kwargs", {}).pop("status", None)
        if status is not None:
            kwargs["status"] = status
        message = ToolMessage(content=content, artifact=artifact, **kwargs)
    elif message_type == "remove":
        message = RemoveMessage(**kwargs)
    else:
        msg = (
            f"Unexpected message type: '{message_type}'. Use one of 'human',"
            f" 'user', 'ai', 'assistant', 'function', 'tool', 'system', or 'developer'."
        )
        msg = create_message(message=msg, error_code=ErrorCode.MESSAGE_COERCION_FAILURE)
        raise ValueError(msg)
    return message


def _convert_to_message(message: MessageLikeRepresentation) -> BaseMessage:
    """Instantiate a `Message` from a variety of message formats.

    The message format can be one of the following:

    - `BaseMessagePromptTemplate`
    - `BaseMessage`
    - 2-tuple of (role string, template); e.g., (`'human'`, `'{user_input}'`)
    - dict: a message dict with role and content keys
    - string: shorthand for (`'human'`, template); e.g., `'{user_input}'`

    Args:
        message: a representation of a message in one of the supported formats.

    Returns:
        An instance of a message or a message template.

    Raises:
        NotImplementedError: if the message type is not supported.
        ValueError: if the message dict does not contain the required keys.

    

    中文翻译:
    从各种消息格式实例化“消息”。
    消息格式可以是以下之一：
    - `BaseMessagePromptTemplate`
    - `基本消息`
    - （角色字符串，模板）的 2 元组；例如，(`'人类'`，`'{user_input}'`)
    - 字典：带有角色和内容键的消息字典
    - 字符串：(`'human'`, template) 的简写；例如，“{user_input}”
    参数：
        消息：以受支持的格式之一表示消息。
    返回：
        消息或消息模板的实例。
    加薪：
        NotImplementedError：如果消息类型不受支持。
        ValueError：如果消息字典不包含所需的键。"""
    if isinstance(message, BaseMessage):
        message_ = message
    elif isinstance(message, Sequence):
        if isinstance(message, str):
            message_ = _create_message_from_message_type("human", message)
        else:
            try:
                message_type_str, template = message
            except ValueError as e:
                msg = "Message as a sequence must be (role string, template)"
                raise NotImplementedError(msg) from e
            message_ = _create_message_from_message_type(message_type_str, template)
    elif isinstance(message, dict):
        msg_kwargs = message.copy()
        try:
            try:
                msg_type = msg_kwargs.pop("role")
            except KeyError:
                msg_type = msg_kwargs.pop("type")
            # None msg content is not allowed
            # 中文: None 不允许的消息内容
            msg_content = msg_kwargs.pop("content") or ""
        except KeyError as e:
            msg = f"Message dict must contain 'role' and 'content' keys, got {message}"
            msg = create_message(
                message=msg, error_code=ErrorCode.MESSAGE_COERCION_FAILURE
            )
            raise ValueError(msg) from e
        message_ = _create_message_from_message_type(
            msg_type, msg_content, **msg_kwargs
        )
    else:
        msg = f"Unsupported message type: {type(message)}"
        msg = create_message(message=msg, error_code=ErrorCode.MESSAGE_COERCION_FAILURE)
        raise NotImplementedError(msg)

    return message_


def convert_to_messages(
    messages: Iterable[MessageLikeRepresentation] | PromptValue,
) -> list[BaseMessage]:
    """Convert a sequence of messages to a list of messages.

    Args:
        messages: Sequence of messages to convert.

    Returns:
        list of messages (BaseMessages).

    

    中文翻译:
    将消息序列转换为消息列表。
    参数：
        messages：要转换的消息序列。
    返回：
        消息列表 (BaseMessages)。"""
    # Import here to avoid circular imports
    # 中文: 在这里导入以避免循环导入
    from langchain_core.prompt_values import PromptValue  # noqa: PLC0415

    if isinstance(messages, PromptValue):
        return messages.to_messages()
    return [_convert_to_message(m) for m in messages]


_P = ParamSpec("_P")
_R_co = TypeVar("_R_co", covariant=True)


class _RunnableSupportCallable(Protocol[_P, _R_co]):
    @overload
    def __call__(
        self,
        messages: None = None,
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> Runnable[Sequence[MessageLikeRepresentation], _R_co]: ...

    @overload
    def __call__(
        self,
        messages: Sequence[MessageLikeRepresentation] | PromptValue,
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> _R_co: ...

    def __call__(
        self,
        messages: Sequence[MessageLikeRepresentation] | PromptValue | None = None,
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> _R_co | Runnable[Sequence[MessageLikeRepresentation], _R_co]: ...


def _runnable_support(
    func: Callable[
        Concatenate[Sequence[MessageLikeRepresentation] | PromptValue, _P], _R_co
    ],
) -> _RunnableSupportCallable[_P, _R_co]:
    @wraps(func)
    def wrapped(
        messages: Sequence[MessageLikeRepresentation] | PromptValue | None = None,
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> _R_co | Runnable[Sequence[MessageLikeRepresentation], _R_co]:
        # Import locally to prevent circular import.
        # 中文: 本地导入，防止循环导入。
        from langchain_core.runnables.base import RunnableLambda  # noqa: PLC0415

        if messages is not None:
            return func(messages, *args, **kwargs)
        return RunnableLambda(partial(func, **kwargs), name=func.__name__)

    return cast("_RunnableSupportCallable[_P, _R_co]", wrapped)


@_runnable_support
def filter_messages(
    messages: Iterable[MessageLikeRepresentation] | PromptValue,
    *,
    include_names: Sequence[str] | None = None,
    exclude_names: Sequence[str] | None = None,
    include_types: Sequence[str | type[BaseMessage]] | None = None,
    exclude_types: Sequence[str | type[BaseMessage]] | None = None,
    include_ids: Sequence[str] | None = None,
    exclude_ids: Sequence[str] | None = None,
    exclude_tool_calls: Sequence[str] | bool | None = None,
) -> list[BaseMessage]:
    """Filter messages based on `name`, `type` or `id`.

    Args:
        messages: Sequence Message-like objects to filter.
        include_names: Message names to include.
        exclude_names: Messages names to exclude.
        include_types: Message types to include. Can be specified as string names
            (e.g. `'system'`, `'human'`, `'ai'`, ...) or as `BaseMessage`
            classes (e.g. `SystemMessage`, `HumanMessage`, `AIMessage`, ...).

        exclude_types: Message types to exclude. Can be specified as string names
            (e.g. `'system'`, `'human'`, `'ai'`, ...) or as `BaseMessage`
            classes (e.g. `SystemMessage`, `HumanMessage`, `AIMessage`, ...).

        include_ids: Message IDs to include.
        exclude_ids: Message IDs to exclude.
        exclude_tool_calls: Tool call IDs to exclude.
            Can be one of the following:
            - `True`: All `AIMessage` objects with tool calls and all `ToolMessage`
                objects will be excluded.
            - a sequence of tool call IDs to exclude:
                - `ToolMessage` objects with the corresponding tool call ID will be
                    excluded.
                - The `tool_calls` in the AIMessage will be updated to exclude
                    matching tool calls. If all `tool_calls` are filtered from an
                    AIMessage, the whole message is excluded.

    Returns:
        A list of Messages that meets at least one of the `incl_*` conditions and none
        of the `excl_*` conditions. If not `incl_*` conditions are specified then
        anything that is not explicitly excluded will be included.

    Raises:
        ValueError: If two incompatible arguments are provided.

    Example:
        ```python
        from langchain_core.messages import (
            filter_messages,
            AIMessage,
            HumanMessage,
            SystemMessage,
        )

        messages = [
            SystemMessage("you're a good assistant."),
            HumanMessage("what's your name", id="foo", name="example_user"),
            AIMessage("steve-o", id="bar", name="example_assistant"),
            HumanMessage(
                "what's your favorite color",
                id="baz",
            ),
            AIMessage(
                "silicon blue",
                id="blah",
            ),
        ]

        filter_messages(
            messages,
            incl_names=("example_user", "example_assistant"),
            incl_types=("system",),
            excl_ids=("bar",),
        )
        ```

        ```python
        [
            SystemMessage("you're a good assistant."),
            HumanMessage("what's your name", id="foo", name="example_user"),
        ]
        ```
    

    中文翻译:
    根据“名称”、“类型”或“id”过滤消息。
    参数：
        messages：要过滤的类似消息对象的序列。
        include_names：要包含的消息名称。
        except_names：要排除的消息名称。
        include_types：要包含的消息类型。可以指定为字符串名称
            （例如“系统”、“人类”、“人工智能”……）或作为“BaseMessage”
            类（例如“SystemMessage”、“HumanMessage”、“AIMessage”……）。
        except_types：要排除的消息类型。可以指定为字符串名称
            （例如“系统”、“人类”、“人工智能”……）或作为“BaseMessage”
            类（例如“SystemMessage”、“HumanMessage”、“AIMessage”……）。
        include_ids：要包含的消息 ID。
        except_ids：要排除的消息 ID。
        except_tool_calls：要排除的工具调用 ID。
            可以是以下之一：
            - `True`：具有工具调用的所有`AIMessage`对象和所有`ToolMessage`
                对象将被排除。
            - 要排除的一系列工具调用 ID：
                - 具有相应工具调用 ID 的“ToolMessage”对象将是
                    排除。
                - AIMessage 中的“tool_calls”将更新为排除
                    匹配工具调用。如果所有“tool_calls”都从
                    AIMessage，整个消息被排除。
    返回：
        至少满足一个“incl_*”条件且无条件的消息列表
        `excl_*` 条件。如果没有指定“incl_*”条件，则
        任何未明确排除的内容都将包括在内。
    加薪：
        ValueError：如果提供了两个不兼容的参数。
    示例：
        ````蟒蛇
        从 langchain_core.messages 导入（
            过滤消息，
            人工智能留言，
            人类讯息，
            系统消息，
        ）
        消息 = [
            SystemMessage("你是一个好助手。"),
            HumanMessage("你叫什么名字", id="foo", name="example_user"),
            AIMessage("steve-o", id="bar", name="example_assistant"),
            人类讯息(
                “你最喜欢什么颜色”，
                id=“巴兹”，
            ),
            人工智能留言(
                “硅蓝”，
                id =“废话”，
            ),
        ]
        过滤消息（
            消息，
            incl_names=("example_user", "example_assistant"),
            incl_types=("系统",),
            excl_ids=("酒吧",),
        ）
        ````
        ````蟒蛇
        [
            SystemMessage("你是一个好助手。"),
            HumanMessage("你叫什么名字", id="foo", name="example_user"),
        ]
        ````"""
    messages = convert_to_messages(messages)
    filtered: list[BaseMessage] = []
    for msg in messages:
        if (
            (exclude_names and msg.name in exclude_names)
            or (exclude_types and _is_message_type(msg, exclude_types))
            or (exclude_ids and msg.id in exclude_ids)
        ):
            continue

        if exclude_tool_calls is True and (
            (isinstance(msg, AIMessage) and msg.tool_calls)
            or isinstance(msg, ToolMessage)
        ):
            continue

        if isinstance(exclude_tool_calls, (list, tuple, set)):
            if isinstance(msg, AIMessage) and msg.tool_calls:
                tool_calls = [
                    tool_call
                    for tool_call in msg.tool_calls
                    if tool_call["id"] not in exclude_tool_calls
                ]
                if not tool_calls:
                    continue

                content = msg.content
                # handle Anthropic content blocks
                # 中文: 处理人为内容块
                if isinstance(msg.content, list):
                    content = [
                        content_block
                        for content_block in msg.content
                        if (
                            not isinstance(content_block, dict)
                            or content_block.get("type") != "tool_use"
                            or content_block.get("id") not in exclude_tool_calls
                        )
                    ]

                msg = msg.model_copy(  # noqa: PLW2901
                    update={"tool_calls": tool_calls, "content": content}
                )
            elif (
                isinstance(msg, ToolMessage) and msg.tool_call_id in exclude_tool_calls
            ):
                continue

        # default to inclusion when no inclusion criteria given.
        # 中文: 当未给出纳入标准时默认纳入。
        if (
            not (include_types or include_ids or include_names)
            or (include_names and msg.name in include_names)
            or (include_types and _is_message_type(msg, include_types))
            or (include_ids and msg.id in include_ids)
        ):
            filtered.append(msg)

    return filtered


@_runnable_support
def merge_message_runs(
    messages: Iterable[MessageLikeRepresentation] | PromptValue,
    *,
    chunk_separator: str = "\n",
) -> list[BaseMessage]:
    r"""Merge consecutive Messages of the same type.

    !!! note
        `ToolMessage` objects are not merged, as each has a distinct tool call id that
        can't be merged.

    Args:
        messages: Sequence Message-like objects to merge.
        chunk_separator: Specify the string to be inserted between message chunks.

    Returns:
        list of BaseMessages with consecutive runs of message types merged into single
        messages. By default, if two messages being merged both have string contents,
        the merged content is a concatenation of the two strings with a new-line
        separator.
        The separator inserted between message chunks can be controlled by specifying
        any string with `chunk_separator`. If at least one of the messages has a list
        of content blocks, the merged content is a list of content blocks.

    Example:
        ```python
        from langchain_core.messages import (
            merge_message_runs,
            AIMessage,
            HumanMessage,
            SystemMessage,
            ToolCall,
        )

        messages = [
            SystemMessage("you're a good assistant."),
            HumanMessage(
                "what's your favorite color",
                id="foo",
            ),
            HumanMessage(
                "wait your favorite food",
                id="bar",
            ),
            AIMessage(
                "my favorite colo",
                tool_calls=[
                    ToolCall(
                        name="blah_tool", args={"x": 2}, id="123", type="tool_call"
                    )
                ],
                id="baz",
            ),
            AIMessage(
                [{"type": "text", "text": "my favorite dish is lasagna"}],
                tool_calls=[
                    ToolCall(
                        name="blah_tool",
                        args={"x": -10},
                        id="456",
                        type="tool_call",
                    )
                ],
                id="blur",
            ),
        ]

        merge_message_runs(messages)
        ```

        ```python
        [
            SystemMessage("you're a good assistant."),
            HumanMessage(
                "what's your favorite color\\n"
                "wait your favorite food", id="foo",
            ),
            AIMessage(
                [
                    "my favorite colo",
                    {"type": "text", "text": "my favorite dish is lasagna"}
                ],
                tool_calls=[
                    ToolCall({
                        "name": "blah_tool",
                        "args": {"x": 2},
                        "id": "123",
                        "type": "tool_call"
                    }),
                    ToolCall({
                        "name": "blah_tool",
                        "args": {"x": -10},
                        "id": "456",
                        "type": "tool_call"
                    })
                ]
                id="baz"
            ),
        ]

        ```
    

中文翻译:
合并相同类型的连续消息。
    ！！！注释
        `ToolMessage` 对象不会合并，因为每个对象都有一个不同的工具调用 ID
        无法合并。
    参数：
        messages：要合并的类似消息对象的序列。
        chunk_separator：指定要插入消息块之间的字符串。
    返回：
        连续运行的消息类型合并为单个的 BaseMessage 列表
        消息。默认情况下，如果要合并的两条消息都有字符串内容，
        合并的内容是两个字符串与换行符的串联
        分隔符。
        消息块之间插入的分隔符可以通过指定来控制
        任何带有“chunk_separator”的字符串。如果至少有一条消息有一个列表
        的内容块，合并的内容是内容块的列表。
    示例：
        ````蟒蛇
        从 langchain_core.messages 导入（
            合并消息运行，
            人工智能留言，
            人类讯息，
            系统消息，
            工具调用，
        ）
        消息 = [
            SystemMessage("你是一个好助手。"),
            人类讯息(
                “你最喜欢什么颜色”，
                id =“富”，
            ),
            人类讯息(
                “等待你最喜欢的食物”，
                id=“酒吧”，
            ),
            人工智能留言(
                “我最喜欢的颜色”，
                工具调用=[
                    工具调用(
                        名称=“blah_tool”，args={“x”：2}，id=“123”，类型=“tool_call”
                    ）
                ],
                id=“巴兹”，
            ),
            人工智能留言(
                [{"type": "text", "text": "我最喜欢的菜是烤宽面条"}],
                工具调用=[
                    工具调用(
                        名称=“blah_tool”，
                        args={"x": -10},
                        ID =“456”，
                        类型=“工具调用”，
                    ）
                ],
                id =“模糊”，
            ),
        ]
        merge_message_runs（消息）
        ````
        ````蟒蛇
        [
            SystemMessage("你是一个好助手。"),
            人类讯息(
                “你最喜欢什么颜色\n”
                "等待你最喜欢的食物", id="foo",
            ),
            人工智能留言(
                [
                    “我最喜欢的颜色”，
                    {"type": "text", "text": "我最喜欢的菜是烤宽面条"}
                ],
                工具调用=[
                    工具调用({
                        “名称”：“blah_tool”，
                        “args”：{“x”：2}，
                        “id”：“123”，
                        “类型”：“工具调用”
                    }),
                    工具调用({
                        “名称”：“blah_tool”，
                        “args”：{“x”：-10}，
                        “id”：“456”，
                        “类型”：“工具调用”
                    })
                ]
                id=“巴兹”
            ),
        ]
        ````"""
    if not messages:
        return []
    messages = convert_to_messages(messages)
    merged: list[BaseMessage] = []
    for msg in messages:
        last = merged.pop() if merged else None
        if not last:
            merged.append(msg)
        elif isinstance(msg, ToolMessage) or not isinstance(msg, last.__class__):
            merged.extend([last, msg])
        else:
            last_chunk = _msg_to_chunk(last)
            curr_chunk = _msg_to_chunk(msg)
            if curr_chunk.response_metadata:
                curr_chunk.response_metadata.clear()
            if (
                isinstance(last_chunk.content, str)
                and isinstance(curr_chunk.content, str)
                and last_chunk.content
                and curr_chunk.content
            ):
                last_chunk.content += chunk_separator
            merged.append(_chunk_to_msg(last_chunk + curr_chunk))
    return merged


# TODO: Update so validation errors (for token_counter, for example) are raised on
# init not at runtime.
# 中文: 不在运行时初始化。
@_runnable_support
def trim_messages(
    messages: Iterable[MessageLikeRepresentation] | PromptValue,
    *,
    max_tokens: int,
    token_counter: Callable[[list[BaseMessage]], int]
    | Callable[[BaseMessage], int]
    | BaseLanguageModel
    | Literal["approximate"],
    strategy: Literal["first", "last"] = "last",
    allow_partial: bool = False,
    end_on: str | type[BaseMessage] | Sequence[str | type[BaseMessage]] | None = None,
    start_on: str | type[BaseMessage] | Sequence[str | type[BaseMessage]] | None = None,
    include_system: bool = False,
    text_splitter: Callable[[str], list[str]] | TextSplitter | None = None,
) -> list[BaseMessage]:
    r"""Trim messages to be below a token count.

    `trim_messages` can be used to reduce the size of a chat history to a specified
    token or message count.

    In either case, if passing the trimmed chat history back into a chat model
    directly, the resulting chat history should usually satisfy the following
    properties:

    1. The resulting chat history should be valid. Most chat models expect that chat
        history starts with either (1) a `HumanMessage` or (2) a `SystemMessage`
        followed by a `HumanMessage`. To achieve this, set `start_on='human'`.
        In addition, generally a `ToolMessage` can only appear after an `AIMessage`
        that involved a tool call.
    2. It includes recent messages and drops old messages in the chat history.
        To achieve this set the `strategy='last'`.
    3. Usually, the new chat history should include the `SystemMessage` if it
        was present in the original chat history since the `SystemMessage` includes
        special instructions to the chat model. The `SystemMessage` is almost always
        the first message in the history if present. To achieve this set the
        `include_system=True`.

    !!! note
        The examples below show how to configure `trim_messages` to achieve a behavior
        consistent with the above properties.

    Args:
        messages: Sequence of Message-like objects to trim.
        max_tokens: Max token count of trimmed messages.
        token_counter: Function or llm for counting tokens in a `BaseMessage` or a
            list of `BaseMessage`.

            If a `BaseLanguageModel` is passed in then
            `BaseLanguageModel.get_num_tokens_from_messages()` will be used. Set to
            `len` to count the number of **messages** in the chat history.

            You can also use string shortcuts for convenience:

            - `'approximate'`: Uses `count_tokens_approximately` for fast, approximate
                token counts.

            !!! note

                `count_tokens_approximately` (or the shortcut `'approximate'`) is
                recommended for using `trim_messages` on the hot path, where exact token
                counting is not necessary.

        strategy: Strategy for trimming.

            - `'first'`: Keep the first `<= n_count` tokens of the messages.
            - `'last'`: Keep the last `<= n_count` tokens of the messages.
        allow_partial: Whether to split a message if only part of the message can be
            included.

            If `strategy='last'` then the last partial contents of a message are
            included. If `strategy='first'` then the first partial contents of a
            message are included.
        end_on: The message type to end on.

            If specified then every message after the last occurrence of this type is
            ignored. If `strategy='last'` then this is done before we attempt to get the
            last `max_tokens`. If `strategy='first'` then this is done after we get the
            first `max_tokens`. Can be specified as string names (e.g. `'system'`,
            `'human'`, `'ai'`, ...) or as `BaseMessage` classes (e.g. `SystemMessage`,
            `HumanMessage`, `AIMessage`, ...). Can be a single type or a list of types.

        start_on: The message type to start on.

            Should only be specified if `strategy='last'`. If specified then every
            message before the first occurrence of this type is ignored. This is done
            after we trim the initial messages to the last `max_tokens`. Does not apply
            to a `SystemMessage` at index 0 if `include_system=True`. Can be specified
            as string names (e.g. `'system'`, `'human'`, `'ai'`, ...) or as
            `BaseMessage` classes (e.g. `SystemMessage`, `HumanMessage`, `AIMessage`,
            ...). Can be a single type or a list of types.

        include_system: Whether to keep the `SystemMessage` if there is one at index
            `0`.

            Should only be specified if `strategy="last"`.
        text_splitter: Function or `langchain_text_splitters.TextSplitter` for
            splitting the string contents of a message.

            Only used if `allow_partial=True`. If `strategy='last'` then the last split
            tokens from a partial message will be included. if `strategy='first'` then
            the first split tokens from a partial message will be included. Token
            splitter assumes that separators are kept, so that split contents can be
            directly concatenated to recreate the original text. Defaults to splitting
            on newlines.

    Returns:
        List of trimmed `BaseMessage`.

    Raises:
        ValueError: if two incompatible arguments are specified or an unrecognized
            `strategy` is specified.

    Example:
        Trim chat history based on token count, keeping the `SystemMessage` if
        present, and ensuring that the chat history starts with a `HumanMessage` (or a
        `SystemMessage` followed by a `HumanMessage`).

        ```python
        from langchain_core.messages import (
            AIMessage,
            HumanMessage,
            BaseMessage,
            SystemMessage,
            trim_messages,
        )

        messages = [
            SystemMessage("you're a good assistant, you always respond with a joke."),
            HumanMessage("i wonder why it's called langchain"),
            AIMessage(
                'Well, I guess they thought "WordRope" and "SentenceString" just '
                "didn't have the same ring to it!"
            ),
            HumanMessage("and who is harrison chasing anyways"),
            AIMessage(
                "Hmmm let me think.\n\nWhy, he's probably chasing after the last "
                "cup of coffee in the office!"
            ),
            HumanMessage("what do you call a speechless parrot"),
        ]


        trim_messages(
            messages,
            max_tokens=45,
            strategy="last",
            token_counter=ChatOpenAI(model="gpt-4o"),
            # Most chat models expect that chat history starts with either:
            # 中文: 大多数聊天模型期望聊天历史记录以以下任一开头：
            # (1) a HumanMessage or
            # 中文: (1) HumanMessage 或
            # (2) a SystemMessage followed by a HumanMessage
            # 中文: (2) SystemMessage 后跟 HumanMessage
            start_on="human",
            # Usually, we want to keep the SystemMessage
            # 中文: 通常，我们希望保留 SystemMessage
            # if it's present in the original history.
            # 中文: 如果它存在于原始历史中。
            # The SystemMessage has special instructions for the model.
            # 中文: SystemMessage 对模型有特殊说明。
            include_system=True,
            allow_partial=False,
        )
        ```

        ```python
        [
            SystemMessage(
                content="you're a good assistant, you always respond with a joke."
            ),
            HumanMessage(content="what do you call a speechless parrot"),
        ]
        ```

        Trim chat history using approximate token counting with `'approximate'`:

        ```python
        trim_messages(
            messages,
            max_tokens=45,
            strategy="last",
            # Using the "approximate" shortcut for fast token counting
            # 中文: 使用“近似”快捷方式进行快速令牌计数
            token_counter="approximate",
            start_on="human",
            include_system=True,
        )

        # This is equivalent to using `count_tokens_approximately` directly
        # 中文: 这相当于直接使用 `count_tokens_approximately`
        from langchain_core.messages.utils import count_tokens_approximately

        trim_messages(
            messages,
            max_tokens=45,
            strategy="last",
            token_counter=count_tokens_approximately,
            start_on="human",
            include_system=True,
        )
        ```

        Trim chat history based on the message count, keeping the `SystemMessage` if
        present, and ensuring that the chat history starts with a HumanMessage (
        or a `SystemMessage` followed by a `HumanMessage`).

            trim_messages(
                messages,
                # When `len` is passed in as the token counter function,
                # 中文: 当“len”作为令牌计数器函数传入时，
                # max_tokens will count the number of messages in the chat history.
                # 中文: max_tokens 将计算聊天记录中的消息数量。
                max_tokens=4,
                strategy="last",
                # Passing in `len` as a token counter function will
                # 中文: 传入 `len` 作为令牌计数器函数将
                # count the number of messages in the chat history.
                # 中文: 统计聊天记录中的消息数量。
                token_counter=len,
                # Most chat models expect that chat history starts with either:
                # 中文: 大多数聊天模型期望聊天历史记录以以下任一开头：
                # (1) a HumanMessage or
                # 中文: (1) HumanMessage 或
                # (2) a SystemMessage followed by a HumanMessage
                # 中文: (2) SystemMessage 后跟 HumanMessage
                start_on="human",
                # Usually, we want to keep the SystemMessage
                # 中文: 通常，我们希望保留 SystemMessage
                # if it's present in the original history.
                # 中文: 如果它存在于原始历史中。
                # The SystemMessage has special instructions for the model.
                # 中文: SystemMessage 对模型有特殊说明。
                include_system=True,
                allow_partial=False,
            )

        ```python
        [
            SystemMessage(
                content="you're a good assistant, you always respond with a joke."
            ),
            HumanMessage(content="and who is harrison chasing anyways"),
            AIMessage(
                content="Hmmm let me think.\n\nWhy, he's probably chasing after "
                "the last cup of coffee in the office!"
            ),
            HumanMessage(content="what do you call a speechless parrot"),
        ]
        ```
        Trim chat history using a custom token counter function that counts the
        number of tokens in each message.

        ```python
        messages = [
            SystemMessage("This is a 4 token text. The full message is 10 tokens."),
            HumanMessage(
                "This is a 4 token text. The full message is 10 tokens.", id="first"
            ),
            AIMessage(
                [
                    {"type": "text", "text": "This is the FIRST 4 token block."},
                    {"type": "text", "text": "This is the SECOND 4 token block."},
                ],
                id="second",
            ),
            HumanMessage(
                "This is a 4 token text. The full message is 10 tokens.", id="third"
            ),
            AIMessage(
                "This is a 4 token text. The full message is 10 tokens.",
                id="fourth",
            ),
        ]


        def dummy_token_counter(messages: list[BaseMessage]) -> int:
            # treat each message like it adds 3 default tokens at the beginning
            # 中文: 将每条消息视为在开头添加 3 个默认标记
            # of the message and at the end of the message. 3 + 4 + 3 = 10 tokens
            # 中文: 消息的开头和消息的末尾。 3 + 4 + 3 = 10 个代币
            # per message.
            # 中文: 每条消息。

            default_content_len = 4
            default_msg_prefix_len = 3
            default_msg_suffix_len = 3

            count = 0
            for msg in messages:
                if isinstance(msg.content, str):
                    count += (
                        default_msg_prefix_len
                        + default_content_len
                        + default_msg_suffix_len
                    )
                if isinstance(msg.content, list):
                    count += (
                        default_msg_prefix_len
                        + len(msg.content) * default_content_len
                        + default_msg_suffix_len
                    )
            return count
        ```

        First 30 tokens, allowing partial messages:
        ```python
        trim_messages(
            messages,
            max_tokens=30,
            token_counter=dummy_token_counter,
            strategy="first",
            allow_partial=True,
        )
        ```

        ```python
        [
            SystemMessage("This is a 4 token text. The full message is 10 tokens."),
            HumanMessage(
                "This is a 4 token text. The full message is 10 tokens.",
                id="first",
            ),
            AIMessage(
                [{"type": "text", "text": "This is the FIRST 4 token block."}],
                id="second",
            ),
        ]
        ```
    

中文翻译:
将消息修剪为令牌计数以下。
    `trim_messages` 可用于将聊天历史记录的大小减少到指定的大小
    令牌或消息计数。
    无论哪种情况，如果将修剪后的聊天历史记录传回聊天模型
    直接生成的聊天记录通常应满足以下条件
    属性：
    1. 生成的聊天记录应该是有效的。大多数聊天模型都期望聊天
        历史记录以 (1) `HumanMessage` 或 (2) `SystemMessage` 开始
        接下来是“HumanMessage”。要实现此目的，请设置 `start_on=' human'`。
        另外，一般`ToolMessage`只能出现在`AIMessage`之后
        这涉及到工具调用。
    2.它包括最近的消息并删除聊天记录中的旧消息。
        要实现此目的，请设置`strategy='last'`。
    3. 通常，新的聊天记录应该包含`SystemMessage`，如果
        由于“SystemMessage”包含，因此存在于原始聊天记录中
        对聊天模型的特殊说明。 `SystemMessage` 几乎总是
        历史记录中的第一条消息（如果存在）。为了实现这一目标，设置
        `include_system=True`。
    ！！！注释
        下面的示例展示了如何配置 `trim_messages` 来实现某种行为
        与上述性质一致。
    参数：
        messages：要修剪的类似消息对象的序列。
        max_tokens：修剪消息的最大令牌计数。
        token_counter：用于计算“BaseMessage”或 a 中的令牌的函数或 llm
            `BaseMessage` 列表。
            如果传入了`BaseLanguageModel`
            将使用“BaseLanguageModel.get_num_tokens_from_messages()”。设置为
            `len` 用于计算聊天历史记录中的**消息**数量。
            为了方便起见，您还可以使用字符串快捷方式：
            - `'approximate'`：使用 `count_tokens_approximately` 进行快速、近似
                令牌计数。
            ！！！注释
                “count_tokens_approximately”（或快捷方式“approximate”）是
                建议在热路径上使用“trim_messages”，其中确切的标记
                计数是没有必要的。
        策略：修剪策略。
            - `'first'`：保留消息的第一个 `<= n_count` 标记。
            - `'last'`：保留消息的最后一个 `<= n_count` 标记。
        allowed_partial：如果只能分割部分消息，是否分割消息
            包括在内。
            如果 `strategy='last'` 则消息的最后部分内容是
            包括在内。如果 `strategy='first'` 则 a 的第一个部分内容
            消息都包含在内。
        end_on：结束的消息类型。
            如果指定，则该类型最后一次出现后的每条消息都是
            被忽略。如果 `strategy='last'` 那么这是在我们尝试获取之前完成的
            最后一个“max_tokens”。如果 `strategy='first'` 那么这是在我们得到之后完成的
            第一个“max_tokens”。可以指定为字符串名称（例如“system”，
            `' human'`，`'ai'`，...）或作为 `BaseMessage` 类（例如 `SystemMessage`，
            `HumanMessage`、`AIMessage`、...)。可以是单个类型或类型列表。
        start_on：开始的消息类型。
            仅当`strategy='last'`时才应指定。如果指定则每个
            该类型第一次出现之前的消息将被忽略。这样就完成了
            在我们将初始消息修剪到最后一个“max_tokens”之后。不适用
            如果“include_system=True”，则指向索引 0 处的“SystemMessage”。可以指定
            作为字符串名称（例如`'system'`，`' human'`，`'ai'`，...）或作为
            `BaseMessage` 类（例如 `SystemMessage`、`HumanMessage`、`AIMessage`、
            ...）。可以是单个类型或类型列表。
        include_system: 如果索引处有`SystemMessage`是否保留
            ‘0’。
            仅当`strategy="last"`时才应指定。
        text_splitter：函数或 `langchain_text_splitters.TextSplitter`
            分割消息的字符串内容。
            仅当“allow_partial=True”时使用。如果 `strategy='last'` 则最后一次分割
            来自部分消息的令牌将被包括在内。如果 `strategy='first'` 那么
            将包含部分消息中的第一个分割令牌。代币splitter 假设保留分隔符，因此可以拆分内容
            直接连接以重新创建原始文本。默认为分割
            在换行符上。
    返回：
        修剪后的“BaseMessage”列表。
    加薪：
        ValueError：如果指定了两个不兼容的参数或无法识别的参数
            指定了“策略”。
    示例：
        根据令牌计数修剪聊天历史记录，如果
        存在，并确保聊天历史记录以“HumanMessage”（或
        “SystemMessage”后跟一个“HumanMessage”）。
        ````蟒蛇
        从 langchain_core.messages 导入（
            人工智能留言，
            人类讯息，
            基本消息，
            系统消息，
            修剪消息，
        ）
        消息 = [
            SystemMessage("你是一个好助手，你总是用笑话来回应。"),
            HumanMessage("我想知道为什么它被称为 langchain"),
            人工智能留言(
                '好吧，我猜他们认为“WordRope”和“SentenceString”只是'
                “没有同样的戒指！”
            ),
            HumanMessage(“哈里森到底在追谁”),
            人工智能留言(
                “嗯，让我想想。\n\n为什么，他可能在追最后一个”
                “在办公室喝杯咖啡吧！”
            ),
            HumanMessage("你怎么称呼一只不会说话的鹦鹉"),
        ]
        修剪消息（
            消息，
            最大令牌=45，
            策略=“最后”，
            token_counter=ChatOpenAI(型号=“gpt-4o”),
            # 大多数聊天模型期望聊天历史记录以以下任一开头：
            # (1) 一条 HumanMessage 或
            # (2) SystemMessage 后跟 HumanMessage
            start_on=“人类”，
            # 通常，我们希望保留 SystemMessage
            # 如果它存在于原始历史中。
            # SystemMessage 对模型有特殊说明。
            include_system=真，
            允许部分=假，
        ）
        ````
        ````蟒蛇
        [
            系统消息(
                content="你是个好助手，你总是用笑话来回应。"
            ),
            HumanMessage(content="你怎么称呼一只不会说话的鹦鹉"),
        ]
        ````
        使用“approximate”进行近似令牌计数来修剪聊天历史记录：
        ````蟒蛇
        修剪消息（
            消息，
            最大令牌=45，
            策略=“最后”，
            # 使用“近似”快捷方式进行快速令牌计数
            token_counter=“大约”，
            start_on=“人类”，
            include_system=真，
        ）
        # 这相当于直接使用 `count_tokens_approximately`
        从 langchain_core.messages.utils 导入 count_tokens_approximately
        修剪消息（
            消息，
            最大令牌=45，
            策略=“最后”，
            token_counter=count_tokens_approximately,
            start_on=“人类”，
            include_system=真，
        ）
        ````
        根据消息计数修剪聊天记录，如果存在则保留“SystemMessage”
        存在，并确保聊天历史记录以 HumanMessage 开头（
        或“SystemMessage”后跟“HumanMessage”）。
            修剪消息（
                消息，
                # 当 `len` 作为令牌计数器函数传入时，
                # max_tokens 将计算聊天记录中的消息数量。
                最大令牌数=4,
                策略=“最后”，
                # 传入 `len` 作为令牌计数器函数将
                # 统计聊天记录中的消息数量。
                token_counter=len,
                # 大多数聊天模型期望聊天历史记录以以下任一开头：
                # (1) 一条 HumanMessage 或
                # (2) SystemMessage 后跟 HumanMessage
                start_on=“人类”，
                # 通常，我们希望保留 SystemMessage
                # 如果它存在于原始历史中。
                # SystemMessage 对模型有特殊说明。
                include_system=真，
                允许部分=假，
            ）
        ````蟒蛇
        [
            系统消息(
                content="你是个好助手，你总是用笑话来回应。"
            ),HumanMessage(content="哈里森到底在追谁"),
            人工智能留言(
                content="嗯，让我想想。\n\n为什么，他可能在追"
                “办公室最后一杯咖啡！”
            ),
            HumanMessage(content="你怎么称呼一只不会说话的鹦鹉"),
        ]
        ````
        使用自定义令牌计数器功能修剪聊天历史记录
        每条消息中的令牌数。
        ````蟒蛇
        消息 = [
            SystemMessage("这是一个 4 个令牌的文本。完整的消息是 10 个令牌。"),
            人类讯息(
                “这是一个 4 个令牌的文本。完整的消息是 10 个令牌。”, id="first"
            ),
            人工智能留言(
                [
                    {"type": "text", "text": "这是第 4 个代币块。"},
                    {"type": "text", "text": "这是第二个 4 代币块。"},
                ],
                id=“第二个”，
            ),
            人类讯息(
                “这是一个 4 个令牌的文本。完整的消息是 10 个令牌。”, id="third"
            ),
            人工智能留言(
                "这是 4 个令牌的文本。完整的消息是 10 个令牌。",
                id=“第四”，
            ),
        ]
        def dummy_token_counter(消息: list[BaseMessage]) -> int:
            # 将每条消息视为在开头添加 3 个默认标记
            消息的 # 和消息的末尾。 3 + 4 + 3 = 10 个代币
            # 每条消息。
            默认内容长度 = 4
            默认消息前缀长度 = 3
            default_msg_suffix_len = 3
            计数 = 0
            对于消息中的消息：
                if isinstance(msg.content, str):
                    计数 += (
                        默认消息前缀长度
                        + 默认内容长度
                        + default_msg_suffix_len
                    ）
                if isinstance(msg.content, 列表):
                    计数 += (
                        默认消息前缀长度
                        + len(msg.content) * default_content_len
                        + default_msg_suffix_len
                    ）
            返回计数
        ````
        前 30 个令牌，允许部分消息：
        ````蟒蛇
        修剪消息（
            消息，
            最大令牌数=30,
            token_counter=dummy_token_counter,
            策略=“第一”，
            允许部分=真，
        ）
        ````
        ````蟒蛇
        [
            SystemMessage("这是一个 4 个令牌的文本。完整的消息是 10 个令牌。"),
            人类讯息(
                "这是 4 个令牌的文本。完整的消息是 10 个令牌。",
                id=“第一个”，
            ),
            人工智能留言(
                [{"type": "text", "text": "这是第一个 4 个代币块。"}],
                id=“第二个”，
            ),
        ]
        ````"""
    # Validate arguments
    # 中文: 验证参数
    if start_on and strategy == "first":
        msg = "start_on parameter is only valid with strategy='last'"
        raise ValueError(msg)
    if include_system and strategy == "first":
        msg = "include_system parameter is only valid with strategy='last'"
        raise ValueError(msg)

    messages = convert_to_messages(messages)

    # Handle string shortcuts for token counter
    # 中文: 处理令牌计数器的字符串快捷方式
    if isinstance(token_counter, str):
        if token_counter in _TOKEN_COUNTER_SHORTCUTS:
            actual_token_counter = _TOKEN_COUNTER_SHORTCUTS[token_counter]
        else:
            available_shortcuts = ", ".join(
                f"'{key}'" for key in _TOKEN_COUNTER_SHORTCUTS
            )
            msg = (
                f"Invalid token_counter shortcut '{token_counter}'. "
                f"Available shortcuts: {available_shortcuts}."
            )
            raise ValueError(msg)
    else:
        # Type narrowing: at this point token_counter is not a str
        # 中文: 类型缩小：此时 token_counter 不是 str
        actual_token_counter = token_counter  # type: ignore[assignment]

    if hasattr(actual_token_counter, "get_num_tokens_from_messages"):
        list_token_counter = actual_token_counter.get_num_tokens_from_messages
    elif callable(actual_token_counter):
        if (
            next(
                iter(inspect.signature(actual_token_counter).parameters.values())
            ).annotation
            is BaseMessage
        ):

            def list_token_counter(messages: Sequence[BaseMessage]) -> int:
                return sum(actual_token_counter(msg) for msg in messages)  # type: ignore[arg-type, misc]

        else:
            list_token_counter = actual_token_counter
    else:
        msg = (
            f"'token_counter' expected to be a model that implements "
            f"'get_num_tokens_from_messages()' or a function. Received object of type "
            f"{type(actual_token_counter)}."
        )
        raise ValueError(msg)

    if _HAS_LANGCHAIN_TEXT_SPLITTERS and isinstance(text_splitter, TextSplitter):
        text_splitter_fn = text_splitter.split_text
    elif text_splitter:
        text_splitter_fn = cast("Callable", text_splitter)
    else:
        text_splitter_fn = _default_text_splitter

    if strategy == "first":
        return _first_max_tokens(
            messages,
            max_tokens=max_tokens,
            token_counter=list_token_counter,
            text_splitter=text_splitter_fn,
            partial_strategy="first" if allow_partial else None,
            end_on=end_on,
        )
    if strategy == "last":
        return _last_max_tokens(
            messages,
            max_tokens=max_tokens,
            token_counter=list_token_counter,
            allow_partial=allow_partial,
            include_system=include_system,
            start_on=start_on,
            end_on=end_on,
            text_splitter=text_splitter_fn,
        )
    msg = f"Unrecognized {strategy=}. Supported strategies are 'last' and 'first'."
    raise ValueError(msg)


_SingleMessage = BaseMessage | str | dict[str, Any]
_T = TypeVar("_T", bound=_SingleMessage)
# A sequence of _SingleMessage that is NOT a bare str
# 中文: _SingleMessage 序列，不是裸露的 str
_MultipleMessages = Sequence[_T]


@overload
def convert_to_openai_messages(
    messages: _SingleMessage,
    *,
    text_format: Literal["string", "block"] = "string",
    include_id: bool = False,
    pass_through_unknown_blocks: bool = True,
) -> dict: ...


@overload
def convert_to_openai_messages(
    messages: _MultipleMessages,
    *,
    text_format: Literal["string", "block"] = "string",
    include_id: bool = False,
    pass_through_unknown_blocks: bool = True,
) -> list[dict]: ...


def convert_to_openai_messages(
    messages: MessageLikeRepresentation | Sequence[MessageLikeRepresentation],
    *,
    text_format: Literal["string", "block"] = "string",
    include_id: bool = False,
    pass_through_unknown_blocks: bool = True,
) -> dict | list[dict]:
    """Convert LangChain messages into OpenAI message dicts.

    Args:
        messages: Message-like object or iterable of objects whose contents are
            in OpenAI, Anthropic, Bedrock Converse, or VertexAI formats.
        text_format: How to format string or text block contents:
            - `'string'`:
                If a message has a string content, this is left as a string. If
                a message has content blocks that are all of type `'text'`, these
                are joined with a newline to make a single string. If a message has
                content blocks and at least one isn't of type `'text'`, then
                all blocks are left as dicts.
            - `'block'`:
                If a message has a string content, this is turned into a list
                with a single content block of type `'text'`. If a message has
                content blocks these are left as is.
        include_id: Whether to include message IDs in the openai messages, if they
            are present in the source messages.
        pass_through_unknown_blocks: Whether to include content blocks with unknown
            formats in the output. If `False`, an error is raised if an unknown
            content block is encountered.

    Raises:
        ValueError: if an unrecognized `text_format` is specified, or if a message
            content block is missing expected keys.

    Returns:
        The return type depends on the input type:

        - dict:
            If a single message-like object is passed in, a single OpenAI message
            dict is returned.
        - list[dict]:
            If a sequence of message-like objects are passed in, a list of OpenAI
            message dicts is returned.

    Example:
        ```python
        from langchain_core.messages import (
            convert_to_openai_messages,
            AIMessage,
            SystemMessage,
            ToolMessage,
        )

        messages = [
            SystemMessage([{"type": "text", "text": "foo"}]),
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "whats in this"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,'/9j/4AAQSk'"},
                    },
                ],
            },
            AIMessage(
                "",
                tool_calls=[
                    {
                        "name": "analyze",
                        "args": {"baz": "buz"},
                        "id": "1",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage("foobar", tool_call_id="1", name="bar"),
            {"role": "assistant", "content": "thats nice"},
        ]
        oai_messages = convert_to_openai_messages(messages)
        # -> [
        # 中文: None
        #   {'role': 'system', 'content': 'foo'},
        #   中文: {'角色': '系统', '内容': 'foo'},
        #   {'role': 'user', 'content': [{'type': 'text', 'text': 'whats in this'}, {'type': 'image_url', 'image_url': {'url': "data:image/png;base64,'/9j/4AAQSk'"}}]},
        #   中文: {'role': 'user', 'content': [{'type': 'text', 'text': '这里面有什么'}, {'type': 'image_url', 'image_url': {'url': "data:image/png;base64,'/9j/4AAQSk'"}}]},
        #   {'role': 'assistant', 'tool_calls': [{'type': 'function', 'id': '1','function': {'name': 'analyze', 'arguments': '{"baz": "buz"}'}}], 'content': ''},
        #   中文: {'role': 'assistant', 'tool_calls': [{'type': 'function', 'id': '1','function': {'name': 'analyze', 'arguments': '{"baz": "buz"}'}}], 'content': ''},
        #   {'role': 'tool', 'name': 'bar', 'content': 'foobar'},
        #   中文: {'角色': '工具', '名称': 'bar', '内容': 'foobar'},
        #   {'role': 'assistant', 'content': 'thats nice'}
        #   中文: {'角色': '助理', '内容': '那很好'}
        # ]
        ```

    !!! version-added "Added in `langchain-core` 0.3.11"

    

    中文翻译:
    将 LangChain 消息转换为 OpenAI 消息字典。
    参数：
        messages：类似消息的对象或可迭代的对象，其内容为
            OpenAI、Anthropic、Bedrock Converse 或 VertexAI 格式。
        text_format：如何格式化字符串或文本块内容：
            - `'字符串'`：
                如果消息具有字符串内容，则将其保留为字符串。如果
                消息的内容块都是“文本”类型，这些
                与换行符连接以形成单个字符串。如果一条消息有
                内容块且至少有一个不是“文本”类型，然后
                所有块都保留为字典。
            - `'阻止'`：
                如果消息有字符串内容，则会将其转换为列表
                具有“文本”类型的单个内容块。如果一条消息有
                内容块保留原样。
        include_id: 是否在openai消息中包含消息ID，如果
            存在于源消息中。
        pass_through_unknown_blocks：是否包含未知的内容块
            输出中的格式。如果为“False”，则在未知的情况下会引发错误
            遇到内容块。
    加薪：
        ValueError：如果指定了无法识别的“text_format”，或者如果消息
            内容块缺少预期的键。
    返回：
        返回类型取决于输入类型：
        - 字典：
            如果传入单个类似消息的对象，则为单个 OpenAI 消息
            返回字典。
        - 列表[字典]：
            如果传入一系列类似消息的对象，则为 OpenAI 的列表
            返回消息字典。
    示例：
        ````蟒蛇
        从 langchain_core.messages 导入（
            Convert_to_openai_messages,
            人工智能留言，
            系统消息，
            工具消息，
        ）
        消息 = [
            SystemMessage([{"type": "text", "text": "foo"}]),
            {
                “角色”：“用户”，
                “内容”：[
                    {"type": "text", "text": "这是什么"},
                    {
                        “类型”：“图像_url”，
                        "image_url": {"url": "data:image/png;base64,'/9j/4AAQSk'"},
                    },
                ],
            },
            人工智能留言(
                "",
                工具调用=[
                    {
                        “名称”：“分析”，
                        "args": {"baz": "buz"},
                        “id”：“1”，
                        “类型”：“工具调用”，
                    }
                ],
            ),
            ToolMessage("foobar", tool_call_id="1", name="bar"),
            {"role": "助理", "content": "不错"},
        ]
        oai_messages = Convert_to_openai_messages(消息)
        # -> [
        # 中文: None
        # {'角色': '系统', '内容': 'foo'},
        # {'role': 'user', 'content': [{'type': 'text', 'text': '这里面有什么'}, {'type': 'image_url', 'image_url': {'url': "data:image/png;base64,'/9j/4AAQSk'"}}]},
        # {'role': 'assistant', 'tool_calls': [{'type': 'function', 'id': '1','function': {'name': 'analyze', 'arguments': '{"baz": "buz"}'}}], 'content': ''},
        # 中文: {'role': 'assistant', 'tool_calls': [{'type': 'function', 'id': '1','function': {'name': 'analyze', 'arguments': '{"baz": "buz"}'}}], 'content': ''},
        # {'角色': '工具', '名称': 'bar', '内容': 'foobar'},
        # {'角色': '助理', '内容': '那很好'}
        #]
        ````
    !!! version-added “在 `langchain-core` 0.3.11 中添加”"""  # noqa: E501
    if text_format not in {"string", "block"}:
        err = f"Unrecognized {text_format=}, expected one of 'string' or 'block'."
        raise ValueError(err)

    oai_messages: list[dict] = []

    if is_single := isinstance(messages, (BaseMessage, dict, str)):
        messages = [messages]

    messages = convert_to_messages(messages)

    for i, message in enumerate(messages):
        oai_msg: dict = {"role": _get_message_openai_role(message)}
        tool_messages: list = []
        content: str | list[dict]

        if message.name:
            oai_msg["name"] = message.name
        if isinstance(message, AIMessage) and message.tool_calls:
            oai_msg["tool_calls"] = _convert_to_openai_tool_calls(message.tool_calls)
        if message.additional_kwargs.get("refusal"):
            oai_msg["refusal"] = message.additional_kwargs["refusal"]
        if isinstance(message, ToolMessage):
            oai_msg["tool_call_id"] = message.tool_call_id
        if include_id and message.id:
            oai_msg["id"] = message.id

        if not message.content:
            content = "" if text_format == "string" else []
        elif isinstance(message.content, str):
            if text_format == "string":
                content = message.content
            else:
                content = [{"type": "text", "text": message.content}]
        elif text_format == "string" and all(
            isinstance(block, str) or block.get("type") == "text"
            for block in message.content
        ):
            content = "\n".join(
                block if isinstance(block, str) else block["text"]
                for block in message.content
            )
        else:
            content = []
            for j, block in enumerate(message.content):
                # OpenAI format
                # 中文: OpenAI 格式
                if isinstance(block, str):
                    content.append({"type": "text", "text": block})
                elif block.get("type") == "text":
                    if missing := [k for k in ("text",) if k not in block]:
                        err = (
                            f"Unrecognized content block at "
                            f"messages[{i}].content[{j}] has 'type': 'text' "
                            f"but is missing expected key(s) "
                            f"{missing}. Full content block:\n\n{block}"
                        )
                        raise ValueError(err)
                    content.append({"type": block["type"], "text": block["text"]})
                elif block.get("type") == "image_url":
                    if missing := [k for k in ("image_url",) if k not in block]:
                        err = (
                            f"Unrecognized content block at "
                            f"messages[{i}].content[{j}] has 'type': 'image_url' "
                            f"but is missing expected key(s) "
                            f"{missing}. Full content block:\n\n{block}"
                        )
                        raise ValueError(err)
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": block["image_url"],
                        }
                    )
                # Standard multi-modal content block
                # 中文: 标准多模式内容块
                elif is_data_content_block(block):
                    formatted_block = convert_to_openai_data_block(block)
                    if (
                        formatted_block.get("type") == "file"
                        and "file" in formatted_block
                        and "filename" not in formatted_block["file"]
                    ):
                        logger.info("Generating a fallback filename.")
                        formatted_block["file"]["filename"] = "LC_AUTOGENERATED"
                    content.append(formatted_block)
                # Anthropic and Bedrock converse format
                # 中文: Anthropic and Bedrock converse format
                elif (block.get("type") == "image") or "image" in block:
                    # Anthropic
                    # 中文: 人择
                    if source := block.get("source"):
                        if missing := [
                            k for k in ("media_type", "type", "data") if k not in source
                        ]:
                            err = (
                                f"Unrecognized content block at "
                                f"messages[{i}].content[{j}] has 'type': 'image' "
                                f"but 'source' is missing expected key(s) "
                                f"{missing}. Full content block:\n\n{block}"
                            )
                            raise ValueError(err)
                        content.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": (
                                        f"data:{source['media_type']};"
                                        f"{source['type']},{source['data']}"
                                    )
                                },
                            }
                        )
                    # Bedrock converse
                    # 中文: 基岩匡威
                    elif image := block.get("image"):
                        if missing := [
                            k for k in ("source", "format") if k not in image
                        ]:
                            err = (
                                f"Unrecognized content block at "
                                f"messages[{i}].content[{j}] has key 'image', "
                                f"but 'image' is missing expected key(s) "
                                f"{missing}. Full content block:\n\n{block}"
                            )
                            raise ValueError(err)
                        b64_image = _bytes_to_b64_str(image["source"]["bytes"])
                        content.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": (
                                        f"data:image/{image['format']};base64,{b64_image}"
                                    )
                                },
                            }
                        )
                    else:
                        err = (
                            f"Unrecognized content block at "
                            f"messages[{i}].content[{j}] has 'type': 'image' "
                            f"but does not have a 'source' or 'image' key. Full "
                            f"content block:\n\n{block}"
                        )
                        raise ValueError(err)
                # OpenAI file format
                # 中文: OpenAI 文件格式
                elif (
                    block.get("type") == "file"
                    and isinstance(block.get("file"), dict)
                    and isinstance(block.get("file", {}).get("file_data"), str)
                ):
                    if block.get("file", {}).get("filename") is None:
                        logger.info("Generating a fallback filename.")
                        block["file"]["filename"] = "LC_AUTOGENERATED"
                    content.append(block)
                # OpenAI audio format
                # 中文: OpenAI 音频格式
                elif (
                    block.get("type") == "input_audio"
                    and isinstance(block.get("input_audio"), dict)
                    and isinstance(block.get("input_audio", {}).get("data"), str)
                    and isinstance(block.get("input_audio", {}).get("format"), str)
                ):
                    content.append(block)
                elif block.get("type") == "tool_use":
                    if missing := [
                        k for k in ("id", "name", "input") if k not in block
                    ]:
                        err = (
                            f"Unrecognized content block at "
                            f"messages[{i}].content[{j}] has 'type': "
                            f"'tool_use', but is missing expected key(s) "
                            f"{missing}. Full content block:\n\n{block}"
                        )
                        raise ValueError(err)
                    if not any(
                        tool_call["id"] == block["id"]
                        for tool_call in cast("AIMessage", message).tool_calls
                    ):
                        oai_msg["tool_calls"] = oai_msg.get("tool_calls", [])
                        oai_msg["tool_calls"].append(
                            {
                                "type": "function",
                                "id": block["id"],
                                "function": {
                                    "name": block["name"],
                                    "arguments": json.dumps(
                                        block["input"], ensure_ascii=False
                                    ),
                                },
                            }
                        )
                elif block.get("type") == "function_call":  # OpenAI Responses
                    if not any(
                        tool_call["id"] == block.get("call_id")
                        for tool_call in cast("AIMessage", message).tool_calls
                    ):
                        if missing := [
                            k
                            for k in ("call_id", "name", "arguments")
                            if k not in block
                        ]:
                            err = (
                                f"Unrecognized content block at "
                                f"messages[{i}].content[{j}] has 'type': "
                                f"'tool_use', but is missing expected key(s) "
                                f"{missing}. Full content block:\n\n{block}"
                            )
                            raise ValueError(err)
                        oai_msg["tool_calls"] = oai_msg.get("tool_calls", [])
                        oai_msg["tool_calls"].append(
                            {
                                "type": "function",
                                "id": block.get("call_id"),
                                "function": {
                                    "name": block.get("name"),
                                    "arguments": block.get("arguments"),
                                },
                            }
                        )
                    if pass_through_unknown_blocks:
                        content.append(block)
                elif block.get("type") == "tool_result":
                    if missing := [
                        k for k in ("content", "tool_use_id") if k not in block
                    ]:
                        msg = (
                            f"Unrecognized content block at "
                            f"messages[{i}].content[{j}] has 'type': "
                            f"'tool_result', but is missing expected key(s) "
                            f"{missing}. Full content block:\n\n{block}"
                        )
                        raise ValueError(msg)
                    tool_message = ToolMessage(
                        block["content"],
                        tool_call_id=block["tool_use_id"],
                        status="error" if block.get("is_error") else "success",
                    )
                    # Recurse to make sure tool message contents are OpenAI format.
                    # 中文: 递归以确保工具消息内容为 OpenAI 格式。
                    tool_messages.extend(
                        convert_to_openai_messages(
                            [tool_message], text_format=text_format
                        )
                    )
                elif (block.get("type") == "json") or "json" in block:
                    if "json" not in block:
                        msg = (
                            f"Unrecognized content block at "
                            f"messages[{i}].content[{j}] has 'type': 'json' "
                            f"but does not have a 'json' key. Full "
                            f"content block:\n\n{block}"
                        )
                        raise ValueError(msg)
                    content.append(
                        {
                            "type": "text",
                            "text": json.dumps(block["json"]),
                        }
                    )
                elif (block.get("type") == "guard_content") or "guard_content" in block:
                    if (
                        "guard_content" not in block
                        or "text" not in block["guard_content"]
                    ):
                        msg = (
                            f"Unrecognized content block at "
                            f"messages[{i}].content[{j}] has 'type': "
                            f"'guard_content' but does not have a "
                            f"messages[{i}].content[{j}]['guard_content']['text'] "
                            f"key. Full content block:\n\n{block}"
                        )
                        raise ValueError(msg)
                    text = block["guard_content"]["text"]
                    if isinstance(text, dict):
                        text = text["text"]
                    content.append({"type": "text", "text": text})
                # VertexAI format
                # 中文: VertexAI格式
                elif block.get("type") == "media":
                    if missing := [k for k in ("mime_type", "data") if k not in block]:
                        err = (
                            f"Unrecognized content block at "
                            f"messages[{i}].content[{j}] has 'type': "
                            f"'media' but does not have key(s) {missing}. Full "
                            f"content block:\n\n{block}"
                        )
                        raise ValueError(err)
                    if "image" not in block["mime_type"]:
                        err = (
                            f"OpenAI messages can only support text and image data."
                            f" Received content block with media of type:"
                            f" {block['mime_type']}"
                        )
                        raise ValueError(err)
                    b64_image = _bytes_to_b64_str(block["data"])
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": (f"data:{block['mime_type']};base64,{b64_image}")
                            },
                        }
                    )
                elif (
                    block.get("type") in {"thinking", "reasoning"}
                    or pass_through_unknown_blocks
                ):
                    content.append(block)
                else:
                    err = (
                        f"Unrecognized content block at "
                        f"messages[{i}].content[{j}] does not match OpenAI, "
                        f"Anthropic, Bedrock Converse, or VertexAI format. Full "
                        f"content block:\n\n{block}"
                    )
                    raise ValueError(err)
            if text_format == "string" and not any(
                block["type"] != "text" for block in content
            ):
                content = "\n".join(block["text"] for block in content)
        oai_msg["content"] = content
        if message.content and not oai_msg["content"] and tool_messages:
            oai_messages.extend(tool_messages)
        else:
            oai_messages.extend([oai_msg, *tool_messages])

    if is_single:
        return oai_messages[0]
    return oai_messages


def _first_max_tokens(
    messages: Sequence[BaseMessage],
    *,
    max_tokens: int,
    token_counter: Callable[[list[BaseMessage]], int],
    text_splitter: Callable[[str], list[str]],
    partial_strategy: Literal["first", "last"] | None = None,
    end_on: str | type[BaseMessage] | Sequence[str | type[BaseMessage]] | None = None,
) -> list[BaseMessage]:
    messages = list(messages)
    if not messages:
        return messages

    # Check if all messages already fit within token limit
    # 中文: 检查所有消息是否已符合令牌限制
    if token_counter(messages) <= max_tokens:
        # When all messages fit, only apply end_on filtering if needed
        # 中文: 当所有消息都适合时，仅在需要时应用 end_on 过滤
        if end_on:
            for _ in range(len(messages)):
                if not _is_message_type(messages[-1], end_on):
                    messages.pop()
                else:
                    break
        return messages

    # Use binary search to find the maximum number of messages within token limit
    # 中文: 使用二分查找查找令牌限制内的最大消息数
    left, right = 0, len(messages)
    max_iterations = len(messages).bit_length()
    for _ in range(max_iterations):
        if left >= right:
            break
        mid = (left + right + 1) // 2
        if token_counter(messages[:mid]) <= max_tokens:
            left = mid
            idx = mid
        else:
            right = mid - 1

    # idx now contains the maximum number of complete messages we can include
    # 中文: idx 现在包含我们可以包含的完整消息的最大数量
    idx = left

    if partial_strategy and idx < len(messages):
        included_partial = False
        copied = False
        if isinstance(messages[idx].content, list):
            excluded = messages[idx].model_copy(deep=True)
            copied = True
            num_block = len(excluded.content)
            if partial_strategy == "last":
                excluded.content = list(reversed(excluded.content))
            for _ in range(1, num_block):
                excluded.content = excluded.content[:-1]
                if token_counter([*messages[:idx], excluded]) <= max_tokens:
                    messages = [*messages[:idx], excluded]
                    idx += 1
                    included_partial = True
                    break
            if included_partial and partial_strategy == "last":
                excluded.content = list(reversed(excluded.content))
        if not included_partial:
            if not copied:
                excluded = messages[idx].model_copy(deep=True)
                copied = True

            # Extract text content efficiently
            # 中文: 高效提取文本内容
            text = None
            if isinstance(excluded.content, str):
                text = excluded.content
            elif isinstance(excluded.content, list) and excluded.content:
                for block in excluded.content:
                    if isinstance(block, str):
                        text = block
                        break
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text")
                        break

            if text:
                if not copied:
                    excluded = excluded.model_copy(deep=True)

                split_texts = text_splitter(text)
                base_message_count = token_counter(messages[:idx])
                if partial_strategy == "last":
                    split_texts = list(reversed(split_texts))

                # Binary search for the maximum number of splits we can include
                # 中文: 二分搜索我们可以包含的最大分割数
                left, right = 0, len(split_texts)
                max_iterations = len(split_texts).bit_length()
                for _ in range(max_iterations):
                    if left >= right:
                        break
                    mid = (left + right + 1) // 2
                    excluded.content = "".join(split_texts[:mid])
                    if base_message_count + token_counter([excluded]) <= max_tokens:
                        left = mid
                    else:
                        right = mid - 1

                if left > 0:
                    content_splits = split_texts[:left]
                    if partial_strategy == "last":
                        content_splits = list(reversed(content_splits))
                    excluded.content = "".join(content_splits)
                    messages = [*messages[:idx], excluded]
                    idx += 1

    if end_on:
        for _ in range(idx):
            if idx > 0 and not _is_message_type(messages[idx - 1], end_on):
                idx -= 1
            else:
                break

    return messages[:idx]


def _last_max_tokens(
    messages: Sequence[BaseMessage],
    *,
    max_tokens: int,
    token_counter: Callable[[list[BaseMessage]], int],
    text_splitter: Callable[[str], list[str]],
    allow_partial: bool = False,
    include_system: bool = False,
    start_on: str | type[BaseMessage] | Sequence[str | type[BaseMessage]] | None = None,
    end_on: str | type[BaseMessage] | Sequence[str | type[BaseMessage]] | None = None,
) -> list[BaseMessage]:
    messages = list(messages)
    if len(messages) == 0:
        return []

    # Filter out messages after end_on type
    # 中文: 过滤掉end_on类型之后的消息
    if end_on:
        for _ in range(len(messages)):
            if not _is_message_type(messages[-1], end_on):
                messages.pop()
            else:
                break

    # Handle system message preservation
    # 中文: 处理系统消息保存
    system_message = None
    if include_system and len(messages) > 0 and isinstance(messages[0], SystemMessage):
        system_message = messages[0]
        messages = messages[1:]

    # Reverse messages to use _first_max_tokens with reversed logic
    # 中文: 反向消息以使用具有反向逻辑的 _first_max_tokens
    reversed_messages = messages[::-1]

    # Calculate remaining tokens after accounting for system message if present
    # 中文: 考虑系统消息（如果存在）后计算剩余令牌
    remaining_tokens = max_tokens
    if system_message:
        system_tokens = token_counter([system_message])
        remaining_tokens = max(0, max_tokens - system_tokens)

    reversed_result = _first_max_tokens(
        reversed_messages,
        max_tokens=remaining_tokens,
        token_counter=token_counter,
        text_splitter=text_splitter,
        partial_strategy="last" if allow_partial else None,
        end_on=start_on,
    )

    # Re-reverse the messages and add back the system message if needed
    # 中文: 如果需要，重新反转消息并添加回系统消息
    result = reversed_result[::-1]
    if system_message:
        result = [system_message, *result]

    return result


_MSG_CHUNK_MAP: dict[type[BaseMessage], type[BaseMessageChunk]] = {
    HumanMessage: HumanMessageChunk,
    AIMessage: AIMessageChunk,
    SystemMessage: SystemMessageChunk,
    ToolMessage: ToolMessageChunk,
    FunctionMessage: FunctionMessageChunk,
    ChatMessage: ChatMessageChunk,
}
_CHUNK_MSG_MAP = {v: k for k, v in _MSG_CHUNK_MAP.items()}


def _msg_to_chunk(message: BaseMessage) -> BaseMessageChunk:
    if message.__class__ in _MSG_CHUNK_MAP:
        return _MSG_CHUNK_MAP[message.__class__](**message.model_dump(exclude={"type"}))

    for msg_cls, chunk_cls in _MSG_CHUNK_MAP.items():
        if isinstance(message, msg_cls):
            return chunk_cls(**message.model_dump(exclude={"type"}))

    msg = (
        f"Unrecognized message class {message.__class__}. Supported classes are "
        f"{list(_MSG_CHUNK_MAP.keys())}"
    )
    msg = create_message(message=msg, error_code=ErrorCode.MESSAGE_COERCION_FAILURE)
    raise ValueError(msg)


def _chunk_to_msg(chunk: BaseMessageChunk) -> BaseMessage:
    if chunk.__class__ in _CHUNK_MSG_MAP:
        return _CHUNK_MSG_MAP[chunk.__class__](
            **chunk.model_dump(exclude={"type", "tool_call_chunks", "chunk_position"})
        )
    for chunk_cls, msg_cls in _CHUNK_MSG_MAP.items():
        if isinstance(chunk, chunk_cls):
            return msg_cls(
                **chunk.model_dump(
                    exclude={"type", "tool_call_chunks", "chunk_position"}
                )
            )

    msg = (
        f"Unrecognized message chunk class {chunk.__class__}. Supported classes are "
        f"{list(_CHUNK_MSG_MAP.keys())}"
    )
    msg = create_message(message=msg, error_code=ErrorCode.MESSAGE_COERCION_FAILURE)
    raise ValueError(msg)


def _default_text_splitter(text: str) -> list[str]:
    splits = text.split("\n")
    return [s + "\n" for s in splits[:-1]] + splits[-1:]


def _is_message_type(
    message: BaseMessage,
    type_: str | type[BaseMessage] | Sequence[str | type[BaseMessage]],
) -> bool:
    types = [type_] if isinstance(type_, (str, type)) else type_
    types_str = [t for t in types if isinstance(t, str)]
    types_types = tuple(t for t in types if isinstance(t, type))

    return message.type in types_str or isinstance(message, types_types)


def _bytes_to_b64_str(bytes_: bytes) -> str:
    return base64.b64encode(bytes_).decode("utf-8")


def _get_message_openai_role(message: BaseMessage) -> str:
    if isinstance(message, AIMessage):
        return "assistant"
    if isinstance(message, HumanMessage):
        return "user"
    if isinstance(message, ToolMessage):
        return "tool"
    if isinstance(message, SystemMessage):
        role = message.additional_kwargs.get("__openai_role__", "system")
        if not isinstance(role, str):
            msg = f"Expected '__openai_role__' to be a str, got {type(role).__name__}"
            raise TypeError(msg)
        return role
    if isinstance(message, FunctionMessage):
        return "function"
    if isinstance(message, ChatMessage):
        return message.role
    msg = f"Unknown BaseMessage type {message.__class__}."
    raise ValueError(msg)


def _convert_to_openai_tool_calls(tool_calls: list[ToolCall]) -> list[dict]:
    return [
        {
            "type": "function",
            "id": tool_call["id"],
            "function": {
                "name": tool_call["name"],
                "arguments": json.dumps(tool_call["args"], ensure_ascii=False),
            },
        }
        for tool_call in tool_calls
    ]


def count_tokens_approximately(
    messages: Iterable[MessageLikeRepresentation],
    *,
    chars_per_token: float = 4.0,
    extra_tokens_per_message: float = 3.0,
    count_name: bool = True,
) -> int:
    """Approximate the total number of tokens in messages.

    The token count includes stringified message content, role, and (optionally) name.

    - For AI messages, the token count also includes stringified tool calls.
    - For tool messages, the token count also includes the tool call ID.

    Args:
        messages: List of messages to count tokens for.
        chars_per_token: Number of characters per token to use for the approximation.

            One token corresponds to ~4 chars for common English text.

            You can also specify `float` values for more fine-grained control.
            [See more here](https://platform.openai.com/tokenizer).
        extra_tokens_per_message: Number of extra tokens to add per message, e.g.
            special tokens, including beginning/end of message.

            You can also specify `float` values for more fine-grained control.
            [See more here](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb).
        count_name: Whether to include message names in the count.

    Returns:
        Approximate number of tokens in the messages.

    Note:
        This is a simple approximation that may not match the exact token count used by
        specific models. For accurate counts, use model-specific tokenizers.

    Warning:
        This function does not currently support counting image tokens.

    !!! version-added "Added in `langchain-core` 0.3.46"
    

    中文翻译:
    估计消息中令牌的总数。
    令牌计数包括字符串化消息内容、角色和（可选）名称。
    - 对于 AI 消息，令牌计数还包括字符串化工具调用。
    - 对于工具消息，令牌计数还包括工具调用 ID。
    参数：
        messages：要计算令牌的消息列表。
        chars_per_token：用于近似的每个标记的字符数。
            对于常见的英语文本，1 个标记对应约 4 个字符。
            您还可以指定“float”值以进行更细粒度的控制。
            [在此处查看更多信息](https://platform.openai.com/tokenizer)。
        extra_tokens_per_message：每条消息添加的额外令牌的数量，例如
            特殊标记，包括消息的开始/结束。
            您还可以指定“float”值以进行更细粒度的控制。
            [在此处查看更多信息](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb)。
        count_name：计数中是否包含消息名称。
    返回：
        消息中令牌的大致数量。
    注意：
        这是一个简单的近似值，可能与所使用的确切令牌计数不匹配
        具体型号。为了获得准确的计数，请使用特定于模型的分词器。
    警告：
        该功能目前不支持对图像token进行计数。
    ！！！ version-added “在 `langchain-core` 0.3.46 中添加”"""
    token_count = 0.0
    for message in convert_to_messages(messages):
        message_chars = 0
        if isinstance(message.content, str):
            message_chars += len(message.content)

        # TODO: add support for approximate counting for image blocks
        else:
            content = repr(message.content)
            message_chars += len(content)

        if (
            isinstance(message, AIMessage)
            # exclude Anthropic format as tool calls are already included in the content
            # 中文: 排除 Anthropic 格式，因为工具调用已包含在内容中
            and not isinstance(message.content, list)
            and message.tool_calls
        ):
            tool_calls_content = repr(message.tool_calls)
            message_chars += len(tool_calls_content)

        if isinstance(message, ToolMessage):
            message_chars += len(message.tool_call_id)

        role = _get_message_openai_role(message)
        message_chars += len(role)

        if message.name and count_name:
            message_chars += len(message.name)

        # NOTE: we're rounding up per message to ensure that
        # individual message token counts add up to the total count
        # 中文: 单个消息令牌计数加起来等于总计数
        # for a list of messages
        # 中文: 获取消息列表
        token_count += math.ceil(message_chars / chars_per_token)

        # add extra tokens per message
        # 中文: 为每条消息添加额外的令牌
        token_count += extra_tokens_per_message

    # round up once more time in case extra_tokens_per_message is a float
    # 中文: 如果 extra_tokens_per_message 是浮点数，则再次四舍五入
    return math.ceil(token_count)


# Mapping from string shortcuts to token counter functions
# 中文: 从字符串快捷方式到令牌计数器函数的映射
def _approximate_token_counter(messages: Sequence[BaseMessage]) -> int:
    """Wrapper for `count_tokens_approximately` that matches expected signature.

    中文翻译:
    与预期签名匹配的“count_tokens_approximately”的包装器。"""
    return count_tokens_approximately(messages)


_TOKEN_COUNTER_SHORTCUTS = {
    "approximate": _approximate_token_counter,
}
