"""Human message.

中文翻译:
人类的讯息。"""

from typing import Any, Literal, cast, overload

from langchain_core.messages import content as types
from langchain_core.messages.base import BaseMessage, BaseMessageChunk


class HumanMessage(BaseMessage):
    """Message from the user.

    A `HumanMessage` is a message that is passed in from a user to the model.

    Example:
        ```python
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content="You are a helpful assistant! Your name is Bob."),
            HumanMessage(content="What is your name?"),
        ]

        # Instantiate a chat model and invoke it with the messages
        # 中文: 实例化聊天模型并使用消息调用它
        model = ...
        print(model.invoke(messages))
        ```
    

    中文翻译:
    来自用户的消息。
    “HumanMessage”是从用户传递到模型的消息。
    示例：
        ````蟒蛇
        从 langchain_core.messages 导入 HumanMessage、SystemMessage
        消息 = [
            SystemMessage(content="你是一个有用的助手！你的名字是鲍勃。"),
            HumanMessage(content="你叫什么名字？"),
        ]
        # 实例化聊天模型并使用消息调用它
        型号=...
        打印（模型.调用（消息））
        ````"""

    type: Literal["human"] = "human"
    """The type of the message (used for serialization).

    中文翻译:
    消息的类型（用于序列化）。"""

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
        """Specify `content` as positional arg or `content_blocks` for typing.

        中文翻译:
        将 `content` 指定为位置参数或用于输入的 `content_blocks`。"""
        if content_blocks is not None:
            super().__init__(
                content=cast("str | list[str | dict]", content_blocks),
                **kwargs,
            )
        else:
            super().__init__(content=content, **kwargs)


class HumanMessageChunk(HumanMessage, BaseMessageChunk):
    """Human Message chunk.

    中文翻译:
    人类消息块。"""

    # Ignoring mypy re-assignment here since we're overriding the value
    # 中文: 此处忽略 mypy 重新分配，因为我们要覆盖该值
    # to make sure that the chunk variant can be discriminated from the
    # 中文: 以确保可以将块变体与
    # non-chunk variant.
    # 中文: 非块变体。
    type: Literal["HumanMessageChunk"] = "HumanMessageChunk"  # type: ignore[assignment]
    """The type of the message (used for serialization).

    中文翻译:
    消息的类型（用于序列化）。"""
