"""Message responsible for deleting other messages.

中文翻译:
负责删除其他消息的消息。"""

from typing import Any, Literal

from langchain_core.messages.base import BaseMessage


class RemoveMessage(BaseMessage):
    """Message responsible for deleting other messages.

    中文翻译:
    负责删除其他消息的消息。"""

    type: Literal["remove"] = "remove"
    """The type of the message (used for serialization).

    中文翻译:
    消息的类型（用于序列化）。"""

    def __init__(
        self,
        id: str,
        **kwargs: Any,
    ) -> None:
        """Create a RemoveMessage.

        Args:
            id: The ID of the message to remove.
            **kwargs: Additional fields to pass to the message.

        Raises:
            ValueError: If the 'content' field is passed in kwargs.

        

        中文翻译:
        创建删除消息。
        参数：
            id：要删除的消息的 ID。
            **kwargs：传递给消息的附加字段。
        加薪：
            ValueError：如果“内容”字段以 kwargs 形式传递。"""
        if kwargs.pop("content", None):
            msg = "RemoveMessage does not support 'content' field."
            raise ValueError(msg)

        super().__init__("", id=id, **kwargs)
