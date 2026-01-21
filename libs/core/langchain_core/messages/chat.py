"""Chat Message.

中文翻译:
聊天消息。"""

from typing import Any, Literal

from typing_extensions import override

from langchain_core.messages.base import (
    BaseMessage,
    BaseMessageChunk,
    merge_content,
)
from langchain_core.utils._merge import merge_dicts


class ChatMessage(BaseMessage):
    """Message that can be assigned an arbitrary speaker (i.e. role).

    中文翻译:
    可以分配任意发言者（即角色）的消息。"""

    role: str
    """The speaker / role of the Message.

    中文翻译:
    消息的发言者/角色。"""

    type: Literal["chat"] = "chat"
    """The type of the message (used during serialization).

    中文翻译:
    消息的类型（在序列化期间使用）。"""


class ChatMessageChunk(ChatMessage, BaseMessageChunk):
    """Chat Message chunk.

    中文翻译:
    聊天消息块。"""

    # Ignoring mypy re-assignment here since we're overriding the value
    # 中文: 此处忽略 mypy 重新分配，因为我们要覆盖该值
    # to make sure that the chunk variant can be discriminated from the
    # 中文: 以确保可以将块变体与
    # non-chunk variant.
    # 中文: 非块变体。
    type: Literal["ChatMessageChunk"] = "ChatMessageChunk"  # type: ignore[assignment]
    """The type of the message (used during serialization).

    中文翻译:
    消息的类型（在序列化期间使用）。"""

    @override
    def __add__(self, other: Any) -> BaseMessageChunk:  # type: ignore[override]
        if isinstance(other, ChatMessageChunk):
            if self.role != other.role:
                msg = "Cannot concatenate ChatMessageChunks with different roles."
                raise ValueError(msg)

            return self.__class__(
                role=self.role,
                content=merge_content(self.content, other.content),
                additional_kwargs=merge_dicts(
                    self.additional_kwargs, other.additional_kwargs
                ),
                response_metadata=merge_dicts(
                    self.response_metadata, other.response_metadata
                ),
                id=self.id,
            )
        if isinstance(other, BaseMessageChunk):
            return self.__class__(
                role=self.role,
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
