"""Chat generation output classes.

中文翻译:
聊天生成输出类。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import model_validator

from langchain_core.messages import BaseMessage, BaseMessageChunk
from langchain_core.outputs.generation import Generation
from langchain_core.utils._merge import merge_dicts

if TYPE_CHECKING:
    from typing_extensions import Self


class ChatGeneration(Generation):
    """A single chat generation output.

    A subclass of `Generation` that represents the response from a chat model
    that generates chat messages.

    The `message` attribute is a structured representation of the chat message.
    Most of the time, the message will be of type `AIMessage`.

    Users working with chat models will usually access information via either
    `AIMessage` (returned from runnable interfaces) or `LLMResult` (available
    via callbacks).
    

    中文翻译:
    单个聊天生成输出。
    “Generation”的子类，表示聊天模型的响应
    生成聊天消息。
    “message”属性是聊天消息的结构化表示。
    大多数时候，消息的类型为“AIMessage”。
    使用聊天模型的用户通常会通过以下任一方式访问信息
    `AIMessage`（从可运行接口返回）或`LLMResult`（可用
    通过回调）。"""

    text: str = ""
    """The text contents of the output message.

    !!! warning
        SHOULD NOT BE SET DIRECTLY!

    

    中文翻译:
    输出消息的文本内容。
    !!!警告
        不应该直接设置！"""
    message: BaseMessage
    """The message output by the chat model.

    中文翻译:
    聊天模型输出的消息。"""
    # Override type to be ChatGeneration, ignore mypy error as this is intentional
    # 中文: 将类型覆盖为 ChatGeneration，忽略 mypy 错误，因为这是故意的
    type: Literal["ChatGeneration"] = "ChatGeneration"  # type: ignore[assignment]
    """Type is used exclusively for serialization purposes.

    中文翻译:
    类型专门用于序列化目的。"""

    @model_validator(mode="after")
    def set_text(self) -> Self:
        """Set the text attribute to be the contents of the message.

        Args:
            values: The values of the object.

        Returns:
            The values of the object with the text attribute set.

        Raises:
            ValueError: If the message is not a string or a list.
        

        中文翻译:
        将文本属性设置为消息的内容。
        参数：
            值：对象的值。
        返回：
            设置了文本属性的对象的值。
        加薪：
            ValueError：如果消息不是字符串或列表。"""
        text = ""
        if isinstance(self.message.content, str):
            text = self.message.content
        # Assumes text in content blocks in OpenAI format.
        # 中文: 假定内容块中的文本采用 OpenAI 格式。
        # Uses first text block.
        # 中文: 使用第一个文本块。
        elif isinstance(self.message.content, list):
            for block in self.message.content:
                if isinstance(block, str):
                    text = block
                    break
                if isinstance(block, dict) and "text" in block:
                    text = block["text"]
                    break
        self.text = text
        return self


class ChatGenerationChunk(ChatGeneration):
    """`ChatGeneration` chunk.

    `ChatGeneration` chunks can be concatenated with other `ChatGeneration` chunks.
    

    中文翻译:
    `ChatGeneration` 块。
    `ChatGeneration` 块可以与其他 `ChatGeneration` 块连接。"""

    message: BaseMessageChunk
    """The message chunk output by the chat model.

    中文翻译:
    聊天模型输出的消息块。"""
    # Override type to be ChatGeneration, ignore mypy error as this is intentional
    # 中文: 将类型覆盖为 ChatGeneration，忽略 mypy 错误，因为这是故意的
    type: Literal["ChatGenerationChunk"] = "ChatGenerationChunk"  # type: ignore[assignment]
    """Type is used exclusively for serialization purposes.

    中文翻译:
    类型专门用于序列化目的。"""

    def __add__(
        self, other: ChatGenerationChunk | list[ChatGenerationChunk]
    ) -> ChatGenerationChunk:
        """Concatenate two `ChatGenerationChunk`s.

        Args:
            other: The other `ChatGenerationChunk` or list of `ChatGenerationChunk`
                to concatenate.

        Raises:
            TypeError: If other is not a `ChatGenerationChunk` or list of
                `ChatGenerationChunk`.

        Returns:
            A new `ChatGenerationChunk` concatenated from self and other.
        

        中文翻译:
        连接两个“ChatGenerationChunk”。
        参数：
            other: 其他 `ChatGenerationChunk` 或 `ChatGenerationChunk` 列表
                连接。
        加薪：
            类型错误：如果 other 不是 `ChatGenerationChunk` 或列表
                `ChatGenerationChunk`。
        返回：
            一个新的“ChatGenerationChunk”由自己和其他人连接而成。"""
        if isinstance(other, ChatGenerationChunk):
            generation_info = merge_dicts(
                self.generation_info or {},
                other.generation_info or {},
            )
            return ChatGenerationChunk(
                message=self.message + other.message,
                generation_info=generation_info or None,
            )
        if isinstance(other, list) and all(
            isinstance(x, ChatGenerationChunk) for x in other
        ):
            generation_info = merge_dicts(
                self.generation_info or {},
                *[chunk.generation_info for chunk in other if chunk.generation_info],
            )
            return ChatGenerationChunk(
                message=self.message + [chunk.message for chunk in other],
                generation_info=generation_info or None,
            )
        msg = f"unsupported operand type(s) for +: '{type(self)}' and '{type(other)}'"
        raise TypeError(msg)


def merge_chat_generation_chunks(
    chunks: list[ChatGenerationChunk],
) -> ChatGenerationChunk | None:
    """Merge a list of `ChatGenerationChunk`s into a single `ChatGenerationChunk`.

    Args:
        chunks: A list of `ChatGenerationChunk` to merge.

    Returns:
        A merged `ChatGenerationChunk`, or None if the input list is empty.
    

    中文翻译:
    将“ChatGenerationChunk”列表合并为单个“ChatGenerationChunk”。
    参数：
        chunks：要合并的“ChatGenerationChunk”列表。
    返回：
        合并的“ChatGenerationChunk”，如果输入列表为空，则为 None。"""
    if not chunks:
        return None

    if len(chunks) == 1:
        return chunks[0]

    return chunks[0] + chunks[1:]
