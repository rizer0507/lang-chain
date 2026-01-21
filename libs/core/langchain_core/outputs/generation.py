"""Generation output schema.

中文翻译:
生成输出模式。"""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.load import Serializable
from langchain_core.utils._merge import merge_dicts


class Generation(Serializable):
    """A single text generation output.

    Generation represents the response from an "old-fashioned" LLM (string-in,
    string-out) that generates regular text (not chat messages).

    This model is used internally by chat model and will eventually
    be mapped to a more general `LLMResult` object, and then projected into
    an `AIMessage` object.

    LangChain users working with chat models will usually access information via
    `AIMessage` (returned from runnable interfaces) or `LLMResult` (available
    via callbacks). Please refer to `AIMessage` and `LLMResult` for more information.
    

    中文翻译:
    单个文本生成输出。
    Generation 代表“老式”LLM（字符串输入、
    string-out）生成常规文本（不是聊天消息）。
    该模型由聊天模型内部使用，最终将
    映射到更通用的“LLMResult”对象，然后投影到
    一个“AIMessage”对象。
    使用聊天模型的 LangChain 用户通常会通过以下方式访问信息
    `AIMessage`（从可运行接口返回）或`LLMResult`（可用
    通过回调）。请参阅“AIMessage”和“LLMResult”了解更多信息。"""

    text: str
    """Generated text output.

    中文翻译:
    生成的文本输出。"""

    generation_info: dict[str, Any] | None = None
    """Raw response from the provider.

    May include things like the reason for finishing or token log probabilities.
    

    中文翻译:
    来自提供商的原始响应。
    可能包括诸如完成原因或令牌日志概率之类的内容。"""
    type: Literal["Generation"] = "Generation"
    """Type is used exclusively for serialization purposes.

    Set to "Generation" for this class.
    

    中文翻译:
    类型专门用于序列化目的。
    将该类设置为“Generation”。"""

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return `True` as this class is serializable.

        中文翻译:
        返回“True”，因为此类是可序列化的。"""
        return True

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the LangChain object.

        Returns:
            `["langchain", "schema", "output"]`
        

        中文翻译:
        获取LangChain对象的命名空间。
        返回：
            `[“langchain”，“模式”，“输出”]`"""
        return ["langchain", "schema", "output"]


class GenerationChunk(Generation):
    """`GenerationChunk`, which can be concatenated with other Generation chunks.

    中文翻译:
    `GenerationChunk`，可以与其他 Generation 块连接。"""

    def __add__(self, other: GenerationChunk) -> GenerationChunk:
        """Concatenate two `GenerationChunk`s.

        Args:
            other: Another `GenerationChunk` to concatenate with.

        Raises:
            TypeError: If other is not a `GenerationChunk`.

        Returns:
            A new `GenerationChunk` concatenated from self and other.
        

        中文翻译:
        连接两个“GenerationChunk”。
        参数：
            other：另一个要连接的“GenerationChunk”。
        加薪：
            类型错误：如果 other 不是 `GenerationChunk`。
        返回：
            一个新的“GenerationChunk”由 self 和 other 连接而成。"""
        if isinstance(other, GenerationChunk):
            generation_info = merge_dicts(
                self.generation_info or {},
                other.generation_info or {},
            )
            return GenerationChunk(
                text=self.text + other.text,
                generation_info=generation_info or None,
            )
        msg = f"unsupported operand type(s) for +: '{type(self)}' and '{type(other)}'"
        raise TypeError(msg)
