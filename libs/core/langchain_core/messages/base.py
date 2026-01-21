"""Base message.

中文翻译:
基本消息。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast, overload

from pydantic import ConfigDict, Field

from langchain_core._api.deprecation import warn_deprecated
from langchain_core.load.serializable import Serializable
from langchain_core.utils import get_bolded_text
from langchain_core.utils._merge import merge_dicts, merge_lists
from langchain_core.utils.interactive_env import is_interactive_env

if TYPE_CHECKING:
    from collections.abc import Sequence

    from typing_extensions import Self

    from langchain_core.messages import content as types
    from langchain_core.prompts.chat import ChatPromptTemplate


def _extract_reasoning_from_additional_kwargs(
    message: BaseMessage,
) -> types.ReasoningContentBlock | None:
    """Extract `reasoning_content` from `additional_kwargs`.

    Handles reasoning content stored in various formats:
    - `additional_kwargs["reasoning_content"]` (string) - Ollama, DeepSeek, XAI, Groq

    Args:
        message: The message to extract reasoning from.

    Returns:
        A `ReasoningContentBlock` if reasoning content is found, None otherwise.
    

    中文翻译:
    从“additional_kwargs”中提取“reasoning_content”。
    处理以各种格式存储的推理内容：
    - `additional_kwargs["reasoning_content"]` (字符串) - Ollama、DeepSeek、XAI、Groq
    参数：
        message：从中提取推理的消息。
    返回：
        如果找到推理内容，则为“ReasoningContentBlock”，否则为 None。"""
    additional_kwargs = getattr(message, "additional_kwargs", {})

    reasoning_content = additional_kwargs.get("reasoning_content")
    if reasoning_content is not None and isinstance(reasoning_content, str):
        return {"type": "reasoning", "reasoning": reasoning_content}

    return None


class TextAccessor(str):
    """String-like object that supports both property and method access patterns.

    Exists to maintain backward compatibility while transitioning from method-based to
    property-based text access in message objects. In LangChain <v1.0, message text was
    accessed via `.text()` method calls. In v1.0=<, the preferred pattern is property
    access via `.text`.

    Rather than breaking existing code immediately, `TextAccessor` allows both
    patterns:
    - Modern property access: `message.text` (returns string directly)
    - Legacy method access: `message.text()` (callable, emits deprecation warning)

    

    中文翻译:
    支持属性和方法访问模式的类字符串对象。
    存在是为了在从基于方法过渡到基于方法的过程中保持向后兼容性
    消息对象中基于属性的文本访问。在 LangChain <v1.0 中，消息文本为
    通过“.text()”方法调用访问。在 v1.0=< 中，首选模式是属性
    通过“.text”访问。
    `TextAccessor` 不会立即破坏现有代码，而是允许两者
    模式：
    - 现代属性访问：`message.text`（直接返回字符串）
    - 旧方法访问：`message.text()`（可调用，发出弃用警告）"""

    __slots__ = ()

    def __new__(cls, value: str) -> Self:
        """Create new TextAccessor instance.

        中文翻译:
        创建新的 TextAccessor 实例。"""
        return str.__new__(cls, value)

    def __call__(self) -> str:
        """Enable method-style text access for backward compatibility.

        This method exists solely to support legacy code that calls `.text()`
        as a method. New code should use property access (`.text`) instead.

        !!! deprecated
            As of `langchain-core` 1.0.0, calling `.text()` as a method is deprecated.
            Use `.text` as a property instead. This method will be removed in 2.0.0.

        Returns:
            The string content, identical to property access.

        

        中文翻译:
        启用方法样式文本访问以实现向后兼容性。
        此方法的存在只是为了支持调用“.text()”的遗留代码
        作为一种方法。新代码应该使用属性访问（`.text`）。
        ！！！已弃用
            从 langchain-core 1.0.0 开始，不推荐调用 .text() 作为方法。
            请使用“.text”作为属性。该方法将在2.0.0中被删除。
        返回：
            字符串内容，与属性访问相同。"""
        warn_deprecated(
            since="1.0.0",
            message=(
                "Calling .text() as a method is deprecated. "
                "Use .text as a property instead (e.g., message.text)."
            ),
            removal="2.0.0",
        )
        return str(self)


class BaseMessage(Serializable):
    """Base abstract message class.

    Messages are the inputs and outputs of a chat model.

    Examples include [`HumanMessage`][langchain.messages.HumanMessage],
    [`AIMessage`][langchain.messages.AIMessage], and
    [`SystemMessage`][langchain.messages.SystemMessage].
    

    中文翻译:
    基本抽象消息类。
    消息是聊天模型的输入和输出。
    示例包括 [`HumanMessage`][langchain.messages.HumanMessage]，
    [`AIMessage`][langchain.messages.AIMessage]，以及
    [`SystemMessage`][langchain.messages.SystemMessage]。"""

    content: str | list[str | dict]
    """The contents of the message.

    中文翻译:
    消息的内容。"""

    additional_kwargs: dict = Field(default_factory=dict)
    """Reserved for additional payload data associated with the message.

    For example, for a message from an AI, this could include tool calls as
    encoded by the model provider.

    

    中文翻译:
    保留用于与消息关联的附加有效负载数据。
    例如，对于来自人工智能的消息，这可能包括工具调用
    由模型提供者编码。"""

    response_metadata: dict = Field(default_factory=dict)
    """Examples: response headers, logprobs, token counts, model name.

    中文翻译:
    示例：响应标头、日志概率、令牌计数、模型名称。"""

    type: str
    """The type of the message. Must be a string that is unique to the message type.

    The purpose of this field is to allow for easy identification of the message type
    when deserializing messages.

    

    中文翻译:
    消息的类型。必须是消息类型唯一的字符串。
    该字段的目的是为了方便识别消息类型
    反序列化消息时。"""

    name: str | None = None
    """An optional name for the message.

    This can be used to provide a human-readable name for the message.

    Usage of this field is optional, and whether it's used or not is up to the
    model implementation.

    

    中文翻译:
    消息的可选名称。
    这可用于为消息提供人类可读的名称。
    该字段的使用是可选的，是否使用由
    模型实施。"""

    id: str | None = Field(default=None, coerce_numbers_to_str=True)
    """An optional unique identifier for the message.

    This should ideally be provided by the provider/model which created the message.

    

    中文翻译:
    消息的可选唯一标识符。
    理想情况下，这应该由创建消息的提供者/模型提供。"""

    model_config = ConfigDict(
        extra="allow",
    )

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
        """Initialize a `BaseMessage`.

        Specify `content` as positional arg or `content_blocks` for typing.

        Args:
            content: The contents of the message.
            content_blocks: Typed standard content.
            **kwargs: Additional arguments to pass to the parent class.
        

        中文翻译:
        初始化一个“BaseMessage”。
        将 `content` 指定为位置参数或用于输入的 `content_blocks`。
        参数：
            内容：消息的内容。
            content_blocks：键入的标准内容。
            **kwargs：传递给父类的附加参数。"""
        if content_blocks is not None:
            super().__init__(content=content_blocks, **kwargs)
        else:
            super().__init__(content=content, **kwargs)

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """`BaseMessage` is serializable.

        Returns:
            True
        

        中文翻译:
        `BaseMessage` 是可序列化的。
        返回：
            真实"""
        return True

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the LangChain object.

        Returns:
            `["langchain", "schema", "messages"]`
        

        中文翻译:
        获取LangChain对象的命名空间。
        返回：
            `[“langchain”、“模式”、“消息”]`"""
        return ["langchain", "schema", "messages"]

    @property
    def content_blocks(self) -> list[types.ContentBlock]:
        r"""Load content blocks from the message content.

        !!! version-added "Added in `langchain-core` 1.0.0"

        

中文翻译:
从消息内容中加载内容块。
        !!! version-added “在 `langchain-core` 1.0.0 中添加”"""
        # Needed here to avoid circular import, as these classes import BaseMessages
        # 中文: 这里需要避免循环导入，因为这些类导入 BaseMessages
        from langchain_core.messages import content as types  # noqa: PLC0415
        from langchain_core.messages.block_translators.anthropic import (  # noqa: PLC0415
            _convert_to_v1_from_anthropic_input,
        )
        from langchain_core.messages.block_translators.bedrock_converse import (  # noqa: PLC0415
            _convert_to_v1_from_converse_input,
        )
        from langchain_core.messages.block_translators.google_genai import (  # noqa: PLC0415
            _convert_to_v1_from_genai_input,
        )
        from langchain_core.messages.block_translators.langchain_v0 import (  # noqa: PLC0415
            _convert_v0_multimodal_input_to_v1,
        )
        from langchain_core.messages.block_translators.openai import (  # noqa: PLC0415
            _convert_to_v1_from_chat_completions_input,
        )

        blocks: list[types.ContentBlock] = []
        content = (
            # Transpose string content to list, otherwise assumed to be list
            # 中文: 将字符串内容转置为列表，否则假定为列表
            [self.content]
            if isinstance(self.content, str) and self.content
            else self.content
        )
        for item in content:
            if isinstance(item, str):
                # Plain string content is treated as a text block
                # 中文: 纯字符串内容被视为文本块
                blocks.append({"type": "text", "text": item})
            elif isinstance(item, dict):
                item_type = item.get("type")
                if item_type not in types.KNOWN_BLOCK_TYPES:
                    # Handle all provider-specific or None type blocks as non-standard -
                    # 中文: 将所有特定于提供者或 None 类型的块作为非标准处理 -
                    # we'll come back to these later
                    # 中文: 我们稍后再讨论这些
                    blocks.append({"type": "non_standard", "value": item})
                else:
                    # Guard against v0 blocks that share the same `type` keys
                    # 中文: 防范共享相同“type”键的 v0 块
                    if "source_type" in item:
                        blocks.append({"type": "non_standard", "value": item})
                        continue

                    # This can't be a v0 block (since they require `source_type`),
                    # 中文: 这不能是 v0 块（因为它们需要 `source_type`），
                    # so it's a known v1 block type
                    # 中文: 所以它是已知的 v1 块类型
                    blocks.append(cast("types.ContentBlock", item))

        # Subsequent passes: attempt to unpack non-standard blocks.
        # 中文: 后续通道：尝试解压非标准块。
        # This is the last stop - if we can't parse it here, it is left as non-standard
        # 中文: 这是最后一站 - 如果我们无法在这里解析它，它将被视为非标准
        for parsing_step in [
            _convert_v0_multimodal_input_to_v1,
            _convert_to_v1_from_chat_completions_input,
            _convert_to_v1_from_anthropic_input,
            _convert_to_v1_from_genai_input,
            _convert_to_v1_from_converse_input,
        ]:
            blocks = parsing_step(blocks)
        return blocks

    @property
    def text(self) -> TextAccessor:
        """Get the text content of the message as a string.

        Can be used as both property (`message.text`) and method (`message.text()`).

        Handles both string and list content types (e.g. for content blocks). Only
        extracts blocks with `type: 'text'`; other block types are ignored.

        !!! deprecated
            As of `langchain-core` 1.0.0, calling `.text()` as a method is deprecated.
            Use `.text` as a property instead. This method will be removed in 2.0.0.

        Returns:
            The text content of the message.

        

        中文翻译:
        获取字符串形式的消息文本内容。
        可以用作属性（`message.text`）和方法（`message.text()`）。
        处理字符串和列表内容类型（例如内容块）。仅
        提取带有 `type: 'text'` 的块；其他块类型将被忽略。
        !!!已弃用
            从 langchain-core 1.0.0 开始，不推荐调用 .text() 作为方法。
            请使用“.text”作为属性。该方法将在2.0.0中被删除。
        返回：
            消息的文本内容。"""
        if isinstance(self.content, str):
            text_value = self.content
        else:
            # Must be a list
            # 中文: 必须是一个列表
            blocks = [
                block
                for block in self.content
                if isinstance(block, str)
                or (block.get("type") == "text" and isinstance(block.get("text"), str))
            ]
            text_value = "".join(
                block if isinstance(block, str) else block["text"] for block in blocks
            )
        return TextAccessor(text_value)

    def __add__(self, other: Any) -> ChatPromptTemplate:
        """Concatenate this message with another message.

        Args:
            other: Another message to concatenate with this one.

        Returns:
            A ChatPromptTemplate containing both messages.
        

        中文翻译:
        将此消息与另一消息连接起来。
        参数：
            other：与此消息连接的另一条消息。
        返回：
            包含两条消息的 ChatPromptTemplate。"""
        # Import locally to prevent circular imports.
        # 中文: 本地导入，防止循环导入。
        from langchain_core.prompts.chat import ChatPromptTemplate  # noqa: PLC0415

        prompt = ChatPromptTemplate(messages=[self])
        return prompt.__add__(other)

    def pretty_repr(
        self,
        html: bool = False,  # noqa: FBT001,FBT002
    ) -> str:
        """Get a pretty representation of the message.

        Args:
            html: Whether to format the message as HTML. If `True`, the message will be
                formatted with HTML tags.

        Returns:
            A pretty representation of the message.

        

        中文翻译:
        获得消息的漂亮表示。
        参数：
            html: Whether to format the message as HTML. If `True`, the message will be
                使用 HTML 标签格式化。
        返回：
            很好地表达了该消息。"""
        title = get_msg_title_repr(self.type.title() + " Message", bold=html)
        # TODO: handle non-string content.
        if self.name is not None:
            title += f"\nName: {self.name}"
        return f"{title}\n\n{self.content}"

    def pretty_print(self) -> None:
        """Print a pretty representation of the message.

        中文翻译:
        打印消息的漂亮表示。"""
        print(self.pretty_repr(html=is_interactive_env()))  # noqa: T201


def merge_content(
    first_content: str | list[str | dict],
    *contents: str | list[str | dict],
) -> str | list[str | dict]:
    """Merge multiple message contents.

    Args:
        first_content: The first `content`. Can be a string or a list.
        contents: The other `content`s. Can be a string or a list.

    Returns:
        The merged content.

    

    中文翻译:
    合并多个消息内容。
    参数：
        第一个内容：第一个“内容”。可以是字符串或列表。
        内容：其他“内容”。可以是字符串或列表。
    返回：
        合并后的内容。"""
    merged: str | list[str | dict]
    merged = "" if first_content is None else first_content

    for content in contents:
        # If current is a string
        # 中文: 如果当前是一个字符串
        if isinstance(merged, str):
            # If the next chunk is also a string, then merge them naively
            # 中文: 如果下一个块也是字符串，则天真地合并它们
            if isinstance(content, str):
                merged += content
            # If the next chunk is a list, add the current to the start of the list
            # 中文: 如果下一个块是列表，则将当前块添加到列表的开头
            else:
                merged = [merged, *content]
        elif isinstance(content, list):
            # If both are lists
            # 中文: 如果两者都是列表
            merged = merge_lists(cast("list", merged), content)  # type: ignore[assignment]
        # If the first content is a list, and the second content is a string
        # 中文: 如果第一个内容是列表，第二个内容是字符串
        # If the last element of the first content is a string
        # 中文: 如果第一个内容的最后一个元素是字符串
        # Add the second content to the last element
        # 中文: 将第二个内容添加到最后一个元素
        elif merged and isinstance(merged[-1], str):
            merged[-1] += content
        # If second content is an empty string, treat as a no-op
        # 中文: 如果第二个内容是空字符串，则视为无操作
        elif content == "":
            pass
        # Otherwise, add the second content as a new element of the list
        # 中文: 否则，将第二个内容添加为列表的新元素
        elif merged:
            merged.append(content)
    return merged


class BaseMessageChunk(BaseMessage):
    """Message chunk, which can be concatenated with other Message chunks.

    中文翻译:
    消息块，可以与其他消息块连接。"""

    def __add__(self, other: Any) -> BaseMessageChunk:  # type: ignore[override]
        """Message chunks support concatenation with other message chunks.

        This functionality is useful to combine message chunks yielded from
        a streaming model into a complete message.

        Args:
            other: Another message chunk to concatenate with this one.

        Returns:
            A new message chunk that is the concatenation of this message chunk
            and the other message chunk.

        Raises:
            TypeError: If the other object is not a message chunk.

        Example:
            ```txt
              AIMessageChunk(content="Hello", ...)
            + AIMessageChunk(content=" World", ...)
            = AIMessageChunk(content="Hello World", ...)
            ```
        

        中文翻译:
        消息块支持与其他消息块的串联。
        此功能对于组合从以下位置生成的消息块非常有用
        将流模型转化为完整的消息。
        参数：
            other：与此消息块连接的另一个消息块。
        返回：
            一个新的消息块，它是该消息块的串联
            和其他消息块。
        加薪：
            类型错误：如果其他对象不是消息块。
        示例：
            ````txt
              AIMessageChunk(内容=“你好”，...)
            + AIMessageChunk(内容=“世界”，...)
            = AIMessageChunk(内容=“你好世界”，...)
            ````"""
        if isinstance(other, BaseMessageChunk):
            # If both are (subclasses of) BaseMessageChunk,
            # 中文: 如果两者都是 BaseMessageChunk（的子类），
            # concat into a single BaseMessageChunk
            # 中文: concat 成单个 BaseMessageChunk

            return self.__class__(
                id=self.id,
                type=self.type,
                content=merge_content(self.content, other.content),
                additional_kwargs=merge_dicts(
                    self.additional_kwargs, other.additional_kwargs
                ),
                response_metadata=merge_dicts(
                    self.response_metadata, other.response_metadata
                ),
            )
        if isinstance(other, list) and all(
            isinstance(o, BaseMessageChunk) for o in other
        ):
            content = merge_content(self.content, *(o.content for o in other))
            additional_kwargs = merge_dicts(
                self.additional_kwargs, *(o.additional_kwargs for o in other)
            )
            response_metadata = merge_dicts(
                self.response_metadata, *(o.response_metadata for o in other)
            )
            return self.__class__(  # type: ignore[call-arg]
                id=self.id,
                content=content,
                additional_kwargs=additional_kwargs,
                response_metadata=response_metadata,
            )
        msg = (
            'unsupported operand type(s) for +: "'
            f"{self.__class__.__name__}"
            f'" and "{other.__class__.__name__}"'
        )
        raise TypeError(msg)


def message_to_dict(message: BaseMessage) -> dict:
    """Convert a Message to a dictionary.

    Args:
        message: Message to convert.

    Returns:
        Message as a dict. The dict will have a `type` key with the message type
        and a `data` key with the message data as a dict.

    

    中文翻译:
    将消息转换为字典。
    参数：
        message：要转换的消息。
    返回：
        消息作为字典。该字典将有一个包含消息类型的“type”键
        和一个“data”键，其中消息数据作为字典。"""
    return {"type": message.type, "data": message.model_dump()}


def messages_to_dict(messages: Sequence[BaseMessage]) -> list[dict]:
    """Convert a sequence of Messages to a list of dictionaries.

    Args:
        messages: Sequence of messages (as `BaseMessage`s) to convert.

    Returns:
        List of messages as dicts.

    

    中文翻译:
    将消息序列转换为字典列表。
    参数：
        messages：要转换的消息序列（如“BaseMessage”）。
    返回：
        作为字典的消息列表。"""
    return [message_to_dict(m) for m in messages]


def get_msg_title_repr(title: str, *, bold: bool = False) -> str:
    """Get a title representation for a message.

    Args:
        title: The title.
        bold: Whether to bold the title.

    Returns:
        The title representation.

    

    中文翻译:
    获取消息的标题表示。
    参数：
        标题：标题。
        粗体：标题是否加粗。
    返回：
        标题表示。"""
    padded = " " + title + " "
    sep_len = (80 - len(padded)) // 2
    sep = "=" * sep_len
    second_sep = sep + "=" if len(padded) % 2 else sep
    if bold:
        padded = get_bolded_text(padded)
    return f"{sep}{padded}{second_sep}"
