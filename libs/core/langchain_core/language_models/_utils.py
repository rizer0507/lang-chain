import re
from collections.abc import Sequence
from typing import (
    TYPE_CHECKING,
    Literal,
    TypedDict,
    TypeVar,
)

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage
from langchain_core.messages.content import (
    ContentBlock,
)


def is_openai_data_block(
    block: dict, filter_: Literal["image", "audio", "file"] | None = None
) -> bool:
    """Check whether a block contains multimodal data in OpenAI Chat Completions format.

    Supports both data and ID-style blocks (e.g. `'file_data'` and `'file_id'`)

    If additional keys are present, they are ignored / will not affect outcome as long
    as the required keys are present and valid.

    Args:
        block: The content block to check.
        filter_: If provided, only return True for blocks matching this specific type.
            - "image": Only match image_url blocks
            - "audio": Only match input_audio blocks
            - "file": Only match file blocks
            If `None`, match any valid OpenAI data block type. Note that this means that
            if the block has a valid OpenAI data type but the filter_ is set to a
            different type, this function will return False.

    Returns:
        `True` if the block is a valid OpenAI data block and matches the filter_
        (if provided).

    

    中文翻译:
    检查块是否包含 OpenAI Chat Completions 格式的多模态数据。
    支持数据和 ID 样式块（例如“file_data”和“file_id”）
    如果存在其他键，它们将被忽略/只要不影响结果
    因为所需的密钥存在且有效。
    参数：
        block：要检查的内容块。
        filter_：如果提供，则仅对匹配此特定类型的块返回 True。
            - “image”：仅匹配 image_url 块
            - “audio”：仅匹配 input_audio 块
            - “file”：仅匹配文件块
            如果为“无”，则匹配任何有效的 OpenAI 数据块类型。请注意，这意味着
            如果该块具有有效的 OpenAI 数据类型，但 filter_ 设置为
            不同类型，该函数将返回 False。
    返回：
        如果该块是有效的 OpenAI 数据块并且与 filter_ 匹配，则为“True”
        （如果提供）。"""
    if block.get("type") == "image_url":
        if filter_ is not None and filter_ != "image":
            return False
        if (
            (set(block.keys()) <= {"type", "image_url", "detail"})
            and (image_url := block.get("image_url"))
            and isinstance(image_url, dict)
        ):
            url = image_url.get("url")
            if isinstance(url, str):
                # Required per OpenAI spec
                # 中文: 根据 OpenAI 规范需要
                return True
            # Ignore `'detail'` since it's optional and specific to OpenAI
            # 中文: 忽略“细节”，因为它是可选的并且特定于 OpenAI

    elif block.get("type") == "input_audio":
        if filter_ is not None and filter_ != "audio":
            return False
        if (audio := block.get("input_audio")) and isinstance(audio, dict):
            audio_data = audio.get("data")
            audio_format = audio.get("format")
            # Both required per OpenAI spec
            # 中文: 根据 OpenAI 规范，两者都需要
            if isinstance(audio_data, str) and isinstance(audio_format, str):
                return True

    elif block.get("type") == "file":
        if filter_ is not None and filter_ != "file":
            return False
        if (file := block.get("file")) and isinstance(file, dict):
            file_data = file.get("file_data")
            file_id = file.get("file_id")
            # Files can be either base64-encoded or pre-uploaded with an ID
            # 中文: 文件可以进行 base64 编码，也可以使用 ID 进行预上传
            if isinstance(file_data, str) or isinstance(file_id, str):
                return True

    else:
        return False

    # Has no `'type'` key
    # 中文: 没有“type”键
    return False


class ParsedDataUri(TypedDict):
    source_type: Literal["base64"]
    data: str
    mime_type: str


def _parse_data_uri(uri: str) -> ParsedDataUri | None:
    """Parse a data URI into its components.

    If parsing fails, return `None`. If either MIME type or data is missing, return
    `None`.

    Example:
        ```python
        data_uri = "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
        parsed = _parse_data_uri(data_uri)

        assert parsed == {
            "source_type": "base64",
            "mime_type": "image/jpeg",
            "data": "/9j/4AAQSkZJRg...",
        }
        ```
    

    中文翻译:
    将数据 URI 解析为其组件。
    如果解析失败，则返回“None”。如果 MIME 类型或数据丢失，则返回
    “没有”。
    示例：
        ````蟒蛇
        data_uri =“数据：图像/jpeg;base64，/9j/4AAQSkZJRg...”
        解析 = _parse_data_uri(data_uri)
        断言已解析 == {
            “source_type”：“base64”，
            "mime_type": "图像/jpeg",
            "数据": "/9j/4AAQSkZJRg...",
        }
        ````"""
    regex = r"^data:(?P<mime_type>[^;]+);base64,(?P<data>.+)$"
    match = re.match(regex, uri)
    if match is None:
        return None

    mime_type = match.group("mime_type")
    data = match.group("data")
    if not mime_type or not data:
        return None

    return {
        "source_type": "base64",
        "data": data,
        "mime_type": mime_type,
    }


def _normalize_messages(
    messages: Sequence["BaseMessage"],
) -> list["BaseMessage"]:
    """Normalize message formats to LangChain v1 standard content blocks.

    Chat models already implement support for:
    - Images in OpenAI Chat Completions format
        These will be passed through unchanged
    - LangChain v1 standard content blocks

    This function extends support to:
    - `[Audio](https://platform.openai.com/docs/api-reference/chat/create) and
        `[file](https://platform.openai.com/docs/api-reference/files) data in OpenAI
        Chat Completions format
        - Images are technically supported but we expect chat models to handle them
            directly; this may change in the future
    - LangChain v0 standard content blocks for backward compatibility

    !!! warning "Behavior changed in `langchain-core` 1.0.0"

        In previous versions, this function returned messages in LangChain v0 format.
        Now, it returns messages in LangChain v1 format, which upgraded chat models now
        expect to receive when passing back in message history. For backward
        compatibility, this function will convert v0 message content to v1 format.

    ??? note "v0 Content Block Schemas"

        `URLContentBlock`:

        ```python
        {
            mime_type: NotRequired[str]
            type: Literal['image', 'audio', 'file'],
            source_type: Literal['url'],
            url: str,
        }
        ```

        `Base64ContentBlock`:

        ```python
        {
            mime_type: NotRequired[str]
            type: Literal['image', 'audio', 'file'],
            source_type: Literal['base64'],
            data: str,
        }
        ```

        `IDContentBlock`:

        (In practice, this was never used)

        ```python
        {
            type: Literal["image", "audio", "file"],
            source_type: Literal["id"],
            id: str,
        }
        ```

        `PlainTextContentBlock`:

        ```python
        {
            mime_type: NotRequired[str]
            type: Literal['file'],
            source_type: Literal['text'],
            url: str,
        }
        ```

    If a v1 message is passed in, it will be returned as-is, meaning it is safe to
    always pass in v1 messages to this function for assurance.

    For posterity, here are the OpenAI Chat Completions schemas we expect:

    Chat Completions image. Can be URL-based or base64-encoded. Supports MIME types
    png, jpeg/jpg, webp, static gif:
    {
        "type": Literal['image_url'],
        "image_url": {
            "url": Union["data:$MIME_TYPE;base64,$BASE64_ENCODED_IMAGE", "$IMAGE_URL"],
            "detail": Literal['low', 'high', 'auto'] = 'auto',  # Supported by OpenAI
        }
    }

    Chat Completions audio:
    {
        "type": Literal['input_audio'],
        "input_audio": {
            "format": Literal['wav', 'mp3'],
            "data": str = "$BASE64_ENCODED_AUDIO",
        },
    }

    Chat Completions files: either base64 or pre-uploaded file ID
    {
        "type": Literal['file'],
        "file": Union[
            {
                "filename": str | None = "$FILENAME",
                "file_data": str = "$BASE64_ENCODED_FILE",
            },
            {
                "file_id": str = "$FILE_ID",  # For pre-uploaded files to OpenAI
            },
        ],
    }

    

    中文翻译:
    将消息格式标准化为LangChain v1标准内容块。
    聊天模型已经实现了对以下方面的支持：
    - OpenAI 聊天完成格式的图像
        这些将不变地通过
    - LangChain v1标准内容块
    此功能将支持扩展到：
    - `[音频](https://platform.openai.com/docs/api-reference/chat/create) 和
        `[文件](https://platform.openai.com/docs/api-reference/files) OpenAI 中的数据
        聊天完成格式
        - 图像在技术上得到支持，但我们希望聊天模型能够处理它们
            直接；这将来可能会改变
    - LangChain v0标准内容块向后兼容
    !!!警告“`langchain-core` 1.0.0 中的行为已更改”
        在之前的版本中，该函数返回LangChain v0格式的消息。
        现在，它返回 LangChain v1 格式的消息，该格式现在升级了聊天模型
        期望在消息历史记录中传回时收到。对于落后的
        兼容性，该函数会将v0消息内容转换为v1格式。
    ???注意“v0 内容块架构”
        `URL内容块`:
        ````蟒蛇
        {
            mime_type: 不需要[str]
            类型：文字['图像'，'音频'，'文件']，
            source_type: 文字['url'],
            网址：str，
        }
        ````
        `Base64ContentBlock`：
        ````蟒蛇
        {
            mime_type: 不需要[str]
            类型：文字['图像'，'音频'，'文件']，
            source_type: 文字['base64'],
            数据：str，
        }
        ````
        `IDContentBlock`:
        （实际中从未使用过）
        ````蟒蛇
        {
            类型：文字[“图像”，“音频”，“文件”]，
            source_type：文字[“id”]，
            id：str，
        }
        ````
        `纯文本内容块`:
        ````蟒蛇
        {
            mime_type: 不需要[str]
            类型：文字['文件']，
            source_type: 文字['文本'],
            网址：str，
        }
        ````
    如果传入 v1 消息，它将按原样返回，这意味着它是安全的
    始终将 v1 消息传递给此函数以确保安全。
    对于后代，以下是我们期望的 OpenAI 聊天完成模式：
    聊天完成图像。可以基于 URL 或 Base64 编码。支持 MIME 类型
    png、jpeg/jpg、webp、静态 gif：
    {
        “类型”：文字['image_url']，
        “图像网址”：{
            "url": Union["数据:$MIME_TYPE;base64,$BASE64_ENCODED_IMAGE", "$IMAGE_URL"],
            "detail": Literal['low', 'high', 'auto'] = 'auto', # OpenAI 支持
        }
    }
    聊天完成音频：
    {
        “类型”：文字['input_audio']，
        “输入音频”：{
            “格式”：文字['wav'，'mp3']，
            “数据”：str =“$BASE64_ENCODED_AUDIO”，
        },
    }
    聊天完成文件：base64 或预上传的文件 ID
    {
        “类型”：文字['文件']，
        “文件”：联盟[
            {
                “文件名”：str |无 = "$FILENAME",
                “文件数据”：str =“$BASE64_ENCODED_FILE”，
            },
            {
                "file_id": str = "$FILE_ID", # 用于预先上传到 OpenAI 的文件
            },
        ],
    }"""
    from langchain_core.messages.block_translators.langchain_v0 import (  # noqa: PLC0415
        _convert_legacy_v0_content_block_to_v1,
    )
    from langchain_core.messages.block_translators.openai import (  # noqa: PLC0415
        _convert_openai_format_to_data_block,
    )

    formatted_messages = []
    for message in messages:
        # We preserve input messages - the caller may reuse them elsewhere and expects
        # 中文: 我们保留输入消息 - 调用者可以在其他地方重用它们并期望
        # them to remain unchanged. We only create a copy if we need to translate.
        # 中文: 他们保持不变。我们仅在需要翻译时创建副本。
        formatted_message = message

        if isinstance(message.content, list):
            for idx, block in enumerate(message.content):
                # OpenAI Chat Completions multimodal data blocks to v1 standard
                # 中文: OpenAI Chat 根据 v1 标准完成多模式数据块
                if (
                    isinstance(block, dict)
                    and block.get("type") in {"input_audio", "file"}
                    # Discriminate between OpenAI/LC format since they share `'type'`
                    # 中文: 区分 OpenAI/LC 格式，因为它们共享“类型”
                    and is_openai_data_block(block)
                ):
                    formatted_message = _ensure_message_copy(message, formatted_message)

                    converted_block = _convert_openai_format_to_data_block(block)
                    _update_content_block(formatted_message, idx, converted_block)

                # Convert multimodal LangChain v0 to v1 standard content blocks
                # 中文: 将多模式 LangChain v0 转换为 v1 标准内容块
                elif (
                    isinstance(block, dict)
                    and block.get("type")
                    in {
                        "image",
                        "audio",
                        "file",
                    }
                    and block.get("source_type")  # v1 doesn't have `source_type`
                    in {
                        "url",
                        "base64",
                        "id",
                        "text",
                    }
                ):
                    formatted_message = _ensure_message_copy(message, formatted_message)

                    converted_block = _convert_legacy_v0_content_block_to_v1(block)
                    _update_content_block(formatted_message, idx, converted_block)
                    continue

                # else, pass through blocks that look like they have v1 format unchanged
                # 中文: 否则，传递看起来 v1 格式未更改的块

        formatted_messages.append(formatted_message)

    return formatted_messages


T = TypeVar("T", bound="BaseMessage")


def _ensure_message_copy(message: T, formatted_message: T) -> T:
    """Create a copy of the message if it hasn't been copied yet.

    中文翻译:
    如果尚未复制消息，请创建该消息的副本。"""
    if formatted_message is message:
        formatted_message = message.model_copy()
        # Shallow-copy content list to allow modifications
        # 中文: 浅复制内容列表以允许修改
        formatted_message.content = list(formatted_message.content)
    return formatted_message


def _update_content_block(
    formatted_message: "BaseMessage", idx: int, new_block: ContentBlock | dict
) -> None:
    """Update a content block at the given index, handling type issues.

    中文翻译:
    更新给定索引处的内容块，处理类型问题。"""
    # Type ignore needed because:
    # 中文: 需要类型忽略，因为：
    # - `BaseMessage.content` is typed as `Union[str, list[Union[str, dict]]]`
    # 中文: - `BaseMessage.content` 的类型为 `Union[str, list[Union[str, dict]]]`
    # - When content is str, indexing fails (index error)
    # 中文: - 当内容为str时，索引失败（索引错误）
    # - When content is list, the items are `Union[str, dict]` but we're assigning
    # 中文: - 当内容为列表时，项目为“Union[str, dict]”，但我们正在分配
    #   `Union[ContentBlock, dict]` where ContentBlock is richer than dict
    #   中文: `Union[ContentBlock, dict]` 其中 ContentBlock 比 dict 更丰富
    # - This is safe because we only call this when we've verified content is a list and
    # 中文: - 这是安全的，因为我们仅在验证内容是列表并且
    #   we're doing content block conversions
    #   中文: 我们正在进行内容块转换
    formatted_message.content[idx] = new_block  # type: ignore[index, assignment]


def _update_message_content_to_blocks(message: T, output_version: str) -> T:
    return message.model_copy(
        update={
            "content": message.content_blocks,
            "response_metadata": {
                **message.response_metadata,
                "output_version": output_version,
            },
        }
    )
