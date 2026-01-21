"""Standard, multimodal content blocks for Large Language Model I/O.

This module provides standardized data structures for representing inputs to and outputs
from LLMs. The core abstraction is the **Content Block**, a `TypedDict`.

**Rationale**

Different LLM providers use distinct and incompatible API schemas. This module provides
a unified, provider-agnostic format to facilitate these interactions. A message to or
from a model is simply a list of content blocks, allowing for the natural interleaving
of text, images, and other content in a single ordered sequence.

An adapter for a specific provider is responsible for translating this standard list of
blocks into the format required by its API.

**Extensibility**

Data **not yet mapped** to a standard block may be represented using the
`NonStandardContentBlock`, which allows for provider-specific data to be included
without losing the benefits of type checking and validation.

Furthermore, provider-specific fields **within** a standard block are fully supported
by default in the `extras` field of each block. This allows for additional metadata
to be included without breaking the standard structure. For example, Google's thought
signature:

```python
AIMessage(
    content=[
        {
            "type": "text",
            "text": "J'adore la programmation.",
            "extras": {"signature": "EpoWCpc..."},  # Thought signature
        }
    ], ...
)
```


!!! note

    Following widespread adoption of [PEP 728](https://peps.python.org/pep-0728/), we
    intend to add `extra_items=Any` as a param to Content Blocks. This will signify to
    type checkers that additional provider-specific fields are allowed outside of the
    `extras` field, and that will become the new standard approach to adding
    provider-specific metadata.

    ??? note

        **Example with PEP 728 provider-specific fields:**

        ```python
        # Content block definition
        # 中文: 内容块定义
        # NOTE: `extra_items=Any`
        class TextContentBlock(TypedDict, extra_items=Any):
            type: Literal["text"]
            id: NotRequired[str]
            text: str
            annotations: NotRequired[list[Annotation]]
            index: NotRequired[int]
        ```

        ```python
        from langchain_core.messages.content import TextContentBlock

        # Create a text content block with provider-specific fields
        # 中文: 使用特定于提供商的字段创建文本内容块
        my_block: TextContentBlock = {
            # Add required fields
            # 中文: 添加必填字段
            "type": "text",
            "text": "Hello, world!",
            # Additional fields not specified in the TypedDict
            # 中文: TypedDict 中未指定的其他字段
            # These are valid with PEP 728 and are typed as Any
            # 中文: 这些对于 PEP 728 有效，并且类型为 Any
            "openai_metadata": {"model": "gpt-4", "temperature": 0.7},
            "anthropic_usage": {"input_tokens": 10, "output_tokens": 20},
            "custom_field": "any value",
        }

        # Mutating an existing block to add provider-specific fields
        # 中文: 改变现有块以添加特定于提供者的字段
        openai_data = my_block["openai_metadata"]  # Type: Any
        ```

**Example Usage**

```python
# Direct construction
# 中文: 直接施工
from langchain_core.messages.content import TextContentBlock, ImageContentBlock

multimodal_message: AIMessage(
    content_blocks=[
        TextContentBlock(type="text", text="What is shown in this image?"),
        ImageContentBlock(
            type="image",
            url="https://www.langchain.com/images/brand/langchain_logo_text_w_white.png",
            mime_type="image/png",
        ),
    ]
)

# Using factories
# 中文: 使用工厂
from langchain_core.messages.content import create_text_block, create_image_block

multimodal_message: AIMessage(
    content=[
        create_text_block("What is shown in this image?"),
        create_image_block(
            url="https://www.langchain.com/images/brand/langchain_logo_text_w_white.png",
            mime_type="image/png",
        ),
    ]
)
```

Factory functions offer benefits such as:

- Automatic ID generation (when not provided)
- No need to manually specify the `type` field

中文翻译:
用于大型语言模型 I/O 的标准多模式内容块。
该模块提供标准化数据结构来表示输入和输出
来自法学硕士。核心抽象是**内容块**，即“TypedDict”。
**理由**
不同的 LLM 提供商使用不同且不兼容的 API 模式。该模块提供
一种与提供商无关的统一格式来促进这些交互。留言给 或
模型中的内容只是一个内容块列表，允许自然交错
单个有序序列中的文本、图像和其他内容。
特定提供商的适配器负责翻译此标准列表
块转换成其 API 所需的格式。
**可扩展性**
**尚未映射**到标准块的数据可以使用
“NonStandardContentBlock”，允许包含特定于提供者的数据
而不失去类型检查和验证的好处。
此外，完全支持标准块**内**的特定于提供者的字段
默认情况下在每个块的“extras”字段中。这允许额外的元数据
在不破坏标准结构的情况下包含在内。例如，谷歌的想法
签名：
````蟒蛇
人工智能留言(
    内容=[
        {
            “类型”：“文本”，
            "text": "我喜欢编程。",
            "extras": {"signature": "EpoWCpc..."}, # 思想签名
        }
    ]、...
）
````
!!!注释
    随着 [PEP 728](https://peps.python.org/pep-0728/) 的广泛采用，我们
    打算将 `extra_items=Any` 作为参数添加到内容块中。这将意味着
    类型检查器，允许在
    “extras”字段，这将成为添加的新标准方法
    特定于提供商的元数据。
    ???注释
        **PEP 728 提供商特定字段的示例：**
        ````蟒蛇
        # 内容块定义
        # 注意：`extra_items=任何`
        类 TextContentBlock(TypedDict, extra_items=Any):
            类型：文字[“文本”]
            id: 不需要[str]
            文本：str
            注释：NotRequired[列表[注释]]
            索引： 不需要[int]
        ````
        ````蟒蛇
        从 langchain_core.messages.content 导入 TextContentBlock
        # 使用特定于提供者的字段创建文本内容块
        my_block: TextContentBlock = {
            # 添加必填字段
            “类型”：“文本”，
            "text": "你好，世界！",
            # TypedDict 中未指定的其他字段
            # 这些对于 PEP 728 有效，并且键入为 Any
            “openai_metadata”：{“模型”：“gpt-4”，“温度”：0.7}，
            “anthropic_usage”：{“input_tokens”：10，“output_tokens”：20}，
            "custom_field": "任意值",
        }
        # 改变现有块以添加特定于提供者的字段
        openai_data = my_block["openai_metadata"] # 类型：任意
        ````
**用法示例**
````蟒蛇
# 直接构建
从 langchain_core.messages.content 导入 TextContentBlock、ImageContentBlock
多模式消息：AIMessage(
    内容块=[
        TextContentBlock(type="text", text="这张图片显示了什么？"),
        图像内容块(
            类型=“图像”，
            url =“https://www.langchain.com/images/brand/langchain_logo_text_w_white.png”，
            mime_type =“图像/ png”，
        ),
    ]
）
# 使用工厂
从 langchain_core.messages.content 导入 create_text_block、create_image_block
多模式消息：AIMessage(
    内容=[
        create_text_block("这张图片显示了什么？"),
        创建图像块（
            url =“https://www.langchain.com/images/brand/langchain_logo_text_w_white.png”，
            mime_type =“图像/ png”，
        ),
    ]
）
````
工厂功能具有以下优势：
- 自动生成 ID（如果未提供）
- 无需手动指定`type`字段
"""

from typing import Any, Literal, get_args, get_type_hints

from typing_extensions import NotRequired, TypedDict

from langchain_core.utils.utils import ensure_id


class Citation(TypedDict):
    """Annotation for citing data from a document.

    !!! note

        `start`/`end` indices refer to the **response text**,
        not the source text. This means that the indices are relative to the model's
        response, not the original document (as specified in the `url`).

    !!! note "Factory function"

        `create_citation` may also be used as a factory to create a `Citation`.
        Benefits include:

        * Automatic ID generation (when not provided)
        * Required arguments strictly validated at creation time
    

    中文翻译:
    用于引用文档中的数据的注释。
    ！！！注释
        `start`/`end` 索引指的是 **响应文本**，
        不是源文本。这意味着指数与模型的相关
        响应，而不是原始文档（如“url”中指定）。
    ！！！注意“工厂功能”
        `create_itation` 也可以用作创建 `Citation` 的工厂。
        好处包括：
        * 自动生成 ID（未提供时）
        * 必需的参数在创建时经过严格验证"""

    type: Literal["citation"]
    """Type of the content block. Used for discrimination.

    中文翻译:
    内容块的类型。用于歧视。"""

    id: NotRequired[str]
    """Unique identifier for this content block.

    Either:

    - Generated by the provider
    - Generated by LangChain upon creation (`UUID4` prefixed with `'lc_'`))
    

    中文翻译:
    该内容块的唯一标识符。
    要么：
    - 由提供商生成
    - 由 LangChain 在创建时生成（`UUID4` 前缀为 `'lc_'`））"""

    url: NotRequired[str]
    """URL of the document source.

    中文翻译:
    文档来源的 URL。"""

    title: NotRequired[str]
    """Source document title.

    For example, the page title for a web page or the title of a paper.
    

    中文翻译:
    源文档标题。
    例如，网页的页面标题或论文的标题。"""

    start_index: NotRequired[int]
    """Start index of the **response text** (`TextContentBlock.text`).

    中文翻译:
    **响应文本** (`TextContentBlock.text`) 的起始索引。"""

    end_index: NotRequired[int]
    """End index of the **response text** (`TextContentBlock.text`)

    中文翻译:
    **响应文本**的结束索引（`TextContentBlock.text`）"""

    cited_text: NotRequired[str]
    """Excerpt of source text being cited.

    中文翻译:
    所引用的源文本摘录。"""

    # NOTE: not including spans for the raw document text (such as `text_start_index`
    # and `text_end_index`) as this is not currently supported by any provider. The
    # 中文: 和 `text_end_index`），因为目前任何提供商都不支持此功能。这
    # thinking is that the `cited_text` should be sufficient for most use cases, and it
    # 中文: 我们的想法是“cited_text”对于大多数用例来说应该足够了，而且它
    # is difficult to reliably extract spans from the raw document text across file
    # 中文: 很难从跨文件的原始文档文本中可靠地提取跨度
    # formats or encoding schemes.
    # 中文: 格式或编码方案。

    extras: NotRequired[dict[str, Any]]
    """Provider-specific metadata.

    中文翻译:
    提供商特定的元数据。"""


class NonStandardAnnotation(TypedDict):
    """Provider-specific annotation format.

    中文翻译:
    提供者特定的注释格式。"""

    type: Literal["non_standard_annotation"]
    """Type of the content block. Used for discrimination.

    中文翻译:
    内容块的类型。用于歧视。"""

    id: NotRequired[str]
    """Unique identifier for this content block.

    Either:

    - Generated by the provider
    - Generated by LangChain upon creation (`UUID4` prefixed with `'lc_'`))
    

    中文翻译:
    该内容块的唯一标识符。
    要么：
    - 由提供商生成
    - 由 LangChain 在创建时生成（`UUID4` 前缀为 `'lc_'`））"""

    value: dict[str, Any]
    """Provider-specific annotation data.

    中文翻译:
    特定于提供者的注释数据。"""


Annotation = Citation | NonStandardAnnotation
"""A union of all defined `Annotation` types.

中文翻译:
所有定义的“注释”类型的联合。"""


class TextContentBlock(TypedDict):
    """Text output from a LLM.

    This typically represents the main text content of a message, such as the response
    from a language model or the text of a user message.

    !!! note "Factory function"

        `create_text_block` may also be used as a factory to create a
        `TextContentBlock`. Benefits include:

        * Automatic ID generation (when not provided)
        * Required arguments strictly validated at creation time
    

    中文翻译:
    LLM 的文本输出。
    这通常表示消息的主要文本内容，例如响应
    来自语言模型或用户消息的文本。
    !!!注意“工厂功能”
        `create_text_block` 也可以用作工厂来创建
        `文本内容块`。好处包括：
        * 自动生成 ID（未提供时）
        * 必需的参数在创建时经过严格验证"""

    type: Literal["text"]
    """Type of the content block. Used for discrimination.

    中文翻译:
    内容块的类型。用于歧视。"""

    id: NotRequired[str]
    """Unique identifier for this content block.

    Either:

    - Generated by the provider
    - Generated by LangChain upon creation (`UUID4` prefixed with `'lc_'`))
    

    中文翻译:
    该内容块的唯一标识符。
    要么：
    - 由提供商生成
    - 由 LangChain 在创建时生成（`UUID4` 前缀为 `'lc_'`））"""

    text: str
    """Block text.

    中文翻译:
    块文本。"""

    annotations: NotRequired[list[Annotation]]
    """`Citation`s and other annotations.

    中文翻译:
    `引文`和其他注释。"""

    index: NotRequired[int | str]
    """Index of block in aggregate response. Used during streaming.

    中文翻译:
    聚合响应中块的索引。在流式传输期间使用。"""

    extras: NotRequired[dict[str, Any]]
    """Provider-specific metadata.

    中文翻译:
    提供商特定的元数据。"""


class ToolCall(TypedDict):
    """Represents an AI's request to call a tool.

    Example:
        ```python
        {"name": "foo", "args": {"a": 1}, "id": "123"}
        ```

        This represents a request to call the tool named "foo" with arguments {"a": 1}
        and an identifier of "123".

    !!! note "Factory function"

        `create_tool_call` may also be used as a factory to create a
        `ToolCall`. Benefits include:

        * Automatic ID generation (when not provided)
        * Required arguments strictly validated at creation time
    

    中文翻译:
    代表AI调用工具的请求。
    示例：
        ````蟒蛇
        {“name”：“foo”，“args”：{“a”：1}，“id”：“123”}
        ````
        这表示使用参数 {"a": 1} 调用名为“foo”的工具的请求
        和标识符“123”。
    ！！！注意“工厂功能”
        `create_tool_call` 也可以用作工厂来创建
        `工具调用`。好处包括：
        * 自动生成 ID（未提供时）
        * 必需的参数在创建时经过严格验证"""

    type: Literal["tool_call"]
    """Used for discrimination.

    中文翻译:
    用于歧视。"""

    id: str | None
    """An identifier associated with the tool call.

    An identifier is needed to associate a tool call request with a tool
    call result in events when multiple concurrent tool calls are made.
    

    中文翻译:
    与工具调用关联的标识符。
    需要一个标识符来将工具调用请求与工具关联起来
    当进行多个并发工具调用时，调用会导致事件。"""
    # TODO: Consider making this NotRequired[str] in the future.

    name: str
    """The name of the tool to be called.

    中文翻译:
    要调用的工具的名称。"""

    args: dict[str, Any]
    """The arguments to the tool call.

    中文翻译:
    工具调用的参数。"""

    index: NotRequired[int | str]
    """Index of block in aggregate response. Used during streaming.

    中文翻译:
    聚合响应中块的索引。在流式传输期间使用。"""

    extras: NotRequired[dict[str, Any]]
    """Provider-specific metadata.

    中文翻译:
    提供商特定的元数据。"""


class ToolCallChunk(TypedDict):
    """A chunk of a tool call (yielded when streaming).

    When merging `ToolCallChunks` (e.g., via `AIMessageChunk.__add__`),
    all string attributes are concatenated. Chunks are only merged if their
    values of `index` are equal and not `None`.

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
    合并`ToolCallChunks`时（例如，通过`AIMessageChunk.__add__`），
    所有字符串属性都连接在一起。仅当它们的块被合并时
    `index` 的值相等而不是 `None`。
    示例：
    ````蟒蛇
    left_chunks = [ToolCallChunk(name="foo", args='{"a":', index=0)]
    right_chunks = [ToolCallChunk(name=None, args="1}", index=0)]
    （
        AIMessageChunk(content="", tool_call_chunks=left_chunks)
        + AIMessageChunk(content="", tool_call_chunks=right_chunks)
    ).tool_call_chunks == [ToolCallChunk(name="foo", args='{"a":1}', index=0)]
    ````"""

    # TODO: Consider making fields NotRequired[str] in the future.

    type: Literal["tool_call_chunk"]
    """Used for serialization.

    中文翻译:
    用于序列化。"""

    id: str | None
    """An identifier associated with the tool call.

    An identifier is needed to associate a tool call request with a tool
    call result in events when multiple concurrent tool calls are made.
    

    中文翻译:
    与工具调用关联的标识符。
    需要一个标识符来将工具调用请求与工具关联起来
    当进行多个并发工具调用时，调用会导致事件。"""
    # TODO: Consider making this NotRequired[str] in the future.

    name: str | None
    """The name of the tool to be called.

    中文翻译:
    要调用的工具的名称。"""

    args: str | None
    """The arguments to the tool call.

    中文翻译:
    工具调用的参数。"""

    index: NotRequired[int | str]
    """The index of the tool call in a sequence.

    中文翻译:
    序列中工具调用的索引。"""

    extras: NotRequired[dict[str, Any]]
    """Provider-specific metadata.

    中文翻译:
    提供商特定的元数据。"""


class InvalidToolCall(TypedDict):
    """Allowance for errors made by LLM.

    Here we add an `error` key to surface errors made during generation
    (e.g., invalid JSON arguments.)
    

    中文翻译:
    LLM 允许犯错。
    这里我们添加一个“error”键来显示生成过程中发生的错误
    （例如，无效的 JSON 参数。）"""

    # TODO: Consider making fields NotRequired[str] in the future.

    type: Literal["invalid_tool_call"]
    """Used for discrimination.

    中文翻译:
    用于歧视。"""

    id: str | None
    """An identifier associated with the tool call.

    An identifier is needed to associate a tool call request with a tool
    call result in events when multiple concurrent tool calls are made.
    

    中文翻译:
    与工具调用关联的标识符。
    需要一个标识符来将工具调用请求与工具关联起来
    当进行多个并发工具调用时，调用会导致事件。"""
    # TODO: Consider making this NotRequired[str] in the future.

    name: str | None
    """The name of the tool to be called.

    中文翻译:
    要调用的工具的名称。"""

    args: str | None
    """The arguments to the tool call.

    中文翻译:
    工具调用的参数。"""

    error: str | None
    """An error message associated with the tool call.

    中文翻译:
    与工具调用相关的错误消息。"""

    index: NotRequired[int | str]
    """Index of block in aggregate response. Used during streaming.

    中文翻译:
    聚合响应中块的索引。在流式传输期间使用。"""

    extras: NotRequired[dict[str, Any]]
    """Provider-specific metadata.

    中文翻译:
    提供商特定的元数据。"""


class ServerToolCall(TypedDict):
    """Tool call that is executed server-side.

    For example: code execution, web search, etc.
    

    中文翻译:
    在服务器端执行的工具调用。
    例如：代码执行、网页搜索等。"""

    type: Literal["server_tool_call"]
    """Used for discrimination.

    中文翻译:
    用于歧视。"""

    id: str
    """An identifier associated with the tool call.

    中文翻译:
    与工具调用关联的标识符。"""

    name: str
    """The name of the tool to be called.

    中文翻译:
    要调用的工具的名称。"""

    args: dict[str, Any]
    """The arguments to the tool call.

    中文翻译:
    工具调用的参数。"""

    index: NotRequired[int | str]
    """Index of block in aggregate response. Used during streaming.

    中文翻译:
    聚合响应中块的索引。在流式传输期间使用。"""

    extras: NotRequired[dict[str, Any]]
    """Provider-specific metadata.

    中文翻译:
    提供商特定的元数据。"""


class ServerToolCallChunk(TypedDict):
    """A chunk of a server-side tool call (yielded when streaming).

    中文翻译:
    服务器端工具调用的一大块（流式传输时产生）。"""

    type: Literal["server_tool_call_chunk"]
    """Used for discrimination.

    中文翻译:
    用于歧视。"""

    name: NotRequired[str]
    """The name of the tool to be called.

    中文翻译:
    要调用的工具的名称。"""

    args: NotRequired[str]
    """JSON substring of the arguments to the tool call.

    中文翻译:
    工具调用参数的 JSON 子字符串。"""

    id: NotRequired[str]
    """Unique identifier for this server tool call chunk.

    Either:

    - Generated by the provider
    - Generated by LangChain upon creation (`UUID4` prefixed with `'lc_'`))
    

    中文翻译:
    该服务器工具调用块的唯一标识符。
    要么：
    - 由提供商生成
    - 由 LangChain 在创建时生成（`UUID4` 前缀为 `'lc_'`））"""

    index: NotRequired[int | str]
    """Index of block in aggregate response. Used during streaming.

    中文翻译:
    聚合响应中块的索引。在流式传输期间使用。"""

    extras: NotRequired[dict[str, Any]]
    """Provider-specific metadata.

    中文翻译:
    提供商特定的元数据。"""


class ServerToolResult(TypedDict):
    """Result of a server-side tool call.

    中文翻译:
    服务器端工具调用的结果。"""

    type: Literal["server_tool_result"]
    """Used for discrimination.

    中文翻译:
    用于歧视。"""

    id: NotRequired[str]
    """Unique identifier for this server tool result.

    Either:

    - Generated by the provider
    - Generated by LangChain upon creation (`UUID4` prefixed with `'lc_'`))
    

    中文翻译:
    该服务器工具结果的唯一标识符。
    要么：
    - 由提供商生成
    - 由 LangChain 在创建时生成（`UUID4` 前缀为 `'lc_'`））"""

    tool_call_id: str
    """ID of the corresponding server tool call.

    中文翻译:
    对应的服务器工具调用的ID。"""

    status: Literal["success", "error"]
    """Execution status of the server-side tool.

    中文翻译:
    服务器端工具的执行状态。"""

    output: NotRequired[Any]
    """Output of the executed tool.

    中文翻译:
    已执行工具的输出。"""

    index: NotRequired[int | str]
    """Index of block in aggregate response. Used during streaming.

    中文翻译:
    聚合响应中块的索引。在流式传输期间使用。"""

    extras: NotRequired[dict[str, Any]]
    """Provider-specific metadata.

    中文翻译:
    提供商特定的元数据。"""


class ReasoningContentBlock(TypedDict):
    """Reasoning output from a LLM.

    !!! note "Factory function"

        `create_reasoning_block` may also be used as a factory to create a
        `ReasoningContentBlock`. Benefits include:

        * Automatic ID generation (when not provided)
        * Required arguments strictly validated at creation time
    

    中文翻译:
    LLM 的推理输出。
    ！！！注意“工厂功能”
        `create_reasoning_block` 也可以用作工厂来创建
        `推理内容块`。好处包括：
        * 自动生成 ID（未提供时）
        * 必需的参数在创建时经过严格验证"""

    type: Literal["reasoning"]
    """Type of the content block. Used for discrimination.

    中文翻译:
    内容块的类型。用于歧视。"""

    id: NotRequired[str]
    """Unique identifier for this content block.

    Either:

    - Generated by the provider
    - Generated by LangChain upon creation (`UUID4` prefixed with `'lc_'`))
    

    中文翻译:
    该内容块的唯一标识符。
    要么：
    - 由提供商生成
    - 由 LangChain 在创建时生成（`UUID4` 前缀为 `'lc_'`））"""

    reasoning: NotRequired[str]
    """Reasoning text.

    Either the thought summary or the raw reasoning text itself. This is often parsed
    from `<think>` tags in the model's response.
    

    中文翻译:
    推理文本。
    思想摘要或原始推理文本本身。这个经常被解析
    来自模型响应中的“<think>”标签。"""

    index: NotRequired[int | str]
    """Index of block in aggregate response. Used during streaming.

    中文翻译:
    聚合响应中块的索引。在流式传输期间使用。"""

    extras: NotRequired[dict[str, Any]]
    """Provider-specific metadata.

    中文翻译:
    提供商特定的元数据。"""


# Note: `title` and `context` are fields that could be used to provide additional
# 中文: 注意：“title”和“context”是可用于提供附加信息的字段
# information about the file, such as a description or summary of its content.
# 中文: 有关文件的信息，例如其内容的描述或摘要。
# E.g. with Claude, you can provide a context for a file which is passed to the model.
# 中文: 例如。使用 Claude，您可以为传递给模型的文件提供上下文。
class ImageContentBlock(TypedDict):
    """Image data.

    !!! note "Factory function"

        `create_image_block` may also be used as a factory to create an
        `ImageContentBlock`. Benefits include:

        * Automatic ID generation (when not provided)
        * Required arguments strictly validated at creation time
    

    中文翻译:
    图像数据。
    !!!注意“工厂功能”
        `create_image_block` 也可以用作工厂来创建
        `图像内容块`。好处包括：
        * 自动生成 ID（未提供时）
        * 必需的参数在创建时经过严格验证"""

    type: Literal["image"]
    """Type of the content block. Used for discrimination.

    中文翻译:
    内容块的类型。用于歧视。"""

    id: NotRequired[str]
    """Unique identifier for this content block.

    Either:

    - Generated by the provider
    - Generated by LangChain upon creation (`UUID4` prefixed with `'lc_'`))
    

    中文翻译:
    该内容块的唯一标识符。
    要么：
    - 由提供商生成
    - 由 LangChain 在创建时生成（`UUID4` 前缀为 `'lc_'`））"""

    file_id: NotRequired[str]
    """Reference to the image in an external file storage system.

    For example, OpenAI or Anthropic's Files API.
    

    中文翻译:
    引用外部文件存储系统中的图像。
    例如，OpenAI 或 Anthropic 的文件 API。"""

    mime_type: NotRequired[str]
    """MIME type of the image.

    Required for base64 data.

    [Examples from IANA](https://www.iana.org/assignments/media-types/media-types.xhtml#image)
    

    中文翻译:
    图像的 MIME 类型。
    对于 Base64 数据是必需的。
    [来自 IANA 的示例](https://www.iana.org/assignments/media-types/media-types.xhtml#image)"""

    index: NotRequired[int | str]
    """Index of block in aggregate response. Used during streaming.

    中文翻译:
    聚合响应中块的索引。在流式传输期间使用。"""

    url: NotRequired[str]
    """URL of the image.

    中文翻译:
    图像的 URL。"""

    base64: NotRequired[str]
    """Data as a base64 string.

    中文翻译:
    Base64 字符串形式的数据。"""

    extras: NotRequired[dict[str, Any]]
    """Provider-specific metadata. This shouldn't be used for the image data itself.

    中文翻译:
    提供商特定的元数据。这不应该用于图像数据本身。"""


class VideoContentBlock(TypedDict):
    """Video data.

    !!! note "Factory function"

        `create_video_block` may also be used as a factory to create a
        `VideoContentBlock`. Benefits include:

        * Automatic ID generation (when not provided)
        * Required arguments strictly validated at creation time
    

    中文翻译:
    视频数据。
    ！！！注意“工厂功能”
        `create_video_block` 也可以用作工厂来创建
        `视频内容块`。好处包括：
        * 自动生成 ID（未提供时）
        * 必需的参数在创建时经过严格验证"""

    type: Literal["video"]
    """Type of the content block. Used for discrimination.

    中文翻译:
    内容块的类型。用于歧视。"""

    id: NotRequired[str]
    """Unique identifier for this content block.

    Either:

    - Generated by the provider
    - Generated by LangChain upon creation (`UUID4` prefixed with `'lc_'`))
    

    中文翻译:
    该内容块的唯一标识符。
    要么：
    - 由提供商生成
    - 由 LangChain 在创建时生成（`UUID4` 前缀为 `'lc_'`））"""

    file_id: NotRequired[str]
    """Reference to the video in an external file storage system.

    For example, OpenAI or Anthropic's Files API.
    

    中文翻译:
    参考外部文件存储系统中的视频。
    例如，OpenAI 或 Anthropic 的文件 API。"""

    mime_type: NotRequired[str]
    """MIME type of the video.

    Required for base64 data.

    [Examples from IANA](https://www.iana.org/assignments/media-types/media-types.xhtml#video)
    

    中文翻译:
    视频的 MIME 类型。
    对于 Base64 数据是必需的。
    [来自 IANA 的示例](https://www.iana.org/assignments/media-types/media-types.xhtml#video)"""

    index: NotRequired[int | str]
    """Index of block in aggregate response. Used during streaming.

    中文翻译:
    聚合响应中块的索引。在流式传输期间使用。"""

    url: NotRequired[str]
    """URL of the video.

    中文翻译:
    视频的网址。"""

    base64: NotRequired[str]
    """Data as a base64 string.

    中文翻译:
    Base64 字符串形式的数据。"""

    extras: NotRequired[dict[str, Any]]
    """Provider-specific metadata. This shouldn't be used for the video data itself.

    中文翻译:
    提供商特定的元数据。这不应该用于视频数据本身。"""


class AudioContentBlock(TypedDict):
    """Audio data.

    !!! note "Factory function"

        `create_audio_block` may also be used as a factory to create an
        `AudioContentBlock`. Benefits include:

        * Automatic ID generation (when not provided)
        * Required arguments strictly validated at creation time
    

    中文翻译:
    音频数据。
    !!!注意“工厂功能”
        `create_audio_block` 也可以用作工厂来创建
        `音频内容块`。好处包括：
        * 自动生成 ID（未提供时）
        * 必需的参数在创建时经过严格验证"""

    type: Literal["audio"]
    """Type of the content block. Used for discrimination.

    中文翻译:
    内容块的类型。用于歧视。"""

    id: NotRequired[str]
    """Unique identifier for this content block.

    Either:

    - Generated by the provider
    - Generated by LangChain upon creation (`UUID4` prefixed with `'lc_'`))
    

    中文翻译:
    该内容块的唯一标识符。
    要么：
    - 由提供商生成
    - 由 LangChain 在创建时生成（`UUID4` 前缀为 `'lc_'`））"""

    file_id: NotRequired[str]
    """Reference to the audio file in an external file storage system.

    For example, OpenAI or Anthropic's Files API.
    

    中文翻译:
    引用外部文件存储系统中的音频文件。
    例如，OpenAI 或 Anthropic 的文件 API。"""

    mime_type: NotRequired[str]
    """MIME type of the audio.

    Required for base64 data.

    [Examples from IANA](https://www.iana.org/assignments/media-types/media-types.xhtml#audio)
    

    中文翻译:
    音频的 MIME 类型。
    对于 Base64 数据是必需的。
    [来自 IANA 的示例](https://www.iana.org/assignments/media-types/media-types.xhtml#audio)"""

    index: NotRequired[int | str]
    """Index of block in aggregate response. Used during streaming.

    中文翻译:
    聚合响应中块的索引。在流式传输期间使用。"""

    url: NotRequired[str]
    """URL of the audio.

    中文翻译:
    音频的 URL。"""

    base64: NotRequired[str]
    """Data as a base64 string.

    中文翻译:
    Base64 字符串形式的数据。"""

    extras: NotRequired[dict[str, Any]]
    """Provider-specific metadata. This shouldn't be used for the audio data itself.

    中文翻译:
    提供商特定的元数据。这不应该用于音频数据本身。"""


class PlainTextContentBlock(TypedDict):
    """Plaintext data (e.g., from a `.txt` or `.md` document).

    !!! note

        A `PlainTextContentBlock` existed in `langchain-core<1.0.0`. Although the
        name has carried over, the structure has changed significantly. The only shared
        keys between the old and new versions are `type` and `text`, though the
        `type` value has changed from `'text'` to `'text-plain'`.

    !!! note

        Title and context are optional fields that may be passed to the model. See
        Anthropic [example](https://platform.claude.com/docs/en/build-with-claude/citations#citable-vs-non-citable-content).

    !!! note "Factory function"

        `create_plaintext_block` may also be used as a factory to create a
        `PlainTextContentBlock`. Benefits include:

        * Automatic ID generation (when not provided)
        * Required arguments strictly validated at creation time
    

    中文翻译:
    纯文本数据（例如，来自“.txt”或“.md”文档）。
    ！！！注释
        “langchain-core<1.0.0”中存在“PlainTextContentBlock”。虽然
        名称已沿用，结构已发生重大变化。唯一共享的
        新旧版本之间的键是“type”和“text”，尽管
        “type”值已从“text”更改为“text-plain”。
    ！！！注释
        标题和上下文是可以传递给模型的可选字段。参见
        人择 [示例](https://platform.claude.com/docs/en/build-with-claude/itations#citable-vs-non-citable-content)。
    ！！！注意“工厂功能”
        `create_plaintext_block` 也可以用作工厂来创建
        `纯文本内容块`。好处包括：
        * 自动生成 ID（未提供时）
        * 必需的参数在创建时经过严格验证"""

    type: Literal["text-plain"]
    """Type of the content block. Used for discrimination.

    中文翻译:
    内容块的类型。用于歧视。"""

    id: NotRequired[str]
    """Unique identifier for this content block.

    Either:

    - Generated by the provider
    - Generated by LangChain upon creation (`UUID4` prefixed with `'lc_'`))
    

    中文翻译:
    该内容块的唯一标识符。
    要么：
    - 由提供商生成
    - 由 LangChain 在创建时生成（`UUID4` 前缀为 `'lc_'`））"""

    file_id: NotRequired[str]
    """Reference to the plaintext file in an external file storage system.

    For example, OpenAI or Anthropic's Files API.
    

    中文翻译:
    引用外部文件存储系统中的明文文件。
    例如，OpenAI 或 Anthropic 的文件 API。"""

    mime_type: Literal["text/plain"]
    """MIME type of the file.

    Required for base64 data.
    

    中文翻译:
    文件的 MIME 类型。
    对于 Base64 数据是必需的。"""

    index: NotRequired[int | str]
    """Index of block in aggregate response. Used during streaming.

    中文翻译:
    聚合响应中块的索引。在流式传输期间使用。"""

    url: NotRequired[str]
    """URL of the plaintext.

    中文翻译:
    明文的 URL。"""

    base64: NotRequired[str]
    """Data as a base64 string.

    中文翻译:
    Base64 字符串形式的数据。"""

    text: NotRequired[str]
    """Plaintext content. This is optional if the data is provided as base64.

    中文翻译:
    明文内容。如果数据以 base64 形式提供，则此选项是可选的。"""

    title: NotRequired[str]
    """Title of the text data, e.g., the title of a document.

    中文翻译:
    文本数据的标题，例如文档的标题。"""

    context: NotRequired[str]
    """Context for the text, e.g., a description or summary of the text's content.

    中文翻译:
    文本的上下文，例如文本内容的描述或摘要。"""

    extras: NotRequired[dict[str, Any]]
    """Provider-specific metadata. This shouldn't be used for the data itself.

    中文翻译:
    提供商特定的元数据。这不应该用于数据本身。"""


class FileContentBlock(TypedDict):
    """File data that doesn't fit into other multimodal block types.

    This block is intended for files that are not images, audio, or plaintext. For
    example, it can be used for PDFs, Word documents, etc.

    If the file is an image, audio, or plaintext, you should use the corresponding
    content block type (e.g., `ImageContentBlock`, `AudioContentBlock`,
    `PlainTextContentBlock`).

    !!! note "Factory function"

        `create_file_block` may also be used as a factory to create a
        `FileContentBlock`. Benefits include:

        * Automatic ID generation (when not provided)
        * Required arguments strictly validated at creation time
    

    中文翻译:
    不适合其他多模式块类型的文件数据。
    该块适用于非图像、音频或纯文本的文件。对于
    例如，它可以用于 PDF、Word 文档等。
    如果文件是图像、音频或明文，则应使用相应的
    内容块类型（例如，`ImageContentBlock`、`AudioContentBlock`、
    `纯文本内容块`)。
    !!!注意“工厂功能”
        `create_file_block` 也可以用作工厂来创建
        `文件内容块`。好处包括：
        * 自动生成 ID（未提供时）
        * 必需的参数在创建时经过严格验证"""

    type: Literal["file"]
    """Type of the content block. Used for discrimination.

    中文翻译:
    内容块的类型。用于歧视。"""

    id: NotRequired[str]
    """Unique identifier for this content block.

    Used for tracking and referencing specific blocks (e.g., during streaming).

    Not to be confused with `file_id`, which references an external file in a
    storage system.

    Either:

    - Generated by the provider
    - Generated by LangChain upon creation (`UUID4` prefixed with `'lc_'`))
    

    中文翻译:
    该内容块的唯一标识符。
    用于跟踪和引用特定块（例如，在流式传输期间）。
    不要与“file_id”混淆，后者引用了外部文件
    存储系统。
    要么：
    - 由提供商生成
    - 由 LangChain 在创建时生成（`UUID4` 前缀为 `'lc_'`））"""

    file_id: NotRequired[str]
    """Reference to the file in an external file storage system.

    For example, a file ID from OpenAI's Files API or another cloud storage provider.
    This is distinct from `id`, which identifies the content block itself.
    

    中文翻译:
    引用外部文件存储系统中的文件。
    例如，来自 OpenAI 的文件 API 或其他云存储提供商的文件 ID。
    这与“id”不同，“id”标识内容块本身。"""

    mime_type: NotRequired[str]
    """MIME type of the file.

    Required for base64 data.

    [Examples from IANA](https://www.iana.org/assignments/media-types/media-types.xhtml)
    

    中文翻译:
    文件的 MIME 类型。
    对于 Base64 数据是必需的。
    [来自 IANA 的示例](https://www.iana.org/assignments/media-types/media-types.xhtml)"""

    index: NotRequired[int | str]
    """Index of block in aggregate response. Used during streaming.

    中文翻译:
    聚合响应中块的索引。在流式传输期间使用。"""

    url: NotRequired[str]
    """URL of the file.

    中文翻译:
    文件的 URL。"""

    base64: NotRequired[str]
    """Data as a base64 string.

    中文翻译:
    Base64 字符串形式的数据。"""

    extras: NotRequired[dict[str, Any]]
    """Provider-specific metadata. This shouldn't be used for the file data itself.

    中文翻译:
    提供商特定的元数据。这不应该用于文件数据本身。"""


# Future modalities to consider:
# 中文: 未来要考虑的方式：
# - 3D models
# 中文: - 3D模型
# - Tabular data
# 中文: - 表格数据


class NonStandardContentBlock(TypedDict):
    """Provider-specific content data.

    This block contains data for which there is not yet a standard type.

    The purpose of this block should be to simply hold a provider-specific payload.
    If a provider's non-standard output includes reasoning and tool calls, it should be
    the adapter's job to parse that payload and emit the corresponding standard
    `ReasoningContentBlock` and `ToolCalls`.

    Has no `extras` field, as provider-specific data should be included in the
    `value` field.

    !!! note "Factory function"

        `create_non_standard_block` may also be used as a factory to create a
        `NonStandardContentBlock`. Benefits include:

        * Automatic ID generation (when not provided)
        * Required arguments strictly validated at creation time
    

    中文翻译:
    提供商特定的内容数据。
    该块包含尚无标准类型的数据。
    该块的目的应该是简单地保存特定于提供者的有效负载。
    如果提供者的非标准输出包括推理和工具调用，那么它应该是
    适配器的工作是解析该有效负载并发出相应的标准
    `ReasoningContentBlock` 和 `ToolCalls`。
    没有“extras”字段，因为提供商特定的数据应包含在
    ‘值’字段。
    ！！！注意“工厂功能”
        `create_non_standard_block` 也可以用作工厂来创建
        `非标准内容块`。好处包括：
        * 自动生成 ID（未提供时）
        * 必需的参数在创建时经过严格验证"""

    type: Literal["non_standard"]
    """Type of the content block. Used for discrimination.

    中文翻译:
    内容块的类型。用于歧视。"""

    id: NotRequired[str]
    """Unique identifier for this content block.

    Either:

    - Generated by the provider
    - Generated by LangChain upon creation (`UUID4` prefixed with `'lc_'`))
    

    中文翻译:
    该内容块的唯一标识符。
    要么：
    - 由提供商生成
    - 由 LangChain 在创建时生成（`UUID4` 前缀为 `'lc_'`））"""

    value: dict[str, Any]
    """Provider-specific content data.

    中文翻译:
    提供商特定的内容数据。"""

    index: NotRequired[int | str]
    """Index of block in aggregate response. Used during streaming.

    中文翻译:
    聚合响应中块的索引。在流式传输期间使用。"""


# --- Aliases ---
# 中文: --- 别名 ---
DataContentBlock = (
    ImageContentBlock
    | VideoContentBlock
    | AudioContentBlock
    | PlainTextContentBlock
    | FileContentBlock
)
"""A union of all defined multimodal data `ContentBlock` types.

中文翻译:
所有定义的多模式数据“ContentBlock”类型的联合。"""

ToolContentBlock = (
    ToolCall | ToolCallChunk | ServerToolCall | ServerToolCallChunk | ServerToolResult
)

ContentBlock = (
    TextContentBlock
    | InvalidToolCall
    | ReasoningContentBlock
    | NonStandardContentBlock
    | DataContentBlock
    | ToolContentBlock
)
"""A union of all defined `ContentBlock` types and aliases.

中文翻译:
所有定义的“ContentBlock”类型和别名的联合。"""


KNOWN_BLOCK_TYPES = {
    # Text output
    # 中文: 文本输出
    "text",
    "reasoning",
    # Tools
    # 中文: 工具
    "tool_call",
    "invalid_tool_call",
    "tool_call_chunk",
    # Multimodal data
    # 中文: 多模态数据
    "image",
    "audio",
    "file",
    "text-plain",
    "video",
    # Server-side tool calls
    # 中文: 服务器端工具调用
    "server_tool_call",
    "server_tool_call_chunk",
    "server_tool_result",
    # Catch-all
    # 中文: 包罗万象
    "non_standard",
    # citation and non_standard_annotation intentionally omitted
    # 中文: 故意省略引用和非标准注释
}
"""These are block types known to `langchain-core >= 1.0.0`.

If a block has a type not in this set, it is considered to be provider-specific.

中文翻译:
这些是“langchain-core >= 1.0.0”已知的块类型。
如果一个块的类型不在此集合中，则它被认为是特定于提供者的。
"""


def _get_data_content_block_types() -> tuple[str, ...]:
    """Get type literals from DataContentBlock union members dynamically.

    Example: ("image", "video", "audio", "text-plain", "file")

    Note that old style multimodal blocks type literals with new style blocks.
    Specifically, "image", "audio", and "file".

    See the docstring of `_normalize_messages` in `language_models._utils` for details.
    

    中文翻译:
    动态从 DataContentBlock 联合成员获取类型文字。
    示例：（“图像”、“视频”、“音频”、“纯文本”、“文件”）
    请注意，旧样式多模式块使用新样式块键入文字。
    具体来说，“图像”、“音频”和“文件”。
    有关详细信息，请参阅“language_models._utils”中“_normalize_messages”的文档字符串。"""
    data_block_types = []

    for block_type in get_args(DataContentBlock):
        hints = get_type_hints(block_type)
        if "type" in hints:
            type_annotation = hints["type"]
            if hasattr(type_annotation, "__args__"):
                # This is a Literal type, get the literal value
                # 中文: 这是一个 Literal 类型，获取字面值
                literal_value = type_annotation.__args__[0]
                data_block_types.append(literal_value)

    return tuple(data_block_types)


def is_data_content_block(block: dict) -> bool:
    """Check if the provided content block is a data content block.

    Returns True for both v0 (old-style) and v1 (new-style) multimodal data blocks.

    Args:
        block: The content block to check.

    Returns:
        `True` if the content block is a data content block, `False` otherwise.
    

    中文翻译:
    检查提供的内容块是否是数据内容块。
    对于 v0（旧式）和 v1（新式）多模式数据块返回 True。
    参数：
        block：要检查的内容块。
    返回：
        如果内容块是数据内容块，则为“True”，否则为“False”。"""
    if block.get("type") not in _get_data_content_block_types():
        return False

    if any(key in block for key in ("url", "base64", "file_id", "text")):
        # Type is valid and at least one data field is present
        # 中文: 类型有效且至少存在一个数据字段
        # (Accepts old-style image and audio URLContentBlock)
        # 中文: （接受旧式图像和音频 URLContentBlock）

        # 'text' is checked to support v0 PlainTextContentBlock types
        # 中文: 检查“text”以支持 v0 PlainTextContentBlock 类型
        # We must guard against new style TextContentBlock which also has 'text' `type`
        # 中文: 我们必须警惕新样式的 TextContentBlock，它也有“文本”“类型”
        # by ensuring the presence of `source_type`
        # 中文: 通过确保“source_type”的存在
        if block["type"] == "text" and "source_type" not in block:  # noqa: SIM103  # This is more readable
            return False

        return True

    if "source_type" in block:
        # Old-style content blocks had possible types of 'image', 'audio', and 'file'
        # 中文: 旧式内容块可能具有“图像”、“音频”和“文件”类型
        # which is not captured in the prior check
        # 中文: 先前检查中未捕获的内容
        source_type = block["source_type"]
        if (source_type == "url" and "url" in block) or (
            source_type == "base64" and "data" in block
        ):
            return True
        if (source_type == "id" and "id" in block) or (
            source_type == "text" and "url" in block
        ):
            return True

    return False


def create_text_block(
    text: str,
    *,
    id: str | None = None,
    annotations: list[Annotation] | None = None,
    index: int | str | None = None,
    **kwargs: Any,
) -> TextContentBlock:
    """Create a `TextContentBlock`.

    Args:
        text: The text content of the block.
        id: Content block identifier.

            Generated automatically if not provided.
        annotations: `Citation`s and other annotations for the text.
        index: Index of block in aggregate response.

            Used during streaming.

    Returns:
        A properly formatted `TextContentBlock`.

    !!! note

        The `id` is generated automatically if not provided, using a UUID4 format
        prefixed with `'lc_'` to indicate it is a LangChain-generated ID.
    

    中文翻译:
    创建一个“TextContentBlock”。
    参数：
        text：块的文本内容。
        id：内容块标识符。
            如果未提供则自动生成。
        注释：文本的“引文”和其他注释。
        index：聚合响应中块的索引。
            在流式传输期间使用。
    返回：
        格式正确的“TextContentBlock”。
    ！！！注释
        如果未提供，则会使用 UUID4 格式自动生成“id”
        前缀为“lc_”，表示它是 LangChain 生成的 ID。"""
    block = TextContentBlock(
        type="text",
        text=text,
        id=ensure_id(id),
    )
    if annotations is not None:
        block["annotations"] = annotations
    if index is not None:
        block["index"] = index

    extras = {k: v for k, v in kwargs.items() if v is not None}
    if extras:
        block["extras"] = extras

    return block


def create_image_block(
    *,
    url: str | None = None,
    base64: str | None = None,
    file_id: str | None = None,
    mime_type: str | None = None,
    id: str | None = None,
    index: int | str | None = None,
    **kwargs: Any,
) -> ImageContentBlock:
    """Create an `ImageContentBlock`.

    Args:
        url: URL of the image.
        base64: Base64-encoded image data.
        file_id: ID of the image file from a file storage system.
        mime_type: MIME type of the image.

            Required for base64 data.
        id: Content block identifier.

            Generated automatically if not provided.
        index: Index of block in aggregate response.

            Used during streaming.

    Returns:
        A properly formatted `ImageContentBlock`.

    Raises:
        ValueError: If no image source is provided or if `base64` is used without
            `mime_type`.

    !!! note

        The `id` is generated automatically if not provided, using a UUID4 format
        prefixed with `'lc_'` to indicate it is a LangChain-generated ID.
    

    中文翻译:
    创建一个“ImageContentBlock”。
    参数：
        url：图像的 URL。
        base64：Base64 编码的图像数据。
        file_id：文件存储系统中图像文件的ID。
        mime_type：图像的 MIME 类型。
            对于 Base64 数据是必需的。
        id：内容块标识符。
            如果未提供则自动生成。
        index：聚合响应中块的索引。
            在流式传输期间使用。
    返回：
        格式正确的“ImageContentBlock”。
    加薪：
        ValueError：如果未提供图像源或使用“base64”而没有
            `mime_type`。
    ！！！注释
        如果未提供，则会使用 UUID4 格式自动生成“id”
        前缀为“lc_”，表示它是 LangChain 生成的 ID。"""
    if not any([url, base64, file_id]):
        msg = "Must provide one of: url, base64, or file_id"
        raise ValueError(msg)

    block = ImageContentBlock(type="image", id=ensure_id(id))

    if url is not None:
        block["url"] = url
    if base64 is not None:
        block["base64"] = base64
    if file_id is not None:
        block["file_id"] = file_id
    if mime_type is not None:
        block["mime_type"] = mime_type
    if index is not None:
        block["index"] = index

    extras = {k: v for k, v in kwargs.items() if v is not None}
    if extras:
        block["extras"] = extras

    return block


def create_video_block(
    *,
    url: str | None = None,
    base64: str | None = None,
    file_id: str | None = None,
    mime_type: str | None = None,
    id: str | None = None,
    index: int | str | None = None,
    **kwargs: Any,
) -> VideoContentBlock:
    """Create a `VideoContentBlock`.

    Args:
        url: URL of the video.
        base64: Base64-encoded video data.
        file_id: ID of the video file from a file storage system.
        mime_type: MIME type of the video.

            Required for base64 data.
        id: Content block identifier.

            Generated automatically if not provided.
        index: Index of block in aggregate response.

            Used during streaming.

    Returns:
        A properly formatted `VideoContentBlock`.

    Raises:
        ValueError: If no video source is provided or if `base64` is used without
            `mime_type`.

    !!! note

        The `id` is generated automatically if not provided, using a UUID4 format
        prefixed with `'lc_'` to indicate it is a LangChain-generated ID.
    

    中文翻译:
    创建一个“VideoContentBlock”。
    参数：
        url：视频的 URL。
        base64：Base64 编码的视频数据。
        file_id：文件存储系统中视频文件的ID。
        mime_type：视频的 MIME 类型。
            对于 Base64 数据是必需的。
        id：内容块标识符。
            如果未提供则自动生成。
        index：聚合响应中块的索引。
            在流式传输期间使用。
    返回：
        格式正确的“VideoContentBlock”。
    加薪：
        ValueError：如果未提供视频源或使用“base64”而没有
            `mime_type`。
    !!!注释
        如果未提供，则会使用 UUID4 格式自动生成“id”
        前缀为“lc_”，表示它是 LangChain 生成的 ID。"""
    if not any([url, base64, file_id]):
        msg = "Must provide one of: url, base64, or file_id"
        raise ValueError(msg)

    if base64 and not mime_type:
        msg = "mime_type is required when using base64 data"
        raise ValueError(msg)

    block = VideoContentBlock(type="video", id=ensure_id(id))

    if url is not None:
        block["url"] = url
    if base64 is not None:
        block["base64"] = base64
    if file_id is not None:
        block["file_id"] = file_id
    if mime_type is not None:
        block["mime_type"] = mime_type
    if index is not None:
        block["index"] = index

    extras = {k: v for k, v in kwargs.items() if v is not None}
    if extras:
        block["extras"] = extras

    return block


def create_audio_block(
    *,
    url: str | None = None,
    base64: str | None = None,
    file_id: str | None = None,
    mime_type: str | None = None,
    id: str | None = None,
    index: int | str | None = None,
    **kwargs: Any,
) -> AudioContentBlock:
    """Create an `AudioContentBlock`.

    Args:
        url: URL of the audio.
        base64: Base64-encoded audio data.
        file_id: ID of the audio file from a file storage system.
        mime_type: MIME type of the audio.

            Required for base64 data.
        id: Content block identifier.

            Generated automatically if not provided.
        index: Index of block in aggregate response.

            Used during streaming.

    Returns:
        A properly formatted `AudioContentBlock`.

    Raises:
        ValueError: If no audio source is provided or if `base64` is used without
            `mime_type`.

    !!! note

        The `id` is generated automatically if not provided, using a UUID4 format
        prefixed with `'lc_'` to indicate it is a LangChain-generated ID.
    

    中文翻译:
    创建一个“AudioContentBlock”。
    参数：
        url：音频的 URL。
        base64：Base64 编码的音频数据。
        file_id：文件存储系统中音频文件的ID。
        mime_type：音频的 MIME 类型。
            对于 Base64 数据是必需的。
        id：内容块标识符。
            如果未提供则自动生成。
        index：聚合响应中块的索引。
            在流式传输期间使用。
    返回：
        格式正确的“AudioContentBlock”。
    加薪：
        ValueError：如果未提供音频源或使用“base64”而没有
            `mime_type`。
    !!!注释
        如果未提供，则会使用 UUID4 格式自动生成“id”
        前缀为“lc_”，表示它是 LangChain 生成的 ID。"""
    if not any([url, base64, file_id]):
        msg = "Must provide one of: url, base64, or file_id"
        raise ValueError(msg)

    if base64 and not mime_type:
        msg = "mime_type is required when using base64 data"
        raise ValueError(msg)

    block = AudioContentBlock(type="audio", id=ensure_id(id))

    if url is not None:
        block["url"] = url
    if base64 is not None:
        block["base64"] = base64
    if file_id is not None:
        block["file_id"] = file_id
    if mime_type is not None:
        block["mime_type"] = mime_type
    if index is not None:
        block["index"] = index

    extras = {k: v for k, v in kwargs.items() if v is not None}
    if extras:
        block["extras"] = extras

    return block


def create_file_block(
    *,
    url: str | None = None,
    base64: str | None = None,
    file_id: str | None = None,
    mime_type: str | None = None,
    id: str | None = None,
    index: int | str | None = None,
    **kwargs: Any,
) -> FileContentBlock:
    """Create a `FileContentBlock`.

    Args:
        url: URL of the file.
        base64: Base64-encoded file data.
        file_id: ID of the file from a file storage system.
        mime_type: MIME type of the file.

            Required for base64 data.
        id: Content block identifier.

            Generated automatically if not provided.
        index: Index of block in aggregate response.

            Used during streaming.

    Returns:
        A properly formatted `FileContentBlock`.

    Raises:
        ValueError: If no file source is provided or if `base64` is used without
            `mime_type`.

    !!! note

        The `id` is generated automatically if not provided, using a UUID4 format
        prefixed with `'lc_'` to indicate it is a LangChain-generated ID.
    

    中文翻译:
    创建一个“FileContentBlock”。
    参数：
        url：文件的 URL。
        base64：Base64 编码的文件数据。
        file_id：文件存储系统中的文件的ID。
        mime_type：文件的 MIME 类型。
            对于 Base64 数据是必需的。
        id：内容块标识符。
            如果未提供则自动生成。
        index：聚合响应中块的索引。
            在流式传输期间使用。
    返回：
        格式正确的“FileContentBlock”。
    加薪：
        ValueError：如果未提供文件源或使用“base64”而没有
            `mime_type`。
    ！！！注释
        如果未提供，则会使用 UUID4 格式自动生成“id”
        前缀为“lc_”，表示它是 LangChain 生成的 ID。"""
    if not any([url, base64, file_id]):
        msg = "Must provide one of: url, base64, or file_id"
        raise ValueError(msg)

    if base64 and not mime_type:
        msg = "mime_type is required when using base64 data"
        raise ValueError(msg)

    block = FileContentBlock(type="file", id=ensure_id(id))

    if url is not None:
        block["url"] = url
    if base64 is not None:
        block["base64"] = base64
    if file_id is not None:
        block["file_id"] = file_id
    if mime_type is not None:
        block["mime_type"] = mime_type
    if index is not None:
        block["index"] = index

    extras = {k: v for k, v in kwargs.items() if v is not None}
    if extras:
        block["extras"] = extras

    return block


def create_plaintext_block(
    text: str | None = None,
    url: str | None = None,
    base64: str | None = None,
    file_id: str | None = None,
    title: str | None = None,
    context: str | None = None,
    id: str | None = None,
    index: int | str | None = None,
    **kwargs: Any,
) -> PlainTextContentBlock:
    """Create a `PlainTextContentBlock`.

    Args:
        text: The plaintext content.
        url: URL of the plaintext file.
        base64: Base64-encoded plaintext data.
        file_id: ID of the plaintext file from a file storage system.
        title: Title of the text data.
        context: Context or description of the text content.
        id: Content block identifier.

            Generated automatically if not provided.
        index: Index of block in aggregate response.

            Used during streaming.

    Returns:
        A properly formatted `PlainTextContentBlock`.

    !!! note

        The `id` is generated automatically if not provided, using a UUID4 format
        prefixed with `'lc_'` to indicate it is a LangChain-generated ID.
    

    中文翻译:
    创建一个“PlainTextContentBlock”。
    参数：
        text：明文内容。
        url：明文文件的URL。
        base64：Base64 编码的纯文本数据。
        file_id：文件存储系统中的明文文件的ID。
        title：文本数据的标题。
        context：文本内容的上下文或描述。
        id：内容块标识符。
            如果未提供则自动生成。
        index：聚合响应中块的索引。
            在流式传输期间使用。
    返回：
        格式正确的“PlainTextContentBlock”。
    ！！！注释
        如果未提供，则会使用 UUID4 格式自动生成“id”
        前缀为“lc_”，表示它是 LangChain 生成的 ID。"""
    block = PlainTextContentBlock(
        type="text-plain",
        mime_type="text/plain",
        id=ensure_id(id),
    )

    if text is not None:
        block["text"] = text
    if url is not None:
        block["url"] = url
    if base64 is not None:
        block["base64"] = base64
    if file_id is not None:
        block["file_id"] = file_id
    if title is not None:
        block["title"] = title
    if context is not None:
        block["context"] = context
    if index is not None:
        block["index"] = index

    extras = {k: v for k, v in kwargs.items() if v is not None}
    if extras:
        block["extras"] = extras

    return block


def create_tool_call(
    name: str,
    args: dict[str, Any],
    *,
    id: str | None = None,
    index: int | str | None = None,
    **kwargs: Any,
) -> ToolCall:
    """Create a `ToolCall`.

    Args:
        name: The name of the tool to be called.
        args: The arguments to the tool call.
        id: An identifier for the tool call.

            Generated automatically if not provided.
        index: Index of block in aggregate response.

            Used during streaming.

    Returns:
        A properly formatted `ToolCall`.

    !!! note

        The `id` is generated automatically if not provided, using a UUID4 format
        prefixed with `'lc_'` to indicate it is a LangChain-generated ID.
    

    中文翻译:
    创建一个“工具调用”。
    参数：
        name：要调用的工具的名称。
        args：工具调用的参数。
        id：工具调用的标识符。
            如果未提供则自动生成。
        index：聚合响应中块的索引。
            在流式传输期间使用。
    返回：
        格式正确的“ToolCall”。
    !!!注释
        如果未提供，则会使用 UUID4 格式自动生成“id”
        前缀为“lc_”，表示它是 LangChain 生成的 ID。"""
    block = ToolCall(
        type="tool_call",
        name=name,
        args=args,
        id=ensure_id(id),
    )

    if index is not None:
        block["index"] = index

    extras = {k: v for k, v in kwargs.items() if v is not None}
    if extras:
        block["extras"] = extras

    return block


def create_reasoning_block(
    reasoning: str | None = None,
    id: str | None = None,
    index: int | str | None = None,
    **kwargs: Any,
) -> ReasoningContentBlock:
    """Create a `ReasoningContentBlock`.

    Args:
        reasoning: The reasoning text or thought summary.
        id: Content block identifier.

            Generated automatically if not provided.
        index: Index of block in aggregate response.

            Used during streaming.

    Returns:
        A properly formatted `ReasoningContentBlock`.

    !!! note

        The `id` is generated automatically if not provided, using a UUID4 format
        prefixed with `'lc_'` to indicate it is a LangChain-generated ID.
    

    中文翻译:
    创建一个“推理内容块”。
    参数：
        推理：推理文本或思想总结。
        id：内容块标识符。
            如果未提供则自动生成。
        index：聚合响应中块的索引。
            在流式传输期间使用。
    返回：
        格式正确的“ReasoningContentBlock”。
    ！！！注释
        如果未提供，则会使用 UUID4 格式自动生成“id”
        前缀为“lc_”，表示它是 LangChain 生成的 ID。"""
    block = ReasoningContentBlock(
        type="reasoning",
        reasoning=reasoning or "",
        id=ensure_id(id),
    )

    if index is not None:
        block["index"] = index

    extras = {k: v for k, v in kwargs.items() if v is not None}
    if extras:
        block["extras"] = extras

    return block


def create_citation(
    *,
    url: str | None = None,
    title: str | None = None,
    start_index: int | None = None,
    end_index: int | None = None,
    cited_text: str | None = None,
    id: str | None = None,
    **kwargs: Any,
) -> Citation:
    """Create a `Citation`.

    Args:
        url: URL of the document source.
        title: Source document title.
        start_index: Start index in the response text where citation applies.
        end_index: End index in the response text where citation applies.
        cited_text: Excerpt of source text being cited.
        id: Content block identifier.

            Generated automatically if not provided.

    Returns:
        A properly formatted `Citation`.

    !!! note

        The `id` is generated automatically if not provided, using a UUID4 format
        prefixed with `'lc_'` to indicate it is a LangChain-generated ID.
    

    中文翻译:
    创建一个“引文”。
    参数：
        url：文档源的 URL。
        title：源文档标题。
        start_index：响应文本中引用的起始索引。
        end_index：响应文本中引用的结束索引。
        被引用的文本：被引用的源文本的摘录。
        id：内容块标识符。
            如果未提供则自动生成。
    返回：
        格式正确的“引文”。
    !!!注释
        如果未提供，则会使用 UUID4 格式自动生成“id”
        前缀为“lc_”，表示它是 LangChain 生成的 ID。"""
    block = Citation(type="citation", id=ensure_id(id))

    if url is not None:
        block["url"] = url
    if title is not None:
        block["title"] = title
    if start_index is not None:
        block["start_index"] = start_index
    if end_index is not None:
        block["end_index"] = end_index
    if cited_text is not None:
        block["cited_text"] = cited_text

    extras = {k: v for k, v in kwargs.items() if v is not None}
    if extras:
        block["extras"] = extras

    return block


def create_non_standard_block(
    value: dict[str, Any],
    *,
    id: str | None = None,
    index: int | str | None = None,
) -> NonStandardContentBlock:
    """Create a `NonStandardContentBlock`.

    Args:
        value: Provider-specific content data.
        id: Content block identifier.

            Generated automatically if not provided.
        index: Index of block in aggregate response.

            Used during streaming.

    Returns:
        A properly formatted `NonStandardContentBlock`.

    !!! note

        The `id` is generated automatically if not provided, using a UUID4 format
        prefixed with `'lc_'` to indicate it is a LangChain-generated ID.
    

    中文翻译:
    创建一个“NonStandardContentBlock”。
    参数：
        值：提供商特定的内容数据。
        id：内容块标识符。
            如果未提供则自动生成。
        index：聚合响应中块的索引。
            在流式传输期间使用。
    返回：
        格式正确的“NonStandardContentBlock”。
    !!!注释
        如果未提供，则会使用 UUID4 格式自动生成“id”
        前缀为“lc_”，表示它是 LangChain 生成的 ID。"""
    block = NonStandardContentBlock(
        type="non_standard",
        value=value,
        id=ensure_id(id),
    )

    if index is not None:
        block["index"] = index

    return block
