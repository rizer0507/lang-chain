"""Model profile types and utilities.

中文翻译:
模型配置文件类型和实用程序。"""

from typing_extensions import TypedDict


class ModelProfile(TypedDict, total=False):
    """Model profile.

    !!! warning "Beta feature"

        This is a beta feature. The format of model profiles is subject to change.

    Provides information about chat model capabilities, such as context window sizes
    and supported features.
    

    中文翻译:
    模型简介。
    ！！！警告“测试版功能”
        这是测试版功能。模型配置文件的格式可能会发生变化。
    提供有关聊天模型功能的信息，例如上下文窗口大小
    和支持的功能。"""

    # --- Input constraints ---
    # 中文: --- 输入约束 ---

    max_input_tokens: int
    """Maximum context window (tokens)

    中文翻译:
    最大上下文窗口（令牌）"""

    image_inputs: bool
    """Whether image inputs are supported.

    中文翻译:
    是否支持图片输入。"""
    # TODO: add more detail about formats?

    image_url_inputs: bool
    """Whether [image URL inputs](https://docs.langchain.com/oss/python/langchain/models#multimodal)
    are supported.

    中文翻译:
    是否[图像URL输入](https://docs.langchain.com/oss/python/langchain/models#multimodal)
    都支持。"""

    pdf_inputs: bool
    """Whether [PDF inputs](https://docs.langchain.com/oss/python/langchain/models#multimodal)
    are supported.

    中文翻译:
    是否[PDF输入](https://docs.langchain.com/oss/python/langchain/models#multimodal)
    都支持。"""
    # TODO: add more detail about formats? e.g. bytes or base64

    audio_inputs: bool
    """Whether [audio inputs](https://docs.langchain.com/oss/python/langchain/models#multimodal)
    are supported.

    中文翻译:
    是否[音频输入](https://docs.langchain.com/oss/python/langchain/models#multimodal)
    都支持。"""
    # TODO: add more detail about formats? e.g. bytes or base64

    video_inputs: bool
    """Whether [video inputs](https://docs.langchain.com/oss/python/langchain/models#multimodal)
    are supported.

    中文翻译:
    是否[视频输入](https://docs.langchain.com/oss/python/langchain/models#multimodal)
    都支持。"""
    # TODO: add more detail about formats? e.g. bytes or base64

    image_tool_message: bool
    """Whether images can be included in tool messages.

    中文翻译:
    工具消息中是否可以包含图像。"""

    pdf_tool_message: bool
    """Whether PDFs can be included in tool messages.

    中文翻译:
    PDF 是否可以包含在工具消息中。"""

    # --- Output constraints ---
    # 中文: --- 输出限制 ---

    max_output_tokens: int
    """Maximum output tokens

    中文翻译:
    最大输出令牌"""

    reasoning_output: bool
    """Whether the model supports [reasoning / chain-of-thought](https://docs.langchain.com/oss/python/langchain/models#reasoning)

    中文翻译:
    模型是否支持[reasoning / chain-of-thought](https://docs.langchain.com/oss/python/langchain/models#reasoning)"""

    image_outputs: bool
    """Whether [image outputs](https://docs.langchain.com/oss/python/langchain/models#multimodal)
    are supported.

    中文翻译:
    是否[图像输出](https://docs.langchain.com/oss/python/langchain/models#multimodal)
    都支持。"""

    audio_outputs: bool
    """Whether [audio outputs](https://docs.langchain.com/oss/python/langchain/models#multimodal)
    are supported.

    中文翻译:
    是否[音频输出](https://docs.langchain.com/oss/python/langchain/models#multimodal)
    都支持。"""

    video_outputs: bool
    """Whether [video outputs](https://docs.langchain.com/oss/python/langchain/models#multimodal)
    are supported.

    中文翻译:
    是否[视频输出](https://docs.langchain.com/oss/python/langchain/models#multimodal)
    都支持。"""

    # --- Tool calling ---
    # 中文: --- 工具调用 ---
    tool_calling: bool
    """Whether the model supports [tool calling](https://docs.langchain.com/oss/python/langchain/models#tool-calling)

    中文翻译:
    模型是否支持【工具调用】(https://docs.langchain.com/oss/python/langchain/models#tool-calling)"""

    tool_choice: bool
    """Whether the model supports [tool choice](https://docs.langchain.com/oss/python/langchain/models#forcing-tool-calls)

    中文翻译:
    模型是否支持【工具选择】(https://docs.langchain.com/oss/python/langchain/models#forcing-tool-calls)"""

    # --- Structured output ---
    # 中文: --- 结构化输出 ---
    structured_output: bool
    """Whether the model supports a native [structured output](https://docs.langchain.com/oss/python/langchain/models#structured-outputs)
    feature

    中文翻译:
    模型是否支持原生[结构化输出](https://docs.langchain.com/oss/python/langchain/models#structed-outputs)
    特征"""


ModelProfileRegistry = dict[str, ModelProfile]
"""Registry mapping model identifiers or names to their ModelProfile.

中文翻译:
注册表将模型标识符或名称映射到其 ModelProfile。"""
