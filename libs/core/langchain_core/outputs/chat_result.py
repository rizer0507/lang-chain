"""Chat result schema.

中文翻译:
聊天结果架构。"""

from pydantic import BaseModel

from langchain_core.outputs.chat_generation import ChatGeneration


class ChatResult(BaseModel):
    """Use to represent the result of a chat model call with a single prompt.

    This container is used internally by some implementations of chat model,
    it will eventually be mapped to a more general `LLMResult` object, and
    then projected into an `AIMessage` object.

    LangChain users working with chat models will usually access information via
    `AIMessage` (returned from runnable interfaces) or `LLMResult` (available
    via callbacks). Please refer the `AIMessage` and `LLMResult` schema documentation
    for more information.
    

    中文翻译:
    用于通过单个提示表示聊天模型调用的结果。
    该容器由聊天模型的某些实现在内部使用，
    它最终会被映射到一个更通用的“LLMResult”对象，并且
    然后投影到“AIMessage”对象中。
    使用聊天模型的 LangChain 用户通常会通过以下方式访问信息
    `AIMessage`（从可运行接口返回）或`LLMResult`（可用
    通过回调）。请参阅“AIMessage”和“LLMResult”架构文档
    了解更多信息。"""

    generations: list[ChatGeneration]
    """List of the chat generations.

    Generations is a list to allow for multiple candidate generations for a single
    input prompt.
    

    中文翻译:
    聊天世代列表。
    Generations 是一个列表，允许单个代有多个候选代
    输入提示。"""
    llm_output: dict | None = None
    """For arbitrary LLM provider specific output.

    This dictionary is a free-form dictionary that can contain any information that the
    provider wants to return. It is not standardized and is provider-specific.

    Users should generally avoid relying on this field and instead rely on
    accessing relevant information from standardized fields present in
    AIMessage.
    

    中文翻译:
    对于任意 LLM 提供商的特定输出。
    该字典是一个自由格式的字典，可以包含该字典中的任何信息。
    提供者想要返回。它不是标准化的，并且是特定于提供商的。
    用户通常应该避免依赖该字段，而应该依赖
    从存在的标准化字段中访问相关信息
    人工智能消息。"""
