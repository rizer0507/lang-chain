"""Output classes.

Used to represent the output of a language model call and the output of a chat.

The top container for information is the `LLMResult` object. `LLMResult` is used by both
chat models and LLMs. This object contains the output of the language model and any
additional information that the model provider wants to return.

When invoking models via the standard runnable methods (e.g. invoke, batch, etc.):

- Chat models will return `AIMessage` objects.
- LLMs will return regular text strings.

In addition, users can access the raw output of either LLMs or chat models via
callbacks. The `on_chat_model_end` and `on_llm_end` callbacks will return an
LLMResult object containing the generated outputs and any additional information
returned by the model provider.

In general, if information is already available in the AIMessage object, it is
recommended to access it from there rather than from the `LLMResult` object.

中文翻译:
输出类。
用于表示语言模型调用的输出和聊天的输出。
信息的顶层容器是“LLMResult”对象。两者都使用“LLMResult”
聊天模型和法学硕士。该对象包含语言模型的输出和任何
模型提供者想要返回的附加信息。
通过标准可运行方法（例如调用、批处理等）调用模型时：
- 聊天模型将返回“AIMessage”对象。
- 法学硕士将返回常规文本字符串。
此外，用户可以通过以下方式访问法学硕士或聊天模型的原始输出
回调。 `on_chat_model_end` 和 `on_llm_end` 回调将返回
LLMResult 对象包含生成的输出和任何附加信息
由模型提供者返回。
一般来说，如果 AIMessage 对象中已有信息，则它是
建议从那里访问它而不是从“LLMResult”对象。
"""

from typing import TYPE_CHECKING

from langchain_core._import_utils import import_attr

if TYPE_CHECKING:
    from langchain_core.outputs.chat_generation import (
        ChatGeneration,
        ChatGenerationChunk,
    )
    from langchain_core.outputs.chat_result import ChatResult
    from langchain_core.outputs.generation import Generation, GenerationChunk
    from langchain_core.outputs.llm_result import LLMResult
    from langchain_core.outputs.run_info import RunInfo

__all__ = (
    "ChatGeneration",
    "ChatGenerationChunk",
    "ChatResult",
    "Generation",
    "GenerationChunk",
    "LLMResult",
    "RunInfo",
)

_dynamic_imports = {
    "ChatGeneration": "chat_generation",
    "ChatGenerationChunk": "chat_generation",
    "ChatResult": "chat_result",
    "Generation": "generation",
    "GenerationChunk": "generation",
    "LLMResult": "llm_result",
    "RunInfo": "run_info",
}


def __getattr__(attr_name: str) -> object:
    module_name = _dynamic_imports.get(attr_name)
    result = import_attr(attr_name, module_name, __spec__.parent)
    globals()[attr_name] = result
    return result


def __dir__() -> list[str]:
    return list(__all__)
