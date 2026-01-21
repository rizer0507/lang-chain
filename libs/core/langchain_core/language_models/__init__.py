"""Core language model abstractions.

LangChain has two main classes to work with language models: chat models and
"old-fashioned" LLMs (string-in, string-out).

**Chat models**

Language models that use a sequence of messages as inputs and return chat messages
as outputs (as opposed to using plain text).

Chat models support the assignment of distinct roles to conversation messages, helping
to distinguish messages from the AI, users, and instructions such as system messages.

The key abstraction for chat models is
[`BaseChatModel`][langchain_core.language_models.BaseChatModel]. Implementations should
inherit from this class.

See existing [chat model integrations](https://docs.langchain.com/oss/python/integrations/chat).

**LLMs (legacy)**

Language models that takes a string as input and returns a string.

These are traditionally older models (newer models generally are chat models).

Although the underlying models are string in, string out, the LangChain wrappers also
allow these models to take messages as input. This gives them the same interface as
chat models. When messages are passed in as input, they will be formatted into a string
under the hood before being passed to the underlying model.

中文翻译:
核心语言模型抽象。
LangChain 有两个主要的类来处理语言模型：聊天模型和
“老式”法学硕士（字符串输入，字符串输出）。
**聊天模型**
使用消息序列作为输入并返回聊天消息的语言模型
作为输出（而不是使用纯文本）。
聊天模型支持为对话消息分配不同的角色，从而帮助
区分来自AI、用户的消息和系统消息等指令。
聊天模型的关键抽象是
[`BaseChatModel`][langchain_core.language_models.BaseChatModel]。实施应该
从这个类继承。
请参阅现有的[聊天模型集成](https://docs.langchain.com/oss/python/integrations/chat)。
**法学硕士（传统）**
将字符串作为输入并返回字符串的语言模型。
这些传统上是较旧的模型（较新的模型通常是聊天模型）。
虽然底层模型是 string in、string out，但 LangChain 包装器也
允许这些模型将消息作为输入。这给了他们相同的界面
聊天模型。当消息作为输入传入时，它们将被格式化为字符串
在传递到底层模型之前。
"""

from typing import TYPE_CHECKING

from langchain_core._import_utils import import_attr
from langchain_core.language_models._utils import is_openai_data_block

if TYPE_CHECKING:
    from langchain_core.language_models.base import (
        BaseLanguageModel,
        LangSmithParams,
        LanguageModelInput,
        LanguageModelLike,
        LanguageModelOutput,
        get_tokenizer,
    )
    from langchain_core.language_models.chat_models import (
        BaseChatModel,
        SimpleChatModel,
    )
    from langchain_core.language_models.fake import FakeListLLM, FakeStreamingListLLM
    from langchain_core.language_models.fake_chat_models import (
        FakeListChatModel,
        FakeMessagesListChatModel,
        GenericFakeChatModel,
        ParrotFakeChatModel,
    )
    from langchain_core.language_models.llms import LLM, BaseLLM
    from langchain_core.language_models.model_profile import (
        ModelProfile,
        ModelProfileRegistry,
    )

__all__ = (
    "LLM",
    "BaseChatModel",
    "BaseLLM",
    "BaseLanguageModel",
    "FakeListChatModel",
    "FakeListLLM",
    "FakeMessagesListChatModel",
    "FakeStreamingListLLM",
    "GenericFakeChatModel",
    "LangSmithParams",
    "LanguageModelInput",
    "LanguageModelLike",
    "LanguageModelOutput",
    "ModelProfile",
    "ModelProfileRegistry",
    "ParrotFakeChatModel",
    "SimpleChatModel",
    "get_tokenizer",
    "is_openai_data_block",
)

_dynamic_imports = {
    "BaseLanguageModel": "base",
    "LangSmithParams": "base",
    "LanguageModelInput": "base",
    "LanguageModelLike": "base",
    "LanguageModelOutput": "base",
    "get_tokenizer": "base",
    "BaseChatModel": "chat_models",
    "SimpleChatModel": "chat_models",
    "FakeListLLM": "fake",
    "FakeStreamingListLLM": "fake",
    "FakeListChatModel": "fake_chat_models",
    "FakeMessagesListChatModel": "fake_chat_models",
    "GenericFakeChatModel": "fake_chat_models",
    "ParrotFakeChatModel": "fake_chat_models",
    "LLM": "llms",
    "ModelProfile": "model_profile",
    "ModelProfileRegistry": "model_profile",
    "BaseLLM": "llms",
    "is_openai_data_block": "_utils",
}


def __getattr__(attr_name: str) -> object:
    module_name = _dynamic_imports.get(attr_name)
    result = import_attr(attr_name, module_name, __spec__.parent)
    globals()[attr_name] = result
    return result


def __dir__() -> list[str]:
    return list(__all__)
