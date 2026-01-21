"""Chat models for conversational AI.

中文翻译:
对话式人工智能的聊天模型。"""

from __future__ import annotations

import asyncio
import inspect
import json
import typing
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from functools import cached_property
from operator import itemgetter
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import override

from langchain_core.caches import BaseCache
from langchain_core.callbacks import (
    AsyncCallbackManager,
    AsyncCallbackManagerForLLMRun,
    CallbackManager,
    CallbackManagerForLLMRun,
    Callbacks,
)
from langchain_core.globals import get_llm_cache
from langchain_core.language_models._utils import (
    _normalize_messages,
    _update_message_content_to_blocks,
)
from langchain_core.language_models.base import (
    BaseLanguageModel,
    LangSmithParams,
    LanguageModelInput,
)
from langchain_core.language_models.model_profile import ModelProfile
from langchain_core.load import dumpd, dumps
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    AnyMessage,
    BaseMessage,
    convert_to_messages,
    is_data_content_block,
    message_chunk_to_message,
)
from langchain_core.messages import content as types
from langchain_core.messages.block_translators.openai import (
    convert_to_openai_image_block,
)
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    PydanticToolsParser,
)
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    ChatResult,
    Generation,
    LLMResult,
    RunInfo,
)
from langchain_core.outputs.chat_generation import merge_chat_generation_chunks
from langchain_core.prompt_values import ChatPromptValue, PromptValue, StringPromptValue
from langchain_core.rate_limiters import BaseRateLimiter
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from langchain_core.runnables.config import ensure_config, run_in_executor
from langchain_core.tracers._streaming import _StreamingCallbackHandler
from langchain_core.utils.function_calling import (
    convert_to_json_schema,
    convert_to_openai_tool,
)
from langchain_core.utils.pydantic import TypeBaseModel, is_basemodel_subclass
from langchain_core.utils.utils import LC_ID_PREFIX, from_env

if TYPE_CHECKING:
    import uuid

    from langchain_core.output_parsers.base import OutputParserLike
    from langchain_core.runnables import Runnable, RunnableConfig
    from langchain_core.tools import BaseTool


def _generate_response_from_error(error: BaseException) -> list[ChatGeneration]:
    if hasattr(error, "response"):
        response = error.response
        metadata: dict = {}
        if hasattr(response, "json"):
            try:
                metadata["body"] = response.json()
            except Exception:
                try:
                    metadata["body"] = getattr(response, "text", None)
                except Exception:
                    metadata["body"] = None
        if hasattr(response, "headers"):
            try:
                metadata["headers"] = dict(response.headers)
            except Exception:
                metadata["headers"] = None
        if hasattr(response, "status_code"):
            metadata["status_code"] = response.status_code
        if hasattr(error, "request_id"):
            metadata["request_id"] = error.request_id
        generations = [
            ChatGeneration(message=AIMessage(content="", response_metadata=metadata))
        ]
    else:
        generations = []

    return generations


def _format_for_tracing(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Format messages for tracing in `on_chat_model_start`.

    - Update image content blocks to OpenAI Chat Completions format (backward
    compatibility).
    - Add `type` key to content blocks that have a single key.

    Args:
        messages: List of messages to format.

    Returns:
        List of messages formatted for tracing.

    

    中文翻译:
    格式化消息以在“on_chat_model_start”中进行跟踪。
    - 将图像内容块更新为 OpenAI Chat Completions 格式（向后
    兼容性）。
    - 将“type”键添加到具有单个键的内容块。
    参数：
        messages：要格式化的消息列表。
    返回：
        为跟踪而格式化的消息列表。"""
    messages_to_trace = []
    for message in messages:
        message_to_trace = message
        if isinstance(message.content, list):
            for idx, block in enumerate(message.content):
                if isinstance(block, dict):
                    # Update image content blocks to OpenAI # Chat Completions format.
                    # 中文: 将图像内容块更新为 OpenAI # Chat Completions 格式。
                    if (
                        block.get("type") == "image"
                        and is_data_content_block(block)
                        and not ("file_id" in block or block.get("source_type") == "id")
                    ):
                        if message_to_trace is message:
                            # Shallow copy
                            # 中文: 浅拷贝
                            message_to_trace = message.model_copy()
                            message_to_trace.content = list(message_to_trace.content)

                        message_to_trace.content[idx] = (  # type: ignore[index]  # mypy confused by .model_copy
                            convert_to_openai_image_block(block)
                        )
                    elif (
                        block.get("type") == "file"
                        and is_data_content_block(block)  # v0 (image/audio/file) or v1
                        and "base64" in block
                        # Backward compat: convert v1 base64 blocks to v0
                        # 中文: 向后兼容：将 v1 base64 块转换为 v0
                    ):
                        if message_to_trace is message:
                            # Shallow copy
                            # 中文: 浅拷贝
                            message_to_trace = message.model_copy()
                            message_to_trace.content = list(message_to_trace.content)

                        message_to_trace.content[idx] = {  # type: ignore[index]
                            **{k: v for k, v in block.items() if k != "base64"},
                            "data": block["base64"],
                            "source_type": "base64",
                        }
                    elif len(block) == 1 and "type" not in block:
                        # Tracing assumes all content blocks have a "type" key. Here
                        # 中文: 跟踪假设所有内容块都有一个“type”键。这里
                        # we add this key if it is missing, and there's an obvious
                        # 中文: 如果缺少此键，我们将其添加，并且有一个明显的
                        # choice for the type (e.g., a single key in the block).
                        # 中文: 类型的选择（例如，块中的单个键）。
                        if message_to_trace is message:
                            # Shallow copy
                            # 中文: 浅拷贝
                            message_to_trace = message.model_copy()
                            message_to_trace.content = list(message_to_trace.content)
                        key = next(iter(block))
                        message_to_trace.content[idx] = {  # type: ignore[index]
                            "type": key,
                            key: block[key],
                        }
        messages_to_trace.append(message_to_trace)

    return messages_to_trace


def generate_from_stream(stream: Iterator[ChatGenerationChunk]) -> ChatResult:
    """Generate from a stream.

    Args:
        stream: Iterator of `ChatGenerationChunk`.

    Raises:
        ValueError: If no generations are found in the stream.

    Returns:
        Chat result.

    

    中文翻译:
    从流生成。
    参数：
        流：“ChatGenerationChunk”的迭代器。
    加薪：
        ValueError：如果在流中未找到任何代。
    返回：
        聊天结果。"""
    generation = next(stream, None)
    if generation:
        generation += list(stream)
    if generation is None:
        msg = "No generations found in stream."
        raise ValueError(msg)
    return ChatResult(
        generations=[
            ChatGeneration(
                message=message_chunk_to_message(generation.message),
                generation_info=generation.generation_info,
            )
        ]
    )


async def agenerate_from_stream(
    stream: AsyncIterator[ChatGenerationChunk],
) -> ChatResult:
    """Async generate from a stream.

    Args:
        stream: Iterator of `ChatGenerationChunk`.

    Returns:
        Chat result.

    

    中文翻译:
    从流中异步生成。
    参数：
        流：“ChatGenerationChunk”的迭代器。
    返回：
        聊天结果。"""
    chunks = [chunk async for chunk in stream]
    return await run_in_executor(None, generate_from_stream, iter(chunks))


def _format_ls_structured_output(ls_structured_output_format: dict | None) -> dict:
    if ls_structured_output_format:
        try:
            ls_structured_output_format_dict = {
                "ls_structured_output_format": {
                    "kwargs": ls_structured_output_format.get("kwargs", {}),
                    "schema": convert_to_json_schema(
                        ls_structured_output_format["schema"]
                    ),
                }
            }
        except ValueError:
            ls_structured_output_format_dict = {}
    else:
        ls_structured_output_format_dict = {}

    return ls_structured_output_format_dict


class BaseChatModel(BaseLanguageModel[AIMessage], ABC):
    r"""Base class for chat models.

    Key imperative methods:
        Methods that actually call the underlying model.

        This table provides a brief overview of the main imperative methods. Please see the base `Runnable` reference for full documentation.

        | Method                 | Input                                                        | Output                                                     | Description                                                                      |
        | ---------------------- | ------------------------------------------------------------ | ---------------------------------------------------------- | -------------------------------------------------------------------------------- |
        | `invoke`               | `str` \| `list[dict | tuple | BaseMessage]` \| `PromptValue` | `BaseMessage`                                              | A single chat model call.                                                        |
        | `ainvoke`              | `'''`                                                        | `BaseMessage`                                              | Defaults to running `invoke` in an async executor.                               |
        | `stream`               | `'''`                                                        | `Iterator[BaseMessageChunk]`                               | Defaults to yielding output of `invoke`.                                         |
        | `astream`              | `'''`                                                        | `AsyncIterator[BaseMessageChunk]`                          | Defaults to yielding output of `ainvoke`.                                        |
        | `astream_events`       | `'''`                                                        | `AsyncIterator[StreamEvent]`                               | Event types: `on_chat_model_start`, `on_chat_model_stream`, `on_chat_model_end`. |
        | `batch`                | `list[''']`                                                  | `list[BaseMessage]`                                        | Defaults to running `invoke` in concurrent threads.                              |
        | `abatch`               | `list[''']`                                                  | `list[BaseMessage]`                                        | Defaults to running `ainvoke` in concurrent threads.                             |
        | `batch_as_completed`   | `list[''']`                                                  | `Iterator[tuple[int, Union[BaseMessage, Exception]]]`      | Defaults to running `invoke` in concurrent threads.                              |
        | `abatch_as_completed`  | `list[''']`                                                  | `AsyncIterator[tuple[int, Union[BaseMessage, Exception]]]` | Defaults to running `ainvoke` in concurrent threads.                             |

    Key declarative methods:
        Methods for creating another `Runnable` using the chat model.

        This table provides a brief overview of the main declarative methods. Please see the reference for each method for full documentation.

        | Method                       | Description                                                                                |
        | ---------------------------- | ------------------------------------------------------------------------------------------ |
        | `bind_tools`                 | Create chat model that can call tools.                                                     |
        | `with_structured_output`     | Create wrapper that structures model output using schema.                                  |
        | `with_retry`                 | Create wrapper that retries model calls on failure.                                        |
        | `with_fallbacks`             | Create wrapper that falls back to other models on failure.                                 |
        | `configurable_fields`        | Specify init args of the model that can be configured at runtime via the `RunnableConfig`. |
        | `configurable_alternatives`  | Specify alternative models which can be swapped in at runtime via the `RunnableConfig`.    |

    Creating custom chat model:
        Custom chat model implementations should inherit from this class.
        Please reference the table below for information about which
        methods and properties are required or optional for implementations.

        | Method/Property                  | Description                                                        | Required          |
        | -------------------------------- | ------------------------------------------------------------------ | ----------------- |
        | `_generate`                      | Use to generate a chat result from a prompt                        | Required          |
        | `_llm_type` (property)           | Used to uniquely identify the type of the model. Used for logging. | Required          |
        | `_identifying_params` (property) | Represent model parameterization for tracing purposes.             | Optional          |
        | `_stream`                        | Use to implement streaming                                         | Optional          |
        | `_agenerate`                     | Use to implement a native async method                             | Optional          |
        | `_astream`                       | Use to implement async version of `_stream`                        | Optional          |

    

中文翻译:
聊天模型的基类。
    关键命令式方法：
        实际调用底层模型的方法。
        该表提供了主要命令式方法的简要概述。请参阅基本“Runnable”参考以获取完整文档。
        |方法|输入 |输出|描述 |
        | ---------------------- | ------------------------------------------------------------------------ | ---------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
        | `调用` | `str` \| `列表[字典|元组|基本消息]` \| `提示值` | `基本消息` |单个聊天模型调用。                                                        |
        | `ainvoke` | '''` | `基本消息` |默认在异步执行器中运行“invoke”。                               |
        | `流` | '''` | `迭代器[BaseMessageChunk]` |默认生成“invoke”的输出。                                         |
        | `astream` | '''` | `AsyncIterator[BaseMessageChunk]` |默认生成 `ainvoke` 的输出。                                        |
        | `astream_events` | '''` | `AsyncIterator[StreamEvent]` |事件类型：`on_chat_model_start`、`on_chat_model_stream`、`on_chat_model_end`。 |
        | `批量` | `列表[''']` | `列表[基本消息]` |默认在并发线程中运行“invoke”。                              |
        | `批量` | `列表[''']` | `列表[基本消息]` |默认在并发线程中运行“ainvoke”。                             |
        | `batch_as_completed` | `列表[''']` | `迭代器[元组[int, Union[BaseMessage, Exception]]]` |默认在并发线程中运行“invoke”。                              |
        | `批量完成` | `列表[''']` | `AsyncIterator[tuple[int, Union[BaseMessage, Exception]]]` |默认在并发线程中运行“ainvoke”。                             |
    关键声明方法：
        使用聊天模型创建另一个“Runnable”的方法。
        该表提供了主要声明方法的简要概述。请参阅每种方法的参考以获取完整文档。
        |方法|描述 |
        | ---------------------------- | ------------------------------------------------------------------------------------------ |
        | `bind_tools` |创建可以调用工具的聊天模型。                                                     |
        | `with_structed_output` |创建使用架构构建模型输出的包装器。                                  |
        | `with_retry` |创建在失败时重试模型调用的包装器。                                        |
        | `with_fallbacks` |创建在失败时回退到其他模型的包装器。                                 |
        | `可配置字段` |指定可以在运行时通过“RunnableConfig”配置的模型的初始化参数。 |
        | `可配置的替代品` |指定可以在运行时通过“RunnableConfig”交换的替代模型。    |
    创建自定义聊天模型：
        自定义聊天模型实现应该继承自此类。
        请参考下表了解有关哪些信息
        方法和属性对于实现来说是必需的或可选的。|方法/属性 |描述 |必填 |
        | -------------------------------- | ------------------------------------------------------------------ | ----------------- |
        | `_生成` |用于根据提示生成聊天结果 |必填 |
        | `_llm_type`（属性）|用于唯一标识模型的类型。用于记录。 |必填|
        | `_identifying_params`（属性）|表示用于跟踪目的的模型参数化。             |可选|
        | `_stream` |用于实现流式 |可选|
        | `_生成` |用于实现本机异步方法 |可选|
        | `_astream` |用于实现 `_stream` 的异步版本 |可选|"""  # noqa: E501

    rate_limiter: BaseRateLimiter | None = Field(default=None, exclude=True)
    "An optional rate limiter to use for limiting the number of requests."

    disable_streaming: bool | Literal["tool_calling"] = False
    """Whether to disable streaming for this model.

    If streaming is bypassed, then `stream`/`astream`/`astream_events` will
    defer to `invoke`/`ainvoke`.

    - If `True`, will always bypass streaming case.
    - If `'tool_calling'`, will bypass streaming case only when the model is called
        with a `tools` keyword argument. In other words, LangChain will automatically
        switch to non-streaming behavior (`invoke`) only when the tools argument is
        provided. This offers the best of both worlds.
    - If `False` (Default), will always use streaming case if available.

    The main reason for this flag is that code might be written using `stream` and
    a user may want to swap out a given model for another model whose the implementation
    does not properly support streaming.
    

    中文翻译:
    是否禁用此模型的流式传输。
    如果绕过流传输，则 `stream`/`astream`/`astream_events` 将
    遵循 `invoke`/`ainvoke`。
    - 如果为“True”，将始终绕过流情况。
    - 如果是“tool_calling”，则仅在调用模型时才会绕过流情况
        带有“tools”关键字参数。也就是说，浪链会自动
        仅当工具参数为时才切换到非流行为（“调用”）
        提供。这提供了两全其美的优点。
    - 如果为“False”（默认），则将始终使用流式传输情况（如果可用）。
    该标志的主要原因是代码可以使用“stream”和
    用户可能想要将给定模型替换为另一个模型，该模型的实现
    无法正确支持流媒体。"""

    output_version: str | None = Field(
        default_factory=from_env("LC_OUTPUT_VERSION", default=None)
    )
    """Version of `AIMessage` output format to store in message content.

    `AIMessage.content_blocks` will lazily parse the contents of `content` into a
    standard format. This flag can be used to additionally store the standard format
    in message content, e.g., for serialization purposes.

    Supported values:

    - `'v0'`: provider-specific format in content (can lazily-parse with
        `content_blocks`)
    - `'v1'`: standardized format in content (consistent with `content_blocks`)

    Partner packages (e.g.,
    [`langchain-openai`](https://pypi.org/project/langchain-openai)) can also use this
    field to roll out new content formats in a backward-compatible way.

    !!! version-added "Added in `langchain-core` 1.0.0"

    

    中文翻译:
    用于存储在消息内容中的“AIMessage”输出格式的版本。
    `AIMessage.content_blocks` 会将 `content` 的内容延迟解析为
    标准格式。该标志可用于额外存储标准格式
    在消息内容中，例如用于序列化目的。
    支持的值：
    - `'v0'`：内容中特定于提供者的格式（可以延迟解析
        `内容块`)
    - `'v1'`：内容的标准化格式（与`content_blocks`一致）
    合作伙伴包（例如，
    [`langchain-openai`](https://pypi.org/project/langchain-openai)) 也可以使用这个
    领域以向后兼容的方式推出新的内容格式。
    !!! version-added “在 `langchain-core` 1.0.0 中添加”"""

    profile: ModelProfile | None = Field(default=None, exclude=True)
    """Profile detailing model capabilities.

    !!! warning "Beta feature"

        This is a beta feature. The format of model profiles is subject to change.

    If not specified, automatically loaded from the provider package on initialization
    if data is available.

    Example profile data includes context window sizes, supported modalities, or support
    for tool calling, structured output, and other features.

    !!! version-added "Added in `langchain-core` 1.1.0"
    

    中文翻译:
    详细描述模型功能的概要文件。
    !!!警告“测试版功能”
        这是测试版功能。模型配置文件的格式可能会发生变化。
    如果未指定，则在初始化时自动从提供程序包加载
    如果数据可用。
    示例配置文件数据包括上下文窗口大小、支持的模式或支持
    用于工具调用、结构化输出和其他功能。
    !!! version-added “在 `langchain-core` 1.1.0 中添加”"""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @cached_property
    def _serialized(self) -> dict[str, Any]:
        # self is always a Serializable object in this case, thus the result is
        # 中文: 在这种情况下，self 始终是可序列化对象，因此结果是
        # guaranteed to be a dict since dumps uses the default callback, which uses
        # 中文: 保证是一个字典，因为 dumps 使用默认回调，它使用
        # obj.to_json which always returns TypedDict subclasses
        # 中文: obj.to_json 总是返回 TypedDict 子类
        return cast("dict[str, Any]", dumpd(self))

    # --- Runnable methods ---
    # 中文: --- 可运行的方法 ---

    @property
    @override
    def OutputType(self) -> Any:
        """Get the output type for this `Runnable`.

        中文翻译:
        获取此“Runnable”的输出类型。"""
        return AnyMessage

    def _convert_input(self, model_input: LanguageModelInput) -> PromptValue:
        if isinstance(model_input, PromptValue):
            return model_input
        if isinstance(model_input, str):
            return StringPromptValue(text=model_input)
        if isinstance(model_input, Sequence):
            return ChatPromptValue(messages=convert_to_messages(model_input))
        msg = (
            f"Invalid input type {type(model_input)}. "
            "Must be a PromptValue, str, or list of BaseMessages."
        )
        raise ValueError(msg)

    @override
    def invoke(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AIMessage:
        config = ensure_config(config)
        return cast(
            "AIMessage",
            cast(
                "ChatGeneration",
                self.generate_prompt(
                    [self._convert_input(input)],
                    stop=stop,
                    callbacks=config.get("callbacks"),
                    tags=config.get("tags"),
                    metadata=config.get("metadata"),
                    run_name=config.get("run_name"),
                    run_id=config.pop("run_id", None),
                    **kwargs,
                ).generations[0][0],
            ).message,
        )

    @override
    async def ainvoke(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AIMessage:
        config = ensure_config(config)
        llm_result = await self.agenerate_prompt(
            [self._convert_input(input)],
            stop=stop,
            callbacks=config.get("callbacks"),
            tags=config.get("tags"),
            metadata=config.get("metadata"),
            run_name=config.get("run_name"),
            run_id=config.pop("run_id", None),
            **kwargs,
        )
        return cast(
            "AIMessage", cast("ChatGeneration", llm_result.generations[0][0]).message
        )

    def _should_stream(
        self,
        *,
        async_api: bool,
        run_manager: CallbackManagerForLLMRun
        | AsyncCallbackManagerForLLMRun
        | None = None,
        **kwargs: Any,
    ) -> bool:
        """Determine if a given model call should hit the streaming API.

        中文翻译:
        确定给定的模型调用是否应访问流 API。"""
        sync_not_implemented = type(self)._stream == BaseChatModel._stream  # noqa: SLF001
        async_not_implemented = type(self)._astream == BaseChatModel._astream  # noqa: SLF001

        # Check if streaming is implemented.
        # 中文: 检查是否实现了流式传输。
        if (not async_api) and sync_not_implemented:
            return False
        # Note, since async falls back to sync we check both here.
        # 中文: 请注意，由于异步回退到同步，因此我们在这里检查两者。
        if async_api and async_not_implemented and sync_not_implemented:
            return False

        # Check if streaming has been disabled on this instance.
        # 中文: 检查此实例上是否已禁用流式传输。
        if self.disable_streaming is True:
            return False
        # We assume tools are passed in via "tools" kwarg in all models.
        # 中文: 我们假设所有模型中的工具都是通过“tools”kwarg 传入的。
        if self.disable_streaming == "tool_calling" and kwargs.get("tools"):
            return False

        # Check if a runtime streaming flag has been passed in.
        # 中文: 检查是否传入了运行时流标志。
        if "stream" in kwargs:
            return bool(kwargs["stream"])

        if "streaming" in self.model_fields_set:
            streaming_value = getattr(self, "streaming", None)
            if isinstance(streaming_value, bool):
                return streaming_value

        # Check if any streaming callback handlers have been passed in.
        # 中文: 检查是否传入了任何流式回调处理程序。
        handlers = run_manager.handlers if run_manager else []
        return any(isinstance(h, _StreamingCallbackHandler) for h in handlers)

    @override
    def stream(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> Iterator[AIMessageChunk]:
        if not self._should_stream(async_api=False, **{**kwargs, "stream": True}):
            # Model doesn't implement streaming, so use default implementation
            # 中文: 模型没有实现流式传输，因此使用默认实现
            yield cast(
                "AIMessageChunk",
                self.invoke(input, config=config, stop=stop, **kwargs),
            )
        else:
            config = ensure_config(config)
            messages = self._convert_input(input).to_messages()
            ls_structured_output_format = kwargs.pop(
                "ls_structured_output_format", None
            ) or kwargs.pop("structured_output_format", None)
            ls_structured_output_format_dict = _format_ls_structured_output(
                ls_structured_output_format
            )

            params = self._get_invocation_params(stop=stop, **kwargs)
            options = {"stop": stop, **kwargs, **ls_structured_output_format_dict}
            inheritable_metadata = {
                **(config.get("metadata") or {}),
                **self._get_ls_params(stop=stop, **kwargs),
            }
            callback_manager = CallbackManager.configure(
                config.get("callbacks"),
                self.callbacks,
                self.verbose,
                config.get("tags"),
                self.tags,
                inheritable_metadata,
                self.metadata,
            )
            (run_manager,) = callback_manager.on_chat_model_start(
                self._serialized,
                [_format_for_tracing(messages)],
                invocation_params=params,
                options=options,
                name=config.get("run_name"),
                run_id=config.pop("run_id", None),
                batch_size=1,
            )

            chunks: list[ChatGenerationChunk] = []

            if self.rate_limiter:
                self.rate_limiter.acquire(blocking=True)

            try:
                input_messages = _normalize_messages(messages)
                run_id = "-".join((LC_ID_PREFIX, str(run_manager.run_id)))
                yielded = False
                index = -1
                index_type = ""
                for chunk in self._stream(input_messages, stop=stop, **kwargs):
                    if chunk.message.id is None:
                        chunk.message.id = run_id
                    chunk.message.response_metadata = _gen_info_and_msg_metadata(chunk)
                    if self.output_version == "v1":
                        # Overwrite .content with .content_blocks
                        # 中文: 用 .content_blocks 覆盖 .content
                        chunk.message = _update_message_content_to_blocks(
                            chunk.message, "v1"
                        )
                        for block in cast(
                            "list[types.ContentBlock]", chunk.message.content
                        ):
                            if block["type"] != index_type:
                                index_type = block["type"]
                                index += 1
                            if "index" not in block:
                                block["index"] = index
                    run_manager.on_llm_new_token(
                        cast("str", chunk.message.content), chunk=chunk
                    )
                    chunks.append(chunk)
                    yield cast("AIMessageChunk", chunk.message)
                    yielded = True

                # Yield a final empty chunk with chunk_position="last" if not yet
                # 中文: 如果尚未生成一个带有 chunk_position="last" 的最终空块
                # yielded
                # 中文: 产生了
                if (
                    yielded
                    and isinstance(chunk.message, AIMessageChunk)
                    and not chunk.message.chunk_position
                ):
                    empty_content: str | list = (
                        "" if isinstance(chunk.message.content, str) else []
                    )
                    msg_chunk = AIMessageChunk(
                        content=empty_content, chunk_position="last", id=run_id
                    )
                    run_manager.on_llm_new_token(
                        "", chunk=ChatGenerationChunk(message=msg_chunk)
                    )
                    yield msg_chunk
            except BaseException as e:
                generations_with_error_metadata = _generate_response_from_error(e)
                chat_generation_chunk = merge_chat_generation_chunks(chunks)
                if chat_generation_chunk:
                    generations = [
                        [chat_generation_chunk],
                        generations_with_error_metadata,
                    ]
                else:
                    generations = [generations_with_error_metadata]
                run_manager.on_llm_error(
                    e,
                    response=LLMResult(generations=generations),
                )
                raise

            generation = merge_chat_generation_chunks(chunks)
            if generation is None:
                err = ValueError("No generation chunks were returned")
                run_manager.on_llm_error(err, response=LLMResult(generations=[]))
                raise err

            run_manager.on_llm_end(LLMResult(generations=[[generation]]))

    @override
    async def astream(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[AIMessageChunk]:
        if not self._should_stream(async_api=True, **{**kwargs, "stream": True}):
            # No async or sync stream is implemented, so fall back to ainvoke
            # 中文: 没有实现异步或同步流，因此回退到 ainvoke
            yield cast(
                "AIMessageChunk",
                await self.ainvoke(input, config=config, stop=stop, **kwargs),
            )
            return

        config = ensure_config(config)
        messages = self._convert_input(input).to_messages()

        ls_structured_output_format = kwargs.pop(
            "ls_structured_output_format", None
        ) or kwargs.pop("structured_output_format", None)
        ls_structured_output_format_dict = _format_ls_structured_output(
            ls_structured_output_format
        )

        params = self._get_invocation_params(stop=stop, **kwargs)
        options = {"stop": stop, **kwargs, **ls_structured_output_format_dict}
        inheritable_metadata = {
            **(config.get("metadata") or {}),
            **self._get_ls_params(stop=stop, **kwargs),
        }
        callback_manager = AsyncCallbackManager.configure(
            config.get("callbacks"),
            self.callbacks,
            self.verbose,
            config.get("tags"),
            self.tags,
            inheritable_metadata,
            self.metadata,
        )
        (run_manager,) = await callback_manager.on_chat_model_start(
            self._serialized,
            [_format_for_tracing(messages)],
            invocation_params=params,
            options=options,
            name=config.get("run_name"),
            run_id=config.pop("run_id", None),
            batch_size=1,
        )

        if self.rate_limiter:
            await self.rate_limiter.aacquire(blocking=True)

        chunks: list[ChatGenerationChunk] = []

        try:
            input_messages = _normalize_messages(messages)
            run_id = "-".join((LC_ID_PREFIX, str(run_manager.run_id)))
            yielded = False
            index = -1
            index_type = ""
            async for chunk in self._astream(
                input_messages,
                stop=stop,
                **kwargs,
            ):
                if chunk.message.id is None:
                    chunk.message.id = run_id
                chunk.message.response_metadata = _gen_info_and_msg_metadata(chunk)
                if self.output_version == "v1":
                    # Overwrite .content with .content_blocks
                    # 中文: 用 .content_blocks 覆盖 .content
                    chunk.message = _update_message_content_to_blocks(
                        chunk.message, "v1"
                    )
                    for block in cast(
                        "list[types.ContentBlock]", chunk.message.content
                    ):
                        if block["type"] != index_type:
                            index_type = block["type"]
                            index += 1
                        if "index" not in block:
                            block["index"] = index
                await run_manager.on_llm_new_token(
                    cast("str", chunk.message.content), chunk=chunk
                )
                chunks.append(chunk)
                yield cast("AIMessageChunk", chunk.message)
                yielded = True

            # Yield a final empty chunk with chunk_position="last" if not yet yielded
            # 中文: 如果尚未产生，则产生一个带有 chunk_position="last" 的最终空块
            if (
                yielded
                and isinstance(chunk.message, AIMessageChunk)
                and not chunk.message.chunk_position
            ):
                empty_content: str | list = (
                    "" if isinstance(chunk.message.content, str) else []
                )
                msg_chunk = AIMessageChunk(
                    content=empty_content, chunk_position="last", id=run_id
                )
                await run_manager.on_llm_new_token(
                    "", chunk=ChatGenerationChunk(message=msg_chunk)
                )
                yield msg_chunk
        except BaseException as e:
            generations_with_error_metadata = _generate_response_from_error(e)
            chat_generation_chunk = merge_chat_generation_chunks(chunks)
            if chat_generation_chunk:
                generations = [[chat_generation_chunk], generations_with_error_metadata]
            else:
                generations = [generations_with_error_metadata]
            await run_manager.on_llm_error(
                e,
                response=LLMResult(generations=generations),
            )
            raise

        generation = merge_chat_generation_chunks(chunks)
        if not generation:
            err = ValueError("No generation chunks were returned")
            await run_manager.on_llm_error(err, response=LLMResult(generations=[]))
            raise err

        await run_manager.on_llm_end(
            LLMResult(generations=[[generation]]),
        )

    # --- Custom methods ---
    # 中文: --- 自定义方法 ---

    def _combine_llm_outputs(self, _llm_outputs: list[dict | None], /) -> dict:
        return {}

    def _convert_cached_generations(self, cache_val: list) -> list[ChatGeneration]:
        """Convert cached Generation objects to ChatGeneration objects.

        Handle case where cache contains Generation objects instead of
        ChatGeneration objects. This can happen due to serialization/deserialization
        issues or legacy cache data (see #22389).

        Args:
            cache_val: List of cached generation objects.

        Returns:
            List of ChatGeneration objects.

        

        中文翻译:
        将缓存的 Generation 对象转换为 ChatGeneration 对象。
        处理缓存包含 Generation 对象而不是
        ChatGeneration 对象。这可能是由于序列化/反序列化而发生的
        问题或遗留缓存数据（参见#22389）。
        参数：
            cache_val：缓存的生成对象列表。
        返回：
            ChatGeneration 对象的列表。"""
        converted_generations = []
        for gen in cache_val:
            if isinstance(gen, Generation) and not isinstance(gen, ChatGeneration):
                # Convert Generation to ChatGeneration by creating AIMessage
                # 中文: 通过创建 AIMessage 将 Generation 转换为 ChatGeneration
                # from the text content
                # 中文: 从文字内容来看
                chat_gen = ChatGeneration(
                    message=AIMessage(content=gen.text),
                    generation_info=gen.generation_info,
                )
                converted_generations.append(chat_gen)
            else:
                # Already a ChatGeneration or other expected type
                # 中文: 已经是 ChatGeneration 或其他预期类型
                if hasattr(gen, "message") and isinstance(gen.message, AIMessage):
                    # We zero out cost on cache hits
                    # 中文: 我们将缓存命中的成本归零
                    gen.message = gen.message.model_copy(
                        update={
                            "usage_metadata": {
                                **(gen.message.usage_metadata or {}),
                                "total_cost": 0,
                            }
                        }
                    )
                converted_generations.append(gen)
        return converted_generations

    def _get_invocation_params(
        self,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        params = self.dict()
        params["stop"] = stop
        return {**params, **kwargs}

    def _get_ls_params(
        self,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> LangSmithParams:
        """Get standard params for tracing.

        中文翻译:
        获取用于跟踪的标准参数。"""
        # get default provider from class name
        # 中文: 从类名获取默认提供者
        default_provider = self.__class__.__name__
        if default_provider.startswith("Chat"):
            default_provider = default_provider[4:].lower()
        elif default_provider.endswith("Chat"):
            default_provider = default_provider[:-4]
        default_provider = default_provider.lower()

        ls_params = LangSmithParams(ls_provider=default_provider, ls_model_type="chat")
        if stop:
            ls_params["ls_stop"] = stop

        # model
        # 中文: 模型
        if "model" in kwargs and isinstance(kwargs["model"], str):
            ls_params["ls_model_name"] = kwargs["model"]
        elif hasattr(self, "model") and isinstance(self.model, str):
            ls_params["ls_model_name"] = self.model
        elif hasattr(self, "model_name") and isinstance(self.model_name, str):
            ls_params["ls_model_name"] = self.model_name

        # temperature
        # 中文: 温度
        if "temperature" in kwargs and isinstance(kwargs["temperature"], float):
            ls_params["ls_temperature"] = kwargs["temperature"]
        elif hasattr(self, "temperature") and isinstance(self.temperature, float):
            ls_params["ls_temperature"] = self.temperature

        # max_tokens
        # 中文: 最大令牌数
        if "max_tokens" in kwargs and isinstance(kwargs["max_tokens"], int):
            ls_params["ls_max_tokens"] = kwargs["max_tokens"]
        elif hasattr(self, "max_tokens") and isinstance(self.max_tokens, int):
            ls_params["ls_max_tokens"] = self.max_tokens

        return ls_params

    def _get_llm_string(self, stop: list[str] | None = None, **kwargs: Any) -> str:
        if self.is_lc_serializable():
            params = {**kwargs, "stop": stop}
            param_string = str(sorted(params.items()))
            # This code is not super efficient as it goes back and forth between
            # 中文: 这段代码并不是非常高效，因为它在
            # json and dict.
            # 中文: json 和 dict.
            serialized_repr = self._serialized
            _cleanup_llm_representation(serialized_repr, 1)
            llm_string = json.dumps(serialized_repr, sort_keys=True)
            return llm_string + "---" + param_string
        params = self._get_invocation_params(stop=stop, **kwargs)
        params = {**params, **kwargs}
        return str(sorted(params.items()))

    def generate(
        self,
        messages: list[list[BaseMessage]],
        stop: list[str] | None = None,
        callbacks: Callbacks = None,
        *,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        run_name: str | None = None,
        run_id: uuid.UUID | None = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Pass a sequence of prompts to the model and return model generations.

        This method should make use of batched calls for models that expose a batched
        API.

        Use this method when you want to:

        1. Take advantage of batched calls,
        2. Need more output from the model than just the top generated value,
        3. Are building chains that are agnostic to the underlying language model
            type (e.g., pure text completion models vs chat models).

        Args:
            messages: List of list of messages.
            stop: Stop words to use when generating.

                Model output is cut off at the first occurrence of any of these
                substrings.
            callbacks: `Callbacks` to pass through.

                Used for executing additional functionality, such as logging or
                streaming, throughout generation.
            tags: The tags to apply.
            metadata: The metadata to apply.
            run_name: The name of the run.
            run_id: The ID of the run.
            **kwargs: Arbitrary additional keyword arguments.

                These are usually passed to the model provider API call.

        Returns:
            An `LLMResult`, which contains a list of candidate `Generations` for each
                input prompt and additional model provider-specific output.

        

        中文翻译:
        将一系列提示传递给模型并返回模型生成。
        此方法应该对公开批量的模型使用批量调用
        API。
        当您想要执行以下操作时，请使用此方法：
        1.利用批量调用的优势，
        2. 需要模型的更多输出而不仅仅是顶部生成的值，
        3.正在构建与底层语言模型无关的链
            类型（例如，纯文本完成模型与聊天模型）。
        参数：
            messages：消息列表列表。
            stop：生成时使用的停止词。
                模型输出在第一次出现这些情况时被切断
                子串。
            回调：要传递的“回调”。
                用于执行附加功能，例如日志记录或
                流式传输，贯穿一代。
            标签：要应用的标签。
            元数据：要应用的元数据。
            run_name：运行的名称。
            run_id：运行的 ID。
            **kwargs：任意附加关键字参数。
                这些通常会传递给模型提供者 API 调用。
        返回：
            一个“LLMResult”，其中包含每个候选“Generations”的列表
                输入提示和附加模型提供者特定的输出。"""
        ls_structured_output_format = kwargs.pop(
            "ls_structured_output_format", None
        ) or kwargs.pop("structured_output_format", None)
        ls_structured_output_format_dict = _format_ls_structured_output(
            ls_structured_output_format
        )

        params = self._get_invocation_params(stop=stop, **kwargs)
        options = {"stop": stop, **ls_structured_output_format_dict}
        inheritable_metadata = {
            **(metadata or {}),
            **self._get_ls_params(stop=stop, **kwargs),
        }

        callback_manager = CallbackManager.configure(
            callbacks,
            self.callbacks,
            self.verbose,
            tags,
            self.tags,
            inheritable_metadata,
            self.metadata,
        )
        messages_to_trace = [
            _format_for_tracing(message_list) for message_list in messages
        ]
        run_managers = callback_manager.on_chat_model_start(
            self._serialized,
            messages_to_trace,
            invocation_params=params,
            options=options,
            name=run_name,
            run_id=run_id,
            batch_size=len(messages),
        )
        results = []
        input_messages = [
            _normalize_messages(message_list) for message_list in messages
        ]
        for i, m in enumerate(input_messages):
            try:
                results.append(
                    self._generate_with_cache(
                        m,
                        stop=stop,
                        run_manager=run_managers[i] if run_managers else None,
                        **kwargs,
                    )
                )
            except BaseException as e:
                if run_managers:
                    generations_with_error_metadata = _generate_response_from_error(e)
                    run_managers[i].on_llm_error(
                        e,
                        response=LLMResult(
                            generations=[generations_with_error_metadata]
                        ),
                    )
                raise
        flattened_outputs = [
            LLMResult(generations=[res.generations], llm_output=res.llm_output)
            for res in results
        ]
        llm_output = self._combine_llm_outputs([res.llm_output for res in results])
        generations = [res.generations for res in results]
        output = LLMResult(generations=generations, llm_output=llm_output)
        if run_managers:
            run_infos = []
            for manager, flattened_output in zip(
                run_managers, flattened_outputs, strict=False
            ):
                manager.on_llm_end(flattened_output)
                run_infos.append(RunInfo(run_id=manager.run_id))
            output.run = run_infos
        return output

    async def agenerate(
        self,
        messages: list[list[BaseMessage]],
        stop: list[str] | None = None,
        callbacks: Callbacks = None,
        *,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        run_name: str | None = None,
        run_id: uuid.UUID | None = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Asynchronously pass a sequence of prompts to a model and return generations.

        This method should make use of batched calls for models that expose a batched
        API.

        Use this method when you want to:

        1. Take advantage of batched calls,
        2. Need more output from the model than just the top generated value,
        3. Are building chains that are agnostic to the underlying language model
            type (e.g., pure text completion models vs chat models).

        Args:
            messages: List of list of messages.
            stop: Stop words to use when generating.

                Model output is cut off at the first occurrence of any of these
                substrings.
            callbacks: `Callbacks` to pass through.

                Used for executing additional functionality, such as logging or
                streaming, throughout generation.
            tags: The tags to apply.
            metadata: The metadata to apply.
            run_name: The name of the run.
            run_id: The ID of the run.
            **kwargs: Arbitrary additional keyword arguments.

                These are usually passed to the model provider API call.

        Returns:
            An `LLMResult`, which contains a list of candidate `Generations` for each
                input prompt and additional model provider-specific output.

        

        中文翻译:
        将一系列提示异步传递给模型并返回生成。
        此方法应该对公开批量的模型使用批量调用
        API。
        当您想要执行以下操作时，请使用此方法：
        1.利用批量调用的优势，
        2. 需要模型的更多输出而不仅仅是顶部生成的值，
        3.正在构建与底层语言模型无关的链
            类型（例如，纯文本完成模型与聊天模型）。
        参数：
            messages：消息列表列表。
            stop：生成时使用的停止词。
                模型输出在第一次出现这些情况时被切断
                子串。
            回调：要传递的“回调”。
                用于执行附加功能，例如日志记录或
                流式传输，贯穿一代。
            标签：要应用的标签。
            元数据：要应用的元数据。
            run_name：运行的名称。
            run_id：运行的 ID。
            **kwargs：任意附加关键字参数。
                这些通常会传递给模型提供者 API 调用。
        返回：
            一个“LLMResult”，其中包含每个候选“Generations”的列表
                输入提示和附加模型提供者特定的输出。"""
        ls_structured_output_format = kwargs.pop(
            "ls_structured_output_format", None
        ) or kwargs.pop("structured_output_format", None)
        ls_structured_output_format_dict = _format_ls_structured_output(
            ls_structured_output_format
        )

        params = self._get_invocation_params(stop=stop, **kwargs)
        options = {"stop": stop, **ls_structured_output_format_dict}
        inheritable_metadata = {
            **(metadata or {}),
            **self._get_ls_params(stop=stop, **kwargs),
        }

        callback_manager = AsyncCallbackManager.configure(
            callbacks,
            self.callbacks,
            self.verbose,
            tags,
            self.tags,
            inheritable_metadata,
            self.metadata,
        )

        messages_to_trace = [
            _format_for_tracing(message_list) for message_list in messages
        ]
        run_managers = await callback_manager.on_chat_model_start(
            self._serialized,
            messages_to_trace,
            invocation_params=params,
            options=options,
            name=run_name,
            batch_size=len(messages),
            run_id=run_id,
        )

        input_messages = [
            _normalize_messages(message_list) for message_list in messages
        ]
        results = await asyncio.gather(
            *[
                self._agenerate_with_cache(
                    m,
                    stop=stop,
                    run_manager=run_managers[i] if run_managers else None,
                    **kwargs,
                )
                for i, m in enumerate(input_messages)
            ],
            return_exceptions=True,
        )
        exceptions = []
        for i, res in enumerate(results):
            if isinstance(res, BaseException):
                if run_managers:
                    generations_with_error_metadata = _generate_response_from_error(res)
                    await run_managers[i].on_llm_error(
                        res,
                        response=LLMResult(
                            generations=[generations_with_error_metadata]
                        ),
                    )
                exceptions.append(res)
        if exceptions:
            if run_managers:
                await asyncio.gather(
                    *[
                        run_manager.on_llm_end(
                            LLMResult(
                                generations=[res.generations],  # type: ignore[union-attr]
                                llm_output=res.llm_output,  # type: ignore[union-attr]
                            )
                        )
                        for run_manager, res in zip(run_managers, results, strict=False)
                        if not isinstance(res, Exception)
                    ]
                )
            raise exceptions[0]
        flattened_outputs = [
            LLMResult(generations=[res.generations], llm_output=res.llm_output)  # type: ignore[union-attr]
            for res in results
        ]
        llm_output = self._combine_llm_outputs([res.llm_output for res in results])  # type: ignore[union-attr]
        generations = [res.generations for res in results]  # type: ignore[union-attr]
        output = LLMResult(generations=generations, llm_output=llm_output)
        await asyncio.gather(
            *[
                run_manager.on_llm_end(flattened_output)
                for run_manager, flattened_output in zip(
                    run_managers, flattened_outputs, strict=False
                )
            ]
        )
        if run_managers:
            output.run = [
                RunInfo(run_id=run_manager.run_id) for run_manager in run_managers
            ]
        return output

    @override
    def generate_prompt(
        self,
        prompts: list[PromptValue],
        stop: list[str] | None = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> LLMResult:
        prompt_messages = [p.to_messages() for p in prompts]
        return self.generate(prompt_messages, stop=stop, callbacks=callbacks, **kwargs)

    @override
    async def agenerate_prompt(
        self,
        prompts: list[PromptValue],
        stop: list[str] | None = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> LLMResult:
        prompt_messages = [p.to_messages() for p in prompts]
        return await self.agenerate(
            prompt_messages, stop=stop, callbacks=callbacks, **kwargs
        )

    def _generate_with_cache(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        llm_cache = self.cache if isinstance(self.cache, BaseCache) else get_llm_cache()
        # We should check the cache unless it's explicitly set to False
        # 中文: 我们应该检查缓存，除非它明确设置为 False
        # A None cache means we should use the default global cache
        # 中文: None 缓存意味着我们应该使用默认的全局缓存
        # if it's configured.
        # 中文: 如果已配置。
        check_cache = self.cache or self.cache is None
        if check_cache:
            if llm_cache:
                llm_string = self._get_llm_string(stop=stop, **kwargs)
                normalized_messages = [
                    (
                        msg.model_copy(update={"id": None})
                        if getattr(msg, "id", None) is not None
                        else msg
                    )
                    for msg in messages
                ]
                prompt = dumps(normalized_messages)
                cache_val = llm_cache.lookup(prompt, llm_string)
                if isinstance(cache_val, list):
                    converted_generations = self._convert_cached_generations(cache_val)
                    return ChatResult(generations=converted_generations)
            elif self.cache is None:
                pass
            else:
                msg = "Asked to cache, but no cache found at `langchain.cache`."
                raise ValueError(msg)

        # Apply the rate limiter after checking the cache, since
        # 中文: 检查缓存后应用速率限制器，因为
        # we usually don't want to rate limit cache lookups, but
        # 中文: 我们通常不想限制缓存查找的速率，但是
        # we do want to rate limit API requests.
        # 中文: 我们确实想要限制 API 请求的速率。
        if self.rate_limiter:
            self.rate_limiter.acquire(blocking=True)

        # If stream is not explicitly set, check if implicitly requested by
        # 中文: 如果未显式设置流，请检查是否隐式请求
        # astream_events() or astream_log(). Bail out if _stream not implemented
        # 中文: astream_events() 或 astream_log()。如果 _stream 未实现则退出
        if self._should_stream(
            async_api=False,
            run_manager=run_manager,
            **kwargs,
        ):
            chunks: list[ChatGenerationChunk] = []
            run_id: str | None = (
                f"{LC_ID_PREFIX}-{run_manager.run_id}" if run_manager else None
            )
            yielded = False
            index = -1
            index_type = ""
            for chunk in self._stream(messages, stop=stop, **kwargs):
                chunk.message.response_metadata = _gen_info_and_msg_metadata(chunk)
                if self.output_version == "v1":
                    # Overwrite .content with .content_blocks
                    # 中文: 用 .content_blocks 覆盖 .content
                    chunk.message = _update_message_content_to_blocks(
                        chunk.message, "v1"
                    )
                    for block in cast(
                        "list[types.ContentBlock]", chunk.message.content
                    ):
                        if block["type"] != index_type:
                            index_type = block["type"]
                            index += 1
                        if "index" not in block:
                            block["index"] = index
                if run_manager:
                    if chunk.message.id is None:
                        chunk.message.id = run_id
                    run_manager.on_llm_new_token(
                        cast("str", chunk.message.content), chunk=chunk
                    )
                chunks.append(chunk)
                yielded = True

            # Yield a final empty chunk with chunk_position="last" if not yet yielded
            # 中文: 如果尚未产生，则产生一个带有 chunk_position="last" 的最终空块
            if (
                yielded
                and isinstance(chunk.message, AIMessageChunk)
                and not chunk.message.chunk_position
            ):
                empty_content: str | list = (
                    "" if isinstance(chunk.message.content, str) else []
                )
                chunk = ChatGenerationChunk(
                    message=AIMessageChunk(
                        content=empty_content, chunk_position="last", id=run_id
                    )
                )
                if run_manager:
                    run_manager.on_llm_new_token("", chunk=chunk)
                chunks.append(chunk)
            result = generate_from_stream(iter(chunks))
        elif inspect.signature(self._generate).parameters.get("run_manager"):
            result = self._generate(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
        else:
            result = self._generate(messages, stop=stop, **kwargs)

        if self.output_version == "v1":
            # Overwrite .content with .content_blocks
            # 中文: 用 .content_blocks 覆盖 .content
            for generation in result.generations:
                generation.message = _update_message_content_to_blocks(
                    generation.message, "v1"
                )

        # Add response metadata to each generation
        # 中文: 为每一代添加响应元数据
        for idx, generation in enumerate(result.generations):
            if run_manager and generation.message.id is None:
                generation.message.id = f"{LC_ID_PREFIX}-{run_manager.run_id}-{idx}"
            generation.message.response_metadata = _gen_info_and_msg_metadata(
                generation
            )
        if len(result.generations) == 1 and result.llm_output is not None:
            result.generations[0].message.response_metadata = {
                **result.llm_output,
                **result.generations[0].message.response_metadata,
            }
        if check_cache and llm_cache:
            llm_cache.update(prompt, llm_string, result.generations)
        return result

    async def _agenerate_with_cache(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        llm_cache = self.cache if isinstance(self.cache, BaseCache) else get_llm_cache()
        # We should check the cache unless it's explicitly set to False
        # 中文: 我们应该检查缓存，除非它明确设置为 False
        # A None cache means we should use the default global cache
        # 中文: None 缓存意味着我们应该使用默认的全局缓存
        # if it's configured.
        # 中文: 如果已配置。
        check_cache = self.cache or self.cache is None
        if check_cache:
            if llm_cache:
                llm_string = self._get_llm_string(stop=stop, **kwargs)
                normalized_messages = [
                    (
                        msg.model_copy(update={"id": None})
                        if getattr(msg, "id", None) is not None
                        else msg
                    )
                    for msg in messages
                ]
                prompt = dumps(normalized_messages)
                cache_val = await llm_cache.alookup(prompt, llm_string)
                if isinstance(cache_val, list):
                    converted_generations = self._convert_cached_generations(cache_val)
                    return ChatResult(generations=converted_generations)
            elif self.cache is None:
                pass
            else:
                msg = "Asked to cache, but no cache found at `langchain.cache`."
                raise ValueError(msg)

        # Apply the rate limiter after checking the cache, since
        # 中文: 检查缓存后应用速率限制器，因为
        # we usually don't want to rate limit cache lookups, but
        # 中文: 我们通常不想限制缓存查找的速率，但是
        # we do want to rate limit API requests.
        # 中文: 我们确实想要限制 API 请求的速率。
        if self.rate_limiter:
            await self.rate_limiter.aacquire(blocking=True)

        # If stream is not explicitly set, check if implicitly requested by
        # 中文: 如果未显式设置流，请检查是否隐式请求
        # astream_events() or astream_log(). Bail out if _astream not implemented
        # 中文: astream_events() 或 astream_log()。如果 _astream 未实现则退出
        if self._should_stream(
            async_api=True,
            run_manager=run_manager,
            **kwargs,
        ):
            chunks: list[ChatGenerationChunk] = []
            run_id: str | None = (
                f"{LC_ID_PREFIX}-{run_manager.run_id}" if run_manager else None
            )
            yielded = False
            index = -1
            index_type = ""
            async for chunk in self._astream(messages, stop=stop, **kwargs):
                chunk.message.response_metadata = _gen_info_and_msg_metadata(chunk)
                if self.output_version == "v1":
                    # Overwrite .content with .content_blocks
                    # 中文: 用 .content_blocks 覆盖 .content
                    chunk.message = _update_message_content_to_blocks(
                        chunk.message, "v1"
                    )
                    for block in cast(
                        "list[types.ContentBlock]", chunk.message.content
                    ):
                        if block["type"] != index_type:
                            index_type = block["type"]
                            index += 1
                        if "index" not in block:
                            block["index"] = index
                if run_manager:
                    if chunk.message.id is None:
                        chunk.message.id = run_id
                    await run_manager.on_llm_new_token(
                        cast("str", chunk.message.content), chunk=chunk
                    )
                chunks.append(chunk)
                yielded = True

            # Yield a final empty chunk with chunk_position="last" if not yet yielded
            # 中文: 如果尚未产生，则产生一个带有 chunk_position="last" 的最终空块
            if (
                yielded
                and isinstance(chunk.message, AIMessageChunk)
                and not chunk.message.chunk_position
            ):
                empty_content: str | list = (
                    "" if isinstance(chunk.message.content, str) else []
                )
                chunk = ChatGenerationChunk(
                    message=AIMessageChunk(
                        content=empty_content, chunk_position="last", id=run_id
                    )
                )
                if run_manager:
                    await run_manager.on_llm_new_token("", chunk=chunk)
                chunks.append(chunk)
            result = generate_from_stream(iter(chunks))
        elif inspect.signature(self._agenerate).parameters.get("run_manager"):
            result = await self._agenerate(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
        else:
            result = await self._agenerate(messages, stop=stop, **kwargs)

        if self.output_version == "v1":
            # Overwrite .content with .content_blocks
            # 中文: 用 .content_blocks 覆盖 .content
            for generation in result.generations:
                generation.message = _update_message_content_to_blocks(
                    generation.message, "v1"
                )

        # Add response metadata to each generation
        # 中文: 为每一代添加响应元数据
        for idx, generation in enumerate(result.generations):
            if run_manager and generation.message.id is None:
                generation.message.id = f"{LC_ID_PREFIX}-{run_manager.run_id}-{idx}"
            generation.message.response_metadata = _gen_info_and_msg_metadata(
                generation
            )
        if len(result.generations) == 1 and result.llm_output is not None:
            result.generations[0].message.response_metadata = {
                **result.llm_output,
                **result.generations[0].message.response_metadata,
            }
        if check_cache and llm_cache:
            await llm_cache.aupdate(prompt, llm_string, result.generations)
        return result

    @abstractmethod
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate the result.

        Args:
            messages: The messages to generate from.
            stop: Optional list of stop words to use when generating.
            run_manager: Optional callback manager to use for this call.
            **kwargs: Additional keyword arguments to pass to the model.

        Returns:
            The chat result.
        

        中文翻译:
        生成结果。
        参数：
            messages：要生成的消息。
            stop：生成时使用的可选停用词列表。
            run_manager：用于此调用的可选回调管理器。
            **kwargs：传递给模型的附加关键字参数。
        返回：
            聊天结果。"""

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate the result.

        Args:
            messages: The messages to generate from.
            stop: Optional list of stop words to use when generating.
            run_manager: Optional callback manager to use for this call.
            **kwargs: Additional keyword arguments to pass to the model.

        Returns:
            The chat result.
        

        中文翻译:
        生成结果。
        参数：
            messages：要生成的消息。
            stop：生成时使用的可选停用词列表。
            run_manager：用于此调用的可选回调管理器。
            **kwargs：传递给模型的附加关键字参数。
        返回：
            聊天结果。"""
        return await run_in_executor(
            None,
            self._generate,
            messages,
            stop,
            run_manager.get_sync() if run_manager else None,
            **kwargs,
        )

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream the output of the model.

        Args:
            messages: The messages to generate from.
            stop: Optional list of stop words to use when generating.
            run_manager: Optional callback manager to use for this call.
            **kwargs: Additional keyword arguments to pass to the model.

        Yields:
            The chat generation chunks.
        

        中文翻译:
        流式传输模型的输出。
        参数：
            messages：要生成的消息。
            stop：生成时使用的可选停用词列表。
            run_manager：用于此调用的可选回调管理器。
            **kwargs：传递给模型的附加关键字参数。
        产量：
            聊天生成块。"""
        raise NotImplementedError

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Stream the output of the model.

        Args:
            messages: The messages to generate from.
            stop: Optional list of stop words to use when generating.
            run_manager: Optional callback manager to use for this call.
            **kwargs: Additional keyword arguments to pass to the model.

        Yields:
            The chat generation chunks.
        

        中文翻译:
        流式传输模型的输出。
        参数：
            messages：要生成的消息。
            stop：生成时使用的可选停用词列表。
            run_manager：用于此调用的可选回调管理器。
            **kwargs：传递给模型的附加关键字参数。
        产量：
            聊天生成块。"""
        iterator = await run_in_executor(
            None,
            self._stream,
            messages,
            stop,
            run_manager.get_sync() if run_manager else None,
            **kwargs,
        )
        done = object()
        while True:
            item = await run_in_executor(
                None,
                next,
                iterator,
                done,
            )
            if item is done:
                break
            yield item  # type: ignore[misc]

    async def _call_async(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> BaseMessage:
        result = await self.agenerate(
            [messages], stop=stop, callbacks=callbacks, **kwargs
        )
        generation = result.generations[0][0]
        if isinstance(generation, ChatGeneration):
            return generation.message
        msg = "Unexpected generation type"
        raise ValueError(msg)

    @property
    @abstractmethod
    def _llm_type(self) -> str:
        """Return type of chat model.

        中文翻译:
        聊天模型的返回类型。"""

    @override
    def dict(self, **kwargs: Any) -> dict:
        """Return a dictionary of the LLM.

        中文翻译:
        返回 LLM 的字典。"""
        starter_dict = dict(self._identifying_params)
        starter_dict["_type"] = self._llm_type
        return starter_dict

    def bind_tools(
        self,
        tools: Sequence[
            typing.Dict[str, Any] | type | Callable | BaseTool  # noqa: UP006
        ],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """Bind tools to the model.

        Args:
            tools: Sequence of tools to bind to the model.
            tool_choice: The tool to use. If "any" then any tool can be used.

        Returns:
            A Runnable that returns a message.

        

        中文翻译:
        将工具绑定到模型。
        参数：
            工具：绑定到模型的工具序列。
            tool_choice：要使用的工具。如果“任何”，则可以使用任何工具。
        返回：
            返回消息的 Runnable。"""
        raise NotImplementedError

    def with_structured_output(
        self,
        schema: typing.Dict | type,  # noqa: UP006
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, typing.Dict | BaseModel]:  # noqa: UP006
        """Model wrapper that returns outputs formatted to match the given schema.

        Args:
            schema: The output schema. Can be passed in as:

                - An OpenAI function/tool schema,
                - A JSON Schema,
                - A `TypedDict` class,
                - Or a Pydantic class.

                If `schema` is a Pydantic class then the model output will be a
                Pydantic instance of that class, and the model-generated fields will be
                validated by the Pydantic class. Otherwise the model output will be a
                dict and will not be validated.

                See `langchain_core.utils.function_calling.convert_to_openai_tool` for
                more on how to properly specify types and descriptions of schema fields
                when specifying a Pydantic or `TypedDict` class.

            include_raw:
                If `False` then only the parsed structured output is returned.

                If an error occurs during model output parsing it will be raised.

                If `True` then both the raw model response (a `BaseMessage`) and the
                parsed model response will be returned.

                If an error occurs during output parsing it will be caught and returned
                as well.

                The final output is always a `dict` with keys `'raw'`, `'parsed'`, and
                `'parsing_error'`.

        Raises:
            ValueError: If there are any unsupported `kwargs`.
            NotImplementedError: If the model does not implement
                `with_structured_output()`.

        Returns:
            A `Runnable` that takes same inputs as a
                `langchain_core.language_models.chat.BaseChatModel`. If `include_raw` is
                `False` and `schema` is a Pydantic class, `Runnable` outputs an instance
                of `schema` (i.e., a Pydantic object). Otherwise, if `include_raw` is
                `False` then `Runnable` outputs a `dict`.

                If `include_raw` is `True`, then `Runnable` outputs a `dict` with keys:

                - `'raw'`: `BaseMessage`
                - `'parsed'`: `None` if there was a parsing error, otherwise the type
                    depends on the `schema` as described above.
                - `'parsing_error'`: `BaseException | None`

        ???+ example "Pydantic schema (`include_raw=False`)"

            ```python
            from pydantic import BaseModel


            class AnswerWithJustification(BaseModel):
                '''An answer to the user question along with justification for the answer.'''

                answer: str
                justification: str


            model = ChatModel(model="model-name", temperature=0)
            structured_model = model.with_structured_output(AnswerWithJustification)

            structured_model.invoke(
                "What weighs more a pound of bricks or a pound of feathers"
            )

            # -> AnswerWithJustification(
            # 中文: -> 有理由的回答(
            #     answer='They weigh the same',
            #     中文: 答案='它们的重量相同',
            #     justification='Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ.'
            #     中文: justification='一磅砖块和一磅羽毛都重一磅。重量相同，但物体的体积或密度可能不同。
            # )
            ```

        ??? example "Pydantic schema (`include_raw=True`)"

            ```python
            from pydantic import BaseModel


            class AnswerWithJustification(BaseModel):
                '''An answer to the user question along with justification for the answer.'''

                answer: str
                justification: str


            model = ChatModel(model="model-name", temperature=0)
            structured_model = model.with_structured_output(
                AnswerWithJustification, include_raw=True
            )

            structured_model.invoke(
                "What weighs more a pound of bricks or a pound of feathers"
            )
            # -> {
            # 中文: None
            #     'raw': AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_Ao02pnFYXD6GN1yzc0uXPsvF', 'function': {'arguments': '{"answer":"They weigh the same.","justification":"Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ."}', 'name': 'AnswerWithJustification'}, 'type': 'function'}]}),
            #     中文: 'raw': AIMessage(content='', extra_kwargs={'tool_calls': [{'id': 'call_Ao02pnFYXD6GN1yzc0uXPsvF', 'function': {'arguments': '{"answer":"它们的重量相同。","justification":"一磅砖块和一磅羽毛都重一磅。重量为相同，但物体的体积或密度可能不同。"}', 'name': 'AnswerWithJustification'}, 'type': 'function'}]}),
            #     'parsed': AnswerWithJustification(answer='They weigh the same.', justification='Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ.'),
            #     中文: 'parsed': AnswerWithJustification(answer='它们的重量相同。', justification='一磅砖块和一磅羽毛都重一磅。重量相同，但物体的体积或密度可能不同。'),
            #     'parsing_error': None
            #     中文: “解析错误”：无
            # }
            ```

        ??? example "Dictionary schema (`include_raw=False`)"

            ```python
            from pydantic import BaseModel
            from langchain_core.utils.function_calling import convert_to_openai_tool


            class AnswerWithJustification(BaseModel):
                '''An answer to the user question along with justification for the answer.'''

                answer: str
                justification: str


            dict_schema = convert_to_openai_tool(AnswerWithJustification)
            model = ChatModel(model="model-name", temperature=0)
            structured_model = model.with_structured_output(dict_schema)

            structured_model.invoke(
                "What weighs more a pound of bricks or a pound of feathers"
            )
            # -> {
            # 中文: None
            #     'answer': 'They weigh the same',
            #     中文: 'answer': '它们的重量相同',
            #     'justification': 'Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume and density of the two substances differ.'
            #     中文: 'justification': '一磅砖块和一磅羽毛都重一磅。重量相同，但两种物质的体积和密度不同。
            # }
            ```

        !!! warning "Behavior changed in `langchain-core` 0.2.26"

            Added support for `TypedDict` class.

        

        中文翻译:
        返回格式化以匹配给定模式的输出的模型包装器。
        参数：
            schema：输出模式。可以传入为：
                - OpenAI 函数/工具模式，
                - JSON 模式，
                - 一个“TypedDict”类，
                - 或者 Pydantic 类。
                如果“schema”是 Pydantic 类，那么模型输出将是
                该类的 Pydantic 实例，模型生成的字段将是
                由 Pydantic 类验证。否则模型输出将是
                dict 并且不会被验证。
                请参阅“langchain_core.utils.function_calling.convert_to_openai_tool”
                有关如何正确指定模式字段的类型和描述的更多信息
                当指定 Pydantic 或 `TypedDict` 类时。
            包括原始：
                如果“False”，则仅返回解析的结构化输出。
                如果模型输出解析期间发生错误，则会引发错误。
                如果“True”，则原始模型响应（“BaseMessage”）和
                将返回解析后的模型响应。
                如果在输出解析期间发生错误，它将被捕获并返回
                以及。
                最终输出始终是一个带有键“raw”、“parsed”和
                `'解析错误'`。
        加薪：
            ValueError：如果有任何不支持的`kwargs`。
            NotImplementedError：如果模型未实现
                `with_structed_output()`。
        返回：
            一个“Runnable”，其输入与
                `langchain_core.language_models.chat.BaseChatModel`。如果 `include_raw` 是
                `False` 和 `schema` 是一个 Pydantic 类，`Runnable` 输出一个实例
                “schema”（即 Pydantic 对象）。否则，如果 `include_raw` 是
                `False` 然后 `Runnable` 输出一个 `dict`。
                如果“include_raw”为“True”，则“Runnable”输出一个带有键的“dict”：
                - `'原始'`：`BaseMessage`
                - `'parsed'`：如果出现解析错误则为`None`，否则为类型
                    取决于上面描述的“模式”。
                - `'parsing_error'`：`BaseException |无`
        ???+ 示例“Pydantic 模式 (`include_raw=False`)”
            ````蟒蛇
            从 pydantic 导入 BaseModel
            类 AnswerWithJustification(BaseModel)：
                '''对用户问题的回答以及回答的理由。'''
                答案：str
                理由：str
            模型 = ChatModel(模型=“模型名称”，温度=0)
            结构化模型 = model.with_structed_output(AnswerWithJustification)
            结构化模型.调用(
                “一磅砖头和一磅羽毛哪个更重”
            ）
            # -> 回答有理由(
            #answer='它们的重量相同',
            # justification='一磅砖块和一磅羽毛的重量都是一磅。重量相同，但物体的体积或密度可能不同。
            ＃）
            ````
        ???例如“Pydantic 模式 (`include_raw=True`)”
            ````蟒蛇
            从 pydantic 导入 BaseModel
            类 AnswerWithJustification(BaseModel)：
                '''对用户问题的回答以及回答的理由。'''
                答案：str
                理由：str
            模型 = ChatModel(模型=“模型名称”，温度=0)
            结构化模型 = model.with_structed_output(
                AnswerWithJustification，include_raw=True
            ）
            结构化模型.调用(
                “一磅砖头和一磅羽毛哪个更重”
            ）
            ＃ - > {
            # 'raw': AIMessage(content='', extra_kwargs={'tool_calls': [{'id': 'call_Ao02pnFYXD6GN1yzc0uXPsvF', 'function': {'arguments': '{"answer":"它们的重量相同。","justification":"一磅砖块和一磅羽毛都重一磅。重量相同，但物体的体积或密度可能不同。"}', 'name': 'AnswerWithJustification'}, 'type': 'function'}]}),# 'parsed': AnswerWithJustification(answer='它们的重量相同。', justification='一磅砖块和一磅羽毛都重一磅。重量相同，但物体的体积或密度可能不同。'),
            # 'parsing_error': 无
            # }
            ````
        ???示例“字典模式 (`include_raw=False`)”
            ````蟒蛇
            从 pydantic 导入 BaseModel
            从langchain_core.utils.function_calling导入convert_to_openai_tool
            类 AnswerWithJustification(BaseModel)：
                '''对用户问题的回答以及回答的理由。'''
                答案：str
                理由：str
            dict_schema = Convert_to_openai_tool(AnswerWithJustification)
            模型 = ChatModel(模型=“模型名称”，温度=0)
            结构化模型 = model.with_structed_output(dict_schema)
            结构化模型.调用(
                “一磅砖头和一磅羽毛哪个更重”
            ）
            ＃ - > {
            # 'answer': '它们的重量相同',
            # 'justification': '一磅砖块和一磅羽毛都重一磅。重量相同，但两种物质的体积和密度不同。
            # }
            ````
        !!!警告“‘langchain-core’ 0.2.26 中的行为已更改”
            添加了对“TypedDict”类的支持。"""  # noqa: E501
        _ = kwargs.pop("method", None)
        _ = kwargs.pop("strict", None)
        if kwargs:
            msg = f"Received unsupported arguments {kwargs}"
            raise ValueError(msg)

        if type(self).bind_tools is BaseChatModel.bind_tools:
            msg = "with_structured_output is not implemented for this model."
            raise NotImplementedError(msg)

        llm = self.bind_tools(
            [schema],
            tool_choice="any",
            ls_structured_output_format={
                "kwargs": {"method": "function_calling"},
                "schema": schema,
            },
        )
        if isinstance(schema, type) and is_basemodel_subclass(schema):
            output_parser: OutputParserLike = PydanticToolsParser(
                tools=[cast("TypeBaseModel", schema)], first_tool_only=True
            )
        else:
            key_name = convert_to_openai_tool(schema)["function"]["name"]
            output_parser = JsonOutputKeyToolsParser(
                key_name=key_name, first_tool_only=True
            )
        if include_raw:
            parser_assign = RunnablePassthrough.assign(
                parsed=itemgetter("raw") | output_parser, parsing_error=lambda _: None
            )
            parser_none = RunnablePassthrough.assign(parsed=lambda _: None)
            parser_with_fallback = parser_assign.with_fallbacks(
                [parser_none], exception_key="parsing_error"
            )
            return RunnableMap(raw=llm) | parser_with_fallback
        return llm | output_parser


class SimpleChatModel(BaseChatModel):
    """Simplified implementation for a chat model to inherit from.

    !!! note
        This implementation is primarily here for backwards compatibility. For new
        implementations, please use `BaseChatModel` directly.

    

    中文翻译:
    继承聊天模型的简化实现。
    !!!注释
        此实现主要是为了向后兼容。对于新的
        实现，请直接使用`BaseChatModel`。"""

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        output_str = self._call(messages, stop=stop, run_manager=run_manager, **kwargs)
        message = AIMessage(content=output_str)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    @abstractmethod
    def _call(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        """Simpler interface.

        中文翻译:
        更简单的界面。"""

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return await run_in_executor(
            None,
            self._generate,
            messages,
            stop=stop,
            run_manager=run_manager.get_sync() if run_manager else None,
            **kwargs,
        )


def _gen_info_and_msg_metadata(
    generation: ChatGeneration | ChatGenerationChunk,
) -> dict:
    return {
        **(generation.generation_info or {}),
        **generation.message.response_metadata,
    }


_MAX_CLEANUP_DEPTH = 100


def _cleanup_llm_representation(serialized: Any, depth: int) -> None:
    """Remove non-serializable objects from a serialized object.

    中文翻译:
    从序列化对象中删除不可序列化对象。"""
    if depth > _MAX_CLEANUP_DEPTH:  # Don't cooperate for pathological cases
        return

    if not isinstance(serialized, dict):
        return

    if (
        "type" in serialized
        and serialized["type"] == "not_implemented"
        and "repr" in serialized
    ):
        del serialized["repr"]

    if "graph" in serialized:
        del serialized["graph"]

    if "kwargs" in serialized:
        kwargs = serialized["kwargs"]

        for value in kwargs.values():
            _cleanup_llm_representation(value, depth + 1)
