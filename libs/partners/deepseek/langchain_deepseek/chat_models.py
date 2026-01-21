"""DeepSeek chat models.

中文翻译:
DeepSeek 聊天模型。
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Sequence
from json import JSONDecodeError
from typing import Any, Literal, TypeAlias, cast

import openai
from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import (
    LangSmithParams,
    LanguageModelInput,
    ModelProfile,
    ModelProfileRegistry,
)
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils import from_env, secret_from_env
from langchain_openai.chat_models.base import BaseChatOpenAI
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

from langchain_deepseek.data._profiles import _PROFILES

DEFAULT_API_BASE = "https://api.deepseek.com/v1"
DEFAULT_BETA_API_BASE = "https://api.deepseek.com/beta"

_DictOrPydanticClass: TypeAlias = dict[str, Any] | type[BaseModel]
_DictOrPydantic: TypeAlias = dict[str, Any] | BaseModel


_MODEL_PROFILES = cast("ModelProfileRegistry", _PROFILES)


def _get_default_model_profile(model_name: str) -> ModelProfile:
    default = _MODEL_PROFILES.get(model_name) or {}
    return default.copy()


class ChatDeepSeek(BaseChatOpenAI):
    """DeepSeek chat model integration to access models hosted in DeepSeek's API.

    Setup:
        Install `langchain-deepseek` and set environment variable `DEEPSEEK_API_KEY`.

        ```bash
        pip install -U langchain-deepseek
        export DEEPSEEK_API_KEY="your-api-key"
        ```

    Key init args — completion params:
        model:
            Name of DeepSeek model to use, e.g. `'deepseek-chat'`.
        temperature:
            Sampling temperature.
        max_tokens:
            Max number of tokens to generate.

    Key init args — client params:
        timeout:
            Timeout for requests.
        max_retries:
            Max number of retries.
        api_key:
            DeepSeek API key. If not passed in will be read from env var `DEEPSEEK_API_KEY`.

    See full list of supported init args and their descriptions in the params section.

    Instantiate:
        ```python
        from langchain_deepseek import ChatDeepSeek

        model = ChatDeepSeek(
            model="...",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            # api_key="...",
            # other params...
        )
        ```

    Invoke:
        ```python
        messages = [
            ("system", "You are a helpful translator. Translate the user sentence to French."),
            ("human", "I love programming."),
        ]
        model.invoke(messages)
        ```

    Stream:
        ```python
        for chunk in model.stream(messages):
            print(chunk.text, end="")
        ```
        ```python
        stream = model.stream(messages)
        full = next(stream)
        for chunk in stream:
            full += chunk
        full
        ```

    Async:
        ```python
        await model.ainvoke(messages)

        # stream:
        # 中文: 溪流：
        # async for chunk in (await model.astream(messages))
        # 中文: async for chunk in (await model.astream(messages))

        # batch:
        # 中文: 批：
        # await model.abatch([messages])
        # 中文: 等待 model.abatch([消息])
        ```

    Tool calling:
        ```python
        from pydantic import BaseModel, Field


        class GetWeather(BaseModel):
            '''Get the current weather in a given location'''

            location: str = Field(..., description="The city and state, e.g. San Francisco, CA")


        class GetPopulation(BaseModel):
            '''Get the current population in a given location'''

            location: str = Field(..., description="The city and state, e.g. San Francisco, CA")


        model_with_tools = model.bind_tools([GetWeather, GetPopulation])
        ai_msg = model_with_tools.invoke("Which city is hotter today and which is bigger: LA or NY?")
        ai_msg.tool_calls
        ```

        See `ChatDeepSeek.bind_tools()` method for more.

    Structured output:
        ```python
        from typing import Optional

        from pydantic import BaseModel, Field


        class Joke(BaseModel):
            '''Joke to tell user.'''

            setup: str = Field(description="The setup of the joke")
            punchline: str = Field(description="The punchline to the joke")
            rating: int | None = Field(description="How funny the joke is, from 1 to 10")


        structured_model = model.with_structured_output(Joke)
        structured_model.invoke("Tell me a joke about cats")
        ```

        See `ChatDeepSeek.with_structured_output()` for more.

    Token usage:
        ```python
        ai_msg = model.invoke(messages)
        ai_msg.usage_metadata
        ```
        ```python
        {"input_tokens": 28, "output_tokens": 5, "total_tokens": 33}
        ```

    Response metadata:
        ```python
        ai_msg = model.invoke(messages)
        ai_msg.response_metadata
        ```


    中文翻译:
    DeepSeek 聊天模型集成可访问 DeepSeek API 中托管的模型。
    设置：
        安装“langchain-deepseek”并设置环境变量“DEEPSEEK_API_KEY”。
        ````bash
        pip install -U langchain-deepseek
        导出 DEEPSEEK_API_KEY="your-api-key"
        ````
    关键初始化参数 - 完成参数：
        型号：
            要使用的 DeepSeek 模型的名称，例如“深度搜索聊天”。
        温度：
            取样温度。
        最大令牌数：
            生成的最大令牌数。
    关键初始化参数 — 客户端参数：
        超时：
            请求超时。
        最大重试次数：
            最大重试次数。
        api_key:
            DeepSeek API 密钥。如果没有传入，将从环境变量`DEEPSEEK_API_KEY`中读取。
    请参阅参数部分中支持的 init args 及其描述的完整列表。
    实例化：
        ````蟒蛇
        从 langchain_deepseek 导入 ChatDeepSeek
        模型 = ChatDeepSeek(
            型号=“...”，
            温度=0，
            max_tokens=无,
            超时=无，
            最大重试次数=2，
            # api_key="...",
            # 其他参数...
        ）
        ````
    调用：
        ````蟒蛇
        消息 = [
            ("system", "你是一位很有帮助的翻译。将用户句子翻译成法语。"),
            （“人类”，“我喜欢编程。”），
        ]
        模型.调用（消息）
        ````
    流：
        ````蟒蛇
        对于 model.stream(messages) 中的块：
            打印（块.文本，结束=“”）
        ````
        ````蟒蛇
        流 = model.stream(消息)
        完整 = 下一个（流）
        对于流中的块：
            完整+=块
        满
        ````
    异步：
        ````蟒蛇
        等待 model.ainvoke(消息)
        # 流：
        # async for chunk in (await model.astream(messages))
        # 中文: async for chunk in (await model.astream(messages))
        # 批次：
        # 等待 model.abatch([messages])
        ````
    工具调用：
        ````蟒蛇
        从 pydantic 导入 BaseModel、Field
        类 GetWeather(BaseModel):
            '''获取给定位置的当前天气'''
            位置：str = Field（...，description =“城市和州，例如加利福尼亚州旧金山”）
        类 GetPopulation(BaseModel):
            '''获取给定位置的当前人口'''
            位置：str = Field（...，description =“城市和州，例如加利福尼亚州旧金山”）
        model_with_tools = model.bind_tools([GetWeather, GetPopulation])
        ai_msg = model_with_tools.invoke("今天哪个城市更热，哪个城市更大：洛杉矶还是纽约？")
        ai_msg.tool_calls
        ````
        有关更多信息，请参阅“ChatDeepSeek.bind_tools()”方法。
    结构化输出：
        ````蟒蛇
        从输入 import 可选
        从 pydantic 导入 BaseModel、Field
        笑话类（基础模型）：
            '''告诉用户的笑话。'''
            setup: str = Field(description="笑话的设置")
            笑点：str = Field（描述=“笑话的笑点”）
            评级：int | None = Field(description="这个笑话有多好笑，从 1 到 10")
        Structured_Model = model.with_Structured_Output（笑话）
        Structured_model.invoke("给我讲一个关于猫的笑话")
        ````
        有关更多信息，请参阅“ChatDeepSeek.with_structed_output()”。
    代币使用：
        ````蟒蛇
        ai_msg = model.invoke(消息)
        ai_msg.usage_metadata
        ````
        ````蟒蛇
        {“input_tokens”：28，“output_tokens”：5，“total_tokens”：33}
        ````
    响应元数据：
        ````蟒蛇
        ai_msg = model.invoke(消息)
        ai_msg.response_metadata
        ````
    """

    model_name: str = Field(alias="model")
    """The name of the model

    中文翻译:
    型号名称
    """
    api_key: SecretStr | None = Field(
        default_factory=secret_from_env("DEEPSEEK_API_KEY", default=None),
    )
    """DeepSeek API key

    中文翻译:
    DeepSeek API 密钥
    """
    api_base: str = Field(
        default_factory=from_env("DEEPSEEK_API_BASE", default=DEFAULT_API_BASE),
    )
    """DeepSeek API base URL

    中文翻译:
    DeepSeek API 基本 URL
    """

    model_config = ConfigDict(populate_by_name=True)

    @property
    def _llm_type(self) -> str:
        """Return type of chat model.

        中文翻译:
        聊天模型的返回类型。
        """
        return "chat-deepseek"

    @property
    def lc_secrets(self) -> dict[str, str]:
        """A map of constructor argument names to secret ids.

        中文翻译:
        构造函数参数名称到秘密 ID 的映射。
        """
        return {"api_key": "DEEPSEEK_API_KEY"}

    def _get_ls_params(
        self,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> LangSmithParams:
        ls_params = super()._get_ls_params(stop=stop, **kwargs)
        ls_params["ls_provider"] = "deepseek"
        return ls_params

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate necessary environment vars and client params.

        中文翻译:
        验证必要的环境变量和客户端参数。
        """
        if self.api_base == DEFAULT_API_BASE and not (
            self.api_key and self.api_key.get_secret_value()
        ):
            msg = "If using default api base, DEEPSEEK_API_KEY must be set."
            raise ValueError(msg)
        client_params: dict = {
            k: v
            for k, v in {
                "api_key": self.api_key.get_secret_value() if self.api_key else None,
                "base_url": self.api_base,
                "timeout": self.request_timeout,
                "max_retries": self.max_retries,
                "default_headers": self.default_headers,
                "default_query": self.default_query,
            }.items()
            if v is not None
        }

        if not (self.client or None):
            sync_specific: dict = {"http_client": self.http_client}
            self.root_client = openai.OpenAI(**client_params, **sync_specific)
            self.client = self.root_client.chat.completions
        if not (self.async_client or None):
            async_specific: dict = {"http_client": self.http_async_client}
            self.root_async_client = openai.AsyncOpenAI(
                **client_params,
                **async_specific,
            )
            self.async_client = self.root_async_client.chat.completions
        return self

    @model_validator(mode="after")
    def _set_model_profile(self) -> Self:
        """Set model profile if not overridden.

        中文翻译:
        如果未覆盖，则设置模型配置文件。
        """
        if self.profile is None:
            self.profile = _get_default_model_profile(self.model_name)
        return self

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        for message in payload["messages"]:
            if message["role"] == "tool" and isinstance(message["content"], list):
                message["content"] = json.dumps(message["content"])
            elif message["role"] == "assistant" and isinstance(
                message["content"], list
            ):
                # DeepSeek API expects assistant content to be a string, not a list.
                # 中文: DeepSeek API 期望助手内容是字符串，而不是列表。
                # Extract text blocks and join them, or use empty string if none exist.
                # 中文: 提取文本块并连接它们，或者如果不存在则使用空字符串。
                text_parts = [
                    block.get("text", "")
                    for block in message["content"]
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                message["content"] = "".join(text_parts) if text_parts else ""
        return payload

    def _create_chat_result(
        self,
        response: dict | openai.BaseModel,
        generation_info: dict | None = None,
    ) -> ChatResult:
        rtn = super()._create_chat_result(response, generation_info)

        if not isinstance(response, openai.BaseModel):
            return rtn

        for generation in rtn.generations:
            if generation.message.response_metadata is None:
                generation.message.response_metadata = {}
            generation.message.response_metadata["model_provider"] = "deepseek"

        choices = getattr(response, "choices", None)
        if choices and hasattr(choices[0].message, "reasoning_content"):
            rtn.generations[0].message.additional_kwargs["reasoning_content"] = choices[
                0
            ].message.reasoning_content
        # Handle use via OpenRouter
        # 中文: 通过 OpenRouter 处理使用
        elif choices and hasattr(choices[0].message, "model_extra"):
            model_extra = choices[0].message.model_extra
            if isinstance(model_extra, dict) and (
                reasoning := model_extra.get("reasoning")
            ):
                rtn.generations[0].message.additional_kwargs["reasoning_content"] = (
                    reasoning
                )

        return rtn

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: type,
        base_generation_info: dict | None,
    ) -> ChatGenerationChunk | None:
        generation_chunk = super()._convert_chunk_to_generation_chunk(
            chunk,
            default_chunk_class,
            base_generation_info,
        )
        if (choices := chunk.get("choices")) and generation_chunk:
            top = choices[0]
            if isinstance(generation_chunk.message, AIMessageChunk):
                generation_chunk.message.response_metadata = {
                    **generation_chunk.message.response_metadata,
                    "model_provider": "deepseek",
                }
                if (
                    reasoning_content := top.get("delta", {}).get("reasoning_content")
                ) is not None:
                    generation_chunk.message.additional_kwargs["reasoning_content"] = (
                        reasoning_content
                    )
                # Handle use via OpenRouter
                # 中文: 通过 OpenRouter 处理使用
                elif (reasoning := top.get("delta", {}).get("reasoning")) is not None:
                    generation_chunk.message.additional_kwargs["reasoning_content"] = (
                        reasoning
                    )

        return generation_chunk

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        try:
            yield from super()._stream(
                messages,
                stop=stop,
                run_manager=run_manager,
                **kwargs,
            )
        except JSONDecodeError as e:
            msg = (
                "DeepSeek API returned an invalid response. "
                "Please check the API status and try again."
            )
            raise JSONDecodeError(
                msg,
                e.doc,
                e.pos,
            ) from e

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        try:
            return super()._generate(
                messages,
                stop=stop,
                run_manager=run_manager,
                **kwargs,
            )
        except JSONDecodeError as e:
            msg = (
                "DeepSeek API returned an invalid response. "
                "Please check the API status and try again."
            )
            raise JSONDecodeError(
                msg,
                e.doc,
                e.pos,
            ) from e

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        *,
        tool_choice: dict | str | bool | None = None,
        strict: bool | None = None,
        parallel_tool_calls: bool | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """Bind tool-like objects to this chat model.

        Overrides parent to use beta endpoint when `strict=True`.

        Args:
            tools: A list of tool definitions to bind to this chat model.
            tool_choice: Which tool to require the model to call.
            strict: If True, uses beta API for strict schema validation.
            parallel_tool_calls: Set to `False` to disable parallel tool use.
            **kwargs: Additional parameters passed to parent `bind_tools`.

        Returns:
            A Runnable that takes same inputs as a chat model.


        中文翻译:
        将类似工具的对象绑定到此聊天模型。
        当 `strict=True` 时覆盖父级以使用 beta 端点。
        参数：
            工具：绑定到此聊天模型的工具定义列表。
            tool_choice：模型需要调用哪个工具。
            strict：如果为 True，则使用 beta API 进行严格的模式验证。
            parallel_tool_calls：设置为“False”以禁用并行工具使用。
            **kwargs：传递给父级“bind_tools”的附加参数。
        返回：
            与聊天模型采用相同输入的 Runnable。
        """
        # If strict mode is enabled and using default API base, switch to beta endpoint
        # 中文: 如果启用了严格模式并使用默认 API 库，请切换到 beta 端点
        if strict is True and self.api_base == DEFAULT_API_BASE:
            # Create a new instance with beta endpoint
            # 中文: 使用 beta 端点创建新实例
            beta_model = self.model_copy(update={"api_base": DEFAULT_BETA_API_BASE})
            return beta_model.bind_tools(
                tools,
                tool_choice=tool_choice,
                strict=strict,
                parallel_tool_calls=parallel_tool_calls,
                **kwargs,
            )

        # Otherwise use parent implementation
        # 中文: 否则使用父实现
        return super().bind_tools(
            tools,
            tool_choice=tool_choice,
            strict=strict,
            parallel_tool_calls=parallel_tool_calls,
            **kwargs,
        )

    def with_structured_output(
        self,
        schema: _DictOrPydanticClass | None = None,
        *,
        method: Literal[
            "function_calling",
            "json_mode",
            "json_schema",
        ] = "function_calling",
        include_raw: bool = False,
        strict: bool | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, _DictOrPydantic]:
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

            method: The method for steering model generation, one of:

                - `'function_calling'`:
                    Uses DeepSeek's [tool-calling features](https://api-docs.deepseek.com/guides/function_calling).
                - `'json_mode'`:
                    Uses DeepSeek's [JSON mode feature](https://api-docs.deepseek.com/guides/json_mode).

            include_raw:
                If `False` then only the parsed structured output is returned.

                If an error occurs during model output parsing it will be raised.

                If `True` then both the raw model response (a `BaseMessage`) and the
                parsed model response will be returned.

                If an error occurs during output parsing it will be caught and returned
                as well.

                The final output is always a `dict` with keys `'raw'`, `'parsed'`, and
                `'parsing_error'`.

            strict:
                Whether to enable strict schema adherence when generating the function
                call. When set to `True`, DeepSeek will use the beta API endpoint
                (`https://api.deepseek.com/beta`) for strict schema validation.
                This ensures model outputs exactly match the defined schema.

                !!! note

                    DeepSeek's strict mode requires all object properties to be marked
                    as required in the schema.

            kwargs: Additional keyword args aren't supported.

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
            method：转向模型生成方法，其中之一：
                - `'函数调用'`：
                    使用 DeepSeek 的[工具调用功能](https://api-docs.deepseek.com/guides/function_calling)。
                - `'json_mode'`：
                    使用 DeepSeek 的 [JSON 模式功能](https://api-docs.deepseek.com/guides/json_mode)。
            包括原始：
                如果“False”，则仅返回解析的结构化输出。
                如果模型输出解析期间发生错误，则会引发错误。
                如果“True”，则原始模型响应（“BaseMessage”）和
                将返回解析后的模型响应。
                如果在输出解析期间发生错误，它将被捕获并返回
                以及。
                最终输出始终是一个带有键“raw”、“parsed”和
                `'解析错误'`。
            严格：
                生成函数时是否启用严格的架构遵循
                打电话。当设置为“True”时，DeepSeek 将使用 beta API 端点
                (`https://api.deepseek.com/beta`) 用于严格的模式验证。
                这可确保模型输出与定义的模式完全匹配。
                !!!注释
                    DeepSeek的严格模式要求所有对象属性都被标记
                    根据架构中的要求。
            kwargs：不支持其他关键字参数。
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
        """
        # Some applications require that incompatible parameters (e.g., unsupported
        # 中文: 某些应用程序需要不兼容的参数（例如，不支持的参数）
        # methods) be handled.
        # 中文: 方法）进行处理。
        if method == "json_schema":
            method = "function_calling"

        # If strict mode is enabled and using default API base, switch to beta endpoint
        # 中文: 如果启用了严格模式并使用默认 API 库，请切换到 beta 端点
        if strict is True and self.api_base == DEFAULT_API_BASE:
            # Create a new instance with beta endpoint
            # 中文: 使用 beta 端点创建新实例
            beta_model = self.model_copy(update={"api_base": DEFAULT_BETA_API_BASE})
            return beta_model.with_structured_output(
                schema,
                method=method,
                include_raw=include_raw,
                strict=strict,
                **kwargs,
            )

        return super().with_structured_output(
            schema,
            method=method,
            include_raw=include_raw,
            strict=strict,
            **kwargs,
        )
