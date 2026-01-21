"""Agent 工厂模块，用于创建具有中间件支持的 Agent。.

本模块提供 `create_agent` 函数，用于构建支持工具调用和中间件的 Agent 图。

核心功能:
---------
**create_agent**: 创建一个循环调用工具直到满足停止条件的 Agent 图

工作原理:
---------
1. Agent 节点使用消息列表调用语言模型
2. 如果 AIMessage 包含 tool_calls，则调用工具节点
3. 工具节点执行工具并将结果添加为 ToolMessage
4. 重复此过程直到没有更多 tool_calls
5. 返回完整的消息列表

支持的功能:
-----------
- 多种模型（通过字符串或 BaseChatModel 实例）
- 工具调用（函数、BaseTool 或字典格式）
- 中间件链（执行控制、重试、回退等）
- 结构化输出（ToolStrategy 或 ProviderStrategy）
- 持久化（checkpointer 和 store）
- 调试模式

使用示例:
---------
>>> from langchain.agents import create_agent
>>>
>>> def check_weather(location: str) -> str:
...     '''返回指定位置的天气预报。'''
...     return f"{location}的天气是晴天"
>>>
>>> graph = create_agent(
...     model="anthropic:claude-sonnet-4-5-20250929",
...     tools=[check_weather],
...     system_prompt="你是一个有用的助手",
... )
>>>
>>> inputs = {"messages": [{"role": "user", "content": "北京天气怎么样？"}]}
>>> for chunk in graph.stream(inputs, stream_mode="updates"):
...     print(chunk)
"""  # noqa: RUF002

from __future__ import annotations

import itertools
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph._internal._runnable import RunnableCallable
from langgraph.constants import END, START
from langgraph.graph.state import StateGraph
from langgraph.prebuilt.tool_node import ToolCallWithContext, ToolNode
from langgraph.types import Command, Send
from typing_extensions import NotRequired, Required, TypedDict

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    JumpTo,
    ModelRequest,
    ModelResponse,
    OmitFromSchema,
    ResponseT,
    StateT_co,
    _InputAgentState,
    _OutputAgentState,
)
from langchain.agents.structured_output import (
    AutoStrategy,
    MultipleStructuredOutputsError,
    OutputToolBinding,
    ProviderStrategy,
    ProviderStrategyBinding,
    ResponseFormat,
    StructuredOutputError,
    StructuredOutputValidationError,
    ToolStrategy,
)
from langchain.chat_models import init_chat_model

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from langchain_core.runnables import Runnable
    from langgraph.cache.base import BaseCache
    from langgraph.graph.state import CompiledStateGraph
    from langgraph.runtime import Runtime
    from langgraph.store.base import BaseStore
    from langgraph.types import Checkpointer
    from langgraph.typing import ContextT

    from langchain.agents.middleware.types import ToolCallRequest, ToolCallWrapper

STRUCTURED_OUTPUT_ERROR_TEMPLATE = "Error: {error}\n Please fix your mistakes."

FALLBACK_MODELS_WITH_STRUCTURED_OUTPUT = [
    # if model profile data are not available, these models are assumed to support
    # 中文: 如果模型配置文件数据不可用，则假定这些模型支持  # noqa: RUF003
    # structured output
    # 中文: 结构化输出  # noqa: ERA001
    "grok",
    "gpt-5",
    "gpt-4.1",
    "gpt-4o",
    "gpt-oss",
    "o3-pro",
    "o3-mini",
]


def _normalize_to_model_response(result: ModelResponse | AIMessage) -> ModelResponse:
    """Normalize middleware return value to ModelResponse.

    中文翻译:
    将中间件返回值标准化为 ModelResponse。
    """
    if isinstance(result, AIMessage):
        return ModelResponse(result=[result], structured_response=None)
    return result


def _chain_model_call_handlers(
    handlers: Sequence[
        Callable[
            [ModelRequest, Callable[[ModelRequest], ModelResponse]],
            ModelResponse | AIMessage,
        ]
    ],
) -> (
    Callable[
        [ModelRequest, Callable[[ModelRequest], ModelResponse]],
        ModelResponse,
    ]
    | None
):
    """Compose multiple wrap_model_call handlers into single middleware stack.

    Composes handlers so first in list becomes outermost layer. Each handler
    receives a handler callback to execute inner layers.

    Args:
        handlers: List of handlers. First handler wraps all others.

    Returns:
        Composed handler, or `None` if handlers empty.

    Example:
        ```python
        # handlers=[auth, retry] means: auth wraps retry
        # 中文: handlers=[auth, retry] 表示：auth 包装重试
        # Flow: auth calls retry, retry calls base handler
        # 中文: 流程：auth调用retry，retry调用base handler
        def auth(req, state, runtime, handler):
            try:
                return handler(req)
            except UnauthorizedError:
                refresh_token()
                return handler(req)


        def retry(req, state, runtime, handler):
            for attempt in range(3):
                try:
                    return handler(req)
                except Exception:
                    if attempt == 2:
                        raise


        handler = _chain_model_call_handlers([auth, retry])
        ```


    中文翻译:
    将多个wrap_model_call处理程序组合成单个中间件堆栈。
    组成处理程序，使列表中的第一个成为最外层。每个处理程序
    接收处理程序回调以执行内层。
    参数：
        处理程序：处理程序列表。第一个处理程序包装所有其他处理程序。
    返回：
        组合处理程序，如果处理程序为空，则为“无”。
    示例：
        ````蟒蛇
        # handlers=[auth, retry] 意思是：auth 包装重试
        # 流程：auth调用retry，retry调用base handler
        def auth(请求、状态、运行时、处理程序):
            尝试：
                返回处理程序（请求）
            除了未经授权的错误：
                刷新令牌（）
                返回处理程序（请求）
        def 重试（请求、状态、运行时、处理程序）：
            对于范围（3）中的尝试：
                尝试：
                    返回处理程序（请求）
                除了例外：
                    如果尝试 == 2：
                        提高
        处理程序 = _chain_model_call_handlers([auth, 重试])
        ````
    """  # noqa: RUF002
    if not handlers:
        return None

    if len(handlers) == 1:
        # Single handler - wrap to normalize output
        # 中文: 单个处理程序 - 换行以规范化输出  # noqa: ERA001
        single_handler = handlers[0]

        def normalized_single(
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse],
        ) -> ModelResponse:
            result = single_handler(request, handler)
            return _normalize_to_model_response(result)

        return normalized_single

    def compose_two(
        outer: Callable[
            [ModelRequest, Callable[[ModelRequest], ModelResponse]],
            ModelResponse | AIMessage,
        ],
        inner: Callable[
            [ModelRequest, Callable[[ModelRequest], ModelResponse]],
            ModelResponse | AIMessage,
        ],
    ) -> Callable[
        [ModelRequest, Callable[[ModelRequest], ModelResponse]],
        ModelResponse,
    ]:
        """Compose two handlers where outer wraps inner.

        中文翻译:
        组成两个处理程序，其中外部包裹内部。
        """  # noqa: RUF002

        def composed(
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse],
        ) -> ModelResponse:
            # Create a wrapper that calls inner with the base handler and normalizes
            # 中文: 创建一个包装器，使用基本处理程序调用内部并规范化
            def inner_handler(req: ModelRequest) -> ModelResponse:
                inner_result = inner(req, handler)
                return _normalize_to_model_response(inner_result)

            # Call outer with the wrapped inner as its handler and normalize
            # 中文: 使用包装的内部作为其处理程序调用外部并规范化  # noqa: ERA001
            outer_result = outer(request, inner_handler)
            return _normalize_to_model_response(outer_result)

        return composed

    # Compose right-to-left: outer(inner(innermost(handler)))
    # 中文: 从右到左编写：outer(inner(innermost(handler)))
    result = handlers[-1]
    for handler in reversed(handlers[:-1]):
        result = compose_two(handler, result)

    # Wrap to ensure final return type is exactly ModelResponse
    # 中文: 换行以确保最终返回类型恰好是 ModelResponse
    def final_normalized(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        # result here is typed as returning ModelResponse | AIMessage but compose_two normalizes
        # 中文: 这里的结果被输入为返回 ModelResponse | AIMessage 但 compose_two 标准化
        final_result = result(request, handler)
        return _normalize_to_model_response(final_result)

    return final_normalized


def _chain_async_model_call_handlers(
    handlers: Sequence[
        Callable[
            [ModelRequest, Callable[[ModelRequest], Awaitable[ModelResponse]]],
            Awaitable[ModelResponse | AIMessage],
        ]
    ],
) -> (
    Callable[
        [ModelRequest, Callable[[ModelRequest], Awaitable[ModelResponse]]],
        Awaitable[ModelResponse],
    ]
    | None
):
    """Compose multiple async `wrap_model_call` handlers into single middleware stack.

    Args:
        handlers: List of async handlers. First handler wraps all others.

    Returns:
        Composed async handler, or `None` if handlers empty.


    中文翻译:
    将多个异步“wrap_model_call”处理程序组合成单个中间件堆栈。
    参数：
        handlers：异步处理程序列表。第一个处理程序包装所有其他处理程序。
    返回：
        组合异步处理程序，如果处理程序为空，则为“无”。
    """
    if not handlers:
        return None

    if len(handlers) == 1:
        # Single handler - wrap to normalize output
        # 中文: 单个处理程序 - 换行以规范化输出
        single_handler = handlers[0]

        async def normalized_single(
            request: ModelRequest,
            handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            result = await single_handler(request, handler)
            return _normalize_to_model_response(result)

        return normalized_single

    def compose_two(
        outer: Callable[
            [ModelRequest, Callable[[ModelRequest], Awaitable[ModelResponse]]],
            Awaitable[ModelResponse | AIMessage],
        ],
        inner: Callable[
            [ModelRequest, Callable[[ModelRequest], Awaitable[ModelResponse]]],
            Awaitable[ModelResponse | AIMessage],
        ],
    ) -> Callable[
        [ModelRequest, Callable[[ModelRequest], Awaitable[ModelResponse]]],
        Awaitable[ModelResponse],
    ]:
        """Compose two async handlers where outer wraps inner.

        中文翻译:
        编写两个异步处理程序，其中外部包裹内部。"""

        async def composed(
            request: ModelRequest,
            handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            # Create a wrapper that calls inner with the base handler and normalizes
            # 中文: 创建一个包装器，使用基本处理程序调用内部并规范化
            async def inner_handler(req: ModelRequest) -> ModelResponse:
                inner_result = await inner(req, handler)
                return _normalize_to_model_response(inner_result)

            # Call outer with the wrapped inner as its handler and normalize
            # 中文: 使用包装的内部作为其处理程序调用外部并规范化
            outer_result = await outer(request, inner_handler)
            return _normalize_to_model_response(outer_result)

        return composed

    # Compose right-to-left: outer(inner(innermost(handler)))
    # 中文: 从右到左编写：outer(inner(innermost(handler)))
    result = handlers[-1]
    for handler in reversed(handlers[:-1]):
        result = compose_two(handler, result)

    # Wrap to ensure final return type is exactly ModelResponse
    # 中文: 换行以确保最终返回类型恰好是 ModelResponse
    async def final_normalized(
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        # result here is typed as returning ModelResponse | AIMessage but compose_two normalizes
        # 中文: 这里的结果被输入为返回 ModelResponse | AIMessage 但 compose_two 标准化
        final_result = await result(request, handler)
        return _normalize_to_model_response(final_result)

    return final_normalized


def _resolve_schema(schemas: set[type], schema_name: str, omit_flag: str | None = None) -> type:
    """Resolve schema by merging schemas and optionally respecting `OmitFromSchema` annotations.

    Args:
        schemas: List of schema types to merge
        schema_name: Name for the generated `TypedDict`
        omit_flag: If specified, omit fields with this flag set (`'input'` or
            `'output'`)


    中文翻译:
    通过合并模式并可选择遵守“OmitFromSchema”注释来解析模式。
    参数：
        schemas：要合并的模式类型列表
        schema_name：生成的“TypedDict”的名称
        omit_flag：如果指定，则省略设置此标志的字段（“输入”或
            `'输出'`)"""
    all_annotations = {}

    for schema in schemas:
        hints = get_type_hints(schema, include_extras=True)

        for field_name, field_type in hints.items():
            should_omit = False

            if omit_flag:
                # Check for omission in the annotation metadata
                # 中文: 检查注释元数据中是否有遗漏
                metadata = _extract_metadata(field_type)
                for meta in metadata:
                    if isinstance(meta, OmitFromSchema) and getattr(meta, omit_flag) is True:
                        should_omit = True
                        break

            if not should_omit:
                all_annotations[field_name] = field_type

    return TypedDict(schema_name, all_annotations)  # type: ignore[operator]


def _extract_metadata(type_: type) -> list:
    """Extract metadata from a field type, handling Required/NotRequired and Annotated wrappers.

    中文翻译:
    从字段类型中提取元数据，处理必需/非必需和带注释的包装器。"""
    # Handle Required[Annotated[...]] or NotRequired[Annotated[...]]
    if get_origin(type_) in {Required, NotRequired}:
        inner_type = get_args(type_)[0]
        if get_origin(inner_type) is Annotated:
            return list(get_args(inner_type)[1:])

    # Handle direct Annotated[...]
    elif get_origin(type_) is Annotated:
        return list(get_args(type_)[1:])

    return []


def _get_can_jump_to(middleware: AgentMiddleware[Any, Any], hook_name: str) -> list[JumpTo]:
    """Get the `can_jump_to` list from either sync or async hook methods.

    Args:
        middleware: The middleware instance to inspect.
        hook_name: The name of the hook (`'before_model'` or `'after_model'`).

    Returns:
        List of jump destinations, or empty list if not configured.


    中文翻译:
    从同步或异步挂钩方法获取“can_jump_to”列表。
    参数：
        middleware：要检查的中间件实例。
        hook_name：钩子的名称（“before_model”或“after_model”）。
    返回：
        跳转目标列表，如果未配置则为空列表。"""
    # Get the base class method for comparison
    # 中文: 获取基类方法进行比较
    base_sync_method = getattr(AgentMiddleware, hook_name, None)
    base_async_method = getattr(AgentMiddleware, f"a{hook_name}", None)

    # Try sync method first - only if it's overridden from base class
    # 中文: 首先尝试同步方法 - 仅当它被基类覆盖时
    sync_method = getattr(middleware.__class__, hook_name, None)
    if (
        sync_method
        and sync_method is not base_sync_method
        and hasattr(sync_method, "__can_jump_to__")
    ):
        return sync_method.__can_jump_to__

    # Try async method - only if it's overridden from base class
    # 中文: 尝试异步方法 - 仅当它被基类覆盖时
    async_method = getattr(middleware.__class__, f"a{hook_name}", None)
    if (
        async_method
        and async_method is not base_async_method
        and hasattr(async_method, "__can_jump_to__")
    ):
        return async_method.__can_jump_to__

    return []


def _supports_provider_strategy(model: str | BaseChatModel, tools: list | None = None) -> bool:
    """Check if a model supports provider-specific structured output.

    Args:
        model: Model name string or `BaseChatModel` instance.
        tools: Optional list of tools provided to the agent. Needed because some models
            don't support structured output together with tool calling.

    Returns:
        `True` if the model supports provider-specific structured output, `False` otherwise.


    中文翻译:
    检查模型是否支持特定于提供者的结构化输出。
    参数：
        model：模型名称字符串或“BaseChatModel”实例。
        工具：提供给代理的可选工具列表。需要，因为某些型号
            不支持结构化输出和工具调用。
    返回：
        如果模型支持特定于提供者的结构化输出，则为“True”，否则为“False”。"""
    model_name: str | None = None
    if isinstance(model, str):
        model_name = model
    elif isinstance(model, BaseChatModel):
        model_name = (
            getattr(model, "model_name", None)
            or getattr(model, "model", None)
            or getattr(model, "model_id", "")
        )
        model_profile = model.profile
        if (
            model_profile is not None
            and model_profile.get("structured_output")
            # We make an exception for Gemini models, which currently do not support
            # 中文: 我们对 Gemini 型号例外，目前不支持
            # simultaneous tool use with structured output
            # 中文: 同时使用工具和结构化输出
            and not (tools and isinstance(model_name, str) and "gemini" in model_name.lower())
        ):
            return True

    return (
        any(part in model_name.lower() for part in FALLBACK_MODELS_WITH_STRUCTURED_OUTPUT)
        if model_name
        else False
    )


def _handle_structured_output_error(
    exception: Exception,
    response_format: ResponseFormat,
) -> tuple[bool, str]:
    """Handle structured output error. Returns `(should_retry, retry_tool_message)`.

    中文翻译:
    处理结构化输出错误。返回“(should_retry, retry_tool_message)”。"""
    if not isinstance(response_format, ToolStrategy):
        return False, ""

    handle_errors = response_format.handle_errors

    if handle_errors is False:
        return False, ""
    if handle_errors is True:
        return True, STRUCTURED_OUTPUT_ERROR_TEMPLATE.format(error=str(exception))
    if isinstance(handle_errors, str):
        return True, handle_errors
    if isinstance(handle_errors, type) and issubclass(handle_errors, Exception):
        if isinstance(exception, handle_errors):
            return True, STRUCTURED_OUTPUT_ERROR_TEMPLATE.format(error=str(exception))
        return False, ""
    if isinstance(handle_errors, tuple):
        if any(isinstance(exception, exc_type) for exc_type in handle_errors):
            return True, STRUCTURED_OUTPUT_ERROR_TEMPLATE.format(error=str(exception))
        return False, ""
    if callable(handle_errors):
        # type narrowing not working appropriately w/ callable check, can fix later
        # 中文: 类型缩小在可调用检查中无法正常工作，可以稍后修复
        return True, handle_errors(exception)  # type: ignore[return-value,call-arg]
    return False, ""


def _chain_tool_call_wrappers(
    wrappers: Sequence[ToolCallWrapper],
) -> ToolCallWrapper | None:
    """Compose wrappers into middleware stack (first = outermost).

    Args:
        wrappers: Wrappers in middleware order.

    Returns:
        Composed wrapper, or `None` if empty.

    Example:
        wrapper = _chain_tool_call_wrappers([auth, cache, retry])
        # Request flows: auth -> cache -> retry -> tool
        # 中文: 请求流程：auth -> 缓存 -> 重试 -> 工具
        # Response flows: tool -> retry -> cache -> auth
        # 中文: 响应流程：工具->重试->缓存->身份验证


    中文翻译:
    将包装器组成中间件堆栈（第一个=最外层）。
    参数：
        包装器：按中间件顺序的包装器。
    返回：
        组合包装器，如果为空则为“None”。
    示例：
        包装器 = _chain_tool_call_wrappers([验证、缓存、重试])
        # 请求流程：auth -> 缓存 -> 重试 -> 工具
        # 响应流程：工具 -> 重试 -> 缓存 -> 身份验证"""
    if not wrappers:
        return None

    if len(wrappers) == 1:
        return wrappers[0]

    def compose_two(outer: ToolCallWrapper, inner: ToolCallWrapper) -> ToolCallWrapper:
        """Compose two wrappers where outer wraps inner.

        中文翻译:
        将两个包装纸组合在一起，外层包裹内层。"""

        def composed(
            request: ToolCallRequest,
            execute: Callable[[ToolCallRequest], ToolMessage | Command],
        ) -> ToolMessage | Command:
            # Create a callable that invokes inner with the original execute
            # 中文: 创建一个通过原始执行调用内部的可调用对象
            def call_inner(req: ToolCallRequest) -> ToolMessage | Command:
                return inner(req, execute)

            # Outer can call call_inner multiple times
            # 中文: Outer可以多次调用call_inner
            return outer(request, call_inner)

        return composed

    # Chain all wrappers: first -> second -> ... -> last
    result = wrappers[-1]
    for wrapper in reversed(wrappers[:-1]):
        result = compose_two(wrapper, result)

    return result


def _chain_async_tool_call_wrappers(
    wrappers: Sequence[
        Callable[
            [ToolCallRequest, Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]]],
            Awaitable[ToolMessage | Command],
        ]
    ],
) -> (
    Callable[
        [ToolCallRequest, Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]]],
        Awaitable[ToolMessage | Command],
    ]
    | None
):
    """Compose async wrappers into middleware stack (first = outermost).

    Args:
        wrappers: Async wrappers in middleware order.

    Returns:
        Composed async wrapper, or `None` if empty.


    中文翻译:
    将异步包装器组合到中间件堆栈中（第一个=最外层）。
    参数：
        包装器：按中间件顺序的异步包装器。
    返回：
        组合异步包装器，如果为空则为“None”。"""
    if not wrappers:
        return None

    if len(wrappers) == 1:
        return wrappers[0]

    def compose_two(
        outer: Callable[
            [ToolCallRequest, Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]]],
            Awaitable[ToolMessage | Command],
        ],
        inner: Callable[
            [ToolCallRequest, Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]]],
            Awaitable[ToolMessage | Command],
        ],
    ) -> Callable[
        [ToolCallRequest, Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]]],
        Awaitable[ToolMessage | Command],
    ]:
        """Compose two async wrappers where outer wraps inner.

        中文翻译:
        编写两个异步包装器，其中外部包装内部。"""

        async def composed(
            request: ToolCallRequest,
            execute: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
        ) -> ToolMessage | Command:
            # Create an async callable that invokes inner with the original execute
            # 中文: 创建一个异步可调用对象，通过原始执行调用内部
            async def call_inner(req: ToolCallRequest) -> ToolMessage | Command:
                return await inner(req, execute)

            # Outer can call call_inner multiple times
            # 中文: Outer可以多次调用call_inner
            return await outer(request, call_inner)

        return composed

    # Chain all wrappers: first -> second -> ... -> last
    result = wrappers[-1]
    for wrapper in reversed(wrappers[:-1]):
        result = compose_two(wrapper, result)

    return result


def create_agent(
    model: str | BaseChatModel,
    tools: Sequence[BaseTool | Callable | dict[str, Any]] | None = None,
    *,
    system_prompt: str | SystemMessage | None = None,
    middleware: Sequence[AgentMiddleware[StateT_co, ContextT]] = (),
    response_format: ResponseFormat[ResponseT] | type[ResponseT] | None = None,
    state_schema: type[AgentState[ResponseT]] | None = None,
    context_schema: type[ContextT] | None = None,
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    interrupt_before: list[str] | None = None,
    interrupt_after: list[str] | None = None,
    debug: bool = False,
    name: str | None = None,
    cache: BaseCache | None = None,
) -> CompiledStateGraph[
    AgentState[ResponseT], ContextT, _InputAgentState, _OutputAgentState[ResponseT]
]:
    """Creates an agent graph that calls tools in a loop until a stopping condition is met.

    For more details on using `create_agent`,
    visit the [Agents](https://docs.langchain.com/oss/python/langchain/agents) docs.

    Args:
        model: The language model for the agent.

            Can be a string identifier (e.g., `"openai:gpt-4"`) or a direct chat model
            instance (e.g., [`ChatOpenAI`][langchain_openai.ChatOpenAI] or other another
            [LangChain chat model](https://docs.langchain.com/oss/python/integrations/chat)).

            For a full list of supported model strings, see
            [`init_chat_model`][langchain.chat_models.init_chat_model(model_provider)].

            !!! tip ""

                See the [Models](https://docs.langchain.com/oss/python/langchain/models)
                docs for more information.
        tools: A list of tools, `dict`, or `Callable`.

            If `None` or an empty list, the agent will consist of a model node without a
            tool calling loop.


            !!! tip ""

                See the [Tools](https://docs.langchain.com/oss/python/langchain/tools)
                docs for more information.
        system_prompt: An optional system prompt for the LLM.

            Can be a `str` (which will be converted to a `SystemMessage`) or a
            `SystemMessage` instance directly. The system message is added to the
            beginning of the message list when calling the model.
        middleware: A sequence of middleware instances to apply to the agent.

            Middleware can intercept and modify agent behavior at various stages.

            !!! tip ""

                See the [Middleware](https://docs.langchain.com/oss/python/langchain/middleware)
                docs for more information.
        response_format: An optional configuration for structured responses.

            Can be a `ToolStrategy`, `ProviderStrategy`, or a Pydantic model class.

            If provided, the agent will handle structured output during the
            conversation flow.

            Raw schemas will be wrapped in an appropriate strategy based on model
            capabilities.

            !!! tip ""

                See the [Structured output](https://docs.langchain.com/oss/python/langchain/structured-output)
                docs for more information.
        state_schema: An optional `TypedDict` schema that extends `AgentState`.

            When provided, this schema is used instead of `AgentState` as the base
            schema for merging with middleware state schemas. This allows users to
            add custom state fields without needing to create custom middleware.

            Generally, it's recommended to use `state_schema` extensions via middleware
            to keep relevant extensions scoped to corresponding hooks / tools.
        context_schema: An optional schema for runtime context.
        checkpointer: An optional checkpoint saver object.

            Used for persisting the state of the graph (e.g., as chat memory) for a
            single thread (e.g., a single conversation).
        store: An optional store object.

            Used for persisting data across multiple threads (e.g., multiple
            conversations / users).
        interrupt_before: An optional list of node names to interrupt before.

            Useful if you want to add a user confirmation or other interrupt
            before taking an action.
        interrupt_after: An optional list of node names to interrupt after.

            Useful if you want to return directly or run additional processing
            on an output.
        debug: Whether to enable verbose logging for graph execution.

            When enabled, prints detailed information about each node execution, state
            updates, and transitions during agent runtime. Useful for debugging
            middleware behavior and understanding agent execution flow.
        name: An optional name for the `CompiledStateGraph`.

            This name will be automatically used when adding the agent graph to
            another graph as a subgraph node - particularly useful for building
            multi-agent systems.
        cache: An optional `BaseCache` instance to enable caching of graph execution.

    Returns:
        A compiled `StateGraph` that can be used for chat interactions.

    The agent node calls the language model with the messages list (after applying
    the system prompt). If the resulting [`AIMessage`][langchain.messages.AIMessage]
    contains `tool_calls`, the graph will then call the tools. The tools node executes
    the tools and adds the responses to the messages list as
    [`ToolMessage`][langchain.messages.ToolMessage] objects. The agent node then calls
    the language model again. The process repeats until no more `tool_calls` are present
    in the response. The agent then returns the full list of messages.

    Example:
        ```python
        from langchain.agents import create_agent


        def check_weather(location: str) -> str:
            '''Return the weather forecast for the specified location.'''
            return f"It's always sunny in {location}"


        graph = create_agent(
            model="anthropic:claude-sonnet-4-5-20250929",
            tools=[check_weather],
            system_prompt="You are a helpful assistant",
        )
        inputs = {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
        for chunk in graph.stream(inputs, stream_mode="updates"):
            print(chunk)
        ```


    中文翻译:
    创建一个代理图，该代理图循环调用工具，直到满足停止条件。
    有关使用“create_agent”的更多详细信息，
    访问[代理](https://docs.langchain.com/oss/python/langchain/agents)文档。
    参数：
        model：代理的语言模型。
            可以是字符串标识符（例如“openai:gpt-4”）或直接聊天模型
            实例（例如，[`ChatOpenAI`][langchain_openai.ChatOpenAI] 或其他
            [LangChain聊天模型](https://docs.langchain.com/oss/python/integrations/chat))。
            有关支持的模型字符串的完整列表，请参阅
            [`init_chat_model`][langchain.chat_models.init_chat_model(model_provider)]。
            !!!提示“”
                参见[模型](https://docs.langchain.com/oss/python/langchain/models)
                文档以获取更多信息。
        工具：工具列表、“dict”或“Callable”。
            如果为“None”或空列表，则代理将由一个没有模型节点的模型节点组成
            工具调用循环。
            !!!提示“”
                参见【工具】(https://docs.langchain.com/oss/python/langchain/tools)
                文档以获取更多信息。
        system_prompt：LLM 的可选系统提示。
            可以是 `str` （将被转换为 `SystemMessage`）或
            直接`SystemMessage`实例。系统消息添加到
            调用模型时消息列表的开头。
        中间件：应用于代理的一系列中间件实例。
            中间件可以拦截并修改代理在各个阶段的行为。
            !!!提示“”
                参见[中间件](https://docs.langchain.com/oss/python/langchain/middleware)
                文档以获取更多信息。
        response_format：结构化响应的可选配置。
            可以是 `ToolStrategy`、`ProviderStrategy` 或 Pydantic 模型类。
            如果提供，代理将在期间处理结构化输出
            对话流程。
            原始模式将被包装在基于模型的适当策略中
            能力。
            !!!提示“”
                参见[结构化输出](https://docs.langchain.com/oss/python/langchain/structed-output)
                文档以获取更多信息。
        state_schema：扩展“AgentState”的可选“TypedDict”模式。
            提供后，将使用此架构而不是“AgentState”作为基础
            用于与中间件状态模式合并的模式。这允许用户
            添加自定义状态字段，无需创建自定义中间件。
            一般来说，建议通过中间件使用 `state_schema` 扩展
            将相关扩展保持在相应的钩子/工具范围内。
        context_schema：运行时上下文的可选架构。
        checkpointer：可选的检查点保护程序对象。
            用于持久保存图的状态（例如，作为聊天内存）
            单线程（例如，单个对话）。
        store：可选的存储对象。
            用于跨多个线程（例如，多个
            对话/用户）。
        Interrupt_before：之前要中断的节点名称的可选列表。
            如果您想添加用户确认或其他中断，则很有用
            在采取行动之前。
        Interrupt_after：要中断的节点名称的可选列表。
            如果您想直接返回或运行附加处理，则很有用
            在输出上。
        debug：是否为图形执行启用详细日志记录。
            启用后，打印有关每个节点执行、状态的详细信息
            代理运行时期间的更新和转换。对于调试很有用
            中间件行为和理解代理执行流程。
        name：“CompiledStateGraph”的可选名称。
            将代理图添加到时将自动使用此名称
            另一个图作为子图节点 - 对于构建特别有用
            多代理系统。
        缓存：可选的“BaseCache”实例，用于启用图形执行的缓存。
    返回：
        已编译的“StateGraph”，可用于聊天交互。
    代理节点使用消息列表调用语言模型（应用后系统提示）。如果生成的 [`AIMessage`][langchain.messages.AIMessage]
    包含“tool_calls”，图表将调用工具。工具节点执行
    工具并将响应添加到消息列表中，如下所示
    [`ToolMessage`][langchain.messages.ToolMessage] 对象。然后代理节点调用
    再次是语言模型。重复该过程，直到不再存在“tool_calls”
    在回应中。然后代理返回完整的消息列表。
    示例：
        ````蟒蛇
        从 langchain.agents 导入 create_agent
        def check_weather(位置: str) -> str:
            '''返回指定地点的天气预报。'''
            return f“{location} 总是阳光明媚”
        图=创建代理（
            模型=“人类：克劳德-sonnet-4-5-20250929”，
            工具=[检查天气],
            system_prompt="你是一个有用的助手",
        ）
        input = {"messages": [{"role": "user", "content": "旧金山的天气怎么样"}]}
        对于 graph.stream 中的块（输入，stream_mode =“更新”）：
            打印（块）
        ````"""
    # init chat model
    # 中文: 初始化聊天模型
    if isinstance(model, str):
        model = init_chat_model(model)

    # Convert system_prompt to SystemMessage if needed
    # 中文: 如果需要，将 system_prompt 转换为 SystemMessage
    system_message: SystemMessage | None = None
    if system_prompt is not None:
        if isinstance(system_prompt, SystemMessage):
            system_message = system_prompt
        else:
            system_message = SystemMessage(content=system_prompt)

    # Handle tools being None or empty
    # 中文: 处理无或空的工具
    if tools is None:
        tools = []

    # Convert response format and setup structured output tools
    # 中文: 转换响应格式并设置结构化输出工具
    # Raw schemas are wrapped in AutoStrategy to preserve auto-detection intent.
    # 中文: 原始模式包装在 AutoStrategy 中以保留自动检测意图。
    # AutoStrategy is converted to ToolStrategy upfront to calculate tools during agent creation,
    # 中文: AutoStrategy 预先转换为 ToolStrategy，以便在代理创建期间计算工具，
    # but may be replaced with ProviderStrategy later based on model capabilities.
    # 中文: 但稍后可能会根据模型功能替换为 ProviderStrategy。
    initial_response_format: ToolStrategy | ProviderStrategy | AutoStrategy | None
    if response_format is None:
        initial_response_format = None
    elif isinstance(response_format, (ToolStrategy, ProviderStrategy)):
        # Preserve explicitly requested strategies
        # 中文: 保留明确要求的策略
        initial_response_format = response_format
    elif isinstance(response_format, AutoStrategy):
        # AutoStrategy provided - preserve it for later auto-detection
        # 中文: 提供 AutoStrategy - 保留它以供以后自动检测
        initial_response_format = response_format
    else:
        # Raw schema - wrap in AutoStrategy to enable auto-detection
        # 中文: 原始模式 - 包装在 AutoStrategy 中以启用自动检测
        initial_response_format = AutoStrategy(schema=response_format)

    # For AutoStrategy, convert to ToolStrategy to setup tools upfront
    # 中文: 对于 AutoStrategy，转换为 ToolStrategy 以预先设置工具
    # (may be replaced with ProviderStrategy later based on model)
    # 中文: （后期可能会根据模型替换为ProviderStrategy）
    tool_strategy_for_setup: ToolStrategy | None = None
    if isinstance(initial_response_format, AutoStrategy):
        tool_strategy_for_setup = ToolStrategy(schema=initial_response_format.schema)
    elif isinstance(initial_response_format, ToolStrategy):
        tool_strategy_for_setup = initial_response_format

    structured_output_tools: dict[str, OutputToolBinding] = {}
    if tool_strategy_for_setup:
        for response_schema in tool_strategy_for_setup.schema_specs:
            structured_tool_info = OutputToolBinding.from_schema_spec(response_schema)
            structured_output_tools[structured_tool_info.tool.name] = structured_tool_info
    middleware_tools = [t for m in middleware for t in getattr(m, "tools", [])]

    # Collect middleware with wrap_tool_call or awrap_tool_call hooks
    # 中文: 使用wrap_tool_call或awrap_tool_call钩子收集中间件
    # Include middleware with either implementation to ensure NotImplementedError is raised
    # 中文: 在任一实现中包含中间件以确保引发 NotImplementedError
    # when middleware doesn't support the execution path
    # 中文: 当中间件不支持执行路径时
    middleware_w_wrap_tool_call = [
        m
        for m in middleware
        if m.__class__.wrap_tool_call is not AgentMiddleware.wrap_tool_call
        or m.__class__.awrap_tool_call is not AgentMiddleware.awrap_tool_call
    ]

    # Chain all wrap_tool_call handlers into a single composed handler
    # 中文: 将所有wrap_tool_call处理程序链接到一个组合处理程序中
    wrap_tool_call_wrapper = None
    if middleware_w_wrap_tool_call:
        wrappers = [m.wrap_tool_call for m in middleware_w_wrap_tool_call]
        wrap_tool_call_wrapper = _chain_tool_call_wrappers(wrappers)

    # Collect middleware with awrap_tool_call or wrap_tool_call hooks
    # 中文: 使用awrap_tool_call或wrap_tool_call钩子收集中间件
    # Include middleware with either implementation to ensure NotImplementedError is raised
    # 中文: 在任一实现中包含中间件以确保引发 NotImplementedError
    # when middleware doesn't support the execution path
    # 中文: 当中间件不支持执行路径时
    middleware_w_awrap_tool_call = [
        m
        for m in middleware
        if m.__class__.awrap_tool_call is not AgentMiddleware.awrap_tool_call
        or m.__class__.wrap_tool_call is not AgentMiddleware.wrap_tool_call
    ]

    # Chain all awrap_tool_call handlers into a single composed async handler
    # 中文: 将所有 awrap_tool_call 处理程序链接到一个组合的异步处理程序中
    awrap_tool_call_wrapper = None
    if middleware_w_awrap_tool_call:
        async_wrappers = [m.awrap_tool_call for m in middleware_w_awrap_tool_call]
        awrap_tool_call_wrapper = _chain_async_tool_call_wrappers(async_wrappers)

    # Setup tools
    # 中文: 设置工具
    tool_node: ToolNode | None = None
    # Extract built-in provider tools (dict format) and regular tools (BaseTool/callables)
    # 中文: 提取内置提供者工具（dict格式）和常规工具（BaseTool/callables）
    built_in_tools = [t for t in tools if isinstance(t, dict)]
    regular_tools = [t for t in tools if not isinstance(t, dict)]

    # Tools that require client-side execution (must be in ToolNode)
    # 中文: 需要客户端执行的工具（必须位于 ToolNode 中）
    available_tools = middleware_tools + regular_tools

    # Only create ToolNode if we have client-side tools
    # 中文: 仅当我们有客户端工具时才创建 ToolNode
    tool_node = (
        ToolNode(
            tools=available_tools,
            wrap_tool_call=wrap_tool_call_wrapper,
            awrap_tool_call=awrap_tool_call_wrapper,
        )
        if available_tools
        else None
    )

    # Default tools for ModelRequest initialization
    # 中文: ModelRequest 初始化的默认工具
    # Use converted BaseTool instances from ToolNode (not raw callables)
    # 中文: 使用来自 ToolNode 的转换后的 BaseTool 实例（不是原始可调用对象）
    # Include built-ins and converted tools (can be changed dynamically by middleware)
    # 中文: 包括内置和转换工具（可以通过中间件动态更改）
    # Structured tools are NOT included - they're added dynamically based on response_format
    # 中文: 不包括结构化工具 - 它们是根据response_format动态添加的
    if tool_node:
        default_tools = list(tool_node.tools_by_name.values()) + built_in_tools
    else:
        default_tools = list(built_in_tools)

    # validate middleware
    # 中文: 验证中间件
    if len({m.name for m in middleware}) != len(middleware):
        msg = "Please remove duplicate middleware instances."
        raise AssertionError(msg)
    middleware_w_before_agent = [
        m
        for m in middleware
        if m.__class__.before_agent is not AgentMiddleware.before_agent
        or m.__class__.abefore_agent is not AgentMiddleware.abefore_agent
    ]
    middleware_w_before_model = [
        m
        for m in middleware
        if m.__class__.before_model is not AgentMiddleware.before_model
        or m.__class__.abefore_model is not AgentMiddleware.abefore_model
    ]
    middleware_w_after_model = [
        m
        for m in middleware
        if m.__class__.after_model is not AgentMiddleware.after_model
        or m.__class__.aafter_model is not AgentMiddleware.aafter_model
    ]
    middleware_w_after_agent = [
        m
        for m in middleware
        if m.__class__.after_agent is not AgentMiddleware.after_agent
        or m.__class__.aafter_agent is not AgentMiddleware.aafter_agent
    ]
    # Collect middleware with wrap_model_call or awrap_model_call hooks
    # 中文: 使用wrap_model_call或awrap_model_call钩子收集中间件
    # Include middleware with either implementation to ensure NotImplementedError is raised
    # 中文: 在任一实现中包含中间件以确保引发 NotImplementedError
    # when middleware doesn't support the execution path
    # 中文: 当中间件不支持执行路径时
    middleware_w_wrap_model_call = [
        m
        for m in middleware
        if m.__class__.wrap_model_call is not AgentMiddleware.wrap_model_call
        or m.__class__.awrap_model_call is not AgentMiddleware.awrap_model_call
    ]
    # Collect middleware with awrap_model_call or wrap_model_call hooks
    # 中文: 使用awrap_model_call或wrap_model_call钩子收集中间件
    # Include middleware with either implementation to ensure NotImplementedError is raised
    # 中文: 在任一实现中包含中间件以确保引发 NotImplementedError
    # when middleware doesn't support the execution path
    # 中文: 当中间件不支持执行路径时
    middleware_w_awrap_model_call = [
        m
        for m in middleware
        if m.__class__.awrap_model_call is not AgentMiddleware.awrap_model_call
        or m.__class__.wrap_model_call is not AgentMiddleware.wrap_model_call
    ]

    # Compose wrap_model_call handlers into a single middleware stack (sync)
    # 中文: 将wrap_model_call处理程序组合成单个中间件堆栈（同步）
    wrap_model_call_handler = None
    if middleware_w_wrap_model_call:
        sync_handlers = [m.wrap_model_call for m in middleware_w_wrap_model_call]
        wrap_model_call_handler = _chain_model_call_handlers(sync_handlers)

    # Compose awrap_model_call handlers into a single middleware stack (async)
    # 中文: 将 awrap_model_call 处理程序组合到单个中间件堆栈中（异步）
    awrap_model_call_handler = None
    if middleware_w_awrap_model_call:
        async_handlers = [m.awrap_model_call for m in middleware_w_awrap_model_call]
        awrap_model_call_handler = _chain_async_model_call_handlers(async_handlers)

    state_schemas: set[type] = {m.state_schema for m in middleware}
    # Use provided state_schema if available, otherwise use base AgentState
    # 中文: 如果可用，请使用提供的 state_schema，否则使用基本 AgentState
    base_state = state_schema if state_schema is not None else AgentState
    state_schemas.add(base_state)

    resolved_state_schema = _resolve_schema(state_schemas, "StateSchema", None)
    input_schema = _resolve_schema(state_schemas, "InputSchema", "input")
    output_schema = _resolve_schema(state_schemas, "OutputSchema", "output")

    # create graph, add nodes
    # 中文: 创建图，添加节点
    graph: StateGraph[
        AgentState[ResponseT], ContextT, _InputAgentState, _OutputAgentState[ResponseT]
    ] = StateGraph(
        state_schema=resolved_state_schema,
        input_schema=input_schema,
        output_schema=output_schema,
        context_schema=context_schema,
    )

    def _handle_model_output(
        output: AIMessage, effective_response_format: ResponseFormat | None
    ) -> dict[str, Any]:
        """Handle model output including structured responses.

        Args:
            output: The AI message output from the model.
            effective_response_format: The actual strategy used
                (may differ from initial if auto-detected).


        中文翻译:
        处理模型输出，包括结构化响应。
        参数：
            输出：模型输出的AI消息。
            effective_response_format：实际使用的策略
                （如果自动检测到，可能与初始值不同）。"""
        # Handle structured output with provider strategy
        # 中文: 通过提供商策略处理结构化输出
        if isinstance(effective_response_format, ProviderStrategy):
            if not output.tool_calls:
                provider_strategy_binding = ProviderStrategyBinding.from_schema_spec(
                    effective_response_format.schema_spec
                )
                try:
                    structured_response = provider_strategy_binding.parse(output)
                except Exception as exc:
                    schema_name = getattr(
                        effective_response_format.schema_spec.schema, "__name__", "response_format"
                    )
                    validation_error = StructuredOutputValidationError(schema_name, exc, output)
                    raise validation_error from exc
                else:
                    return {"messages": [output], "structured_response": structured_response}
            return {"messages": [output]}

        # Handle structured output with tool strategy
        # 中文: 使用工具策略处理结构化输出
        if (
            isinstance(effective_response_format, ToolStrategy)
            and isinstance(output, AIMessage)
            and output.tool_calls
        ):
            structured_tool_calls = [
                tc for tc in output.tool_calls if tc["name"] in structured_output_tools
            ]

            if structured_tool_calls:
                exception: StructuredOutputError | None = None
                if len(structured_tool_calls) > 1:
                    # Handle multiple structured outputs error
                    # 中文: 处理多个结构化输出错误
                    tool_names = [tc["name"] for tc in structured_tool_calls]
                    exception = MultipleStructuredOutputsError(tool_names, output)
                    should_retry, error_message = _handle_structured_output_error(
                        exception, effective_response_format
                    )
                    if not should_retry:
                        raise exception

                    # Add error messages and retry
                    # 中文: 添加错误消息并重试
                    tool_messages = [
                        ToolMessage(
                            content=error_message,
                            tool_call_id=tc["id"],
                            name=tc["name"],
                        )
                        for tc in structured_tool_calls
                    ]
                    return {"messages": [output, *tool_messages]}

                # Handle single structured output
                # 中文: 处理单一结构化输出
                tool_call = structured_tool_calls[0]
                try:
                    structured_tool_binding = structured_output_tools[tool_call["name"]]
                    structured_response = structured_tool_binding.parse(tool_call["args"])

                    tool_message_content = (
                        effective_response_format.tool_message_content
                        or f"Returning structured response: {structured_response}"
                    )

                    return {
                        "messages": [
                            output,
                            ToolMessage(
                                content=tool_message_content,
                                tool_call_id=tool_call["id"],
                                name=tool_call["name"],
                            ),
                        ],
                        "structured_response": structured_response,
                    }
                except Exception as exc:
                    exception = StructuredOutputValidationError(tool_call["name"], exc, output)
                    should_retry, error_message = _handle_structured_output_error(
                        exception, effective_response_format
                    )
                    if not should_retry:
                        raise exception from exc

                    return {
                        "messages": [
                            output,
                            ToolMessage(
                                content=error_message,
                                tool_call_id=tool_call["id"],
                                name=tool_call["name"],
                            ),
                        ],
                    }

        return {"messages": [output]}

    def _get_bound_model(request: ModelRequest) -> tuple[Runnable, ResponseFormat | None]:
        """Get the model with appropriate tool bindings.

        Performs auto-detection of strategy if needed based on model capabilities.

        Args:
            request: The model request containing model, tools, and response format.

        Returns:
            Tuple of `(bound_model, effective_response_format)` where
            `effective_response_format` is the actual strategy used (may differ from
            initial if auto-detected).


        中文翻译:
        使用适当的工具绑定获取模型。
        如果需要，根据模型功能执行策略自动检测。
        参数：
            request：模型请求，包含模型、工具和响应格式。
        返回：
            “(bound_model, effective_response_format)”的元组，其中
            ` effective_response_format` 是实际使用的策略（可能与
            如果自动检测到则为初始值）。"""
        # Validate ONLY client-side tools that need to exist in tool_node
        # 中文: 仅验证需要存在于 tool_node 中的客户端工具
        # Build map of available client-side tools from the ToolNode
        # 中文: 从 ToolNode 构建可用客户端工具的地图
        # (which has already converted callables)
        # 中文: （已经转换了可调用对象）
        available_tools_by_name = {}
        if tool_node:
            available_tools_by_name = tool_node.tools_by_name.copy()

        # Check if any requested tools are unknown CLIENT-SIDE tools
        # 中文: 检查是否有任何请求的工具是未知的客户端工具
        unknown_tool_names = []
        for t in request.tools:
            # Only validate BaseTool instances (skip built-in dict tools)
            # 中文: 仅验证 BaseTool 实例（跳过内置字典工具）
            if isinstance(t, dict):
                continue
            if isinstance(t, BaseTool) and t.name not in available_tools_by_name:
                unknown_tool_names.append(t.name)

        if unknown_tool_names:
            available_tool_names = sorted(available_tools_by_name.keys())
            msg = (
                f"Middleware returned unknown tool names: {unknown_tool_names}\n\n"
                f"Available client-side tools: {available_tool_names}\n\n"
                "To fix this issue:\n"
                "1. Ensure the tools are passed to create_agent() via "
                "the 'tools' parameter\n"
                "2. If using custom middleware with tools, ensure "
                "they're registered via middleware.tools attribute\n"
                "3. Verify that tool names in ModelRequest.tools match "
                "the actual tool.name values\n"
                "Note: Built-in provider tools (dict format) can be added dynamically."
            )
            raise ValueError(msg)

        # Determine effective response format (auto-detect if needed)
        # 中文: 确定有效的响应格式（如果需要，自动检测）
        effective_response_format: ResponseFormat | None
        if isinstance(request.response_format, AutoStrategy):
            # User provided raw schema via AutoStrategy - auto-detect best strategy based on model
            # 中文: 用户通过 AutoStrategy 提供原始模式 - 根据模型自动检测最佳策略
            if _supports_provider_strategy(request.model, tools=request.tools):
                # Model supports provider strategy - use it
                # 中文: 模型支持提供商策略 - 使用它
                effective_response_format = ProviderStrategy(schema=request.response_format.schema)
            else:
                # Model doesn't support provider strategy - use ToolStrategy
                # 中文: 模型不支持提供者策略 - 使用 ToolStrategy
                effective_response_format = ToolStrategy(schema=request.response_format.schema)
        else:
            # User explicitly specified a strategy - preserve it
            # 中文: 用户明确指定策略 - 保留它
            effective_response_format = request.response_format

        # Build final tools list including structured output tools
        # 中文: 构建最终工具列表，包括结构化输出工具
        # request.tools now only contains BaseTool instances (converted from callables)
        # 中文: request.tools 现在仅包含 BaseTool 实例（从可调用对象转换而来）
        # and dicts (built-ins)
        # 中文: 和字典（内置）
        final_tools = list(request.tools)
        if isinstance(effective_response_format, ToolStrategy):
            # Add structured output tools to final tools list
            # 中文: 将结构化输出工具添加到最终工具列表中
            structured_tools = [info.tool for info in structured_output_tools.values()]
            final_tools.extend(structured_tools)

        # Bind model based on effective response format
        # 中文: 基于有效响应格式的绑定模型
        if isinstance(effective_response_format, ProviderStrategy):
            # (Backward compatibility) Use OpenAI format structured output
            # 中文: （向后兼容）使用OpenAI格式结构化输出
            kwargs = effective_response_format.to_model_kwargs()
            return (
                request.model.bind_tools(
                    final_tools, strict=True, **kwargs, **request.model_settings
                ),
                effective_response_format,
            )

        if isinstance(effective_response_format, ToolStrategy):
            # Current implementation requires that tools used for structured output
            # 中文: 当前的实现需要用于结构化输出的工具
            # have to be declared upfront when creating the agent as part of the
            # 中文: 在创建代理作为代理的一部分时必须预先声明
            # response format. Middleware is allowed to change the response format
            # 中文: 响应格式。中间件允许改变响应格式
            # to a subset of the original structured tools when using ToolStrategy,
            # 中文: 使用 ToolStrategy 时原始结构化工具的子集，
            # but not to add new structured tools that weren't declared upfront.
            # 中文: 但不添加未预先声明的新的结构化工具。
            # Compute output binding
            # 中文: 计算输出绑定
            for tc in effective_response_format.schema_specs:
                if tc.name not in structured_output_tools:
                    msg = (
                        f"ToolStrategy specifies tool '{tc.name}' "
                        "which wasn't declared in the original "
                        "response format when creating the agent."
                    )
                    raise ValueError(msg)

            # Force tool use if we have structured output tools
            # 中文: 如果我们有结构化输出工具，则强制使用工具
            tool_choice = "any" if structured_output_tools else request.tool_choice
            return (
                request.model.bind_tools(
                    final_tools, tool_choice=tool_choice, **request.model_settings
                ),
                effective_response_format,
            )

        # No structured output - standard model binding
        # 中文: 无结构化输出 - 标准模型绑定
        if final_tools:
            return (
                request.model.bind_tools(
                    final_tools, tool_choice=request.tool_choice, **request.model_settings
                ),
                None,
            )
        return request.model.bind(**request.model_settings), None

    def _execute_model_sync(request: ModelRequest) -> ModelResponse:
        """Execute model and return response.

        This is the core model execution logic wrapped by `wrap_model_call` handlers.
        Raises any exceptions that occur during model invocation.


        中文翻译:
        执行模型并返回响应。
        这是由“wrap_model_call”处理程序包装的核心模型执行逻辑。
        引发模型调用期间发生的任何异常。"""
        # Get the bound model (with auto-detection if needed)
        # 中文: 获取绑定模型（如果需要，可以自动检测）
        model_, effective_response_format = _get_bound_model(request)
        messages = request.messages
        if request.system_message:
            messages = [request.system_message, *messages]

        output = model_.invoke(messages)
        if name:
            output.name = name

        # Handle model output to get messages and structured_response
        # 中文: 处理模型输出以获取消息和结构化响应
        handled_output = _handle_model_output(output, effective_response_format)
        messages_list = handled_output["messages"]
        structured_response = handled_output.get("structured_response")

        return ModelResponse(
            result=messages_list,
            structured_response=structured_response,
        )

    def model_node(state: AgentState, runtime: Runtime[ContextT]) -> dict[str, Any]:
        """Sync model request handler with sequential middleware processing.

        中文翻译:
        将模型请求处理程序与顺序中间件处理同步。"""
        request = ModelRequest(
            model=model,
            tools=default_tools,
            system_message=system_message,
            response_format=initial_response_format,
            messages=state["messages"],
            tool_choice=None,
            state=state,
            runtime=runtime,
        )

        if wrap_model_call_handler is None:
            # No handlers - execute directly
            # 中文: 没有处理程序 - 直接执行
            response = _execute_model_sync(request)
        else:
            # Call composed handler with base handler
            # 中文: 使用基本处理程序调用组合处理程序
            response = wrap_model_call_handler(request, _execute_model_sync)

        # Extract state updates from ModelResponse
        # 中文: 从 ModelResponse 中提取状态更新
        state_updates = {"messages": response.result}
        if response.structured_response is not None:
            state_updates["structured_response"] = response.structured_response

        return state_updates

    async def _execute_model_async(request: ModelRequest) -> ModelResponse:
        """Execute model asynchronously and return response.

        This is the core async model execution logic wrapped by `wrap_model_call`
        handlers.

        Raises any exceptions that occur during model invocation.


        中文翻译:
        异步执行模型并返回响应。
        这是由“wrap_model_call”包装的核心异步模型执行逻辑
        处理程序。
        引发模型调用期间发生的任何异常。"""
        # Get the bound model (with auto-detection if needed)
        # 中文: 获取绑定模型（如果需要，可以自动检测）
        model_, effective_response_format = _get_bound_model(request)
        messages = request.messages
        if request.system_message:
            messages = [request.system_message, *messages]

        output = await model_.ainvoke(messages)
        if name:
            output.name = name

        # Handle model output to get messages and structured_response
        # 中文: 处理模型输出以获取消息和结构化响应
        handled_output = _handle_model_output(output, effective_response_format)
        messages_list = handled_output["messages"]
        structured_response = handled_output.get("structured_response")

        return ModelResponse(
            result=messages_list,
            structured_response=structured_response,
        )

    async def amodel_node(state: AgentState, runtime: Runtime[ContextT]) -> dict[str, Any]:
        """Async model request handler with sequential middleware processing.

        中文翻译:
        具有顺序中间件处理的异步模型请求处理程序。"""
        request = ModelRequest(
            model=model,
            tools=default_tools,
            system_message=system_message,
            response_format=initial_response_format,
            messages=state["messages"],
            tool_choice=None,
            state=state,
            runtime=runtime,
        )

        if awrap_model_call_handler is None:
            # No async handlers - execute directly
            # 中文: 没有异步处理程序 - 直接执行
            response = await _execute_model_async(request)
        else:
            # Call composed async handler with base handler
            # 中文: 使用基本处理程序调用组合的异步处理程序
            response = await awrap_model_call_handler(request, _execute_model_async)

        # Extract state updates from ModelResponse
        # 中文: 从 ModelResponse 中提取状态更新
        state_updates = {"messages": response.result}
        if response.structured_response is not None:
            state_updates["structured_response"] = response.structured_response

        return state_updates

    # Use sync or async based on model capabilities
    # 中文: 根据模型功能使用同步或异步
    graph.add_node("model", RunnableCallable(model_node, amodel_node, trace=False))

    # Only add tools node if we have tools
    # 中文: 仅当我们有工具时才添加工具节点
    if tool_node is not None:
        graph.add_node("tools", tool_node)

    # Add middleware nodes
    # 中文: 添加中间件节点
    for m in middleware:
        if (
            m.__class__.before_agent is not AgentMiddleware.before_agent
            or m.__class__.abefore_agent is not AgentMiddleware.abefore_agent
        ):
            # Use RunnableCallable to support both sync and async
            # 中文: 使用 RunnableCallable 支持同步和异步
            # Pass None for sync if not overridden to avoid signature conflicts
            # 中文: 如果不覆盖，则传递 None 进行同步以避免签名冲突
            sync_before_agent = (
                m.before_agent
                if m.__class__.before_agent is not AgentMiddleware.before_agent
                else None
            )
            async_before_agent = (
                m.abefore_agent
                if m.__class__.abefore_agent is not AgentMiddleware.abefore_agent
                else None
            )
            before_agent_node = RunnableCallable(sync_before_agent, async_before_agent, trace=False)
            graph.add_node(
                f"{m.name}.before_agent", before_agent_node, input_schema=resolved_state_schema
            )

        if (
            m.__class__.before_model is not AgentMiddleware.before_model
            or m.__class__.abefore_model is not AgentMiddleware.abefore_model
        ):
            # Use RunnableCallable to support both sync and async
            # 中文: 使用 RunnableCallable 支持同步和异步
            # Pass None for sync if not overridden to avoid signature conflicts
            # 中文: 如果不覆盖，则传递 None 进行同步以避免签名冲突
            sync_before = (
                m.before_model
                if m.__class__.before_model is not AgentMiddleware.before_model
                else None
            )
            async_before = (
                m.abefore_model
                if m.__class__.abefore_model is not AgentMiddleware.abefore_model
                else None
            )
            before_node = RunnableCallable(sync_before, async_before, trace=False)
            graph.add_node(
                f"{m.name}.before_model", before_node, input_schema=resolved_state_schema
            )

        if (
            m.__class__.after_model is not AgentMiddleware.after_model
            or m.__class__.aafter_model is not AgentMiddleware.aafter_model
        ):
            # Use RunnableCallable to support both sync and async
            # 中文: 使用 RunnableCallable 支持同步和异步
            # Pass None for sync if not overridden to avoid signature conflicts
            # 中文: 如果不覆盖，则传递 None 进行同步以避免签名冲突
            sync_after = (
                m.after_model
                if m.__class__.after_model is not AgentMiddleware.after_model
                else None
            )
            async_after = (
                m.aafter_model
                if m.__class__.aafter_model is not AgentMiddleware.aafter_model
                else None
            )
            after_node = RunnableCallable(sync_after, async_after, trace=False)
            graph.add_node(f"{m.name}.after_model", after_node, input_schema=resolved_state_schema)

        if (
            m.__class__.after_agent is not AgentMiddleware.after_agent
            or m.__class__.aafter_agent is not AgentMiddleware.aafter_agent
        ):
            # Use RunnableCallable to support both sync and async
            # 中文: 使用 RunnableCallable 支持同步和异步
            # Pass None for sync if not overridden to avoid signature conflicts
            # 中文: 如果不覆盖，则传递 None 进行同步以避免签名冲突
            sync_after_agent = (
                m.after_agent
                if m.__class__.after_agent is not AgentMiddleware.after_agent
                else None
            )
            async_after_agent = (
                m.aafter_agent
                if m.__class__.aafter_agent is not AgentMiddleware.aafter_agent
                else None
            )
            after_agent_node = RunnableCallable(sync_after_agent, async_after_agent, trace=False)
            graph.add_node(
                f"{m.name}.after_agent", after_agent_node, input_schema=resolved_state_schema
            )

    # Determine the entry node (runs once at start): before_agent -> before_model -> model
    # 中文: 确定入口节点（启动时运行一次）：before_agent -> before_model -> model
    if middleware_w_before_agent:
        entry_node = f"{middleware_w_before_agent[0].name}.before_agent"
    elif middleware_w_before_model:
        entry_node = f"{middleware_w_before_model[0].name}.before_model"
    else:
        entry_node = "model"

    # Determine the loop entry node (beginning of agent loop, excludes before_agent)
    # 中文: 确定循环入口节点（agent循环的开始，不包括before_agent）
    # This is where tools will loop back to for the next iteration
    # 中文: 这是工具将循环回到下一次迭代的地方
    if middleware_w_before_model:
        loop_entry_node = f"{middleware_w_before_model[0].name}.before_model"
    else:
        loop_entry_node = "model"

    # Determine the loop exit node (end of each iteration, can run multiple times)
    # 中文: 确定循环退出节点（每次迭代结束，可以运行多次）
    # This is after_model or model, but NOT after_agent
    # 中文: 这是 after_model 或 model，但不是 after_agent
    if middleware_w_after_model:
        loop_exit_node = f"{middleware_w_after_model[0].name}.after_model"
    else:
        loop_exit_node = "model"

    # Determine the exit node (runs once at end): after_agent or END
    # 中文: 确定退出节点（结束时运行一次）：after_agent 或 END
    if middleware_w_after_agent:
        exit_node = f"{middleware_w_after_agent[-1].name}.after_agent"
    else:
        exit_node = END

    graph.add_edge(START, entry_node)
    # add conditional edges only if tools exist
    # 中文: 仅当工具存在时才添加条件边
    if tool_node is not None:
        # Only include exit_node in destinations if any tool has return_direct=True
        # 中文: 仅当任何工具具有 return_direct=True 时，才在目标中包含 exit_node
        # or if there are structured output tools
        # 中文: 或者是否有结构化输出工具
        tools_to_model_destinations = [loop_entry_node]
        if (
            any(tool.return_direct for tool in tool_node.tools_by_name.values())
            or structured_output_tools
        ):
            tools_to_model_destinations.append(exit_node)

        graph.add_conditional_edges(
            "tools",
            RunnableCallable(
                _make_tools_to_model_edge(
                    tool_node=tool_node,
                    model_destination=loop_entry_node,
                    structured_output_tools=structured_output_tools,
                    end_destination=exit_node,
                ),
                trace=False,
            ),
            tools_to_model_destinations,
        )

        # base destinations are tools and exit_node
        # 中文: 基本目的地是tools和exit_node
        # we add the loop_entry node to edge destinations if:
        # 中文: 如果满足以下条件，我们将loop_entry节点添加到边缘目的地：
        # - there is an after model hook(s) -- allows jump_to to model
        # 中文: - 有一个模型后钩子 - 允许跳转到模型
        #   potentially artificially injected tool messages, ex HITL
        #   中文: 可能人为注入的工具消息，例如 HITL
        # - there is a response format -- to allow for jumping to model to handle
        # 中文: - 有一个响应格式——允许跳转到模型来处理
        #   regenerating structured output tool calls
        #   中文: 重新生成结构化输出工具调用
        model_to_tools_destinations = ["tools", exit_node]
        if response_format or loop_exit_node != "model":
            model_to_tools_destinations.append(loop_entry_node)

        graph.add_conditional_edges(
            loop_exit_node,
            RunnableCallable(
                _make_model_to_tools_edge(
                    model_destination=loop_entry_node,
                    structured_output_tools=structured_output_tools,
                    end_destination=exit_node,
                ),
                trace=False,
            ),
            model_to_tools_destinations,
        )
    elif len(structured_output_tools) > 0:
        graph.add_conditional_edges(
            loop_exit_node,
            RunnableCallable(
                _make_model_to_model_edge(
                    model_destination=loop_entry_node,
                    end_destination=exit_node,
                ),
                trace=False,
            ),
            [loop_entry_node, exit_node],
        )
    elif loop_exit_node == "model":
        # If no tools and no after_model, go directly to exit_node
        # 中文: 如果没有工具也没有after_model，则直接进入exit_node
        graph.add_edge(loop_exit_node, exit_node)
    # No tools but we have after_model - connect after_model to exit_node
    # 中文: 没有工具，但我们有 after_model - 将 after_model 连接到 exit_node
    else:
        _add_middleware_edge(
            graph,
            name=f"{middleware_w_after_model[0].name}.after_model",
            default_destination=exit_node,
            model_destination=loop_entry_node,
            end_destination=exit_node,
            can_jump_to=_get_can_jump_to(middleware_w_after_model[0], "after_model"),
        )

    # Add before_agent middleware edges
    # 中文: 添加 before_agent 中间件边缘
    if middleware_w_before_agent:
        for m1, m2 in itertools.pairwise(middleware_w_before_agent):
            _add_middleware_edge(
                graph,
                name=f"{m1.name}.before_agent",
                default_destination=f"{m2.name}.before_agent",
                model_destination=loop_entry_node,
                end_destination=exit_node,
                can_jump_to=_get_can_jump_to(m1, "before_agent"),
            )
        # Connect last before_agent to loop_entry_node (before_model or model)
        # 中文: 将最后一个 before_agent 连接到loop_entry_node（before_model 或 model）
        _add_middleware_edge(
            graph,
            name=f"{middleware_w_before_agent[-1].name}.before_agent",
            default_destination=loop_entry_node,
            model_destination=loop_entry_node,
            end_destination=exit_node,
            can_jump_to=_get_can_jump_to(middleware_w_before_agent[-1], "before_agent"),
        )

    # Add before_model middleware edges
    # 中文: 添加 before_model 中间件边缘
    if middleware_w_before_model:
        for m1, m2 in itertools.pairwise(middleware_w_before_model):
            _add_middleware_edge(
                graph,
                name=f"{m1.name}.before_model",
                default_destination=f"{m2.name}.before_model",
                model_destination=loop_entry_node,
                end_destination=exit_node,
                can_jump_to=_get_can_jump_to(m1, "before_model"),
            )
        # Go directly to model after the last before_model
        # 中文: 直接进入最后一个before_model之后的model
        _add_middleware_edge(
            graph,
            name=f"{middleware_w_before_model[-1].name}.before_model",
            default_destination="model",
            model_destination=loop_entry_node,
            end_destination=exit_node,
            can_jump_to=_get_can_jump_to(middleware_w_before_model[-1], "before_model"),
        )

    # Add after_model middleware edges
    # 中文: 添加 after_model 中间件边缘
    if middleware_w_after_model:
        graph.add_edge("model", f"{middleware_w_after_model[-1].name}.after_model")
        for idx in range(len(middleware_w_after_model) - 1, 0, -1):
            m1 = middleware_w_after_model[idx]
            m2 = middleware_w_after_model[idx - 1]
            _add_middleware_edge(
                graph,
                name=f"{m1.name}.after_model",
                default_destination=f"{m2.name}.after_model",
                model_destination=loop_entry_node,
                end_destination=exit_node,
                can_jump_to=_get_can_jump_to(m1, "after_model"),
            )
        # Note: Connection from after_model to after_agent/END is handled above
        # 中文: 注意：上面处理了从 after_model 到 after_agent/END 的连接
        # in the conditional edges section
        # 中文: 在条件边部分

    # Add after_agent middleware edges
    # 中文: 添加 after_agent 中间件边缘
    if middleware_w_after_agent:
        # Chain after_agent middleware (runs once at the very end, before END)
        # 中文: Chain after_agent 中间件（在最后运行一次，在 END 之前）
        for idx in range(len(middleware_w_after_agent) - 1, 0, -1):
            m1 = middleware_w_after_agent[idx]
            m2 = middleware_w_after_agent[idx - 1]
            _add_middleware_edge(
                graph,
                name=f"{m1.name}.after_agent",
                default_destination=f"{m2.name}.after_agent",
                model_destination=loop_entry_node,
                end_destination=exit_node,
                can_jump_to=_get_can_jump_to(m1, "after_agent"),
            )

        # Connect the last after_agent to END
        # 中文: 将最后一个 after_agent 连接到 END
        _add_middleware_edge(
            graph,
            name=f"{middleware_w_after_agent[0].name}.after_agent",
            default_destination=END,
            model_destination=loop_entry_node,
            end_destination=exit_node,
            can_jump_to=_get_can_jump_to(middleware_w_after_agent[0], "after_agent"),
        )

    return graph.compile(
        checkpointer=checkpointer,
        store=store,
        interrupt_before=interrupt_before,
        interrupt_after=interrupt_after,
        debug=debug,
        name=name,
        cache=cache,
    ).with_config({"recursion_limit": 10_000})


def _resolve_jump(
    jump_to: JumpTo | None,
    *,
    model_destination: str,
    end_destination: str,
) -> str | None:
    if jump_to == "model":
        return model_destination
    if jump_to == "end":
        return end_destination
    if jump_to == "tools":
        return "tools"
    return None


def _fetch_last_ai_and_tool_messages(
    messages: list[AnyMessage],
) -> tuple[AIMessage, list[ToolMessage]]:
    last_ai_index: int
    last_ai_message: AIMessage

    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], AIMessage):
            last_ai_index = i
            last_ai_message = cast("AIMessage", messages[i])
            break

    tool_messages = [m for m in messages[last_ai_index + 1 :] if isinstance(m, ToolMessage)]
    return last_ai_message, tool_messages


def _make_model_to_tools_edge(
    *,
    model_destination: str,
    structured_output_tools: dict[str, OutputToolBinding],
    end_destination: str,
) -> Callable[[dict[str, Any]], str | list[Send] | None]:
    def model_to_tools(
        state: dict[str, Any],
    ) -> str | list[Send] | None:
        # 1. if there's an explicit jump_to in the state, use it
        # 中文: 1.如果状态中有显式的jump_to，则使用它
        if jump_to := state.get("jump_to"):
            return _resolve_jump(
                jump_to,
                model_destination=model_destination,
                end_destination=end_destination,
            )

        last_ai_message, tool_messages = _fetch_last_ai_and_tool_messages(state["messages"])
        tool_message_ids = [m.tool_call_id for m in tool_messages]

        # 2. if the model hasn't called any tools, exit the loop
        # 中文: 2.如果模型没有调用任何工具，则退出循环
        # this is the classic exit condition for an agent loop
        # 中文: 这是代理循环的经典退出条件
        if len(last_ai_message.tool_calls) == 0:
            return end_destination

        pending_tool_calls = [
            c
            for c in last_ai_message.tool_calls
            if c["id"] not in tool_message_ids and c["name"] not in structured_output_tools
        ]

        # 3. if there are pending tool calls, jump to the tool node
        # 中文: 3.如果有待处理的工具调用，跳转到工具节点
        if pending_tool_calls:
            return [
                Send(
                    "tools",
                    ToolCallWithContext(
                        __type="tool_call_with_context",
                        tool_call=tool_call,
                        state=state,
                    ),
                )
                for tool_call in pending_tool_calls
            ]

        # 4. if there is a structured response, exit the loop
        # 中文: 4. 如果有结构化响应，则退出循环
        if "structured_response" in state:
            return end_destination

        # 5. AIMessage has tool calls, but there are no pending tool calls
        # 中文: 5. AIMessage有工具调用，但没有待处理的工具调用
        # which suggests the injection of artificial tool messages. jump to the model node
        # 中文: 这表明注入人工工具消息。跳转到模型节点
        return model_destination

    return model_to_tools


def _make_model_to_model_edge(
    *,
    model_destination: str,
    end_destination: str,
) -> Callable[[dict[str, Any]], str | list[Send] | None]:
    def model_to_model(
        state: dict[str, Any],
    ) -> str | list[Send] | None:
        # 1. Priority: Check for explicit jump_to directive from middleware
        # 中文: 1. 优先级：检查中间件中是否有显式的 Jump_to 指令
        if jump_to := state.get("jump_to"):
            return _resolve_jump(
                jump_to,
                model_destination=model_destination,
                end_destination=end_destination,
            )

        # 2. Exit condition: A structured response was generated
        # 中文: 2. 退出条件：生成结构化响应
        if "structured_response" in state:
            return end_destination

        # 3. Default: Continue the loop, there may have been an issue
        # 中文: 3.默认：继续循环，可能有问题
        #     with structured output generation, so we need to retry
        #     中文: 具有结构化输出生成，因此我们需要重试
        return model_destination

    return model_to_model


def _make_tools_to_model_edge(
    *,
    tool_node: ToolNode,
    model_destination: str,
    structured_output_tools: dict[str, OutputToolBinding],
    end_destination: str,
) -> Callable[[dict[str, Any]], str | None]:
    def tools_to_model(state: dict[str, Any]) -> str | None:
        last_ai_message, tool_messages = _fetch_last_ai_and_tool_messages(state["messages"])

        # 1. Exit condition: All executed tools have return_direct=True
        # 中文: 1.退出条件：所有执行的工具都有return_direct=True
        # Filter to only client-side tools (provider tools are not in tool_node)
        # 中文: 仅过滤客户端工具（提供程序工具不在 tool_node 中）
        client_side_tool_calls = [
            c for c in last_ai_message.tool_calls if c["name"] in tool_node.tools_by_name
        ]
        if client_side_tool_calls and all(
            tool_node.tools_by_name[c["name"]].return_direct for c in client_side_tool_calls
        ):
            return end_destination

        # 2. Exit condition: A structured output tool was executed
        # 中文: 2.退出条件：结构化输出工具被执行
        if any(t.name in structured_output_tools for t in tool_messages):
            return end_destination

        # 3. Default: Continue the loop
        # 中文: 3.默认：继续循环
        #    Tool execution completed successfully, route back to the model
        #    中文: 工具执行成功完成，返回模型
        #    so it can process the tool results and decide the next action.
        #    中文: 因此它可以处理工具结果并决定下一步行动。
        return model_destination

    return tools_to_model


def _add_middleware_edge(
    graph: StateGraph[
        AgentState[ResponseT], ContextT, _InputAgentState, _OutputAgentState[ResponseT]
    ],
    *,
    name: str,
    default_destination: str,
    model_destination: str,
    end_destination: str,
    can_jump_to: list[JumpTo] | None,
) -> None:
    """Add an edge to the graph for a middleware node.

    Args:
        graph: The graph to add the edge to.
        name: The name of the middleware node.
        default_destination: The default destination for the edge.
        model_destination: The destination for the edge to the model.
        end_destination: The destination for the edge to the end.
        can_jump_to: The conditionally jumpable destinations for the edge.


    中文翻译:
    向中间件节点的图形添加一条边。
    参数：
        graph：要添加边的图。
        name：中间件节点的名称。
        default_destination：边的默认目的地。
        model_destination：模型边的目的地。
        end_destination：边缘到末端的目的地。
        can_jump_to：边缘的有条件可跳转目的地。"""
    if can_jump_to:

        def jump_edge(state: dict[str, Any]) -> str:
            return (
                _resolve_jump(
                    state.get("jump_to"),
                    model_destination=model_destination,
                    end_destination=end_destination,
                )
                or default_destination
            )

        destinations = [default_destination]

        if "end" in can_jump_to:
            destinations.append(end_destination)
        if "tools" in can_jump_to:
            destinations.append("tools")
        if "model" in can_jump_to and name != model_destination:
            destinations.append(model_destination)

        graph.add_conditional_edges(name, RunnableCallable(jump_edge, trace=False), destinations)

    else:
        graph.add_edge(name, default_destination)


__all__ = [
    "create_agent",
]
