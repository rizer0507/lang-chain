"""Agent 中间件类型定义模块。

本模块定义了 Agent 系统的核心类型和数据结构。

核心类型:
---------
**AgentMiddleware**: 中间件基类，定义了各种钩子方法
**AgentState**: Agent 状态 TypedDict，包含消息历史
**ModelRequest**: 封装模型调用的请求信息
**ModelResponse**: 封装模型调用的响应信息

钩子装饰器:
-----------
- `@before_agent`: Agent 执行开始前
- `@after_agent`: Agent 执行完成后
- `@before_model`: 模型调用前
- `@after_model`: 模型调用后
- `@wrap_model_call`: 包装模型调用（可重试、缓存等）
- `@wrap_tool_call`: 包装工具调用
- `@dynamic_prompt`: 动态生成系统提示

跳转目标:
---------
中间件可以通过返回 `{"jump_to": "..."}` 改变执行流程：
- `"tools"`: 跳转到工具节点
- `"model"`: 跳转回模型节点
- `"end"`: 结束执行

使用示例:
---------
>>> from langchain.agents.middleware import AgentMiddleware, before_model
>>>
>>> class MyMiddleware(AgentMiddleware):
...     def before_model(self, state, runtime):
...         # 在模型调用前执行自定义逻辑
...         return {"messages": [SystemMessage(content="额外指令")]}
>>>
>>> # 或使用装饰器创建简单中间件
>>> @before_model
... def log_middleware(state, runtime):
...     print(f"调用模型，消息数: {len(state['messages'])}")
...     return None
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field, replace
from inspect import iscoroutinefunction
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Generic,
    Literal,
    Protocol,
    cast,
    overload,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable

    from langgraph.types import Command

# Needed as top level import for Pydantic schema generation on AgentState
# 中文: 需要作为 AgentState 上 Pydantic 架构生成的顶级导入
import warnings
from typing import TypeAlias

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.channels.ephemeral_value import EphemeralValue
from langgraph.graph.message import add_messages
from langgraph.prebuilt.tool_node import ToolCallRequest, ToolCallWrapper
from langgraph.typing import ContextT
from typing_extensions import NotRequired, Required, TypedDict, TypeVar, Unpack

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.tools import BaseTool
    from langgraph.runtime import Runtime

    from langchain.agents.structured_output import ResponseFormat

__all__ = [
    "AgentMiddleware",
    "AgentState",
    "ContextT",
    "ModelRequest",
    "ModelResponse",
    "OmitFromSchema",
    "ResponseT",
    "StateT_co",
    "ToolCallRequest",
    "ToolCallWrapper",
    "after_agent",
    "after_model",
    "before_agent",
    "before_model",
    "dynamic_prompt",
    "hook_config",
    "wrap_tool_call",
]

JumpTo = Literal["tools", "model", "end"]
"""中间件节点返回时可跳转的目标。"""

ResponseT = TypeVar("ResponseT")


class _ModelRequestOverrides(TypedDict, total=False):
    """ModelRequest.override() 方法的可能覆盖选项。"""

    model: BaseChatModel
    system_message: SystemMessage | None
    messages: list[AnyMessage]
    tool_choice: Any | None
    tools: list[BaseTool | dict]
    response_format: ResponseFormat | None
    model_settings: dict[str, Any]


@dataclass(init=False)
class ModelRequest:
    """Agent 的模型请求信息。

    封装了发送给语言模型的所有必要信息。

    属性:
    -----
    model : BaseChatModel
        使用的聊天模型
    messages : list[AnyMessage]
        消息列表（不含系统消息）
    system_message : SystemMessage | None
        系统消息
    tool_choice : Any | None
        工具选择配置
    tools : list[BaseTool | dict]
        可用工具列表
    response_format : ResponseFormat | None
        响应格式规范
    state : AgentState
        Agent 状态
    runtime : Runtime
        运行时上下文
    model_settings : dict[str, Any]
        额外的模型设置
    """

    model: BaseChatModel
    messages: list[AnyMessage]  # excluding system message
    system_message: SystemMessage | None
    tool_choice: Any | None
    tools: list[BaseTool | dict]
    response_format: ResponseFormat | None
    state: AgentState
    runtime: Runtime[ContextT]  # type: ignore[valid-type]
    model_settings: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        *,
        model: BaseChatModel,
        messages: list[AnyMessage],
        system_message: SystemMessage | None = None,
        system_prompt: str | None = None,
        tool_choice: Any | None = None,
        tools: list[BaseTool | dict] | None = None,
        response_format: ResponseFormat | None = None,
        state: AgentState | None = None,
        runtime: Runtime[ContextT] | None = None,
        model_settings: dict[str, Any] | None = None,
    ) -> None:
        """Initialize ModelRequest with backward compatibility for system_prompt.

        Args:
            model: The chat model to use.
            messages: List of messages (excluding system prompt).
            tool_choice: Tool choice configuration.
            tools: List of available tools.
            response_format: Response format specification.
            state: Agent state.
            runtime: Runtime context.
            model_settings: Additional model settings.
            system_message: System message instance (preferred).
            system_prompt: System prompt string (deprecated, converted to SystemMessage).
        

        中文翻译:
        初始化 ModelRequest 并向后兼容 system_prompt。
        参数：
            model：要使用的聊天模型。
            messages：消息列表（不包括系统提示）。
            tool_choice：工具选择配置。
            工具：可用工具的列表。
            response_format：响应格式规范。
            状态：代理状态。
            运行时：运行时上下文。
            model_settings：附加模型设置。
            system_message：系统消息实例（首选）。
            system_prompt：系统提示字符串（已弃用，转换为 SystemMessage）。"""
        # Handle system_prompt/system_message conversion and validation
        # 中文: 处理system_prompt/system_message转换和验证
        if system_prompt is not None and system_message is not None:
            msg = "Cannot specify both system_prompt and system_message"
            raise ValueError(msg)

        if system_prompt is not None:
            system_message = SystemMessage(content=system_prompt)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=DeprecationWarning)
            self.model = model
            self.messages = messages
            self.system_message = system_message
            self.tool_choice = tool_choice
            self.tools = tools if tools is not None else []
            self.response_format = response_format
            self.state = state if state is not None else {"messages": []}
            self.runtime = runtime  # type: ignore[assignment]
            self.model_settings = model_settings if model_settings is not None else {}

    @property
    def system_prompt(self) -> str | None:
        """Get system prompt text from system_message.

        Returns:
            The content of the system message if present, otherwise `None`.
        

        中文翻译:
        从system_message获取系统提示文本。
        返回：
            系统消息的内容（如果存在），否则为“无”。"""
        if self.system_message is None:
            return None
        return self.system_message.text

    def __setattr__(self, name: str, value: Any) -> None:
        """Set an attribute with a deprecation warning.

        Direct attribute assignment on `ModelRequest` is deprecated. Use the
        `override()` method instead to create a new request with modified attributes.

        Args:
            name: Attribute name.
            value: Attribute value.
        

        中文翻译:
        设置带有弃用警告的属性。
        不推荐对“ModelRequest”进行直接属性分配。使用
        `override()` 方法改为创建具有修改属性的新请求。
        参数：
            名称：属性名称。
            value：属性值。"""
        # Special handling for system_prompt - convert to system_message
        # 中文: system_prompt 的特殊处理 - 转换为 system_message
        if name == "system_prompt":
            warnings.warn(
                "Direct attribute assignment to ModelRequest.system_prompt is deprecated. "
                "Use request.override(system_message=SystemMessage(...)) instead to create "
                "a new request with the modified system message.",
                DeprecationWarning,
                stacklevel=2,
            )
            if value is None:
                object.__setattr__(self, "system_message", None)
            else:
                object.__setattr__(self, "system_message", SystemMessage(content=value))
            return

        warnings.warn(
            f"Direct attribute assignment to ModelRequest.{name} is deprecated. "
            f"Use request.override({name}=...) instead to create a new request "
            f"with the modified attribute.",
            DeprecationWarning,
            stacklevel=2,
        )
        object.__setattr__(self, name, value)

    def override(self, **overrides: Unpack[_ModelRequestOverrides]) -> ModelRequest:
        """Replace the request with a new request with the given overrides.

        Returns a new `ModelRequest` instance with the specified attributes replaced.

        This follows an immutable pattern, leaving the original request unchanged.

        Args:
            **overrides: Keyword arguments for attributes to override.

                Supported keys:

                - `model`: `BaseChatModel` instance
                - `system_prompt`: deprecated, use `system_message` instead
                - `system_message`: `SystemMessage` instance
                - `messages`: `list` of messages
                - `tool_choice`: Tool choice configuration
                - `tools`: `list` of available tools
                - `response_format`: Response format specification
                - `model_settings`: Additional model settings

        Returns:
            New `ModelRequest` instance with specified overrides applied.

        Examples:
            !!! example "Create a new request with different model"

                ```python
                new_request = request.override(model=different_model)
                ```

            !!! example "Override system message (preferred)"

                ```python
                from langchain_core.messages import SystemMessage

                new_request = request.override(
                    system_message=SystemMessage(content="New instructions")
                )
                ```

            !!! example "Override multiple attributes"

                ```python
                new_request = request.override(
                    model=ChatOpenAI(model="gpt-4o"),
                    system_message=SystemMessage(content="New instructions"),
                )
                ```
        

        中文翻译:
        将请求替换为具有给定覆盖的新请求。
        返回一个新的“ModelRequest”实例，并替换指定的属性。
        这遵循不可变的模式，使原始请求保持不变。
        参数：
            **overrides：要覆盖的属性的关键字参数。
                支持的按键：
                - `model`: `BaseChatModel` 实例
                - `system_prompt`：已弃用，请使用 `system_message` 代替
                - `system_message`：`SystemMessage` 实例
                - `messages`：消息的`列表`
                - `tool_choice`：工具选择配置
                - `tools`：可用工具的`列表`
                - `response_format`：响应格式规范
                - `model_settings`：附加模型设置
        返回：
            应用了指定覆盖的新“ModelRequest”实例。
        示例：
            !!!示例“使用不同模型创建新请求”
                ````蟒蛇
                new_request = request.override(model= different_model)
                ````
            !!!示例“覆盖系统消息（首选）”
                ````蟒蛇
                从 langchain_core.messages 导入 SystemMessage
                new_request = request.override（
                    system_message=SystemMessage(content="新说明")
                ）
                ````
            !!!示例“覆盖多个属性”
                ````蟒蛇
                new_request = request.override（
                    模型=ChatOpenAI(模型=“gpt-4o”),
                    system_message=SystemMessage(content="新说明"),
                ）
                ````"""
        # Handle system_prompt/system_message conversion
        # 中文: 处理system_prompt/system_message转换
        if "system_prompt" in overrides and "system_message" in overrides:
            msg = "Cannot specify both system_prompt and system_message"
            raise ValueError(msg)

        if "system_prompt" in overrides:
            system_prompt = cast("str", overrides.pop("system_prompt"))  # type: ignore[typeddict-item]
            if system_prompt is None:
                overrides["system_message"] = None
            else:
                overrides["system_message"] = SystemMessage(content=system_prompt)

        return replace(self, **overrides)


@dataclass
class ModelResponse:
    """Response from model execution including messages and optional structured output.

    The result will usually contain a single `AIMessage`, but may include an additional
    `ToolMessage` if the model used a tool for structured output.
    

    中文翻译:
    来自模型执行的响应，包括消息和可选的结构化输出。
    结果通常包含一个“AIMessage”，但可能包含一个附加的
    如果模型使用结构化输出工具，则为“ToolMessage”。"""

    result: list[BaseMessage]
    """List of messages from model execution.

    中文翻译:
    来自模型执行的消息列表。"""

    structured_response: Any = None
    """Parsed structured output if `response_format` was specified, `None` otherwise.

    中文翻译:
    如果指定了“response_format”，则解析结构化输出，否则为“None”。"""


# Type alias for middleware return type - allows returning either full response or just AIMessage
# 中文: Type alias for middleware return type - allows returning either full response or just AIMessage
ModelCallResult: TypeAlias = ModelResponse | AIMessage
"""`TypeAlias` for model call handler return value.

Middleware can return either:

- `ModelResponse`: Full response with messages and optional structured output
- `AIMessage`: Simplified return for simple use cases

中文翻译:
模型调用处理程序返回值的“TypeAlias”。
中间件可以返回：
- `ModelResponse`：带有消息和可选结构化输出的完整响应
- `AIMessage`：简单用例的简化返回
"""


@dataclass
class OmitFromSchema:
    """Annotation used to mark state attributes as omitted from input or output schemas.

    中文翻译:
    用于将状态属性标记为从输入或输出模式中省略的注释。"""

    input: bool = True
    """Whether to omit the attribute from the input schema.

    中文翻译:
    是否从输入模式中省略该属性。"""

    output: bool = True
    """Whether to omit the attribute from the output schema.

    中文翻译:
    是否从输出架构中省略该属性。"""


OmitFromInput = OmitFromSchema(input=True, output=False)
"""Annotation used to mark state attributes as omitted from input schema.

中文翻译:
用于将状态属性标记为从输入模式中省略的注释。"""

OmitFromOutput = OmitFromSchema(input=False, output=True)
"""Annotation used to mark state attributes as omitted from output schema.

中文翻译:
用于将状态属性标记为从输出模式中省略的注释。"""

PrivateStateAttr = OmitFromSchema(input=True, output=True)
"""Annotation used to mark state attributes as purely internal for a given middleware.

中文翻译:
用于将状态属性标记为给定中间件的纯内部属性的注释。"""


class AgentState(TypedDict, Generic[ResponseT]):
    """State schema for the agent.

    中文翻译:
    代理的状态架构。"""

    messages: Required[Annotated[list[AnyMessage], add_messages]]
    jump_to: NotRequired[Annotated[JumpTo | None, EphemeralValue, PrivateStateAttr]]
    structured_response: NotRequired[Annotated[ResponseT, OmitFromInput]]


class _InputAgentState(TypedDict):  # noqa: PYI049
    """Input state schema for the agent.

    中文翻译:
    代理的输入状态架构。"""

    messages: Required[Annotated[list[AnyMessage | dict], add_messages]]


class _OutputAgentState(TypedDict, Generic[ResponseT]):  # noqa: PYI049
    """Output state schema for the agent.

    中文翻译:
    代理的输出状态架构。"""

    messages: Required[Annotated[list[AnyMessage], add_messages]]
    structured_response: NotRequired[ResponseT]


StateT = TypeVar("StateT", bound=AgentState, default=AgentState)
StateT_co = TypeVar("StateT_co", bound=AgentState, default=AgentState, covariant=True)
StateT_contra = TypeVar("StateT_contra", bound=AgentState, contravariant=True)


class AgentMiddleware(Generic[StateT, ContextT]):
    """Base middleware class for an agent.

    Subclass this and implement any of the defined methods to customize agent behavior
    between steps in the main agent loop.
    

    中文翻译:
    代理的基础中间件类。
    对此进行子类化并实现任何已定义的方法来自定义代理行为
    主代理循环中的步骤之间。"""

    state_schema: type[StateT] = cast("type[StateT]", AgentState)
    """The schema for state passed to the middleware nodes.

    中文翻译:
    传递到中间件节点的状态模式。"""

    tools: Sequence[BaseTool]
    """Additional tools registered by the middleware.

    中文翻译:
    中间件注册的其他工具。"""

    @property
    def name(self) -> str:
        """The name of the middleware instance.

        Defaults to the class name, but can be overridden for custom naming.
        

        中文翻译:
        中间件实例的名称。
        默认为类名，但可以覆盖自定义命名。"""
        return self.__class__.__name__

    def before_agent(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        """Logic to run before the agent execution starts.

        Async version is `abefore_agent`
        

        中文翻译:
        在代理执行开始之前运行的逻辑。
        异步版本是 `abefore_agent`"""

    async def abefore_agent(
        self, state: StateT, runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """Async logic to run before the agent execution starts.

        中文翻译:
        在代理执行开始之前运行的异步逻辑。"""

    def before_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        """Logic to run before the model is called.

        Async version is `abefore_model`
        

        中文翻译:
        在调用模型之前运行的逻辑。
        异步版本是 `abefore_model`"""

    async def abefore_model(
        self, state: StateT, runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """Async logic to run before the model is called.

        中文翻译:
        在调用模型之前运行的异步逻辑。"""

    def after_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        """Logic to run after the model is called.

        Async version is `aafter_model`
        

        中文翻译:
        调用模型后运行的逻辑。
        异步版本是“aafter_model”"""

    async def aafter_model(
        self, state: StateT, runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """Async logic to run after the model is called.

        中文翻译:
        调用模型后运行的异步逻辑。"""

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Intercept and control model execution via handler callback.

        Async version is `awrap_model_call`

        The handler callback executes the model request and returns a `ModelResponse`.
        Middleware can call the handler multiple times for retry logic, skip calling
        it to short-circuit, or modify the request/response. Multiple middleware
        compose with first in list as outermost layer.

        Args:
            request: Model request to execute (includes state and runtime).
            handler: Callback that executes the model request and returns
                `ModelResponse`.

                Call this to execute the model.

                Can be called multiple times for retry logic.

                Can skip calling it to short-circuit.

        Returns:
            `ModelCallResult`

        Examples:
            !!! example "Retry on error"

                ```python
                def wrap_model_call(self, request, handler):
                    for attempt in range(3):
                        try:
                            return handler(request)
                        except Exception:
                            if attempt == 2:
                                raise
                ```

            !!! example "Rewrite response"

                ```python
                def wrap_model_call(self, request, handler):
                    response = handler(request)
                    ai_msg = response.result[0]
                    return ModelResponse(
                        result=[AIMessage(content=f"[{ai_msg.content}]")],
                        structured_response=response.structured_response,
                    )
                ```

            !!! example "Error to fallback"

                ```python
                def wrap_model_call(self, request, handler):
                    try:
                        return handler(request)
                    except Exception:
                        return ModelResponse(result=[AIMessage(content="Service unavailable")])
                ```

            !!! example "Cache/short-circuit"

                ```python
                def wrap_model_call(self, request, handler):
                    if cached := get_cache(request):
                        return cached  # Short-circuit with cached result
                    response = handler(request)
                    save_cache(request, response)
                    return response
                ```

            !!! example "Simple `AIMessage` return (converted automatically)"

                ```python
                def wrap_model_call(self, request, handler):
                    response = handler(request)
                    # Can return AIMessage directly for simple cases
                    # 中文: 简单情况可以直接返回AIMessage
                    return AIMessage(content="Simplified response")
                ```
        

        中文翻译:
        通过处理程序回调拦截和控制模型执行。
        异步版本是“awrap_model_call”
        处理程序回调执行模型请求并返回“ModelResponse”。
        中间件可以多次调用handler进行重试逻辑，跳过调用
        它短路，或修改请求/响应。多种中间件
        以列表中的第一个作为最外层组成。
        参数：
            request：要执行的模型请求（包括状态和运行时）。
            handler：执行模型请求并返回的回调
                `模型响应`。
                调用它来执行模型。
                可以多次调用重试逻辑。
                可以跳过调用它来短路。
        返回：
            `模型调用结果`
        示例：
            !!!示例“出错时重试”
                ````蟒蛇
                defwrapp_model_call（自身，请求，处理程序）：
                    对于范围（3）中的尝试：
                        尝试：
                            返回处理程序（请求）
                        除了例外：
                            如果尝试 == 2：
                                提高
                ````
            !!!示例“重写响应”
                ````蟒蛇
                defwrapp_model_call（自身，请求，处理程序）：
                    响应=处理程序（请求）
                    ai_msg = 响应.结果[0]
                    返回模型响应（
                        结果=[AIMessage(内容=f"[{ai_msg.content}]")],
                        结构化响应=响应.结构化响应，
                    ）
                ````
            !!!示例“回退错误”
                ````蟒蛇
                defwrapp_model_call（自身，请求，处理程序）：
                    尝试：
                        返回处理程序（请求）
                    除了例外：
                        return ModelResponse(结果=[AIMessage(content="服务不可用")])
                ````
            !!!示例“缓存/短路”
                ````蟒蛇
                defwrapp_model_call（自身，请求，处理程序）：
                    如果缓存:= get_cache(请求):
                        return cached # 与缓存结果短路
                    响应=处理程序（请求）
                    save_cache（请求，响应）
                    返回响应
                ````
            !!!示例“简单的`AIMessage`返回（自动转换）”
                ````蟒蛇
                defwrapp_model_call（自身，请求，处理程序）：
                    响应=处理程序（请求）
                    # 简单情况可以直接返回AIMessage
                    return AIMessage(content="简化响应")
                ````"""
        msg = (
            "Synchronous implementation of wrap_model_call is not available. "
            "You are likely encountering this error because you defined only the async version "
            "(awrap_model_call) and invoked your agent in a synchronous context "
            "(e.g., using `stream()` or `invoke()`). "
            "To resolve this, either: "
            "(1) subclass AgentMiddleware and implement the synchronous wrap_model_call method, "
            "(2) use the @wrap_model_call decorator on a standalone sync function, or "
            "(3) invoke your agent asynchronously using `astream()` or `ainvoke()`."
        )
        raise NotImplementedError(msg)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Intercept and control async model execution via handler callback.

        The handler callback executes the model request and returns a `ModelResponse`.

        Middleware can call the handler multiple times for retry logic, skip calling
        it to short-circuit, or modify the request/response. Multiple middleware
        compose with first in list as outermost layer.

        Args:
            request: Model request to execute (includes state and runtime).
            handler: Async callback that executes the model request and returns
                `ModelResponse`.

                Call this to execute the model.

                Can be called multiple times for retry logic.

                Can skip calling it to short-circuit.

        Returns:
            `ModelCallResult`

        Examples:
            !!! example "Retry on error"

                ```python
                async def awrap_model_call(self, request, handler):
                    for attempt in range(3):
                        try:
                            return await handler(request)
                        except Exception:
                            if attempt == 2:
                                raise
                ```
        

        中文翻译:
        通过处理程序回调拦截和控制异步模型执行。
        处理程序回调执行模型请求并返回“ModelResponse”。
        中间件可以多次调用handler进行重试逻辑，跳过调用
        它短路，或修改请求/响应。多种中间件
        以列表中的第一个作为最外层组成。
        参数：
            request：要执行的模型请求（包括状态和运行时）。
            handler：执行模型请求并返回的异步回调
                `模型响应`。
                调用它来执行模型。
                可以多次调用重试逻辑。
                可以跳过调用它来短路。
        返回：
            `模型调用结果`
        示例：
            !!!示例“出错时重试”
                ````蟒蛇
                异步 def awrap_model_call(self, request, handler):
                    对于范围（3）中的尝试：
                        尝试：
                            返回等待处理程序（请求）
                        除了例外：
                            如果尝试 == 2：
                                提高
                ````"""
        msg = (
            "Asynchronous implementation of awrap_model_call is not available. "
            "You are likely encountering this error because you defined only the sync version "
            "(wrap_model_call) and invoked your agent in an asynchronous context "
            "(e.g., using `astream()` or `ainvoke()`). "
            "To resolve this, either: "
            "(1) subclass AgentMiddleware and implement the asynchronous awrap_model_call method, "
            "(2) use the @wrap_model_call decorator on a standalone async function, or "
            "(3) invoke your agent synchronously using `stream()` or `invoke()`."
        )
        raise NotImplementedError(msg)

    def after_agent(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        """Logic to run after the agent execution completes.

        中文翻译:
        代理执行完成后运行的逻辑。"""

    async def aafter_agent(
        self, state: StateT, runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """Async logic to run after the agent execution completes.

        中文翻译:
        代理执行完成后运行的异步逻辑。"""

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Intercept tool execution for retries, monitoring, or modification.

        Async version is `awrap_tool_call`

        Multiple middleware compose automatically (first defined = outermost).

        Exceptions propagate unless `handle_tool_errors` is configured on `ToolNode`.

        Args:
            request: Tool call request with call `dict`, `BaseTool`, state, and runtime.

                Access state via `request.state` and runtime via `request.runtime`.
            handler: `Callable` to execute the tool (can be called multiple times).

        Returns:
            `ToolMessage` or `Command` (the final result).

        The handler `Callable` can be invoked multiple times for retry logic.

        Each call to handler is independent and stateless.

        Examples:
            !!! example "Modify request before execution"

                ```python
                def wrap_tool_call(self, request, handler):
                    modified_call = {
                        **request.tool_call,
                        "args": {
                            **request.tool_call["args"],
                            "value": request.tool_call["args"]["value"] * 2,
                        },
                    }
                    request = request.override(tool_call=modified_call)
                    return handler(request)
                ```

            !!! example "Retry on error (call handler multiple times)"

                ```python
                def wrap_tool_call(self, request, handler):
                    for attempt in range(3):
                        try:
                            result = handler(request)
                            if is_valid(result):
                                return result
                        except Exception:
                            if attempt == 2:
                                raise
                    return result
                ```

            !!! example "Conditional retry based on response"

                ```python
                def wrap_tool_call(self, request, handler):
                    for attempt in range(3):
                        result = handler(request)
                        if isinstance(result, ToolMessage) and result.status != "error":
                            return result
                        if attempt < 2:
                            continue
                        return result
                ```
        

        中文翻译:
        拦截工具执行以进行重试、监视或修改。
        异步版本是“awrap_tool_call”
        多个中间件自动组合（第一个定义=最外层）。
        除非在“ToolNode”上配置了“handle_tool_errors”，否则异常会传播。
        参数：
            request：工具调用请求，包含调用 `dict`、`BaseTool`、状态和运行时。
                通过“request.state”访问状态，通过“request.runtime”访问运行时。
            handler：`Callable` 来执行该工具（可以多次调用）。
        返回：
            `ToolMessage` 或 `Command` （最终结果）。
        可以多次调用处理程序“Callable”以实现重试逻辑。
        对处理程序的每次调用都是独立且无状态的。
        示例：
            !!!示例“执行前修改请求”
                ````蟒蛇
                defwrapp_tool_call（自身，请求，处理程序）：
                    修改后的调用 = {
                        **请求.tool_call，
                        “参数”：{
                            **request.tool_call["args"],
                            "值": request.tool_call["args"]["value"] * 2,
                        },
                    }
                    请求 = request.override(tool_call=modified_call)
                    返回处理程序（请求）
                ````
            !!!示例“错误重试（多次调用处理程序）”
                ````蟒蛇
                defwrapp_tool_call（自身，请求，处理程序）：
                    对于范围（3）中的尝试：
                        尝试：
                            结果 = 处理程序（请求）
                            如果 is_valid（结果）：
                                返回结果
                        除了例外：
                            如果尝试 == 2：
                                提高
                    返回结果
                ````
            !!!示例“根据响应有条件重试”
                ````蟒蛇
                defwrapp_tool_call（自身，请求，处理程序）：
                    对于范围（3）中的尝试：
                        结果 = 处理程序（请求）
                        if isinstance(result, ToolMessage) 且 result.status != "error":
                            返回结果
                        如果尝试 < 2：
                            继续
                        返回结果
                ````"""
        msg = (
            "Synchronous implementation of wrap_tool_call is not available. "
            "You are likely encountering this error because you defined only the async version "
            "(awrap_tool_call) and invoked your agent in a synchronous context "
            "(e.g., using `stream()` or `invoke()`). "
            "To resolve this, either: "
            "(1) subclass AgentMiddleware and implement the synchronous wrap_tool_call method, "
            "(2) use the @wrap_tool_call decorator on a standalone sync function, or "
            "(3) invoke your agent asynchronously using `astream()` or `ainvoke()`."
        )
        raise NotImplementedError(msg)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Intercept and control async tool execution via handler callback.

        The handler callback executes the tool call and returns a `ToolMessage` or
        `Command`. Middleware can call the handler multiple times for retry logic, skip
        calling it to short-circuit, or modify the request/response. Multiple middleware
        compose with first in list as outermost layer.

        Args:
            request: Tool call request with call `dict`, `BaseTool`, state, and runtime.

                Access state via `request.state` and runtime via `request.runtime`.
            handler: Async callable to execute the tool and returns `ToolMessage` or
                `Command`.

                Call this to execute the tool.

                Can be called multiple times for retry logic.

                Can skip calling it to short-circuit.

        Returns:
            `ToolMessage` or `Command` (the final result).

        The handler `Callable` can be invoked multiple times for retry logic.

        Each call to handler is independent and stateless.

        Examples:
            !!! example "Async retry on error"

                ```python
                async def awrap_tool_call(self, request, handler):
                    for attempt in range(3):
                        try:
                            result = await handler(request)
                            if is_valid(result):
                                return result
                        except Exception:
                            if attempt == 2:
                                raise
                    return result
                ```

                ```python
                async def awrap_tool_call(self, request, handler):
                    if cached := await get_cache_async(request):
                        return ToolMessage(content=cached, tool_call_id=request.tool_call["id"])
                    result = await handler(request)
                    await save_cache_async(request, result)
                    return result
                ```
        

        中文翻译:
        通过处理程序回调拦截和控制异步工具执行。
        处理程序回调执行工具调用并返回“ToolMessage”或
        ‘命令’。中间件可以多次调用处理程序进行重试逻辑，跳过
        调用它来短路，或修改请求/响应。多种中间件
        以列表中的第一个作为最外层组成。
        参数：
            request：工具调用请求，包含调用 `dict`、`BaseTool`、状态和运行时。
                通过“request.state”访问状态，通过“request.runtime”访问运行时。
            handler：异步调用来执行工具并返回“ToolMessage”或
                ‘命令’。
                调用它来执行该工具。
                可以多次调用重试逻辑。
                可以跳过调用它来短路。
        返回：
            `ToolMessage` 或 `Command` （最终结果）。
        可以多次调用处理程序“Callable”以实现重试逻辑。
        对处理程序的每次调用都是独立且无状态的。
        示例：
            !!!示例“出错时异步重试”
                ````蟒蛇
                异步 def awrap_tool_call(self, request, handler):
                    对于范围（3）中的尝试：
                        尝试：
                            结果=等待处理程序（请求）
                            如果 is_valid（结果）：
                                返回结果
                        除了例外：
                            如果尝试 == 2：
                                提高
                    返回结果
                ````
                ````蟒蛇
                异步 def awrap_tool_call(self, request, handler):
                    如果缓存：=等待get_cache_async（请求）：
                        返回 ToolMessage(content=cached, tool_call_id=request.tool_call["id"])
                    结果=等待处理程序（请求）
                    等待 save_cache_async（请求，结果）
                    返回结果
                ````"""
        msg = (
            "Asynchronous implementation of awrap_tool_call is not available. "
            "You are likely encountering this error because you defined only the sync version "
            "(wrap_tool_call) and invoked your agent in an asynchronous context "
            "(e.g., using `astream()` or `ainvoke()`). "
            "To resolve this, either: "
            "(1) subclass AgentMiddleware and implement the asynchronous awrap_tool_call method, "
            "(2) use the @wrap_tool_call decorator on a standalone async function, or "
            "(3) invoke your agent synchronously using `stream()` or `invoke()`."
        )
        raise NotImplementedError(msg)


class _CallableWithStateAndRuntime(Protocol[StateT_contra, ContextT]):
    """Callable with `AgentState` and `Runtime` as arguments.

    中文翻译:
    可使用“AgentState”和“Runtime”作为参数进行调用。"""

    def __call__(
        self, state: StateT_contra, runtime: Runtime[ContextT]
    ) -> dict[str, Any] | Command | None | Awaitable[dict[str, Any] | Command | None]:
        """Perform some logic with the state and runtime.

        中文翻译:
        使用状态和运行时执行一些逻辑。"""
        ...


class _CallableReturningSystemMessage(Protocol[StateT_contra, ContextT]):  # type: ignore[misc]
    """Callable that returns a prompt string or SystemMessage given `ModelRequest`.

    中文翻译:
    给定“ModelRequest”返回提示字符串或 SystemMessage 的可调用函数。"""

    def __call__(
        self, request: ModelRequest
    ) -> str | SystemMessage | Awaitable[str | SystemMessage]:
        """Generate a system prompt string or SystemMessage based on the request.

        中文翻译:
        根据请求生成系统提示字符串或SystemMessage。"""
        ...


class _CallableReturningModelResponse(Protocol[StateT_contra, ContextT]):  # type: ignore[misc]
    """Callable for model call interception with handler callback.

    Receives handler callback to execute model and returns `ModelResponse` or
    `AIMessage`.
    

    中文翻译:
    可通过处理程序回调调用模型调用拦截。
    接收处理程序回调以执行模型并返回“ModelResponse”或
    “AI消息”。"""

    def __call__(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Intercept model execution via handler callback.

        中文翻译:
        通过处理程序回调拦截模型执行。"""
        ...


class _CallableReturningToolResponse(Protocol):
    """Callable for tool call interception with handler callback.

    Receives handler callback to execute tool and returns final `ToolMessage` or
    `Command`.
    

    中文翻译:
    可通过处理程序回调调用工具调用拦截。
    接收处理程序回调以执行工具并返回最终的“ToolMessage”或
    ‘命令’。"""

    def __call__(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Intercept tool execution via handler callback.

        中文翻译:
        通过处理程序回调拦截工具执行。"""
        ...


CallableT = TypeVar("CallableT", bound=Callable[..., Any])


def hook_config(
    *,
    can_jump_to: list[JumpTo] | None = None,
) -> Callable[[CallableT], CallableT]:
    """Decorator to configure hook behavior in middleware methods.

    Use this decorator on `before_model` or `after_model` methods in middleware classes
    to configure their behavior. Currently supports specifying which destinations they
    can jump to, which establishes conditional edges in the agent graph.

    Args:
        can_jump_to: Optional list of valid jump destinations.

            Can be:

            - `'tools'`: Jump to the tools node
            - `'model'`: Jump back to the model node
            - `'end'`: Jump to the end of the graph

    Returns:
        Decorator function that marks the method with configuration metadata.

    Examples:
        !!! example "Using decorator on a class method"

            ```python
            class MyMiddleware(AgentMiddleware):
                @hook_config(can_jump_to=["end", "model"])
                def before_model(self, state: AgentState) -> dict[str, Any] | None:
                    if some_condition(state):
                        return {"jump_to": "end"}
                    return None
            ```

        Alternative: Use the `can_jump_to` parameter in `before_model`/`after_model`
        decorators:

        ```python
        @before_model(can_jump_to=["end"])
        def conditional_middleware(state: AgentState) -> dict[str, Any] | None:
            if should_exit(state):
                return {"jump_to": "end"}
            return None
        ```
    

    中文翻译:
    用于在中间件方法中配置挂钩行为的装饰器。
    在中间件类中的“before_model”或“after_model”方法上使用此装饰器
    配置他们的行为。目前支持指定目的地
    可以跳转到，这在代理图中建立了条件边。
    参数：
        can_jump_to：有效跳转目的地的可选列表。
            可以是：
            - `'tools'`：跳转到工具节点
            - `'model'`：跳回模型节点
            - `'end'`：跳转到图表的末尾
    返回：
        使用配置元数据标记方法的装饰器函数。
    示例：
        !!!示例“在类方法上使用装饰器”
            ````蟒蛇
            类 MyMiddleware(AgentMiddleware):
                @hook_config(can_jump_to=[“结束”,“模型”])
                def before_model(self, 状态: AgentState) -> dict[str, Any] |无：
                    如果某些_条件（状态）：
                        返回 {"jump_to": "end"}
                    返回无
            ````
        替代方案：在“before_model”/“after_model”中使用“can_jump_to”参数
        装饰器：
        ````蟒蛇
        @before_model(can_jump_to=[“结束”])
        def conditional_middleware(state: AgentState) -> dict[str, Any] | 字典[str, Any] |无：
            如果应该_退出（状态）：
                返回 {"jump_to": "end"}
            返回无
        ````"""

    def decorator(func: CallableT) -> CallableT:
        if can_jump_to is not None:
            func.__can_jump_to__ = can_jump_to  # type: ignore[attr-defined]
        return func

    return decorator


@overload
def before_model(
    func: _CallableWithStateAndRuntime[StateT, ContextT],
) -> AgentMiddleware[StateT, ContextT]: ...


@overload
def before_model(
    func: None = None,
    *,
    state_schema: type[StateT] | None = None,
    tools: list[BaseTool] | None = None,
    can_jump_to: list[JumpTo] | None = None,
    name: str | None = None,
) -> Callable[
    [_CallableWithStateAndRuntime[StateT, ContextT]], AgentMiddleware[StateT, ContextT]
]: ...


def before_model(
    func: _CallableWithStateAndRuntime[StateT, ContextT] | None = None,
    *,
    state_schema: type[StateT] | None = None,
    tools: list[BaseTool] | None = None,
    can_jump_to: list[JumpTo] | None = None,
    name: str | None = None,
) -> (
    Callable[[_CallableWithStateAndRuntime[StateT, ContextT]], AgentMiddleware[StateT, ContextT]]
    | AgentMiddleware[StateT, ContextT]
):
    """Decorator used to dynamically create a middleware with the `before_model` hook.

    Args:
        func: The function to be decorated.

            Must accept: `state: StateT, runtime: Runtime[ContextT]` - State and runtime
                context
        state_schema: Optional custom state schema type.

            If not provided, uses the default `AgentState` schema.
        tools: Optional list of additional tools to register with this middleware.
        can_jump_to: Optional list of valid jump destinations for conditional edges.

            Valid values are: `'tools'`, `'model'`, `'end'`
        name: Optional name for the generated middleware class.

            If not provided, uses the decorated function's name.

    Returns:
        Either an `AgentMiddleware` instance (if func is provided directly) or a
            decorator function that can be applied to a function it is wrapping.

    The decorated function should return:

    - `dict[str, Any]` - State updates to merge into the agent state
    - `Command` - A command to control flow (e.g., jump to different node)
    - `None` - No state updates or flow control

    Examples:
        !!! example "Basic usage"

            ```python
            @before_model
            def log_before_model(state: AgentState, runtime: Runtime) -> None:
                print(f"About to call model with {len(state['messages'])} messages")
            ```

        !!! example "With conditional jumping"

            ```python
            @before_model(can_jump_to=["end"])
            def conditional_before_model(
                state: AgentState, runtime: Runtime
            ) -> dict[str, Any] | None:
                if some_condition(state):
                    return {"jump_to": "end"}
                return None
            ```

        !!! example "With custom state schema"

            ```python
            @before_model(state_schema=MyCustomState)
            def custom_before_model(state: MyCustomState, runtime: Runtime) -> dict[str, Any]:
                return {"custom_field": "updated_value"}
            ```

        !!! example "Streaming custom events before model call"

            Use `runtime.stream_writer` to emit custom events before each model invocation.
            Events are received when streaming with `stream_mode="custom"`.

            ```python
            @before_model
            async def notify_model_call(state: AgentState, runtime: Runtime) -> None:
                '''Notify user before model is called.'''
                runtime.stream_writer(
                    {
                        "type": "status",
                        "message": "Thinking...",
                    }
                )
            ```
    

    中文翻译:
    装饰器用于通过“before_model”钩子动态创建中间件。
    参数：
        func：要装饰的函数。
            必须接受：`state：StateT，runtime：Runtime[ContextT]` - 状态和运行时
                上下文
        state_schema：可选的自定义状态模式类型。
            如果未提供，则使用默认的“AgentState”架构。
        工具：向此中间件注册的附加工具的可选列表。
        can_jump_to：条件边的有效跳转目标的可选列表。
            有效值为：“工具”、“模型”、“结束”
        name：生成的中间件类的可选名称。
            如果未提供，则使用修饰函数的名称。
    返回：
        要么是 `AgentMiddleware` 实例（如果直接提供 func），要么是
            可以应用于它所包装的函数的装饰器函数。
    装饰函数应该返回：
    - `dict[str, Any]` - 状态更新以合并到代理状态
    - `Command` - 控制流程的命令（例如，跳转到不同的节点）
    - `None` - 无状态更新或流量控制
    示例：
        !!!示例“基本用法”
            ````蟒蛇
            @之前_模型
            def log_before_model(状态: AgentState, 运行时: 运行时) -> 无:
                print(f"即将使用 {len(state['messages'])} 消息调用模型")
            ````
        !!!例如“有条件跳转”
            ````蟒蛇
            @before_model(can_jump_to=[“结束”])
            def 条件模型之前（
                状态：代理状态，运行时：运行时
            ) -> dict[str, 任意] |无：
                如果某些_条件（状态）：
                    返回 {"jump_to": "end"}
                返回无
            ````
        !!!示例“使用自定义状态模式”
            ````蟒蛇
            @before_model(state_schema=MyCustomState)
            def custom_before_model(状态: MyCustomState, 运行时: Runtime) -> dict[str, Any]:
                返回 {"custom_field": "updated_value"}
            ````
        !!!示例“在模型调用之前流式传输自定义事件”
            使用“runtime.stream_writer”在每次模型调用之前发出自定义事件。
            使用 `stream_mode="custom"` 进行流式传输时会收到事件。
            ````蟒蛇
            @之前_模型
            async def notification_model_call(state: AgentState, runtime: Runtime) -> None:
                '''在调用模型之前通知用户。'''
                运行时.stream_writer(
                    {
                        “类型”：“状态”，
                        "message": "思考...",
                    }
                ）
            ````"""

    def decorator(
        func: _CallableWithStateAndRuntime[StateT, ContextT],
    ) -> AgentMiddleware[StateT, ContextT]:
        is_async = iscoroutinefunction(func)

        func_can_jump_to = (
            can_jump_to if can_jump_to is not None else getattr(func, "__can_jump_to__", [])
        )

        if is_async:

            async def async_wrapped(
                _self: AgentMiddleware[StateT, ContextT],
                state: StateT,
                runtime: Runtime[ContextT],
            ) -> dict[str, Any] | Command | None:
                return await func(state, runtime)  # type: ignore[misc]

            # Preserve can_jump_to metadata on the wrapped function
            # 中文: 保留包装函数上的 can_jump_to 元数据
            if func_can_jump_to:
                async_wrapped.__can_jump_to__ = func_can_jump_to  # type: ignore[attr-defined]

            middleware_name = name or cast(
                "str", getattr(func, "__name__", "BeforeModelMiddleware")
            )

            return type(
                middleware_name,
                (AgentMiddleware,),
                {
                    "state_schema": state_schema or AgentState,
                    "tools": tools or [],
                    "abefore_model": async_wrapped,
                },
            )()

        def wrapped(
            _self: AgentMiddleware[StateT, ContextT],
            state: StateT,
            runtime: Runtime[ContextT],
        ) -> dict[str, Any] | Command | None:
            return func(state, runtime)  # type: ignore[return-value]

        # Preserve can_jump_to metadata on the wrapped function
        # 中文: 保留包装函数上的 can_jump_to 元数据
        if func_can_jump_to:
            wrapped.__can_jump_to__ = func_can_jump_to  # type: ignore[attr-defined]

        # Use function name as default if no name provided
        # 中文: 如果未提供名称，则使用函数名称作为默认值
        middleware_name = name or cast("str", getattr(func, "__name__", "BeforeModelMiddleware"))

        return type(
            middleware_name,
            (AgentMiddleware,),
            {
                "state_schema": state_schema or AgentState,
                "tools": tools or [],
                "before_model": wrapped,
            },
        )()

    if func is not None:
        return decorator(func)
    return decorator


@overload
def after_model(
    func: _CallableWithStateAndRuntime[StateT, ContextT],
) -> AgentMiddleware[StateT, ContextT]: ...


@overload
def after_model(
    func: None = None,
    *,
    state_schema: type[StateT] | None = None,
    tools: list[BaseTool] | None = None,
    can_jump_to: list[JumpTo] | None = None,
    name: str | None = None,
) -> Callable[
    [_CallableWithStateAndRuntime[StateT, ContextT]], AgentMiddleware[StateT, ContextT]
]: ...


def after_model(
    func: _CallableWithStateAndRuntime[StateT, ContextT] | None = None,
    *,
    state_schema: type[StateT] | None = None,
    tools: list[BaseTool] | None = None,
    can_jump_to: list[JumpTo] | None = None,
    name: str | None = None,
) -> (
    Callable[[_CallableWithStateAndRuntime[StateT, ContextT]], AgentMiddleware[StateT, ContextT]]
    | AgentMiddleware[StateT, ContextT]
):
    """Decorator used to dynamically create a middleware with the `after_model` hook.

    Args:
        func: The function to be decorated.

            Must accept: `state: StateT, runtime: Runtime[ContextT]` - State and runtime
            context
        state_schema: Optional custom state schema type.

            If not provided, uses the default `AgentState` schema.
        tools: Optional list of additional tools to register with this middleware.
        can_jump_to: Optional list of valid jump destinations for conditional edges.

            Valid values are: `'tools'`, `'model'`, `'end'`
        name: Optional name for the generated middleware class.

            If not provided, uses the decorated function's name.

    Returns:
        Either an `AgentMiddleware` instance (if func is provided) or a decorator
            function that can be applied to a function.

    The decorated function should return:

    - `dict[str, Any]` - State updates to merge into the agent state
    - `Command` - A command to control flow (e.g., jump to different node)
    - `None` - No state updates or flow control

    Examples:
        !!! example "Basic usage for logging model responses"

            ```python
            @after_model
            def log_latest_message(state: AgentState, runtime: Runtime) -> None:
                print(state["messages"][-1].content)
            ```

        !!! example "With custom state schema"

            ```python
            @after_model(state_schema=MyCustomState, name="MyAfterModelMiddleware")
            def custom_after_model(state: MyCustomState, runtime: Runtime) -> dict[str, Any]:
                return {"custom_field": "updated_after_model"}
            ```

        !!! example "Streaming custom events after model call"

            Use `runtime.stream_writer` to emit custom events after model responds.
            Events are received when streaming with `stream_mode="custom"`.

            ```python
            @after_model
            async def notify_model_response(state: AgentState, runtime: Runtime) -> None:
                '''Notify user after model has responded.'''
                last_message = state["messages"][-1]
                has_tool_calls = hasattr(last_message, "tool_calls") and last_message.tool_calls
                runtime.stream_writer(
                    {
                        "type": "status",
                        "message": "Using tools..." if has_tool_calls else "Response ready!",
                    }
                )
            ```
    

    中文翻译:
    装饰器用于通过“after_model”钩子动态创建中间件。
    参数：
        func：要装饰的函数。
            必须接受：`state：StateT，runtime：Runtime[ContextT]` - 状态和运行时
            上下文
        state_schema：可选的自定义状态模式类型。
            如果未提供，则使用默认的“AgentState”架构。
        工具：向此中间件注册的附加工具的可选列表。
        can_jump_to：条件边的有效跳转目标的可选列表。
            有效值为：“工具”、“模型”、“结束”
        name：生成的中间件类的可选名称。
            如果未提供，则使用修饰函数的名称。
    返回：
        一个 `AgentMiddleware` 实例（如果提供了 func）或者一个装饰器
            可以应用于函数的函数。
    装饰函数应该返回：
    - `dict[str, Any]` - 状态更新以合并到代理状态
    - `Command` - 控制流程的命令（例如，跳转到不同的节点）
    - `None` - 无状态更新或流量控制
    示例：
        !!!示例“记录模型响应的基本用法”
            ````蟒蛇
            @after_model
            def log_latest_message(状态: AgentState, 运行时: 运行时) -> 无:
                打印（状态[“消息”][-1].内容）
            ````
        !!!示例“使用自定义状态模式”
            ````蟒蛇
            @after_model(state_schema=MyCustomState, name="MyAfterModelMiddleware")
            def custom_after_model(状态: MyCustomState, 运行时: Runtime) -> dict[str, Any]:
                返回{“custom_field”：“updated_after_model”}
            ````
        !!!示例“模型调用后流式传输自定义事件”
            使用“runtime.stream_writer”在模型响应后发出自定义事件。
            使用 `stream_mode="custom"` 进行流式传输时会收到事件。
            ````蟒蛇
            @after_model
            async def notification_model_response(state: AgentState, runtime: Runtime) -> None:
                '''模型响应后通知用户。'''
                最后一条消息 = 状态["消息"][-1]
                has_tool_calls = hasattr(last_message, "tool_calls") 和 last_message.tool_calls
                运行时.stream_writer(
                    {
                        “类型”：“状态”，
                        "message": "正在使用工具..." if has_tool_calls else "响应就绪！",
                    }
                ）
            ````"""

    def decorator(
        func: _CallableWithStateAndRuntime[StateT, ContextT],
    ) -> AgentMiddleware[StateT, ContextT]:
        is_async = iscoroutinefunction(func)
        # Extract can_jump_to from decorator parameter or from function metadata
        # 中文: 从装饰器参数或函数元数据中提取 can_jump_to
        func_can_jump_to = (
            can_jump_to if can_jump_to is not None else getattr(func, "__can_jump_to__", [])
        )

        if is_async:

            async def async_wrapped(
                _self: AgentMiddleware[StateT, ContextT],
                state: StateT,
                runtime: Runtime[ContextT],
            ) -> dict[str, Any] | Command | None:
                return await func(state, runtime)  # type: ignore[misc]

            # Preserve can_jump_to metadata on the wrapped function
            # 中文: 保留包装函数上的 can_jump_to 元数据
            if func_can_jump_to:
                async_wrapped.__can_jump_to__ = func_can_jump_to  # type: ignore[attr-defined]

            middleware_name = name or cast("str", getattr(func, "__name__", "AfterModelMiddleware"))

            return type(
                middleware_name,
                (AgentMiddleware,),
                {
                    "state_schema": state_schema or AgentState,
                    "tools": tools or [],
                    "aafter_model": async_wrapped,
                },
            )()

        def wrapped(
            _self: AgentMiddleware[StateT, ContextT],
            state: StateT,
            runtime: Runtime[ContextT],
        ) -> dict[str, Any] | Command | None:
            return func(state, runtime)  # type: ignore[return-value]

        # Preserve can_jump_to metadata on the wrapped function
        # 中文: 保留包装函数上的 can_jump_to 元数据
        if func_can_jump_to:
            wrapped.__can_jump_to__ = func_can_jump_to  # type: ignore[attr-defined]

        # Use function name as default if no name provided
        # 中文: 如果未提供名称，则使用函数名称作为默认值
        middleware_name = name or cast("str", getattr(func, "__name__", "AfterModelMiddleware"))

        return type(
            middleware_name,
            (AgentMiddleware,),
            {
                "state_schema": state_schema or AgentState,
                "tools": tools or [],
                "after_model": wrapped,
            },
        )()

    if func is not None:
        return decorator(func)
    return decorator


@overload
def before_agent(
    func: _CallableWithStateAndRuntime[StateT, ContextT],
) -> AgentMiddleware[StateT, ContextT]: ...


@overload
def before_agent(
    func: None = None,
    *,
    state_schema: type[StateT] | None = None,
    tools: list[BaseTool] | None = None,
    can_jump_to: list[JumpTo] | None = None,
    name: str | None = None,
) -> Callable[
    [_CallableWithStateAndRuntime[StateT, ContextT]], AgentMiddleware[StateT, ContextT]
]: ...


def before_agent(
    func: _CallableWithStateAndRuntime[StateT, ContextT] | None = None,
    *,
    state_schema: type[StateT] | None = None,
    tools: list[BaseTool] | None = None,
    can_jump_to: list[JumpTo] | None = None,
    name: str | None = None,
) -> (
    Callable[[_CallableWithStateAndRuntime[StateT, ContextT]], AgentMiddleware[StateT, ContextT]]
    | AgentMiddleware[StateT, ContextT]
):
    """Decorator used to dynamically create a middleware with the `before_agent` hook.

    Args:
        func: The function to be decorated.

            Must accept: `state: StateT, runtime: Runtime[ContextT]` - State and runtime
            context
        state_schema: Optional custom state schema type.

            If not provided, uses the default `AgentState` schema.
        tools: Optional list of additional tools to register with this middleware.
        can_jump_to: Optional list of valid jump destinations for conditional edges.

            Valid values are: `'tools'`, `'model'`, `'end'`
        name: Optional name for the generated middleware class.

            If not provided, uses the decorated function's name.

    Returns:
        Either an `AgentMiddleware` instance (if func is provided directly) or a
            decorator function that can be applied to a function it is wrapping.

    The decorated function should return:

    - `dict[str, Any]` - State updates to merge into the agent state
    - `Command` - A command to control flow (e.g., jump to different node)
    - `None` - No state updates or flow control

    Examples:
        !!! example "Basic usage"

            ```python
            @before_agent
            def log_before_agent(state: AgentState, runtime: Runtime) -> None:
                print(f"Starting agent with {len(state['messages'])} messages")
            ```

        !!! example "With conditional jumping"

            ```python
            @before_agent(can_jump_to=["end"])
            def conditional_before_agent(
                state: AgentState, runtime: Runtime
            ) -> dict[str, Any] | None:
                if some_condition(state):
                    return {"jump_to": "end"}
                return None
            ```

        !!! example "With custom state schema"

            ```python
            @before_agent(state_schema=MyCustomState)
            def custom_before_agent(state: MyCustomState, runtime: Runtime) -> dict[str, Any]:
                return {"custom_field": "initialized_value"}
            ```

        !!! example "Streaming custom events"

            Use `runtime.stream_writer` to emit custom events during agent execution.
            Events are received when streaming with `stream_mode="custom"`.

            ```python
            from langchain.agents import create_agent
            from langchain.agents.middleware import before_agent, AgentState
            from langchain.messages import HumanMessage
            from langgraph.runtime import Runtime


            @before_agent
            async def notify_start(state: AgentState, runtime: Runtime) -> None:
                '''Notify user that agent is starting.'''
                runtime.stream_writer(
                    {
                        "type": "status",
                        "message": "Initializing agent session...",
                    }
                )
                # Perform prerequisite tasks here
                # 中文: Perform prerequisite tasks here
                runtime.stream_writer({"type": "status", "message": "Agent ready!"})


            agent = create_agent(
                model="openai:gpt-5.2",
                tools=[...],
                middleware=[notify_start],
            )

            # Consume with stream_mode="custom" to receive events
            # 中文: 使用stream_mode =“custom”来接收事件
            async for mode, event in agent.astream(
                {"messages": [HumanMessage("Hello")]},
                stream_mode=["updates", "custom"],
            ):
                if mode == "custom":
                    print(f"Status: {event}")
            ```
    

    中文翻译:
    装饰器用于通过“before_agent”钩子动态创建中间件。
    参数：
        func：要装饰的函数。
            必须接受：`state：StateT，runtime：Runtime[ContextT]` - 状态和运行时
            上下文
        state_schema：可选的自定义状态模式类型。
            如果未提供，则使用默认的“AgentState”架构。
        工具：向此中间件注册的附加工具的可选列表。
        can_jump_to：条件边的有效跳转目标的可选列表。
            有效值为：“工具”、“模型”、“结束”
        name：生成的中间件类的可选名称。
            如果未提供，则使用修饰函数的名称。
    返回：
        要么是 `AgentMiddleware` 实例（如果直接提供 func），要么是
            可以应用于它所包装的函数的装饰器函数。
    装饰函数应该返回：
    - `dict[str, Any]` - 状态更新以合并到代理状态
    - `Command` - 控制流程的命令（例如，跳转到不同的节点）
    - `None` - 无状态更新或流量控制
    示例：
        !!!示例“基本用法”
            ````蟒蛇
            @before_agent
            def log_before_agent(状态: AgentState, 运行时: 运行时) -> 无:
                print(f"使用 {len(state['messages'])} 消息启动代理")
            ````
        !!!例如“有条件跳转”
            ````蟒蛇
            @before_agent(can_jump_to=[“结束”])
            def conditional_before_agent(
                状态：代理状态，运行时：运行时
            ) -> dict[str, 任意] |无：
                如果某些_条件（状态）：
                    返回 {"jump_to": "end"}
                返回无
            ````
        !!!示例“使用自定义状态模式”
            ````蟒蛇
            @before_agent(state_schema=MyCustomState)
            def custom_before_agent(状态: MyCustomState, 运行时: Runtime) -> dict[str, Any]:
                返回 {"custom_field": "initialized_value"}
            ````
        !!!示例“流式传输自定义事件”
            使用“runtime.stream_writer”在代理执行期间发出自定义事件。
            使用 `stream_mode="custom"` 进行流式传输时会收到事件。
            ````蟒蛇
            从 langchain.agents 导入 create_agent
            从 langchain.agents.middleware 导入 before_agent, AgentState
            从 langchain.messages 导入 HumanMessage
            从 langgraph.runtime 导入运行时
            @before_agent
            async def notification_start(state: AgentState, runtime: Runtime) -> None:
                '''通知用户代理正在启动。'''
                运行时.stream_writer(
                    {
                        “类型”：“状态”，
                        "message": "正在初始化代理会话...",
                    }
                ）
                # 在此执行先决任务
                runtime.stream_writer({"type": "status", "message": "代理准备就绪！"})
            代理=创建_代理（
                型号=“openai：gpt-5.2”，
                工具=[...],
                中间件=[notify_start],
            ）
            # 使用stream_mode =“custom”来接收事件
            异步模式，agent.astream 中的事件（
                {“消息”：[HumanMessage（“你好”）]}，
                Stream_mode=["更新", "自定义"],
            ）：
                如果模式==“自定义”：
                    print(f"状态：{事件}")
            ````"""

    def decorator(
        func: _CallableWithStateAndRuntime[StateT, ContextT],
    ) -> AgentMiddleware[StateT, ContextT]:
        is_async = iscoroutinefunction(func)

        func_can_jump_to = (
            can_jump_to if can_jump_to is not None else getattr(func, "__can_jump_to__", [])
        )

        if is_async:

            async def async_wrapped(
                _self: AgentMiddleware[StateT, ContextT],
                state: StateT,
                runtime: Runtime[ContextT],
            ) -> dict[str, Any] | Command | None:
                return await func(state, runtime)  # type: ignore[misc]

            # Preserve can_jump_to metadata on the wrapped function
            # 中文: 保留包装函数上的 can_jump_to 元数据
            if func_can_jump_to:
                async_wrapped.__can_jump_to__ = func_can_jump_to  # type: ignore[attr-defined]

            middleware_name = name or cast(
                "str", getattr(func, "__name__", "BeforeAgentMiddleware")
            )

            return type(
                middleware_name,
                (AgentMiddleware,),
                {
                    "state_schema": state_schema or AgentState,
                    "tools": tools or [],
                    "abefore_agent": async_wrapped,
                },
            )()

        def wrapped(
            _self: AgentMiddleware[StateT, ContextT],
            state: StateT,
            runtime: Runtime[ContextT],
        ) -> dict[str, Any] | Command | None:
            return func(state, runtime)  # type: ignore[return-value]

        # Preserve can_jump_to metadata on the wrapped function
        # 中文: 保留包装函数上的 can_jump_to 元数据
        if func_can_jump_to:
            wrapped.__can_jump_to__ = func_can_jump_to  # type: ignore[attr-defined]

        # Use function name as default if no name provided
        # 中文: 如果未提供名称，则使用函数名称作为默认值
        middleware_name = name or cast("str", getattr(func, "__name__", "BeforeAgentMiddleware"))

        return type(
            middleware_name,
            (AgentMiddleware,),
            {
                "state_schema": state_schema or AgentState,
                "tools": tools or [],
                "before_agent": wrapped,
            },
        )()

    if func is not None:
        return decorator(func)
    return decorator


@overload
def after_agent(
    func: _CallableWithStateAndRuntime[StateT, ContextT],
) -> AgentMiddleware[StateT, ContextT]: ...


@overload
def after_agent(
    func: None = None,
    *,
    state_schema: type[StateT] | None = None,
    tools: list[BaseTool] | None = None,
    can_jump_to: list[JumpTo] | None = None,
    name: str | None = None,
) -> Callable[
    [_CallableWithStateAndRuntime[StateT, ContextT]], AgentMiddleware[StateT, ContextT]
]: ...


def after_agent(
    func: _CallableWithStateAndRuntime[StateT, ContextT] | None = None,
    *,
    state_schema: type[StateT] | None = None,
    tools: list[BaseTool] | None = None,
    can_jump_to: list[JumpTo] | None = None,
    name: str | None = None,
) -> (
    Callable[[_CallableWithStateAndRuntime[StateT, ContextT]], AgentMiddleware[StateT, ContextT]]
    | AgentMiddleware[StateT, ContextT]
):
    """Decorator used to dynamically create a middleware with the `after_agent` hook.

    Async version is `aafter_agent`.

    Args:
        func: The function to be decorated.

            Must accept: `state: StateT, runtime: Runtime[ContextT]` - State and runtime
            context
        state_schema: Optional custom state schema type.

            If not provided, uses the default `AgentState` schema.
        tools: Optional list of additional tools to register with this middleware.
        can_jump_to: Optional list of valid jump destinations for conditional edges.

            Valid values are: `'tools'`, `'model'`, `'end'`
        name: Optional name for the generated middleware class.

            If not provided, uses the decorated function's name.

    Returns:
        Either an `AgentMiddleware` instance (if func is provided) or a decorator
            function that can be applied to a function.

    The decorated function should return:

    - `dict[str, Any]` - State updates to merge into the agent state
    - `Command` - A command to control flow (e.g., jump to different node)
    - `None` - No state updates or flow control

    Examples:
        !!! example "Basic usage for logging agent completion"

            ```python
            @after_agent
            def log_completion(state: AgentState, runtime: Runtime) -> None:
                print(f"Agent completed with {len(state['messages'])} messages")
            ```

        !!! example "With custom state schema"

            ```python
            @after_agent(state_schema=MyCustomState, name="MyAfterAgentMiddleware")
            def custom_after_agent(state: MyCustomState, runtime: Runtime) -> dict[str, Any]:
                return {"custom_field": "finalized_value"}
            ```

        !!! example "Streaming custom events on completion"

            Use `runtime.stream_writer` to emit custom events when agent completes.
            Events are received when streaming with `stream_mode="custom"`.

            ```python
            @after_agent
            async def notify_completion(state: AgentState, runtime: Runtime) -> None:
                '''Notify user that agent has completed.'''
                runtime.stream_writer(
                    {
                        "type": "status",
                        "message": "Agent execution complete!",
                        "total_messages": len(state["messages"]),
                    }
                )
            ```
    

    中文翻译:
    装饰器用于通过“after_agent”钩子动态创建中间件。
    异步版本是“aafter_agent”。
    参数：
        func：要装饰的函数。
            必须接受：`state：StateT，runtime：Runtime[ContextT]` - 状态和运行时
            上下文
        state_schema：可选的自定义状态模式类型。
            如果未提供，则使用默认的“AgentState”架构。
        工具：向此中间件注册的附加工具的可选列表。
        can_jump_to：条件边的有效跳转目标的可选列表。
            有效值为：“工具”、“模型”、“结束”
        name：生成的中间件类的可选名称。
            如果未提供，则使用修饰函数的名称。
    返回：
        一个 `AgentMiddleware` 实例（如果提供了 func）或者一个装饰器
            可以应用于函数的函数。
    装饰函数应该返回：
    - `dict[str, Any]` - 状态更新以合并到代理状态
    - `Command` - 控制流程的命令（例如，跳转到不同的节点）
    - `None` - 无状态更新或流量控制
    示例：
        !!!示例“记录代理完成的基本用法”
            ````蟒蛇
            @after_agent
            def log_completion(状态: AgentState, 运行时: 运行时) -> 无:
                print(f"代理已完成 {len(state['messages'])} 消息")
            ````
        !!!示例“使用自定义状态模式”
            ````蟒蛇
            @after_agent(state_schema=MyCustomState, name="MyAfterAgentMiddleware")
            def custom_after_agent(状态: MyCustomState, 运行时: Runtime) -> dict[str, Any]:
                返回 {"custom_field": "finalized_value"}
            ````
        !!!示例“完成时流式传输自定义事件”
            使用“runtime.stream_writer”在代理完成时发出自定义事件。
            使用 `stream_mode="custom"` 进行流式传输时会收到事件。
            ````蟒蛇
            @after_agent
            async def notification_completion(state: AgentState, runtime: Runtime) -> None:
                '''通知用户代理已完成。'''
                运行时.stream_writer(
                    {
                        “类型”：“状态”，
                        "message": "代理执行完成！",
                        “total_messages”：len（状态[“消息”]），
                    }
                ）
            ````"""

    def decorator(
        func: _CallableWithStateAndRuntime[StateT, ContextT],
    ) -> AgentMiddleware[StateT, ContextT]:
        is_async = iscoroutinefunction(func)
        # Extract can_jump_to from decorator parameter or from function metadata
        # 中文: 从装饰器参数或函数元数据中提取 can_jump_to
        func_can_jump_to = (
            can_jump_to if can_jump_to is not None else getattr(func, "__can_jump_to__", [])
        )

        if is_async:

            async def async_wrapped(
                _self: AgentMiddleware[StateT, ContextT],
                state: StateT,
                runtime: Runtime[ContextT],
            ) -> dict[str, Any] | Command | None:
                return await func(state, runtime)  # type: ignore[misc]

            # Preserve can_jump_to metadata on the wrapped function
            # 中文: 保留包装函数上的 can_jump_to 元数据
            if func_can_jump_to:
                async_wrapped.__can_jump_to__ = func_can_jump_to  # type: ignore[attr-defined]

            middleware_name = name or cast("str", getattr(func, "__name__", "AfterAgentMiddleware"))

            return type(
                middleware_name,
                (AgentMiddleware,),
                {
                    "state_schema": state_schema or AgentState,
                    "tools": tools or [],
                    "aafter_agent": async_wrapped,
                },
            )()

        def wrapped(
            _self: AgentMiddleware[StateT, ContextT],
            state: StateT,
            runtime: Runtime[ContextT],
        ) -> dict[str, Any] | Command | None:
            return func(state, runtime)  # type: ignore[return-value]

        # Preserve can_jump_to metadata on the wrapped function
        # 中文: 保留包装函数上的 can_jump_to 元数据
        if func_can_jump_to:
            wrapped.__can_jump_to__ = func_can_jump_to  # type: ignore[attr-defined]

        # Use function name as default if no name provided
        # 中文: 如果未提供名称，则使用函数名称作为默认值
        middleware_name = name or cast("str", getattr(func, "__name__", "AfterAgentMiddleware"))

        return type(
            middleware_name,
            (AgentMiddleware,),
            {
                "state_schema": state_schema or AgentState,
                "tools": tools or [],
                "after_agent": wrapped,
            },
        )()

    if func is not None:
        return decorator(func)
    return decorator


@overload
def dynamic_prompt(
    func: _CallableReturningSystemMessage[StateT, ContextT],
) -> AgentMiddleware[StateT, ContextT]: ...


@overload
def dynamic_prompt(
    func: None = None,
) -> Callable[
    [_CallableReturningSystemMessage[StateT, ContextT]],
    AgentMiddleware[StateT, ContextT],
]: ...


def dynamic_prompt(
    func: _CallableReturningSystemMessage[StateT, ContextT] | None = None,
) -> (
    Callable[
        [_CallableReturningSystemMessage[StateT, ContextT]],
        AgentMiddleware[StateT, ContextT],
    ]
    | AgentMiddleware[StateT, ContextT]
):
    """Decorator used to dynamically generate system prompts for the model.

    This is a convenience decorator that creates middleware using `wrap_model_call`
    specifically for dynamic prompt generation. The decorated function should return
    a string that will be set as the system prompt for the model request.

    Args:
        func: The function to be decorated.

            Must accept: `request: ModelRequest` - Model request (contains state and
            runtime)

    Returns:
        Either an `AgentMiddleware` instance (if func is provided) or a decorator
            function that can be applied to a function.

    The decorated function should return:
        - `str` – The system prompt string to use for the model request
        - `SystemMessage` – A complete system message to use for the model request

    Examples:
        Basic usage with dynamic content:

        ```python
        @dynamic_prompt
        def my_prompt(request: ModelRequest) -> str:
            user_name = request.runtime.context.get("user_name", "User")
            return f"You are a helpful assistant helping {user_name}."
        ```

        Using state to customize the prompt:

        ```python
        @dynamic_prompt
        def context_aware_prompt(request: ModelRequest) -> str:
            msg_count = len(request.state["messages"])
            if msg_count > 10:
                return "You are in a long conversation. Be concise."
            return "You are a helpful assistant."
        ```

        Using with agent:

        ```python
        agent = create_agent(model, middleware=[my_prompt])
        ```
    

    中文翻译:
    装饰器用于为模型动态生成系统提示。
    这是一个方便的装饰器，使用“wrap_model_call”创建中间件
    专门用于动态提示生成。装饰函数应该返回
    将设置为模型请求的系统提示的字符串。
    参数：
        func：要装饰的函数。
            必须接受：`request: ModelRequest` - 模型请求（包含状态和
            运行时）
    返回：
        一个 `AgentMiddleware` 实例（如果提供了 func）或者一个装饰器
            可以应用于函数的函数。
    装饰函数应该返回：
        - `str` – 用于模型请求的系统提示字符串
        - `SystemMessage` – 用于模型请求的完整系统消息
    示例：
        动态内容的基本用法：
        ````蟒蛇
        @动态提示
        def my_prompt(请求: ModelRequest) -> str:
            user_name = request.runtime.context.get("user_name", "用户")
            return f“您是帮助 {user_name} 的有用助手。”
        ````
        使用状态来自定义提示：
        ````蟒蛇
        @动态提示
        def context_aware_prompt(请求: ModelRequest) -> str:
            msg_count = len(request.state["消息"])
            如果 msg_count > 10：
                return “您正在进行一个很长的对话。请简明扼要。”
            return “你是一个有用的助手。”
        ````
        与代理一起使用：
        ````蟒蛇
        代理= create_agent（模型，中间件= [my_prompt]）
        ````"""

    def decorator(
        func: _CallableReturningSystemMessage[StateT, ContextT],
    ) -> AgentMiddleware[StateT, ContextT]:
        is_async = iscoroutinefunction(func)

        if is_async:

            async def async_wrapped(
                _self: AgentMiddleware[StateT, ContextT],
                request: ModelRequest,
                handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
            ) -> ModelCallResult:
                prompt = await func(request)  # type: ignore[misc]
                if isinstance(prompt, SystemMessage):
                    request = request.override(system_message=prompt)
                else:
                    request = request.override(system_message=SystemMessage(content=prompt))
                return await handler(request)

            middleware_name = cast("str", getattr(func, "__name__", "DynamicPromptMiddleware"))

            return type(
                middleware_name,
                (AgentMiddleware,),
                {
                    "state_schema": AgentState,
                    "tools": [],
                    "awrap_model_call": async_wrapped,
                },
            )()

        def wrapped(
            _self: AgentMiddleware[StateT, ContextT],
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse],
        ) -> ModelCallResult:
            prompt = cast("Callable[[ModelRequest], SystemMessage | str]", func)(request)
            if isinstance(prompt, SystemMessage):
                request = request.override(system_message=prompt)
            else:
                request = request.override(system_message=SystemMessage(content=prompt))
            return handler(request)

        async def async_wrapped_from_sync(
            _self: AgentMiddleware[StateT, ContextT],
            request: ModelRequest,
            handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
        ) -> ModelCallResult:
            # Delegate to sync function
            # 中文: 委托同步功能
            prompt = cast("Callable[[ModelRequest], SystemMessage | str]", func)(request)
            if isinstance(prompt, SystemMessage):
                request = request.override(system_message=prompt)
            else:
                request = request.override(system_message=SystemMessage(content=prompt))
            return await handler(request)

        middleware_name = cast("str", getattr(func, "__name__", "DynamicPromptMiddleware"))

        return type(
            middleware_name,
            (AgentMiddleware,),
            {
                "state_schema": AgentState,
                "tools": [],
                "wrap_model_call": wrapped,
                "awrap_model_call": async_wrapped_from_sync,
            },
        )()

    if func is not None:
        return decorator(func)
    return decorator


@overload
def wrap_model_call(
    func: _CallableReturningModelResponse[StateT, ContextT],
) -> AgentMiddleware[StateT, ContextT]: ...


@overload
def wrap_model_call(
    func: None = None,
    *,
    state_schema: type[StateT] | None = None,
    tools: list[BaseTool] | None = None,
    name: str | None = None,
) -> Callable[
    [_CallableReturningModelResponse[StateT, ContextT]],
    AgentMiddleware[StateT, ContextT],
]: ...


def wrap_model_call(
    func: _CallableReturningModelResponse[StateT, ContextT] | None = None,
    *,
    state_schema: type[StateT] | None = None,
    tools: list[BaseTool] | None = None,
    name: str | None = None,
) -> (
    Callable[
        [_CallableReturningModelResponse[StateT, ContextT]],
        AgentMiddleware[StateT, ContextT],
    ]
    | AgentMiddleware[StateT, ContextT]
):
    """Create middleware with `wrap_model_call` hook from a function.

    Converts a function with handler callback into middleware that can intercept model
    calls, implement retry logic, handle errors, and rewrite responses.

    Args:
        func: Function accepting (request, handler) that calls handler(request)
            to execute the model and returns `ModelResponse` or `AIMessage`.

            Request contains state and runtime.
        state_schema: Custom state schema.

            Defaults to `AgentState`.
        tools: Additional tools to register with this middleware.
        name: Middleware class name.

            Defaults to function name.

    Returns:
        `AgentMiddleware` instance if func provided, otherwise a decorator.

    Examples:
        !!! example "Basic retry logic"

            ```python
            @wrap_model_call
            def retry_on_error(request, handler):
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        return handler(request)
                    except Exception:
                        if attempt == max_retries - 1:
                            raise
            ```

        !!! example "Model fallback"

            ```python
            @wrap_model_call
            def fallback_model(request, handler):
                # Try primary model
                # 中文: 尝试主要模型
                try:
                    return handler(request)
                except Exception:
                    pass

                # Try fallback model
                # 中文: 尝试后备模型
                request = request.override(model=fallback_model_instance)
                return handler(request)
            ```

        !!! example "Rewrite response content (full `ModelResponse`)"

            ```python
            @wrap_model_call
            def uppercase_responses(request, handler):
                response = handler(request)
                ai_msg = response.result[0]
                return ModelResponse(
                    result=[AIMessage(content=ai_msg.content.upper())],
                    structured_response=response.structured_response,
                )
            ```

        !!! example "Simple `AIMessage` return (converted automatically)"

            ```python
            @wrap_model_call
            def simple_response(request, handler):
                # AIMessage is automatically converted to ModelResponse
                # 中文: AIMessage自动转换为ModelResponse
                return AIMessage(content="Simple response")
            ```
    

    中文翻译:
    使用函数中的“wrap_model_call”挂钩创建中间件。
    将具有处理程序回调的函数转换为可以拦截模型的中间件
    调用、实现重试逻辑、处理错误和重写响应。
    参数：
        func：接受（请求，处理程序）的函数，调用处理程序（请求）
            执行模型并返回“ModelResponse”或“AIMessage”。
            请求包含状态和运行时。
        state_schema：自定义状态模式。
            默认为“AgentState”。
        工具：向此中间件注册的附加工具。
        name：中间件类名。
            默认为函数名称。
    返回：
        如果提供了 func，则为 `AgentMiddleware` 实例，否则为装饰器。
    示例：
        !!!示例“基本重试逻辑”
            ````蟒蛇
            @wrap_model_call
            def retry_on_error(请求, 处理程序):
                最大重试次数 = 3
                对于范围内的尝试（max_retries）：
                    尝试：
                        返回处理程序（请求）
                    除了例外：
                        如果尝试 == max_retries - 1：
                            提高
            ````
        !!!示例“模型后备”
            ````蟒蛇
            @wrap_model_call
            deffallback_model（请求，处理程序）：
                # 尝试主要模型
                尝试：
                    返回处理程序（请求）
                除了例外：
                    通过
                # 尝试后备模型
                请求 = request.override(model=fallback_model_instance)
                返回处理程序（请求）
            ````
        !!!示例“重写响应内容（完整的`ModelResponse`）”
            ````蟒蛇
            @wrap_model_call
            def uppercase_responses(请求, 处理程序):
                响应=处理程序（请求）
                ai_msg = 响应.结果[0]
                返回模型响应（
                    结果=[AIMessage(内容=ai_msg.content.upper())],
                    结构化响应=响应.结构化响应，
                ）
            ````
        !!!示例“简单的`AIMessage`返回（自动转换）”
            ````蟒蛇
            @wrap_model_call
            def simple_response(请求, 处理程序):
                # AIMessage自动转换为ModelResponse
                return AIMessage(content="简单回复")
            ````"""

    def decorator(
        func: _CallableReturningModelResponse[StateT, ContextT],
    ) -> AgentMiddleware[StateT, ContextT]:
        is_async = iscoroutinefunction(func)

        if is_async:

            async def async_wrapped(
                _self: AgentMiddleware[StateT, ContextT],
                request: ModelRequest,
                handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
            ) -> ModelCallResult:
                return await func(request, handler)  # type: ignore[misc, arg-type]

            middleware_name = name or cast(
                "str", getattr(func, "__name__", "WrapModelCallMiddleware")
            )

            return type(
                middleware_name,
                (AgentMiddleware,),
                {
                    "state_schema": state_schema or AgentState,
                    "tools": tools or [],
                    "awrap_model_call": async_wrapped,
                },
            )()

        def wrapped(
            _self: AgentMiddleware[StateT, ContextT],
            request: ModelRequest,
            handler: Callable[[ModelRequest], ModelResponse],
        ) -> ModelCallResult:
            return func(request, handler)

        middleware_name = name or cast("str", getattr(func, "__name__", "WrapModelCallMiddleware"))

        return type(
            middleware_name,
            (AgentMiddleware,),
            {
                "state_schema": state_schema or AgentState,
                "tools": tools or [],
                "wrap_model_call": wrapped,
            },
        )()

    if func is not None:
        return decorator(func)
    return decorator


@overload
def wrap_tool_call(
    func: _CallableReturningToolResponse,
) -> AgentMiddleware: ...


@overload
def wrap_tool_call(
    func: None = None,
    *,
    tools: list[BaseTool] | None = None,
    name: str | None = None,
) -> Callable[
    [_CallableReturningToolResponse],
    AgentMiddleware,
]: ...


def wrap_tool_call(
    func: _CallableReturningToolResponse | None = None,
    *,
    tools: list[BaseTool] | None = None,
    name: str | None = None,
) -> (
    Callable[
        [_CallableReturningToolResponse],
        AgentMiddleware,
    ]
    | AgentMiddleware
):
    """Create middleware with `wrap_tool_call` hook from a function.

    Async version is `awrap_tool_call`.

    Converts a function with handler callback into middleware that can intercept
    tool calls, implement retry logic, monitor execution, and modify responses.

    Args:
        func: Function accepting (request, handler) that calls
            handler(request) to execute the tool and returns final `ToolMessage` or
            `Command`.

            Can be sync or async.
        tools: Additional tools to register with this middleware.
        name: Middleware class name.

            Defaults to function name.

    Returns:
        `AgentMiddleware` instance if func provided, otherwise a decorator.

    Examples:
        !!! example "Retry logic"

            ```python
            @wrap_tool_call
            def retry_on_error(request, handler):
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        return handler(request)
                    except Exception:
                        if attempt == max_retries - 1:
                            raise
            ```

        !!! example "Async retry logic"

            ```python
            @wrap_tool_call
            async def async_retry(request, handler):
                for attempt in range(3):
                    try:
                        return await handler(request)
                    except Exception:
                        if attempt == 2:
                            raise
            ```

        !!! example "Modify request"

            ```python
            @wrap_tool_call
            def modify_args(request, handler):
                modified_call = {
                    **request.tool_call,
                    "args": {
                        **request.tool_call["args"],
                        "value": request.tool_call["args"]["value"] * 2,
                    },
                }
                request = request.override(tool_call=modified_call)
                return handler(request)
            ```

        !!! example "Short-circuit with cached result"

            ```python
            @wrap_tool_call
            def with_cache(request, handler):
                if cached := get_cache(request):
                    return ToolMessage(content=cached, tool_call_id=request.tool_call["id"])
                result = handler(request)
                save_cache(request, result)
                return result
            ```
    

    中文翻译:
    使用函数中的“wrap_tool_call”挂钩创建中间件。
    异步版本是“awrap_tool_call”。
    将具有处理程序回调的函数转换为可以拦截的中间件
    工具调用、实现重试逻辑、监视执行和修改响应。
    参数：
        func：接受调用的函数（请求、处理程序）
            handler(request) 执行该工具并返回最终的 `ToolMessage` 或
            ‘命令’。
            可以是同步或异步。
        工具：向此中间件注册的附加工具。
        name：中间件类名。
            默认为函数名称。
    返回：
        如果提供了 func，则为 `AgentMiddleware` 实例，否则为装饰器。
    示例：
        !!!示例“重试逻辑”
            ````蟒蛇
            @wrap_tool_call
            def retry_on_error(请求, 处理程序):
                最大重试次数 = 3
                对于范围内的尝试（max_retries）：
                    尝试：
                        返回处理程序（请求）
                    除了例外：
                        如果尝试 == max_retries - 1：
                            提高
            ````
        !!!示例“异步重试逻辑”
            ````蟒蛇
            @wrap_tool_call
            async def async_retry(请求, 处理程序):
                对于范围（3）中的尝试：
                    尝试：
                        返回等待处理程序（请求）
                    除了例外：
                        如果尝试 == 2：
                            提高
            ````
        !!!示例“修改请求”
            ````蟒蛇
            @wrap_tool_call
            defmodify_args（请求，处理程序）：
                修改后的调用 = {
                    **请求.tool_call，
                    “参数”：{
                        **request.tool_call["args"],
                        "值": request.tool_call["args"]["value"] * 2,
                    },
                }
                请求 = request.override(tool_call=modified_call)
                返回处理程序（请求）
            ````
        !!!示例“缓存结果短路”
            ````蟒蛇
            @wrap_tool_call
            def with_cache（请求，处理程序）：
                如果缓存:= get_cache(请求):
                    返回 ToolMessage(content=cached, tool_call_id=request.tool_call["id"])
                结果 = 处理程序（请求）
                save_cache（请求，结果）
                返回结果
            ````"""

    def decorator(
        func: _CallableReturningToolResponse,
    ) -> AgentMiddleware:
        is_async = iscoroutinefunction(func)

        if is_async:

            async def async_wrapped(
                _self: AgentMiddleware,
                request: ToolCallRequest,
                handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
            ) -> ToolMessage | Command:
                return await func(request, handler)  # type: ignore[arg-type,misc]

            middleware_name = name or cast(
                "str", getattr(func, "__name__", "WrapToolCallMiddleware")
            )

            return type(
                middleware_name,
                (AgentMiddleware,),
                {
                    "state_schema": AgentState,
                    "tools": tools or [],
                    "awrap_tool_call": async_wrapped,
                },
            )()

        def wrapped(
            _self: AgentMiddleware,
            request: ToolCallRequest,
            handler: Callable[[ToolCallRequest], ToolMessage | Command],
        ) -> ToolMessage | Command:
            return func(request, handler)

        middleware_name = name or cast("str", getattr(func, "__name__", "WrapToolCallMiddleware"))

        return type(
            middleware_name,
            (AgentMiddleware,),
            {
                "state_schema": AgentState,
                "tools": tools or [],
                "wrap_tool_call": wrapped,
            },
        )()

    if func is not None:
        return decorator(func)
    return decorator
