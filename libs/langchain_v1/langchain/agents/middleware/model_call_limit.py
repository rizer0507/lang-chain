"""模型调用次数限制中间件模块。

本模块提供跟踪和限制模型调用次数的能力。

核心类:
--------
**ModelCallLimitMiddleware**: 模型调用限制中间件

功能特性:
---------
- 线程级限制：跨多次调用持久化
- 运行级限制：单次调用内限制
- 可配置超限行为

超限行为:
---------
- `end`: 结束执行并注入消息
- `error`: 抛出 ModelCallLimitExceededError

使用示例:
---------
>>> from langchain.agents import create_agent
>>> from langchain.agents.middleware import ModelCallLimitMiddleware
>>>
>>> limiter = ModelCallLimitMiddleware(
...     thread_limit=10,  # 线程内最多10次
...     run_limit=5,      # 单次运行最多5次
...     exit_behavior="end",
... )
>>>
>>> agent = create_agent(
...     model="openai:gpt-4o",
...     middleware=[limiter],
... )
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Literal

from langchain_core.messages import AIMessage
from langgraph.channels.untracked_value import UntrackedValue
from typing_extensions import NotRequired, override

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    PrivateStateAttr,
    hook_config,
)

if TYPE_CHECKING:
    from langgraph.runtime import Runtime


class ModelCallLimitState(AgentState):
    """State schema for `ModelCallLimitMiddleware`.

    Extends `AgentState` with model call tracking fields.
    

    中文翻译:
    “ModelCallLimitMiddleware”的状态模式。
    使用模型调用跟踪字段扩展“AgentState”。"""

    thread_model_call_count: NotRequired[Annotated[int, PrivateStateAttr]]
    run_model_call_count: NotRequired[Annotated[int, UntrackedValue, PrivateStateAttr]]


def _build_limit_exceeded_message(
    thread_count: int,
    run_count: int,
    thread_limit: int | None,
    run_limit: int | None,
) -> str:
    """Build a message indicating which limits were exceeded.

    Args:
        thread_count: Current thread model call count.
        run_count: Current run model call count.
        thread_limit: Thread model call limit (if set).
        run_limit: Run model call limit (if set).

    Returns:
        A formatted message describing which limits were exceeded.
    

    中文翻译:
    构建一条消息，指示超出了哪些限制。
    参数：
        thread_count：当前线程模型调用计数。
        run_count：当前运行模型调用计数。
        thread_limit：线程模型调用限制（如果设置）。
        run_limit：运行模型调用限制（如果设置）。
    返回：
        描述超出哪些限制的格式化消息。"""
    exceeded_limits = []
    if thread_limit is not None and thread_count >= thread_limit:
        exceeded_limits.append(f"thread limit ({thread_count}/{thread_limit})")
    if run_limit is not None and run_count >= run_limit:
        exceeded_limits.append(f"run limit ({run_count}/{run_limit})")

    return f"Model call limits exceeded: {', '.join(exceeded_limits)}"


class ModelCallLimitExceededError(Exception):
    """Exception raised when model call limits are exceeded.

    This exception is raised when the configured exit behavior is `'error'` and either
    the thread or run model call limit has been exceeded.
    

    中文翻译:
    超出模型调用限制时引发异常。
    当配置的退出行为是“错误”并且
    已超出线程或运行模型调用限制。"""

    def __init__(
        self,
        thread_count: int,
        run_count: int,
        thread_limit: int | None,
        run_limit: int | None,
    ) -> None:
        """Initialize the exception with call count information.

        Args:
            thread_count: Current thread model call count.
            run_count: Current run model call count.
            thread_limit: Thread model call limit (if set).
            run_limit: Run model call limit (if set).
        

        中文翻译:
        使用调用计数信息初始化异常。
        参数：
            thread_count：当前线程模型调用计数。
            run_count：当前运行模型调用计数。
            thread_limit：线程模型调用限制（如果设置）。
            run_limit：运行模型调用限制（如果设置）。"""
        self.thread_count = thread_count
        self.run_count = run_count
        self.thread_limit = thread_limit
        self.run_limit = run_limit

        msg = _build_limit_exceeded_message(thread_count, run_count, thread_limit, run_limit)
        super().__init__(msg)


class ModelCallLimitMiddleware(AgentMiddleware[ModelCallLimitState, Any]):
    """Tracks model call counts and enforces limits.

    This middleware monitors the number of model calls made during agent execution
    and can terminate the agent when specified limits are reached. It supports
    both thread-level and run-level call counting with configurable exit behaviors.

    Thread-level: The middleware tracks the number of model calls and persists
    call count across multiple runs (invocations) of the agent.

    Run-level: The middleware tracks the number of model calls made during a single
    run (invocation) of the agent.

    Example:
        ```python
        from langchain.agents.middleware.call_tracking import ModelCallLimitMiddleware
        from langchain.agents import create_agent

        # Create middleware with limits
        # 中文: 创建有限制的中间件
        call_tracker = ModelCallLimitMiddleware(thread_limit=10, run_limit=5, exit_behavior="end")

        agent = create_agent("openai:gpt-4o", middleware=[call_tracker])

        # Agent will automatically jump to end when limits are exceeded
        # 中文: 超过限制时Agent会自动跳转到结束
        result = await agent.invoke({"messages": [HumanMessage("Help me with a task")]})
        ```
    

    中文翻译:
    跟踪模型调用计数并强制执行限制。
    该中间件监视代理执行期间进行的模型调用的数量
    并可以在达到指定限制时终止代理。它支持
    具有可配置退出行为的线程级和运行级调用计数。
    线程级：中间件跟踪模型调用次数并持久化
    代理多次运行（调用）的调用计数。
    运行级别：中间件跟踪单个过程中进行的模型调用数量
    运行（调用）代理。
    示例：
        ````蟒蛇
        从 langchain.agents.middleware.call_tracking 导入 ModelCallLimitMiddleware
        从 langchain.agents 导入 create_agent
        # 创建有限制的中间件
        call_tracker = ModelCallLimitMiddleware(thread_limit=10, run_limit=5, exit_behavior="end")
        代理 = create_agent("openai:gpt-4o", middleware=[call_tracker])
        # 超出限制时Agent会自动跳转到结束
        result = wait agent.invoke({"messages": [HumanMessage("帮我完成任务")]})
        ````"""

    state_schema = ModelCallLimitState

    def __init__(
        self,
        *,
        thread_limit: int | None = None,
        run_limit: int | None = None,
        exit_behavior: Literal["end", "error"] = "end",
    ) -> None:
        """Initialize the call tracking middleware.

        Args:
            thread_limit: Maximum number of model calls allowed per thread.

                `None` means no limit.
            run_limit: Maximum number of model calls allowed per run.

                `None` means no limit.
            exit_behavior: What to do when limits are exceeded.

                - `'end'`: Jump to the end of the agent execution and
                    inject an artificial AI message indicating that the limit was
                    exceeded.
                - `'error'`: Raise a `ModelCallLimitExceededError`

        Raises:
            ValueError: If both limits are `None` or if `exit_behavior` is invalid.
        

        中文翻译:
        初始化呼叫跟踪中间件。
        参数：
            thread_limit：每个线程允许的最大模型调用数。
                “无”意味着没有限制。
            run_limit：每次运行允许的最大模型调用次数。
                “无”意味着没有限制。
            exit_behavior：超出限制时该怎么办。
                - `'end'`：跳转到代理执行的末尾并
                    注入人工 AI 消息，表明限制已达到
                    超过了。
                -“错误”：引发“ModelCallLimitExceededError”
        加薪：
            ValueError：如果两个限制均为“无”或“exit_behavior”无效。"""
        super().__init__()

        if thread_limit is None and run_limit is None:
            msg = "At least one limit must be specified (thread_limit or run_limit)"
            raise ValueError(msg)

        if exit_behavior not in {"end", "error"}:
            msg = f"Invalid exit_behavior: {exit_behavior}. Must be 'end' or 'error'"
            raise ValueError(msg)

        self.thread_limit = thread_limit
        self.run_limit = run_limit
        self.exit_behavior = exit_behavior

    @hook_config(can_jump_to=["end"])
    @override
    def before_model(self, state: ModelCallLimitState, runtime: Runtime) -> dict[str, Any] | None:
        """Check model call limits before making a model call.

        Args:
            state: The current agent state containing call counts.
            runtime: The langgraph runtime.

        Returns:
            If limits are exceeded and exit_behavior is `'end'`, returns
                a `Command` to jump to the end with a limit exceeded message. Otherwise
                returns `None`.

        Raises:
            ModelCallLimitExceededError: If limits are exceeded and `exit_behavior`
                is `'error'`.
        

        中文翻译:
        在进行模型调用之前检查模型调用限制。
        参数：
            状态：包含呼叫计数的当前代理状态。
            运行时：语言图运行时。
        返回：
            如果超出限制且 exit_behavior 为“end”，则返回
                一个“命令”，用于跳转到末尾并显示超出限制的消息。否则
                返回“无”。
        加薪：
            ModelCallLimitExceededError：如果超出限制并且“exit_behavior”
                是“错误”。"""
        thread_count = state.get("thread_model_call_count", 0)
        run_count = state.get("run_model_call_count", 0)

        # Check if any limits will be exceeded after the next call
        # 中文: 检查下一次调用后是否会超出任何限制
        thread_limit_exceeded = self.thread_limit is not None and thread_count >= self.thread_limit
        run_limit_exceeded = self.run_limit is not None and run_count >= self.run_limit

        if thread_limit_exceeded or run_limit_exceeded:
            if self.exit_behavior == "error":
                raise ModelCallLimitExceededError(
                    thread_count=thread_count,
                    run_count=run_count,
                    thread_limit=self.thread_limit,
                    run_limit=self.run_limit,
                )
            if self.exit_behavior == "end":
                # Create a message indicating the limit was exceeded
                # 中文: 创建一条消息，指示超出限制
                limit_message = _build_limit_exceeded_message(
                    thread_count, run_count, self.thread_limit, self.run_limit
                )
                limit_ai_message = AIMessage(content=limit_message)

                return {"jump_to": "end", "messages": [limit_ai_message]}

        return None

    @hook_config(can_jump_to=["end"])
    async def abefore_model(
        self,
        state: ModelCallLimitState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Async check model call limits before making a model call.

        Args:
            state: The current agent state containing call counts.
            runtime: The langgraph runtime.

        Returns:
            If limits are exceeded and exit_behavior is `'end'`, returns
                a `Command` to jump to the end with a limit exceeded message. Otherwise
                returns `None`.

        Raises:
            ModelCallLimitExceededError: If limits are exceeded and `exit_behavior`
                is `'error'`.
        

        中文翻译:
        在进行模型调用之前异步检查模型调用限制。
        参数：
            状态：包含呼叫计数的当前代理状态。
            运行时：语言图运行时。
        返回：
            如果超出限制且 exit_behavior 为“end”，则返回
                一个“命令”，用于跳转到末尾并显示超出限制的消息。否则
                返回“无”。
        加薪：
            ModelCallLimitExceededError：如果超出限制并且“exit_behavior”
                是“错误”。"""
        return self.before_model(state, runtime)

    @override
    def after_model(self, state: ModelCallLimitState, runtime: Runtime) -> dict[str, Any] | None:
        """Increment model call counts after a model call.

        Args:
            state: The current agent state.
            runtime: The langgraph runtime.

        Returns:
            State updates with incremented call counts.
        

        中文翻译:
        模型调用后增加模型调用计数。
        参数：
            状态：当前代理状态。
            运行时：语言图运行时。
        返回：
            状态随着调用计数的增加而更新。"""
        return {
            "thread_model_call_count": state.get("thread_model_call_count", 0) + 1,
            "run_model_call_count": state.get("run_model_call_count", 0) + 1,
        }

    async def aafter_model(
        self,
        state: ModelCallLimitState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Async increment model call counts after a model call.

        Args:
            state: The current agent state.
            runtime: The langgraph runtime.

        Returns:
            State updates with incremented call counts.
        

        中文翻译:
        模型调用后异步增量模型调用计数。
        参数：
            状态：当前代理状态。
            运行时：语言图运行时。
        返回：
            状态随着调用计数的增加而更新。"""
        return self.after_model(state, runtime)
