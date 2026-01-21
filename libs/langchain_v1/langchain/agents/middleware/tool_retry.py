"""工具调用重试中间件模块。

本模块提供自动重试失败工具调用的中间件，支持可配置的退避策略。

核心类:
--------
**ToolRetryMiddleware**: 工具调用重试中间件

功能特性:
---------
- 指数退避重试策略
- 可配置重试异常类型
- 可限制到特定工具
- 支持同步和异步调用
- 自定义失败处理

使用示例:
---------
>>> from langchain.agents import create_agent
>>> from langchain.agents.middleware import ToolRetryMiddleware
>>>
>>> # 基本用法（默认2次重试）
>>> agent = create_agent(
...     model="openai:gpt-4o",
...     tools=[my_tool],
...     middleware=[ToolRetryMiddleware()],
... )
>>>
>>> # 仅对特定工具重试
>>> retry = ToolRetryMiddleware(
...     max_retries=4,
...     tools=["search_database"],
...     retry_on=(TimeoutError,),
... )
"""

from __future__ import annotations

import asyncio
import time
import warnings
from typing import TYPE_CHECKING

from langchain_core.messages import ToolMessage

from langchain.agents.middleware._retry import (
    OnFailure,
    RetryOn,
    calculate_delay,
    should_retry_exception,
    validate_retry_params,
)
from langchain.agents.middleware.types import AgentMiddleware

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langgraph.types import Command

    from langchain.agents.middleware.types import ToolCallRequest
    from langchain.tools import BaseTool


class ToolRetryMiddleware(AgentMiddleware):
    """Middleware that automatically retries failed tool calls with configurable backoff.

    Supports retrying on specific exceptions and exponential backoff.

    Examples:
        !!! example "Basic usage with default settings (2 retries, exponential backoff)"

            ```python
            from langchain.agents import create_agent
            from langchain.agents.middleware import ToolRetryMiddleware

            agent = create_agent(model, tools=[search_tool], middleware=[ToolRetryMiddleware()])
            ```

        !!! example "Retry specific exceptions only"

            ```python
            from requests.exceptions import RequestException, Timeout

            retry = ToolRetryMiddleware(
                max_retries=4,
                retry_on=(RequestException, Timeout),
                backoff_factor=1.5,
            )
            ```

        !!! example "Custom exception filtering"

            ```python
            from requests.exceptions import HTTPError


            def should_retry(exc: Exception) -> bool:
                # Only retry on 5xx errors
                # 中文: 仅重试 5xx 错误
                if isinstance(exc, HTTPError):
                    return 500 <= exc.status_code < 600
                return False


            retry = ToolRetryMiddleware(
                max_retries=3,
                retry_on=should_retry,
            )
            ```

        !!! example "Apply to specific tools with custom error handling"

            ```python
            def format_error(exc: Exception) -> str:
                return "Database temporarily unavailable. Please try again later."


            retry = ToolRetryMiddleware(
                max_retries=4,
                tools=["search_database"],
                on_failure=format_error,
            )
            ```

        !!! example "Apply to specific tools using `BaseTool` instances"

            ```python
            from langchain_core.tools import tool


            @tool
            def search_database(query: str) -> str:
                '''Search the database.'''
                return results


            retry = ToolRetryMiddleware(
                max_retries=4,
                tools=[search_database],  # Pass BaseTool instance
            )
            ```

        !!! example "Constant backoff (no exponential growth)"

            ```python
            retry = ToolRetryMiddleware(
                max_retries=5,
                backoff_factor=0.0,  # No exponential growth
                initial_delay=2.0,  # Always wait 2 seconds
            )
            ```

        !!! example "Raise exception on failure"

            ```python
            retry = ToolRetryMiddleware(
                max_retries=2,
                on_failure="error",  # Re-raise exception instead of returning message
            )
            ```
    

    中文翻译:
    通过可配置的退避自动重试失败的工具调用的中间件。
    支持重试特定异常和指数退避。
    示例：
        !!!示例“默认设置的基本用法（2 次重试，指数退避）”
            ````蟒蛇
            从 langchain.agents 导入 create_agent
            从 langchain.agents.middleware 导入 ToolRetryMiddleware
            代理 = create_agent(模型、工具=[search_tool]、中间件=[ToolRetryMiddleware()])
            ````
        !!!示例“仅重试特定异常”
            ````蟒蛇
            从 requests.exceptions 导入 RequestException、超时
            重试 = ToolRetryMiddleware(
                最大重试次数=4，
                retry_on=(RequestException, 超时),
                退避因子=1.5，
            ）
            ````
        !!!示例“自定义异常过滤”
            ````蟒蛇
            从 requests.exceptions 导入 HTTPError
            def should_retry(exc: 异常) -> bool:
                # 仅重试 5xx 错误
                if isinstance(exc, HTTPError):
                    返回 500 <= exc.status_code < 600
                返回错误
            重试 = ToolRetryMiddleware(
                最大重试次数=3，
                retry_on=应该重试，
            ）
            ````
        !!!示例“通过自定义错误处理应用于特定工具”
            ````蟒蛇
            def format_error(exc: 异常) -> str:
                返回“数据库暂时不可用。请稍后重试。”
            重试 = ToolRetryMiddleware(
                最大重试次数=4，
                工具=[“搜索数据库”]，
                on_failure=格式错误，
            ）
            ````
        !!!示例“使用 `BaseTool` 实例应用于特定工具”
            ````蟒蛇
            从 langchain_core.tools 导入工具
            @工具
            def search_database(查询: str) -> str:
                '''搜索数据库。'''
                返回结果
            重试 = ToolRetryMiddleware(
                最大重试次数=4，
                tools=[search_database], # 传递BaseTool实例
            ）
            ````
        !!!示例“恒定退避（无指数增长）”
            ````蟒蛇
            重试 = ToolRetryMiddleware(
                最大重试次数=5，
                backoff_factor=0.0, # 无指数增长
                initial_delay=2.0, # 始终等待 2 秒
            ）
            ````
        !!!示例“失败时引发异常”
            ````蟒蛇
            重试 = ToolRetryMiddleware(
                最大重试次数=2，
                on_failure="error", # 重新引发异常而不是返回消息
            ）
            ````"""

    def __init__(
        self,
        *,
        max_retries: int = 2,
        tools: list[BaseTool | str] | None = None,
        retry_on: RetryOn = (Exception,),
        on_failure: OnFailure = "continue",
        backoff_factor: float = 2.0,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        jitter: bool = True,
    ) -> None:
        """Initialize `ToolRetryMiddleware`.

        Args:
            max_retries: Maximum number of retry attempts after the initial call.

                Must be `>= 0`.
            tools: Optional list of tools or tool names to apply retry logic to.

                Can be a list of `BaseTool` instances or tool name strings.

                If `None`, applies to all tools.
            retry_on: Either a tuple of exception types to retry on, or a callable
                that takes an exception and returns `True` if it should be retried.

                Default is to retry on all exceptions.
            on_failure: Behavior when all retries are exhausted.

                Options:

                - `'continue'`: Return a `ToolMessage` with error details,
                    allowing the LLM to handle the failure and potentially recover.
                - `'error'`: Re-raise the exception, stopping agent execution.
                - **Custom callable:** Function that takes the exception and returns a
                    string for the `ToolMessage` content, allowing custom error
                    formatting.

                **Deprecated values** (for backwards compatibility):

                - `'return_message'`: Use `'continue'` instead.
                - `'raise'`: Use `'error'` instead.
            backoff_factor: Multiplier for exponential backoff.

                Each retry waits `initial_delay * (backoff_factor ** retry_number)`
                seconds.

                Set to `0.0` for constant delay.
            initial_delay: Initial delay in seconds before first retry.
            max_delay: Maximum delay in seconds between retries.

                Caps exponential backoff growth.
            jitter: Whether to add random jitter (`±25%`) to delay to avoid thundering herd.

        Raises:
            ValueError: If `max_retries < 0` or delays are negative.
        

        中文翻译:
        初始化“ToolRetryMiddleware”。
        参数：
            max_retries：初始调用后重试的最大次数。
                必须是 `>= 0`。
            工具：要应用重试逻辑的工具或工具名称的可选列表。
                可以是“BaseTool”实例或工具名称字符串的列表。
                如果为“无”，则适用于所有工具。
            retry_on：要重试的异常类型元组，或者可调用的
                如果应该重试，则接受异常并返回“True”。
                默认是重试所有异常。
            on_failure：所有重试都用尽时的行为。
                选项：
                - “继续”：返回包含错误详细信息的“ToolMessage”，
                    允许法学硕士处理失败并可能恢复。
                - `'error'`：重新引发异常，停止代理执行。
                - **自定义可调用：** 接受异常并返回的函数
                    `ToolMessage` 内容的字符串，允许自定义错误
                    格式化。
                **弃用的值**（为了向后兼容）：
                - `'return_message'`：使用`'继续'`代替。
                - “raise”：使用“error”代替。
            backoff_factor：指数退避的乘数。
                每次重试都会等待 `initial_delay * (backoff_factor ** retry_number)`
                秒。
                设置为“0.0”以获得恒定延迟。
            初始延迟：第一次重试之前的初始延迟（以秒为单位）。
            max_delay：重试之间的最大延迟（以秒为单位）。
                限制指数退避增长。
            jitter：是否添加随机抖动（`±25%`）来延迟以避免雷群。
        加薪：
            ValueError：如果“max_retries < 0”或延迟为负数。"""
        super().__init__()

        # Validate parameters
        # 中文: 验证参数
        validate_retry_params(max_retries, initial_delay, max_delay, backoff_factor)

        # Handle backwards compatibility for deprecated on_failure values
        # 中文: 处理已弃用的 on_failure 值的向后兼容性
        if on_failure == "raise":  # type: ignore[comparison-overlap]
            msg = (
                "on_failure='raise' is deprecated and will be removed in a future version. "
                "Use on_failure='error' instead."
            )
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            on_failure = "error"
        elif on_failure == "return_message":  # type: ignore[comparison-overlap]
            msg = (
                "on_failure='return_message' is deprecated and will be removed "
                "in a future version. Use on_failure='continue' instead."
            )
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            on_failure = "continue"

        self.max_retries = max_retries

        # Extract tool names from BaseTool instances or strings
        # 中文: 从 BaseTool 实例或字符串中提取工具名称
        self._tool_filter: list[str] | None
        if tools is not None:
            self._tool_filter = [tool.name if not isinstance(tool, str) else tool for tool in tools]
        else:
            self._tool_filter = None

        self.tools = []  # No additional tools registered by this middleware
        self.retry_on = retry_on
        self.on_failure = on_failure
        self.backoff_factor = backoff_factor
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.jitter = jitter

    def _should_retry_tool(self, tool_name: str) -> bool:
        """Check if retry logic should apply to this tool.

        Args:
            tool_name: Name of the tool being called.

        Returns:
            `True` if retry logic should apply, `False` otherwise.
        

        中文翻译:
        检查重试逻辑是否适用于此工具。
        参数：
            tool_name：正在调用的工具的名称。
        返回：
            如果重试逻辑应适用，则为“True”，否则为“False”。"""
        if self._tool_filter is None:
            return True
        return tool_name in self._tool_filter

    def _format_failure_message(self, tool_name: str, exc: Exception, attempts_made: int) -> str:
        """Format the failure message when retries are exhausted.

        Args:
            tool_name: Name of the tool that failed.
            exc: The exception that caused the failure.
            attempts_made: Number of attempts actually made.

        Returns:
            Formatted error message string.
        

        中文翻译:
        当重试次数用尽时，格式化失败消息。
        参数：
            tool_name：失败的工具的名称。
            exc：导致失败的异常。
            attempts_made：实际尝试的次数。
        返回：
            格式化的错误消息字符串。"""
        exc_type = type(exc).__name__
        exc_msg = str(exc)
        attempt_word = "attempt" if attempts_made == 1 else "attempts"
        return (
            f"Tool '{tool_name}' failed after {attempts_made} {attempt_word} "
            f"with {exc_type}: {exc_msg}. Please try again."
        )

    def _handle_failure(
        self, tool_name: str, tool_call_id: str | None, exc: Exception, attempts_made: int
    ) -> ToolMessage:
        """Handle failure when all retries are exhausted.

        Args:
            tool_name: Name of the tool that failed.
            tool_call_id: ID of the tool call (may be `None`).
            exc: The exception that caused the failure.
            attempts_made: Number of attempts actually made.

        Returns:
            `ToolMessage` with error details.

        Raises:
            Exception: If `on_failure` is `'error'`, re-raises the exception.
        

        中文翻译:
        当所有重试都用尽时处理失败。
        参数：
            tool_name：失败的工具的名称。
            tool_call_id：工具调用的 ID（可以是“None”）。
            exc：导致失败的异常。
            attempts_made：实际尝试的次数。
        返回：
            带有错误详细信息的“ToolMessage”。
        加薪：
            异常：如果`on_failure`是`'error'`，则重新引发异常。"""
        if self.on_failure == "error":
            raise exc

        if callable(self.on_failure):
            content = self.on_failure(exc)
        else:
            content = self._format_failure_message(tool_name, exc, attempts_made)

        return ToolMessage(
            content=content,
            tool_call_id=tool_call_id,
            name=tool_name,
            status="error",
        )

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Intercept tool execution and retry on failure.

        Args:
            request: Tool call request with call dict, `BaseTool`, state, and runtime.
            handler: Callable to execute the tool (can be called multiple times).

        Returns:
            `ToolMessage` or `Command` (the final result).
        

        中文翻译:
        拦截工具执行并在失败时重试。
        参数：
            request：工具调用请求，包含调用字典、`BaseTool`、状态和运行时。
            handler：可调用来执行工具（可以调用多次）。
        返回：
            `ToolMessage` 或 `Command` （最终结果）。"""
        tool_name = request.tool.name if request.tool else request.tool_call["name"]

        # Check if retry should apply to this tool
        # 中文: 检查重试是否适用于此工具
        if not self._should_retry_tool(tool_name):
            return handler(request)

        tool_call_id = request.tool_call["id"]

        # Initial attempt + retries
        # 中文: 初次尝试+重试
        for attempt in range(self.max_retries + 1):
            try:
                return handler(request)
            except Exception as exc:
                attempts_made = attempt + 1  # attempt is 0-indexed

                # Check if we should retry this exception
                # 中文: 检查我们是否应该重试此异常
                if not should_retry_exception(exc, self.retry_on):
                    # Exception is not retryable, handle failure immediately
                    # 中文: 异常不可重试，失败立即处理
                    return self._handle_failure(tool_name, tool_call_id, exc, attempts_made)

                # Check if we have more retries left
                # 中文: 检查是否还有更多重试机会
                if attempt < self.max_retries:
                    # Calculate and apply backoff delay
                    # 中文: 计算并应用退避延迟
                    delay = calculate_delay(
                        attempt,
                        backoff_factor=self.backoff_factor,
                        initial_delay=self.initial_delay,
                        max_delay=self.max_delay,
                        jitter=self.jitter,
                    )
                    if delay > 0:
                        time.sleep(delay)
                    # Continue to next retry
                    # 中文: 继续下一次重试
                else:
                    # No more retries, handle failure
                    # 中文: 不再重试，处理失败
                    return self._handle_failure(tool_name, tool_call_id, exc, attempts_made)

        # Unreachable: loop always returns via handler success or _handle_failure
        # 中文: 无法访问：循环始终通过处理程序 success 或 _handle_failure 返回
        msg = "Unexpected: retry loop completed without returning"
        raise RuntimeError(msg)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Intercept and control async tool execution with retry logic.

        Args:
            request: Tool call request with call `dict`, `BaseTool`, state, and runtime.
            handler: Async callable to execute the tool and returns `ToolMessage` or
                `Command`.

        Returns:
            `ToolMessage` or `Command` (the final result).
        

        中文翻译:
        使用重试逻辑拦截和控制异步工具执行。
        参数：
            request：工具调用请求，包含调用 `dict`、`BaseTool`、状态和运行时。
            handler：异步调用来执行工具并返回“ToolMessage”或
                ‘命令’。
        返回：
            `ToolMessage` 或 `Command` （最终结果）。"""
        tool_name = request.tool.name if request.tool else request.tool_call["name"]

        # Check if retry should apply to this tool
        # 中文: 检查重试是否适用于此工具
        if not self._should_retry_tool(tool_name):
            return await handler(request)

        tool_call_id = request.tool_call["id"]

        # Initial attempt + retries
        # 中文: 初次尝试+重试
        for attempt in range(self.max_retries + 1):
            try:
                return await handler(request)
            except Exception as exc:
                attempts_made = attempt + 1  # attempt is 0-indexed

                # Check if we should retry this exception
                # 中文: 检查我们是否应该重试此异常
                if not should_retry_exception(exc, self.retry_on):
                    # Exception is not retryable, handle failure immediately
                    # 中文: 异常不可重试，失败立即处理
                    return self._handle_failure(tool_name, tool_call_id, exc, attempts_made)

                # Check if we have more retries left
                # 中文: 检查是否还有更多重试机会
                if attempt < self.max_retries:
                    # Calculate and apply backoff delay
                    # 中文: 计算并应用退避延迟
                    delay = calculate_delay(
                        attempt,
                        backoff_factor=self.backoff_factor,
                        initial_delay=self.initial_delay,
                        max_delay=self.max_delay,
                        jitter=self.jitter,
                    )
                    if delay > 0:
                        await asyncio.sleep(delay)
                    # Continue to next retry
                    # 中文: 继续下一次重试
                else:
                    # No more retries, handle failure
                    # 中文: 不再重试，处理失败
                    return self._handle_failure(tool_name, tool_call_id, exc, attempts_made)

        # Unreachable: loop always returns via handler success or _handle_failure
        # 中文: 无法访问：循环始终通过处理程序 success 或 _handle_failure 返回
        msg = "Unexpected: retry loop completed without returning"
        raise RuntimeError(msg)
