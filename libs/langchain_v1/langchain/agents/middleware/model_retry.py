"""模型调用重试中间件模块。

本模块提供自动重试失败模型调用的中间件，支持可配置的退避策略。

核心类:
--------
**ModelRetryMiddleware**: 模型调用重试中间件

功能特性:
---------
- 指数退避重试策略
- 可配置重试异常类型
- 支持同步和异步调用
- 自定义失败处理

使用示例:
---------
>>> from langchain.agents import create_agent
>>> from langchain.agents.middleware import ModelRetryMiddleware
>>>
>>> # 基本用法（默认2次重试，指数退避）
>>> agent = create_agent(
...     model="openai:gpt-4o",
...     middleware=[ModelRetryMiddleware()],
... )
>>>
>>> # 自定义重试配置
>>> retry = ModelRetryMiddleware(
...     max_retries=4,
...     retry_on=(TimeoutError, ConnectionError),
...     backoff_factor=1.5,
... )
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage

from langchain.agents.middleware._retry import (
    OnFailure,
    RetryOn,
    calculate_delay,
    should_retry_exception,
    validate_retry_params,
)
from langchain.agents.middleware.types import AgentMiddleware, ModelResponse

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain.agents.middleware.types import ModelRequest


class ModelRetryMiddleware(AgentMiddleware):
    """Middleware that automatically retries failed model calls with configurable backoff.

    Supports retrying on specific exceptions and exponential backoff.

    Examples:
        !!! example "Basic usage with default settings (2 retries, exponential backoff)"

            ```python
            from langchain.agents import create_agent
            from langchain.agents.middleware import ModelRetryMiddleware

            agent = create_agent(model, tools=[search_tool], middleware=[ModelRetryMiddleware()])
            ```

        !!! example "Retry specific exceptions only"

            ```python
            from anthropic import RateLimitError
            from openai import APITimeoutError

            retry = ModelRetryMiddleware(
                max_retries=4,
                retry_on=(APITimeoutError, RateLimitError),
                backoff_factor=1.5,
            )
            ```

        !!! example "Custom exception filtering"

            ```python
            from anthropic import APIStatusError


            def should_retry(exc: Exception) -> bool:
                # Only retry on 5xx errors
                # 中文: 仅重试 5xx 错误
                if isinstance(exc, APIStatusError):
                    return 500 <= exc.status_code < 600
                return False


            retry = ModelRetryMiddleware(
                max_retries=3,
                retry_on=should_retry,
            )
            ```

        !!! example "Custom error handling"

            ```python
            def format_error(exc: Exception) -> str:
                return "Model temporarily unavailable. Please try again later."


            retry = ModelRetryMiddleware(
                max_retries=4,
                on_failure=format_error,
            )
            ```

        !!! example "Constant backoff (no exponential growth)"

            ```python
            retry = ModelRetryMiddleware(
                max_retries=5,
                backoff_factor=0.0,  # No exponential growth
                initial_delay=2.0,  # Always wait 2 seconds
            )
            ```

        !!! example "Raise exception on failure"

            ```python
            retry = ModelRetryMiddleware(
                max_retries=2,
                on_failure="error",  # Re-raise exception instead of returning message
            )
            ```
    

    中文翻译:
    通过可配置的退避自动重试失败的模型调用的中间件。
    支持重试特定异常和指数退避。
    示例：
        !!!示例“默认设置的基本用法（2 次重试，指数退避）”
            ````蟒蛇
            从 langchain.agents 导入 create_agent
            从 langchain.agents.middleware 导入 ModelRetryMiddleware
            代理 = create_agent(模型、工具=[search_tool]、中间件=[ModelRetryMiddleware()])
            ````
        !!!示例“仅重试特定异常”
            ````蟒蛇
            来自 anthropic import RateLimitError
            从 openai 导入 APITimeoutError
            重试 = ModelRetryMiddleware(
                最大重试次数=4，
                retry_on=(APITimeoutError, RateLimitError),
                退避因子=1.5，
            ）
            ````
        !!!示例“自定义异常过滤”
            ````蟒蛇
            来自 anthropic import APIStatusError
            def should_retry(exc: 异常) -> bool:
                # 仅重试 5xx 错误
                if isinstance(exc, APIStatusError):
                    返回 500 <= exc.status_code < 600
                返回错误
            重试 = ModelRetryMiddleware(
                最大重试次数=3，
                retry_on=应该重试，
            ）
            ````
        !!!示例“自定义错误处理”
            ````蟒蛇
            def format_error(exc: 异常) -> str:
                return“模型暂时不可用，请稍后重试。”
            重试 = ModelRetryMiddleware(
                最大重试次数=4，
                on_failure=格式错误，
            ）
            ````
        !!!示例“恒定退避（无指数增长）”
            ````蟒蛇
            重试 = ModelRetryMiddleware(
                最大重试次数=5，
                backoff_factor=0.0, # 无指数增长
                initial_delay=2.0, # 始终等待 2 秒
            ）
            ````
        !!!示例“失败时引发异常”
            ````蟒蛇
            重试 = ModelRetryMiddleware(
                最大重试次数=2，
                on_failure="error", # 重新引发异常而不是返回消息
            ）
            ````"""

    def __init__(
        self,
        *,
        max_retries: int = 2,
        retry_on: RetryOn = (Exception,),
        on_failure: OnFailure = "continue",
        backoff_factor: float = 2.0,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        jitter: bool = True,
    ) -> None:
        """Initialize `ModelRetryMiddleware`.

        Args:
            max_retries: Maximum number of retry attempts after the initial call.

                Must be `>= 0`.
            retry_on: Either a tuple of exception types to retry on, or a callable
                that takes an exception and returns `True` if it should be retried.

                Default is to retry on all exceptions.
            on_failure: Behavior when all retries are exhausted.

                Options:

                - `'continue'`: Return an `AIMessage` with error details,
                    allowing the agent to continue with an error response.
                - `'error'`: Re-raise the exception, stopping agent execution.
                - **Custom callable:** Function that takes the exception and returns a
                    string for the `AIMessage` content, allowing custom error
                    formatting.
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
        初始化“ModelRetryMiddleware”。
        参数：
            max_retries：初始调用后重试的最大次数。
                必须是 `>= 0`。
            retry_on：要重试的异常类型元组，或者可调用的
                如果应该重试，则接受异常并返回“True”。
                默认是重试所有异常。
            on_failure：所有重试都用尽时的行为。
                选项：
                - “继续”：返回包含错误详细信息的“AIMessage”，
                    允许代理继续执行错误响应。
                - `'error'`：重新引发异常，停止代理执行。
                - **自定义可调用：** 接受异常并返回的函数
                    “AIMessage”内容的字符串，允许自定义错误
                    格式化。
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

        self.max_retries = max_retries
        self.tools = []  # No additional tools registered by this middleware
        self.retry_on = retry_on
        self.on_failure = on_failure
        self.backoff_factor = backoff_factor
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.jitter = jitter

    def _format_failure_message(self, exc: Exception, attempts_made: int) -> AIMessage:
        """Format the failure message when retries are exhausted.

        Args:
            exc: The exception that caused the failure.
            attempts_made: Number of attempts actually made.

        Returns:
            `AIMessage` with formatted error message.
        

        中文翻译:
        当重试次数用尽时，格式化失败消息。
        参数：
            exc：导致失败的异常。
            attempts_made：实际尝试的次数。
        返回：
            带有格式化错误消息的“AIMessage”。"""
        exc_type = type(exc).__name__
        exc_msg = str(exc)
        attempt_word = "attempt" if attempts_made == 1 else "attempts"
        content = (
            f"Model call failed after {attempts_made} {attempt_word} with {exc_type}: {exc_msg}"
        )
        return AIMessage(content=content)

    def _handle_failure(self, exc: Exception, attempts_made: int) -> ModelResponse:
        """Handle failure when all retries are exhausted.

        Args:
            exc: The exception that caused the failure.
            attempts_made: Number of attempts actually made.

        Returns:
            `ModelResponse` with error details.

        Raises:
            Exception: If `on_failure` is `'error'`, re-raises the exception.
        

        中文翻译:
        当所有重试都用尽时处理失败。
        参数：
            exc：导致失败的异常。
            attempts_made：实际尝试的次数。
        返回：
            带有错误详细信息的“ModelResponse”。
        加薪：
            异常：如果`on_failure`是`'error'`，则重新引发异常。"""
        if self.on_failure == "error":
            raise exc

        if callable(self.on_failure):
            content = self.on_failure(exc)
            ai_msg = AIMessage(content=content)
        else:
            ai_msg = self._format_failure_message(exc, attempts_made)

        return ModelResponse(result=[ai_msg])

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse | AIMessage:
        """Intercept model execution and retry on failure.

        Args:
            request: Model request with model, messages, state, and runtime.
            handler: Callable to execute the model (can be called multiple times).

        Returns:
            `ModelResponse` or `AIMessage` (the final result).
        

        中文翻译:
        拦截模型执行并在失败时重试。
        参数：
            请求：包含模型、消息、状态和运行时的模型请求。
            handler：可调用来执行模型（可以调用多次）。
        返回：
            `ModelResponse` 或 `AIMessage` （最终结果）。"""
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
                    return self._handle_failure(exc, attempts_made)

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
                    return self._handle_failure(exc, attempts_made)

        # Unreachable: loop always returns via handler success or _handle_failure
        # 中文: 无法访问：循环始终通过处理程序 success 或 _handle_failure 返回
        msg = "Unexpected: retry loop completed without returning"
        raise RuntimeError(msg)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse | AIMessage:
        """Intercept and control async model execution with retry logic.

        Args:
            request: Model request with model, messages, state, and runtime.
            handler: Async callable to execute the model and returns `ModelResponse`.

        Returns:
            `ModelResponse` or `AIMessage` (the final result).
        

        中文翻译:
        使用重试逻辑拦截和控制异步模型执行。
        参数：
            请求：包含模型、消息、状态和运行时的模型请求。
            handler：异步调用来执行模型并返回“ModelResponse”。
        返回：
            `ModelResponse` 或 `AIMessage` （最终结果）。"""
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
                    return self._handle_failure(exc, attempts_made)

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
                    return self._handle_failure(exc, attempts_made)

        # Unreachable: loop always returns via handler success or _handle_failure
        # 中文: 无法访问：循环始终通过处理程序 success 或 _handle_failure 返回
        msg = "Unexpected: retry loop completed without returning"
        raise RuntimeError(msg)
