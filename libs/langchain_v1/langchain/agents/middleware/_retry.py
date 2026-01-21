"""重试工具模块（内部使用）。

本模块包含模型和工具重试中间件共享的常量、工具和逻辑。

核心功能:
---------
- `RetryOn`: 指定重试异常类型
- `OnFailure`: 指定失败处理行为
- `calculate_delay`: 计算指数退避延迟
- `should_retry_exception`: 检查是否应该重试
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Literal

# Type aliases
# 中文: 类型别名
RetryOn = tuple[type[Exception], ...] | Callable[[Exception], bool]
"""Type for specifying which exceptions to retry on.

Can be either:
- A tuple of exception types to retry on (based on `isinstance` checks)
- A callable that takes an exception and returns `True` if it should be retried

中文翻译:
用于指定要重试的异常的类型。
可以是：
- 要重试的异常类型元组（基于“isinstance”检查）
- 一个可调用的，如果应该重试，则接受异常并返回“True”
"""

OnFailure = Literal["error", "continue"] | Callable[[Exception], str]
"""Type for specifying failure handling behavior.

Can be either:
- A literal action string (`'error'` or `'continue'`)
    - `'error'`: Re-raise the exception, stopping agent execution.
    - `'continue'`: Inject a message with the error details, allowing the agent to continue.
       For tool retries, a `ToolMessage` with the error details will be injected.
       For model retries, an `AIMessage` with the error details will be returned.
- A callable that takes an exception and returns a string for error message content

中文翻译:
用于指定故障处理行为的类型。
可以是：
- 文字操作字符串（“错误”或“继续”）
    - `'error'`：重新引发异常，停止代理执行。
    - `'继续'`：注入包含错误详细信息的消息，允许代理继续。
       对于工具重试，将注入包含错误详细信息的“ToolMessage”。
       对于模型重试，将返回包含错误详细信息的“AIMessage”。
- 一个可调用的函数，它接受异常并返回错误消息内容的字符串
"""


def validate_retry_params(
    max_retries: int,
    initial_delay: float,
    max_delay: float,
    backoff_factor: float,
) -> None:
    """Validate retry parameters.

    Args:
        max_retries: Maximum number of retry attempts.
        initial_delay: Initial delay in seconds before first retry.
        max_delay: Maximum delay in seconds between retries.
        backoff_factor: Multiplier for exponential backoff.

    Raises:
        ValueError: If any parameter is invalid (negative values).
    

    中文翻译:
    验证重试参数。
    参数：
        max_retries：最大重试次数。
        初始延迟：第一次重试之前的初始延迟（以秒为单位）。
        max_delay：重试之间的最大延迟（以秒为单位）。
        backoff_factor：指数退避的乘数。
    加薪：
        ValueError：如果任何参数无效（负值）。"""
    if max_retries < 0:
        msg = "max_retries must be >= 0"
        raise ValueError(msg)
    if initial_delay < 0:
        msg = "initial_delay must be >= 0"
        raise ValueError(msg)
    if max_delay < 0:
        msg = "max_delay must be >= 0"
        raise ValueError(msg)
    if backoff_factor < 0:
        msg = "backoff_factor must be >= 0"
        raise ValueError(msg)


def should_retry_exception(
    exc: Exception,
    retry_on: RetryOn,
) -> bool:
    """Check if an exception should trigger a retry.

    Args:
        exc: The exception that occurred.
        retry_on: Either a tuple of exception types to retry on, or a callable
            that takes an exception and returns `True` if it should be retried.

    Returns:
        `True` if the exception should be retried, `False` otherwise.
    

    中文翻译:
    检查异常是否应触发重试。
    参数：
        exc：发生的异常。
        retry_on：要重试的异常类型元组，或者可调用的
            如果应该重试，则接受异常并返回“True”。
    返回：
        如果应该重试异常，则为“True”，否则为“False”。"""
    if callable(retry_on):
        return retry_on(exc)
    return isinstance(exc, retry_on)


def calculate_delay(
    retry_number: int,
    *,
    backoff_factor: float,
    initial_delay: float,
    max_delay: float,
    jitter: bool,
) -> float:
    """Calculate delay for a retry attempt with exponential backoff and optional jitter.

    Args:
        retry_number: The retry attempt number (0-indexed).
        backoff_factor: Multiplier for exponential backoff.

            Set to `0.0` for constant delay.
        initial_delay: Initial delay in seconds before first retry.
        max_delay: Maximum delay in seconds between retries.

            Caps exponential backoff growth.
        jitter: Whether to add random jitter to delay to avoid thundering herd.

    Returns:
        Delay in seconds before next retry.
    

    中文翻译:
    使用指数退避和可选抖动计算重试尝试的延迟。
    参数：
        retry_number：重试尝试次数（从 0 开始索引）。
        backoff_factor：指数退避的乘数。
            设置为“0.0”以获得恒定延迟。
        初始延迟：第一次重试之前的初始延迟（以秒为单位）。
        max_delay：重试之间的最大延迟（以秒为单位）。
            限制指数退避增长。
        jitter：是否添加随机抖动来延迟以避免惊群。
    返回：
        下次重试之前的延迟（以秒为单位）。"""
    if backoff_factor == 0.0:
        delay = initial_delay
    else:
        delay = initial_delay * (backoff_factor**retry_number)

    # Cap at max_delay
    # 中文: max_delay 上限
    delay = min(delay, max_delay)

    if jitter and delay > 0:
        jitter_amount = delay * 0.25  # ±25% jitter
        delay += random.uniform(-jitter_amount, jitter_amount)  # noqa: S311
        # Ensure delay is not negative after jitter
        # 中文: 确保抖动后延迟不为负
        delay = max(0, delay)

    return delay
