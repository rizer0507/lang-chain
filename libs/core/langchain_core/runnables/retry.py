"""可重试的 Runnable 模块。

本模块提供 `RunnableRetry`，它为 Runnable 添加重试逻辑。
当调用可能因暂时性错误而失败时（如网络错误），重试特别有用。

核心类:
--------
**RunnableRetry**: 为任何 Runnable 添加重试功能

重试参数:
---------
- **retry_exception_types**: 要重试的异常类型元组
- **wait_exponential_jitter**: 是否使用指数退避加抖动
- **max_attempt_number**: 最大尝试次数
- **exponential_jitter_params**: 指数退避参数

使用示例:
---------
>>> from langchain_core.runnables import RunnableLambda
>>>
>>> def flaky_function(x):
...     '''可能失败的函数'''
...     import random
...     if random.random() < 0.5:
...         raise ValueError("随机失败")
...     return x * 2
>>>
>>> # 方式1: 使用 with_retry 方法（推荐）
>>> runnable = RunnableLambda(flaky_function)
>>> runnable_with_retry = runnable.with_retry(
...     retry_if_exception_type=(ValueError,),
...     wait_exponential_jitter=True,
...     stop_after_attempt=3,
... )
>>>
>>> # 方式2: 直接使用 RunnableRetry
>>> from langchain_core.runnables.retry import RunnableRetry
>>> runnable_with_retry = RunnableRetry(
...     bound=runnable,
...     retry_exception_types=(ValueError,),
...     max_attempt_number=3,
...     wait_exponential_jitter=True,
... )

最佳实践:
---------
**将重试范围保持尽可能小**

```python
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

template = PromptTemplate.from_template("讲一个关于{topic}的笑话")
model = ChatOpenAI(temperature=0.5)

# ✅ 好: 只重试可能失败的部分
chain = template | model.with_retry()

# ❌ 不好: 重试整个链
chain = template | model
retryable_chain = chain.with_retry()
```
"""

from typing import (
    TYPE_CHECKING,
    Any,
    TypeVar,
    cast,
)

from tenacity import (
    AsyncRetrying,
    RetryCallState,
    RetryError,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)
from typing_extensions import TypedDict, override

from langchain_core.runnables.base import RunnableBindingBase
from langchain_core.runnables.config import RunnableConfig, patch_config
from langchain_core.runnables.utils import Input, Output

if TYPE_CHECKING:
    from langchain_core.callbacks.manager import (
        AsyncCallbackManagerForChainRun,
        CallbackManagerForChainRun,
    )

    T = TypeVar("T", CallbackManagerForChainRun, AsyncCallbackManagerForChainRun)
U = TypeVar("U")


class ExponentialJitterParams(TypedDict, total=False):
    """`tenacity.wait_exponential_jitter` 的参数。

    这些参数控制指数退避的行为。
    """

    initial: float
    """初始等待时间（秒）。"""

    max: float
    """最大等待时间（秒）。"""

    exp_base: float
    """指数退避的底数。"""

    jitter: float
    """随机抖动，从 random.uniform(0, jitter) 采样。"""


class RunnableRetry(RunnableBindingBase[Input, Output]):  # type: ignore[no-redef]
    """失败时重试 Runnable 的包装器。

    RunnableRetry 可用于为任何继承自 Runnable 的对象添加重试逻辑。

    这种重试对于可能因暂时性错误而失败的网络调用特别有用，
    例如 API 速率限制、服务器过载等。

    RunnableRetry 实现为 RunnableBinding。最简单的使用方式是
    通过所有 Runnable 上的 `.with_retry()` 方法。

    属性:
    -----
    retry_exception_types : tuple[type[BaseException], ...]
        要重试的异常类型。默认重试所有异常。
    wait_exponential_jitter : bool
        是否在指数退避中添加抖动。
    exponential_jitter_params : ExponentialJitterParams | None
        指数退避的参数。
    max_attempt_number : int
        最大重试次数。

    使用示例:
        ```python
        import time

        def foo(input) -> None:
            '''会抛出异常的假函数'''
            raise ValueError(f"调用失败，时间: {time.time()}")

        runnable = RunnableLambda(foo)

        runnable_with_retries = runnable.with_retry(
            retry_if_exception_type=(ValueError,),  # 只重试 ValueError
            wait_exponential_jitter=True,  # 添加抖动
            stop_after_attempt=2,  # 最多尝试2次
            exponential_jitter_params={"initial": 2},  # 自定义退避参数
        )

        # 上述方法调用等同于:
        runnable_with_retries = RunnableRetry(
            bound=runnable,
            retry_exception_types=(ValueError,),
            max_attempt_number=2,
            wait_exponential_jitter=True,
            exponential_jitter_params={"initial": 2},
        )
        ```

    注意事项:
        - 这个逻辑可以用于重试任何 Runnable，包括 Runnable 链
        - 但最佳实践是将重试范围保持尽可能小
        - 例如，如果有一个 Runnable 链，应该只重试可能失败的那个，
          而不是整个链
    """

    retry_exception_types: tuple[type[BaseException], ...] = (Exception,)
    """要重试的异常类型。默认重试所有异常。

    通常应该只重试可能是暂时性的异常，如网络错误。

    建议重试的异常:
    - 所有服务器错误 (5xx)
    - 选定的客户端错误 (4xx)，如 429 Too Many Requests
    """

    wait_exponential_jitter: bool = True
    """是否在指数退避中添加抖动。

    抖动可以防止多个请求同时重试（惊群效应）。
    """

    exponential_jitter_params: ExponentialJitterParams | None = None
    """`tenacity.wait_exponential_jitter` 的参数。

    包括: `initial`（初始等待）、`max`（最大等待）、
    `exp_base`（指数底数）和 `jitter`（抖动），都是浮点值。
    """

    max_attempt_number: int = 3
    """重试 Runnable 的最大尝试次数。"""

    @property
    def _kwargs_retrying(self) -> dict[str, Any]:
        """获取 tenacity Retrying 的参数。"""
        kwargs: dict[str, Any] = {}

        if self.max_attempt_number:
            kwargs["stop"] = stop_after_attempt(self.max_attempt_number)

        if self.wait_exponential_jitter:
            kwargs["wait"] = wait_exponential_jitter(
                **(self.exponential_jitter_params or {})
            )

        if self.retry_exception_types:
            kwargs["retry"] = retry_if_exception_type(self.retry_exception_types)

        return kwargs

    def _sync_retrying(self, **kwargs: Any) -> Retrying:
        """创建同步重试器。"""
        return Retrying(**self._kwargs_retrying, **kwargs)

    def _async_retrying(self, **kwargs: Any) -> AsyncRetrying:
        """创建异步重试器。"""
        return AsyncRetrying(**self._kwargs_retrying, **kwargs)

    @staticmethod
    def _patch_config(
        config: RunnableConfig,
        run_manager: "T",
        retry_state: RetryCallState,
    ) -> RunnableConfig:
        """为重试尝试更新配置。"""
        attempt = retry_state.attempt_number
        tag = f"retry:attempt:{attempt}" if attempt > 1 else None
        return patch_config(config, callbacks=run_manager.get_child(tag))

    def _patch_config_list(
        self,
        config: list[RunnableConfig],
        run_manager: list["T"],
        retry_state: RetryCallState,
    ) -> list[RunnableConfig]:
        """为批量重试更新配置列表。"""
        return [
            self._patch_config(c, rm, retry_state)
            for c, rm in zip(config, run_manager, strict=False)
        ]

    def _invoke(
        self,
        input_: Input,
        run_manager: "CallbackManagerForChainRun",
        config: RunnableConfig,
        **kwargs: Any,
    ) -> Output:
        """带重试的同步调用内部实现。"""
        for attempt in self._sync_retrying(reraise=True):
            with attempt:
                result = super().invoke(
                    input_,
                    self._patch_config(config, run_manager, attempt.retry_state),
                    **kwargs,
                )
            if attempt.retry_state.outcome and not attempt.retry_state.outcome.failed:
                attempt.retry_state.set_result(result)
        return result

    @override
    def invoke(
        self, input: Input, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Output:
        """带重试的同步调用。"""
        return self._call_with_config(self._invoke, input, config, **kwargs)

    async def _ainvoke(
        self,
        input_: Input,
        run_manager: "AsyncCallbackManagerForChainRun",
        config: RunnableConfig,
        **kwargs: Any,
    ) -> Output:
        """带重试的异步调用内部实现。"""
        async for attempt in self._async_retrying(reraise=True):
            with attempt:
                result = await super().ainvoke(
                    input_,
                    self._patch_config(config, run_manager, attempt.retry_state),
                    **kwargs,
                )
            if attempt.retry_state.outcome and not attempt.retry_state.outcome.failed:
                attempt.retry_state.set_result(result)
        return result

    @override
    async def ainvoke(
        self, input: Input, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Output:
        """带重试的异步调用。"""
        return await self._acall_with_config(self._ainvoke, input, config, **kwargs)

    def _batch(
        self,
        inputs: list[Input],
        run_manager: list["CallbackManagerForChainRun"],
        config: list[RunnableConfig],
        **kwargs: Any,
    ) -> list[Output | Exception]:
        """带重试的批量调用内部实现。

        智能地只重试失败的输入。
        """
        results_map: dict[int, Output] = {}

        not_set: list[Output] = []
        result = not_set
        try:
            for attempt in self._sync_retrying():
                with attempt:
                    # 只重试尚未成功的输入
                    # 确定哪些原始索引仍然需要处理
                    remaining_indices = [
                        i for i in range(len(inputs)) if i not in results_map
                    ]
                    if not remaining_indices:
                        break
                    pending_inputs = [inputs[i] for i in remaining_indices]
                    pending_configs = [config[i] for i in remaining_indices]
                    pending_run_managers = [run_manager[i] for i in remaining_indices]
                    # 只对剩余元素调用底层批量处理
                    result = super().batch(
                        pending_inputs,
                        self._patch_config_list(
                            pending_configs, pending_run_managers, attempt.retry_state
                        ),
                        return_exceptions=True,
                        **kwargs,
                    )
                    # 注册成功的输入结果，映射回原始索引
                    first_exception = None
                    for offset, r in enumerate(result):
                        if isinstance(r, Exception):
                            if not first_exception:
                                first_exception = r
                            continue
                        orig_idx = remaining_indices[offset]
                        results_map[orig_idx] = r
                    # 如果有任何异常，抛出它以重试失败的输入
                    if first_exception:
                        raise first_exception
                if (
                    attempt.retry_state.outcome
                    and not attempt.retry_state.outcome.failed
                ):
                    attempt.retry_state.set_result(result)
        except RetryError as e:
            if result is not_set:
                result = cast("list[Output]", [e] * len(inputs))

        outputs: list[Output | Exception] = []
        for idx in range(len(inputs)):
            if idx in results_map:
                outputs.append(results_map[idx])
            else:
                outputs.append(result.pop(0))
        return outputs

    @override
    def batch(
        self,
        inputs: list[Input],
        config: RunnableConfig | list[RunnableConfig] | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any,
    ) -> list[Output]:
        """带重试的批量调用。"""
        return self._batch_with_config(
            self._batch, inputs, config, return_exceptions=return_exceptions, **kwargs
        )

    async def _abatch(
        self,
        inputs: list[Input],
        run_manager: list["AsyncCallbackManagerForChainRun"],
        config: list[RunnableConfig],
        **kwargs: Any,
    ) -> list[Output | Exception]:
        """带重试的异步批量调用内部实现。"""
        results_map: dict[int, Output] = {}

        not_set: list[Output] = []
        result = not_set
        try:
            async for attempt in self._async_retrying():
                with attempt:
                    # 只重试尚未成功的输入
                    remaining_indices = [
                        i for i in range(len(inputs)) if i not in results_map
                    ]
                    if not remaining_indices:
                        break
                    pending_inputs = [inputs[i] for i in remaining_indices]
                    pending_configs = [config[i] for i in remaining_indices]
                    pending_run_managers = [run_manager[i] for i in remaining_indices]
                    result = await super().abatch(
                        pending_inputs,
                        self._patch_config_list(
                            pending_configs, pending_run_managers, attempt.retry_state
                        ),
                        return_exceptions=True,
                        **kwargs,
                    )
                    first_exception = None
                    for offset, r in enumerate(result):
                        if isinstance(r, Exception):
                            if not first_exception:
                                first_exception = r
                            continue
                        orig_idx = remaining_indices[offset]
                        results_map[orig_idx] = r
                    if first_exception:
                        raise first_exception
                if (
                    attempt.retry_state.outcome
                    and not attempt.retry_state.outcome.failed
                ):
                    attempt.retry_state.set_result(result)
        except RetryError as e:
            if result is not_set:
                result = cast("list[Output]", [e] * len(inputs))

        outputs: list[Output | Exception] = []
        for idx in range(len(inputs)):
            if idx in results_map:
                outputs.append(results_map[idx])
            else:
                outputs.append(result.pop(0))
        return outputs

    @override
    async def abatch(
        self,
        inputs: list[Input],
        config: RunnableConfig | list[RunnableConfig] | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any,
    ) -> list[Output]:
        """带重试的异步批量调用。"""
        return await self._abatch_with_config(
            self._abatch, inputs, config, return_exceptions=return_exceptions, **kwargs
        )

    # stream() 和 transform() 不支持重试
    # 因为重试流式操作不太直观
