"""Runnable 配置工具模块。

本模块提供 `RunnableConfig` 类型定义和相关的配置管理工具函数。
配置用于控制 Runnable 的执行行为，包括回调、标签、元数据等。

核心类型:
---------
**RunnableConfig**: Runnable 的配置字典，包含以下字段:
- tags: 标签列表，用于过滤和追踪
- metadata: 元数据字典，用于存储额外信息
- callbacks: 回调处理器，用于监控和日志
- run_name: 运行名称，用于追踪器
- max_concurrency: 最大并发数
- recursion_limit: 递归深度限制
- configurable: 可配置字段的运行时值
- run_id: 唯一运行 ID

核心函数:
---------
- **ensure_config**: 确保配置字典包含所有必需的键
- **get_config_list**: 从单个或多个配置创建配置列表
- **patch_config**: 用新值更新配置
- **merge_configs**: 合并多个配置
- **get_callback_manager_for_config**: 从配置获取回调管理器
- **run_in_executor**: 在执行器中运行函数

使用示例:
---------
>>> from langchain_core.runnables import RunnableConfig
>>>
>>> # 创建配置
>>> config: RunnableConfig = {
...     "tags": ["production", "high-priority"],
...     "metadata": {"user_id": "123"},
...     "callbacks": [my_callback],
...     "max_concurrency": 5,
... }
>>>
>>> # 使用配置调用 Runnable
>>> result = my_runnable.invoke(input, config=config)

配置传递:
---------
配置会自动传递给子 Runnable，某些字段（如 tags 和 metadata）会被继承:

>>> chain = prompt | llm | parser
>>> chain.invoke(input, config={"tags": ["test"]})
# 链中的所有组件都会接收到 "test" 标签
"""

from __future__ import annotations

import asyncio

# uuid 不能移到 TYPE_CHECKING 中，因为 RunnableConfig 用于 Pydantic 模型
import uuid  # noqa: TC003
import warnings
from collections.abc import Awaitable, Callable, Generator, Iterable, Iterator, Sequence
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import Context, ContextVar, Token, copy_context
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    ParamSpec,
    TypeVar,
    cast,
)

from langsmith.run_helpers import _set_tracing_context, get_tracing_context
from typing_extensions import TypedDict

from langchain_core.callbacks.manager import AsyncCallbackManager, CallbackManager
from langchain_core.runnables.utils import (
    Input,
    Output,
    accepts_config,
    accepts_run_manager,
)
from langchain_core.tracers.langchain import LangChainTracer

if TYPE_CHECKING:
    from langchain_core.callbacks.base import BaseCallbackManager, Callbacks
    from langchain_core.callbacks.manager import (
        AsyncCallbackManagerForChainRun,
        CallbackManagerForChainRun,
    )
else:
    # Pydantic 通过 TypedDict 验证，但回调需要更新前向引用
    Callbacks = list | Any | None


class EmptyDict(TypedDict, total=False):
    """空字典类型。"""


class RunnableConfig(TypedDict, total=False):
    """Runnable 的配置类型。

    这是一个 TypedDict，定义了所有可用的配置选项。
    配置用于控制 Runnable 的执行行为，并会自动传递给子 Runnable。

    使用示例:
        ```python
        config: RunnableConfig = {
            "tags": ["production"],
            "metadata": {"user_id": "123"},
            "max_concurrency": 5,
        }
        result = runnable.invoke(input, config=config)
        ```

    参考文档:
        https://reference.langchain.com/python/langchain_core/runnables/#langchain_core.runnables.RunnableConfig
    """

    tags: list[str]
    """此调用及任何子调用的标签。

    标签可用于过滤和分组调用。
    标签会传递给所有回调处理器。

    示例:
        ```python
        config = {"tags": ["production", "high-priority"]}
        ```
    """

    metadata: dict[str, Any]
    """此调用及任何子调用的元数据。

    键应为字符串，值应为 JSON 可序列化的。
    元数据会传递给 handle*Start 回调。
    """

    callbacks: Callbacks
    """此调用及任何子调用的回调处理器。

    可以是:
    - 回调处理器列表
    - CallbackManager 实例
    - None
    """

    run_name: str
    """此调用的追踪器运行名称。

    默认为类名。
    """

    max_concurrency: int | None
    """最大并行调用数。

    如果未提供，默认使用 ThreadPoolExecutor 的默认值。
    """

    recursion_limit: int
    """最大递归次数。

    如果未提供，默认为 25。
    这用于防止无限递归。
    """

    configurable: dict[str, Any]
    """可配置字段的运行时值。

    用于通过 `configurable_fields` 或 `configurable_alternatives`
    设置为可配置的属性的运行时值。

    示例:
        ```python
        config = {"configurable": {"temperature": 0.5}}
        ```
    """

    run_id: uuid.UUID | None
    """此调用的唯一追踪器运行 ID。

    如果未提供，将生成新的 UUID。
    """


# 配置键列表
CONFIG_KEYS = [
    "tags",
    "metadata",
    "callbacks",
    "run_name",
    "max_concurrency",
    "recursion_limit",
    "configurable",
    "run_id",
]

# 可复制的键（这些键的值在传递时会被复制）
COPIABLE_KEYS = [
    "tags",
    "metadata",
    "callbacks",
    "configurable",
]

# 默认递归限制
DEFAULT_RECURSION_LIMIT = 25


# 子 Runnable 配置的上下文变量
var_child_runnable_config: ContextVar[RunnableConfig | None] = ContextVar(
    "child_runnable_config", default=None
)


# 这个函数在 langgraph 中导入和使用，请勿更改
def _set_config_context(
    config: RunnableConfig,
) -> tuple[Token[RunnableConfig | None], dict[str, Any] | None]:
    """设置子 Runnable 配置和追踪上下文。

    Args:
        config: 要设置的配置。

    Returns:
        重置配置的令牌和之前的追踪上下文。
    """
    config_token = var_child_runnable_config.set(config)
    current_context = None
    if (
        (callbacks := config.get("callbacks"))
        and (
            parent_run_id := getattr(callbacks, "parent_run_id", None)
        )  # 是回调管理器
        and (
            tracer := next(
                (
                    handler
                    for handler in getattr(callbacks, "handlers", [])
                    if isinstance(handler, LangChainTracer)
                ),
                None,
            )
        )
        and (run := tracer.run_map.get(str(parent_run_id)))
    ):
        current_context = get_tracing_context()
        _set_tracing_context({"parent": run})
    return config_token, current_context


@contextmanager
def set_config_context(config: RunnableConfig) -> Generator[Context, None, None]:
    """设置子 Runnable 配置和追踪上下文。

    这是一个上下文管理器，用于在执行期间设置配置上下文。

    Args:
        config: 要设置的配置。

    Yields:
        配置上下文。

    使用示例:
        ```python
        with set_config_context(config) as ctx:
            # 在此上下文中执行的代码将使用此配置
            pass
        ```
    """
    ctx = copy_context()
    config_token, _ = ctx.run(_set_config_context, config)
    try:
        yield ctx
    finally:
        ctx.run(var_child_runnable_config.reset, config_token)
        ctx.run(
            _set_tracing_context,
            {
                "parent": None,
                "project_name": None,
                "tags": None,
                "metadata": None,
                "enabled": None,
                "client": None,
            },
        )


def ensure_config(config: RunnableConfig | None = None) -> RunnableConfig:
    """确保配置是包含所有键的字典。

    如果配置为 None 或缺少某些键，将使用默认值填充。
    还会从上下文变量中继承配置。

    Args:
        config: 要确保的配置。

    Returns:
        确保后的配置，包含所有必需的键。

    使用示例:
        ```python
        # 传入 None 时返回默认配置
        config = ensure_config(None)

        # 传入部分配置时补充缺失的键
        config = ensure_config({"tags": ["test"]})
        ```
    """
    empty = RunnableConfig(
        tags=[],
        metadata={},
        callbacks=None,
        recursion_limit=DEFAULT_RECURSION_LIMIT,
        configurable={},
    )
    # 从上下文变量继承配置
    if var_config := var_child_runnable_config.get():
        empty.update(
            cast(
                "RunnableConfig",
                {
                    k: v.copy() if k in COPIABLE_KEYS else v  # type: ignore[attr-defined]
                    for k, v in var_config.items()
                    if v is not None
                },
            )
        )
    # 合并用户提供的配置
    if config is not None:
        empty.update(
            cast(
                "RunnableConfig",
                {
                    k: v.copy() if k in COPIABLE_KEYS else v  # type: ignore[attr-defined]
                    for k, v in config.items()
                    if v is not None and k in CONFIG_KEYS
                },
            )
        )
    # 将不在 CONFIG_KEYS 中的键移到 configurable 中
    if config is not None:
        for k, v in config.items():
            if k not in CONFIG_KEYS and v is not None:
                empty["configurable"][k] = v
    # 将简单类型的 configurable 值复制到 metadata
    for key, value in empty.get("configurable", {}).items():
        if (
            not key.startswith("__")
            and isinstance(value, (str, int, float, bool))
            and key not in empty["metadata"]
            and key != "api_key"
        ):
            empty["metadata"][key] = value
    return empty


def get_config_list(
    config: RunnableConfig | Sequence[RunnableConfig] | None, length: int
) -> list[RunnableConfig]:
    """从单个配置或配置列表获取配置列表。

    这对于重写 batch() 或 abatch() 的子类很有用。

    Args:
        config: 配置或配置列表。
        length: 列表的长度。

    Returns:
        配置列表。

    Raises:
        ValueError: 如果列表长度与输入长度不匹配。
    """
    if length < 0:
        msg = f"length 必须 >= 0，但得到 {length}"
        raise ValueError(msg)
    if isinstance(config, Sequence) and len(config) != length:
        msg = (
            f"config 必须是与输入长度相同的列表，"
            f"但得到 {len(config)} 个配置用于 {length} 个输入"
        )
        raise ValueError(msg)

    if isinstance(config, Sequence):
        return list(map(ensure_config, config))
    # 如果提供了 run_id 且长度大于 1，发出警告
    if length > 1 and isinstance(config, dict) and config.get("run_id") is not None:
        warnings.warn(
            "提供的 run_id 仅用于批量的第一个元素。",
            category=RuntimeWarning,
            stacklevel=3,
        )
        subsequent = cast(
            "RunnableConfig", {k: v for k, v in config.items() if k != "run_id"}
        )
        return [
            ensure_config(subsequent) if i else ensure_config(config)
            for i in range(length)
        ]
    return [ensure_config(config) for i in range(length)]


def patch_config(
    config: RunnableConfig | None,
    *,
    callbacks: BaseCallbackManager | None = None,
    recursion_limit: int | None = None,
    max_concurrency: int | None = None,
    run_name: str | None = None,
    configurable: dict[str, Any] | None = None,
) -> RunnableConfig:
    """用新值更新配置。

    这会创建一个新的配置字典，不会修改原始配置。

    Args:
        config: 要更新的配置。
        callbacks: 要设置的回调。
        recursion_limit: 要设置的递归限制。
        max_concurrency: 要设置的最大并发数。
        run_name: 要设置的运行名称。
        configurable: 要设置的可配置字段。

    Returns:
        更新后的配置。
    """
    config = ensure_config(config)
    if callbacks is not None:
        # 如果替换回调，需要取消设置 run_name
        # 因为那应该只适用于原始回调的同一运行
        config["callbacks"] = callbacks
        if "run_name" in config:
            del config["run_name"]
        if "run_id" in config:
            del config["run_id"]
    if recursion_limit is not None:
        config["recursion_limit"] = recursion_limit
    if max_concurrency is not None:
        config["max_concurrency"] = max_concurrency
    if run_name is not None:
        config["run_name"] = run_name
    if configurable is not None:
        config["configurable"] = {**config.get("configurable", {}), **configurable}
    return config


def merge_configs(*configs: RunnableConfig | None) -> RunnableConfig:
    """合并多个配置为一个。

    合并规则:
    - tags: 合并并去重
    - metadata: 后面的覆盖前面的
    - configurable: 后面的覆盖前面的
    - callbacks: 智能合并

    Args:
        *configs: 要合并的配置。

    Returns:
        合并后的配置。
    """
    base: RunnableConfig = {}
    for config in (ensure_config(c) for c in configs if c is not None):
        for key in config:
            if key == "metadata":
                base["metadata"] = {
                    **base.get("metadata", {}),
                    **(config.get("metadata") or {}),
                }
            elif key == "tags":
                base["tags"] = sorted(
                    set(base.get("tags", []) + (config.get("tags") or [])),
                )
            elif key == "configurable":
                base["configurable"] = {
                    **base.get("configurable", {}),
                    **(config.get("configurable") or {}),
                }
            elif key == "callbacks":
                base_callbacks = base.get("callbacks")
                these_callbacks = config["callbacks"]
                # 回调可以是 None、list[handler] 或 manager
                # 因此合并两个回调值有 6 种情况
                if isinstance(these_callbacks, list):
                    if base_callbacks is None:
                        base["callbacks"] = these_callbacks.copy()
                    elif isinstance(base_callbacks, list):
                        base["callbacks"] = base_callbacks + these_callbacks
                    else:
                        # base_callbacks 是管理器
                        mngr = base_callbacks.copy()
                        for callback in these_callbacks:
                            mngr.add_handler(callback, inherit=True)
                        base["callbacks"] = mngr
                elif these_callbacks is not None:
                    # these_callbacks 是管理器
                    if base_callbacks is None:
                        base["callbacks"] = these_callbacks.copy()
                    elif isinstance(base_callbacks, list):
                        mngr = these_callbacks.copy()
                        for callback in base_callbacks:
                            mngr.add_handler(callback, inherit=True)
                        base["callbacks"] = mngr
                    else:
                        # base_callbacks 也是管理器
                        base["callbacks"] = base_callbacks.merge(these_callbacks)
            elif key == "recursion_limit":
                if config["recursion_limit"] != DEFAULT_RECURSION_LIMIT:
                    base["recursion_limit"] = config["recursion_limit"]
            elif key in COPIABLE_KEYS and config[key] is not None:  # type: ignore[literal-required]
                base[key] = config[key].copy()  # type: ignore[literal-required]
            else:
                base[key] = config[key] or base.get(key)  # type: ignore[literal-required]
    return base


def call_func_with_variable_args(
    func: Callable[[Input], Output]
    | Callable[[Input, RunnableConfig], Output]
    | Callable[[Input, CallbackManagerForChainRun], Output]
    | Callable[[Input, CallbackManagerForChainRun, RunnableConfig], Output],
    input: Input,
    config: RunnableConfig,
    run_manager: CallbackManagerForChainRun | None = None,
    **kwargs: Any,
) -> Output:
    """调用可能接受 run_manager 和/或 config 的函数。

    这个辅助函数会检查函数签名，只传递它接受的参数。

    Args:
        func: 要调用的函数。
        input: 函数的输入。
        config: 要传递给函数的配置。
        run_manager: 要传递给函数的运行管理器。
        **kwargs: 要传递给函数的其他关键字参数。

    Returns:
        函数的输出。
    """
    if accepts_config(func):
        if run_manager is not None:
            kwargs["config"] = patch_config(config, callbacks=run_manager.get_child())
        else:
            kwargs["config"] = config
    if run_manager is not None and accepts_run_manager(func):
        kwargs["run_manager"] = run_manager
    return func(input, **kwargs)  # type: ignore[call-arg]


def acall_func_with_variable_args(
    func: Callable[[Input], Awaitable[Output]]
    | Callable[[Input, RunnableConfig], Awaitable[Output]]
    | Callable[[Input, AsyncCallbackManagerForChainRun], Awaitable[Output]]
    | Callable[
        [Input, AsyncCallbackManagerForChainRun, RunnableConfig], Awaitable[Output]
    ],
    input: Input,
    config: RunnableConfig,
    run_manager: AsyncCallbackManagerForChainRun | None = None,
    **kwargs: Any,
) -> Awaitable[Output]:
    """异步调用可能接受 run_manager 和/或 config 的函数。

    与 call_func_with_variable_args 相同，但用于异步函数。
    """
    if accepts_config(func):
        if run_manager is not None:
            kwargs["config"] = patch_config(config, callbacks=run_manager.get_child())
        else:
            kwargs["config"] = config
    if run_manager is not None and accepts_run_manager(func):
        kwargs["run_manager"] = run_manager
    return func(input, **kwargs)  # type: ignore[call-arg]


def get_callback_manager_for_config(config: RunnableConfig) -> CallbackManager:
    """从配置获取回调管理器。

    Args:
        config: 配置。

    Returns:
        回调管理器。
    """
    return CallbackManager.configure(
        inheritable_callbacks=config.get("callbacks"),
        inheritable_tags=config.get("tags"),
        inheritable_metadata=config.get("metadata"),
    )


def get_async_callback_manager_for_config(
    config: RunnableConfig,
) -> AsyncCallbackManager:
    """从配置获取异步回调管理器。

    Args:
        config: 配置。

    Returns:
        异步回调管理器。
    """
    return AsyncCallbackManager.configure(
        inheritable_callbacks=config.get("callbacks"),
        inheritable_tags=config.get("tags"),
        inheritable_metadata=config.get("metadata"),
    )


P = ParamSpec("P")
T = TypeVar("T")


class ContextThreadPoolExecutor(ThreadPoolExecutor):
    """将上下文复制到子线程的 ThreadPoolExecutor。

    这确保了在线程池中执行的任务可以访问正确的上下文变量，
    例如配置和追踪信息。
    """

    def submit(  # type: ignore[override]
        self,
        func: Callable[P, T],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Future[T]:
        """向执行器提交函数。

        Args:
            func: 要提交的函数。
            *args: 函数的位置参数。
            **kwargs: 函数的关键字参数。

        Returns:
            函数的 Future。
        """
        return super().submit(
            cast("Callable[..., T]", partial(copy_context().run, func, *args, **kwargs))
        )

    def map(
        self,
        fn: Callable[..., T],
        *iterables: Iterable[Any],
        **kwargs: Any,
    ) -> Iterator[T]:
        """将函数映射到多个可迭代对象。

        Args:
            fn: 要映射的函数。
            *iterables: 要映射的可迭代对象。
            **kwargs: 额外参数（timeout、chunksize 等）。

        Returns:
            映射函数的迭代器。
        """
        contexts = [copy_context() for _ in range(len(iterables[0]))]  # type: ignore[arg-type]

        def _wrapped_fn(*args: Any) -> T:
            return contexts.pop().run(fn, *args)

        return super().map(
            _wrapped_fn,
            *iterables,
            **kwargs,
        )


@contextmanager
def get_executor_for_config(
    config: RunnableConfig | None,
) -> Generator[Executor, None, None]:
    """获取配置的执行器。

    Args:
        config: 配置。

    Yields:
        执行器。
    """
    config = config or {}
    with ContextThreadPoolExecutor(
        max_workers=config.get("max_concurrency")
    ) as executor:
        yield executor


async def run_in_executor(
    executor_or_config: Executor | RunnableConfig | None,
    func: Callable[P, T],
    *args: P.args,
    **kwargs: P.kwargs,
) -> T:
    """在执行器中运行函数。

    这用于在异步上下文中运行同步函数，而不阻塞事件循环。

    Args:
        executor_or_config: 执行器或配置。
            如果是配置或 None，将使用默认执行器。
        func: 要运行的函数。
        *args: 函数的位置参数。
        **kwargs: 函数的关键字参数。

    Returns:
        函数的输出。

    使用示例:
        ```python
        async def async_main():
            result = await run_in_executor(
                None,
                some_sync_function,
                arg1,
                arg2,
            )
        ```
    """

    def wrapper() -> T:
        try:
            return func(*args, **kwargs)
        except StopIteration as exc:
            # StopIteration 不能设置在 asyncio.Future 上
            # 它会引发 TypeError 并使 Future 永远挂起
            # 所以我们需要将其转换为 RuntimeError
            raise RuntimeError from exc

    if executor_or_config is None or isinstance(executor_or_config, dict):
        # 使用默认执行器，上下文从当前上下文复制
        return await asyncio.get_running_loop().run_in_executor(
            None,
            cast("Callable[..., T]", partial(copy_context().run, wrapper)),
        )

    return await asyncio.get_running_loop().run_in_executor(executor_or_config, wrapper)
