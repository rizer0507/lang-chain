"""路由 Runnable 模块。

本模块提供 `RouterRunnable`，它可以根据输入的键将请求路由到不同的 Runnable。

核心概念:
---------
RouterRunnable 类似于一个路由器或分发器，它:
1. 接收一个包含 `key` 和 `input` 的字典
2. 根据 `key` 选择对应的 Runnable
3. 将 `input` 传递给选中的 Runnable
4. 返回该 Runnable 的输出

使用场景:
---------
- 根据用户意图路由到不同的处理链
- 实现多分支逻辑
- 动态选择不同的处理器

使用示例:
---------
>>> from langchain_core.runnables.router import RouterRunnable
>>> from langchain_core.runnables import RunnableLambda
>>>
>>> # 定义不同的处理函数
>>> add = RunnableLambda(func=lambda x: x + 1)
>>> square = RunnableLambda(func=lambda x: x ** 2)
>>> double = RunnableLambda(func=lambda x: x * 2)
>>>
>>> # 创建路由器
>>> router = RouterRunnable(runnables={
...     "add": add,
...     "square": square,
...     "double": double,
... })
>>>
>>> # 使用路由器
>>> router.invoke({"key": "square", "input": 3})
9
>>> router.invoke({"key": "add", "input": 3})
4
>>> router.invoke({"key": "double", "input": 3})
6

与 RunnableBranch 的区别:
---------
| 特性 | RouterRunnable | RunnableBranch |
|------|----------------|----------------|
| 路由方式 | 显式指定 key | 条件函数求值 |
| 灵活性 | 需要预先定义所有路由 | 支持默认分支 |
| 使用场景 | 已知路由键 | 动态条件判断 |
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)

from pydantic import ConfigDict
from typing_extensions import TypedDict, override

from langchain_core.runnables.base import (
    Runnable,
    RunnableSerializable,
    coerce_to_runnable,
)
from langchain_core.runnables.config import (
    RunnableConfig,
    get_config_list,
    get_executor_for_config,
)
from langchain_core.runnables.utils import (
    ConfigurableFieldSpec,
    Input,
    Output,
    gather_with_concurrency,
    get_unique_config_specs,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterator


class RouterInput(TypedDict):
    """路由器输入类型。

    这是 RouterRunnable 期望的输入格式。
    """

    key: str
    """路由键，用于选择要执行的 Runnable。"""

    input: Any
    """传递给选中 Runnable 的输入。"""


class RouterRunnable(RunnableSerializable[RouterInput, Output]):
    """根据输入键路由到一组 Runnable 的路由器。

    返回选中 Runnable 的输出。

    工作原理:
    1. 接收 `{"key": str, "input": Any}` 格式的输入
    2. 根据 `key` 从 `runnables` 字典中选择对应的 Runnable
    3. 用 `input` 调用选中的 Runnable
    4. 返回该 Runnable 的输出

    属性:
    -----
    runnables : Mapping[str, Runnable]
        键到 Runnable 的映射字典

    使用示例:
        ```python
        from langchain_core.runnables.router import RouterRunnable
        from langchain_core.runnables import RunnableLambda

        add = RunnableLambda(func=lambda x: x + 1)
        square = RunnableLambda(func=lambda x: x ** 2)

        router = RouterRunnable(runnables={"add": add, "square": square})
        router.invoke({"key": "square", "input": 3})
        # 输出: 9
        ```
    """

    runnables: Mapping[str, Runnable[Any, Output]]
    """键到 Runnable 的映射。

    输入中的 `key` 会用于从这个字典中查找要执行的 Runnable。
    """

    @property
    @override
    def config_specs(self) -> list[ConfigurableFieldSpec]:
        """获取所有子 Runnable 的配置规格。"""
        return get_unique_config_specs(
            spec for step in self.runnables.values() for spec in step.config_specs
        )

    def __init__(
        self,
        runnables: Mapping[str, Runnable[Any, Output] | Callable[[Any], Output]],
    ) -> None:
        """创建 RouterRunnable。

        Args:
            runnables: 键到 Runnable 对象（或可调用对象）的映射。
                可调用对象会自动转换为 RunnableLambda。
        """
        super().__init__(
            runnables={key: coerce_to_runnable(r) for key, r in runnables.items()}
        )

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @classmethod
    @override
    def is_lc_serializable(cls) -> bool:
        """返回 True，表示此类可序列化。"""
        return True

    @classmethod
    @override
    def get_lc_namespace(cls) -> list[str]:
        """获取 LangChain 对象的命名空间。

        Returns:
            `["langchain", "schema", "runnable"]`
        """
        return ["langchain", "schema", "runnable"]

    @override
    def invoke(
        self, input: RouterInput, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Output:
        """同步调用路由器。

        Args:
            input: 包含 `key` 和 `input` 的路由器输入。
            config: 运行配置。
            **kwargs: 额外参数。

        Returns:
            选中 Runnable 的输出。

        Raises:
            ValueError: 如果 key 对应的 Runnable 不存在。
        """
        key = input["key"]
        actual_input = input["input"]
        if key not in self.runnables:
            msg = f"没有与键 '{key}' 关联的 Runnable"
            raise ValueError(msg)

        runnable = self.runnables[key]
        return runnable.invoke(actual_input, config)

    @override
    async def ainvoke(
        self,
        input: RouterInput,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> Output:
        """异步调用路由器。"""
        key = input["key"]
        actual_input = input["input"]
        if key not in self.runnables:
            msg = f"没有与键 '{key}' 关联的 Runnable"
            raise ValueError(msg)

        runnable = self.runnables[key]
        return await runnable.ainvoke(actual_input, config)

    @override
    def batch(
        self,
        inputs: list[RouterInput],
        config: RunnableConfig | list[RunnableConfig] | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any | None,
    ) -> list[Output]:
        """批量调用路由器。

        Args:
            inputs: 路由器输入列表。
            config: 运行配置（单个或列表）。
            return_exceptions: 如果为 True，异常会作为结果返回而不是抛出。
            **kwargs: 额外参数。

        Returns:
            输出列表。
        """
        if not inputs:
            return []

        keys = [input_["key"] for input_ in inputs]
        actual_inputs = [input_["input"] for input_ in inputs]
        if any(key not in self.runnables for key in keys):
            msg = "一个或多个键没有对应的 Runnable"
            raise ValueError(msg)

        def invoke(
            runnable: Runnable[Input, Output], input_: Input, config: RunnableConfig
        ) -> Output | Exception:
            if return_exceptions:
                try:
                    return runnable.invoke(input_, config, **kwargs)
                except Exception as e:
                    return e
            else:
                return runnable.invoke(input_, config, **kwargs)

        runnables = [self.runnables[key] for key in keys]
        configs = get_config_list(config, len(inputs))
        with get_executor_for_config(configs[0]) as executor:
            return cast(
                "list[Output]",
                list(executor.map(invoke, runnables, actual_inputs, configs)),
            )

    @override
    async def abatch(
        self,
        inputs: list[RouterInput],
        config: RunnableConfig | list[RunnableConfig] | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any | None,
    ) -> list[Output]:
        """异步批量调用路由器。"""
        if not inputs:
            return []

        keys = [input_["key"] for input_ in inputs]
        actual_inputs = [input_["input"] for input_ in inputs]
        if any(key not in self.runnables for key in keys):
            msg = "一个或多个键没有对应的 Runnable"
            raise ValueError(msg)

        async def ainvoke(
            runnable: Runnable[Input, Output], input_: Input, config: RunnableConfig
        ) -> Output | Exception:
            if return_exceptions:
                try:
                    return await runnable.ainvoke(input_, config, **kwargs)
                except Exception as e:
                    return e
            else:
                return await runnable.ainvoke(input_, config, **kwargs)

        runnables = [self.runnables[key] for key in keys]
        configs = get_config_list(config, len(inputs))
        return await gather_with_concurrency(
            configs[0].get("max_concurrency"),
            *map(ainvoke, runnables, actual_inputs, configs),
        )

    @override
    def stream(
        self,
        input: RouterInput,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> Iterator[Output]:
        """流式调用路由器。

        Args:
            input: 路由器输入。
            config: 运行配置。
            **kwargs: 额外参数。

        Yields:
            选中 Runnable 的流式输出块。
        """
        key = input["key"]
        actual_input = input["input"]
        if key not in self.runnables:
            msg = f"没有与键 '{key}' 关联的 Runnable"
            raise ValueError(msg)

        runnable = self.runnables[key]
        yield from runnable.stream(actual_input, config)

    @override
    async def astream(
        self,
        input: RouterInput,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> AsyncIterator[Output]:
        """异步流式调用路由器。"""
        key = input["key"]
        actual_input = input["input"]
        if key not in self.runnables:
            msg = f"没有与键 '{key}' 关联的 Runnable"
            raise ValueError(msg)

        runnable = self.runnables[key]
        async for output in runnable.astream(actual_input, config):
            yield output
