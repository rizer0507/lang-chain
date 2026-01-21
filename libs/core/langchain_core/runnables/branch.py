"""根据条件选择执行分支的 Runnable 模块。

本模块提供 `RunnableBranch`，它根据条件函数的求值结果选择要执行的分支。

核心概念:
---------
RunnableBranch 类似于编程语言中的 if-elif-else 语句：
1. 依次求值每个条件
2. 执行第一个条件为 True 的分支
3. 如果所有条件都为 False，执行默认分支

与 RouterRunnable 的区别:
---------
| 特性 | RunnableBranch | RouterRunnable |
|------|----------------|----------------|
| 路由方式 | 条件函数求值 | 显式指定 key |
| 条件数量 | 可以有多个 | 固定的键映射 |
| 默认分支 | 必须有 | 无 |
| 使用场景 | 动态条件判断 | 已知路由键 |

使用示例:
---------
>>> from langchain_core.runnables import RunnableBranch
>>>
>>> # 创建分支
>>> branch = RunnableBranch(
...     (lambda x: isinstance(x, str), lambda x: x.upper()),
...     (lambda x: isinstance(x, int), lambda x: x + 1),
...     (lambda x: isinstance(x, float), lambda x: x * 2),
...     lambda x: "默认输出",  # 默认分支
... )
>>>
>>> branch.invoke("hello")  # "HELLO"
>>> branch.invoke(5)        # 6
>>> branch.invoke(3.14)     # 6.28
>>> branch.invoke(None)     # "默认输出"

实际应用示例:
---------
>>> from langchain_core.runnables import RunnableBranch, RunnableLambda
>>> from langchain_openai import ChatOpenAI
>>>
>>> # 根据用户意图路由到不同的链
>>> def is_math_question(x):
...     return "计算" in x or "数学" in x
>>>
>>> def is_translation(x):
...     return "翻译" in x
>>>
>>> math_chain = ChatOpenAI() | (lambda x: f"数学答案: {x}")
>>> translation_chain = ChatOpenAI() | (lambda x: f"翻译结果: {x}")
>>> default_chain = ChatOpenAI()
>>>
>>> branch = RunnableBranch(
...     (is_math_question, math_chain),
...     (is_translation, translation_chain),
...     default_chain,
... )
"""

from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Iterator,
    Mapping,
    Sequence,
)
from typing import (
    Any,
    cast,
)

from pydantic import BaseModel, ConfigDict
from typing_extensions import override

from langchain_core.runnables.base import (
    Runnable,
    RunnableLike,
    RunnableSerializable,
    coerce_to_runnable,
)
from langchain_core.runnables.config import (
    RunnableConfig,
    ensure_config,
    get_async_callback_manager_for_config,
    get_callback_manager_for_config,
    patch_config,
)
from langchain_core.runnables.utils import (
    ConfigurableFieldSpec,
    Input,
    Output,
    get_unique_config_specs,
)

# 最少分支数量（包括默认分支）
_MIN_BRANCHES = 2


class RunnableBranch(RunnableSerializable[Input, Output]):
    """根据条件选择要执行的分支的 Runnable。

    初始化时接受 `(条件, Runnable)` 对的列表和一个默认分支。

    执行时:
    1. 按顺序求值每个条件
    2. 选择第一个求值为 True 的条件对应的 Runnable
    3. 用输入调用该 Runnable
    4. 如果所有条件都为 False，则运行默认分支

    属性:
    -----
    branches : Sequence[tuple[Runnable, Runnable]]
        `(条件, Runnable)` 对的列表
    default : Runnable
        所有条件都不满足时运行的默认 Runnable

    使用示例:
        ```python
        from langchain_core.runnables import RunnableBranch

        branch = RunnableBranch(
            (lambda x: isinstance(x, str), lambda x: x.upper()),
            (lambda x: isinstance(x, int), lambda x: x + 1),
            (lambda x: isinstance(x, float), lambda x: x * 2),
            lambda x: "goodbye",  # 默认分支
        )

        branch.invoke("hello")  # "HELLO"
        branch.invoke(None)     # "goodbye"
        ```
    """

    branches: Sequence[tuple[Runnable[Input, bool], Runnable[Input, Output]]]
    """`(条件, Runnable)` 对的列表。

    条件是一个接受输入并返回布尔值的 Runnable。
    """

    default: Runnable[Input, Output]
    """如果所有条件都不满足时执行的默认 Runnable。"""

    def __init__(
        self,
        *branches: tuple[
            Runnable[Input, bool]
            | Callable[[Input], bool]
            | Callable[[Input], Awaitable[bool]],
            RunnableLike,
        ]
        | RunnableLike,
    ) -> None:
        """创建一个根据条件选择分支的 Runnable。

        参数格式: (条件1, 分支1), (条件2, 分支2), ..., 默认分支

        Args:
            *branches: `(条件, Runnable)` 对的列表，最后一个是默认分支。
                条件可以是返回布尔值的 Runnable、Callable 或异步 Callable。
                分支可以是 RunnableLike（Runnable、Callable 或 Mapping）。

        Raises:
            ValueError: 如果分支数量少于 2。
            TypeError: 如果默认分支不是 Runnable、Callable 或 Mapping。
            TypeError: 如果分支不是元组或列表。
            ValueError: 如果分支的长度不是 2。

        使用示例:
            ```python
            branch = RunnableBranch(
                (条件1, 分支1),  # 如果条件1为True，执行分支1
                (条件2, 分支2),  # 否则如果条件2为True，执行分支2
                默认分支,        # 否则执行默认分支
            )
            ```
        """
        if len(branches) < _MIN_BRANCHES:
            msg = "RunnableBranch 需要至少两个分支"
            raise ValueError(msg)

        # 最后一个参数是默认分支
        default = branches[-1]

        if not isinstance(
            default,
            (Runnable, Callable, Mapping),  # type: ignore[arg-type]
        ):
            msg = "RunnableBranch 的默认分支必须是 Runnable、callable 或 mapping。"
            raise TypeError(msg)

        default_ = cast(
            "Runnable[Input, Output]", coerce_to_runnable(cast("RunnableLike", default))
        )

        branches_ = []

        for branch in branches[:-1]:
            if not isinstance(branch, (tuple, list)):
                msg = (
                    f"RunnableBranch 的分支必须是元组或列表，"
                    f"而不是 {type(branch)}"
                )
                raise TypeError(msg)

            if len(branch) != _MIN_BRANCHES:
                msg = (
                    f"RunnableBranch 的分支必须是长度为 2 的元组或列表，"
                    f"而不是 {len(branch)}"
                )
                raise ValueError(msg)
            condition, runnable = branch
            condition = cast("Runnable[Input, bool]", coerce_to_runnable(condition))
            runnable = coerce_to_runnable(runnable)
            branches_.append((condition, runnable))

        super().__init__(
            branches=branches_,
            default=default_,
        )

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @classmethod
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
    def get_input_schema(self, config: RunnableConfig | None = None) -> type[BaseModel]:
        """获取输入模式。

        尝试从所有分支和默认分支中获取有效的输入模式。
        """
        runnables = (
            [self.default]
            + [r for _, r in self.branches]
            + [r for r, _ in self.branches]
        )

        for runnable in runnables:
            if (
                runnable.get_input_schema(config).model_json_schema().get("type")
                is not None
            ):
                return runnable.get_input_schema(config)

        return super().get_input_schema(config)

    @property
    @override
    def config_specs(self) -> list[ConfigurableFieldSpec]:
        """获取所有分支的配置规格。"""
        return get_unique_config_specs(
            spec
            for step in (
                [self.default]
                + [r for _, r in self.branches]
                + [r for r, _ in self.branches]
            )
            for spec in step.config_specs
        )

    @override
    def invoke(
        self, input: Input, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Output:
        """首先求值条件，然后执行 True 或默认分支。

        工作流程:
        1. 依次求值每个条件
        2. 如果条件为 True，执行对应的分支并返回
        3. 如果所有条件都为 False，执行默认分支

        Args:
            input: 传递给 Runnable 的输入。
            config: Runnable 的配置。
            **kwargs: 传递给 Runnable 的额外关键字参数。

        Returns:
            执行的分支的输出。
        """
        config = ensure_config(config)
        callback_manager = get_callback_manager_for_config(config)
        run_manager = callback_manager.on_chain_start(
            None,
            input,
            name=config.get("run_name") or self.get_name(),
            run_id=config.pop("run_id", None),
        )

        try:
            for idx, branch in enumerate(self.branches):
                condition, runnable = branch

                # 求值条件
                expression_value = condition.invoke(
                    input,
                    config=patch_config(
                        config,
                        callbacks=run_manager.get_child(tag=f"condition:{idx + 1}"),
                    ),
                )

                # 如果条件为 True，执行对应分支
                if expression_value:
                    output = runnable.invoke(
                        input,
                        config=patch_config(
                            config,
                            callbacks=run_manager.get_child(tag=f"branch:{idx + 1}"),
                        ),
                        **kwargs,
                    )
                    break
            else:
                # 所有条件都为 False，执行默认分支
                output = self.default.invoke(
                    input,
                    config=patch_config(
                        config, callbacks=run_manager.get_child(tag="branch:default")
                    ),
                    **kwargs,
                )
        except BaseException as e:
            run_manager.on_chain_error(e)
            raise
        run_manager.on_chain_end(output)
        return output

    @override
    async def ainvoke(
        self, input: Input, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Output:
        """异步版本的 invoke。"""
        config = ensure_config(config)
        callback_manager = get_async_callback_manager_for_config(config)
        run_manager = await callback_manager.on_chain_start(
            None,
            input,
            name=config.get("run_name") or self.get_name(),
            run_id=config.pop("run_id", None),
        )
        try:
            for idx, branch in enumerate(self.branches):
                condition, runnable = branch

                expression_value = await condition.ainvoke(
                    input,
                    config=patch_config(
                        config,
                        callbacks=run_manager.get_child(tag=f"condition:{idx + 1}"),
                    ),
                )

                if expression_value:
                    output = await runnable.ainvoke(
                        input,
                        config=patch_config(
                            config,
                            callbacks=run_manager.get_child(tag=f"branch:{idx + 1}"),
                        ),
                        **kwargs,
                    )
                    break
            else:
                output = await self.default.ainvoke(
                    input,
                    config=patch_config(
                        config, callbacks=run_manager.get_child(tag="branch:default")
                    ),
                    **kwargs,
                )
        except BaseException as e:
            await run_manager.on_chain_error(e)
            raise
        await run_manager.on_chain_end(output)
        return output

    @override
    def stream(
        self,
        input: Input,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> Iterator[Output]:
        """首先求值条件，然后流式执行选中的分支。

        Args:
            input: 传递给 Runnable 的输入。
            config: Runnable 的配置。
            **kwargs: 额外参数。

        Yields:
            执行的分支的流式输出块。
        """
        config = ensure_config(config)
        callback_manager = get_callback_manager_for_config(config)
        run_manager = callback_manager.on_chain_start(
            None,
            input,
            name=config.get("run_name") or self.get_name(),
            run_id=config.pop("run_id", None),
        )
        final_output: Output | None = None
        final_output_supported = True

        try:
            for idx, branch in enumerate(self.branches):
                condition, runnable = branch

                expression_value = condition.invoke(
                    input,
                    config=patch_config(
                        config,
                        callbacks=run_manager.get_child(tag=f"condition:{idx + 1}"),
                    ),
                )

                if expression_value:
                    for chunk in runnable.stream(
                        input,
                        config=patch_config(
                            config,
                            callbacks=run_manager.get_child(tag=f"branch:{idx + 1}"),
                        ),
                        **kwargs,
                    ):
                        yield chunk
                        # 尝试累加块以获得最终输出
                        if final_output_supported:
                            if final_output is None:
                                final_output = chunk
                            else:
                                try:
                                    final_output = final_output + chunk  # type: ignore[operator]
                                except TypeError:
                                    final_output = None
                                    final_output_supported = False
                    break
            else:
                for chunk in self.default.stream(
                    input,
                    config=patch_config(
                        config,
                        callbacks=run_manager.get_child(tag="branch:default"),
                    ),
                    **kwargs,
                ):
                    yield chunk
                    if final_output_supported:
                        if final_output is None:
                            final_output = chunk
                        else:
                            try:
                                final_output = final_output + chunk  # type: ignore[operator]
                            except TypeError:
                                final_output = None
                                final_output_supported = False
        except BaseException as e:
            run_manager.on_chain_error(e)
            raise
        run_manager.on_chain_end(final_output)

    @override
    async def astream(
        self,
        input: Input,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> AsyncIterator[Output]:
        """异步流式版本。"""
        config = ensure_config(config)
        callback_manager = get_async_callback_manager_for_config(config)
        run_manager = await callback_manager.on_chain_start(
            None,
            input,
            name=config.get("run_name") or self.get_name(),
            run_id=config.pop("run_id", None),
        )
        final_output: Output | None = None
        final_output_supported = True

        try:
            for idx, branch in enumerate(self.branches):
                condition, runnable = branch

                expression_value = await condition.ainvoke(
                    input,
                    config=patch_config(
                        config,
                        callbacks=run_manager.get_child(tag=f"condition:{idx + 1}"),
                    ),
                )

                if expression_value:
                    async for chunk in runnable.astream(
                        input,
                        config=patch_config(
                            config,
                            callbacks=run_manager.get_child(tag=f"branch:{idx + 1}"),
                        ),
                        **kwargs,
                    ):
                        yield chunk
                        if final_output_supported:
                            if final_output is None:
                                final_output = chunk
                            else:
                                try:
                                    final_output = final_output + chunk  # type: ignore[operator]
                                except TypeError:
                                    final_output = None
                                    final_output_supported = False
                    break
            else:
                async for chunk in self.default.astream(
                    input,
                    config=patch_config(
                        config,
                        callbacks=run_manager.get_child(tag="branch:default"),
                    ),
                    **kwargs,
                ):
                    yield chunk
                    if final_output_supported:
                        if final_output is None:
                            final_output = chunk
                        else:
                            try:
                                final_output = final_output + chunk  # type: ignore[operator]
                            except TypeError:
                                final_output = None
                                final_output_supported = False
        except BaseException as e:
            await run_manager.on_chain_error(e)
            raise
        await run_manager.on_chain_end(final_output)
