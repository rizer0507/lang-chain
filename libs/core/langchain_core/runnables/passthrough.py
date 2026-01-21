"""RunnablePassthrough 实现模块。

本模块提供直通类 Runnable，用于传递输入或添加额外的键值对。

核心类:
--------
**RunnablePassthrough**: 直通输入，可选地添加额外键
**RunnableAssign**: 向字典输入添加新键值对
**RunnablePick**: 从字典中选择指定的键

常用场景:
---------
1. 在并行处理中保留原始输入
2. 使用 `.assign()` 方法向字典添加计算结果
3. 使用 `RunnablePick` 选择需要的键

使用示例:
---------
>>> from langchain_core.runnables import RunnablePassthrough, RunnableParallel
>>>
>>> # 保留原始值并添加处理结果
>>> chain = RunnableParallel(
...     original=RunnablePassthrough(),
...     doubled=lambda x: x * 2,
... )
>>> chain.invoke(5)  # {'original': 5, 'doubled': 10}
>>>
>>> # 使用 assign 向输出添加新键
>>> chain = {"a": lambda x: x + 1} | RunnablePassthrough.assign(
...     b=lambda d: d["a"] * 2
... )
>>> chain.invoke(1)  # {'a': 2, 'b': 4}
"""

from __future__ import annotations

import asyncio
import inspect
import threading
from collections.abc import Awaitable, Callable
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)

from pydantic import BaseModel, RootModel
from typing_extensions import override

from langchain_core.runnables.base import (
    Other,
    Runnable,
    RunnableParallel,
    RunnableSerializable,
)
from langchain_core.runnables.config import (
    RunnableConfig,
    acall_func_with_variable_args,
    call_func_with_variable_args,
    ensure_config,
    get_executor_for_config,
    patch_config,
)
from langchain_core.runnables.utils import (
    AddableDict,
    ConfigurableFieldSpec,
)
from langchain_core.utils.aiter import atee
from langchain_core.utils.iter import safetee
from langchain_core.utils.pydantic import create_model_v2

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator, Mapping

    from langchain_core.callbacks.manager import (
        AsyncCallbackManagerForChainRun,
        CallbackManagerForChainRun,
    )
    from langchain_core.runnables.graph import Graph


def identity(x: Other) -> Other:
    """恒等函数。

    Args:
        x: 输入。

    Returns:
        原样返回输入。
    """
    return x


async def aidentity(x: Other) -> Other:
    """异步恒等函数。

    Args:
        x: 输入。

    Returns:
        原样返回输入。
    """
    return x


class RunnablePassthrough(RunnableSerializable[Other, Other]):
    """直通输入的 Runnable，可选地添加额外键。

    这个 Runnable 的行为几乎像恒等函数，但如果输入是字典，
    可以配置为向输出添加额外的键。

    主要用途:
    ---------
    1. 在 RunnableParallel 中保留原始输入
    2. 通过 `.assign()` 方法向字典添加新的计算结果
    3. 在执行过程中记录或监控数据（通过 func 参数）

    属性:
    -----
    func : Callable | None
        可选的同步函数，在传递输入时被调用（用于副作用）
    afunc : Callable | None
        可选的异步函数，在传递输入时被调用
    input_type : type | None
        输入类型

    使用示例:
        ```python
        from langchain_core.runnables import (
            RunnableLambda,
            RunnableParallel,
            RunnablePassthrough,
        )

        runnable = RunnableParallel(
            origin=RunnablePassthrough(), modified=lambda x: x + 1
        )

        runnable.invoke(1)  # {'origin': 1, 'modified': 2}

        def fake_llm(prompt: str) -> str:  # 示例用的假 LLM
            return "completion"

        chain = RunnableLambda(fake_llm) | {
            "original": RunnablePassthrough(),  # 原始 LLM 输出
            "parsed": lambda text: text[::-1],  # 解析逻辑
        }

        chain.invoke("hello")  # {'original': 'completion', 'parsed': 'noitelpmoc'}
        ```

    使用 assign 添加键:
        ```python
        from langchain_core.runnables import RunnablePassthrough

        def fake_llm(prompt: str) -> str:
            return "completion"

        runnable = {
            "llm1": fake_llm,
            "llm2": fake_llm,
        } | RunnablePassthrough.assign(
            total_chars=lambda inputs: len(inputs["llm1"] + inputs["llm2"])
        )

        runnable.invoke("hello")
        # {'llm1': 'completion', 'llm2': 'completion', 'total_chars': 20}
        ```
    """

    input_type: type[Other] | None = None

    func: Callable[[Other], None] | Callable[[Other, RunnableConfig], None] | None = (
        None
    )

    afunc: (
        Callable[[Other], Awaitable[None]]
        | Callable[[Other, RunnableConfig], Awaitable[None]]
        | None
    ) = None

    @override
    def __repr_args__(self) -> Any:
        # Without this repr(self) raises a RecursionError
        # See https://github.com/pydantic/pydantic/issues/7327
        return []

    def __init__(
        self,
        func: Callable[[Other], None]
        | Callable[[Other, RunnableConfig], None]
        | Callable[[Other], Awaitable[None]]
        | Callable[[Other, RunnableConfig], Awaitable[None]]
        | None = None,
        afunc: Callable[[Other], Awaitable[None]]
        | Callable[[Other, RunnableConfig], Awaitable[None]]
        | None = None,
        *,
        input_type: type[Other] | None = None,
        **kwargs: Any,
    ) -> None:
        """Create a `RunnablePassthrough`.

        Args:
            func: Function to be called with the input.
            afunc: Async function to be called with the input.
            input_type: Type of the input.
        """
        if inspect.iscoroutinefunction(func):
            afunc = func
            func = None

        super().__init__(func=func, afunc=afunc, input_type=input_type, **kwargs)

    @classmethod
    @override
    def is_lc_serializable(cls) -> bool:
        """Return `True` as this class is serializable."""
        return True

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the LangChain object.

        Returns:
            `["langchain", "schema", "runnable"]`
        """
        return ["langchain", "schema", "runnable"]

    @property
    @override
    def InputType(self) -> Any:
        return self.input_type or Any

    @property
    @override
    def OutputType(self) -> Any:
        return self.input_type or Any

    @classmethod
    @override
    def assign(
        cls,
        **kwargs: Runnable[dict[str, Any], Any]
        | Callable[[dict[str, Any]], Any]
        | Mapping[str, Runnable[dict[str, Any], Any] | Callable[[dict[str, Any]], Any]],
    ) -> RunnableAssign:
        """Merge the Dict input with the output produced by the mapping argument.

        Args:
            **kwargs: `Runnable`, `Callable` or a `Mapping` from keys to `Runnable`
                objects or `Callable`s.

        Returns:
            A `Runnable` that merges the `dict` input with the output produced by the
            mapping argument.
        """
        return RunnableAssign(RunnableParallel[dict[str, Any]](kwargs))

    @override
    def invoke(
        self, input: Other, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Other:
        if self.func is not None:
            call_func_with_variable_args(
                self.func, input, ensure_config(config), **kwargs
            )
        return self._call_with_config(identity, input, config)

    @override
    async def ainvoke(
        self,
        input: Other,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> Other:
        if self.afunc is not None:
            await acall_func_with_variable_args(
                self.afunc, input, ensure_config(config), **kwargs
            )
        elif self.func is not None:
            call_func_with_variable_args(
                self.func, input, ensure_config(config), **kwargs
            )
        return await self._acall_with_config(aidentity, input, config)

    @override
    def transform(
        self,
        input: Iterator[Other],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[Other]:
        if self.func is None:
            for chunk in self._transform_stream_with_config(input, identity, config):
                yield chunk
        else:
            final: Other
            got_first_chunk = False

            for chunk in self._transform_stream_with_config(input, identity, config):
                yield chunk

                if not got_first_chunk:
                    final = chunk
                    got_first_chunk = True
                else:
                    try:
                        final = final + chunk  # type: ignore[operator]
                    except TypeError:
                        final = chunk

            if got_first_chunk:
                call_func_with_variable_args(
                    self.func, final, ensure_config(config), **kwargs
                )

    @override
    async def atransform(
        self,
        input: AsyncIterator[Other],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Other]:
        if self.afunc is None and self.func is None:
            async for chunk in self._atransform_stream_with_config(
                input, identity, config
            ):
                yield chunk
        else:
            got_first_chunk = False

            async for chunk in self._atransform_stream_with_config(
                input, identity, config
            ):
                yield chunk

                # By definitions, a function will operate on the aggregated
                # input. So we'll aggregate the input until we get to the last
                # chunk.
                # If the input is not addable, then we'll assume that we can
                # only operate on the last chunk.
                if not got_first_chunk:
                    final = chunk
                    got_first_chunk = True
                else:
                    try:
                        final = final + chunk  # type: ignore[operator]
                    except TypeError:
                        final = chunk

            if got_first_chunk:
                config = ensure_config(config)
                if self.afunc is not None:
                    await acall_func_with_variable_args(
                        self.afunc, final, config, **kwargs
                    )
                elif self.func is not None:
                    call_func_with_variable_args(self.func, final, config, **kwargs)

    @override
    def stream(
        self,
        input: Other,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[Other]:
        return self.transform(iter([input]), config, **kwargs)

    @override
    async def astream(
        self,
        input: Other,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Other]:
        async def input_aiter() -> AsyncIterator[Other]:
            yield input

        async for chunk in self.atransform(input_aiter(), config, **kwargs):
            yield chunk


_graph_passthrough: RunnablePassthrough = RunnablePassthrough()


class RunnableAssign(RunnableSerializable[dict[str, Any], dict[str, Any]]):
    """向 `dict[str, Any]` 输入添加键值对的 Runnable。

    `RunnableAssign` 类接收输入字典，通过 `RunnableParallel` 实例应用转换，
    然后将结果与原始数据合并，根据 mapper 的逻辑引入新的键值对。

    这通常通过 `RunnablePassthrough.assign()` 方法创建，
    而不是直接实例化。

    工作原理:
    ---------
    1. 接收字典输入
    2. 将输入传递给 mapper（RunnableParallel）
    3. mapper 计算新的键值对
    4. 将新键值对与原始输入合并

    使用示例:
        ```python
        from langchain_core.runnables.passthrough import (
            RunnableAssign,
            RunnableParallel,
        )
        from langchain_core.runnables.base import RunnableLambda

        def add_ten(x: dict[str, int]) -> dict[str, int]:
            return {"added": x["input"] + 10}

        mapper = RunnableParallel(
            {
                "add_step": RunnableLambda(add_ten),
            }
        )

        runnable_assign = RunnableAssign(mapper)

        # 同步示例
        runnable_assign.invoke({"input": 5})
        # 返回 {'input': 5, 'add_step': {'added': 15}}

        # 异步示例
        await runnable_assign.ainvoke({"input": 5})
        # 返回 {'input': 5, 'add_step': {'added': 15}}
        ```
    """

    mapper: RunnableParallel
    """用于计算新键值对的 RunnableParallel。"""

    def __init__(self, mapper: RunnableParallel[dict[str, Any]], **kwargs: Any) -> None:
        """Create a `RunnableAssign`.

        Args:
            mapper: A `RunnableParallel` instance that will be used to transform the
                input dictionary.
        """
        super().__init__(mapper=mapper, **kwargs)

    @classmethod
    @override
    def is_lc_serializable(cls) -> bool:
        """Return `True` as this class is serializable."""
        return True

    @classmethod
    @override
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the LangChain object.

        Returns:
            `["langchain", "schema", "runnable"]`
        """
        return ["langchain", "schema", "runnable"]

    @override
    def get_name(self, suffix: str | None = None, *, name: str | None = None) -> str:
        name = (
            name
            or self.name
            or f"RunnableAssign<{','.join(self.mapper.steps__.keys())}>"
        )
        return super().get_name(suffix, name=name)

    @override
    def get_input_schema(self, config: RunnableConfig | None = None) -> type[BaseModel]:
        map_input_schema = self.mapper.get_input_schema(config)
        if not issubclass(map_input_schema, RootModel):
            # ie. it's a dict
            return map_input_schema

        return super().get_input_schema(config)

    @override
    def get_output_schema(
        self, config: RunnableConfig | None = None
    ) -> type[BaseModel]:
        map_input_schema = self.mapper.get_input_schema(config)
        map_output_schema = self.mapper.get_output_schema(config)
        if not issubclass(map_input_schema, RootModel) and not issubclass(
            map_output_schema, RootModel
        ):
            fields = {}

            for name, field_info in map_input_schema.model_fields.items():
                fields[name] = (field_info.annotation, field_info.default)

            for name, field_info in map_output_schema.model_fields.items():
                fields[name] = (field_info.annotation, field_info.default)

            return create_model_v2("RunnableAssignOutput", field_definitions=fields)
        if not issubclass(map_output_schema, RootModel):
            # ie. only map output is a dict
            # ie. input type is either unknown or inferred incorrectly
            return map_output_schema

        return super().get_output_schema(config)

    @property
    @override
    def config_specs(self) -> list[ConfigurableFieldSpec]:
        return self.mapper.config_specs

    @override
    def get_graph(self, config: RunnableConfig | None = None) -> Graph:
        # get graph from mapper
        graph = self.mapper.get_graph(config)
        # add passthrough node and edges
        input_node = graph.first_node()
        output_node = graph.last_node()
        if input_node is not None and output_node is not None:
            passthrough_node = graph.add_node(_graph_passthrough)
            graph.add_edge(input_node, passthrough_node)
            graph.add_edge(passthrough_node, output_node)
        return graph

    def _invoke(
        self,
        value: dict[str, Any],
        run_manager: CallbackManagerForChainRun,
        config: RunnableConfig,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if not isinstance(value, dict):
            msg = "The input to RunnablePassthrough.assign() must be a dict."
            raise ValueError(msg)  # noqa: TRY004

        return {
            **value,
            **self.mapper.invoke(
                value,
                patch_config(config, callbacks=run_manager.get_child()),
                **kwargs,
            ),
        }

    @override
    def invoke(
        self,
        input: dict[str, Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return self._call_with_config(self._invoke, input, config, **kwargs)

    async def _ainvoke(
        self,
        value: dict[str, Any],
        run_manager: AsyncCallbackManagerForChainRun,
        config: RunnableConfig,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if not isinstance(value, dict):
            msg = "The input to RunnablePassthrough.assign() must be a dict."
            raise ValueError(msg)  # noqa: TRY004

        return {
            **value,
            **await self.mapper.ainvoke(
                value,
                patch_config(config, callbacks=run_manager.get_child()),
                **kwargs,
            ),
        }

    @override
    async def ainvoke(
        self,
        input: dict[str, Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return await self._acall_with_config(self._ainvoke, input, config, **kwargs)

    def _transform(
        self,
        values: Iterator[dict[str, Any]],
        run_manager: CallbackManagerForChainRun,
        config: RunnableConfig,
        **kwargs: Any,
    ) -> Iterator[dict[str, Any]]:
        # collect mapper keys
        mapper_keys = set(self.mapper.steps__.keys())
        # create two streams, one for the map and one for the passthrough
        for_passthrough, for_map = safetee(values, 2, lock=threading.Lock())

        # create map output stream
        map_output = self.mapper.transform(
            for_map,
            patch_config(
                config,
                callbacks=run_manager.get_child(),
            ),
            **kwargs,
        )

        # get executor to start map output stream in background
        with get_executor_for_config(config) as executor:
            # start map output stream
            first_map_chunk_future = executor.submit(
                next,
                map_output,
                None,
            )
            # consume passthrough stream
            for chunk in for_passthrough:
                if not isinstance(chunk, dict):
                    msg = "The input to RunnablePassthrough.assign() must be a dict."
                    raise ValueError(msg)  # noqa: TRY004
                # remove mapper keys from passthrough chunk, to be overwritten by map
                filtered = AddableDict(
                    {k: v for k, v in chunk.items() if k not in mapper_keys}
                )
                if filtered:
                    yield filtered
            # yield map output
            yield cast("dict[str, Any]", first_map_chunk_future.result())
            for chunk in map_output:
                yield chunk

    @override
    def transform(
        self,
        input: Iterator[dict[str, Any]],
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> Iterator[dict[str, Any]]:
        yield from self._transform_stream_with_config(
            input, self._transform, config, **kwargs
        )

    async def _atransform(
        self,
        values: AsyncIterator[dict[str, Any]],
        run_manager: AsyncCallbackManagerForChainRun,
        config: RunnableConfig,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        # collect mapper keys
        mapper_keys = set(self.mapper.steps__.keys())
        # create two streams, one for the map and one for the passthrough
        for_passthrough, for_map = atee(values, 2, lock=asyncio.Lock())
        # create map output stream
        map_output = self.mapper.atransform(
            for_map,
            patch_config(
                config,
                callbacks=run_manager.get_child(),
            ),
            **kwargs,
        )
        # start map output stream
        first_map_chunk_task: asyncio.Task = asyncio.create_task(
            anext(map_output, None),
        )
        # consume passthrough stream
        async for chunk in for_passthrough:
            if not isinstance(chunk, dict):
                msg = "The input to RunnablePassthrough.assign() must be a dict."
                raise ValueError(msg)  # noqa: TRY004

            # remove mapper keys from passthrough chunk, to be overwritten by map output
            filtered = AddableDict(
                {k: v for k, v in chunk.items() if k not in mapper_keys}
            )
            if filtered:
                yield filtered
        # yield map output
        yield await first_map_chunk_task
        async for chunk in map_output:
            yield chunk

    @override
    async def atransform(
        self,
        input: AsyncIterator[dict[str, Any]],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        async for chunk in self._atransform_stream_with_config(
            input, self._atransform, config, **kwargs
        ):
            yield chunk

    @override
    def stream(
        self,
        input: dict[str, Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[dict[str, Any]]:
        return self.transform(iter([input]), config, **kwargs)

    @override
    async def astream(
        self,
        input: dict[str, Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        async def input_aiter() -> AsyncIterator[dict[str, Any]]:
            yield input

        async for chunk in self.atransform(input_aiter(), config, **kwargs):
            yield chunk


class RunnablePick(RunnableSerializable[dict[str, Any], Any]):
    """从 `dict[str, Any]` 输入中选择指定键的 Runnable。

    `RunnablePick` 类代表一个从字典输入中选择性地提取键的 Runnable。
    它允许你指定一个或多个要从输入字典中提取的键。

    返回类型行为:
    -------------
    返回类型取决于 `keys` 参数：

    - 当 `keys` 是 `str` 时：返回该键关联的单个值
    - 当 `keys` 是 `list` 时：返回仅包含所选键的字典

    使用示例:
        ```python
        from langchain_core.runnables.passthrough import RunnablePick

        input_data = {
            "name": "张三",
            "age": 30,
            "city": "北京",
            "country": "中国",
        }

        # 单个键 - 直接返回值
        runnable_single = RunnablePick(keys="name")
        result_single = runnable_single.invoke(input_data)
        print(result_single)  # 输出: "张三"

        # 多个键 - 返回字典
        runnable_multiple = RunnablePick(keys=["name", "age"])
        result_multiple = runnable_multiple.invoke(input_data)
        print(result_multiple)  # 输出: {'name': '张三', 'age': 30}
        ```
    """

    keys: str | list[str]
    """要从输入字典中选择的键，可以是单个字符串或字符串列表。"""

    def __init__(self, keys: str | list[str], **kwargs: Any) -> None:
        """Create a `RunnablePick`.

        Args:
            keys: A single key or a list of keys to pick from the input dictionary.
        """
        super().__init__(keys=keys, **kwargs)

    @classmethod
    @override
    def is_lc_serializable(cls) -> bool:
        """Return `True` as this class is serializable."""
        return True

    @classmethod
    @override
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the LangChain object.

        Returns:
            `["langchain", "schema", "runnable"]`
        """
        return ["langchain", "schema", "runnable"]

    @override
    def get_name(self, suffix: str | None = None, *, name: str | None = None) -> str:
        name = (
            name
            or self.name
            or "RunnablePick"
            f"<{','.join([self.keys] if isinstance(self.keys, str) else self.keys)}>"
        )
        return super().get_name(suffix, name=name)

    def _pick(self, value: dict[str, Any]) -> Any:
        if not isinstance(value, dict):
            msg = "The input to RunnablePassthrough.assign() must be a dict."
            raise ValueError(msg)  # noqa: TRY004

        if isinstance(self.keys, str):
            return value.get(self.keys)
        picked = {k: value.get(k) for k in self.keys if k in value}
        if picked:
            return AddableDict(picked)
        return None

    @override
    def invoke(
        self,
        input: dict[str, Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        return self._call_with_config(self._pick, input, config, **kwargs)

    async def _ainvoke(
        self,
        value: dict[str, Any],
    ) -> Any:
        return self._pick(value)

    @override
    async def ainvoke(
        self,
        input: dict[str, Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        return await self._acall_with_config(self._ainvoke, input, config, **kwargs)

    def _transform(
        self,
        chunks: Iterator[dict[str, Any]],
    ) -> Iterator[Any]:
        for chunk in chunks:
            picked = self._pick(chunk)
            if picked is not None:
                yield picked

    @override
    def transform(
        self,
        input: Iterator[dict[str, Any]],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[Any]:
        yield from self._transform_stream_with_config(
            input, self._transform, config, **kwargs
        )

    async def _atransform(
        self,
        chunks: AsyncIterator[dict[str, Any]],
    ) -> AsyncIterator[Any]:
        async for chunk in chunks:
            picked = self._pick(chunk)
            if picked is not None:
                yield picked

    @override
    async def atransform(
        self,
        input: AsyncIterator[dict[str, Any]],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        async for chunk in self._atransform_stream_with_config(
            input, self._atransform, config, **kwargs
        ):
            yield chunk

    @override
    def stream(
        self,
        input: dict[str, Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[Any]:
        return self.transform(iter([input]), config, **kwargs)

    @override
    async def astream(
        self,
        input: dict[str, Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        async def input_aiter() -> AsyncIterator[dict[str, Any]]:
            yield input

        async for chunk in self.atransform(input_aiter(), config, **kwargs):
            yield chunk
