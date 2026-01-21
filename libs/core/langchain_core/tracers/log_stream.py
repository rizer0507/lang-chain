"""Tracer that streams run logs to a stream.

中文翻译:
将运行日志流式传输到流的跟踪器。"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import threading
from collections import defaultdict
from pprint import pformat
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    TypeVar,
    overload,
)

import jsonpatch  # type: ignore[import-untyped]
from typing_extensions import NotRequired, TypedDict, override

from langchain_core.callbacks.base import BaseCallbackManager
from langchain_core.load import dumps
from langchain_core.load.load import load
from langchain_core.outputs import ChatGenerationChunk, GenerationChunk
from langchain_core.runnables import RunnableConfig, ensure_config
from langchain_core.tracers._streaming import _StreamingCallbackHandler
from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.memory_stream import _MemoryStream

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator, Sequence
    from uuid import UUID

    from langchain_core.runnables import Runnable
    from langchain_core.runnables.utils import Input, Output
    from langchain_core.tracers.schemas import Run


class LogEntry(TypedDict):
    """A single entry in the run log.

    中文翻译:
    运行日志中的单个条目。"""

    id: str
    """ID of the sub-run.

    中文翻译:
    子运行的 ID。"""
    name: str
    """Name of the object being run.

    中文翻译:
    正在运行的对象的名称。"""
    type: str
    """Type of the object being run, eg. prompt, chain, llm, etc.

    中文翻译:
    正在运行的对象的类型，例如。提示、连锁、LLM 等"""
    tags: list[str]
    """List of tags for the run.

    中文翻译:
    运行的标签列表。"""
    metadata: dict[str, Any]
    """Key-value pairs of metadata for the run.

    中文翻译:
    运行的元数据的键值对。"""
    start_time: str
    """ISO-8601 timestamp of when the run started.

    中文翻译:
    运行开始时的 ISO-8601 时间戳。"""

    streamed_output_str: list[str]
    """List of LLM tokens streamed by this run, if applicable.

    中文翻译:
    本次运行流式传输的 LLM 令牌列表（如果适用）。"""
    streamed_output: list[Any]
    """List of output chunks streamed by this run, if available.

    中文翻译:
    此运行流式传输的输出块列表（如果有）。"""
    inputs: NotRequired[Any | None]
    """Inputs to this run. Not available currently via astream_log.

    中文翻译:
    本次运行的输入。目前无法通过 astream_log 获得。"""
    final_output: Any | None
    """Final output of this run.

    Only available after the run has finished successfully.

    中文翻译:
    本次运行的最终输出。
    仅在运行成功完成后可用。"""
    end_time: str | None
    """ISO-8601 timestamp of when the run ended.
    Only available after the run has finished.

    中文翻译:
    运行结束时的 ISO-8601 时间戳。
    仅在运行完成后可用。"""


class RunState(TypedDict):
    """State of the run.

    中文翻译:
    运行状态。"""

    id: str
    """ID of the run.

    中文翻译:
    运行的 ID。"""
    streamed_output: list[Any]
    """List of output chunks streamed by Runnable.stream()

    中文翻译:
    由 Runnable.stream() 流式传输的输出块列表"""
    final_output: Any | None
    """Final output of the run, usually the result of aggregating (`+`) streamed_output.
    Updated throughout the run when supported by the Runnable.

    中文翻译:
    运行的最终输出，通常是聚合（`+`）streamed_output 的结果。
    当 Runnable 支持时，在整个运行过程中进行更新。"""

    name: str
    """Name of the object being run.

    中文翻译:
    正在运行的对象的名称。"""
    type: str
    """Type of the object being run, eg. prompt, chain, llm, etc.

    中文翻译:
    正在运行的对象的类型，例如。提示、连锁、LLM 等"""

    # Do we want tags/metadata on the root run? Client kinda knows it in most situations
    # 中文: 我们想要在根运行上添加标签/元数据吗？在大多数情况下，客户都知道这一点
    # tags: list[str]
    # 中文: 标签： 列表[str]

    logs: dict[str, LogEntry]
    """Map of run names to sub-runs. If filters were supplied, this list will
    contain only the runs that matched the filters.

    中文翻译:
    运行名称到子运行的映射。如果提供了过滤器，此列表将
    仅包含与过滤器匹配的运行。"""


class RunLogPatch:
    """Patch to the run log.

    中文翻译:
    修补运行日志。"""

    ops: list[dict[str, Any]]
    """List of JSONPatch operations, which describe how to create the run state
    from an empty dict. This is the minimal representation of the log, designed to
    be serialized as JSON and sent over the wire to reconstruct the log on the other
    side. Reconstruction of the state can be done with any JSONPatch-compliant library,
    see https://jsonpatch.com for more information.

    中文翻译:
    JSONPatch 操作列表，描述如何创建运行状态
    来自一个空字典。这是日志的最小表示，旨在
    序列化为 JSON 并通过线路发送以重建另一方的日志
    边。可以使用任何兼容 JSONPatch 的库来重建状态，
    请参阅 https://jsonpatch.com 了解更多信息。"""

    def __init__(self, *ops: dict[str, Any]) -> None:
        """Create a RunLogPatch.

        Args:
            *ops: The operations to apply to the state.
        

        中文翻译:
        创建一个 RunLogPatch。
        参数：
            *ops：应用于状态的操作。"""
        self.ops = list(ops)

    def __add__(self, other: RunLogPatch | Any) -> RunLog:
        """Combine two `RunLogPatch` instances.

        Args:
            other: The other `RunLogPatch` to combine with.

        Raises:
            TypeError: If the other object is not a `RunLogPatch`.

        Returns:
            A new `RunLog` representing the combination of the two.
        

        中文翻译:
        组合两个“RunLogPatch”实例。
        参数：
            other：要结合的另一个“RunLogPatch”。
        加薪：
            类型错误：如果另一个对象不是“RunLogPatch”。
        返回：
            一个新的“RunLog”代表两者的组合。"""
        if type(other) is RunLogPatch:
            ops = self.ops + other.ops
            state = jsonpatch.apply_patch(None, copy.deepcopy(ops))
            return RunLog(*ops, state=state)

        msg = f"unsupported operand type(s) for +: '{type(self)}' and '{type(other)}'"
        raise TypeError(msg)

    @override
    def __repr__(self) -> str:
        # 1:-1 to get rid of the [] around the list
        # 中文: 1:-1 去掉列表周围的[]
        return f"RunLogPatch({pformat(self.ops)[1:-1]})"

    @override
    def __eq__(self, other: object) -> bool:
        return isinstance(other, RunLogPatch) and self.ops == other.ops

    __hash__ = None  # type: ignore[assignment]


class RunLog(RunLogPatch):
    """Run log.

    中文翻译:
    运行日志。"""

    state: RunState
    """Current state of the log, obtained from applying all ops in sequence.

    中文翻译:
    日志的当前状态，通过按顺序应用所有操作获得。"""

    def __init__(self, *ops: dict[str, Any], state: RunState) -> None:
        """Create a RunLog.

        Args:
            *ops: The operations to apply to the state.
            state: The initial state of the run log.
        

        中文翻译:
        创建运行日志。
        参数：
            *ops：应用于状态的操作。
            state：运行日志的初始状态。"""
        super().__init__(*ops)
        self.state = state

    def __add__(self, other: RunLogPatch | Any) -> RunLog:
        """Combine two `RunLog`s.

        Args:
            other: The other `RunLog` or `RunLogPatch` to combine with.

        Raises:
            TypeError: If the other object is not a `RunLog` or `RunLogPatch`.

        Returns:
            A new `RunLog` representing the combination of the two.
        

        中文翻译:
        合并两个“RunLog”。
        参数：
            other：要结合的其他“RunLog”或“RunLogPatch”。
        加薪：
            类型错误：如果另一个对象不是“RunLog”或“RunLogPatch”。
        返回：
            一个新的“RunLog”代表两者的组合。"""
        if type(other) is RunLogPatch:
            ops = self.ops + other.ops
            state = jsonpatch.apply_patch(self.state, other.ops)
            return RunLog(*ops, state=state)

        msg = f"unsupported operand type(s) for +: '{type(self)}' and '{type(other)}'"
        raise TypeError(msg)

    @override
    def __repr__(self) -> str:
        return f"RunLog({pformat(self.state)})"

    @override
    def __eq__(self, other: object) -> bool:
        """Check if two `RunLog`s are equal.

        Args:
            other: The other `RunLog` to compare to.

        Returns:
            `True` if the `RunLog`s are equal, `False` otherwise.
        

        中文翻译:
        检查两个 RunLog 是否相等。
        参数：
            other：要比较的另一个“RunLog”。
        返回：
            如果“RunLog”相等，则为“True”，否则为“False”。"""
        # First compare that the state is the same
        # 中文: 首先比较状态是否相同
        if not isinstance(other, RunLog):
            return False
        if self.state != other.state:
            return False
        # Then compare that the ops are the same
        # 中文: 然后比较ops是否相同
        return super().__eq__(other)

    __hash__ = None


T = TypeVar("T")


class LogStreamCallbackHandler(BaseTracer, _StreamingCallbackHandler):
    """Tracer that streams run logs to a stream.

    中文翻译:
    将运行日志流式传输到流的跟踪器。"""

    def __init__(
        self,
        *,
        auto_close: bool = True,
        include_names: Sequence[str] | None = None,
        include_types: Sequence[str] | None = None,
        include_tags: Sequence[str] | None = None,
        exclude_names: Sequence[str] | None = None,
        exclude_types: Sequence[str] | None = None,
        exclude_tags: Sequence[str] | None = None,
        # Schema format is for internal use only.
        # 中文: 架构格式仅供内部使用。
        _schema_format: Literal["original", "streaming_events"] = "streaming_events",
    ) -> None:
        """A tracer that streams run logs to a stream.

        Args:
            auto_close: Whether to close the stream when the root run finishes.
            include_names: Only include runs from Runnables with matching names.
            include_types: Only include runs from Runnables with matching types.
            include_tags: Only include runs from Runnables with matching tags.
            exclude_names: Exclude runs from Runnables with matching names.
            exclude_types: Exclude runs from Runnables with matching types.
            exclude_tags: Exclude runs from Runnables with matching tags.
            _schema_format: Primarily changes how the inputs and outputs are
                handled.

                **For internal use only. This API will change.**

                - 'original' is the format used by all current tracers.
                  This format is slightly inconsistent with respect to inputs
                  and outputs.
                - 'streaming_events' is used for supporting streaming events,
                  for internal usage. It will likely change in the future, or
                  be deprecated entirely in favor of a dedicated async tracer
                  for streaming events.

        Raises:
            ValueError: If an invalid schema format is provided (internal use only).
        

        中文翻译:
        将运行日志流式传输到流的跟踪器。
        参数：
            auto_close：根运行完成后是否关闭流。
            include_names：仅包含具有匹配名称的 Runnable 中的运行。
            include_types：仅包含具有匹配类型的 Runnable 中的运行。
            include_tags：仅包含具有匹配标签的 Runnable 中的运行。
            except_names：从 Runnables 中排除具有匹配名称的运行。
            except_types：从 Runnables 中排除具有匹配类型的运行。
            except_tags：从具有匹配标签的 Runnables 中排除运行。
            _schema_format：主要改变输入和输出的方式
                处理。
                **仅供内部使用。此 API 将发生变化。**
                -“原始”是所有当前跟踪器使用的格式。
                  此格式与输入略有不一致
                  和输出。
                - 'streaming_events' 用于支持流事件，
                  供内部使用。将来可能会改变，或者
                  完全弃用，转而使用专用的异步跟踪器
                  用于流式传输事件。
        加薪：
            ValueError：如果提供了无效的架构格式（仅限内部使用）。"""
        if _schema_format not in {"original", "streaming_events"}:
            msg = (
                f"Invalid schema format: {_schema_format}. "
                f"Expected one of 'original', 'streaming_events'."
            )
            raise ValueError(msg)
        super().__init__(_schema_format=_schema_format)

        self.auto_close = auto_close
        self.include_names = include_names
        self.include_types = include_types
        self.include_tags = include_tags
        self.exclude_names = exclude_names
        self.exclude_types = exclude_types
        self.exclude_tags = exclude_tags

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
        memory_stream = _MemoryStream[RunLogPatch](loop)
        self.lock = threading.Lock()
        self.send_stream = memory_stream.get_send_stream()
        self.receive_stream = memory_stream.get_receive_stream()
        self._key_map_by_run_id: dict[UUID, str] = {}
        self._counter_map_by_name: dict[str, int] = defaultdict(int)
        self.root_id: UUID | None = None

    def __aiter__(self) -> AsyncIterator[RunLogPatch]:
        """Iterate over the stream of run logs.

        Returns:
            An async iterator over the run log patches.
        

        中文翻译:
        迭代运行日志流。
        返回：
            运行日志补丁的异步迭代器。"""
        return self.receive_stream.__aiter__()

    def send(self, *ops: dict[str, Any]) -> bool:
        """Send a patch to the stream, return False if the stream is closed.

        Args:
            *ops: The operations to send to the stream.

        Returns:
            `True` if the patch was sent successfully, False if the stream is closed.
        

        中文翻译:
        向流发送补丁，如果流关闭则返回 False。
        参数：
            *ops：发送到流的操作。
        返回：
            如果补丁发送成功则为“True”，如果流关闭则为 False。"""
        # We will likely want to wrap this in try / except at some point
        # 中文: 我们可能希望在某个时候将其包装在 try / except 中
        # to handle exceptions that might arise at run time.
        # 中文: 处理运行时可能出现的异常。
        # For now we'll let the exception bubble up, and always return
        # 中文: 现在我们让异常冒泡，并始终返回
        # True on the happy path.
        # 中文: 真正走上幸福之路。
        self.send_stream.send_nowait(RunLogPatch(*ops))
        return True

    async def tap_output_aiter(
        self, run_id: UUID, output: AsyncIterator[T]
    ) -> AsyncIterator[T]:
        """Tap an output async iterator to stream its values to the log.

        Args:
            run_id: The ID of the run.
            output: The output async iterator.

        Yields:
            The output value.
        

        中文翻译:
        点击输出异步迭代器将其值流式传输到日志中。
        参数：
            run_id：运行的 ID。
            输出：输出异步迭代器。
        产量：
            输出值。"""
        async for chunk in output:
            # root run is handled in .astream_log()
            # 中文: root 运行在 .astream_log() 中处理
            # if we can't find the run silently ignore
            # 中文: 如果我们找不到运行则默默忽略
            # eg. because this run wasn't included in the log
            # 中文: 例如。因为这次运行没有包含在日志中
            if (
                run_id != self.root_id
                and (key := self._key_map_by_run_id.get(run_id))
                and (
                    not self.send(
                        {
                            "op": "add",
                            "path": f"/logs/{key}/streamed_output/-",
                            "value": chunk,
                        }
                    )
                )
            ):
                break

            yield chunk

    def tap_output_iter(self, run_id: UUID, output: Iterator[T]) -> Iterator[T]:
        """Tap an output async iterator to stream its values to the log.

        Args:
            run_id: The ID of the run.
            output: The output iterator.

        Yields:
            The output value.
        

        中文翻译:
        点击输出异步迭代器将其值流式传输到日志中。
        参数：
            run_id：运行的 ID。
            输出：输出迭代器。
        产量：
            输出值。"""
        for chunk in output:
            # root run is handled in .astream_log()
            # 中文: root 运行在 .astream_log() 中处理
            # if we can't find the run silently ignore
            # 中文: 如果我们找不到运行则默默忽略
            # eg. because this run wasn't included in the log
            # 中文: 例如。因为这次运行没有包含在日志中
            if (
                run_id != self.root_id
                and (key := self._key_map_by_run_id.get(run_id))
                and (
                    not self.send(
                        {
                            "op": "add",
                            "path": f"/logs/{key}/streamed_output/-",
                            "value": chunk,
                        }
                    )
                )
            ):
                break

            yield chunk

    def include_run(self, run: Run) -> bool:
        """Check if a Run should be included in the log.

        Args:
            run: The Run to check.

        Returns:
            `True` if the run should be included, `False` otherwise.
        

        中文翻译:
        检查日志中是否应包含运行。
        参数：
            run：运行来检查。
        返回：
            如果应包含运行，则为“True”，否则为“False”。"""
        if run.id == self.root_id:
            return False

        run_tags = run.tags or []

        if (
            self.include_names is None
            and self.include_types is None
            and self.include_tags is None
        ):
            include = True
        else:
            include = False

        if self.include_names is not None:
            include = include or run.name in self.include_names
        if self.include_types is not None:
            include = include or run.run_type in self.include_types
        if self.include_tags is not None:
            include = include or any(tag in self.include_tags for tag in run_tags)

        if self.exclude_names is not None:
            include = include and run.name not in self.exclude_names
        if self.exclude_types is not None:
            include = include and run.run_type not in self.exclude_types
        if self.exclude_tags is not None:
            include = include and all(tag not in self.exclude_tags for tag in run_tags)

        return include

    def _persist_run(self, run: Run) -> None:
        # This is a legacy method only called once for an entire run tree
        # 中文: 这是一个遗留方法，仅对整个运行树调用一次
        # therefore not useful here
        # 中文: 因此这里没用
        pass

    def _on_run_create(self, run: Run) -> None:
        """Start a run.

        中文翻译:
        开始跑步。"""
        if self.root_id is None:
            self.root_id = run.id
            if not self.send(
                {
                    "op": "replace",
                    "path": "",
                    "value": RunState(
                        id=str(run.id),
                        streamed_output=[],
                        final_output=None,
                        logs={},
                        name=run.name,
                        type=run.run_type,
                    ),
                }
            ):
                return

        if not self.include_run(run):
            return

        # Determine previous index, increment by 1
        # 中文: 确定前一个索引，加1
        with self.lock:
            self._counter_map_by_name[run.name] += 1
            count = self._counter_map_by_name[run.name]
            self._key_map_by_run_id[run.id] = (
                run.name if count == 1 else f"{run.name}:{count}"
            )

        entry = LogEntry(
            id=str(run.id),
            name=run.name,
            type=run.run_type,
            tags=run.tags or [],
            metadata=(run.extra or {}).get("metadata", {}),
            start_time=run.start_time.isoformat(timespec="milliseconds"),
            streamed_output=[],
            streamed_output_str=[],
            final_output=None,
            end_time=None,
        )

        if self._schema_format == "streaming_events":
            # If using streaming events let's add inputs as well
            # 中文: 如果使用流事件，我们也添加输入
            entry["inputs"] = _get_standardized_inputs(run, self._schema_format)

        # Add the run to the stream
        # 中文: 将运行添加到流中
        self.send(
            {
                "op": "add",
                "path": f"/logs/{self._key_map_by_run_id[run.id]}",
                "value": entry,
            }
        )

    def _on_run_update(self, run: Run) -> None:
        """Finish a run.

        中文翻译:
        完成一次跑步。"""
        try:
            index = self._key_map_by_run_id.get(run.id)

            if index is None:
                return

            ops = []

            if self._schema_format == "streaming_events":
                ops.append(
                    {
                        "op": "replace",
                        "path": f"/logs/{index}/inputs",
                        "value": _get_standardized_inputs(run, self._schema_format),
                    }
                )

            ops.extend(
                [
                    # Replace 'inputs' with final inputs
                    # 中文: 将“输入”替换为最终输入
                    # This is needed because in many cases the inputs are not
                    # 中文: 这是必要的，因为在许多情况下，输入不是
                    # known until after the run is finished and the entire
                    # 中文: 直到运行完成并且整个
                    # input stream has been processed by the runnable.
                    # 中文: 输入流已由可运行程序处理。
                    {
                        "op": "add",
                        "path": f"/logs/{index}/final_output",
                        # to undo the dumpd done by some runnables / tracer / etc
                        # 中文: 撤消由某些可运行程序/跟踪器/等完成的转储
                        "value": _get_standardized_outputs(run, self._schema_format),
                    },
                    {
                        "op": "add",
                        "path": f"/logs/{index}/end_time",
                        "value": run.end_time.isoformat(timespec="milliseconds")
                        if run.end_time is not None
                        else None,
                    },
                ]
            )

            self.send(*ops)
        finally:
            if run.id == self.root_id and self.auto_close:
                self.send_stream.close()

    def _on_llm_new_token(
        self,
        run: Run,
        token: str,
        chunk: GenerationChunk | ChatGenerationChunk | None,
    ) -> None:
        """Process new LLM token.

        中文翻译:
        处理新的 LLM 令牌。"""
        index = self._key_map_by_run_id.get(run.id)

        if index is None:
            return

        self.send(
            {
                "op": "add",
                "path": f"/logs/{index}/streamed_output_str/-",
                "value": token,
            },
            {
                "op": "add",
                "path": f"/logs/{index}/streamed_output/-",
                "value": chunk.message
                if isinstance(chunk, ChatGenerationChunk)
                else token,
            },
        )


def _get_standardized_inputs(
    run: Run, schema_format: Literal["original", "streaming_events"]
) -> Any:
    """Extract standardized inputs from a run.

    Standardizes the inputs based on the type of the runnable used.

    Args:
        run: Run object
        schema_format: The schema format to use.

    Returns:
        Valid inputs are only dict. By conventions, inputs always represented
        invocation using named arguments.
        None means that the input is not yet known!
    

    中文翻译:
    从运行中提取标准化输入。
    根据所使用的可运行程序的类型标准化输入。
    参数：
        运行：运行对象
        schema_format：要使用的架构格式。
    返回：
        有效输入只是字典。按照惯例，输入总是表示
        使用命名参数调用。
        None 表示输入尚不可知！"""
    if schema_format == "original":
        msg = (
            "Do not assign inputs with original schema drop the key for now."
            "When inputs are added to astream_log they should be added with "
            "standardized schema for streaming events."
        )
        raise NotImplementedError(msg)

    inputs = load(run.inputs, allowed_objects="all")

    if run.run_type in {"retriever", "llm", "chat_model"}:
        return inputs

    # new style chains
    # 中文: 新型链条
    # These nest an additional 'input' key inside the 'inputs' to make sure
    # 中文: 这些在“输入”内嵌套了一个额外的“输入”键，以确保
    # the input is always a dict. We need to unpack and use the inner value.
    # 中文: 输入始终是一个字典。我们需要解压并利用其内在价值。
    inputs = inputs["input"]
    # We should try to fix this in Runnables and callbacks/tracers
    # 中文: 我们应该尝试在 Runnables 和回调/跟踪器中修复这个问题
    # Runnables should be using a None type here not a placeholder
    # 中文: Runnables 应该在此处使用 None 类型而不是占位符
    # dict.
    # 中文: 字典。
    if inputs == {"input": ""}:  # Workaround for Runnables not using None
        # The input is not known, so we don't assign data['input']
        # 中文: 输入未知，因此我们不分配 data['input']
        return None
    return inputs


def _get_standardized_outputs(
    run: Run, schema_format: Literal["original", "streaming_events", "original+chat"]
) -> Any | None:
    """Extract standardized output from a run.

    Standardizes the outputs based on the type of the runnable used.

    Args:
        run: the run object.
        schema_format: The schema format to use.

    Returns:
        An output if returned, otherwise a None
    

    中文翻译:
    从运行中提取标准化输出。
    根据所使用的可运行程序的类型标准化输出。
    参数：
        run：运行对象。
        schema_format：要使用的架构格式。
    返回：
        如果返回则输出，否则 None"""
    outputs = load(run.outputs, allowed_objects="all")
    if schema_format == "original":
        if run.run_type == "prompt" and "output" in outputs:
            # These were previously dumped before the tracer.
            # 中文: 这些先前已在追踪器之前倾倒。
            # Now we needn't do anything to them.
            # 中文: 现在我们不需要对他们做任何事情。
            return outputs["output"]
        # Return the old schema, without standardizing anything
        # 中文: 返回旧模式，不进行任何标准化
        return outputs

    if run.run_type in {"retriever", "llm", "chat_model"}:
        return outputs

    if isinstance(outputs, dict):
        return outputs.get("output", None)

    return None


@overload
def _astream_log_implementation(
    runnable: Runnable[Input, Output],
    value: Any,
    config: RunnableConfig | None = None,
    *,
    stream: LogStreamCallbackHandler,
    diff: Literal[True] = True,
    with_streamed_output_list: bool = True,
    **kwargs: Any,
) -> AsyncIterator[RunLogPatch]: ...


@overload
def _astream_log_implementation(
    runnable: Runnable[Input, Output],
    value: Any,
    config: RunnableConfig | None = None,
    *,
    stream: LogStreamCallbackHandler,
    diff: Literal[False],
    with_streamed_output_list: bool = True,
    **kwargs: Any,
) -> AsyncIterator[RunLog]: ...


async def _astream_log_implementation(
    runnable: Runnable[Input, Output],
    value: Any,
    config: RunnableConfig | None = None,
    *,
    stream: LogStreamCallbackHandler,
    diff: bool = True,
    with_streamed_output_list: bool = True,
    **kwargs: Any,
) -> AsyncIterator[RunLogPatch] | AsyncIterator[RunLog]:
    """Implementation of astream_log for a given runnable.

    The implementation has been factored out (at least temporarily) as both
    astream_log and astream_events relies on it.

    Args:
        runnable: The runnable to run in streaming mode.
        value: The input to the runnable.
        config: The config to pass to the runnable.
        stream: The stream to send the run logs to.
        diff: Whether to yield run log patches (True) or full run logs (False).
        with_streamed_output_list: Whether to include a list of all streamed
            outputs in each patch. If `False`, only the final output will be included
            in the patches.
        **kwargs: Additional keyword arguments to pass to the runnable.

    Raises:
        ValueError: If the callbacks in the config are of an unexpected type.

    Yields:
        The run log patches or states, depending on the value of `diff`.
    

    中文翻译:
    给定可运行对象的 astream_log 实现。
    实施已被排除（至少暂时），因为两者
    astream_log 和 astream_events 依赖于它。
    参数：
        runnable：以流模式运行的runnable。
        value：可运行的输入。
        config：传递给可运行的配置。
        Stream：将运行日志发送到的流。
        diff：是否生成运行日志补丁（True）或完整运行日志（False）。
        with_streamed_output_list：是否包含所有流的列表
            每个补丁中的输出。如果为“False”，则仅包含最终输出
            在补丁中。
        **kwargs：传递给可运行程序的附加关键字参数。
    加薪：
        ValueError：如果配置中的回调属于意外类型。
    产量：
        运行日志补丁或状态，取决于“diff”的值。"""
    # Assign the stream handler to the config
    # 中文: 将流处理程序分配给配置
    config = ensure_config(config)
    callbacks = config.get("callbacks")
    if callbacks is None:
        config["callbacks"] = [stream]
    elif isinstance(callbacks, list):
        config["callbacks"] = [*callbacks, stream]
    elif isinstance(callbacks, BaseCallbackManager):
        callbacks = callbacks.copy()
        callbacks.add_handler(stream, inherit=True)
        config["callbacks"] = callbacks
    else:
        msg = (
            f"Unexpected type for callbacks: {callbacks}."
            "Expected None, list or AsyncCallbackManager."
        )
        raise ValueError(msg)

    # Call the runnable in streaming mode,
    # 中文: 以流模式调用可运行程序，
    # add each chunk to the output stream
    # 中文: 将每个块添加到输出流
    async def consume_astream() -> None:
        try:
            prev_final_output: Output | None = None
            final_output: Output | None = None

            async for chunk in runnable.astream(value, config, **kwargs):
                prev_final_output = final_output
                if final_output is None:
                    final_output = chunk
                else:
                    try:
                        final_output = final_output + chunk  # type: ignore[operator]
                    except TypeError:
                        prev_final_output = None
                        final_output = chunk
                patches: list[dict[str, Any]] = []
                if with_streamed_output_list:
                    patches.append(
                        {
                            "op": "add",
                            "path": "/streamed_output/-",
                            # chunk cannot be shared between
                            # 中文: 块不能在之间共享
                            # streamed_output and final_output
                            # 中文: Streamed_output 和 Final_output
                            # otherwise jsonpatch.apply will
                            # 中文: 否则 jsonpatch.apply 将会
                            # modify both
                            # 中文: 修改两者
                            "value": copy.deepcopy(chunk),
                        }
                    )
                patches.extend(
                    {**op, "path": f"/final_output{op['path']}"}
                    for op in jsonpatch.JsonPatch.from_diff(
                        prev_final_output, final_output, dumps=dumps
                    )
                )
                await stream.send_stream.send(RunLogPatch(*patches))
        finally:
            await stream.send_stream.aclose()

    # Start the runnable in a task, so we can start consuming output
    # 中文: 在任务中启动可运行程序，以便我们可以开始使用输出
    task = asyncio.create_task(consume_astream())
    try:
        # Yield each chunk from the output stream
        # 中文: 从输出流中生成每个块
        if diff:
            async for log in stream:
                yield log
        else:
            state = RunLog(state=None)  # type: ignore[arg-type]
            async for log in stream:
                state += log
                yield state
    finally:
        # Wait for the runnable to finish, if not cancelled (eg. by break)
        # 中文: 等待可运行完成，如果没有取消（例如通过中断）
        with contextlib.suppress(asyncio.CancelledError):
            await task
