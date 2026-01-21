"""Internal tracer to power the event stream API.

中文翻译:
为事件流 API 提供支持的内部跟踪器。"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import (
    TYPE_CHECKING,
    Any,
    TypedDict,
    TypeVar,
    cast,
)

from typing_extensions import NotRequired, override

from langchain_core.callbacks.base import AsyncCallbackHandler, BaseCallbackManager
from langchain_core.messages import AIMessageChunk, BaseMessage, BaseMessageChunk
from langchain_core.outputs import (
    ChatGenerationChunk,
    GenerationChunk,
    LLMResult,
)
from langchain_core.runnables import ensure_config
from langchain_core.runnables.schema import (
    CustomStreamEvent,
    EventData,
    StandardStreamEvent,
    StreamEvent,
)
from langchain_core.runnables.utils import (
    Input,
    Output,
    _RootEventFilter,
)
from langchain_core.tracers._streaming import _StreamingCallbackHandler
from langchain_core.tracers.log_stream import (
    LogStreamCallbackHandler,
    RunLog,
    _astream_log_implementation,
)
from langchain_core.tracers.memory_stream import _MemoryStream
from langchain_core.utils.aiter import aclosing
from langchain_core.utils.uuid import uuid7

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator, Sequence
    from uuid import UUID

    from langchain_core.documents import Document
    from langchain_core.runnables import Runnable, RunnableConfig
    from langchain_core.tracers.log_stream import LogEntry

logger = logging.getLogger(__name__)


class RunInfo(TypedDict):
    """Information about a run.

    This is used to keep track of the metadata associated with a run.
    

    中文翻译:
    有关跑步的信息。
    这用于跟踪与运行相关的元数据。"""

    name: str
    """The name of the run.

    中文翻译:
    运行的名称。"""
    tags: list[str]
    """The tags associated with the run.

    中文翻译:
    与运行关联的标签。"""
    metadata: dict[str, Any]
    """The metadata associated with the run.

    中文翻译:
    与运行关联的元数据。"""
    run_type: str
    """The type of the run.

    中文翻译:
    运行的类型。"""
    inputs: NotRequired[Any]
    """The inputs to the run.

    中文翻译:
    运行的输入。"""
    parent_run_id: UUID | None
    """The ID of the parent run.

    中文翻译:
    父运行的 ID。"""


def _assign_name(name: str | None, serialized: dict[str, Any] | None) -> str:
    """Assign a name to a run.

    中文翻译:
    为运行指定名称。"""
    if name is not None:
        return name
    if serialized is not None:
        if "name" in serialized:
            return cast("str", serialized["name"])
        if "id" in serialized:
            return cast("str", serialized["id"][-1])
    return "Unnamed"


T = TypeVar("T")


class _AstreamEventsCallbackHandler(AsyncCallbackHandler, _StreamingCallbackHandler):
    """An implementation of an async callback handler for astream events.

    中文翻译:
    astream 事件的异步回调处理程序的实现。"""

    def __init__(
        self,
        *args: Any,
        include_names: Sequence[str] | None = None,
        include_types: Sequence[str] | None = None,
        include_tags: Sequence[str] | None = None,
        exclude_names: Sequence[str] | None = None,
        exclude_types: Sequence[str] | None = None,
        exclude_tags: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the tracer.

        中文翻译:
        初始化跟踪器。"""
        super().__init__(*args, **kwargs)
        # Map of run ID to run info.
        # 中文: 运行 ID 到运行信息的映射。
        # the entry corresponding to a given run id is cleaned
        # 中文: 与给定运行 ID 对应的条目被清理
        # up when each corresponding run ends.
        # 中文: 当每个相应的运行结束时。
        self.run_map: dict[UUID, RunInfo] = {}
        # The callback event that corresponds to the end of a parent run
        # 中文: 对应于父运行结束的回调事件
        # may be invoked BEFORE the callback event that corresponds to the end
        # 中文: 可以在对应于结束的回调事件之前调用
        # of a child run, which results in clean up of run_map.
        # 中文: 子运行，这会导致 run_map 的清理。
        # So we keep track of the mapping between children and parent run IDs
        # 中文: 因此我们跟踪子代和父代运行 ID 之间的映射
        # in a separate container. This container is GCed when the tracer is GCed.
        # 中文: 在一个单独的容器中。当跟踪器被 GC 时，该容器也被 GC。
        self.parent_map: dict[UUID, UUID | None] = {}

        self.is_tapped: dict[UUID, Any] = {}

        # Filter which events will be sent over the queue.
        # 中文: 过滤哪些事件将通过队列发送。
        self.root_event_filter = _RootEventFilter(
            include_names=include_names,
            include_types=include_types,
            include_tags=include_tags,
            exclude_names=exclude_names,
            exclude_types=exclude_types,
            exclude_tags=exclude_tags,
        )

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
        memory_stream = _MemoryStream[StreamEvent](loop)
        self.send_stream = memory_stream.get_send_stream()
        self.receive_stream = memory_stream.get_receive_stream()

    def _get_parent_ids(self, run_id: UUID) -> list[str]:
        """Get the parent IDs of a run (non-recursively) cast to strings.

        中文翻译:
        获取运行（非递归）转换为字符串的父 ID。"""
        parent_ids = []

        while parent_id := self.parent_map.get(run_id):
            str_parent_id = str(parent_id)
            if str_parent_id in parent_ids:
                msg = (
                    f"Parent ID {parent_id} is already in the parent_ids list. "
                    f"This should never happen."
                )
                raise AssertionError(msg)
            parent_ids.append(str_parent_id)
            run_id = parent_id

        # Return the parent IDs in reverse order, so that the first
        # 中文: 以相反的顺序返回父 ID，以便第一个
        # parent ID is the root and the last ID is the immediate parent.
        # 中文: 父 ID 是根 ID，最后一个 ID 是直接父 ID。
        return parent_ids[::-1]

    def _send(self, event: StreamEvent, event_type: str) -> None:
        """Send an event to the stream.

        中文翻译:
        将事件发送到流。"""
        if self.root_event_filter.include_event(event, event_type):
            self.send_stream.send_nowait(event)

    def __aiter__(self) -> AsyncIterator[Any]:
        """Iterate over the receive stream.

        Returns:
            An async iterator over the receive stream.
        

        中文翻译:
        迭代接收流。
        返回：
            接收流上的异步迭代器。"""
        return self.receive_stream.__aiter__()

    async def tap_output_aiter(
        self, run_id: UUID, output: AsyncIterator[T]
    ) -> AsyncIterator[T]:
        """Tap the output aiter.

        This method is used to tap the output of a Runnable that produces
        an async iterator. It is used to generate stream events for the
        output of the Runnable.

        Args:
            run_id: The ID of the run.
            output: The output of the Runnable.

        Yields:
            The output of the Runnable.
        

        中文翻译:
        点击输出 aiter。
        此方法用于点击 Runnable 的输出，该 Runnable 产生
        异步迭代器。它用于生成流事件
        可运行的输出。
        参数：
            run_id：运行的 ID。
            输出：Runnable 的输出。
        产量：
            可运行的输出。"""
        sentinel = object()
        # atomic check and set
        # 中文: 原子检查和设置
        tap = self.is_tapped.setdefault(run_id, sentinel)
        # wait for first chunk
        # 中文: 等待第一个块
        first = await anext(output, sentinel)
        if first is sentinel:
            return
        # get run info
        # 中文: 获取跑步信息
        run_info = self.run_map.get(run_id)
        if run_info is None:
            # run has finished, don't issue any stream events
            # 中文: 运行已完成，不发出任何流事件
            yield cast("T", first)
            return
        if tap is sentinel:
            # if we are the first to tap, issue stream events
            # 中文: 如果我们是第一个点击的，则发出流事件
            event: StandardStreamEvent = {
                "event": f"on_{run_info['run_type']}_stream",
                "run_id": str(run_id),
                "name": run_info["name"],
                "tags": run_info["tags"],
                "metadata": run_info["metadata"],
                "data": {},
                "parent_ids": self._get_parent_ids(run_id),
            }
            self._send({**event, "data": {"chunk": first}}, run_info["run_type"])
            yield cast("T", first)
            # consume the rest of the output
            # 中文: 消耗剩余的输出
            async for chunk in output:
                self._send(
                    {**event, "data": {"chunk": chunk}},
                    run_info["run_type"],
                )
                yield chunk
        else:
            # otherwise just pass through
            # 中文: 否则就通过
            yield cast("T", first)
            # consume the rest of the output
            # 中文: 消耗剩余的输出
            async for chunk in output:
                yield chunk

    def tap_output_iter(self, run_id: UUID, output: Iterator[T]) -> Iterator[T]:
        """Tap the output iter.

        Args:
            run_id: The ID of the run.
            output: The output of the Runnable.

        Yields:
            The output of the Runnable.
        

        中文翻译:
        点击输出迭代器。
        参数：
            run_id：运行的 ID。
            输出：Runnable 的输出。
        产量：
            可运行的输出。"""
        sentinel = object()
        # atomic check and set
        # 中文: 原子检查和设置
        tap = self.is_tapped.setdefault(run_id, sentinel)
        # wait for first chunk
        # 中文: 等待第一个块
        first = next(output, sentinel)
        if first is sentinel:
            return
        # get run info
        # 中文: 获取跑步信息
        run_info = self.run_map.get(run_id)
        if run_info is None:
            # run has finished, don't issue any stream events
            # 中文: 运行已完成，不发出任何流事件
            yield cast("T", first)
            return
        if tap is sentinel:
            # if we are the first to tap, issue stream events
            # 中文: 如果我们是第一个点击的，则发出流事件
            event: StandardStreamEvent = {
                "event": f"on_{run_info['run_type']}_stream",
                "run_id": str(run_id),
                "name": run_info["name"],
                "tags": run_info["tags"],
                "metadata": run_info["metadata"],
                "data": {},
                "parent_ids": self._get_parent_ids(run_id),
            }
            self._send({**event, "data": {"chunk": first}}, run_info["run_type"])
            yield cast("T", first)
            # consume the rest of the output
            # 中文: 消耗剩余的输出
            for chunk in output:
                self._send(
                    {**event, "data": {"chunk": chunk}},
                    run_info["run_type"],
                )
                yield chunk
        else:
            # otherwise just pass through
            # 中文: 否则就通过
            yield cast("T", first)
            # consume the rest of the output
            # 中文: 消耗剩余的输出
            for chunk in output:
                yield chunk

    def _write_run_start_info(
        self,
        run_id: UUID,
        *,
        tags: list[str] | None,
        metadata: dict[str, Any] | None,
        parent_run_id: UUID | None,
        name_: str,
        run_type: str,
        **kwargs: Any,
    ) -> None:
        """Update the run info.

        中文翻译:
        更新运行信息。"""
        info: RunInfo = {
            "tags": tags or [],
            "metadata": metadata or {},
            "name": name_,
            "run_type": run_type,
            "parent_run_id": parent_run_id,
        }

        if "inputs" in kwargs:
            # Handle inputs in a special case to allow inputs to be an
            # 中文: 处理特殊情况下的输入以允许输入成为
            # optionally provided and distinguish between missing value
            # 中文: 可选提供并区分缺失值
            # vs. None value.
            # 中文: 与无值。
            info["inputs"] = kwargs["inputs"]

        self.run_map[run_id] = info
        self.parent_map[run_id] = parent_run_id

    @override
    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        tags: list[str] | None = None,
        parent_run_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Start a trace for a chat model run.

        中文翻译:
        启动聊天模型运行的跟踪。"""
        name_ = _assign_name(name, serialized)
        run_type = "chat_model"

        self._write_run_start_info(
            run_id,
            tags=tags,
            metadata=metadata,
            parent_run_id=parent_run_id,
            name_=name_,
            run_type=run_type,
            inputs={"messages": messages},
        )

        self._send(
            {
                "event": "on_chat_model_start",
                "data": {
                    "input": {"messages": messages},
                },
                "name": name_,
                "tags": tags or [],
                "run_id": str(run_id),
                "metadata": metadata or {},
                "parent_ids": self._get_parent_ids(run_id),
            },
            run_type,
        )

    @override
    async def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        tags: list[str] | None = None,
        parent_run_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Start a trace for a (non-chat model) LLM run.

        中文翻译:
        启动（非聊天模型）LLM 运行的跟踪。"""
        name_ = _assign_name(name, serialized)
        run_type = "llm"

        self._write_run_start_info(
            run_id,
            tags=tags,
            metadata=metadata,
            parent_run_id=parent_run_id,
            name_=name_,
            run_type=run_type,
            inputs={"prompts": prompts},
        )

        self._send(
            {
                "event": "on_llm_start",
                "data": {
                    "input": {
                        "prompts": prompts,
                    }
                },
                "name": name_,
                "tags": tags or [],
                "run_id": str(run_id),
                "metadata": metadata or {},
                "parent_ids": self._get_parent_ids(run_id),
            },
            run_type,
        )

    @override
    async def on_custom_event(
        self,
        name: str,
        data: Any,
        *,
        run_id: UUID,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Generate a custom astream event.

        中文翻译:
        生成自定义 astream 事件。"""
        event = CustomStreamEvent(
            event="on_custom_event",
            run_id=str(run_id),
            name=name,
            tags=tags or [],
            metadata=metadata or {},
            data=data,
            parent_ids=self._get_parent_ids(run_id),
        )
        self._send(event, name)

    @override
    async def on_llm_new_token(
        self,
        token: str,
        *,
        chunk: GenerationChunk | ChatGenerationChunk | None = None,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Run on new output token. Only available when streaming is enabled.

        For both chat models and non-chat models (legacy LLMs).

        Raises:
            ValueError: If the run type is not `llm` or `chat_model`.
            AssertionError: If the run ID is not found in the run map.
        

        中文翻译:
        在新的输出令牌上运行。仅在启用流式传输时可用。
        适用于聊天模型和非聊天模型（传统法学硕士）。
        加薪：
            ValueError：如果运行类型不是“llm”或“chat_model”。
            AssertionError：如果在运行映射中找不到运行 ID。"""
        run_info = self.run_map.get(run_id)
        chunk_: GenerationChunk | BaseMessageChunk

        if run_info is None:
            msg = f"Run ID {run_id} not found in run map."
            raise AssertionError(msg)
        if self.is_tapped.get(run_id):
            return
        if run_info["run_type"] == "chat_model":
            event = "on_chat_model_stream"

            if chunk is None:
                chunk_ = AIMessageChunk(content=token)
            else:
                chunk_ = cast("ChatGenerationChunk", chunk).message

        elif run_info["run_type"] == "llm":
            event = "on_llm_stream"
            if chunk is None:
                chunk_ = GenerationChunk(text=token)
            else:
                chunk_ = cast("GenerationChunk", chunk)
        else:
            msg = f"Unexpected run type: {run_info['run_type']}"
            raise ValueError(msg)

        self._send(
            {
                "event": event,
                "data": {
                    "chunk": chunk_,
                },
                "run_id": str(run_id),
                "name": run_info["name"],
                "tags": run_info["tags"],
                "metadata": run_info["metadata"],
                "parent_ids": self._get_parent_ids(run_id),
            },
            run_info["run_type"],
        )

    @override
    async def on_llm_end(
        self, response: LLMResult, *, run_id: UUID, **kwargs: Any
    ) -> None:
        """End a trace for a model run.

        For both chat models and non-chat models (legacy LLMs).

        Raises:
            ValueError: If the run type is not `'llm'` or `'chat_model'`.
        

        中文翻译:
        结束模型运行的跟踪。
        适用于聊天模型和非聊天模型（传统法学硕士）。
        加薪：
            ValueError：如果运行类型不是“llm”或“chat_model”。"""
        run_info = self.run_map.pop(run_id)
        inputs_ = run_info.get("inputs")

        generations: list[list[GenerationChunk]] | list[list[ChatGenerationChunk]]
        output: dict | BaseMessage = {}

        if run_info["run_type"] == "chat_model":
            generations = cast("list[list[ChatGenerationChunk]]", response.generations)
            for gen in generations:
                if output != {}:
                    break
                for chunk in gen:
                    output = chunk.message
                    break

            event = "on_chat_model_end"
        elif run_info["run_type"] == "llm":
            generations = cast("list[list[GenerationChunk]]", response.generations)
            output = {
                "generations": [
                    [
                        {
                            "text": chunk.text,
                            "generation_info": chunk.generation_info,
                            "type": chunk.type,
                        }
                        for chunk in gen
                    ]
                    for gen in generations
                ],
                "llm_output": response.llm_output,
            }
            event = "on_llm_end"
        else:
            msg = f"Unexpected run type: {run_info['run_type']}"
            raise ValueError(msg)

        self._send(
            {
                "event": event,
                "data": {"output": output, "input": inputs_},
                "run_id": str(run_id),
                "name": run_info["name"],
                "tags": run_info["tags"],
                "metadata": run_info["metadata"],
                "parent_ids": self._get_parent_ids(run_id),
            },
            run_info["run_type"],
        )

    async def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        tags: list[str] | None = None,
        parent_run_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
        run_type: str | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Start a trace for a chain run.

        中文翻译:
        开始跟踪链运行。"""
        name_ = _assign_name(name, serialized)
        run_type_ = run_type or "chain"

        data: EventData = {}

        # Work-around Runnable core code not sending input in some
        # 中文: 解决可运行核心代码在某些情况下不发送输入的问题
        # cases.
        # 中文: 案例。
        if inputs != {"input": ""}:
            data["input"] = inputs
            kwargs["inputs"] = inputs

        self._write_run_start_info(
            run_id,
            tags=tags,
            metadata=metadata,
            parent_run_id=parent_run_id,
            name_=name_,
            run_type=run_type_,
            **kwargs,
        )

        self._send(
            {
                "event": f"on_{run_type_}_start",
                "data": data,
                "name": name_,
                "tags": tags or [],
                "run_id": str(run_id),
                "metadata": metadata or {},
                "parent_ids": self._get_parent_ids(run_id),
            },
            run_type_,
        )

    @override
    async def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """End a trace for a chain run.

        中文翻译:
        结束连锁运行的跟踪。"""
        run_info = self.run_map.pop(run_id)
        run_type = run_info["run_type"]

        event = f"on_{run_type}_end"

        inputs = inputs or run_info.get("inputs") or {}

        data: EventData = {
            "output": outputs,
            "input": inputs,
        }

        self._send(
            {
                "event": event,
                "data": data,
                "run_id": str(run_id),
                "name": run_info["name"],
                "tags": run_info["tags"],
                "metadata": run_info["metadata"],
                "parent_ids": self._get_parent_ids(run_id),
            },
            run_type,
        )

    def _get_tool_run_info_with_inputs(self, run_id: UUID) -> tuple[RunInfo, Any]:
        """Get run info for a tool and extract inputs, with validation.

        Args:
            run_id: The run ID of the tool.

        Returns:
            A tuple of (run_info, inputs).

        Raises:
            AssertionError: If the run ID is a tool call and does not have inputs.
        

        中文翻译:
        获取工具的运行信息并提取输入并进行验证。
        参数：
            run_id：工具的运行 ID。
        返回：
            (run_info, input) 的元组。
        加薪：
            AssertionError：如果运行 ID 是工具调用并且没有输入。"""
        run_info = self.run_map.pop(run_id)
        if "inputs" not in run_info:
            msg = (
                f"Run ID {run_id} is a tool call and is expected to have "
                f"inputs associated with it."
            )
            raise AssertionError(msg)
        inputs = run_info["inputs"]
        return run_info, inputs

    @override
    async def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        tags: list[str] | None = None,
        parent_run_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
        name: str | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Start a trace for a tool run.

        中文翻译:
        启动工具运行的跟踪。"""
        name_ = _assign_name(name, serialized)

        self._write_run_start_info(
            run_id,
            tags=tags,
            metadata=metadata,
            parent_run_id=parent_run_id,
            name_=name_,
            run_type="tool",
            inputs=inputs,
        )

        self._send(
            {
                "event": "on_tool_start",
                "data": {
                    "input": inputs or {},
                },
                "name": name_,
                "tags": tags or [],
                "run_id": str(run_id),
                "metadata": metadata or {},
                "parent_ids": self._get_parent_ids(run_id),
            },
            "tool",
        )

    @override
    async def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Run when tool errors.

        中文翻译:
        当工具出错时运行。"""
        run_info, inputs = self._get_tool_run_info_with_inputs(run_id)

        self._send(
            {
                "event": "on_tool_error",
                "data": {
                    "error": error,
                    "input": inputs,
                },
                "run_id": str(run_id),
                "name": run_info["name"],
                "tags": run_info["tags"],
                "metadata": run_info["metadata"],
                "parent_ids": self._get_parent_ids(run_id),
            },
            "tool",
        )

    @override
    async def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        """End a trace for a tool run.

        中文翻译:
        结束工具运行的跟踪。"""
        run_info, inputs = self._get_tool_run_info_with_inputs(run_id)

        self._send(
            {
                "event": "on_tool_end",
                "data": {
                    "output": output,
                    "input": inputs,
                },
                "run_id": str(run_id),
                "name": run_info["name"],
                "tags": run_info["tags"],
                "metadata": run_info["metadata"],
                "parent_ids": self._get_parent_ids(run_id),
            },
            "tool",
        )

    @override
    async def on_retriever_start(
        self,
        serialized: dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Run when Retriever starts running.

        中文翻译:
        当 Retriever 开始运行时运行。"""
        name_ = _assign_name(name, serialized)
        run_type = "retriever"

        self._write_run_start_info(
            run_id,
            tags=tags,
            metadata=metadata,
            parent_run_id=parent_run_id,
            name_=name_,
            run_type=run_type,
            inputs={"query": query},
        )

        self._send(
            {
                "event": "on_retriever_start",
                "data": {
                    "input": {
                        "query": query,
                    }
                },
                "name": name_,
                "tags": tags or [],
                "run_id": str(run_id),
                "metadata": metadata or {},
                "parent_ids": self._get_parent_ids(run_id),
            },
            run_type,
        )

    @override
    async def on_retriever_end(
        self, documents: Sequence[Document], *, run_id: UUID, **kwargs: Any
    ) -> None:
        """Run when Retriever ends running.

        中文翻译:
        当猎犬结束运行时运行。"""
        run_info = self.run_map.pop(run_id)

        self._send(
            {
                "event": "on_retriever_end",
                "data": {
                    "output": documents,
                    "input": run_info.get("inputs"),
                },
                "run_id": str(run_id),
                "name": run_info["name"],
                "tags": run_info["tags"],
                "metadata": run_info["metadata"],
                "parent_ids": self._get_parent_ids(run_id),
            },
            run_info["run_type"],
        )

    def __deepcopy__(self, memo: dict) -> _AstreamEventsCallbackHandler:
        """Return self.

        中文翻译:
        回归自我。"""
        return self

    def __copy__(self) -> _AstreamEventsCallbackHandler:
        """Return self.

        中文翻译:
        回归自我。"""
        return self


async def _astream_events_implementation_v1(
    runnable: Runnable[Input, Output],
    value: Any,
    config: RunnableConfig | None = None,
    *,
    include_names: Sequence[str] | None = None,
    include_types: Sequence[str] | None = None,
    include_tags: Sequence[str] | None = None,
    exclude_names: Sequence[str] | None = None,
    exclude_types: Sequence[str] | None = None,
    exclude_tags: Sequence[str] | None = None,
    **kwargs: Any,
) -> AsyncIterator[StandardStreamEvent]:
    stream = LogStreamCallbackHandler(
        auto_close=False,
        include_names=include_names,
        include_types=include_types,
        include_tags=include_tags,
        exclude_names=exclude_names,
        exclude_types=exclude_types,
        exclude_tags=exclude_tags,
        _schema_format="streaming_events",
    )

    run_log = RunLog(state=None)  # type: ignore[arg-type]
    encountered_start_event = False

    root_event_filter = _RootEventFilter(
        include_names=include_names,
        include_types=include_types,
        include_tags=include_tags,
        exclude_names=exclude_names,
        exclude_types=exclude_types,
        exclude_tags=exclude_tags,
    )

    config = ensure_config(config)
    root_tags = config.get("tags", [])
    root_metadata = config.get("metadata", {})
    root_name = config.get("run_name", runnable.get_name())

    async for log in _astream_log_implementation(
        runnable,
        value,
        config=config,
        stream=stream,
        diff=True,
        with_streamed_output_list=True,
        **kwargs,
    ):
        run_log += log

        if not encountered_start_event:
            # Yield the start event for the root runnable.
            # 中文: 产生根可运行的启动事件。
            encountered_start_event = True
            state = run_log.state.copy()

            event = StandardStreamEvent(
                event=f"on_{state['type']}_start",
                run_id=state["id"],
                name=root_name,
                tags=root_tags,
                metadata=root_metadata,
                data={
                    "input": value,
                },
                parent_ids=[],  # Not supported in v1
            )

            if root_event_filter.include_event(event, state["type"]):
                yield event

        paths = {
            op["path"].split("/")[2]
            for op in log.ops
            if op["path"].startswith("/logs/")
        }
        # Elements in a set should be iterated in the same order
        # 中文: 集合中的元素应以相同的顺序迭代
        # as they were inserted in modern python versions.
        # 中文: 因为它们被插入到现代 python 版本中。
        for path in paths:
            data: EventData = {}
            log_entry: LogEntry = run_log.state["logs"][path]
            if log_entry["end_time"] is None:
                event_type = "stream" if log_entry["streamed_output"] else "start"
            else:
                event_type = "end"

            if event_type == "start":
                # Include the inputs with the start event if they are available.
                # 中文: 将输入包含在开始事件中（如果可用）。
                # Usually they will NOT be available for components that operate
                # 中文: 通常它们不可用于运行的组件
                # on streams, since those components stream the input and
                # 中文: 在流上，因为这些组件流式传输输入和
                # don't know its final value until the end of the stream.
                # 中文: 在流结束之前不知道其最终值。
                inputs = log_entry.get("inputs")
                if inputs is not None:
                    data["input"] = inputs

            if event_type == "end":
                inputs = log_entry.get("inputs")
                if inputs is not None:
                    data["input"] = inputs

                # None is a VALID output for an end event
                # 中文: None 是结束事件的有效输出
                data["output"] = log_entry["final_output"]

            if event_type == "stream":
                num_chunks = len(log_entry["streamed_output"])
                if num_chunks != 1:
                    msg = (
                        f"Expected exactly one chunk of streamed output, "
                        f"got {num_chunks} instead. This is impossible. "
                        f"Encountered in: {log_entry['name']}"
                    )
                    raise AssertionError(msg)

                data = {"chunk": log_entry["streamed_output"][0]}
                # Clean up the stream, we don't need it anymore.
                # 中文: 清理流，我们不再需要它了。
                # And this avoids duplicates as well!
                # 中文: 这也避免了重复！
                log_entry["streamed_output"] = []

            yield StandardStreamEvent(
                event=f"on_{log_entry['type']}_{event_type}",
                name=log_entry["name"],
                run_id=log_entry["id"],
                tags=log_entry["tags"],
                metadata=log_entry["metadata"],
                data=data,
                parent_ids=[],  # Not supported in v1
            )

        # Finally, we take care of the streaming output from the root chain
        # 中文: 最后，我们处理根链的流输出
        # if there is any.
        # 中文: 如果有的话。
        state = run_log.state
        if state["streamed_output"]:
            num_chunks = len(state["streamed_output"])
            if num_chunks != 1:
                msg = (
                    f"Expected exactly one chunk of streamed output, "
                    f"got {num_chunks} instead. This is impossible. "
                    f"Encountered in: {state['name']}"
                )
                raise AssertionError(msg)

            data = {"chunk": state["streamed_output"][0]}
            # Clean up the stream, we don't need it anymore.
            # 中文: 清理流，我们不再需要它了。
            state["streamed_output"] = []

            event = StandardStreamEvent(
                event=f"on_{state['type']}_stream",
                run_id=state["id"],
                tags=root_tags,
                metadata=root_metadata,
                name=root_name,
                data=data,
                parent_ids=[],  # Not supported in v1
            )
            if root_event_filter.include_event(event, state["type"]):
                yield event

    state = run_log.state

    # Finally yield the end event for the root runnable.
    # 中文: 最后产生根可运行对象的结束事件。
    event = StandardStreamEvent(
        event=f"on_{state['type']}_end",
        name=root_name,
        run_id=state["id"],
        tags=root_tags,
        metadata=root_metadata,
        data={
            "output": state["final_output"],
        },
        parent_ids=[],  # Not supported in v1
    )
    if root_event_filter.include_event(event, state["type"]):
        yield event


async def _astream_events_implementation_v2(
    runnable: Runnable[Input, Output],
    value: Any,
    config: RunnableConfig | None = None,
    *,
    include_names: Sequence[str] | None = None,
    include_types: Sequence[str] | None = None,
    include_tags: Sequence[str] | None = None,
    exclude_names: Sequence[str] | None = None,
    exclude_types: Sequence[str] | None = None,
    exclude_tags: Sequence[str] | None = None,
    **kwargs: Any,
) -> AsyncIterator[StandardStreamEvent]:
    """Implementation of the astream events API for V2 runnables.

    中文翻译:
    为 V2 可运行对象实现 astream 事件 API。"""
    event_streamer = _AstreamEventsCallbackHandler(
        include_names=include_names,
        include_types=include_types,
        include_tags=include_tags,
        exclude_names=exclude_names,
        exclude_types=exclude_types,
        exclude_tags=exclude_tags,
    )

    # Assign the stream handler to the config
    # 中文: 将流处理程序分配给配置
    config = ensure_config(config)
    if "run_id" in config:
        run_id = cast("UUID", config["run_id"])
    else:
        run_id = uuid7()
        config["run_id"] = run_id
    callbacks = config.get("callbacks")
    if callbacks is None:
        config["callbacks"] = [event_streamer]
    elif isinstance(callbacks, list):
        config["callbacks"] = [*callbacks, event_streamer]
    elif isinstance(callbacks, BaseCallbackManager):
        callbacks = callbacks.copy()
        callbacks.add_handler(event_streamer, inherit=True)
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
            # if astream also calls tap_output_aiter this will be a no-op
            # 中文: 如果 astream 也调用 tap_output_aiter 这将是一个空操作
            async with aclosing(runnable.astream(value, config, **kwargs)) as stream:
                async for _ in event_streamer.tap_output_aiter(run_id, stream):
                    # All the content will be picked up
                    # 中文: 所有内容将被拾取
                    pass
        finally:
            await event_streamer.send_stream.aclose()

    # Start the runnable in a task, so we can start consuming output
    # 中文: 在任务中启动可运行程序，以便我们可以开始使用输出
    task = asyncio.create_task(consume_astream())

    first_event_sent = False
    first_event_run_id = None

    try:
        async for event in event_streamer:
            if not first_event_sent:
                first_event_sent = True
                # This is a work-around an issue where the inputs into the
                # 中文: 这是一个解决问题的方法，其中输入
                # chain are not available until the entire input is consumed.
                # 中文: 在消耗整个输入之前，链不可用。
                # As a temporary solution, we'll modify the input to be the input
                # 中文: 作为临时解决方案，我们将输入修改为输入
                # that was passed into the chain.
                # 中文: 被传递到链中。
                event["data"]["input"] = value
                first_event_run_id = event["run_id"]
                yield event
                continue

            # If it's the end event corresponding to the root runnable
            # 中文: 如果是根runnable对应的结束事件
            # we don't include the input in the event since it's guaranteed
            # 中文: 我们不将输入包含在事件中，因为它是有保证的
            # to be included in the first event.
            # 中文: 包含在第一个事件中。
            if (
                event["run_id"] == first_event_run_id
                and event["event"].endswith("_end")
                and "input" in event["data"]
            ):
                del event["data"]["input"]

            yield event
    except asyncio.CancelledError as exc:
        # Cancel the task if it's still running
        # 中文: 如果任务仍在运行，则取消该任务
        task.cancel(exc.args[0] if exc.args else None)
        raise
    finally:
        # Cancel the task if it's still running
        # 中文: 如果任务仍在运行，则取消该任务
        task.cancel()
        # Await it anyway, to run any cleanup code, and propagate any exceptions
        # 中文: 无论如何都要等待，运行任何清理代码并传播任何异常
        with contextlib.suppress(asyncio.CancelledError):
            await task
