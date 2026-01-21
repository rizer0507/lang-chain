"""Base interfaces for tracing runs.

中文翻译:
用于跟踪运行的基本接口。"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
)

from typing_extensions import override

from langchain_core.callbacks.base import AsyncCallbackHandler, BaseCallbackHandler
from langchain_core.exceptions import TracerException  # noqa: F401
from langchain_core.tracers.core import _TracerCore

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

    from tenacity import RetryCallState

    from langchain_core.documents import Document
    from langchain_core.messages import BaseMessage
    from langchain_core.outputs import ChatGenerationChunk, GenerationChunk, LLMResult
    from langchain_core.tracers.schemas import Run

logger = logging.getLogger(__name__)


class BaseTracer(_TracerCore, BaseCallbackHandler, ABC):
    """Base interface for tracers.

    中文翻译:
    跟踪器的基本接口。"""

    @abstractmethod
    def _persist_run(self, run: Run) -> None:
        """Persist a run.

        中文翻译:
        坚持跑步。"""

    def _start_trace(self, run: Run) -> None:
        """Start a trace for a run.

        中文翻译:
        开始追踪跑步。"""
        super()._start_trace(run)
        self._on_run_create(run)

    def _end_trace(self, run: Run) -> None:
        """End a trace for a run.

        中文翻译:
        结束跑步追踪。"""
        if not run.parent_run_id:
            self._persist_run(run)
        self.run_map.pop(str(run.id))
        self._on_run_update(run)

    def on_chat_model_start(
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
    ) -> Run:
        """Start a trace for an LLM run.

        Args:
            serialized: The serialized model.
            messages: The messages to start the chat with.
            run_id: The run ID.
            tags: The tags for the run.
            parent_run_id: The parent run ID.
            metadata: The metadata for the run.
            name: The name of the run.
            **kwargs: Additional arguments.

        Returns:
            The run.
        

        中文翻译:
        启动 LLM 运行的跟踪。
        参数：
            序列化：序列化模型。
            messages：开始聊天的消息。
            run_id：运行 ID。
            标签：运行的标签。
            Parent_run_id：父运行 ID。
            元数据：运行的元数据。
            名称：运行的名称。
            **kwargs：附加参数。
        返回：
            奔跑。"""
        chat_model_run = self._create_chat_model_run(
            serialized=serialized,
            messages=messages,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            name=name,
            **kwargs,
        )
        self._start_trace(chat_model_run)
        self._on_chat_model_start(chat_model_run)
        return chat_model_run

    def on_llm_start(
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
    ) -> Run:
        """Start a trace for an LLM run.

        Args:
            serialized: The serialized model.
            prompts: The prompts to start the LLM with.
            run_id: The run ID.
            tags: The tags for the run.
            parent_run_id: The parent run ID.
            metadata: The metadata for the run.
            name: The name of the run.
            **kwargs: Additional arguments.

        Returns:
            The run.
        

        中文翻译:
        启动 LLM 运行的跟踪。
        参数：
            序列化：序列化模型。
            提示：启动 LLM 的提示。
            run_id：运行 ID。
            标签：运行的标签。
            Parent_run_id：父运行 ID。
            元数据：运行的元数据。
            名称：运行的名称。
            **kwargs：附加参数。
        返回：
            奔跑。"""
        llm_run = self._create_llm_run(
            serialized=serialized,
            prompts=prompts,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            name=name,
            **kwargs,
        )
        self._start_trace(llm_run)
        self._on_llm_start(llm_run)
        return llm_run

    @override
    def on_llm_new_token(
        self,
        token: str,
        *,
        chunk: GenerationChunk | ChatGenerationChunk | None = None,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Run:
        """Run on new LLM token. Only available when streaming is enabled.

        Args:
            token: The token.
            chunk: The chunk.
            run_id: The run ID.
            parent_run_id: The parent run ID.
            **kwargs: Additional arguments.

        Returns:
            The run.
        

        中文翻译:
        在新的 LLM 代币上运行。仅在启用流式传输时可用。
        参数：
            令牌：令牌。
            块：块。
            run_id：运行 ID。
            Parent_run_id：父运行 ID。
            **kwargs：附加参数。
        返回：
            奔跑。"""
        # "chat_model" is only used for the experimental new streaming_events format.
        # 中文: “chat_model”仅用于实验性的新的streaming_events格式。
        # This change should not affect any existing tracers.
        # 中文: 此更改不应影响任何现有的跟踪器。
        llm_run = self._llm_run_with_token_event(
            token=token,
            run_id=run_id,
            chunk=chunk,
            parent_run_id=parent_run_id,
        )
        self._on_llm_new_token(llm_run, token, chunk)
        return llm_run

    @override
    def on_retry(
        self,
        retry_state: RetryCallState,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> Run:
        """Run on retry.

        Args:
            retry_state: The retry state.
            run_id: The run ID.
            **kwargs: Additional arguments.

        Returns:
            The run.
        

        中文翻译:
        重试时运行。
        参数：
            retry_state：重试状态。
            run_id：运行 ID。
            **kwargs：附加参数。
        返回：
            奔跑。"""
        return self._llm_run_with_retry_event(
            retry_state=retry_state,
            run_id=run_id,
        )

    @override
    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> Run:
        """End a trace for an LLM run.

        Args:
            response: The response.
            run_id: The run ID.
            **kwargs: Additional arguments.

        Returns:
            The run.
        

        中文翻译:
        结束 LLM 运行的跟踪。
        参数：
            回应：回应。
            run_id：运行 ID。
            **kwargs：附加参数。
        返回：
            奔跑。"""
        # "chat_model" is only used for the experimental new streaming_events format.
        # 中文: “chat_model”仅用于实验性的新的streaming_events格式。
        # This change should not affect any existing tracers.
        # 中文: 此更改不应影响任何现有的跟踪器。
        llm_run = self._complete_llm_run(
            response=response,
            run_id=run_id,
        )
        self._end_trace(llm_run)
        self._on_llm_end(llm_run)
        return llm_run

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> Run:
        """Handle an error for an LLM run.

        Args:
            error: The error.
            run_id: The run ID.
            **kwargs: Additional arguments.

        Returns:
            The run.
        

        中文翻译:
        处理 LLM 运行的错误。
        参数：
            错误：错误。
            run_id：运行 ID。
            **kwargs：附加参数。
        返回：
            奔跑。"""
        # "chat_model" is only used for the experimental new streaming_events format.
        # 中文: “chat_model”仅用于实验性的新的streaming_events格式。
        # This change should not affect any existing tracers.
        # 中文: 此更改不应影响任何现有的跟踪器。
        llm_run = self._errored_llm_run(
            error=error, run_id=run_id, response=kwargs.pop("response", None)
        )
        self._end_trace(llm_run)
        self._on_llm_error(llm_run)
        return llm_run

    @override
    def on_chain_start(
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
    ) -> Run:
        """Start a trace for a chain run.

        Args:
            serialized: The serialized chain.
            inputs: The inputs for the chain.
            run_id: The run ID.
            tags: The tags for the run.
            parent_run_id: The parent run ID.
            metadata: The metadata for the run.
            run_type: The type of the run.
            name: The name of the run.
            **kwargs: Additional arguments.

        Returns:
            The run.
        

        中文翻译:
        开始跟踪链运行。
        参数：
            序列化：序列化链。
            输入：链的输入。
            run_id：运行 ID。
            标签：运行的标签。
            Parent_run_id：父运行 ID。
            元数据：运行的元数据。
            run_type：运行的类型。
            名称：运行的名称。
            **kwargs：附加参数。
        返回：
            奔跑。"""
        chain_run = self._create_chain_run(
            serialized=serialized,
            inputs=inputs,
            run_id=run_id,
            tags=tags,
            parent_run_id=parent_run_id,
            metadata=metadata,
            run_type=run_type,
            name=name,
            **kwargs,
        )
        self._start_trace(chain_run)
        self._on_chain_start(chain_run)
        return chain_run

    @override
    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Run:
        """End a trace for a chain run.

        Args:
            outputs: The outputs for the chain.
            run_id: The run ID.
            inputs: The inputs for the chain.
            **kwargs: Additional arguments.

        Returns:
            The run.
        

        中文翻译:
        结束连锁运行的跟踪。
        参数：
            输出：链的输出。
            run_id：运行 ID。
            输入：链的输入。
            **kwargs：附加参数。
        返回：
            奔跑。"""
        chain_run = self._complete_chain_run(
            outputs=outputs,
            run_id=run_id,
            inputs=inputs,
        )
        self._end_trace(chain_run)
        self._on_chain_end(chain_run)
        return chain_run

    @override
    def on_chain_error(
        self,
        error: BaseException,
        *,
        inputs: dict[str, Any] | None = None,
        run_id: UUID,
        **kwargs: Any,
    ) -> Run:
        """Handle an error for a chain run.

        Args:
            error: The error.
            inputs: The inputs for the chain.
            run_id: The run ID.
            **kwargs: Additional arguments.

        Returns:
            The run.
        

        中文翻译:
        处理链运行的错误。
        参数：
            错误：错误。
            输入：链的输入。
            run_id：运行 ID。
            **kwargs：附加参数。
        返回：
            奔跑。"""
        chain_run = self._errored_chain_run(
            error=error,
            run_id=run_id,
            inputs=inputs,
        )
        self._end_trace(chain_run)
        self._on_chain_error(chain_run)
        return chain_run

    def on_tool_start(
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
    ) -> Run:
        """Start a trace for a tool run.

        Args:
            serialized: The serialized tool.
            input_str: The input string.
            run_id: The run ID.
            tags: The tags for the run.
            parent_run_id: The parent run ID.
            metadata: The metadata for the run.
            name: The name of the run.
            inputs: The inputs for the tool.
            **kwargs: Additional arguments.

        Returns:
            The run.
        

        中文翻译:
        启动工具运行的跟踪。
        参数：
            序列化：序列化工具。
            input_str：输入字符串。
            run_id：运行 ID。
            标签：运行的标签。
            Parent_run_id：父运行 ID。
            元数据：运行的元数据。
            名称：运行的名称。
            输入：工具的输入。
            **kwargs：附加参数。
        返回：
            奔跑。"""
        tool_run = self._create_tool_run(
            serialized=serialized,
            input_str=input_str,
            run_id=run_id,
            tags=tags,
            parent_run_id=parent_run_id,
            metadata=metadata,
            name=name,
            inputs=inputs,
            **kwargs,
        )
        self._start_trace(tool_run)
        self._on_tool_start(tool_run)
        return tool_run

    @override
    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> Run:
        """End a trace for a tool run.

        Args:
            output: The output for the tool.
            run_id: The run ID.
            **kwargs: Additional arguments.

        Returns:
            The run.
        

        中文翻译:
        结束工具运行的跟踪。
        参数：
            输出：工具的输出。
            run_id：运行 ID。
            **kwargs：附加参数。
        返回：
            奔跑。"""
        tool_run = self._complete_tool_run(
            output=output,
            run_id=run_id,
        )
        self._end_trace(tool_run)
        self._on_tool_end(tool_run)
        return tool_run

    @override
    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> Run:
        """Handle an error for a tool run.

        Args:
            error: The error.
            run_id: The run ID.
            **kwargs: Additional arguments.

        Returns:
            The run.
        

        中文翻译:
        处理工具运行的错误。
        参数：
            错误：错误。
            run_id：运行 ID。
            **kwargs：附加参数。
        返回：
            奔跑。"""
        tool_run = self._errored_tool_run(
            error=error,
            run_id=run_id,
        )
        self._end_trace(tool_run)
        self._on_tool_error(tool_run)
        return tool_run

    def on_retriever_start(
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
    ) -> Run:
        """Run when the Retriever starts running.

        Args:
            serialized: The serialized retriever.
            query: The query.
            run_id: The run ID.
            parent_run_id: The parent run ID.
            tags: The tags for the run.
            metadata: The metadata for the run.
            name: The name of the run.
            **kwargs: Additional arguments.

        Returns:
            The run.
        

        中文翻译:
        当检索器开始运行时运行。
        参数：
            序列化：序列化的检索器。
            查询：查询。
            run_id：运行 ID。
            Parent_run_id：父运行 ID。
            标签：运行的标签。
            元数据：运行的元数据。
            名称：运行的名称。
            **kwargs：附加参数。
        返回：
            奔跑。"""
        retrieval_run = self._create_retrieval_run(
            serialized=serialized,
            query=query,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            name=name,
            **kwargs,
        )
        self._start_trace(retrieval_run)
        self._on_retriever_start(retrieval_run)
        return retrieval_run

    @override
    def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> Run:
        """Run when Retriever errors.

        Args:
            error: The error.
            run_id: The run ID.
            **kwargs: Additional arguments.

        Returns:
            The run.
        

        中文翻译:
        当检索器出错时运行。
        参数：
            错误：错误。
            run_id：运行 ID。
            **kwargs：附加参数。
        返回：
            奔跑。"""
        retrieval_run = self._errored_retrieval_run(
            error=error,
            run_id=run_id,
        )
        self._end_trace(retrieval_run)
        self._on_retriever_error(retrieval_run)
        return retrieval_run

    @override
    def on_retriever_end(
        self, documents: Sequence[Document], *, run_id: UUID, **kwargs: Any
    ) -> Run:
        """Run when the Retriever ends running.

        Args:
            documents: The documents.
            run_id: The run ID.
            **kwargs: Additional arguments.

        Returns:
            The run.
        

        中文翻译:
        当猎犬结束运行时运行。
        参数：
            文件：文件。
            run_id：运行 ID。
            **kwargs：附加参数。
        返回：
            奔跑。"""
        retrieval_run = self._complete_retrieval_run(
            documents=documents,
            run_id=run_id,
        )
        self._end_trace(retrieval_run)
        self._on_retriever_end(retrieval_run)
        return retrieval_run

    def __deepcopy__(self, memo: dict) -> BaseTracer:
        """Return self.

        中文翻译:
        回归自我。"""
        return self

    def __copy__(self) -> BaseTracer:
        """Return self.

        中文翻译:
        回归自我。"""
        return self


class AsyncBaseTracer(_TracerCore, AsyncCallbackHandler, ABC):
    """Async Base interface for tracers.

    中文翻译:
    跟踪器的异步基本接口。"""

    @abstractmethod
    @override
    async def _persist_run(self, run: Run) -> None:
        """Persist a run.

        中文翻译:
        坚持跑步。"""

    @override
    async def _start_trace(self, run: Run) -> None:
        """Start a trace for a run.

        Starting a trace will run concurrently with each _on_[run_type]_start method.
        No _on_[run_type]_start callback should depend on operations in _start_trace.
        

        中文翻译:
        开始追踪跑步。
        启动跟踪将与每个 _on_[run_type]_start 方法同时运行。
        _on_[run_type]_start 回调不应依赖于 _start_trace 中的操作。"""
        super()._start_trace(run)
        await self._on_run_create(run)

    @override
    async def _end_trace(self, run: Run) -> None:
        """End a trace for a run.

        Ending a trace will run concurrently with each _on_[run_type]_end method.
        No _on_[run_type]_end callback should depend on operations in _end_trace.
        

        中文翻译:
        结束跑步追踪。
        结束跟踪将与每个 _on_[run_type]_end 方法同时运行。
        _on_[run_type]_end 回调不应依赖于 _end_trace 中的操作。"""
        if not run.parent_run_id:
            await self._persist_run(run)
        self.run_map.pop(str(run.id))
        await self._on_run_update(run)

    @override
    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> Any:
        chat_model_run = self._create_chat_model_run(
            serialized=serialized,
            messages=messages,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            name=name,
            **kwargs,
        )
        tasks = [
            self._start_trace(chat_model_run),
            self._on_chat_model_start(chat_model_run),
        ]
        await asyncio.gather(*tasks)
        return chat_model_run

    @override
    async def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        llm_run = self._create_llm_run(
            serialized=serialized,
            prompts=prompts,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            **kwargs,
        )
        tasks = [self._start_trace(llm_run), self._on_llm_start(llm_run)]
        await asyncio.gather(*tasks)

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
        llm_run = self._llm_run_with_token_event(
            token=token,
            run_id=run_id,
            chunk=chunk,
            parent_run_id=parent_run_id,
        )
        await self._on_llm_new_token(llm_run, token, chunk)

    @override
    async def on_retry(
        self,
        retry_state: RetryCallState,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._llm_run_with_retry_event(
            retry_state=retry_state,
            run_id=run_id,
        )

    @override
    async def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        llm_run = self._complete_llm_run(
            response=response,
            run_id=run_id,
        )
        tasks = [self._on_llm_end(llm_run), self._end_trace(llm_run)]
        await asyncio.gather(*tasks)

    @override
    async def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        llm_run = self._errored_llm_run(
            error=error,
            run_id=run_id,
        )
        tasks = [self._on_llm_error(llm_run), self._end_trace(llm_run)]
        await asyncio.gather(*tasks)

    @override
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
        chain_run = self._create_chain_run(
            serialized=serialized,
            inputs=inputs,
            run_id=run_id,
            tags=tags,
            parent_run_id=parent_run_id,
            metadata=metadata,
            run_type=run_type,
            name=name,
            **kwargs,
        )
        tasks = [self._start_trace(chain_run), self._on_chain_start(chain_run)]
        await asyncio.gather(*tasks)

    @override
    async def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        chain_run = self._complete_chain_run(
            outputs=outputs,
            run_id=run_id,
            inputs=inputs,
        )
        tasks = [self._end_trace(chain_run), self._on_chain_end(chain_run)]
        await asyncio.gather(*tasks)

    @override
    async def on_chain_error(
        self,
        error: BaseException,
        *,
        inputs: dict[str, Any] | None = None,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        chain_run = self._errored_chain_run(
            error=error,
            inputs=inputs,
            run_id=run_id,
        )
        tasks = [self._end_trace(chain_run), self._on_chain_error(chain_run)]
        await asyncio.gather(*tasks)

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
        tool_run = self._create_tool_run(
            serialized=serialized,
            input_str=input_str,
            run_id=run_id,
            tags=tags,
            parent_run_id=parent_run_id,
            metadata=metadata,
            inputs=inputs,
            **kwargs,
        )
        tasks = [self._start_trace(tool_run), self._on_tool_start(tool_run)]
        await asyncio.gather(*tasks)

    @override
    async def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        tool_run = self._complete_tool_run(
            output=output,
            run_id=run_id,
        )
        tasks = [self._end_trace(tool_run), self._on_tool_end(tool_run)]
        await asyncio.gather(*tasks)

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
        tool_run = self._errored_tool_run(
            error=error,
            run_id=run_id,
        )
        tasks = [self._end_trace(tool_run), self._on_tool_error(tool_run)]
        await asyncio.gather(*tasks)

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
        retriever_run = self._create_retrieval_run(
            serialized=serialized,
            query=query,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            name=name,
        )
        tasks = [
            self._start_trace(retriever_run),
            self._on_retriever_start(retriever_run),
        ]
        await asyncio.gather(*tasks)

    @override
    async def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        retrieval_run = self._errored_retrieval_run(
            error=error,
            run_id=run_id,
        )
        tasks = [
            self._end_trace(retrieval_run),
            self._on_retriever_error(retrieval_run),
        ]
        await asyncio.gather(*tasks)

    @override
    async def on_retriever_end(
        self,
        documents: Sequence[Document],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        retrieval_run = self._complete_retrieval_run(
            documents=documents,
            run_id=run_id,
        )
        tasks = [self._end_trace(retrieval_run), self._on_retriever_end(retrieval_run)]
        await asyncio.gather(*tasks)

    async def _on_run_create(self, run: Run) -> None:
        """Process a run upon creation.

        中文翻译:
        创建时处理运行。"""

    async def _on_run_update(self, run: Run) -> None:
        """Process a run upon update.

        中文翻译:
        更新时处理运行。"""

    async def _on_llm_start(self, run: Run) -> None:
        """Process the LLM Run upon start.

        中文翻译:
        启动时处理 LLM 运行。"""

    async def _on_llm_end(self, run: Run) -> None:
        """Process the LLM Run.

        中文翻译:
        处理 LLM 运行。"""

    async def _on_llm_error(self, run: Run) -> None:
        """Process the LLM Run upon error.

        中文翻译:
        出现错误时处理 LLM 运行。"""

    async def _on_llm_new_token(
        self,
        run: Run,
        token: str,
        chunk: GenerationChunk | ChatGenerationChunk | None,
    ) -> None:
        """Process new LLM token.

        中文翻译:
        处理新的 LLM 令牌。"""

    async def _on_chain_start(self, run: Run) -> None:
        """Process the Chain Run upon start.

        中文翻译:
        启动时处理 Chain Run。"""

    async def _on_chain_end(self, run: Run) -> None:
        """Process the Chain Run.

        中文翻译:
        处理链运行。"""

    async def _on_chain_error(self, run: Run) -> None:
        """Process the Chain Run upon error.

        中文翻译:
        处理出错时的 Chain Run。"""

    async def _on_tool_start(self, run: Run) -> None:
        """Process the Tool Run upon start.

        中文翻译:
        启动时处理工具运行。"""

    async def _on_tool_end(self, run: Run) -> None:
        """Process the Tool Run.

        中文翻译:
        处理工具运行。"""

    async def _on_tool_error(self, run: Run) -> None:
        """Process the Tool Run upon error.

        中文翻译:
        处理出现错误时的工具运行。"""

    async def _on_chat_model_start(self, run: Run) -> None:
        """Process the Chat Model Run upon start.

        中文翻译:
        启动时处理聊天模型运行。"""

    async def _on_retriever_start(self, run: Run) -> None:
        """Process the Retriever Run upon start.

        中文翻译:
        启动时处理 Retriever Run。"""

    async def _on_retriever_end(self, run: Run) -> None:
        """Process the Retriever Run.

        中文翻译:
        处理检索器运行。"""

    async def _on_retriever_error(self, run: Run) -> None:
        """Process the Retriever Run upon error.

        中文翻译:
        处理错误时的检索器运行。"""
