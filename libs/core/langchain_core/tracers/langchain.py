"""A Tracer implementation that records to LangChain endpoint.

中文翻译:
记录到 LangChain 端点的 Tracer 实现。"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from uuid import UUID

from langsmith import Client, get_tracing_context
from langsmith import run_trees as rt
from langsmith import utils as ls_utils
from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)
from typing_extensions import override

from langchain_core.env import get_runtime_environment
from langchain_core.load import dumpd
from langchain_core.messages.ai import UsageMetadata, add_usage
from langchain_core.tracers._compat import run_construct, run_to_dict
from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.schemas import Run

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage
    from langchain_core.outputs import ChatGenerationChunk, GenerationChunk

logger = logging.getLogger(__name__)
_LOGGED = set()
_EXECUTOR: ThreadPoolExecutor | None = None


def log_error_once(method: str, exception: Exception) -> None:
    """Log an error once.

    Args:
        method: The method that raised the exception.
        exception: The exception that was raised.
    

    中文翻译:
    记录一次错误。
    参数：
        method：引发异常的方法。
        异常：引发的异常。"""
    if (method, type(exception)) in _LOGGED:
        return
    _LOGGED.add((method, type(exception)))
    logger.error(exception)


def wait_for_all_tracers() -> None:
    """Wait for all tracers to finish.

    中文翻译:
    等待所有跟踪器完成。"""
    if rt._CLIENT is not None:  # noqa: SLF001
        rt._CLIENT.flush()  # noqa: SLF001


def get_client() -> Client:
    """Get the client.

    Returns:
        The LangSmith client.
    

    中文翻译:
    得到客户。
    返回：
        朗史密斯客户端。"""
    return rt.get_cached_client()


def _get_executor() -> ThreadPoolExecutor:
    """Get the executor.

    中文翻译:
    找执行者。"""
    global _EXECUTOR  # noqa: PLW0603
    if _EXECUTOR is None:
        _EXECUTOR = ThreadPoolExecutor()
    return _EXECUTOR


def _get_usage_metadata_from_generations(
    generations: list[list[dict[str, Any]]],
) -> UsageMetadata | None:
    """Extract and aggregate `usage_metadata` from generations.

    Iterates through generations to find and aggregate all `usage_metadata` found in
    messages. This is typically present in chat model outputs.

    Args:
        generations: List of generation batches, where each batch is a list
            of generation dicts that may contain a `'message'` key with
            `'usage_metadata'`.

    Returns:
        The aggregated `usage_metadata` dict if found, otherwise `None`.
    

    中文翻译:
    从各代中提取并聚合“usage_metadata”。
    迭代几代以查找并聚合在中找到的所有“usage_metadata”
    消息。这通常出现在聊天模型输出中。
    参数：
        Generation：生成批次的列表，其中每个批次是一个列表
            可能包含“message”键的生成字典
            `'usage_metadata'`。
    返回：
        如果找到，则聚合的“usage_metadata”字典，否则为“None”。"""
    output: UsageMetadata | None = None
    for generation_batch in generations:
        for generation in generation_batch:
            if isinstance(generation, dict) and "message" in generation:
                message = generation["message"]
                if isinstance(message, dict) and "usage_metadata" in message:
                    output = add_usage(output, message["usage_metadata"])
    return output


class LangChainTracer(BaseTracer):
    """Implementation of the SharedTracer that POSTS to the LangChain endpoint.

    中文翻译:
    实现 POSTS 到 LangChain 端点的 SharedTracer。"""

    run_inline = True

    def __init__(
        self,
        example_id: UUID | str | None = None,
        project_name: str | None = None,
        client: Client | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the LangChain tracer.

        Args:
            example_id: The example ID.
            project_name: The project name. Defaults to the tracer project.
            client: The client. Defaults to the global client.
            tags: The tags. Defaults to an empty list.
            **kwargs: Additional keyword arguments.
        

        中文翻译:
        初始化LangChain追踪器。
        参数：
            example_id：示例 ID。
            项目名称：项目名称。默认为跟踪器项目。
            客户：客户。默认为全局客户端。
            标签：标签。默认为空列表。
            **kwargs：附加关键字参数。"""
        super().__init__(**kwargs)
        self.example_id = (
            UUID(example_id) if isinstance(example_id, str) else example_id
        )
        self.project_name = project_name or ls_utils.get_tracer_project()
        self.client = client or get_client()
        self.tags = tags or []
        self.latest_run: Run | None = None
        self.run_has_token_event_map: dict[str, bool] = {}

    def _start_trace(self, run: Run) -> None:
        if self.project_name:
            run.session_name = self.project_name
        if self.tags is not None:
            if run.tags:
                run.tags = sorted(set(run.tags + self.tags))
            else:
                run.tags = self.tags.copy()

        super()._start_trace(run)
        if run.ls_client is None:
            run.ls_client = self.client
        if get_tracing_context().get("enabled") is False:
            run.extra["__disabled"] = True

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
            messages: The messages.
            run_id: The run ID.
            tags: The tags.
            parent_run_id: The parent run ID.
            metadata: The metadata.
            name: The name.
            **kwargs: Additional keyword arguments.

        Returns:
            The run.
        

        中文翻译:
        启动 LLM 运行的跟踪。
        参数：
            序列化：序列化模型。
            消息：消息。
            run_id：运行 ID。
            标签：标签。
            Parent_run_id：父运行 ID。
            元数据：元数据。
            姓名：姓名。
            **kwargs：附加关键字参数。
        返回：
            奔跑。"""
        start_time = datetime.now(timezone.utc)
        if metadata:
            kwargs.update({"metadata": metadata})
        chat_model_run = Run(
            id=run_id,
            parent_run_id=parent_run_id,
            serialized=serialized,
            inputs={"messages": [[dumpd(msg) for msg in batch] for batch in messages]},
            extra=kwargs,
            events=[{"name": "start", "time": start_time}],
            start_time=start_time,
            run_type="llm",
            tags=tags,
            name=name,
        )
        self._start_trace(chat_model_run)
        self._on_chat_model_start(chat_model_run)
        return chat_model_run

    def _persist_run(self, run: Run) -> None:
        # We want to free up more memory by avoiding keeping a reference to the
        # 中文: 我们希望通过避免保留对
        # whole nested run tree.
        # 中文: 整个嵌套运行树。
        run_data = run_to_dict(run, exclude={"child_runs", "inputs", "outputs"})
        self.latest_run = run_construct(
            **run_data,
            inputs=run.inputs,
            outputs=run.outputs,
        )

    def get_run_url(self) -> str:
        """Get the LangSmith root run URL.

        Returns:
            The LangSmith root run URL.

        Raises:
            ValueError: If no traced run is found.
            ValueError: If the run URL cannot be found.
        

        中文翻译:
        获取 LangSmith 根运行 URL。
        返回：
            LangSmith 根运行 URL。
        加薪：
            ValueError：如果未找到跟踪的运行。
            ValueError：如果找不到运行 URL。"""
        if not self.latest_run:
            msg = "No traced run found."
            raise ValueError(msg)
        # If this is the first run in a project, the project may not yet be created.
        # 中文: 如果这是项目中的第一次运行，则该项目可能尚未创建。
        # This method is only really useful for debugging flows, so we will assume
        # 中文: 此方法仅对调试流程真正有用，因此我们假设
        # there is some tolerace for latency.
        # 中文: 对延迟有一定的容忍度。
        for attempt in Retrying(
            stop=stop_after_attempt(5),
            wait=wait_exponential_jitter(),
            retry=retry_if_exception_type(ls_utils.LangSmithError),
        ):
            with attempt:
                return self.client.get_run_url(
                    run=self.latest_run, project_name=self.project_name
                )
        msg = "Failed to get run URL."
        raise ValueError(msg)

    def _get_tags(self, run: Run) -> list[str]:
        """Get combined tags for a run.

        中文翻译:
        获取跑步的组合标签。"""
        tags = set(run.tags or [])
        tags.update(self.tags or [])
        return list(tags)

    def _persist_run_single(self, run: Run) -> None:
        """Persist a run.

        中文翻译:
        坚持跑步。"""
        if run.extra.get("__disabled"):
            return
        try:
            run.extra["runtime"] = get_runtime_environment()
            run.tags = self._get_tags(run)
            if run.ls_client is not self.client:
                run.ls_client = self.client
            run.post()
        except Exception as e:
            # Errors are swallowed by the thread executor so we need to log them here
            # 中文: 错误被线程执行器吞掉，所以我们需要在这里记录它们
            log_error_once("post", e)
            raise

    @staticmethod
    def _update_run_single(run: Run) -> None:
        """Update a run.

        中文翻译:
        更新一次运行。"""
        if run.extra.get("__disabled"):
            return
        try:
            run.patch(exclude_inputs=run.extra.get("inputs_is_truthy", False))
        except Exception as e:
            # Errors are swallowed by the thread executor so we need to log them here
            # 中文: 错误被线程执行器吞掉，所以我们需要在这里记录它们
            log_error_once("patch", e)
            raise

    def _on_llm_start(self, run: Run) -> None:
        """Persist an LLM run.

        中文翻译:
        坚持 LLM 运行。"""
        if run.parent_run_id is None:
            run.reference_example_id = self.example_id
        self._persist_run_single(run)

    @override
    def _llm_run_with_token_event(
        self,
        token: str,
        run_id: UUID,
        chunk: GenerationChunk | ChatGenerationChunk | None = None,
        parent_run_id: UUID | None = None,
    ) -> Run:
        run_id_str = str(run_id)
        if run_id_str not in self.run_has_token_event_map:
            self.run_has_token_event_map[run_id_str] = True
        else:
            return self._get_run(run_id, run_type={"llm", "chat_model"})
        return super()._llm_run_with_token_event(
            # Drop the chunk; we don't need to save it
            # 中文: 放下大块；我们不需要保存它
            token,
            run_id,
            chunk=None,
            parent_run_id=parent_run_id,
        )

    def _on_chat_model_start(self, run: Run) -> None:
        """Persist an LLM run.

        中文翻译:
        坚持 LLM 运行。"""
        if run.parent_run_id is None:
            run.reference_example_id = self.example_id
        self._persist_run_single(run)

    def _on_llm_end(self, run: Run) -> None:
        """Process the LLM Run.

        中文翻译:
        处理 LLM 运行。"""
        # Extract usage_metadata from outputs and store in extra.metadata
        # 中文: 从输出中提取 use_metadata 并存储在 extra.metadata 中
        if run.outputs and "generations" in run.outputs:
            usage_metadata = _get_usage_metadata_from_generations(
                run.outputs["generations"]
            )
            if usage_metadata is not None:
                if "metadata" not in run.extra:
                    run.extra["metadata"] = {}
                run.extra["metadata"]["usage_metadata"] = usage_metadata
        self._update_run_single(run)

    def _on_llm_error(self, run: Run) -> None:
        """Process the LLM Run upon error.

        中文翻译:
        出现错误时处理 LLM 运行。"""
        self._update_run_single(run)

    def _on_chain_start(self, run: Run) -> None:
        """Process the Chain Run upon start.

        中文翻译:
        启动时处理 Chain Run。"""
        if run.parent_run_id is None:
            run.reference_example_id = self.example_id
        # Skip persisting if inputs are deferred (e.g., iterator/generator inputs).
        # 中文: 如果输入被延迟（例如，迭代器/生成器输入），则跳过持久化。
        # The run will be posted when _on_chain_end is called with realized inputs.
        # 中文: 当使用已实现的输入调用 _on_chain_end 时，将发布运行。
        if not run.extra.get("defers_inputs"):
            self._persist_run_single(run)

    def _on_chain_end(self, run: Run) -> None:
        """Process the Chain Run.

        中文翻译:
        处理链运行。"""
        # If inputs were deferred, persist (POST) the run now that inputs are realized.
        # 中文: 如果输入被推迟，则在实现输入后继续运行 (POST)。
        # Otherwise, update (PATCH) the existing run.
        # 中文: 否则，更新（修补）现有运行。
        if run.extra.get("defers_inputs"):
            self._persist_run_single(run)
        else:
            self._update_run_single(run)

    def _on_chain_error(self, run: Run) -> None:
        """Process the Chain Run upon error.

        中文翻译:
        处理出错时的 Chain Run。"""
        # If inputs were deferred, persist (POST) the run now that inputs are realized.
        # 中文: 如果输入被推迟，则在实现输入后继续运行 (POST)。
        # Otherwise, update (PATCH) the existing run.
        # 中文: 否则，更新（修补）现有运行。
        if run.extra.get("defers_inputs"):
            self._persist_run_single(run)
        else:
            self._update_run_single(run)

    def _on_tool_start(self, run: Run) -> None:
        """Process the Tool Run upon start.

        中文翻译:
        启动时处理工具运行。"""
        if run.parent_run_id is None:
            run.reference_example_id = self.example_id
        self._persist_run_single(run)

    def _on_tool_end(self, run: Run) -> None:
        """Process the Tool Run.

        中文翻译:
        处理工具运行。"""
        self._update_run_single(run)

    def _on_tool_error(self, run: Run) -> None:
        """Process the Tool Run upon error.

        中文翻译:
        处理出现错误时的工具运行。"""
        self._update_run_single(run)

    def _on_retriever_start(self, run: Run) -> None:
        """Process the Retriever Run upon start.

        中文翻译:
        启动时处理 Retriever Run。"""
        if run.parent_run_id is None:
            run.reference_example_id = self.example_id
        self._persist_run_single(run)

    def _on_retriever_end(self, run: Run) -> None:
        """Process the Retriever Run.

        中文翻译:
        处理检索器运行。"""
        self._update_run_single(run)

    def _on_retriever_error(self, run: Run) -> None:
        """Process the Retriever Run upon error.

        中文翻译:
        处理错误时的检索器运行。"""
        self._update_run_single(run)

    def wait_for_futures(self) -> None:
        """Wait for the given futures to complete.

        中文翻译:
        等待给定的 future 完成。"""
        if self.client is not None:
            self.client.flush()
