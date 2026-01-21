"""A tracer that collects all nested runs in a list.

中文翻译:
收集列表中所有嵌套运行的跟踪器。"""

from typing import Any
from uuid import UUID

from langchain_core.tracers._compat import run_copy
from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.schemas import Run


class RunCollectorCallbackHandler(BaseTracer):
    """Tracer that collects all nested runs in a list.

    This tracer is useful for inspection and evaluation purposes.
    

    中文翻译:
    收集列表中所有嵌套运行的跟踪器。
    该示踪剂可用于检查和评估目的。"""

    name: str = "run-collector_callback_handler"

    def __init__(self, example_id: UUID | str | None = None, **kwargs: Any) -> None:
        """Initialize the RunCollectorCallbackHandler.

        Args:
            example_id: The ID of the example being traced. (default: None).
                It can be either a UUID or a string.
            **kwargs: Additional keyword arguments.
        

        中文翻译:
        初始化 RunCollectorCallbackHandler。
        参数：
            example_id：正在跟踪的示例的 ID。 （默认值：无）。
                它可以是 UUID 或字符串。
            **kwargs：附加关键字参数。"""
        super().__init__(**kwargs)
        self.example_id = (
            UUID(example_id) if isinstance(example_id, str) else example_id
        )
        self.traced_runs: list[Run] = []

    def _persist_run(self, run: Run) -> None:
        """Persist a run by adding it to the traced_runs list.

        Args:
            run: The run to be persisted.
        

        中文翻译:
        通过将运行添加到 traced_runs 列表来保留运行。
        参数：
            run：要持久化的运行。"""
        run_ = run_copy(run)
        run_.reference_example_id = self.example_id
        self.traced_runs.append(run_)
