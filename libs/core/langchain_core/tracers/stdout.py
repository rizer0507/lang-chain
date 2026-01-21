"""Tracers that print to the console.

中文翻译:
打印到控制台的跟踪器。"""

import json
from collections.abc import Callable
from typing import Any

from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.schemas import Run
from langchain_core.utils.input import get_bolded_text, get_colored_text

MILLISECONDS_IN_SECOND = 1000


def try_json_stringify(obj: Any, fallback: str) -> str:
    """Try to stringify an object to JSON.

    Args:
        obj: Object to stringify.
        fallback: Fallback string to return if the object cannot be stringified.

    Returns:
        A JSON string if the object can be stringified, otherwise the fallback string.
    

    中文翻译:
    尝试将对象字符串化为 JSON。
    参数：
        obj：要字符串化的对象。
        Fallback：如果对象无法字符串化，则返回后备字符串。
    返回：
        如果对象可以字符串化，则为 JSON 字符串，否则为后备字符串。"""
    try:
        return json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception:
        return fallback


def elapsed(run: Any) -> str:
    """Get the elapsed time of a run.

    Args:
        run: any object with a start_time and end_time attribute.

    Returns:
        A string with the elapsed time in seconds or
            milliseconds if time is less than a second.

    

    中文翻译:
    获取跑步的已用时间。
    参数：
        run：任何具有 start_time 和 end_time 属性的对象。
    返回：
        一个字符串，其中包含经过的时间（以秒为单位）或
            如果时间小于一秒，则为毫秒。"""
    elapsed_time = run.end_time - run.start_time
    seconds = elapsed_time.total_seconds()
    if seconds < 1:
        return f"{seconds * MILLISECONDS_IN_SECOND:.0f}ms"
    return f"{seconds:.2f}s"


class FunctionCallbackHandler(BaseTracer):
    """Tracer that calls a function with a single str parameter.

    中文翻译:
    使用单个 str 参数调用函数的跟踪器。"""

    name: str = "function_callback_handler"
    """The name of the tracer. This is used to identify the tracer in the logs.

    中文翻译:
    追踪器的名称。这用于识别日志中的跟踪器。"""

    def __init__(self, function: Callable[[str], None], **kwargs: Any) -> None:
        """Create a FunctionCallbackHandler.

        Args:
            function: The callback function to call.
        

        中文翻译:
        创建一个 FunctionCallbackHandler。
        参数：
            function：要调用的回调函数。"""
        super().__init__(**kwargs)
        self.function_callback = function

    def _persist_run(self, run: Run) -> None:
        pass

    def get_parents(self, run: Run) -> list[Run]:
        """Get the parents of a run.

        Args:
            run: The run to get the parents of.

        Returns:
            A list of parent runs.
        

        中文翻译:
        赶紧让家长跑起来吧。
        参数：
            跑：跑去找父母。
        返回：
            父运行列表。"""
        parents = []
        current_run = run
        while current_run.parent_run_id:
            parent = self.run_map.get(str(current_run.parent_run_id))
            if parent:
                parents.append(parent)
                current_run = parent
            else:
                break
        return parents

    def get_breadcrumbs(self, run: Run) -> str:
        """Get the breadcrumbs of a run.

        Args:
            run: The run to get the breadcrumbs of.

        Returns:
            A string with the breadcrumbs of the run.
        

        中文翻译:
        获取跑步的痕迹。
        参数：
            run：获取面包屑的运行。
        返回：
            带有运行的面包屑的字符串。"""
        parents = self.get_parents(run)[::-1]
        return " > ".join(
            f"{parent.run_type}:{parent.name}"
            for i, parent in enumerate([*parents, run])
        )

    # logging methods
    # 中文: 记录方法
    def _on_chain_start(self, run: Run) -> None:
        crumbs = self.get_breadcrumbs(run)
        run_type = run.run_type.capitalize()
        self.function_callback(
            f"{get_colored_text('[chain/start]', color='green')} "
            + get_bolded_text(f"[{crumbs}] Entering {run_type} run with input:\n")
            + f"{try_json_stringify(run.inputs, '[inputs]')}"
        )

    def _on_chain_end(self, run: Run) -> None:
        crumbs = self.get_breadcrumbs(run)
        run_type = run.run_type.capitalize()
        self.function_callback(
            f"{get_colored_text('[chain/end]', color='blue')} "
            + get_bolded_text(
                f"[{crumbs}] [{elapsed(run)}] Exiting {run_type} run with output:\n"
            )
            + f"{try_json_stringify(run.outputs, '[outputs]')}"
        )

    def _on_chain_error(self, run: Run) -> None:
        crumbs = self.get_breadcrumbs(run)
        run_type = run.run_type.capitalize()
        self.function_callback(
            f"{get_colored_text('[chain/error]', color='red')} "
            + get_bolded_text(
                f"[{crumbs}] [{elapsed(run)}] {run_type} run errored with error:\n"
            )
            + f"{try_json_stringify(run.error, '[error]')}"
        )

    def _on_llm_start(self, run: Run) -> None:
        crumbs = self.get_breadcrumbs(run)
        inputs = (
            {"prompts": [p.strip() for p in run.inputs["prompts"]]}
            if "prompts" in run.inputs
            else run.inputs
        )
        self.function_callback(
            f"{get_colored_text('[llm/start]', color='green')} "
            + get_bolded_text(f"[{crumbs}] Entering LLM run with input:\n")
            + f"{try_json_stringify(inputs, '[inputs]')}"
        )

    def _on_llm_end(self, run: Run) -> None:
        crumbs = self.get_breadcrumbs(run)
        self.function_callback(
            f"{get_colored_text('[llm/end]', color='blue')} "
            + get_bolded_text(
                f"[{crumbs}] [{elapsed(run)}] Exiting LLM run with output:\n"
            )
            + f"{try_json_stringify(run.outputs, '[response]')}"
        )

    def _on_llm_error(self, run: Run) -> None:
        crumbs = self.get_breadcrumbs(run)
        self.function_callback(
            f"{get_colored_text('[llm/error]', color='red')} "
            + get_bolded_text(
                f"[{crumbs}] [{elapsed(run)}] LLM run errored with error:\n"
            )
            + f"{try_json_stringify(run.error, '[error]')}"
        )

    def _on_tool_start(self, run: Run) -> None:
        crumbs = self.get_breadcrumbs(run)
        self.function_callback(
            f"{get_colored_text('[tool/start]', color='green')} "
            + get_bolded_text(f"[{crumbs}] Entering Tool run with input:\n")
            + f'"{run.inputs["input"].strip()}"'
        )

    def _on_tool_end(self, run: Run) -> None:
        crumbs = self.get_breadcrumbs(run)
        if run.outputs:
            self.function_callback(
                f"{get_colored_text('[tool/end]', color='blue')} "
                + get_bolded_text(
                    f"[{crumbs}] [{elapsed(run)}] Exiting Tool run with output:\n"
                )
                + f'"{str(run.outputs["output"]).strip()}"'
            )

    def _on_tool_error(self, run: Run) -> None:
        crumbs = self.get_breadcrumbs(run)
        self.function_callback(
            f"{get_colored_text('[tool/error]', color='red')} "
            + get_bolded_text(f"[{crumbs}] [{elapsed(run)}] ")
            + f"Tool run errored with error:\n"
            f"{run.error}"
        )


class ConsoleCallbackHandler(FunctionCallbackHandler):
    """Tracer that prints to the console.

    中文翻译:
    打印到控制台的跟踪器。"""

    name: str = "console_callback_handler"

    def __init__(self, **kwargs: Any) -> None:
        """Create a ConsoleCallbackHandler.

        中文翻译:
        创建一个 ConsoleCallbackHandler。"""
        super().__init__(function=print, **kwargs)
