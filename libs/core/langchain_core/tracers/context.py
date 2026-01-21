"""Context management for tracers.

中文翻译:
跟踪器的上下文管理。"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    cast,
)
from uuid import UUID

from langsmith import run_helpers as ls_rh
from langsmith import utils as ls_utils

from langchain_core.tracers.langchain import LangChainTracer
from langchain_core.tracers.run_collector import RunCollectorCallbackHandler

if TYPE_CHECKING:
    from collections.abc import Generator

    from langsmith import Client as LangSmithClient

    from langchain_core.callbacks.base import BaseCallbackHandler, Callbacks
    from langchain_core.callbacks.manager import AsyncCallbackManager, CallbackManager

# for backwards partial compatibility if this is imported by users but unused
# 中文: 如果用户导入但未使用，则用于向后部分兼容
tracing_callback_var: Any = None
tracing_v2_callback_var: ContextVar[LangChainTracer | None] = ContextVar(
    "tracing_callback_v2", default=None
)
run_collector_var: ContextVar[RunCollectorCallbackHandler | None] = ContextVar(
    "run_collector", default=None
)


@contextmanager
def tracing_v2_enabled(
    project_name: str | None = None,
    *,
    example_id: str | UUID | None = None,
    tags: list[str] | None = None,
    client: LangSmithClient | None = None,
) -> Generator[LangChainTracer, None, None]:
    """Instruct LangChain to log all runs in context to LangSmith.

    Args:
        project_name: The name of the project. Defaults to `'default'`.
        example_id: The ID of the example.
        tags: The tags to add to the run.
        client: The client of the langsmith.

    Yields:
        The LangChain tracer.

    Example:
        >>> with tracing_v2_enabled():
        ...     # LangChain code will automatically be traced

        You can use this to fetch the LangSmith run URL:

        >>> with tracing_v2_enabled() as cb:
        ...     chain.invoke("foo")
        ...     run_url = cb.get_run_url()
    

    中文翻译:
    指示 LangChain 将上下文中的所有运行记录到 LangSmith。
    参数：
        项目名称：项目的名称。默认为“默认”。
        example_id：示例的 ID。
        标签：要添加到运行的标签。
        客户：郎匠的客户。
    产量：
        LangChain 追踪器。
    示例：
        您可以使用它来获取 LangSmith 运行 URL："""
    if isinstance(example_id, str):
        example_id = UUID(example_id)
    cb = LangChainTracer(
        example_id=example_id,
        project_name=project_name,
        tags=tags,
        client=client,
    )
    token = tracing_v2_callback_var.set(cb)
    try:
        yield cb
    finally:
        tracing_v2_callback_var.reset(token)


@contextmanager
def collect_runs() -> Generator[RunCollectorCallbackHandler, None, None]:
    """Collect all run traces in context.

    Yields:
        The run collector callback handler.

    Example:
        >>> with collect_runs() as runs_cb:
                chain.invoke("foo")
                run_id = runs_cb.traced_runs[0].id
    

    中文翻译:
    收集上下文中的所有运行跟踪。
    产量：
        运行收集器回调处理程序。
    示例：
                链.调用（“foo”）
                run_id = running_cb.traced_runs[0].id"""
    cb = RunCollectorCallbackHandler()
    token = run_collector_var.set(cb)
    try:
        yield cb
    finally:
        run_collector_var.reset(token)


def _get_trace_callbacks(
    project_name: str | None = None,
    example_id: str | UUID | None = None,
    callback_manager: CallbackManager | AsyncCallbackManager | None = None,
) -> Callbacks:
    if _tracing_v2_is_enabled():
        project_name_ = project_name or _get_tracer_project()
        tracer = tracing_v2_callback_var.get() or LangChainTracer(
            project_name=project_name_,
            example_id=example_id,
        )
        if callback_manager is None:
            cb = cast("Callbacks", [tracer])
        else:
            if not any(
                isinstance(handler, LangChainTracer)
                for handler in callback_manager.handlers
            ):
                callback_manager.add_handler(tracer)
                # If it already has a LangChainTracer, we don't need to add another one.
                # 中文: 如果已经有LangChainTracer，我们就不需要再添加一个。
                # this would likely mess up the trace hierarchy.
                # 中文: 这可能会扰乱跟踪层次结构。
            cb = callback_manager
    else:
        cb = None
    return cb


def _tracing_v2_is_enabled() -> bool | Literal["local"]:
    if tracing_v2_callback_var.get() is not None:
        return True
    return ls_utils.tracing_is_enabled()


def _get_tracer_project() -> str:
    tracing_context = ls_rh.get_tracing_context()
    run_tree = tracing_context["parent"]
    if run_tree is None and tracing_context["project_name"] is not None:
        return cast("str", tracing_context["project_name"])
    return getattr(
        run_tree,
        "session_name",
        getattr(
            # Note, if people are trying to nest @traceable functions and the
            # 中文: 请注意，如果人们尝试嵌套 @traceable 函数并且
            # tracing_v2_enabled context manager, this will likely mess up the
            # 中文: Tracing_v2_enabled 上下文管理器，这可能会弄乱
            # tree structure.
            # 中文: 树结构。
            tracing_v2_callback_var.get(),
            "project",
            # Have to set this to a string even though it always will return
            # 中文: 必须将其设置为字符串，即使它总是会返回
            # a string because `get_tracer_project` technically can return
            # 中文: 一个字符串，因为“get_tracer_project”在技术上可以返回
            # None, but only when a specific argument is supplied.
            # 中文: 无，但仅当提供特定参数时。
            # Therefore, this just tricks the mypy type checker
            # 中文: 因此，这只是欺骗了 mypy 类型检查器
            str(ls_utils.get_tracer_project()),
        ),
    )


_configure_hooks: list[
    tuple[
        ContextVar[BaseCallbackHandler | None],
        bool,
        type[BaseCallbackHandler] | None,
        str | None,
    ]
] = []


def register_configure_hook(
    context_var: ContextVar[Any | None],
    inheritable: bool,  # noqa: FBT001
    handle_class: type[BaseCallbackHandler] | None = None,
    env_var: str | None = None,
) -> None:
    """Register a configure hook.

    Args:
        context_var: The context variable.
        inheritable: Whether the context variable is inheritable.
        handle_class: The callback handler class.
        env_var: The environment variable.

    Raises:
        ValueError: If env_var is set, handle_class must also be set to a non-None
            value.
    

    中文翻译:
    注册一个配置钩子。
    参数：
        context_var：上下文变量。
        可继承：上下文变量是否可继承。
        handle_class：回调处理程序类。
        env_var：环境变量。
    加薪：
        ValueError：如果设置了 env_var，handle_class 也必须设置为非 None
            值。"""
    if env_var is not None and handle_class is None:
        msg = "If env_var is set, handle_class must also be set to a non-None value."
        raise ValueError(msg)

    _configure_hooks.append(
        (
            # the typings of ContextVar do not have the generic arg set as covariant
            # 中文: ContextVar 的类型没有将通用参数设置为协变
            # so we have to cast it
            # 中文: 所以我们必须投射它
            cast("ContextVar[BaseCallbackHandler | None]", context_var),
            inheritable,
            handle_class,
            env_var,
        )
    )


register_configure_hook(run_collector_var, inheritable=False)
