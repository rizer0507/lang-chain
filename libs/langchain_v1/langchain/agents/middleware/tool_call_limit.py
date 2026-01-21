"""工具调用次数限制中间件模块。

本模块提供跟踪和限制工具调用次数的能力。

核心类:
--------
**ToolCallLimitMiddleware**: 工具调用限制中间件

功能特性:
---------
- 线程级限制：跨多次调用持久化
- 运行级限制：单次调用内限制
- 可限制特定工具
- 可配置超限行为

超限行为:
---------
- `continue`: 阻止超限工具，其他工具继续（默认）
- `error`: 抛出 ToolCallLimitExceededError
- `end`: 立即结束执行

使用示例:
---------
>>> from langchain.agents import create_agent
>>> from langchain.agents.middleware import ToolCallLimitMiddleware
>>>
>>> # 限制所有工具
>>> limiter = ToolCallLimitMiddleware(
...     thread_limit=20,
...     run_limit=10,
... )
>>>
>>> # 限制特定工具
>>> search_limiter = ToolCallLimitMiddleware(
...     tool_name="search",
...     thread_limit=5,
...     exit_behavior="error",
... )
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Generic, Literal

from langchain_core.messages import AIMessage, ToolCall, ToolMessage
from langgraph.channels.untracked_value import UntrackedValue
from langgraph.typing import ContextT
from typing_extensions import NotRequired, override

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    PrivateStateAttr,
    ResponseT,
    hook_config,
)

if TYPE_CHECKING:
    from langgraph.runtime import Runtime

ExitBehavior = Literal["continue", "error", "end"]
"""How to handle execution when tool call limits are exceeded.

- `'continue'`: Block exceeded tools with error messages, let other tools continue
    (default)
- `'error'`: Raise a `ToolCallLimitExceededError` exception
- `'end'`: Stop execution immediately, injecting a `ToolMessage` and an `AIMessage` for
    the single tool call that exceeded the limit. Raises `NotImplementedError` if there
    are other pending tool calls (due to parallel tool calling).

中文翻译:
当超出工具调用限制时如何处理执行。
- `'继续'`：阻止超出的工具并显示错误消息，让其他工具继续
    （默认）
- `'error'`：引发 `ToolCallLimitExceededError` 异常
- `'end'`：立即停止执行，注入一个`ToolMessage`和一个`AIMessage`
    超出限制的单个工具调用。如果存在则引发“NotImplementedError”
    是其他待处理的工具调用（由于并行工具调用）。
"""


class ToolCallLimitState(AgentState[ResponseT], Generic[ResponseT]):
    """State schema for `ToolCallLimitMiddleware`.

    Extends `AgentState` with tool call tracking fields.

    The count fields are dictionaries mapping tool names to execution counts. This
    allows multiple middleware instances to track different tools independently. The
    special key `'__all__'` is used for tracking all tool calls globally.
    

    中文翻译:
    `ToolCallLimitMiddleware` 的状态模式。
    使用工具调用跟踪字段扩展“AgentState”。
    计数字段是将工具名称映射到执行计数的字典。这个
    允许多个中间件实例独立跟踪不同的工具。的
    特殊键“__all__”用于全局跟踪所有工具调用。"""

    thread_tool_call_count: NotRequired[Annotated[dict[str, int], PrivateStateAttr]]
    run_tool_call_count: NotRequired[Annotated[dict[str, int], UntrackedValue, PrivateStateAttr]]


def _build_tool_message_content(tool_name: str | None) -> str:
    """Build the error message content for `ToolMessage` when limit is exceeded.

    This message is sent to the model, so it should not reference thread/run concepts
    that the model has no notion of.

    Args:
        tool_name: Tool name being limited (if specific tool), or `None` for all tools.

    Returns:
        A concise message instructing the model not to call the tool again.
    

    中文翻译:
    当超出限制时，构建“ToolMessage”的错误消息内容。
    该消息被发送到模型，因此它不应引用线程/运行概念
    模型没有这个概念。
    参数：
        tool_name：工具名称受到限制（如果是特定工具），或所有工具为“无”。
    返回：
        指示模型不要再次调用该工具的简洁消息。"""
    # Always instruct the model not to call again, regardless of which limit was hit
    # 中文: 始终指示模型不要再次调用，无论达到哪个限制
    if tool_name:
        return f"Tool call limit exceeded. Do not call '{tool_name}' again."
    return "Tool call limit exceeded. Do not make additional tool calls."


def _build_final_ai_message_content(
    thread_count: int,
    run_count: int,
    thread_limit: int | None,
    run_limit: int | None,
    tool_name: str | None,
) -> str:
    """Build the final AI message content for `'end'` behavior.

    This message is displayed to the user, so it should include detailed information
    about which limits were exceeded.

    Args:
        thread_count: Current thread tool call count.
        run_count: Current run tool call count.
        thread_limit: Thread tool call limit (if set).
        run_limit: Run tool call limit (if set).
        tool_name: Tool name being limited (if specific tool), or `None` for all tools.

    Returns:
        A formatted message describing which limits were exceeded.
    

    中文翻译:
    为“结束”行为构建最终的 AI 消息内容。
    此消息显示给用户，因此应包含详细信息
    关于超出了哪些限制。
    参数：
        thread_count：当前线程工具调用计数。
        run_count：当前运行工具调用次数。
        thread_limit：线程工具调用限制（如果设置）。
        run_limit：运行工具调用限制（如果设置）。
        tool_name：工具名称受到限制（如果是特定工具），或所有工具为“无”。
    返回：
        描述超出哪些限制的格式化消息。"""
    tool_desc = f"'{tool_name}' tool" if tool_name else "Tool"
    exceeded_limits = []

    if thread_limit is not None and thread_count > thread_limit:
        exceeded_limits.append(f"thread limit exceeded ({thread_count}/{thread_limit} calls)")
    if run_limit is not None and run_count > run_limit:
        exceeded_limits.append(f"run limit exceeded ({run_count}/{run_limit} calls)")

    limits_text = " and ".join(exceeded_limits)
    return f"{tool_desc} call limit reached: {limits_text}."


class ToolCallLimitExceededError(Exception):
    """Exception raised when tool call limits are exceeded.

    This exception is raised when the configured exit behavior is `'error'` and either
    the thread or run tool call limit has been exceeded.
    

    中文翻译:
    超出工具调用限制时引发异常。
    当配置的退出行为是“错误”并且
    已超出线程或运行工具调用限制。"""

    def __init__(
        self,
        thread_count: int,
        run_count: int,
        thread_limit: int | None,
        run_limit: int | None,
        tool_name: str | None = None,
    ) -> None:
        """Initialize the exception with call count information.

        Args:
            thread_count: Current thread tool call count.
            run_count: Current run tool call count.
            thread_limit: Thread tool call limit (if set).
            run_limit: Run tool call limit (if set).
            tool_name: Tool name being limited (if specific tool), or None for all tools.
        

        中文翻译:
        使用调用计数信息初始化异常。
        参数：
            thread_count：当前线程工具调用计数。
            run_count：当前运行工具调用次数。
            thread_limit：线程工具调用限制（如果设置）。
            run_limit：运行工具调用限制（如果设置）。
            tool_name：受限制的工具名称（如果是特定工具），或所有工具均无。"""
        self.thread_count = thread_count
        self.run_count = run_count
        self.thread_limit = thread_limit
        self.run_limit = run_limit
        self.tool_name = tool_name

        msg = _build_final_ai_message_content(
            thread_count, run_count, thread_limit, run_limit, tool_name
        )
        super().__init__(msg)


class ToolCallLimitMiddleware(
    AgentMiddleware[ToolCallLimitState[ResponseT], ContextT],
    Generic[ResponseT, ContextT],
):
    """Track tool call counts and enforces limits during agent execution.

    This middleware monitors the number of tool calls made and can terminate or
    restrict execution when limits are exceeded. It supports both thread-level
    (persistent across runs) and run-level (per invocation) call counting.

    Configuration:
        - `exit_behavior`: How to handle when limits are exceeded
            - `'continue'`: Block exceeded tools, let execution continue (default)
            - `'error'`: Raise an exception
            - `'end'`: Stop immediately with a `ToolMessage` + AI message for the single
                tool call that exceeded the limit (raises `NotImplementedError` if there
                are other pending tool calls (due to parallel tool calling).

    Examples:
        !!! example "Continue execution with blocked tools (default)"

            ```python
            from langchain.agents.middleware.tool_call_limit import ToolCallLimitMiddleware
            from langchain.agents import create_agent

            # Block exceeded tools but let other tools and model continue
            # 中文: 阻止超出的工具，但让其他工具和模型继续
            limiter = ToolCallLimitMiddleware(
                thread_limit=20,
                run_limit=10,
                exit_behavior="continue",  # default
            )

            agent = create_agent("openai:gpt-4o", middleware=[limiter])
            ```

        !!! example "Stop immediately when limit exceeded"

            ```python
            # End execution immediately with an AI message
            # 中文: 通过 AI 消息立即结束执行
            limiter = ToolCallLimitMiddleware(run_limit=5, exit_behavior="end")

            agent = create_agent("openai:gpt-4o", middleware=[limiter])
            ```

        !!! example "Raise exception on limit"

            ```python
            # Strict limit with exception handling
            # 中文: 严格限制与异常处理
            limiter = ToolCallLimitMiddleware(
                tool_name="search", thread_limit=5, exit_behavior="error"
            )

            agent = create_agent("openai:gpt-4o", middleware=[limiter])

            try:
                result = await agent.invoke({"messages": [HumanMessage("Task")]})
            except ToolCallLimitExceededError as e:
                print(f"Search limit exceeded: {e}")
            ```

    

    中文翻译:
    跟踪工具调用计数并在代理执行期间强制执行限制。
    该中间件监视进行的工具调用数量，并可以终止或
    当超出限制时限制执行。它支持线程级别
    （跨运行持续）和运行级别（每次调用）调用计数。
    配置：
        - `exit_behavior`：超出限制时如何处理
            - `'继续'`：阻止超出的工具，让执行继续（默认）
            - `'error'`：引发异常
            - `'end'`：使用`ToolMessage` + AI 消息立即停止
                超出限制的工具调用（如果存在则引发“NotImplementedError”
                是其他待处理的工具调用（由于并行工具调用）。
    示例：
        !!!示例“使用阻止的工具继续执行（默认）”
            ````蟒蛇
            从 langchain.agents.middleware.tool_call_limit 导入 ToolCallLimitMiddleware
            从 langchain.agents 导入 create_agent
            # 阻止超出的工具，但让其他工具和模型继续
            限制器 = ToolCallLimitMiddleware(
                线程限制=20，
                运行限制=10，
                exit_behavior="继续", # 默认
            ）
            代理 = create_agent("openai:gpt-4o", middleware=[限制器])
            ````
        !!!示例“超出限制时立即停止”
            ````蟒蛇
            # 立即结束执行并发出 AI 消息
            限制器 = ToolCallLimitMiddleware(run_limit=5, exit_behavior="end")
            代理 = create_agent("openai:gpt-4o", middleware=[限制器])
            ````
        !!!示例“引发限制异常”
            ````蟒蛇
            # 严格限制异常处理
            限制器 = ToolCallLimitMiddleware(
                tool_name =“搜索”，thread_limit = 5，exit_behavior =“错误”
            ）
            代理 = create_agent("openai:gpt-4o", middleware=[限制器])
            尝试：
                结果=等待agent.invoke({"messages": [HumanMessage("Task")]})
            除了 ToolCallLimitExceededError 为 e：
                print(f"超出搜索限制：{e}")
            ````"""

    state_schema = ToolCallLimitState  # type: ignore[assignment]

    def __init__(
        self,
        *,
        tool_name: str | None = None,
        thread_limit: int | None = None,
        run_limit: int | None = None,
        exit_behavior: ExitBehavior = "continue",
    ) -> None:
        """Initialize the tool call limit middleware.

        Args:
            tool_name: Name of the specific tool to limit. If `None`, limits apply
                to all tools.
            thread_limit: Maximum number of tool calls allowed per thread.
                `None` means no limit.
            run_limit: Maximum number of tool calls allowed per run.
                `None` means no limit.
            exit_behavior: How to handle when limits are exceeded.

                - `'continue'`: Block exceeded tools with error messages, let other
                    tools continue. Model decides when to end.
                - `'error'`: Raise a `ToolCallLimitExceededError` exception
                - `'end'`: Stop execution immediately with a `ToolMessage` + AI message
                    for the single tool call that exceeded the limit. Raises
                    `NotImplementedError` if there are multiple parallel tool
                    calls to other tools or multiple pending tool calls.

        Raises:
            ValueError: If both limits are `None`, if `exit_behavior` is invalid,
                or if `run_limit` exceeds `thread_limit`.
        

        中文翻译:
        初始化工具调用限制中间件。
        参数：
            tool_name：要限制的特定工具的名称。如果“无”，则有限制
                到所有工具。
            thread_limit：每个线程允许的最大工具调用次数。
                “无”意味着没有限制。
            run_limit：每次运行允许的最大工具调用次数。
                “无”意味着没有限制。
            exit_behavior：超出限制时如何处理。
                - `'继续'`：阻止带有错误消息的超出工具，让其他工具
                    工具继续。模型决定何时结束。
                - `'error'`：引发 `ToolCallLimitExceededError` 异常
                - `'end'`：通过 `ToolMessage` + AI 消息立即停止执行
                    对于超出限制的单个工具调用。提高
                    如果有多个并行工具，则为“NotImplementedError”
                    对其他工具的调用或多个待处理的工具调用。
        加薪：
            ValueError：如果两个限制都是“None”，如果“exit_behavior”无效，
                或者如果“run_limit”超过“thread_limit”。"""
        super().__init__()

        if thread_limit is None and run_limit is None:
            msg = "At least one limit must be specified (thread_limit or run_limit)"
            raise ValueError(msg)

        valid_behaviors = ("continue", "error", "end")
        if exit_behavior not in valid_behaviors:
            msg = f"Invalid exit_behavior: {exit_behavior!r}. Must be one of {valid_behaviors}"
            raise ValueError(msg)

        if thread_limit is not None and run_limit is not None and run_limit > thread_limit:
            msg = (
                f"run_limit ({run_limit}) cannot exceed thread_limit ({thread_limit}). "
                "The run limit should be less than or equal to the thread limit."
            )
            raise ValueError(msg)

        self.tool_name = tool_name
        self.thread_limit = thread_limit
        self.run_limit = run_limit
        self.exit_behavior = exit_behavior

    @property
    def name(self) -> str:
        """The name of the middleware instance.

        Includes the tool name if specified to allow multiple instances
        of this middleware with different tool names.
        

        中文翻译:
        中间件实例的名称。
        如果指定允许多个实例，则包括工具名称
        该中间件具有不同的工具名称。"""
        base_name = self.__class__.__name__
        if self.tool_name:
            return f"{base_name}[{self.tool_name}]"
        return base_name

    def _would_exceed_limit(self, thread_count: int, run_count: int) -> bool:
        """Check if incrementing the counts would exceed any configured limit.

        Args:
            thread_count: Current thread call count.
            run_count: Current run call count.

        Returns:
            True if either limit would be exceeded by one more call.
        

        中文翻译:
        检查增加计数是否会超出任何配置的限制。
        参数：
            thread_count：当前线程调用计数。
            run_count：当前运行调用计数。
        返回：
            如果再一次调用将超过任一限制，则为 true。"""
        return (self.thread_limit is not None and thread_count + 1 > self.thread_limit) or (
            self.run_limit is not None and run_count + 1 > self.run_limit
        )

    def _matches_tool_filter(self, tool_call: ToolCall) -> bool:
        """Check if a tool call matches this middleware's tool filter.

        Args:
            tool_call: The tool call to check.

        Returns:
            True if this middleware should track this tool call.
        

        中文翻译:
        检查工具调用是否与该中间件的工具过滤器匹配。
        参数：
            tool_call：要检查的工具调用。
        返回：
            如果此中间件应跟踪此工具调用，则为 true。"""
        return self.tool_name is None or tool_call["name"] == self.tool_name

    def _separate_tool_calls(
        self, tool_calls: list[ToolCall], thread_count: int, run_count: int
    ) -> tuple[list[ToolCall], list[ToolCall], int, int]:
        """Separate tool calls into allowed and blocked based on limits.

        Args:
            tool_calls: List of tool calls to evaluate.
            thread_count: Current thread call count.
            run_count: Current run call count.

        Returns:
            Tuple of `(allowed_calls, blocked_calls, final_thread_count,
                final_run_count)`.
        

        中文翻译:
        根据限制将工具调用分为允许和阻止。
        参数：
            tool_calls：要评估的工具调用列表。
            thread_count：当前线程调用计数。
            run_count：当前运行调用计数。
        返回：
            `（allowed_calls、blocked_calls、final_thread_count、的元组）
                最终运行计数）`。"""
        allowed_calls: list[ToolCall] = []
        blocked_calls: list[ToolCall] = []
        temp_thread_count = thread_count
        temp_run_count = run_count

        for tool_call in tool_calls:
            if not self._matches_tool_filter(tool_call):
                continue

            if self._would_exceed_limit(temp_thread_count, temp_run_count):
                blocked_calls.append(tool_call)
            else:
                allowed_calls.append(tool_call)
                temp_thread_count += 1
                temp_run_count += 1

        return allowed_calls, blocked_calls, temp_thread_count, temp_run_count

    @hook_config(can_jump_to=["end"])
    @override
    def after_model(
        self,
        state: ToolCallLimitState[ResponseT],
        runtime: Runtime[ContextT],
    ) -> dict[str, Any] | None:
        """Increment tool call counts after a model call and check limits.

        Args:
            state: The current agent state.
            runtime: The langgraph runtime.

        Returns:
            State updates with incremented tool call counts. If limits are exceeded
                and exit_behavior is `'end'`, also includes a jump to end with a
                `ToolMessage` and AI message for the single exceeded tool call.

        Raises:
            ToolCallLimitExceededError: If limits are exceeded and `exit_behavior`
                is `'error'`.
            NotImplementedError: If limits are exceeded, `exit_behavior` is `'end'`,
                and there are multiple tool calls.
        

        中文翻译:
        在模型调用和检查限制后增加工具调用计数。
        参数：
            状态：当前代理状态。
            运行时：语言图运行时。
        返回：
            状态随着工具调用计数的增加而更新。如果超出限制
                exit_behavior 是 `'end'`，还包括跳转到结尾
                单个超出的工具调用的“ToolMessage”和 AI 消息。
        加薪：
            ToolCallLimitExceededError：如果超出限制并且“exit_behavior”
                是“错误”。
            NotImplementedError：如果超出限制，`exit_behavior` 为 `'end'`，
                并且有多个工具调用。"""
        # Get the last AIMessage to check for tool calls
        # 中文: 获取最后一条AIMessage来检查工具调用
        messages = state.get("messages", [])
        if not messages:
            return None

        # Find the last AIMessage
        # 中文: 查找最后一条 AIMessage
        last_ai_message = None
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                last_ai_message = message
                break

        if not last_ai_message or not last_ai_message.tool_calls:
            return None

        # Get the count key for this middleware instance
        # 中文: 获取此中间件实例的计数键
        count_key = self.tool_name or "__all__"

        # Get current counts
        # 中文: 获取当前计数
        thread_counts = state.get("thread_tool_call_count", {}).copy()
        run_counts = state.get("run_tool_call_count", {}).copy()
        current_thread_count = thread_counts.get(count_key, 0)
        current_run_count = run_counts.get(count_key, 0)

        # Separate tool calls into allowed and blocked
        # 中文: 将工具调用分为允许和阻止
        allowed_calls, blocked_calls, new_thread_count, new_run_count = self._separate_tool_calls(
            last_ai_message.tool_calls, current_thread_count, current_run_count
        )

        # Update counts to include only allowed calls for thread count
        # 中文: 更新计数以仅包括允许的线程计数调用
        # (blocked calls don't count towards thread-level tracking)
        # 中文: （被阻止的调用不计入线程级跟踪）
        # But run count includes blocked calls since they were attempted in this run
        # 中文: 但运行计数包括自本次运行尝试以来被阻止的调用
        thread_counts[count_key] = new_thread_count
        run_counts[count_key] = new_run_count + len(blocked_calls)

        # If no tool calls are blocked, just update counts
        # 中文: 如果没有工具调用被阻止，则仅更新计数
        if not blocked_calls:
            if allowed_calls:
                return {
                    "thread_tool_call_count": thread_counts,
                    "run_tool_call_count": run_counts,
                }
            return None

        # Get final counts for building messages
        # 中文: 获取构建消息的最终计数
        final_thread_count = thread_counts[count_key]
        final_run_count = run_counts[count_key]

        # Handle different exit behaviors
        # 中文: 处理不同的退出行为
        if self.exit_behavior == "error":
            # Use hypothetical thread count to show which limit was exceeded
            # 中文: 使用假设的线程计数来显示超出了哪个限制
            hypothetical_thread_count = final_thread_count + len(blocked_calls)
            raise ToolCallLimitExceededError(
                thread_count=hypothetical_thread_count,
                run_count=final_run_count,
                thread_limit=self.thread_limit,
                run_limit=self.run_limit,
                tool_name=self.tool_name,
            )

        # Build tool message content (sent to model - no thread/run details)
        # 中文: 构建工具消息内容（发送到模型 - 无线程/运行详细信息）
        tool_msg_content = _build_tool_message_content(self.tool_name)

        # Inject artificial error ToolMessages for blocked tool calls
        # 中文: 为阻止的工具调用注入人为错误 ToolMessages
        artificial_messages: list[ToolMessage | AIMessage] = [
            ToolMessage(
                content=tool_msg_content,
                tool_call_id=tool_call["id"],
                name=tool_call.get("name"),
                status="error",
            )
            for tool_call in blocked_calls
        ]

        if self.exit_behavior == "end":
            # Check if there are tool calls to other tools that would continue executing
            # 中文: 检查是否有对其他工具的工具调用将继续执行
            other_tools = [
                tc
                for tc in last_ai_message.tool_calls
                if self.tool_name is not None and tc["name"] != self.tool_name
            ]

            if other_tools:
                tool_names = ", ".join({tc["name"] for tc in other_tools})
                msg = (
                    f"Cannot end execution with other tool calls pending. "
                    f"Found calls to: {tool_names}. Use 'continue' or 'error' behavior instead."
                )
                raise NotImplementedError(msg)

            # Build final AI message content (displayed to user - includes thread/run details)
            # 中文: 构建最终的 AI 消息内容（向用户显示 - 包括线程/运行详细信息）
            # Use hypothetical thread count (what it would have been if call wasn't blocked)
            # 中文: 使用假设的线程数（如果调用没有被阻止，那么线程数会是多少）
            # to show which limit was actually exceeded
            # 中文: 显示实际超出了哪个限制
            hypothetical_thread_count = final_thread_count + len(blocked_calls)
            final_msg_content = _build_final_ai_message_content(
                hypothetical_thread_count,
                final_run_count,
                self.thread_limit,
                self.run_limit,
                self.tool_name,
            )
            artificial_messages.append(AIMessage(content=final_msg_content))

            return {
                "thread_tool_call_count": thread_counts,
                "run_tool_call_count": run_counts,
                "jump_to": "end",
                "messages": artificial_messages,
            }

        # For exit_behavior="continue", return error messages to block exceeded tools
        # 中文: 对于exit_behavior =“继续”，返回错误消息以阻止超出的工具
        return {
            "thread_tool_call_count": thread_counts,
            "run_tool_call_count": run_counts,
            "messages": artificial_messages,
        }

    @hook_config(can_jump_to=["end"])
    async def aafter_model(
        self,
        state: ToolCallLimitState[ResponseT],
        runtime: Runtime[ContextT],
    ) -> dict[str, Any] | None:
        """Async increment tool call counts after a model call and check limits.

        Args:
            state: The current agent state.
            runtime: The langgraph runtime.

        Returns:
            State updates with incremented tool call counts. If limits are exceeded
                and exit_behavior is `'end'`, also includes a jump to end with a
                `ToolMessage` and AI message for the single exceeded tool call.

        Raises:
            ToolCallLimitExceededError: If limits are exceeded and `exit_behavior`
                is `'error'`.
            NotImplementedError: If limits are exceeded, `exit_behavior` is `'end'`,
                and there are multiple tool calls.
        

        中文翻译:
        模型调用和检查限制后异步增量工具调用计数。
        参数：
            状态：当前代理状态。
            运行时：语言图运行时。
        返回：
            状态随着工具调用计数的增加而更新。如果超出限制
                exit_behavior 是 `'end'`，还包括跳转到结尾
                单个超出的工具调用的“ToolMessage”和 AI 消息。
        加薪：
            ToolCallLimitExceededError：如果超出限制并且“exit_behavior”
                是“错误”。
            NotImplementedError：如果超出限制，`exit_behavior` 为 `'end'`，
                并且有多个工具调用。"""
        return self.after_model(state, runtime)
