"""上下文编辑中间件模块。

本模块提供对话上下文的自动修剪功能，当 token 数量超过阈值时
清除旧的工具调用结果以控制上下文长度。

此实现模仿了 Anthropic 的上下文编辑功能：当对话超过配置的 token
阈值时，自动清除较早的工具调用结果。

核心类:
--------
**ContextEditingMiddleware**: 上下文编辑中间件
**ClearToolUsesEdit**: 清除工具使用的编辑策略

使用示例:
---------
>>> from langchain.agents import create_agent
>>> from langchain.agents.middleware import ContextEditingMiddleware, ClearToolUsesEdit
>>>
>>> editor = ContextEditingMiddleware(
...     edits=[ClearToolUsesEdit(trigger=100_000, keep=3)],
... )
>>>
>>> agent = create_agent(
...     model="openai:gpt-4o",
...     middleware=[editor],
... )
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    ToolMessage,
)
from langchain_core.messages.utils import count_tokens_approximately
from typing_extensions import Protocol

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)

DEFAULT_TOOL_PLACEHOLDER = "[cleared]"


TokenCounter = Callable[
    [Sequence[BaseMessage]],
    int,
]


class ContextEdit(Protocol):
    """Protocol describing a context editing strategy.

    中文翻译:
    描述上下文编辑策略的协议。"""

    def apply(
        self,
        messages: list[AnyMessage],
        *,
        count_tokens: TokenCounter,
    ) -> None:
        """Apply an edit to the message list in place.

        中文翻译:
        将编辑应用于适当的消息列表。"""
        ...


@dataclass(slots=True)
class ClearToolUsesEdit(ContextEdit):
    """Configuration for clearing tool outputs when token limits are exceeded.

    中文翻译:
    用于在超出令牌限制时清除工具输出的配置。"""

    trigger: int = 100_000
    """Token count that triggers the edit.

    中文翻译:
    触发编辑的令牌计数。"""

    clear_at_least: int = 0
    """Minimum number of tokens to reclaim when the edit runs.

    中文翻译:
    编辑运行时要回收的最小令牌数。"""

    keep: int = 3
    """Number of most recent tool results that must be preserved.

    中文翻译:
    必须保留的最新工具结果的数量。"""

    clear_tool_inputs: bool = False
    """Whether to clear the originating tool call parameters on the AI message.

    中文翻译:
    是否清除AI消息上的原始工具调用参数。"""

    exclude_tools: Sequence[str] = ()
    """List of tool names to exclude from clearing.

    中文翻译:
    要从清除中排除的工具名称列表。"""

    placeholder: str = DEFAULT_TOOL_PLACEHOLDER
    """Placeholder text inserted for cleared tool outputs.

    中文翻译:
    为清除的工具输出插入占位符文本。"""

    def apply(
        self,
        messages: list[AnyMessage],
        *,
        count_tokens: TokenCounter,
    ) -> None:
        """Apply the clear-tool-uses strategy.

        中文翻译:
        应用明确工具使用策略。"""
        tokens = count_tokens(messages)

        if tokens <= self.trigger:
            return

        candidates = [
            (idx, msg) for idx, msg in enumerate(messages) if isinstance(msg, ToolMessage)
        ]

        if self.keep >= len(candidates):
            candidates = []
        elif self.keep:
            candidates = candidates[: -self.keep]

        cleared_tokens = 0
        excluded_tools = set(self.exclude_tools)

        for idx, tool_message in candidates:
            if tool_message.response_metadata.get("context_editing", {}).get("cleared"):
                continue

            ai_message = next(
                (m for m in reversed(messages[:idx]) if isinstance(m, AIMessage)), None
            )

            if ai_message is None:
                continue

            tool_call = next(
                (
                    call
                    for call in ai_message.tool_calls
                    if call.get("id") == tool_message.tool_call_id
                ),
                None,
            )

            if tool_call is None:
                continue

            if (tool_message.name or tool_call["name"]) in excluded_tools:
                continue

            messages[idx] = tool_message.model_copy(
                update={
                    "artifact": None,
                    "content": self.placeholder,
                    "response_metadata": {
                        **tool_message.response_metadata,
                        "context_editing": {
                            "cleared": True,
                            "strategy": "clear_tool_uses",
                        },
                    },
                }
            )

            if self.clear_tool_inputs:
                messages[messages.index(ai_message)] = self._build_cleared_tool_input_message(
                    ai_message,
                    tool_message.tool_call_id,
                )

            if self.clear_at_least > 0:
                new_token_count = count_tokens(messages)
                cleared_tokens = max(0, tokens - new_token_count)
                if cleared_tokens >= self.clear_at_least:
                    break

        return

    def _build_cleared_tool_input_message(
        self,
        message: AIMessage,
        tool_call_id: str,
    ) -> AIMessage:
        updated_tool_calls = []
        cleared_any = False
        for tool_call in message.tool_calls:
            updated_call = dict(tool_call)
            if updated_call.get("id") == tool_call_id:
                updated_call["args"] = {}
                cleared_any = True
            updated_tool_calls.append(updated_call)

        metadata = dict(getattr(message, "response_metadata", {}))
        context_entry = dict(metadata.get("context_editing", {}))
        if cleared_any:
            cleared_ids = set(context_entry.get("cleared_tool_inputs", []))
            cleared_ids.add(tool_call_id)
            context_entry["cleared_tool_inputs"] = sorted(cleared_ids)
            metadata["context_editing"] = context_entry

        return message.model_copy(
            update={
                "tool_calls": updated_tool_calls,
                "response_metadata": metadata,
            }
        )


class ContextEditingMiddleware(AgentMiddleware):
    """Automatically prune tool results to manage context size.

    The middleware applies a sequence of edits when the total input token count exceeds
    configured thresholds.

    Currently the `ClearToolUsesEdit` strategy is supported, aligning with Anthropic's
    `clear_tool_uses_20250919` behavior [(read more)](https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool).
    

    中文翻译:
    自动修剪工具结果以管理上下文大小。
    当输入令牌总数超过时，中间件将应用一系列编辑
    配置的阈值。
    目前支持“ClearToolUsesEdit”策略，与 Anthropic 的策略保持一致
    `clear_tool_uses_20250919` 行为 [(阅读更多)](https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool)。"""

    edits: list[ContextEdit]
    token_count_method: Literal["approximate", "model"]

    def __init__(
        self,
        *,
        edits: Iterable[ContextEdit] | None = None,
        token_count_method: Literal["approximate", "model"] = "approximate",  # noqa: S107
    ) -> None:
        """Initialize an instance of context editing middleware.

        Args:
            edits: Sequence of edit strategies to apply.

                Defaults to a single `ClearToolUsesEdit` mirroring Anthropic defaults.
            token_count_method: Whether to use approximate token counting
                (faster, less accurate) or exact counting implemented by the
                chat model (potentially slower, more accurate).
        

        中文翻译:
        初始化上下文编辑中间件的实例。
        参数：
            编辑：要应用的编辑策略的顺序。
                默认为单个“ClearToolUsesEdit”镜像 Anthropic 默认值。
            token_count_method：是否使用近似令牌计数
                （更快，不太准确）或由
                聊天模型（可能更慢，更准确）。"""
        super().__init__()
        self.edits = list(edits or (ClearToolUsesEdit(),))
        self.token_count_method = token_count_method

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Apply context edits before invoking the model via handler.

        中文翻译:
        在通过处理程序调用模型之前应用上下文编辑。"""
        if not request.messages:
            return handler(request)

        if self.token_count_method == "approximate":  # noqa: S105

            def count_tokens(messages: Sequence[BaseMessage]) -> int:
                return count_tokens_approximately(messages)

        else:
            system_msg = [request.system_message] if request.system_message else []

            def count_tokens(messages: Sequence[BaseMessage]) -> int:
                return request.model.get_num_tokens_from_messages(
                    system_msg + list(messages), request.tools
                )

        edited_messages = deepcopy(list(request.messages))
        for edit in self.edits:
            edit.apply(edited_messages, count_tokens=count_tokens)

        return handler(request.override(messages=edited_messages))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Apply context edits before invoking the model via handler (async version).

        中文翻译:
        在通过处理程序（异步版本）调用模型之前应用上下文编辑。"""
        if not request.messages:
            return await handler(request)

        if self.token_count_method == "approximate":  # noqa: S105

            def count_tokens(messages: Sequence[BaseMessage]) -> int:
                return count_tokens_approximately(messages)

        else:
            system_msg = [request.system_message] if request.system_message else []

            def count_tokens(messages: Sequence[BaseMessage]) -> int:
                return request.model.get_num_tokens_from_messages(
                    system_msg + list(messages), request.tools
                )

        edited_messages = deepcopy(list(request.messages))
        for edit in self.edits:
            edit.apply(edited_messages, count_tokens=count_tokens)

        return await handler(request.override(messages=edited_messages))


__all__ = [
    "ClearToolUsesEdit",
    "ContextEditingMiddleware",
]
