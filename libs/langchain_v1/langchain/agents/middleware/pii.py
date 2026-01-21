"""PII（个人身份信息）检测和处理中间件模块。

本模块提供对话中个人身份信息的检测和处理能力。

核心类:
--------
**PIIMiddleware**: PII 检测中间件

支持的 PII 类型:
-----------------
- `email`: 电子邮件地址
- `credit_card`: 信用卡号（Luhn 算法验证）
- `ip`: IP 地址
- `mac_address`: MAC 地址
- `url`: URL 链接

处理策略:
---------
- `block`: 检测到 PII 时抛出异常
- `redact`: 用 `[REDACTED_TYPE]` 占位符替换
- `mask`: 部分掩码（如 `****-****-****-1234`）
- `hash`: 用确定性哈希替换

使用示例:
---------
>>> from langchain.agents import create_agent
>>> from langchain.agents.middleware import PIIMiddleware
>>>
>>> # 脱敏用户输入中的邮箱
>>> agent = create_agent(
...     model="openai:gpt-4o",
...     middleware=[PIIMiddleware("email", strategy="redact")],
... )
>>>
>>> # 不同类型使用不同策略
>>> agent = create_agent(
...     model="openai:gpt-4o",
...     middleware=[
...         PIIMiddleware("credit_card", strategy="mask"),
...         PIIMiddleware("ip", strategy="hash"),
...     ],
... )
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from typing_extensions import override

from langchain.agents.middleware._redaction import (
    PIIDetectionError,
    PIIMatch,
    RedactionRule,
    ResolvedRedactionRule,
    apply_strategy,
    detect_credit_card,
    detect_email,
    detect_ip,
    detect_mac_address,
    detect_url,
)
from langchain.agents.middleware.types import AgentMiddleware, AgentState, hook_config

if TYPE_CHECKING:
    from collections.abc import Callable

    from langgraph.runtime import Runtime


class PIIMiddleware(AgentMiddleware):
    """Detect and handle Personally Identifiable Information (PII) in conversations.

    This middleware detects common PII types and applies configurable strategies
    to handle them. It can detect emails, credit cards, IP addresses, MAC addresses, and
    URLs in both user input and agent output.

    Built-in PII types:

    - `email`: Email addresses
    - `credit_card`: Credit card numbers (validated with Luhn algorithm)
    - `ip`: IP addresses (validated with stdlib)
    - `mac_address`: MAC addresses
    - `url`: URLs (both `http`/`https` and bare URLs)

    Strategies:

    - `block`: Raise an exception when PII is detected
    - `redact`: Replace PII with `[REDACTED_TYPE]` placeholders
    - `mask`: Partially mask PII (e.g., `****-****-****-1234` for credit card)
    - `hash`: Replace PII with deterministic hash (e.g., `<email_hash:a1b2c3d4>`)

    Strategy Selection Guide:

    | Strategy | Preserves Identity? | Best For                                |
    | -------- | ------------------- | --------------------------------------- |
    | `block`  | N/A                 | Avoid PII completely                    |
    | `redact` | No                  | General compliance, log sanitization    |
    | `mask`   | No                  | Human readability, customer service UIs |
    | `hash`   | Yes (pseudonymous)  | Analytics, debugging                    |

    Example:
        ```python
        from langchain.agents.middleware import PIIMiddleware
        from langchain.agents import create_agent

        # Redact all emails in user input
        # 中文: 编辑用户输入中的所有电子邮件
        agent = create_agent(
            "openai:gpt-5",
            middleware=[
                PIIMiddleware("email", strategy="redact"),
            ],
        )

        # Use different strategies for different PII types
        # 中文: 针对不同的 PII 类型使用不同的策略
        agent = create_agent(
            "openai:gpt-4o",
            middleware=[
                PIIMiddleware("credit_card", strategy="mask"),
                PIIMiddleware("url", strategy="redact"),
                PIIMiddleware("ip", strategy="hash"),
            ],
        )

        # Custom PII type with regex
        # 中文: 使用正则表达式自定义 PII 类型
        agent = create_agent(
            "openai:gpt-5",
            middleware=[
                PIIMiddleware("api_key", detector=r"sk-[a-zA-Z0-9]{32}", strategy="block"),
            ],
        )
        ```
    

    中文翻译:
    检测并处理对话中的个人身份信息 (PII)。
    该中间件检测常见的 PII 类型并应用可配置策略
    来处理它们。它可以检测电子邮件、信用卡、IP 地址、MAC 地址和
    用户输入和代理输出中的 URL。
    内置 PII 类型：
    - `电子邮件`：电子邮件地址
    - `credit_card`：信用卡号（使用 Luhn 算法验证）
    - `ip`：IP 地址（使用 stdlib 验证）
    - `mac_address`: MAC 地址
    - `url`：URL（`http`/`https` 和裸 URL）
    策略：
    - `block`：检测到 PII 时引发异常
    - `redact`：将 PII 替换为 `[REDACTED_TYPE]` 占位符
    - `mask`：部分屏蔽 PII（例如信用卡的 `****-****-****-1234`）
    - `hash`：用确定性哈希替换 PII（例如，`<email_hash:a1b2c3d4>`）
    策略选择指南：
    |战略|保留身份？ |最适合 |
    | -------- | ------------------- | --------------------------------------- |
    | `块` |不适用 |完全避免 PII |
    | `编辑` |没有 |一般合规性，日志清理 |
    | `面具` |没有 |人类可读性、客户服务用户界面 |
    | `哈希` |是（化名）|分析、调试 |
    示例：
        ````蟒蛇
        从 langchain.agents.middleware 导入 PIIMiddleware
        从 langchain.agents 导入 create_agent
        # 编辑用户输入中的所有电子邮件
        代理=创建_代理（
            “openai：gpt-5”，
            中间件=[
                PIIMiddleware（“电子邮件”，策略=“redact”），
            ],
        ）
        # 针对不同的 PII 类型使用不同的策略
        代理=创建_代理（
            “openai：gpt-4o”，
            中间件=[
                PIIMiddleware(“credit_card”,策略=“掩码”),
                PIIMiddleware(“url”,策略=“redact”),
                PIIMiddleware(“ip”,策略=“哈希”),
            ],
        ）
        # 使用正则表达式自定义 PII 类型
        代理=创建_代理（
            “openai：gpt-5”，
            中间件=[
                PIIMiddleware("api_key", detector=r"sk-[a-zA-Z0-9]{32}", Strategy="block"),
            ],
        ）
        ````"""

    def __init__(
        self,
        # From a typing point of view, the literals are covered by 'str'.
        # 中文: 从打字的角度来看，文字被“str”覆盖。
        # Nonetheless, we escape PYI051 to keep hints and autocompletion for the caller.
        # 中文: 尽管如此，我们还是转义了 PYI051 来为调用者保留提示和自动完成功能。
        pii_type: Literal["email", "credit_card", "ip", "mac_address", "url"] | str,  # noqa: PYI051
        *,
        strategy: Literal["block", "redact", "mask", "hash"] = "redact",
        detector: Callable[[str], list[PIIMatch]] | str | None = None,
        apply_to_input: bool = True,
        apply_to_output: bool = False,
        apply_to_tool_results: bool = False,
    ) -> None:
        """Initialize the PII detection middleware.

        Args:
            pii_type: Type of PII to detect.

                Can be a built-in type (`email`, `credit_card`, `ip`, `mac_address`,
                `url`) or a custom type name.
            strategy: How to handle detected PII.

                Options:

                * `block`: Raise `PIIDetectionError` when PII is detected
                * `redact`: Replace with `[REDACTED_TYPE]` placeholders
                * `mask`: Partially mask PII (show last few characters)
                * `hash`: Replace with deterministic hash (format: `<type_hash:digest>`)

            detector: Custom detector function or regex pattern.

                * If `Callable`: Function that takes content string and returns
                    list of `PIIMatch` objects
                * If `str`: Regex pattern to match PII
                * If `None`: Uses built-in detector for the `pii_type`
            apply_to_input: Whether to check user messages before model call.
            apply_to_output: Whether to check AI messages after model call.
            apply_to_tool_results: Whether to check tool result messages after tool execution.

        Raises:
            ValueError: If `pii_type` is not built-in and no detector is provided.
        

        中文翻译:
        初始化PII检测中间件。
        参数：
            pii_type：要检测的 PII 类型。
                可以是内置类型（`email`、`credit_card`、`ip`、`mac_address`、
                `url`) 或自定义类型名称。
            策略：如何处理检测到的 PII。
                选项：
                * `block`: 当检测到 PII 时引发 `PIIDetectionError`
                * `redact`：替换为 `[REDACTED_TYPE]` 占位符
                * `mask`：部分屏蔽 PII（显示最后几个字符）
                * `hash`：替换为确定性哈希（格式：`<type_hash:digest>`）
            detector：自定义检测器函数或正则表达式模式。
                * If `Callable`: 接受内容字符串并返回的函数
                    `PIIMatch` 对象列表
                * If `str`: 匹配 PII 的正则表达式模式
                * 如果“无”：使用内置检测器检测“pii_type”
            apply_to_input：模型调用前是否检查用户消息。
            apply_to_output：模型调用后是否检查AI消息。
            apply_to_tool_results：工具执行后是否检查工具结果消息。
        加薪：
            ValueError：如果“pii_type”不是内置的并且未提供检测器。"""
        super().__init__()

        self.apply_to_input = apply_to_input
        self.apply_to_output = apply_to_output
        self.apply_to_tool_results = apply_to_tool_results

        self._resolved_rule: ResolvedRedactionRule = RedactionRule(
            pii_type=pii_type,
            strategy=strategy,
            detector=detector,
        ).resolve()
        self.pii_type = self._resolved_rule.pii_type
        self.strategy = self._resolved_rule.strategy
        self.detector = self._resolved_rule.detector

    @property
    def name(self) -> str:
        """Name of the middleware.

        中文翻译:
        中间件的名称。"""
        return f"{self.__class__.__name__}[{self.pii_type}]"

    def _process_content(self, content: str) -> tuple[str, list[PIIMatch]]:
        """Apply the configured redaction rule to the provided content.

        中文翻译:
        将配置的密文规则应用于提供的内容。"""
        matches = self.detector(content)
        if not matches:
            return content, []
        sanitized = apply_strategy(content, matches, self.strategy)
        return sanitized, matches

    @hook_config(can_jump_to=["end"])
    @override
    def before_model(
        self,
        state: AgentState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Check user messages and tool results for PII before model invocation.

        Args:
            state: The current agent state.
            runtime: The langgraph runtime.

        Returns:
            Updated state with PII handled according to strategy, or `None` if no PII
                detected.

        Raises:
            PIIDetectionError: If PII is detected and strategy is `'block'`.
        

        中文翻译:
        在模型调用之前检查 PII 的用户消息和工具结果。
        参数：
            状态：当前代理状态。
            运行时：语言图运行时。
        返回：
            根据策略处理 PII 的更新状态，如果没有 PII，则为“无”
                检测到。
        加薪：
            PIIDetectionError：如果检测到 PII 并且策略为“阻止”。"""
        if not self.apply_to_input and not self.apply_to_tool_results:
            return None

        messages = state["messages"]
        if not messages:
            return None

        new_messages = list(messages)
        any_modified = False

        # Check user input if enabled
        # 中文: 检查用户输入（如果启用）
        if self.apply_to_input:
            # Get last user message
            # 中文: 获取最后一条用户消息
            last_user_msg = None
            last_user_idx = None
            for i in range(len(messages) - 1, -1, -1):
                if isinstance(messages[i], HumanMessage):
                    last_user_msg = messages[i]
                    last_user_idx = i
                    break

            if last_user_idx is not None and last_user_msg and last_user_msg.content:
                # Detect PII in message content
                # 中文: 检测邮件内容中的 PII
                content = str(last_user_msg.content)
                new_content, matches = self._process_content(content)

                if matches:
                    updated_message: AnyMessage = HumanMessage(
                        content=new_content,
                        id=last_user_msg.id,
                        name=last_user_msg.name,
                    )

                    new_messages[last_user_idx] = updated_message
                    any_modified = True

        # Check tool results if enabled
        # 中文: 检查工具结果（如果启用）
        if self.apply_to_tool_results:
            # Find the last AIMessage, then process all `ToolMessage` objects after it
            # 中文: 找到最后一个AIMessage，然后处理它后面的所有`ToolMessage`对象
            last_ai_idx = None
            for i in range(len(messages) - 1, -1, -1):
                if isinstance(messages[i], AIMessage):
                    last_ai_idx = i
                    break

            if last_ai_idx is not None:
                # Get all tool messages after the last AI message
                # 中文: 获取最后一条 AI 消息之后的所有工具消息
                for i in range(last_ai_idx + 1, len(messages)):
                    msg = messages[i]
                    if isinstance(msg, ToolMessage):
                        tool_msg = msg
                        if not tool_msg.content:
                            continue

                        content = str(tool_msg.content)
                        new_content, matches = self._process_content(content)

                        if not matches:
                            continue

                        # Create updated tool message
                        # 中文: 创建更新的工具消息
                        updated_message = ToolMessage(
                            content=new_content,
                            id=tool_msg.id,
                            name=tool_msg.name,
                            tool_call_id=tool_msg.tool_call_id,
                        )

                        new_messages[i] = updated_message
                        any_modified = True

        if any_modified:
            return {"messages": new_messages}

        return None

    @hook_config(can_jump_to=["end"])
    async def abefore_model(
        self,
        state: AgentState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Async check user messages and tool results for PII before model invocation.

        Args:
            state: The current agent state.
            runtime: The langgraph runtime.

        Returns:
            Updated state with PII handled according to strategy, or `None` if no PII
                detected.

        Raises:
            PIIDetectionError: If PII is detected and strategy is `'block'`.
        

        中文翻译:
        在模型调用之前异步检查 PII 的用户消息和工具结果。
        参数：
            状态：当前代理状态。
            运行时：语言图运行时。
        返回：
            根据策略处理 PII 的更新状态，如果没有 PII，则为“无”
                检测到。
        加薪：
            PIIDetectionError：如果检测到 PII 并且策略为“阻止”。"""
        return self.before_model(state, runtime)

    @override
    def after_model(
        self,
        state: AgentState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Check AI messages for PII after model invocation.

        Args:
            state: The current agent state.
            runtime: The langgraph runtime.

        Returns:
            Updated state with PII handled according to strategy, or None if no PII
                detected.

        Raises:
            PIIDetectionError: If PII is detected and strategy is `'block'`.
        

        中文翻译:
        模型调用后检查 PII 的 AI 消息。
        参数：
            状态：当前代理状态。
            运行时：语言图运行时。
        返回：
            根据策略处理 PII 的更新状态，如果没有 PII，则为 None
                检测到。
        加薪：
            PIIDetectionError：如果检测到 PII 并且策略为“阻止”。"""
        if not self.apply_to_output:
            return None

        messages = state["messages"]
        if not messages:
            return None

        # Get last AI message
        # 中文: 获取最后一条 AI 消息
        last_ai_msg = None
        last_ai_idx = None
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if isinstance(msg, AIMessage):
                last_ai_msg = msg
                last_ai_idx = i
                break

        if last_ai_idx is None or not last_ai_msg or not last_ai_msg.content:
            return None

        # Detect PII in message content
        # 中文: 检测邮件内容中的 PII
        content = str(last_ai_msg.content)
        new_content, matches = self._process_content(content)

        if not matches:
            return None

        # Create updated message
        # 中文: 创建更新的消息
        updated_message = AIMessage(
            content=new_content,
            id=last_ai_msg.id,
            name=last_ai_msg.name,
            tool_calls=last_ai_msg.tool_calls,
        )

        # Return updated messages
        # 中文: 返回更新的消息
        new_messages = list(messages)
        new_messages[last_ai_idx] = updated_message

        return {"messages": new_messages}

    async def aafter_model(
        self,
        state: AgentState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Async check AI messages for PII after model invocation.

        Args:
            state: The current agent state.
            runtime: The langgraph runtime.

        Returns:
            Updated state with PII handled according to strategy, or None if no PII
                detected.

        Raises:
            PIIDetectionError: If PII is detected and strategy is `'block'`.
        

        中文翻译:
        模型调用后异步检查 PII 的 AI 消息。
        参数：
            状态：当前代理状态。
            运行时：语言图运行时。
        返回：
            根据策略处理 PII 的更新状态，如果没有 PII，则为 None
                检测到。
        加薪：
            PIIDetectionError：如果检测到 PII 并且策略为“阻止”。"""
        return self.after_model(state, runtime)


__all__ = [
    "PIIDetectionError",
    "PIIMatch",
    "PIIMiddleware",
    "detect_credit_card",
    "detect_email",
    "detect_ip",
    "detect_mac_address",
    "detect_url",
]
