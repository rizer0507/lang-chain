"""对话摘要中间件模块。

本模块提供对话历史过长时自动摘要的能力，保持上下文连续性。

核心类:
--------
**SummarizationMiddleware**: 对话摘要中间件

功能特性:
---------
- 监控消息 token 数量
- 达到阈值时自动摘要旧消息
- 保留最近的消息
- 确保 AI/Tool 消息对不被拆分

上下文大小配置:
---------------
- `("fraction", 0.5)`: 模型最大输入 token 的百分比
- `("tokens", 3000)`: 绝对 token 数量
- `("messages", 50)`: 绝对消息数量

使用示例:
---------
>>> from langchain.agents import create_agent
>>> from langchain.agents.middleware import SummarizationMiddleware
>>>
>>> # 当达到 50 条消息时触发摘要，保留最近 20 条
>>> summarizer = SummarizationMiddleware(
...     model="openai:gpt-4o-mini",
...     trigger=("messages", 50),
...     keep=("messages", 20),
... )
>>>
>>> agent = create_agent(
...     model="openai:gpt-4o",
...     middleware=[summarizer],
... )
"""

import uuid
import warnings
from collections.abc import Callable, Iterable, Mapping
from functools import partial
from typing import Any, Literal, cast

from langchain_core.messages import (
    AnyMessage,
    MessageLikeRepresentation,
    RemoveMessage,
    ToolMessage,
)
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.utils import (
    count_tokens_approximately,
    get_buffer_string,
    trim_messages,
)
from langgraph.graph.message import (
    REMOVE_ALL_MESSAGES,
)
from langgraph.runtime import Runtime
from typing_extensions import override

from langchain.agents.middleware.types import AgentMiddleware, AgentState
from langchain.chat_models import BaseChatModel, init_chat_model

TokenCounter = Callable[[Iterable[MessageLikeRepresentation]], int]

DEFAULT_SUMMARY_PROMPT = """<role>
Context Extraction Assistant
</role>

<primary_objective>
Your sole objective in this task is to extract the highest quality/most relevant context from the conversation history below.
</primary_objective>

<objective_information>
You're nearing the total number of input tokens you can accept, so you must extract the highest quality/most relevant pieces of information from your conversation history.
This context will then overwrite the conversation history presented below. Because of this, ensure the context you extract is only the most important information to your overall goal.
</objective_information>

<instructions>
The conversation history below will be replaced with the context you extract in this step. Because of this, you must do your very best to extract and record all of the most important context from the conversation history.
You want to ensure that you don't repeat any actions you've already completed, so the context you extract from the conversation history should be focused on the most important information to your overall goal.
</instructions>

The user will message you with the full message history you'll be extracting context from, to then replace. Carefully read over it all, and think deeply about what information is most important to your overall goal that should be saved:

With all of this in mind, please carefully read over the entire conversation history, and extract the most important and relevant context to replace it so that you can free up space in the conversation history.
Respond ONLY with the extracted context. Do not include any additional information, or text before or after the extracted context.

<messages>
Messages to summarize:
{messages}
</messages>

 中文翻译:
 <角色>
上下文提取助手
</角色>
<主要目标>
您在此任务中的唯一目标是从下面的对话历史记录中提取最高质量/最相关的上下文。
</主要目标>
<目标信息>
您已接近可以接受的输入令牌总数，因此您必须从对话历史记录中提取最高质量/最相关的信息。
然后，此上下文将覆盖下面显示的对话历史记录。因此，请确保您提取的上下文只是对您的总体目标最重要的信息。
</目标信息>
<说明>
下面的对话历史记录将替换为您在此步骤中提取的上下文。因此，您必须尽最大努力从对话历史记录中提取并记录所有最重要的上下文。
您希望确保不会重复已完成的任何操作，因此从对话历史记录中提取的上下文应集中于对您的总体目标最重要的信息。
</说明>
用户将向您发送消息，其中包含您将从中提取上下文的完整消息历史记录，然后进行替换。仔细阅读所有内容，并深入思考哪些信息对您的总体目标最重要，应该保存：
考虑到所有这些，请仔细阅读整个对话历史记录，并提取最重要且相关的上下文来替换它，以便您可以释放对话历史记录中的空间。
仅使用提取的上下文进行响应。不要在提取的上下文之前或之后包含任何附加信息或文本。
<消息>
消息总结：
{消息}
</消息>"""  # noqa: E501

_DEFAULT_MESSAGES_TO_KEEP = 20
_DEFAULT_TRIM_TOKEN_LIMIT = 4000
_DEFAULT_FALLBACK_MESSAGE_COUNT = 15

ContextFraction = tuple[Literal["fraction"], float]
"""Fraction of model's maximum input tokens.

Example:
    To specify 50% of the model's max input tokens:

    ```python
    ("fraction", 0.5)
    ```

中文翻译:
模型最大输入标记的分数。
示例：
    要指定模型最大输入标记的 50%：
    ````蟒蛇
    （“分数”，0.5）
    ````
"""

ContextTokens = tuple[Literal["tokens"], int]
"""Absolute number of tokens.

Example:
    To specify 3000 tokens:

    ```python
    ("tokens", 3000)
    ```

中文翻译:
代币的绝对数量。
示例：
    要指定 3000 个令牌：
    ````蟒蛇
    （“代币”，3000）
    ````
"""

ContextMessages = tuple[Literal["messages"], int]
"""Absolute number of messages.

Example:
    To specify 50 messages:

    ```python
    ("messages", 50)
    ```

中文翻译:
消息的绝对数量。
示例：
    要指定 50 条消息：
    ````蟒蛇
    （“消息”，50）
    ````
"""

ContextSize = ContextFraction | ContextTokens | ContextMessages
"""Union type for context size specifications.

Can be either:

- [`ContextFraction`][langchain.agents.middleware.summarization.ContextFraction]: A
    fraction of the model's maximum input tokens.
- [`ContextTokens`][langchain.agents.middleware.summarization.ContextTokens]: An absolute
    number of tokens.
- [`ContextMessages`][langchain.agents.middleware.summarization.ContextMessages]: An
    absolute number of messages.

Depending on use with `trigger` or `keep` parameters, this type indicates either
when to trigger summarization or how much context to retain.

Example:
    ```python
    # ContextFraction
    # 中文: 上下文分数
    context_size: ContextSize = ("fraction", 0.5)

    # ContextTokens
    # 中文: 上下文令牌
    context_size: ContextSize = ("tokens", 3000)

    # ContextMessages
    # 中文: 上下文消息
    context_size: ContextSize = ("messages", 50)
    ```

中文翻译:
上下文大小规范的联合类型。
可以是：
- [`ContextFraction`][langchain.agents.middleware.summarization.ContextFraction]：A
    模型最大输入标记的分数。
- [`ContextTokens`][langchain.agents.middleware.summarization.ContextTokens]：绝对值
    代币数量。
- [`ContextMessages`][langchain.agents.middleware.summarization.ContextMessages]：一个
    消息的绝对数量。
根据与“trigger”或“keep”参数的使用，此类型指示
何时触发摘要或保留多少上下文。
示例：
    ````蟒蛇
    # 上下文分数
    context_size: ContextSize = ("分数", 0.5)
    # 上下文令牌
    context_size: ContextSize = ("令牌", 3000)
    # 上下文消息
    context_size: ContextSize = ("消息", 50)
    ````
"""


def _get_approximate_token_counter(model: BaseChatModel) -> TokenCounter:
    """Tune parameters of approximate token counter based on model type.

    中文翻译:
    根据模型类型调整近似令牌计数器的参数。"""
    if model._llm_type == "anthropic-chat":  # noqa: SLF001
        # 3.3 was estimated in an offline experiment, comparing with Claude's token-counting
        # 中文: 3.3 离线实验中估计的，与 Claude 的 token-counting 进行比较
        # API: https://platform.claude.com/docs/en/build-with-claude/token-counting
        # 中文: API：https://platform.claude.com/docs/en/build-with-claude/token-counting
        return partial(count_tokens_approximately, chars_per_token=3.3)
    return count_tokens_approximately


class SummarizationMiddleware(AgentMiddleware):
    """Summarizes conversation history when token limits are approached.

    This middleware monitors message token counts and automatically summarizes older
    messages when a threshold is reached, preserving recent messages and maintaining
    context continuity by ensuring AI/Tool message pairs remain together.
    

    中文翻译:
    当达到令牌限制时总结对话历史记录。
    该中间件监视消息令牌计数并自动总结旧的
    达到阈值时发送消息，保留最近的消息并维护
    通过确保 AI/工具消息对保持在一起来实现上下文连续性。"""

    def __init__(
        self,
        model: str | BaseChatModel,
        *,
        trigger: ContextSize | list[ContextSize] | None = None,
        keep: ContextSize = ("messages", _DEFAULT_MESSAGES_TO_KEEP),
        token_counter: TokenCounter = count_tokens_approximately,
        summary_prompt: str = DEFAULT_SUMMARY_PROMPT,
        trim_tokens_to_summarize: int | None = _DEFAULT_TRIM_TOKEN_LIMIT,
        **deprecated_kwargs: Any,
    ) -> None:
        """Initialize summarization middleware.

        Args:
            model: The language model to use for generating summaries.
            trigger: One or more thresholds that trigger summarization.

                Provide a single
                [`ContextSize`][langchain.agents.middleware.summarization.ContextSize]
                tuple or a list of tuples, in which case summarization runs when any
                threshold is met.

                !!! example

                    ```python
                    # Trigger summarization when 50 messages is reached
                    # 中文: 当达到50条消息时触发汇总
                    ("messages", 50)

                    # Trigger summarization when 3000 tokens is reached
                    # 中文: 当达到3000个token时触发汇总
                    ("tokens", 3000)

                    # Trigger summarization either when 80% of model's max input tokens
                    # 中文: 当模型最大输入标记的 80% 时触发汇总
                    # is reached or when 100 messages is reached (whichever comes first)
                    # 中文: 已达到或达到 100 条消息时（以先到者为准）
                    [("fraction", 0.8), ("messages", 100)]
                    ```

                    See [`ContextSize`][langchain.agents.middleware.summarization.ContextSize]
                    for more details.
            keep: Context retention policy applied after summarization.

                Provide a [`ContextSize`][langchain.agents.middleware.summarization.ContextSize]
                tuple to specify how much history to preserve.

                Defaults to keeping the most recent `20` messages.

                Does not support multiple values like `trigger`.

                !!! example

                    ```python
                    # Keep the most recent 20 messages
                    # 中文: 保留最近20条消息
                    ("messages", 20)

                    # Keep the most recent 3000 tokens
                    # 中文: 保留最近的3000个代币
                    ("tokens", 3000)

                    # Keep the most recent 30% of the model's max input tokens
                    # 中文: 保留模型最大输入令牌的最新 30%
                    ("fraction", 0.3)
                    ```
            token_counter: Function to count tokens in messages.
            summary_prompt: Prompt template for generating summaries.
            trim_tokens_to_summarize: Maximum tokens to keep when preparing messages for
                the summarization call.

                Pass `None` to skip trimming entirely.
        

        中文翻译:
        初始化摘要中间件。
        参数：
            model：用于生成摘要的语言模型。
            触发：触发汇总的一个或多个阈值。
                提供单
                [`ContextSize`][langchain.agents.middleware.summarization.ContextSize]
                元组或元组列表，在这种情况下，当任何
                达到阈值。
                !!!例子
                    ````蟒蛇
                    # 当达到50条消息时触发汇总
                    （“消息”，50）
                    # 当达到3000个token时触发汇总
                    （“代币”，3000）
                    # 当模型最大输入标记的 80% 时触发摘要
                    # 达到或达到 100 条消息时（以先到者为准）
                    [(“分数”，0.8)，(“消息”，100)]
                    ````
                    请参阅[`ContextSize`][langchain.agents.middleware.summarization.ContextSize]
                    了解更多详情。
            keep：汇总后应用上下文保留策略。
                提供 [`ContextSize`][langchain.agents.middleware.summarization.ContextSize]
                元组指定要保留多少历史记录。
                默认保留最近的“20”消息。
                不支持“trigger”等多个值。
                !!!例子
                    ````蟒蛇
                    # 保留最近20条消息
                    （“消息”，20）
                    # 保留最近的3000个代币
                    （“代币”，3000）
                    # 保留模型最大输入标记的最新 30%
                    （“分数”，0.3）
                    ````
            token_counter：计算消息中令牌的函数。
            Summary_prompt：生成摘要的提示模板。
            trim_tokens_to_summarize：准备消息时保留的最大令牌数
                总结调用。
                传递“None”以完全跳过修剪。"""
        # Handle deprecated parameters
        # 中文: 处理已弃用的参数
        if "max_tokens_before_summary" in deprecated_kwargs:
            value = deprecated_kwargs["max_tokens_before_summary"]
            warnings.warn(
                "max_tokens_before_summary is deprecated. Use trigger=('tokens', value) instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if trigger is None and value is not None:
                trigger = ("tokens", value)

        if "messages_to_keep" in deprecated_kwargs:
            value = deprecated_kwargs["messages_to_keep"]
            warnings.warn(
                "messages_to_keep is deprecated. Use keep=('messages', value) instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if keep == ("messages", _DEFAULT_MESSAGES_TO_KEEP):
                keep = ("messages", value)

        super().__init__()

        if isinstance(model, str):
            model = init_chat_model(model)

        self.model = model
        if trigger is None:
            self.trigger: ContextSize | list[ContextSize] | None = None
            trigger_conditions: list[ContextSize] = []
        elif isinstance(trigger, list):
            validated_list = [self._validate_context_size(item, "trigger") for item in trigger]
            self.trigger = validated_list
            trigger_conditions = validated_list
        else:
            validated = self._validate_context_size(trigger, "trigger")
            self.trigger = validated
            trigger_conditions = [validated]
        self._trigger_conditions = trigger_conditions

        self.keep = self._validate_context_size(keep, "keep")
        if token_counter is count_tokens_approximately:
            self.token_counter = _get_approximate_token_counter(self.model)
        else:
            self.token_counter = token_counter
        self.summary_prompt = summary_prompt
        self.trim_tokens_to_summarize = trim_tokens_to_summarize

        requires_profile = any(condition[0] == "fraction" for condition in self._trigger_conditions)
        if self.keep[0] == "fraction":
            requires_profile = True
        if requires_profile and self._get_profile_limits() is None:
            msg = (
                "Model profile information is required to use fractional token limits, "
                "and is unavailable for the specified model. Please use absolute token "
                "counts instead, or pass "
                '`\n\nChatModel(..., profile={"max_input_tokens": ...})`.\n\n'
                "with a desired integer value of the model's maximum input tokens."
            )
            raise ValueError(msg)

    @override
    def before_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Process messages before model invocation, potentially triggering summarization.

        中文翻译:
        在模型调用之前处理消息，可能会触发摘要。"""
        messages = state["messages"]
        self._ensure_message_ids(messages)

        total_tokens = self.token_counter(messages)
        if not self._should_summarize(messages, total_tokens):
            return None

        cutoff_index = self._determine_cutoff_index(messages)

        if cutoff_index <= 0:
            return None

        messages_to_summarize, preserved_messages = self._partition_messages(messages, cutoff_index)

        summary = self._create_summary(messages_to_summarize)
        new_messages = self._build_new_messages(summary)

        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *new_messages,
                *preserved_messages,
            ]
        }

    @override
    async def abefore_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Process messages before model invocation, potentially triggering summarization.

        中文翻译:
        在模型调用之前处理消息，可能会触发摘要。"""
        messages = state["messages"]
        self._ensure_message_ids(messages)

        total_tokens = self.token_counter(messages)
        if not self._should_summarize(messages, total_tokens):
            return None

        cutoff_index = self._determine_cutoff_index(messages)

        if cutoff_index <= 0:
            return None

        messages_to_summarize, preserved_messages = self._partition_messages(messages, cutoff_index)

        summary = await self._acreate_summary(messages_to_summarize)
        new_messages = self._build_new_messages(summary)

        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *new_messages,
                *preserved_messages,
            ]
        }

    def _should_summarize(self, messages: list[AnyMessage], total_tokens: int) -> bool:
        """Determine whether summarization should run for the current token usage.

        中文翻译:
        确定是否应针对当前令牌使用情况运行汇总。"""
        if not self._trigger_conditions:
            return False

        for kind, value in self._trigger_conditions:
            if kind == "messages" and len(messages) >= value:
                return True
            if kind == "tokens" and total_tokens >= value:
                return True
            if kind == "fraction":
                max_input_tokens = self._get_profile_limits()
                if max_input_tokens is None:
                    continue
                threshold = int(max_input_tokens * value)
                if threshold <= 0:
                    threshold = 1
                if total_tokens >= threshold:
                    return True
        return False

    def _determine_cutoff_index(self, messages: list[AnyMessage]) -> int:
        """Choose cutoff index respecting retention configuration.

        中文翻译:
        选择尊重保留配置的截止索引。"""
        kind, value = self.keep
        if kind in {"tokens", "fraction"}:
            token_based_cutoff = self._find_token_based_cutoff(messages)
            if token_based_cutoff is not None:
                return token_based_cutoff
            # None cutoff -> model profile data not available (caught in __init__ but
            # 中文: 无截止 -> 模型配置文件数据不可用（在 __init__ 中捕获，但
            # here for safety), fallback to message count
            # 中文: 为了安全起见），回退到消息计数
            return self._find_safe_cutoff(messages, _DEFAULT_MESSAGES_TO_KEEP)
        return self._find_safe_cutoff(messages, cast("int", value))

    def _find_token_based_cutoff(self, messages: list[AnyMessage]) -> int | None:
        """Find cutoff index based on target token retention.

        中文翻译:
        根据目标令牌保留找到截止索引。"""
        if not messages:
            return 0

        kind, value = self.keep
        if kind == "fraction":
            max_input_tokens = self._get_profile_limits()
            if max_input_tokens is None:
                return None
            target_token_count = int(max_input_tokens * value)
        elif kind == "tokens":
            target_token_count = int(value)
        else:
            return None

        if target_token_count <= 0:
            target_token_count = 1

        if self.token_counter(messages) <= target_token_count:
            return 0

        # Use binary search to identify the earliest message index that keeps the
        # 中文: 使用二分查找来确定保留该消息的最早的消息索引
        # suffix within the token budget.
        # 中文: 代币预算内的后缀。
        left, right = 0, len(messages)
        cutoff_candidate = len(messages)
        max_iterations = len(messages).bit_length() + 1
        for _ in range(max_iterations):
            if left >= right:
                break

            mid = (left + right) // 2
            if self.token_counter(messages[mid:]) <= target_token_count:
                cutoff_candidate = mid
                right = mid
            else:
                left = mid + 1

        if cutoff_candidate == len(messages):
            cutoff_candidate = left

        if cutoff_candidate >= len(messages):
            if len(messages) == 1:
                return 0
            cutoff_candidate = len(messages) - 1

        # Advance past any ToolMessages to avoid splitting AI/Tool pairs
        # 中文: 前进通过任何 ToolMessages 以避免分裂 AI/工具对
        return self._find_safe_cutoff_point(messages, cutoff_candidate)

    def _get_profile_limits(self) -> int | None:
        """Retrieve max input token limit from the model profile.

        中文翻译:
        从模型配置文件中检索最大输入令牌限制。"""
        try:
            profile = self.model.profile
        except AttributeError:
            return None

        if not isinstance(profile, Mapping):
            return None

        max_input_tokens = profile.get("max_input_tokens")

        if not isinstance(max_input_tokens, int):
            return None

        return max_input_tokens

    def _validate_context_size(self, context: ContextSize, parameter_name: str) -> ContextSize:
        """Validate context configuration tuples.

        中文翻译:
        验证上下文配置元组。"""
        kind, value = context
        if kind == "fraction":
            if not 0 < value <= 1:
                msg = f"Fractional {parameter_name} values must be between 0 and 1, got {value}."
                raise ValueError(msg)
        elif kind in {"tokens", "messages"}:
            if value <= 0:
                msg = f"{parameter_name} thresholds must be greater than 0, got {value}."
                raise ValueError(msg)
        else:
            msg = f"Unsupported context size type {kind} for {parameter_name}."
            raise ValueError(msg)
        return context

    def _build_new_messages(self, summary: str) -> list[HumanMessage]:
        return [
            HumanMessage(content=f"Here is a summary of the conversation to date:\n\n{summary}")
        ]

    def _ensure_message_ids(self, messages: list[AnyMessage]) -> None:
        """Ensure all messages have unique IDs for the add_messages reducer.

        中文翻译:
        确保所有消息都具有 add_messages 缩减程序的唯一 ID。"""
        for msg in messages:
            if msg.id is None:
                msg.id = str(uuid.uuid4())

    def _partition_messages(
        self,
        conversation_messages: list[AnyMessage],
        cutoff_index: int,
    ) -> tuple[list[AnyMessage], list[AnyMessage]]:
        """Partition messages into those to summarize and those to preserve.

        中文翻译:
        将消息分为要总结的消息和要保留的消息。"""
        messages_to_summarize = conversation_messages[:cutoff_index]
        preserved_messages = conversation_messages[cutoff_index:]

        return messages_to_summarize, preserved_messages

    def _find_safe_cutoff(self, messages: list[AnyMessage], messages_to_keep: int) -> int:
        """Find safe cutoff point that preserves AI/Tool message pairs.

        Returns the index where messages can be safely cut without separating
        related AI and Tool messages. Returns `0` if no safe cutoff is found.

        This is aggressive with summarization - if the target cutoff lands in the
        middle of tool messages, we advance past all of them (summarizing more).
        

        中文翻译:
        找到保留 AI/工具消息对的安全截止点。
        返回可以安全剪切消息而不分离的索引
        相关AI和工具消息。如果未找到安全截止点，则返回“0”。
        这是积极的总结 - 如果目标截止落在
        在工具消息的中间，我们超越了所有这些（总结更多）。"""
        if len(messages) <= messages_to_keep:
            return 0

        target_cutoff = len(messages) - messages_to_keep
        return self._find_safe_cutoff_point(messages, target_cutoff)

    def _find_safe_cutoff_point(self, messages: list[AnyMessage], cutoff_index: int) -> int:
        """Find a safe cutoff point that doesn't split AI/Tool message pairs.

        If the message at cutoff_index is a ToolMessage, advance until we find
        a non-ToolMessage. This ensures we never cut in the middle of parallel
        tool call responses.
        

        中文翻译:
        找到一个不会分裂 AI/工具消息对的安全截止点。
        如果 cutoff_index 处的消息是 ToolMessage，则前进直到找到
        非 ToolMessage。这确保了我们永远不会在平行的中间切割
        工具调用响应。"""
        while cutoff_index < len(messages) and isinstance(messages[cutoff_index], ToolMessage):
            cutoff_index += 1
        return cutoff_index

    def _create_summary(self, messages_to_summarize: list[AnyMessage]) -> str:
        """Generate summary for the given messages.

        中文翻译:
        生成给定消息的摘要。"""
        if not messages_to_summarize:
            return "No previous conversation history."

        trimmed_messages = self._trim_messages_for_summary(messages_to_summarize)
        if not trimmed_messages:
            return "Previous conversation was too long to summarize."

        # Format messages to avoid token inflation from metadata when str() is called on
        # 中文: 格式化消息以避免调用 str() 时元数据中的令牌膨胀
        # message objects
        # 中文: 消息对象
        formatted_messages = get_buffer_string(trimmed_messages)

        try:
            response = self.model.invoke(self.summary_prompt.format(messages=formatted_messages))
            return response.text.strip()
        except Exception as e:
            return f"Error generating summary: {e!s}"

    async def _acreate_summary(self, messages_to_summarize: list[AnyMessage]) -> str:
        """Generate summary for the given messages.

        中文翻译:
        生成给定消息的摘要。"""
        if not messages_to_summarize:
            return "No previous conversation history."

        trimmed_messages = self._trim_messages_for_summary(messages_to_summarize)
        if not trimmed_messages:
            return "Previous conversation was too long to summarize."

        # Format messages to avoid token inflation from metadata when str() is called on
        # 中文: 格式化消息以避免调用 str() 时元数据中的令牌膨胀
        # message objects
        # 中文: 消息对象
        formatted_messages = get_buffer_string(trimmed_messages)

        try:
            response = await self.model.ainvoke(
                self.summary_prompt.format(messages=formatted_messages)
            )
            return response.text.strip()
        except Exception as e:
            return f"Error generating summary: {e!s}"

    def _trim_messages_for_summary(self, messages: list[AnyMessage]) -> list[AnyMessage]:
        """Trim messages to fit within summary generation limits.

        中文翻译:
        修剪消息以适应摘要生成限制。"""
        try:
            if self.trim_tokens_to_summarize is None:
                return messages
            return cast(
                "list[AnyMessage]",
                trim_messages(
                    messages,
                    max_tokens=self.trim_tokens_to_summarize,
                    token_counter=self.token_counter,
                    start_on="human",
                    strategy="last",
                    allow_partial=True,
                    include_system=True,
                ),
            )
        except Exception:
            return messages[-_DEFAULT_FALLBACK_MESSAGE_COUNT:]
