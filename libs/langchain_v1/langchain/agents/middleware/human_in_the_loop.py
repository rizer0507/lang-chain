"""人工介入（Human-in-the-Loop）中间件模块。

本模块提供在 Agent 执行过程中请求人工审批的能力。

核心类:
--------
**HumanInTheLoopMiddleware**: 人工介入中间件

功能特性:
---------
- 在工具执行前请求人工审批
- 支持批准、编辑、拒绝三种决策
- 可配置每个工具的审批策略
- 支持自定义描述生成

使用示例:
---------
>>> from langchain.agents import create_agent
>>> from langchain.agents.middleware import HumanInTheLoopMiddleware
>>>
>>> hitl = HumanInTheLoopMiddleware(
...     interrupt_on={
...         "delete_file": True,  # 所有决策都允许
...         "send_email": {"allowed_decisions": ["approve", "reject"]},
...     }
... )
>>>
>>> agent = create_agent(
...     model="openai:gpt-4o",
...     tools=[delete_file, send_email],
...     middleware=[hitl],
... )
"""

from typing import Any, Literal, Protocol

from langchain_core.messages import AIMessage, ToolCall, ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import interrupt
from typing_extensions import NotRequired, TypedDict

from langchain.agents.middleware.types import AgentMiddleware, AgentState, ContextT, StateT


class Action(TypedDict):
    """Represents an action with a name and args.

    中文翻译:
    表示具有名称和参数的操作。"""

    name: str
    """The type or name of action being requested (e.g., `'add_numbers'`).

    中文翻译:
    请求的操作的类型或名称（例如“add_numbers”）。"""

    args: dict[str, Any]
    """Key-value pairs of args needed for the action (e.g., `{"a": 1, "b": 2}`).

    中文翻译:
    操作所需的键值对参数（例如，`{"a": 1, "b": 2}`）。"""


class ActionRequest(TypedDict):
    """Represents an action request with a name, args, and description.

    中文翻译:
    表示带有名称、参数和描述的操作请求。"""

    name: str
    """The name of the action being requested.

    中文翻译:
    所请求的操作的名称。"""

    args: dict[str, Any]
    """Key-value pairs of args needed for the action (e.g., `{"a": 1, "b": 2}`).

    中文翻译:
    操作所需的键值对参数（例如，`{"a": 1, "b": 2}`）。"""

    description: NotRequired[str]
    """The description of the action to be reviewed.

    中文翻译:
    要审查的操作的描述。"""


DecisionType = Literal["approve", "edit", "reject"]


class ReviewConfig(TypedDict):
    """Policy for reviewing a HITL request.

    中文翻译:
    审查 HITL 请求的政策。"""

    action_name: str
    """Name of the action associated with this review configuration.

    中文翻译:
    与此审核配置关联的操作的名称。"""

    allowed_decisions: list[DecisionType]
    """The decisions that are allowed for this request.

    中文翻译:
    允许此请求的决定。"""

    args_schema: NotRequired[dict[str, Any]]
    """JSON schema for the args associated with the action, if edits are allowed.

    中文翻译:
    与操作关联的参数的 JSON 架构（如果允许编辑）。"""


class HITLRequest(TypedDict):
    """Request for human feedback on a sequence of actions requested by a model.

    中文翻译:
    请求对模型请求的一系列操作进行人工反馈。"""

    action_requests: list[ActionRequest]
    """A list of agent actions for human review.

    中文翻译:
    供人工审核的代理操作列表。"""

    review_configs: list[ReviewConfig]
    """Review configuration for all possible actions.

    中文翻译:
    检查所有可能操作的配置。"""


class ApproveDecision(TypedDict):
    """Response when a human approves the action.

    中文翻译:
    当人类批准该操作时的响应。"""

    type: Literal["approve"]
    """The type of response when a human approves the action.

    中文翻译:
    人类批准该操作时的响应类型。"""


class EditDecision(TypedDict):
    """Response when a human edits the action.

    中文翻译:
    当人类编辑操作时的响应。"""

    type: Literal["edit"]
    """The type of response when a human edits the action.

    中文翻译:
    人类编辑操作时的响应类型。"""

    edited_action: Action
    """Edited action for the agent to perform.

    Ex: for a tool call, a human reviewer can edit the tool name and args.
    

    中文翻译:
    已编辑代理要执行的操作。
    例如：对于工具调用，人工审阅者可以编辑工具名称和参数。"""


class RejectDecision(TypedDict):
    """Response when a human rejects the action.

    中文翻译:
    当人类拒绝该动作时的反应。"""

    type: Literal["reject"]
    """The type of response when a human rejects the action.

    中文翻译:
    当人类拒绝某个动作时的反应类型。"""

    message: NotRequired[str]
    """The message sent to the model explaining why the action was rejected.

    中文翻译:
    发送到模型的消息解释了操作被拒绝的原因。"""


Decision = ApproveDecision | EditDecision | RejectDecision


class HITLResponse(TypedDict):
    """Response payload for a HITLRequest.

    中文翻译:
    HITLRequest 的响应负载。"""

    decisions: list[Decision]
    """The decisions made by the human.

    中文翻译:
    由人类做出的决定。"""


class _DescriptionFactory(Protocol):
    """Callable that generates a description for a tool call.

    中文翻译:
    可调用，生成工具调用的描述。"""

    def __call__(self, tool_call: ToolCall, state: AgentState, runtime: Runtime[ContextT]) -> str:
        """Generate a description for a tool call.

        中文翻译:
        生成工具调用的描述。"""
        ...


class InterruptOnConfig(TypedDict):
    """Configuration for an action requiring human in the loop.

    This is the configuration format used in the `HumanInTheLoopMiddleware.__init__`
    method.
    

    中文翻译:
    需要人工参与循环的操作的配置。
    这是`HumanInTheLoopMiddleware.__init__`中使用的配置格式
    方法。"""

    allowed_decisions: list[DecisionType]
    """The decisions that are allowed for this action.

    中文翻译:
    允许执行此操作的决策。"""

    description: NotRequired[str | _DescriptionFactory]
    """The description attached to the request for human input.

    Can be either:

    - A static string describing the approval request
    - A callable that dynamically generates the description based on agent state,
        runtime, and tool call information

    Example:
        ```python
        # Static string description
        # 中文: 静态字符串描述
        config = ToolConfig(
            allowed_decisions=["approve", "reject"],
            description="Please review this tool execution"
        )

        # Dynamic callable description
        # 中文: 动态可调用描述
        def format_tool_description(
            tool_call: ToolCall,
            state: AgentState,
            runtime: Runtime[ContextT]
        ) -> str:
            import json
            return (
                f"Tool: {tool_call['name']}\\n"
                f"Arguments:\\n{json.dumps(tool_call['args'], indent=2)}"
            )

        config = InterruptOnConfig(
            allowed_decisions=["approve", "edit", "reject"],
            description=format_tool_description
        )
        ```
    

    中文翻译:
    附加到人工输入请求的描述。
    可以是：
    - 描述批准请求的静态字符串
    - 根据代理状态动态生成描述的可调用项，
        运行时和工具调用信息
    示例：
        ````蟒蛇
        # 静态字符串描述
        配置 = 工具配置(
            allowed_decisions=["批准", "拒绝"],
            description="请检查此工具的执行情况"
        ）
        # 动态可调用描述
        def format_tool_description(
            tool_call：工具调用，
            状态：代理状态，
            运行时：运行时[ContextT]
        ) -> 字符串:
            导入 json
            返回（
                f"工具：{tool_call['name']}\\n"
                f"参数：\\n{json.dumps(tool_call['args'], indent=2)}"
            ）
        配置 = 中断配置（
            allowed_decisions=["批准", "编辑", "拒绝"],
            描述=格式工具描述
        ）
        ````"""
    args_schema: NotRequired[dict[str, Any]]
    """JSON schema for the args associated with the action, if edits are allowed.

    中文翻译:
    与操作关联的参数的 JSON 架构（如果允许编辑）。"""


class HumanInTheLoopMiddleware(AgentMiddleware[StateT, ContextT]):
    """Human in the loop middleware.

    中文翻译:
    人在循环中间件。"""

    def __init__(
        self,
        interrupt_on: dict[str, bool | InterruptOnConfig],
        *,
        description_prefix: str = "Tool execution requires approval",
    ) -> None:
        """Initialize the human in the loop middleware.

        Args:
            interrupt_on: Mapping of tool name to allowed actions.

                If a tool doesn't have an entry, it's auto-approved by default.

                * `True` indicates all decisions are allowed: approve, edit, and reject.
                * `False` indicates that the tool is auto-approved.
                * `InterruptOnConfig` indicates the specific decisions allowed for this
                    tool.

                    The `InterruptOnConfig` can include a `description` field (`str` or
                    `Callable`) for custom formatting of the interrupt description.
            description_prefix: The prefix to use when constructing action requests.

                This is used to provide context about the tool call and the action being
                requested.

                Not used if a tool has a `description` in its `InterruptOnConfig`.
        

        中文翻译:
        初始化人机循环中间件。
        参数：
            Interrupt_on：工具名称到允许操作的映射。
                如果工具没有条目，则默认情况下会自动批准。
                * “True”表示允许所有决定：批准、编辑和拒绝。
                * “False”表示该工具是自动批准的。
                * `InterruptOnConfig` 表示为此允许的具体决定
                    工具。
                    `InterruptOnConfig` 可以包含一个 `description` 字段（`str` 或
                    `Callable`) 用于中断描述的自定义格式。
            description_prefix：构造操作请求时使用的前缀。
                这用于提供有关工具调用和操作的上下文
                要求。
                如果工具的“InterruptOnConfig”中有“描述”，则不使用。"""
        super().__init__()
        resolved_configs: dict[str, InterruptOnConfig] = {}
        for tool_name, tool_config in interrupt_on.items():
            if isinstance(tool_config, bool):
                if tool_config is True:
                    resolved_configs[tool_name] = InterruptOnConfig(
                        allowed_decisions=["approve", "edit", "reject"]
                    )
            elif tool_config.get("allowed_decisions"):
                resolved_configs[tool_name] = tool_config
        self.interrupt_on = resolved_configs
        self.description_prefix = description_prefix

    def _create_action_and_config(
        self,
        tool_call: ToolCall,
        config: InterruptOnConfig,
        state: AgentState,
        runtime: Runtime[ContextT],
    ) -> tuple[ActionRequest, ReviewConfig]:
        """Create an ActionRequest and ReviewConfig for a tool call.

        中文翻译:
        创建用于工具调用的 ActionRequest 和 ReviewConfig。"""
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        # Generate description using the description field (str or callable)
        # 中文: 使用描述字段（str 或 callable）生成描述
        description_value = config.get("description")
        if callable(description_value):
            description = description_value(tool_call, state, runtime)
        elif description_value is not None:
            description = description_value
        else:
            description = f"{self.description_prefix}\n\nTool: {tool_name}\nArgs: {tool_args}"

        # Create ActionRequest with description
        # 中文: 创建带有描述的 ActionRequest
        action_request = ActionRequest(
            name=tool_name,
            args=tool_args,
            description=description,
        )

        # Create ReviewConfig
        # 中文: 创建ReviewConfig
        # eventually can get tool information and populate args_schema from there
        # 中文: 最终可以从那里获取工具信息并填充args_schema
        review_config = ReviewConfig(
            action_name=tool_name,
            allowed_decisions=config["allowed_decisions"],
        )

        return action_request, review_config

    def _process_decision(
        self,
        decision: Decision,
        tool_call: ToolCall,
        config: InterruptOnConfig,
    ) -> tuple[ToolCall | None, ToolMessage | None]:
        """Process a single decision and return the revised tool call and optional tool message.

        中文翻译:
        处理单个决策并返回修改后的工具调用和可选工具消息。"""
        allowed_decisions = config["allowed_decisions"]

        if decision["type"] == "approve" and "approve" in allowed_decisions:
            return tool_call, None
        if decision["type"] == "edit" and "edit" in allowed_decisions:
            edited_action = decision["edited_action"]
            return (
                ToolCall(
                    type="tool_call",
                    name=edited_action["name"],
                    args=edited_action["args"],
                    id=tool_call["id"],
                ),
                None,
            )
        if decision["type"] == "reject" and "reject" in allowed_decisions:
            # Create a tool message with the human's text response
            # 中文: 使用人类的文本响应创建工具消息
            content = decision.get("message") or (
                f"User rejected the tool call for `{tool_call['name']}` with id {tool_call['id']}"
            )
            tool_message = ToolMessage(
                content=content,
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
                status="error",
            )
            return tool_call, tool_message
        msg = (
            f"Unexpected human decision: {decision}. "
            f"Decision type '{decision.get('type')}' "
            f"is not allowed for tool '{tool_call['name']}'. "
            f"Expected one of {allowed_decisions} based on the tool's configuration."
        )
        raise ValueError(msg)

    def after_model(self, state: AgentState, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        """Trigger interrupt flows for relevant tool calls after an `AIMessage`.

        中文翻译:
        在“AIMessage”之后触发相关工具调用的中断流。"""
        messages = state["messages"]
        if not messages:
            return None

        last_ai_msg = next((msg for msg in reversed(messages) if isinstance(msg, AIMessage)), None)
        if not last_ai_msg or not last_ai_msg.tool_calls:
            return None

        # Create action requests and review configs for tools that need approval
        # 中文: 创建操作请求并审查需要批准的工具的配置
        action_requests: list[ActionRequest] = []
        review_configs: list[ReviewConfig] = []
        interrupt_indices: list[int] = []

        for idx, tool_call in enumerate(last_ai_msg.tool_calls):
            if (config := self.interrupt_on.get(tool_call["name"])) is not None:
                action_request, review_config = self._create_action_and_config(
                    tool_call, config, state, runtime
                )
                action_requests.append(action_request)
                review_configs.append(review_config)
                interrupt_indices.append(idx)

        # If no interrupts needed, return early
        # 中文: 如果不需要中断，则尽早返回
        if not action_requests:
            return None

        # Create single HITLRequest with all actions and configs
        # 中文: 使用所有操作和配置创建单个 HITLRequest
        hitl_request = HITLRequest(
            action_requests=action_requests,
            review_configs=review_configs,
        )

        # Send interrupt and get response
        # 中文: 发送中断并获取响应
        decisions = interrupt(hitl_request)["decisions"]

        # Validate that the number of decisions matches the number of interrupt tool calls
        # 中文: 验证决策数量与中断工具调用数量相匹配
        if (decisions_len := len(decisions)) != (interrupt_count := len(interrupt_indices)):
            msg = (
                f"Number of human decisions ({decisions_len}) does not match "
                f"number of hanging tool calls ({interrupt_count})."
            )
            raise ValueError(msg)

        # Process decisions and rebuild tool calls in original order
        # 中文: 按原始顺序处理决策并重建工具调用
        revised_tool_calls: list[ToolCall] = []
        artificial_tool_messages: list[ToolMessage] = []
        decision_idx = 0

        for idx, tool_call in enumerate(last_ai_msg.tool_calls):
            if idx in interrupt_indices:
                # This was an interrupt tool call - process the decision
                # 中文: 这是一个中断工具调用 - 处理决定
                config = self.interrupt_on[tool_call["name"]]
                decision = decisions[decision_idx]
                decision_idx += 1

                revised_tool_call, tool_message = self._process_decision(
                    decision, tool_call, config
                )
                if revised_tool_call is not None:
                    revised_tool_calls.append(revised_tool_call)
                if tool_message:
                    artificial_tool_messages.append(tool_message)
            else:
                # This was auto-approved - keep original
                # 中文: 这是自动批准的 - 保持原始状态
                revised_tool_calls.append(tool_call)

        # Update the AI message to only include approved tool calls
        # 中文: 更新 AI 消息以仅包含批准的工具调用
        last_ai_msg.tool_calls = revised_tool_calls

        return {"messages": [last_ai_msg, *artificial_tool_messages]}

    async def aafter_model(
        self, state: AgentState, runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        """Async trigger interrupt flows for relevant tool calls after an `AIMessage`.

        中文翻译:
        “AIMessage”之后相关工具调用的异步触发中断流。"""
        return self.after_model(state, runtime)
