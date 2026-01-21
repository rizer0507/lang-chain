"""工具选择中间件模块。

本模块提供使用 LLM 选择相关工具的能力，减少 token 使用。

核心类:
--------
**LLMToolSelectorMiddleware**: 工具选择中间件

功能特性:
---------
- 使用 LLM 预筛选相关工具
- 可配置最大工具数量
- 支持必选工具列表
- 减少主模型的 token 消耗

使用示例:
---------
>>> from langchain.agents import create_agent
>>> from langchain.agents.middleware import LLMToolSelectorMiddleware
>>>
>>> # 限制最多 3 个工具
>>> selector = LLMToolSelectorMiddleware(max_tools=3)
>>>
>>> # 使用更小的模型进行选择
>>> selector = LLMToolSelectorMiddleware(
...     model="openai:gpt-4o-mini",
...     max_tools=2,
... )
>>>
>>> agent = create_agent(
...     model="openai:gpt-4o",
...     tools=[tool1, tool2, tool3, tool4, tool5],
...     middleware=[selector],
... )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Literal, Union

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain.tools import BaseTool

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import Field, TypeAdapter
from typing_extensions import TypedDict

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)
from langchain.chat_models.base import init_chat_model

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "Your goal is to select the most relevant tools for answering the user's query."
)


@dataclass
class _SelectionRequest:
    """Prepared inputs for tool selection.

    中文翻译:
    为工具选择准备输入。"""

    available_tools: list[BaseTool]
    system_message: str
    last_user_message: HumanMessage
    model: BaseChatModel
    valid_tool_names: list[str]


def _create_tool_selection_response(tools: list[BaseTool]) -> TypeAdapter:
    """Create a structured output schema for tool selection.

    Args:
        tools: Available tools to include in the schema.

    Returns:
        `TypeAdapter` for a schema where each tool name is a `Literal` with its
            description.
    

    中文翻译:
    创建用于工具选择的结构化输出架构。
    参数：
        工具：要包含在架构中的可用工具。
    返回：
        用于模式的“TypeAdapter”，其中每个工具名称都是“Literal”及其
            描述。"""
    if not tools:
        msg = "Invalid usage: tools must be non-empty"
        raise AssertionError(msg)

    # Create a Union of Annotated Literal types for each tool name with description
    # 中文: 为每个工具名称和描述创建注释文字类型的联合
    # For instance: Union[Annotated[Literal["tool1"], Field(description="...")], ...]
    literals = [
        Annotated[Literal[tool.name], Field(description=tool.description)] for tool in tools
    ]
    selected_tool_type = Union[tuple(literals)]  # type: ignore[valid-type]  # noqa: UP007

    description = "Tools to use. Place the most relevant tools first."

    class ToolSelectionResponse(TypedDict):
        """Use to select relevant tools.

        中文翻译:
        用于选择相关工具。"""

        tools: Annotated[list[selected_tool_type], Field(description=description)]  # type: ignore[valid-type]

    return TypeAdapter(ToolSelectionResponse)


def _render_tool_list(tools: list[BaseTool]) -> str:
    """Format tools as markdown list.

    Args:
        tools: Tools to format.

    Returns:
        Markdown string with each tool on a new line.
    

    中文翻译:
    将工具格式化为降价列表。
    参数：
        工具：格式化工具。
    返回：
        每个工具的 Markdown 字符串都在新行上。"""
    return "\n".join(f"- {tool.name}: {tool.description}" for tool in tools)


class LLMToolSelectorMiddleware(AgentMiddleware):
    """Uses an LLM to select relevant tools before calling the main model.

    When an agent has many tools available, this middleware filters them down
    to only the most relevant ones for the user's query. This reduces token usage
    and helps the main model focus on the right tools.

    Examples:
        !!! example "Limit to 3 tools"

            ```python
            from langchain.agents.middleware import LLMToolSelectorMiddleware

            middleware = LLMToolSelectorMiddleware(max_tools=3)

            agent = create_agent(
                model="openai:gpt-4o",
                tools=[tool1, tool2, tool3, tool4, tool5],
                middleware=[middleware],
            )
            ```

        !!! example "Use a smaller model for selection"

            ```python
            middleware = LLMToolSelectorMiddleware(model="openai:gpt-4o-mini", max_tools=2)
            ```
    

    中文翻译:
    在调用主模型之前使用LLM选择相关工具。
    当代理有许多可用工具时，该中间件会过滤掉它们
    仅与用户的查询最相关的。这减少了令牌的使用
    并帮助主模型专注于正确的工具。
    示例：
        !!!示例“限制为 3 个工具”
            ````蟒蛇
            从 langchain.agents.middleware 导入 LLMToolSelectorMiddleware
            中间件 = LLMToolSelectorMiddleware(max_tools=3)
            代理=创建_代理（
                型号=“openai：gpt-4o”，
                工具=[工具1，工具2，工具3，工具4，工具5]，
                中间件=[中间件],
            ）
            ````
        !!!例如“使用较小的模型进行选择”
            ````蟒蛇
            中间件 = LLMToolSelectorMiddleware(model="openai:gpt-4o-mini", max_tools=2)
            ````"""

    def __init__(
        self,
        *,
        model: str | BaseChatModel | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_tools: int | None = None,
        always_include: list[str] | None = None,
    ) -> None:
        """Initialize the tool selector.

        Args:
            model: Model to use for selection.

                If not provided, uses the agent's main model.

                Can be a model identifier string or `BaseChatModel` instance.
            system_prompt: Instructions for the selection model.
            max_tools: Maximum number of tools to select.

                If the model selects more, only the first `max_tools` will be used.

                If not specified, there is no limit.
            always_include: Tool names to always include regardless of selection.

                These do not count against the `max_tools` limit.
        

        中文翻译:
        初始化工具选择器。
        参数：
            model：用于选择的模型。
                如果未提供，则使用代理的主模型。
                可以是模型标识符字符串或“BaseChatModel”实例。
            system_prompt：选择型号的说明。
            max_tools：要选择的最大工具数。
                如果模型选择更多，则仅使用第一个“max_tools”。
                如果没有指定，则没有限制。
            always_include：无论选择如何，始终包含的工具名称。
                这些不计入“max_tools”限制。"""
        super().__init__()
        self.system_prompt = system_prompt
        self.max_tools = max_tools
        self.always_include = always_include or []

        if isinstance(model, (BaseChatModel, type(None))):
            self.model: BaseChatModel | None = model
        else:
            self.model = init_chat_model(model)

    def _prepare_selection_request(self, request: ModelRequest) -> _SelectionRequest | None:
        """Prepare inputs for tool selection.

        Returns:
            `SelectionRequest` with prepared inputs, or `None` if no selection is
                needed.
        

        中文翻译:
        准备工具选择的输入。
        返回：
            带有准备好的输入的“SelectionRequest”，如果没有选择，则为“None”
                需要。"""
        # If no tools available, return None
        # 中文: 如果没有可用的工具，则返回 None
        if not request.tools or len(request.tools) == 0:
            return None

        # Filter to only BaseTool instances (exclude provider-specific tool dicts)
        # 中文: 仅过滤到 BaseTool 实例（排除特定于提供者的工具字典）
        base_tools = [tool for tool in request.tools if not isinstance(tool, dict)]

        # Validate that always_include tools exist
        # 中文: 验证always_include工具是否存在
        if self.always_include:
            available_tool_names = {tool.name for tool in base_tools}
            missing_tools = [
                name for name in self.always_include if name not in available_tool_names
            ]
            if missing_tools:
                msg = (
                    f"Tools in always_include not found in request: {missing_tools}. "
                    f"Available tools: {sorted(available_tool_names)}"
                )
                raise ValueError(msg)

        # Separate tools that are always included from those available for selection
        # 中文: 始终包含在可供选择的工具中的单独工具
        available_tools = [tool for tool in base_tools if tool.name not in self.always_include]

        # If no tools available for selection, return None
        # 中文: 如果没有可供选择的工具，则返回 None
        if not available_tools:
            return None

        system_message = self.system_prompt
        # If there's a max_tools limit, append instructions to the system prompt
        # 中文: 如果有 max_tools 限制，请将说明附加到系统提示符中
        if self.max_tools is not None:
            system_message += (
                f"\nIMPORTANT: List the tool names in order of relevance, "
                f"with the most relevant first. "
                f"If you exceed the maximum number of tools, "
                f"only the first {self.max_tools} will be used."
            )

        # Get the last user message from the conversation history
        # 中文: 从对话历史记录中获取最后一条用户消息
        last_user_message: HumanMessage
        for message in reversed(request.messages):
            if isinstance(message, HumanMessage):
                last_user_message = message
                break
        else:
            msg = "No user message found in request messages"
            raise AssertionError(msg)

        model = self.model or request.model
        valid_tool_names = [tool.name for tool in available_tools]

        return _SelectionRequest(
            available_tools=available_tools,
            system_message=system_message,
            last_user_message=last_user_message,
            model=model,
            valid_tool_names=valid_tool_names,
        )

    def _process_selection_response(
        self,
        response: dict,
        available_tools: list[BaseTool],
        valid_tool_names: list[str],
        request: ModelRequest,
    ) -> ModelRequest:
        """Process the selection response and return filtered `ModelRequest`.

        中文翻译:
        处理选择响应并返回过滤后的“ModelRequest”。"""
        selected_tool_names: list[str] = []
        invalid_tool_selections = []

        for tool_name in response["tools"]:
            if tool_name not in valid_tool_names:
                invalid_tool_selections.append(tool_name)
                continue

            # Only add if not already selected and within max_tools limit
            # 中文: 仅在尚未选择且在 max_tools 限制内添加
            if tool_name not in selected_tool_names and (
                self.max_tools is None or len(selected_tool_names) < self.max_tools
            ):
                selected_tool_names.append(tool_name)

        if invalid_tool_selections:
            msg = f"Model selected invalid tools: {invalid_tool_selections}"
            raise ValueError(msg)

        # Filter tools based on selection and append always-included tools
        # 中文: 根据选择过滤工具并附加始终包含的工具
        selected_tools: list[BaseTool] = [
            tool for tool in available_tools if tool.name in selected_tool_names
        ]
        always_included_tools: list[BaseTool] = [
            tool
            for tool in request.tools
            if not isinstance(tool, dict) and tool.name in self.always_include
        ]
        selected_tools.extend(always_included_tools)

        # Also preserve any provider-specific tool dicts from the original request
        # 中文: 还保留原始请求中任何特定于提供者的工具指令
        provider_tools = [tool for tool in request.tools if isinstance(tool, dict)]

        return request.override(tools=[*selected_tools, *provider_tools])

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Filter tools based on LLM selection before invoking the model via handler.

        中文翻译:
        在通过处理程序调用模型之前，根据 LLM 选择过滤工具。"""
        selection_request = self._prepare_selection_request(request)
        if selection_request is None:
            return handler(request)

        # Create dynamic response model with Literal enum of available tool names
        # 中文: 使用可用工具名称的文字枚举创建动态响应模型
        type_adapter = _create_tool_selection_response(selection_request.available_tools)
        schema = type_adapter.json_schema()
        structured_model = selection_request.model.with_structured_output(schema)

        response = structured_model.invoke(
            [
                {"role": "system", "content": selection_request.system_message},
                selection_request.last_user_message,
            ]
        )

        # Response should be a dict since we're passing a schema (not a Pydantic model class)
        # 中文: 响应应该是一个字典，因为我们传递的是一个模式（不是 Pydantic 模型类）
        if not isinstance(response, dict):
            msg = f"Expected dict response, got {type(response)}"
            raise AssertionError(msg)  # noqa: TRY004
        modified_request = self._process_selection_response(
            response, selection_request.available_tools, selection_request.valid_tool_names, request
        )
        return handler(modified_request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Filter tools based on LLM selection before invoking the model via handler.

        中文翻译:
        在通过处理程序调用模型之前，根据 LLM 选择过滤工具。"""
        selection_request = self._prepare_selection_request(request)
        if selection_request is None:
            return await handler(request)

        # Create dynamic response model with Literal enum of available tool names
        # 中文: 使用可用工具名称的文字枚举创建动态响应模型
        type_adapter = _create_tool_selection_response(selection_request.available_tools)
        schema = type_adapter.json_schema()
        structured_model = selection_request.model.with_structured_output(schema)

        response = await structured_model.ainvoke(
            [
                {"role": "system", "content": selection_request.system_message},
                selection_request.last_user_message,
            ]
        )

        # Response should be a dict since we're passing a schema (not a Pydantic model class)
        # 中文: 响应应该是一个字典，因为我们传递的是一个模式（不是 Pydantic 模型类）
        if not isinstance(response, dict):
            msg = f"Expected dict response, got {type(response)}"
            raise AssertionError(msg)  # noqa: TRY004
        modified_request = self._process_selection_response(
            response, selection_request.available_tools, selection_request.valid_tool_names, request
        )
        return await handler(modified_request)
