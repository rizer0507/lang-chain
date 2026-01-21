"""工具模拟中间件模块。

本模块提供使用 LLM 模拟工具执行的能力，用于测试目的。

核心类:
--------
**LLMToolEmulator**: 工具模拟中间件

功能特性:
---------
- 使用 LLM 生成模拟工具响应
- 可选择模拟全部或特定工具
- 支持自定义模拟模型

使用示例:
---------
>>> from langchain.agents import create_agent
>>> from langchain.agents.middleware import LLMToolEmulator
>>>
>>> # 模拟所有工具（默认）
>>> emulator = LLMToolEmulator()
>>>
>>> # 仅模拟特定工具
>>> emulator = LLMToolEmulator(tools=["get_weather"])
>>>
>>> # 使用自定义模型
>>> emulator = LLMToolEmulator(model="openai:gpt-4o-mini")
>>>
>>> agent = create_agent(
...     model="openai:gpt-4o",
...     tools=[get_weather, calculator],
...     middleware=[emulator],
... )
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, ToolMessage

from langchain.agents.middleware.types import AgentMiddleware
from langchain.chat_models.base import init_chat_model

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langgraph.types import Command

    from langchain.agents.middleware.types import ToolCallRequest
    from langchain.tools import BaseTool


class LLMToolEmulator(AgentMiddleware):
    """Emulates specified tools using an LLM instead of executing them.

    This middleware allows selective emulation of tools for testing purposes.

    By default (when `tools=None`), all tools are emulated. You can specify which
    tools to emulate by passing a list of tool names or `BaseTool` instances.

    Examples:
        !!! example "Emulate all tools (default behavior)"

            ```python
            from langchain.agents.middleware import LLMToolEmulator

            middleware = LLMToolEmulator()

            agent = create_agent(
                model="openai:gpt-4o",
                tools=[get_weather, get_user_location, calculator],
                middleware=[middleware],
            )
            ```

        !!! example "Emulate specific tools by name"

            ```python
            middleware = LLMToolEmulator(tools=["get_weather", "get_user_location"])
            ```

        !!! example "Use a custom model for emulation"

            ```python
            middleware = LLMToolEmulator(
                tools=["get_weather"], model="anthropic:claude-sonnet-4-5-20250929"
            )
            ```

        !!! example "Emulate specific tools by passing tool instances"

            ```python
            middleware = LLMToolEmulator(tools=[get_weather, get_user_location])
            ```
    

    中文翻译:
    使用 LLM 模拟指定的工具，而不是执行它们。
    该中间件允许选择性地模拟工具以进行测试。
    默认情况下（当“tools=None”时），所有工具都会被模拟。您可以指定哪个
    通过传递工具名称或“BaseTool”实例的列表来模拟的工具。
    示例：
        !!!示例“模拟所有工具（默认行为）”
            ````蟒蛇
            从 langchain.agents.middleware 导入 LLMToolEmulator
            中间件 = LLMToolEmulator()
            代理=创建_代理（
                型号=“openai：gpt-4o”，
                工具=[获取天气、获取用户位置、计算器]、
                中间件=[中间件],
            ）
            ````
        !!!示例“按名称模拟特定工具”
            ````蟒蛇
            中间件 = LLMToolEmulator(工具=["get_weather", "get_user_location"])
            ````
        !!!示例“使用自定义模型进行仿真”
            ````蟒蛇
            中间件 = LLMToolEmulator(
                工具=[“get_weather”]，模型=“人类：claude-sonnet-4-5-20250929”
            ）
            ````
        !!!示例“通过传递工具实例来模拟特定工具”
            ````蟒蛇
            中间件 = LLMToolEmulator(工具=[get_weather, get_user_location])
            ````"""

    def __init__(
        self,
        *,
        tools: list[str | BaseTool] | None = None,
        model: str | BaseChatModel | None = None,
    ) -> None:
        """Initialize the tool emulator.

        Args:
            tools: List of tool names (`str`) or `BaseTool` instances to emulate.

                If `None`, ALL tools will be emulated.

                If empty list, no tools will be emulated.
            model: Model to use for emulation.

                Defaults to `'anthropic:claude-sonnet-4-5-20250929'`.

                Can be a model identifier string or `BaseChatModel` instance.
        

        中文翻译:
        初始化工具模拟器。
        参数：
            工具：要模拟的工具名称（“str”）或“BaseTool”实例的列表。
                如果为“无”，则将模拟所有工具。
                如果列表为空，则不会模拟任何工具。
            model：用于仿真的模型。
                默认为“anthropic:claude-sonnet-4-5-20250929”。
                可以是模型标识符字符串或“BaseChatModel”实例。"""
        super().__init__()

        # Extract tool names from tools
        # 中文: 从工具中提取工具名称
        # None means emulate all tools
        # 中文: None 表示模拟所有工具
        self.emulate_all = tools is None
        self.tools_to_emulate: set[str] = set()

        if not self.emulate_all and tools is not None:
            for tool in tools:
                if isinstance(tool, str):
                    self.tools_to_emulate.add(tool)
                else:
                    # Assume BaseTool with .name attribute
                    # 中文: 假设 BaseTool 具有 .name 属性
                    self.tools_to_emulate.add(tool.name)

        # Initialize emulator model
        # 中文: 初始化模拟器模型
        if model is None:
            self.model = init_chat_model("anthropic:claude-sonnet-4-5-20250929", temperature=1)
        elif isinstance(model, BaseChatModel):
            self.model = model
        else:
            self.model = init_chat_model(model, temperature=1)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Emulate tool execution using LLM if tool should be emulated.

        Args:
            request: Tool call request to potentially emulate.
            handler: Callback to execute the tool (can be called multiple times).

        Returns:
            ToolMessage with emulated response if tool should be emulated,
                otherwise calls handler for normal execution.
        

        中文翻译:
        如果应模拟工具，请使用 LLM 模拟工具执行。
        参数：
            request：要模拟的工具调用请求。
            handler：执行工具的回调（可以多次调用）。
        返回：
            如果应模拟工具，则具有模拟响应的 ToolMessage，
                否则调用处理程序以正常执行。"""
        tool_name = request.tool_call["name"]

        # Check if this tool should be emulated
        # 中文: 检查是否应该模拟该工具
        should_emulate = self.emulate_all or tool_name in self.tools_to_emulate

        if not should_emulate:
            # Let it execute normally by calling the handler
            # 中文: 通过调用handler让它正常执行
            return handler(request)

        # Extract tool information for emulation
        # 中文: 提取仿真工具信息
        tool_args = request.tool_call["args"]
        tool_description = request.tool.description if request.tool else "No description available"

        # Build prompt for emulator LLM
        # 中文: 模拟器 LLM 的构建提示
        prompt = (
            f"You are emulating a tool call for testing purposes.\n\n"
            f"Tool: {tool_name}\n"
            f"Description: {tool_description}\n"
            f"Arguments: {tool_args}\n\n"
            f"Generate a realistic response that this tool would return "
            f"given these arguments.\n"
            f"Return ONLY the tool's output, no explanation or preamble. "
            f"Introduce variation into your responses."
        )

        # Get emulated response from LLM
        # 中文: 获取 LLM 的模拟响应
        response = self.model.invoke([HumanMessage(prompt)])

        # Short-circuit: return emulated result without executing real tool
        # 中文: 短路：返回模拟结果而不执行真实工具
        return ToolMessage(
            content=response.content,
            tool_call_id=request.tool_call["id"],
            name=tool_name,
        )

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Async version of `wrap_tool_call`.

        Emulate tool execution using LLM if tool should be emulated.

        Args:
            request: Tool call request to potentially emulate.
            handler: Async callback to execute the tool (can be called multiple times).

        Returns:
            ToolMessage with emulated response if tool should be emulated,
                otherwise calls handler for normal execution.
        

        中文翻译:
        `wrap_tool_call` 的异步版本。
        如果应模拟工具，请使用 LLM 模拟工具执行。
        参数：
            request：要模拟的工具调用请求。
            handler：执行工具的异步回调（可以调用多次）。
        返回：
            如果应模拟工具，则具有模拟响应的 ToolMessage，
                否则调用处理程序以正常执行。"""
        tool_name = request.tool_call["name"]

        # Check if this tool should be emulated
        # 中文: 检查是否应该模拟该工具
        should_emulate = self.emulate_all or tool_name in self.tools_to_emulate

        if not should_emulate:
            # Let it execute normally by calling the handler
            # 中文: 通过调用handler让它正常执行
            return await handler(request)

        # Extract tool information for emulation
        # 中文: 提取仿真工具信息
        tool_args = request.tool_call["args"]
        tool_description = request.tool.description if request.tool else "No description available"

        # Build prompt for emulator LLM
        # 中文: 模拟器 LLM 的构建提示
        prompt = (
            f"You are emulating a tool call for testing purposes.\n\n"
            f"Tool: {tool_name}\n"
            f"Description: {tool_description}\n"
            f"Arguments: {tool_args}\n\n"
            f"Generate a realistic response that this tool would return "
            f"given these arguments.\n"
            f"Return ONLY the tool's output, no explanation or preamble. "
            f"Introduce variation into your responses."
        )

        # Get emulated response from LLM (using async invoke)
        # 中文: 从 LLM 获取模拟响应（使用异步调用）
        response = await self.model.ainvoke([HumanMessage(prompt)])

        # Short-circuit: return emulated result without executing real tool
        # 中文: 短路：返回模拟结果而不执行真实工具
        return ToolMessage(
            content=response.content,
            tool_call_id=request.tool_call["id"],
            name=tool_name,
        )
