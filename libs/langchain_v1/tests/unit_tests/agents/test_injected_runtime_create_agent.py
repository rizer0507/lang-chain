"""Test ToolRuntime injection with create_agent.

This module tests the injected runtime functionality when using tools
with the create_agent factory. The ToolRuntime provides tools access to:
- state: Current graph state
- tool_call_id: ID of the current tool call
- config: RunnableConfig for the execution
- context: Runtime context from LangGraph
- store: BaseStore for persistent storage
- stream_writer: For streaming custom output

These tests verify that runtime injection works correctly across both
sync and async execution paths, with middleware, and in various agent
configurations.

中文翻译:
使用 create_agent 测试 ToolRuntime 注入。
该模块测试使用工具时注入的运行时功能
与 create_agent 工厂。 ToolRuntime 提供工具访问：
- state：当前图形状态
- tool_call_id：当前工具调用的ID
- config：用于执行的RunnableConfig
- context：来自 LangGraph 的运行时上下文
- store：用于持久存储的BaseStore
-stream_writer：用于流式自定义输出
这些测试验证运行时注入在两者之间都能正常工作
同步和异步执行路径，带有中间件，以及在各种代理中
配置。
"""

from __future__ import annotations

from typing import Annotated, Any

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedStore
from langgraph.store.memory import InMemoryStore

from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentMiddleware, AgentState
from langchain.tools import InjectedState, ToolRuntime

from .model import FakeToolCallingModel


def test_tool_runtime_basic_injection() -> None:
    """Test basic ToolRuntime injection in tools with create_agent.

    中文翻译:
    使用 create_agent 在工具中测试基本 ToolRuntime 注入。"""
    # Track what was injected
    # 中文: 跟踪注入的内容
    injected_data = {}

    @tool
    def runtime_tool(x: int, runtime: ToolRuntime) -> str:
        """Tool that accesses runtime context.

        中文翻译:
        访问运行时上下文的工具。"""
        injected_data["state"] = runtime.state
        injected_data["tool_call_id"] = runtime.tool_call_id
        injected_data["config"] = runtime.config
        injected_data["context"] = runtime.context
        injected_data["store"] = runtime.store
        injected_data["stream_writer"] = runtime.stream_writer
        return f"Processed {x}"

    assert runtime_tool.args

    agent = create_agent(
        model=FakeToolCallingModel(
            tool_calls=[
                [{"args": {"x": 42}, "id": "call_123", "name": "runtime_tool"}],
                [],
            ]
        ),
        tools=[runtime_tool],
        system_prompt="You are a helpful assistant.",
    )

    result = agent.invoke({"messages": [HumanMessage("Test")]})

    # Verify tool executed
    # 中文: 验证工具已执行
    assert len(result["messages"]) == 4
    tool_message = result["messages"][2]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.content == "Processed 42"
    assert tool_message.tool_call_id == "call_123"

    # Verify runtime was injected
    # 中文: 验证运行时是否已注入
    assert injected_data["state"] is not None
    assert "messages" in injected_data["state"]
    assert injected_data["tool_call_id"] == "call_123"
    assert injected_data["config"] is not None
    # Context, store, stream_writer may be None depending on graph setup
    # 中文: Context、store、stream_writer 可能为 None，具体取决于图形设置
    assert "context" in injected_data
    assert "store" in injected_data
    assert "stream_writer" in injected_data


async def test_tool_runtime_async_injection() -> None:
    """Test ToolRuntime injection works with async tools.

    中文翻译:
    测试 ToolRuntime 注入可与异步工具配合使用。"""
    injected_data = {}

    @tool
    async def async_runtime_tool(x: int, runtime: ToolRuntime) -> str:
        """Async tool that accesses runtime context.

        中文翻译:
        访问运行时上下文的异步工具。"""
        injected_data["state"] = runtime.state
        injected_data["tool_call_id"] = runtime.tool_call_id
        injected_data["config"] = runtime.config
        return f"Async processed {x}"

    agent = create_agent(
        model=FakeToolCallingModel(
            tool_calls=[
                [{"args": {"x": 99}, "id": "async_call_456", "name": "async_runtime_tool"}],
                [],
            ]
        ),
        tools=[async_runtime_tool],
        system_prompt="You are a helpful assistant.",
    )

    result = await agent.ainvoke({"messages": [HumanMessage("Test async")]})

    # Verify tool executed
    # 中文: 验证工具已执行
    assert len(result["messages"]) == 4
    tool_message = result["messages"][2]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.content == "Async processed 99"
    assert tool_message.tool_call_id == "async_call_456"

    # Verify runtime was injected
    # 中文: 验证运行时是否已注入
    assert injected_data["state"] is not None
    assert "messages" in injected_data["state"]
    assert injected_data["tool_call_id"] == "async_call_456"
    assert injected_data["config"] is not None


def test_tool_runtime_state_access() -> None:
    """Test that tools can access and use state via ToolRuntime.

    中文翻译:
    测试工具是否可以通过 ToolRuntime 访问和使用状态。"""

    @tool
    def state_aware_tool(query: str, runtime: ToolRuntime) -> str:
        """Tool that uses state to provide context-aware responses.

        中文翻译:
        使用状态提供上下文感知响应的工具。"""
        messages = runtime.state.get("messages", [])
        msg_count = len(messages)
        return f"Query: {query}, Message count: {msg_count}"

    agent = create_agent(
        model=FakeToolCallingModel(
            tool_calls=[
                [{"args": {"query": "test"}, "id": "state_call", "name": "state_aware_tool"}],
                [],
            ]
        ),
        tools=[state_aware_tool],
        system_prompt="You are a helpful assistant.",
    )

    result = agent.invoke({"messages": [HumanMessage("Hello"), HumanMessage("World")]})

    # Check that tool accessed state correctly
    # 中文: 检查工具访问状态是否正确
    tool_message = result["messages"][3]
    assert isinstance(tool_message, ToolMessage)
    # Should have original 2 HumanMessages + 1 AIMessage before tool execution
    # 中文: 在工具执行之前应具有原始 2 个 HumanMessages + 1 个 AIMessage
    assert "Message count: 3" in tool_message.content


def test_tool_runtime_with_store() -> None:
    """Test ToolRuntime provides access to store.

    中文翻译:
    Test ToolRuntime 提供对存储的访问。"""
    # Note: create_agent doesn't currently expose a store parameter,
    # 中文: 注意：create_agent 当前不公开存储参数，
    # so runtime.store will be None in this test.
    # 中文: 所以在这个测试中，runtime.store 将为 None 。
    # This test demonstrates the runtime injection works correctly.
    # 中文: 此测试证明运行时注入工作正常。

    @tool
    def store_tool(key: str, value: str, runtime: ToolRuntime) -> str:
        """Tool that uses store.

        中文翻译:
        使用商店的工具。"""
        if runtime.store is None:
            return f"No store (key={key}, value={value})"
        runtime.store.put(("test",), key, {"data": value})
        return f"Stored {key}={value}"

    @tool
    def check_runtime_tool(runtime: ToolRuntime) -> str:
        """Tool that checks runtime availability.

        中文翻译:
        检查运行时可用性的工具。"""
        has_store = runtime.store is not None
        has_context = runtime.context is not None
        return f"Runtime: store={has_store}, context={has_context}"

    agent = create_agent(
        model=FakeToolCallingModel(
            tool_calls=[
                [{"args": {"key": "foo", "value": "bar"}, "id": "call_1", "name": "store_tool"}],
                [{"args": {}, "id": "call_2", "name": "check_runtime_tool"}],
                [],
            ]
        ),
        tools=[store_tool, check_runtime_tool],
        system_prompt="You are a helpful assistant.",
    )

    result = agent.invoke({"messages": [HumanMessage("Test store")]})

    # Find the tool messages
    # 中文: 查找工具消息
    tool_messages = [msg for msg in result["messages"] if isinstance(msg, ToolMessage)]
    assert len(tool_messages) == 2

    # First tool indicates no store is available (expected since create_agent doesn't expose store)
    # 中文: 第一个工具指示没有可用的存储（这是预期的，因为 create_agent 不公开存储）
    assert "No store" in tool_messages[0].content

    # Second tool confirms runtime was injected
    # 中文: 第二个工具确认运行时已注入
    assert "Runtime:" in tool_messages[1].content


def test_tool_runtime_with_multiple_tools() -> None:
    """Test multiple tools can all access ToolRuntime.

    中文翻译:
    测试多个工具都可以访问ToolRuntime。"""
    call_log = []

    @tool
    def tool_a(x: int, runtime: ToolRuntime) -> str:
        """First tool.

        中文翻译:
        第一个工具。"""
        call_log.append(("tool_a", runtime.tool_call_id, x))
        return f"A: {x}"

    @tool
    def tool_b(y: str, runtime: ToolRuntime) -> str:
        """Second tool.

        中文翻译:
        第二个工具。"""
        call_log.append(("tool_b", runtime.tool_call_id, y))
        return f"B: {y}"

    agent = create_agent(
        model=FakeToolCallingModel(
            tool_calls=[
                [
                    {"args": {"x": 1}, "id": "call_a", "name": "tool_a"},
                    {"args": {"y": "test"}, "id": "call_b", "name": "tool_b"},
                ],
                [],
            ]
        ),
        tools=[tool_a, tool_b],
        system_prompt="You are a helpful assistant.",
    )

    result = agent.invoke({"messages": [HumanMessage("Use both tools")]})

    # Verify both tools were called with correct runtime
    # 中文: 验证这两个工具均以正确的运行时调用
    assert len(call_log) == 2
    # Tools may execute in parallel, so check both calls are present
    # 中文: 工具可能会并行执行，因此请检查两个调用是否都存在
    call_ids = {(name, call_id) for name, call_id, _ in call_log}
    assert ("tool_a", "call_a") in call_ids
    assert ("tool_b", "call_b") in call_ids

    # Verify tool messages
    # 中文: 验证工具消息
    tool_messages = [msg for msg in result["messages"] if isinstance(msg, ToolMessage)]
    assert len(tool_messages) == 2
    contents = {msg.content for msg in tool_messages}
    assert "A: 1" in contents
    assert "B: test" in contents


def test_tool_runtime_config_access() -> None:
    """Test tools can access config through ToolRuntime.

    中文翻译:
    测试工具可以通过ToolRuntime访问配置。"""
    config_data = {}

    @tool
    def config_tool(x: int, runtime: ToolRuntime) -> str:
        """Tool that accesses config.

        中文翻译:
        访问配置的工具。"""
        config_data["config_exists"] = runtime.config is not None
        config_data["has_configurable"] = (
            "configurable" in runtime.config if runtime.config else False
        )
        # Config may have run_id or other fields depending on execution context
        # 中文: 配置可能有 run_id 或其他字段，具体取决于执行上下文
        if runtime.config:
            config_data["config_keys"] = list(runtime.config.keys())
        return f"Config accessed for {x}"

    agent = create_agent(
        model=FakeToolCallingModel(
            tool_calls=[
                [{"args": {"x": 5}, "id": "config_call", "name": "config_tool"}],
                [],
            ]
        ),
        tools=[config_tool],
        system_prompt="You are a helpful assistant.",
    )

    result = agent.invoke({"messages": [HumanMessage("Test config")]})

    # Verify config was accessible
    # 中文: 验证配置是否可访问
    assert config_data["config_exists"] is True
    assert "config_keys" in config_data

    # Verify tool executed
    # 中文: 验证工具已执行
    tool_message = result["messages"][2]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.content == "Config accessed for 5"


def test_tool_runtime_with_custom_state() -> None:
    """Test ToolRuntime works with custom state schemas.

    中文翻译:
    测试 ToolRuntime 使用自定义状态模式。"""

    class CustomState(AgentState):
        custom_field: str

    runtime_state = {}

    @tool
    def custom_state_tool(x: int, runtime: ToolRuntime) -> str:
        """Tool that accesses custom state.

        中文翻译:
        访问自定义状态的工具。"""
        runtime_state["custom_field"] = runtime.state.get("custom_field", "not found")
        return f"Custom: {x}"

    class CustomMiddleware(AgentMiddleware):
        state_schema = CustomState

    agent = create_agent(
        model=FakeToolCallingModel(
            tool_calls=[
                [{"args": {"x": 10}, "id": "custom_call", "name": "custom_state_tool"}],
                [],
            ]
        ),
        tools=[custom_state_tool],
        system_prompt="You are a helpful assistant.",
        middleware=[CustomMiddleware()],
    )

    result = agent.invoke(
        {
            "messages": [HumanMessage("Test custom state")],
            "custom_field": "custom_value",
        }
    )

    # Verify custom field was accessible
    # 中文: 验证自定义字段是否可访问
    assert runtime_state["custom_field"] == "custom_value"

    # Verify tool executed
    # 中文: 验证工具已执行
    tool_message = result["messages"][2]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.content == "Custom: 10"


def test_tool_runtime_no_runtime_parameter() -> None:
    """Test that tools without runtime parameter work normally.

    中文翻译:
    测试没有运行时参数的工具是否正常工作。"""

    @tool
    def regular_tool(x: int) -> str:
        """Regular tool without runtime.

        中文翻译:
        没有运行时间的常规工具。"""
        return f"Regular: {x}"

    @tool
    def runtime_tool(y: int, runtime: ToolRuntime) -> str:
        """Tool with runtime.

        中文翻译:
        具有运行时的工具。"""
        return f"Runtime: {y}, call_id: {runtime.tool_call_id}"

    agent = create_agent(
        model=FakeToolCallingModel(
            tool_calls=[
                [
                    {"args": {"x": 1}, "id": "regular_call", "name": "regular_tool"},
                    {"args": {"y": 2}, "id": "runtime_call", "name": "runtime_tool"},
                ],
                [],
            ]
        ),
        tools=[regular_tool, runtime_tool],
        system_prompt="You are a helpful assistant.",
    )

    result = agent.invoke({"messages": [HumanMessage("Test mixed tools")]})

    # Verify both tools executed correctly
    # 中文: 验证两个工具是否正确执行
    tool_messages = [msg for msg in result["messages"] if isinstance(msg, ToolMessage)]
    assert len(tool_messages) == 2
    assert tool_messages[0].content == "Regular: 1"
    assert "Runtime: 2, call_id: runtime_call" in tool_messages[1].content


async def test_tool_runtime_parallel_execution() -> None:
    """Test ToolRuntime injection works with parallel tool execution.

    中文翻译:
    测试 ToolRuntime 注入与并行工具执行配合使用。"""
    execution_log = []

    @tool
    async def parallel_tool_1(x: int, runtime: ToolRuntime) -> str:
        """First parallel tool.

        中文翻译:
        第一个并行工具。"""
        execution_log.append(("tool_1", runtime.tool_call_id, x))
        return f"Tool1: {x}"

    @tool
    async def parallel_tool_2(y: int, runtime: ToolRuntime) -> str:
        """Second parallel tool.

        中文翻译:
        第二个并行工具。"""
        execution_log.append(("tool_2", runtime.tool_call_id, y))
        return f"Tool2: {y}"

    agent = create_agent(
        model=FakeToolCallingModel(
            tool_calls=[
                [
                    {"args": {"x": 10}, "id": "parallel_1", "name": "parallel_tool_1"},
                    {"args": {"y": 20}, "id": "parallel_2", "name": "parallel_tool_2"},
                ],
                [],
            ]
        ),
        tools=[parallel_tool_1, parallel_tool_2],
        system_prompt="You are a helpful assistant.",
    )

    result = await agent.ainvoke({"messages": [HumanMessage("Run parallel")]})

    # Verify both tools executed
    # 中文: 验证两个工具均已执行
    assert len(execution_log) == 2

    # Find the tool messages (order may vary due to parallel execution)
    # 中文: 查找工具消息（由于并行执行，顺序可能会有所不同）
    tool_messages = [msg for msg in result["messages"] if isinstance(msg, ToolMessage)]
    assert len(tool_messages) == 2

    contents = {msg.content for msg in tool_messages}
    assert "Tool1: 10" in contents
    assert "Tool2: 20" in contents

    call_ids = {msg.tool_call_id for msg in tool_messages}
    assert "parallel_1" in call_ids
    assert "parallel_2" in call_ids


def test_tool_runtime_error_handling() -> None:
    """Test error handling with ToolRuntime injection.

    中文翻译:
    使用 ToolRuntime 注入测试错误处理。"""

    @tool
    def error_tool(x: int, runtime: ToolRuntime) -> str:
        """Tool that may error.

        中文翻译:
        可能会出错的工具。"""
        # Access runtime to ensure it's injected even during errors
        # 中文: 访问运行时以确保即使在错误期间也能注入
        _ = runtime.tool_call_id
        if x == 0:
            msg = "Cannot process zero"
            raise ValueError(msg)
        return f"Processed: {x}"

    # create_agent uses default error handling which doesn't catch ValueError
    # 中文: create_agent 使用默认错误处理，不会捕获 ValueError
    # So we need to handle this differently
    # 中文: 所以我们需要以不同的方式处理这个问题
    @tool
    def safe_tool(x: int, runtime: ToolRuntime) -> str:
        """Tool that handles errors safely.

        中文翻译:
        安全处理错误的工具。"""
        try:
            if x == 0:
                return "Error: Cannot process zero"
        except Exception as e:
            return f"Error: {e}"
        return f"Processed: {x}"

    agent = create_agent(
        model=FakeToolCallingModel(
            tool_calls=[
                [{"args": {"x": 0}, "id": "error_call", "name": "safe_tool"}],
                [{"args": {"x": 5}, "id": "success_call", "name": "safe_tool"}],
                [],
            ]
        ),
        tools=[safe_tool],
        system_prompt="You are a helpful assistant.",
    )

    result = agent.invoke({"messages": [HumanMessage("Test error handling")]})

    # Both tool calls should complete
    # 中文: 两个工具调用都应该完成
    tool_messages = [msg for msg in result["messages"] if isinstance(msg, ToolMessage)]
    assert len(tool_messages) == 2

    # First call returned error message
    # 中文: 第一次调用返回错误消息
    assert "Error:" in tool_messages[0].content or "Cannot process zero" in tool_messages[0].content

    # Second call succeeded
    # 中文: 第二次调用成功
    assert "Processed: 5" in tool_messages[1].content


def test_tool_runtime_with_middleware() -> None:
    """Test ToolRuntime injection works with agent middleware.

    中文翻译:
    测试 ToolRuntime 注入与代理中间件一起使用。"""
    middleware_calls = []
    runtime_calls = []

    class TestMiddleware(AgentMiddleware):
        def before_model(self, state, runtime) -> dict[str, Any]:
            middleware_calls.append("before_model")
            return {}

        def after_model(self, state, runtime) -> dict[str, Any]:
            middleware_calls.append("after_model")
            return {}

    @tool
    def middleware_tool(x: int, runtime: ToolRuntime) -> str:
        """Tool with runtime in middleware agent.

        中文翻译:
        在中间件代理中运行的工具。"""
        runtime_calls.append(("middleware_tool", runtime.tool_call_id))
        return f"Middleware result: {x}"

    agent = create_agent(
        model=FakeToolCallingModel(
            tool_calls=[
                [{"args": {"x": 7}, "id": "mw_call", "name": "middleware_tool"}],
                [],
            ]
        ),
        tools=[middleware_tool],
        system_prompt="You are a helpful assistant.",
        middleware=[TestMiddleware()],
    )

    result = agent.invoke({"messages": [HumanMessage("Test with middleware")]})

    # Verify middleware ran
    # 中文: 验证中间件运行
    assert "before_model" in middleware_calls
    assert "after_model" in middleware_calls

    # Verify tool with runtime executed
    # 中文: 验证工具已执行运行时
    assert len(runtime_calls) == 1
    assert runtime_calls[0] == ("middleware_tool", "mw_call")

    # Verify result
    # 中文: 验证结果
    tool_message = result["messages"][2]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.content == "Middleware result: 7"


def test_tool_runtime_type_hints() -> None:
    """Test that ToolRuntime provides access to state fields.

    中文翻译:
    测试 ToolRuntime 是否提供对状态字段的访问。"""
    typed_runtime = {}

    # Use ToolRuntime without generic type hints to avoid forward reference issues
    # 中文: 使用不带泛型类型提示的 ToolRuntime 以避免前向引用问题
    @tool
    def typed_runtime_tool(x: int, runtime: ToolRuntime) -> str:
        """Tool with runtime access.

        中文翻译:
        具有运行时访问权限的工具。"""
        # Access state dict - verify we can access standard state fields
        # 中文: 访问状态字典 - 验证我们可以访问标准状态字段
        if isinstance(runtime.state, dict):
            # Count messages in state
            # 中文: 统计状态中的消息数
            typed_runtime["message_count"] = len(runtime.state.get("messages", []))
        else:
            typed_runtime["message_count"] = len(getattr(runtime.state, "messages", []))
        return f"Typed: {x}"

    agent = create_agent(
        model=FakeToolCallingModel(
            tool_calls=[
                [{"args": {"x": 3}, "id": "typed_call", "name": "typed_runtime_tool"}],
                [],
            ]
        ),
        tools=[typed_runtime_tool],
        system_prompt="You are a helpful assistant.",
    )

    result = agent.invoke({"messages": [HumanMessage("Test")]})

    # Verify typed runtime worked -
    # 中文: 验证键入的运行时是否有效 -
    # should see 2 messages (HumanMessage + AIMessage) before tool executes
    # 中文: 在工具执行之前应该看到 2 条消息（HumanMessage + AIMessage）
    assert typed_runtime["message_count"] == 2

    tool_message = result["messages"][2]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.content == "Typed: 3"


def test_tool_runtime_name_based_injection() -> None:
    """Test that parameter named 'runtime' gets injected without type annotation.

    中文翻译:
    测试名为“runtime”的参数是否在没有类型注释的情况下注入。"""
    injected_data = {}

    @tool
    def name_based_tool(x: int, runtime: Any) -> str:
        """Tool with 'runtime' parameter without ToolRuntime type annotation.

        中文翻译:
        带有“runtime”参数且没有 ToolRuntime 类型注释的工具。"""
        # Even though type is Any, runtime should still be injected as ToolRuntime
        # 中文: 即使类型为 Any，运行时仍应作为 ToolRuntime 注入
        injected_data["is_tool_runtime"] = isinstance(runtime, ToolRuntime)
        injected_data["has_state"] = hasattr(runtime, "state")
        injected_data["has_tool_call_id"] = hasattr(runtime, "tool_call_id")
        if hasattr(runtime, "tool_call_id"):
            injected_data["tool_call_id"] = runtime.tool_call_id
        if hasattr(runtime, "state"):
            injected_data["state"] = runtime.state
        return f"Processed {x}"

    agent = create_agent(
        model=FakeToolCallingModel(
            tool_calls=[
                [{"args": {"x": 42}, "id": "name_call_123", "name": "name_based_tool"}],
                [],
            ]
        ),
        tools=[name_based_tool],
        system_prompt="You are a helpful assistant.",
    )

    result = agent.invoke({"messages": [HumanMessage("Test")]})

    # Verify tool executed
    # 中文: 验证工具已执行
    assert len(result["messages"]) == 4
    tool_message = result["messages"][2]
    assert isinstance(tool_message, ToolMessage)
    assert tool_message.content == "Processed 42"

    # Verify runtime was injected based on parameter name
    # 中文: 根据参数名称验证运行时是否已注入
    assert injected_data["is_tool_runtime"] is True
    assert injected_data["has_state"] is True
    assert injected_data["has_tool_call_id"] is True
    assert injected_data["tool_call_id"] == "name_call_123"
    assert injected_data["state"] is not None
    assert "messages" in injected_data["state"]


def test_combined_injected_state_runtime_store() -> None:
    """Test that all injection mechanisms work together in create_agent.

    This test verifies that a tool can receive injected state, tool runtime,
    and injected store simultaneously when specified in the function signature
    but not in the explicit args schema. This is modeled after the pattern
    from mre.py where multiple injection types are combined.
    

    中文翻译:
    测试所有注入机制在 create_agent 中是否协同工作。
    此测试验证工具是否可以接收注入状态、工具运行时、
    并在函数签名中指定时同时注入存储
    但不在显式 args 模式中。这是根据模式建模的
    来自 mre.py，其中组合了多种注入类型。"""
    # Track what was injected
    # 中文: 跟踪注入的内容
    injected_data = {}

    # Custom state schema with additional fields
    # 中文: 具有附加字段的自定义状态模式
    class CustomState(AgentState):
        user_id: str
        session_id: str

    # Define explicit args schema that only includes LLM-controlled parameters
    # 中文: 定义仅包含 LLM 控制参数的显式 args 架构
    weather_schema = {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "The location to get weather for"},
        },
        "required": ["location"],
    }

    @tool(args_schema=weather_schema)
    def multi_injection_tool(
        location: str,
        state: Annotated[Any, InjectedState],
        runtime: ToolRuntime,
        store: Annotated[Any, InjectedStore()],
    ) -> str:
        """Tool that uses injected state, runtime, and store together.

        Args:
            location: The location to get weather for (LLM-controlled).
            state: The graph state (injected).
            runtime: The tool runtime context (injected).
            store: The persistent store (injected).
        

        中文翻译:
        一起使用注入状态、运行时和存储的工具。
        参数：
            location：获取天气的位置（LLM 控制）。
            state：图状态（注入）。
            运行时：工具运行时上下文（注入）。
            store：持久存储（注入）。"""
        # Capture all injected parameters
        # 中文: 捕获所有注入的参数
        injected_data["state"] = state
        injected_data["user_id"] = state.get("user_id", "unknown")
        injected_data["session_id"] = state.get("session_id", "unknown")
        injected_data["runtime"] = runtime
        injected_data["tool_call_id"] = runtime.tool_call_id
        injected_data["store"] = store
        injected_data["store_is_none"] = store is None

        # Verify runtime.state matches the state parameter
        # 中文: 验证runtime.state与state参数匹配
        injected_data["runtime_state_matches"] = runtime.state == state

        return f"Weather info for {location}"

    # Create model that calls the tool
    # 中文: 创建调用该工具的模型
    model = FakeToolCallingModel(
        tool_calls=[
            [
                {
                    "name": "multi_injection_tool",
                    "args": {"location": "San Francisco"},  # Only LLM-controlled arg
                    "id": "call_weather_123",
                }
            ],
            [],  # End the loop
        ]
    )

    # Create agent with custom state and store
    # 中文: 创建具有自定义状态和存储的代理
    agent = create_agent(
        model=model,
        tools=[multi_injection_tool],
        state_schema=CustomState,
        store=InMemoryStore(),
    )

    # Verify the tool's args schema only includes LLM-controlled parameters
    # 中文: 验证工具的 args 架构仅包含 LLM 控制的参数
    tool_args_schema = multi_injection_tool.args_schema
    assert "location" in tool_args_schema["properties"]
    assert "state" not in tool_args_schema["properties"]
    assert "runtime" not in tool_args_schema["properties"]
    assert "store" not in tool_args_schema["properties"]

    # Invoke with custom state fields
    # 中文: 使用自定义状态字段进行调用
    result = agent.invoke(
        {
            "messages": [HumanMessage("What's the weather like?")],
            "user_id": "user_42",
            "session_id": "session_abc123",
        }
    )

    # Verify tool executed successfully
    # 中文: 验证工具执行成功
    tool_messages = [msg for msg in result["messages"] if isinstance(msg, ToolMessage)]
    assert len(tool_messages) == 1
    tool_message = tool_messages[0]
    assert tool_message.content == "Weather info for San Francisco"
    assert tool_message.tool_call_id == "call_weather_123"

    # Verify all injections worked correctly
    # 中文: 验证所有注射均正常工作
    assert injected_data["state"] is not None
    assert "messages" in injected_data["state"]

    # Verify custom state fields were accessible
    # 中文: 验证自定义状态字段是否可访问
    assert injected_data["user_id"] == "user_42"
    assert injected_data["session_id"] == "session_abc123"

    # Verify runtime was injected
    # 中文: 验证运行时是否已注入
    assert injected_data["runtime"] is not None
    assert injected_data["tool_call_id"] == "call_weather_123"

    # Verify store was injected
    # 中文: 验证商店已注入
    assert injected_data["store_is_none"] is False
    assert injected_data["store"] is not None

    # Verify runtime.state matches the injected state
    # 中文: 验证runtime.state与注入的状态匹配
    assert injected_data["runtime_state_matches"] is True


async def test_combined_injected_state_runtime_store_async() -> None:
    """Test that all injection mechanisms work together in async execution.

    This async version verifies that injected state, tool runtime, and injected
    store all work correctly with async tools in create_agent.
    

    中文翻译:
    测试所有注入机制是否在异步执行中协同工作。
    此异步版本验证注入状态、工具运行时和注入
    使用 create_agent 中的异步工具正确存储所有工作。"""
    # Track what was injected
    # 中文: 跟踪注入的内容
    injected_data = {}

    # Custom state schema
    # 中文: 自定义状态模式
    class CustomState(AgentState):
        api_key: str
        request_id: str

    # Define explicit args schema that only includes LLM-controlled parameters
    # 中文: 定义仅包含 LLM 控制参数的显式 args 架构
    # Note: state, runtime, and store are NOT in this schema
    # 中文: 注意：状态、运行时和存储不在此模式中
    search_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query"},
            "max_results": {"type": "integer", "description": "Maximum number of results"},
        },
        "required": ["query", "max_results"],
    }

    @tool(args_schema=search_schema)
    async def async_multi_injection_tool(
        query: str,
        max_results: int,
        state: Annotated[Any, InjectedState],
        runtime: ToolRuntime,
        store: Annotated[Any, InjectedStore()],
    ) -> str:
        """Async tool with multiple injection types.

        Args:
            query: The search query (LLM-controlled).
            max_results: Maximum number of results (LLM-controlled).
            state: The graph state (injected).
            runtime: The tool runtime context (injected).
            store: The persistent store (injected).
        

        中文翻译:
        具有多种注入类型的异步工具。
        参数：
            查询：搜索查询（LLM 控制）。
            max_results：最大结果数（LLM 控制）。
            state：图状态（注入）。
            运行时：工具运行时上下文（注入）。
            store：持久存储（注入）。"""
        # Capture all injected parameters
        # 中文: 捕获所有注入的参数
        injected_data["state"] = state
        injected_data["api_key"] = state.get("api_key", "unknown")
        injected_data["request_id"] = state.get("request_id", "unknown")
        injected_data["runtime"] = runtime
        injected_data["tool_call_id"] = runtime.tool_call_id
        injected_data["config"] = runtime.config
        injected_data["store"] = store

        # Verify we can write to the store
        # 中文: 验证我们可以写入商店
        if store is not None:
            await store.aput(("test", "namespace"), "test_key", {"query": query})
            # Read back to verify it worked
            # 中文: 回读以验证其是否有效
            item = await store.aget(("test", "namespace"), "test_key")
            injected_data["store_write_success"] = item is not None

        return f"Found {max_results} results for '{query}'"

    # Create model that calls the async tool
    # 中文: 创建调用异步工具的模型
    model = FakeToolCallingModel(
        tool_calls=[
            [
                {
                    "name": "async_multi_injection_tool",
                    "args": {"query": "test search", "max_results": 10},
                    "id": "call_search_456",
                }
            ],
            [],
        ]
    )

    # Create agent with custom state and store
    # 中文: 创建具有自定义状态和存储的代理
    agent = create_agent(
        model=model,
        tools=[async_multi_injection_tool],
        state_schema=CustomState,
        store=InMemoryStore(),
    )

    # Verify the tool's args schema only includes LLM-controlled parameters
    # 中文: 验证工具的 args 架构仅包含 LLM 控制的参数
    tool_args_schema = async_multi_injection_tool.args_schema
    assert "query" in tool_args_schema["properties"]
    assert "max_results" in tool_args_schema["properties"]
    assert "state" not in tool_args_schema["properties"]
    assert "runtime" not in tool_args_schema["properties"]
    assert "store" not in tool_args_schema["properties"]

    # Invoke async
    # 中文: 调用异步
    result = await agent.ainvoke(
        {
            "messages": [HumanMessage("Search for something")],
            "api_key": "sk-test-key-xyz",
            "request_id": "req_999",
        }
    )

    # Verify tool executed successfully
    # 中文: 验证工具执行成功
    tool_messages = [msg for msg in result["messages"] if isinstance(msg, ToolMessage)]
    assert len(tool_messages) == 1
    tool_message = tool_messages[0]
    assert tool_message.content == "Found 10 results for 'test search'"
    assert tool_message.tool_call_id == "call_search_456"

    # Verify all injections worked correctly
    # 中文: 验证所有注射均正常工作
    assert injected_data["state"] is not None
    assert injected_data["api_key"] == "sk-test-key-xyz"
    assert injected_data["request_id"] == "req_999"

    # Verify runtime was injected
    # 中文: 验证运行时是否已注入
    assert injected_data["runtime"] is not None
    assert injected_data["tool_call_id"] == "call_search_456"
    assert injected_data["config"] is not None

    # Verify store was injected and writable
    # 中文: 验证存储已注入且可写
    assert injected_data["store"] is not None
    assert injected_data["store_write_success"] is True
