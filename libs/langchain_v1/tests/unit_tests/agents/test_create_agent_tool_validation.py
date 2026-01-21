import sys
from typing import Annotated

import pytest
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import InjectedStore, ToolRuntime
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from langchain.agents import AgentState, create_agent
from langchain.tools import InjectedState
from langchain.tools import tool as dec_tool

from .model import FakeToolCallingModel


@pytest.mark.skipif(
    sys.version_info >= (3, 14), reason="Pydantic model rebuild issue in Python 3.14"
)
def test_tool_invocation_error_excludes_injected_state() -> None:
    """Test that tool invocation errors only include LLM-controllable arguments.

    When a tool has InjectedState parameters and the LLM makes an incorrect
    invocation (e.g., missing required arguments), the error message should only
    contain the arguments from the tool call that the LLM controls. This ensures
    the LLM receives relevant context to correct its mistakes, without being
    distracted by system-injected parameters it has no control over.
    This test uses create_agent to ensure the behavior works in a full agent context.
    

    中文翻译:
    测试工具调用错误仅包含 LLM 可控参数。
    当工具具有 InjectedState 参数并且 LLM 做出错误的
    调用（例如，缺少必需的参数），错误消息应该只
    包含 LLM 控制的工具调用的参数。这确保了
    法学硕士收到相关背景信息以纠正其错误，而无需
    被它无法控制的系统注入参数分散了注意力。
    此测试使用 create_agent 来确保该行为在完整的代理上下文中运行。"""

    # Define a custom state schema with injected data
    # 中文: 使用注入的数据定义自定义状态模式
    class TestState(AgentState):
        secret_data: str  # Example of state data not controlled by LLM

    @dec_tool
    def tool_with_injected_state(
        some_val: int,
        state: Annotated[TestState, InjectedState],
    ) -> str:
        """Tool that uses injected state.

        中文翻译:
        使用注入状态的工具。"""
        return f"some_val: {some_val}"

    # Create a fake model that makes an incorrect tool call (missing 'some_val')
    # 中文: 创建一个虚假模型，该模型进行不正确的工具调用（缺少“some_val”）
    # Then returns no tool calls on the second iteration to end the loop
    # 中文: 然后在第二次迭代时不返回任何工具调用来结束循环
    model = FakeToolCallingModel(
        tool_calls=[
            [
                {
                    "name": "tool_with_injected_state",
                    "args": {"wrong_arg": "value"},  # Missing required 'some_val'
                    "id": "call_1",
                }
            ],
            [],  # No tool calls on second iteration to end the loop
        ]
    )

    # Create an agent with the tool and custom state schema
    # 中文: 使用工具和自定义状态架构创建代理
    agent = create_agent(
        model=model,
        tools=[tool_with_injected_state],
        state_schema=TestState,
    )

    # Invoke the agent with injected state data
    # 中文: 使用注入的状态数据调用代理
    result = agent.invoke(
        {
            "messages": [HumanMessage("Test message")],
            "secret_data": "sensitive_secret_123",
        }
    )

    # Find the tool error message
    # 中文: 查找工具错误消息
    tool_messages = [m for m in result["messages"] if m.type == "tool"]
    assert len(tool_messages) == 1
    tool_message = tool_messages[0]
    assert tool_message.status == "error"

    # The error message should contain only the LLM-provided args (wrong_arg)
    # 中文: 错误消息应仅包含 LLM 提供的参数 (wrong_arg)
    # and NOT the system-injected state (secret_data)
    # 中文: 而不是系统注入的状态（secret_data）
    assert "{'wrong_arg': 'value'}" in tool_message.content
    assert "secret_data" not in tool_message.content
    assert "sensitive_secret_123" not in tool_message.content


@pytest.mark.skipif(
    sys.version_info >= (3, 14), reason="Pydantic model rebuild issue in Python 3.14"
)
async def test_tool_invocation_error_excludes_injected_state_async() -> None:
    """Test that async tool invocation errors only include LLM-controllable arguments.

    This test verifies that the async execution path (_execute_tool_async and _arun_one)
    properly filters validation errors to exclude system-injected arguments, ensuring
    the LLM receives only relevant context for correction.
    

    中文翻译:
    测试异步工具调用错误是否仅包含 LLM 可控参数。
    此测试验证异步执行路径（_execute_tool_async 和 _arun_one）
    正确过滤验证错误以排除系统注入的参数，确保
    法学硕士仅收到相关上下文以供纠正。"""

    # Define a custom state schema
    # 中文: 定义自定义状态模式
    class TestState(AgentState):
        internal_data: str

    @dec_tool
    async def async_tool_with_injected_state(
        query: str,
        max_results: int,
        state: Annotated[TestState, InjectedState],
    ) -> str:
        """Async tool that uses injected state.

        中文翻译:
        使用注入状态的异步工具。"""
        return f"query: {query}, max_results: {max_results}"

    # Create a fake model that makes an incorrect tool call
    # 中文: 创建一个虚假模型，导致错误的工具调用
    # - query has wrong type (int instead of str)
    # 中文: - 查询的类型错误（int 而不是 str）
    # - max_results is missing
    # 中文: - max_results 丢失
    model = FakeToolCallingModel(
        tool_calls=[
            [
                {
                    "name": "async_tool_with_injected_state",
                    "args": {"query": 999},  # Wrong type, missing max_results
                    "id": "call_async_1",
                }
            ],
            [],  # End the loop
        ]
    )

    # Create an agent with the async tool
    # 中文: 使用异步工具创建代理
    agent = create_agent(
        model=model,
        tools=[async_tool_with_injected_state],
        state_schema=TestState,
    )

    # Invoke with state data
    # 中文: 使用状态数据调用
    result = await agent.ainvoke(
        {
            "messages": [HumanMessage("Test async")],
            "internal_data": "secret_internal_value_xyz",
        }
    )

    # Find the tool error message
    # 中文: 查找工具错误消息
    tool_messages = [m for m in result["messages"] if m.type == "tool"]
    assert len(tool_messages) == 1
    tool_message = tool_messages[0]
    assert tool_message.status == "error"

    # Verify error mentions LLM-controlled parameters only
    # 中文: 验证错误仅提及 LLM 控制的参数
    content = tool_message.content
    assert "query" in content.lower(), "Error should mention 'query' (LLM-controlled)"
    assert "max_results" in content.lower(), "Error should mention 'max_results' (LLM-controlled)"

    # Verify system-injected state does not appear in the validation errors
    # 中文: 验证系统注入状态不会出现在验证错误中
    # This keeps the error focused on what the LLM can actually fix
    # 中文: 这使得错误集中在法学硕士可以实际解决的问题上
    assert "internal_data" not in content, (
        "Error should NOT mention 'internal_data' (system-injected field)"
    )
    assert "secret_internal_value" not in content, (
        "Error should NOT contain system-injected state values"
    )

    # Verify only LLM-controlled parameters are in the error list
    # 中文: 验证错误列表中仅包含 LLM 控制的参数
    # Should see "query" and "max_results" errors, but not "state"
    # 中文: 应该看到“query”和“max_results”错误，但看不到“state”
    lines = content.split("\n")
    error_lines = [line.strip() for line in lines if line.strip()]
    # Find lines that look like field names (single words at start of line)
    # 中文: 查找看起来像字段名称的行（行开头的单个单词）
    field_errors = [
        line
        for line in error_lines
        if line
        and not line.startswith("input")
        and not line.startswith("field")
        and not line.startswith("error")
        and not line.startswith("please")
        and len(line.split()) <= 2
    ]
    # Verify system-injected 'state' is not in the field error list
    # 中文: 验证系统注入的“状态”不在字段错误列表中
    assert not any(field.lower() == "state" for field in field_errors), (
        "The field 'state' (system-injected) should not appear in validation errors"
    )


@pytest.mark.skipif(
    sys.version_info >= (3, 14), reason="Pydantic model rebuild issue in Python 3.14"
)
async def test_create_agent_error_content_with_multiple_params() -> None:
    """Test that error messages only include LLM-controlled parameter errors.

    Uses create_agent to verify that when a tool with both LLM-controlled
    and system-injected parameters receives invalid arguments, the error message:
    1. Contains details about LLM-controlled parameter errors (query, limit)
    2. Does NOT contain system-injected parameter names (state, store, runtime)
    3. Does NOT contain values from system-injected parameters
    4. Properly formats the validation errors for LLM correction
    This ensures the LLM receives focused, actionable feedback.
    

    中文翻译:
    测试错误消息仅包含 LLM 控制的参数错误。
    使用 create_agent 来验证当一个工具同时具有 LLM 控制时
    并且系统注入的参数接收到无效参数，错误消息：
    1.包含LLM控制的参数错误的详细信息（查询、限制）
    2. 不包含系统注入的参数名称（状态、存储、运行时）
    3. 不包含系统注入参数的值
    4. 正确格式化 LLM 更正的验证错误
    这确保了法学硕士收到有针对性的、可操作的反馈。"""

    class TestState(AgentState):
        user_id: str
        api_key: str
        session_data: dict

    @dec_tool
    def complex_tool(
        query: str,
        limit: int,
        state: Annotated[TestState, InjectedState],
        store: Annotated[BaseStore, InjectedStore()],
        runtime: ToolRuntime,
    ) -> str:
        """A complex tool with multiple injected and non-injected parameters.

        Args:
            query: The search query string.
            limit: Maximum number of results to return.
            state: The graph state (injected).
            store: The persistent store (injected).
            runtime: The tool runtime context (injected).
        

        中文翻译:
        具有多个注入和非注入参数的复杂工具。
        参数：
            查询：搜索查询字符串。
            limit：返回结果的最大数量。
            state：图状态（注入）。
            store：持久存储（注入）。
            运行时：工具运行时上下文（注入）。"""
        # Access injected params to verify they work in normal execution
        # 中文: 访问注入的参数以验证它们在正常执行中工作
        user = state.get("user_id", "unknown")
        return f"Results for '{query}' (limit={limit}, user={user})"

    # Create a model that makes an incorrect tool call with multiple errors:
    # 中文: 创建一个模型，该模型会进行不正确的工具调用并出现多个错误：
    # - query is wrong type (int instead of str)
    # 中文: - 查询类型错误（int 而不是 str）
    # - limit is missing
    # 中文: - 缺少限制
    # Then returns no tool calls to end the loop
    # 中文: 然后不返回任何工具调用来结束循环
    model = FakeToolCallingModel(
        tool_calls=[
            [
                {
                    "name": "complex_tool",
                    "args": {
                        "query": 12345,  # Wrong type - should be str
                        # "limit" is missing - required field
                        # 中文: 缺少“限制” - 必填字段
                    },
                    "id": "call_complex_1",
                }
            ],
            [],  # No tool calls on second iteration to end the loop
        ]
    )

    # Create an agent with the complex tool and custom state
    # 中文: 使用复杂的工具和自定义状态创建代理
    # Need to provide a store since the tool uses InjectedStore
    # 中文: 需要提供一个商店，因为该工具使用 InjectedStore
    agent = create_agent(
        model=model,
        tools=[complex_tool],
        state_schema=TestState,
        store=InMemoryStore(),
    )

    # Invoke with sensitive data in state
    # 中文: 使用状态中的敏感数据进行调用
    result = agent.invoke(
        {
            "messages": [HumanMessage("Search for something")],
            "user_id": "user_12345",
            "api_key": "sk-secret-key-abc123xyz",
            "session_data": {"token": "secret_session_token"},
        }
    )

    # Find the tool error message
    # 中文: 查找工具错误消息
    tool_messages = [m for m in result["messages"] if m.type == "tool"]
    assert len(tool_messages) == 1
    tool_message = tool_messages[0]
    assert tool_message.status == "error"
    assert tool_message.tool_call_id == "call_complex_1"

    content = tool_message.content

    # Verify error mentions LLM-controlled parameter issues
    # 中文: 验证错误提到 LLM 控制的参数问题
    assert "query" in content.lower(), "Error should mention 'query' (LLM-controlled)"
    assert "limit" in content.lower(), "Error should mention 'limit' (LLM-controlled)"

    # Should indicate validation errors occurred
    # 中文: 应指示发生验证错误
    assert "validation error" in content.lower() or "error" in content.lower(), (
        "Error should indicate validation occurred"
    )

    # Verify NO system-injected parameter names appear in error
    # 中文: 验证系统注入的参数名称是否出现错误
    # These are not controlled by the LLM and should be excluded
    # 中文: 这些不受法学硕士控制，应排除在外
    assert "state" not in content.lower(), "Error should NOT mention 'state' (system-injected)"
    assert "store" not in content.lower(), "Error should NOT mention 'store' (system-injected)"
    assert "runtime" not in content.lower(), "Error should NOT mention 'runtime' (system-injected)"

    # Verify NO values from system-injected parameters appear in error
    # 中文: 验证系统注入参数中的值是否出现错误
    # The LLM doesn't control these, so they shouldn't distract from the actual issues
    # 中文: 法学硕士不控制这些，所以它们不应该分散对实际问题的注意力
    assert "user_12345" not in content, "Error should NOT contain user_id value (from state)"
    assert "sk-secret-key" not in content, "Error should NOT contain api_key value (from state)"
    assert "secret_session_token" not in content, (
        "Error should NOT contain session_data value (from state)"
    )

    # Verify the LLM's original tool call args are present
    # 中文: 验证 LLM 的原始工具调用参数是否存在
    # The error should show what the LLM actually provided to help it correct the mistake
    # 中文: 该错误应显示法学硕士实际提供的内容以帮助其纠正错误
    assert "12345" in content, "Error should show the invalid query value provided by LLM (12345)"

    # Check error is well-formatted
    # 中文: 检查错误格式是否正确
    assert "complex_tool" in content, "Error should mention the tool name"


@pytest.mark.skipif(
    sys.version_info >= (3, 14), reason="Pydantic model rebuild issue in Python 3.14"
)
async def test_create_agent_error_only_model_controllable_params() -> None:
    """Test that errors only include LLM-controllable parameter issues.

    Focused test ensuring that validation errors for LLM-controlled parameters
    are clearly reported, while system-injected parameters remain completely
    absent from error messages. This provides focused feedback to the LLM.
    

    中文翻译:
    测试错误仅包括 LLM 可控参数问题。
    集中测试确保LLM控制参数的验证错误
    被清楚地报告，而系统注入的参数完全保留
    错误消息中不存在。这为法学硕士提供了集中的反馈。"""

    class StateWithSecrets(AgentState):
        password: str  # Example of data not controlled by LLM

    @dec_tool
    def secure_tool(
        username: str,
        email: str,
        state: Annotated[StateWithSecrets, InjectedState],
    ) -> str:
        """Tool that validates user credentials.

        Args:
            username: The username (3-20 chars).
            email: The email address.
            state: State with password (system-injected).
        

        中文翻译:
        验证用户凭据的工具。
        参数：
            用户名：用户名（3-20 个字符）。
            电子邮件：电子邮件地址。
            state：带有密码的状态（系统注入）。"""
        return f"Validated {username} with email {email}"

    # LLM provides invalid username (too short) and invalid email
    # 中文: LLM 提供无效的用户名（太短）和无效的电子邮件
    model = FakeToolCallingModel(
        tool_calls=[
            [
                {
                    "name": "secure_tool",
                    "args": {
                        "username": "ab",  # Too short (needs 3-20)
                        "email": "not-an-email",  # Invalid format
                    },
                    "id": "call_secure_1",
                }
            ],
            [],
        ]
    )

    agent = create_agent(
        model=model,
        tools=[secure_tool],
        state_schema=StateWithSecrets,
    )

    result = agent.invoke(
        {
            "messages": [HumanMessage("Create account")],
            "password": "super_secret_password_12345",
        }
    )

    tool_messages = [m for m in result["messages"] if m.type == "tool"]
    assert len(tool_messages) == 1
    content = tool_messages[0].content

    # The error should mention LLM-controlled parameters
    # 中文: 错误应该提到LLM控制的参数
    # Note: Pydantic's default validation may or may not catch format issues,
    # 中文: 注意：Pydantic 的默认验证可能会也可能不会捕获格式问题，
    # but the parameters themselves should be present in error messages
    # 中文: 但参数本身应该出现在错误消息中
    assert "username" in content.lower() or "email" in content.lower(), (
        "Error should mention at least one LLM-controlled parameter"
    )

    # Password is system-injected and should not appear
    # 中文: 密码是系统注入的，不应出现
    # The LLM doesn't control it, so it shouldn't distract from the actual errors
    # 中文: LLM 无法控制它，因此它不应该分散对实际错误的注意力
    assert "password" not in content.lower(), (
        "Error should NOT mention 'password' (system-injected parameter)"
    )
    assert "super_secret_password" not in content, (
        "Error should NOT contain password value (from system-injected state)"
    )
