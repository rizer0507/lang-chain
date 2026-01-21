"""Test agent name parameter in create_agent.

This module tests that the name parameter correctly sets .name on AIMessage outputs.

中文翻译:
测试 create_agent 中的代理名称参数。
该模块测试 name 参数是否正确设置 AIMessage 输出上的 .name。
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from langchain.agents import create_agent

from .model import FakeToolCallingModel


@tool
def simple_tool(x: int) -> str:
    """Simple tool for basic tests.

    中文翻译:
    用于基本测试的简单工具。"""
    return f"Result: {x}"


def test_agent_name_set_on_ai_message() -> None:
    """Test that agent name is set on AIMessage when name is provided.

    中文翻译:
    测试在提供名称时是否在 AIMessage 上设置代理名称。"""
    agent = create_agent(
        model=FakeToolCallingModel(tool_calls=[[]]),
        name="test_agent",
    )

    result = agent.invoke({"messages": [HumanMessage("Hello")]})

    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert len(ai_messages) == 1
    assert ai_messages[0].name == "test_agent"


def test_agent_name_not_set_when_none() -> None:
    """Test that AIMessage.name is not set when name is not provided.

    中文翻译:
    测试未提供名称时是否未设置 AIMessage.name。"""
    agent = create_agent(
        model=FakeToolCallingModel(tool_calls=[[]]),
    )

    result = agent.invoke({"messages": [HumanMessage("Hello")]})

    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert len(ai_messages) == 1
    assert ai_messages[0].name is None


def test_agent_name_on_multiple_iterations() -> None:
    """Test that agent name is set on all AIMessages in multi-turn conversation.

    中文翻译:
    测试多轮对话中的所有 AIMessage 上是否设置了代理名称。"""
    agent = create_agent(
        model=FakeToolCallingModel(
            tool_calls=[[{"args": {"x": 1}, "id": "call_1", "name": "simple_tool"}], []]
        ),
        tools=[simple_tool],
        name="multi_turn_agent",
    )

    result = agent.invoke({"messages": [HumanMessage("Call a tool")]})

    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert len(ai_messages) == 2
    for msg in ai_messages:
        assert msg.name == "multi_turn_agent"


@pytest.mark.asyncio
async def test_agent_name_async() -> None:
    """Test that agent name is set on AIMessage in async execution.

    中文翻译:
    测试异步执行中 AIMessage 上是否设置了代理名称。"""
    agent = create_agent(
        model=FakeToolCallingModel(tool_calls=[[]]),
        name="async_agent",
    )

    result = await agent.ainvoke({"messages": [HumanMessage("Hello async")]})

    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert len(ai_messages) == 1
    assert ai_messages[0].name == "async_agent"


@pytest.mark.asyncio
async def test_agent_name_async_multiple_iterations() -> None:
    """Test that agent name is set on all AIMessages in async multi-turn.

    中文翻译:
    测试是否在异步多轮中的所有 AIMessage 上设置了代理名称。"""
    agent = create_agent(
        model=FakeToolCallingModel(
            tool_calls=[[{"args": {"x": 5}, "id": "call_1", "name": "simple_tool"}], []]
        ),
        tools=[simple_tool],
        name="async_multi_agent",
    )

    result = await agent.ainvoke({"messages": [HumanMessage("Call tool async")]})

    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assert len(ai_messages) == 2
    for msg in ai_messages:
        assert msg.name == "async_multi_agent"
