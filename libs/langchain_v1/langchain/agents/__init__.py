"""LangChain Agent 构建入口模块。

本模块是使用 LangChain 构建 Agent（智能体）的入口点。

核心功能:
---------
**create_agent**: 创建具有中间件支持的 Agent 图

什么是 Agent:
-------------
Agent 是一个能够使用工具循环调用的系统，直到满足停止条件。
它由语言模型驱动，可以根据用户输入自主决定：
- 调用哪些工具
- 如何处理工具返回结果
- 何时停止并返回最终答案

使用示例:
---------
>>> from langchain.agents import create_agent
>>>
>>> def check_weather(location: str) -> str:
...     '''返回指定位置的天气预报。'''
...     return f"{location}的天气是晴天"
>>>
>>> graph = create_agent(
...     model="openai:gpt-4o",
...     tools=[check_weather],
...     system_prompt="你是一个有用的助手",
... )
>>>
>>> result = graph.invoke({"messages": [{"role": "user", "content": "北京天气怎么样？"}]})

更多信息请参阅:
- Agent 文档: https://docs.langchain.com/oss/python/langchain/agents
- 中间件文档: https://docs.langchain.com/oss/python/langchain/middleware
"""

from langchain.agents.factory import create_agent
from langchain.agents.middleware.types import AgentState

__all__ = [
    "AgentState",
    "create_agent",
]
