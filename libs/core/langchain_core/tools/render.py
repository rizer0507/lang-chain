"""Utilities to render tools.

中文翻译:
渲染工具的实用程序。"""

from __future__ import annotations

from collections.abc import Callable
from inspect import signature

from langchain_core.tools.base import BaseTool

ToolsRenderer = Callable[[list[BaseTool]], str]


def render_text_description(tools: list[BaseTool]) -> str:
    """Render the tool name and description in plain text.

    Args:
        tools: The tools to render.

    Returns:
        The rendered text.

    Output will be in the format of:

    ```txt
    search: This tool is used for search
    calculator: This tool is used for math
    ```
    

    中文翻译:
    以纯文本形式呈现工具名称和描述。
    参数：
        工具：渲染工具。
    返回：
        渲染的文本。
    输出将采用以下格式：
    ````txt
    搜索：该工具用于搜索
    计算器：这个工具用于数学计算
    ````"""
    descriptions = []
    for tool in tools:
        if hasattr(tool, "func") and tool.func:
            sig = signature(tool.func)
            description = f"{tool.name}{sig} - {tool.description}"
        else:
            description = f"{tool.name} - {tool.description}"

        descriptions.append(description)
    return "\n".join(descriptions)


def render_text_description_and_args(tools: list[BaseTool]) -> str:
    """Render the tool name, description, and args in plain text.

    Args:
        tools: The tools to render.

    Returns:
        The rendered text.

    Output will be in the format of:

    ```txt
    search: This tool is used for search, args: {"query": {"type": "string"}}
    calculator: This tool is used for math, \
    args: {"expression": {"type": "string"}}
    ```
    

    中文翻译:
    以纯文本形式呈现工具名称、描述和参数。
    参数：
        工具：渲染工具。
    返回：
        渲染的文本。
    输出将采用以下格式：
    ````txt
    search: 该工具用于搜索，args: {"query": {"type": "string"}}
    计算器：这个工具用于数学计算，\
    args: {"表达式": {"类型": "字符串"}}
    ````"""
    tool_strings = []
    for tool in tools:
        args_schema = str(tool.args)
        if hasattr(tool, "func") and tool.func:
            sig = signature(tool.func)
            description = f"{tool.name}{sig} - {tool.description}"
        else:
            description = f"{tool.name} - {tool.description}"
        tool_strings.append(f"{description}, args: {args_schema}")
    return "\n".join(tool_strings)
