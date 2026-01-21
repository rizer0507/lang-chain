"""Redefined messages as a work-around for pydantic issue with AnyStr.

The code below creates version of pydantic models
that will work in unit tests with AnyStr as id field
Please note that the `id` field is assigned AFTER the model is created
to workaround an issue with pydantic ignoring the __eq__ method on
subclassed strings.

中文翻译:
重新定义消息作为 AnyStr 的 pydantic 问题的解决方法。
下面的代码创建 pydantic 模型的版本
它将在使用 AnyStr 作为 id 字段的单元测试中工作
请注意，“id”字段是在模型创建后分配的
解决 pydantic 忽略 __eq__ 方法的问题
子类字符串。
"""

from typing import Any

from langchain_core.messages import HumanMessage, ToolMessage

from .any_str import AnyStr


def _AnyIdHumanMessage(**kwargs: Any) -> HumanMessage:  # noqa: N802
    """Create a human message with an any id field.

    中文翻译:
    创建带有任何 id 字段的人工消息。"""
    message = HumanMessage(**kwargs)
    message.id = AnyStr()
    return message


def _AnyIdToolMessage(**kwargs: Any) -> ToolMessage:  # noqa: N802
    """Create a tool message with an any id field.

    中文翻译:
    创建带有任意 id 字段的工具消息。"""
    message = ToolMessage(**kwargs)
    message.id = AnyStr()
    return message
