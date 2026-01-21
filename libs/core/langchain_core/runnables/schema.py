"""Runnable 对象使用的类型定义模块。

本模块包含与 `Runnable` 对象一起使用的 TypedDict 类型定义，
主要用于 `astream_events` 方法产生的流式事件。

核心类型:
---------
1. **StreamEvent**: 流式事件的联合类型
2. **StandardStreamEvent**: 标准流式事件
3. **CustomStreamEvent**: 自定义流式事件
4. **EventData**: 事件数据

事件类型说明:
---------
事件名称格式: `on_[runnable_type]_(start|stream|end)`

Runnable 类型:
- **llm**: 非聊天模型
- **chat_model**: 聊天模型
- **prompt**: 提示模板（如 ChatPromptTemplate）
- **tool**: 工具（通过 @tool 装饰器定义或继承自 Tool/BaseTool）
- **chain**: 大多数 Runnable 对象

事件阶段:
- **start**: Runnable 开始执行时
- **stream**: Runnable 流式输出时
- **end**: Runnable 结束执行时

使用示例:
---------
>>> from langchain_core.runnables import RunnableLambda
>>>
>>> async def reverse(s: str) -> str:
...     return s[::-1]
>>>
>>> chain = RunnableLambda(func=reverse)
>>>
>>> events = [event async for event in chain.astream_events("hello")]
>>> # events 会包含:
>>> # [
>>> #     {"event": "on_chain_start", "data": {"input": "hello"}, ...},
>>> #     {"event": "on_chain_stream", "data": {"chunk": "olleh"}, ...},
>>> #     {"event": "on_chain_end", "data": {"output": "olleh"}, ...},
>>> # ]
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from typing_extensions import NotRequired, TypedDict

if TYPE_CHECKING:
    from collections.abc import Sequence


class EventData(TypedDict, total=False):
    """与流式事件关联的数据。

    包含事件的输入、输出、错误和流式块信息。
    不同阶段的事件包含不同的字段。
    """

    input: Any
    """传递给生成此事件的 `Runnable` 的输入。

    输入有时在 *START* 事件时可用，有时在 *END* 事件时可用。

    如果 Runnable 能够流式处理其输入，则其输入在流式处理完成（*END*）之前
    是未知的。
    """

    error: NotRequired[BaseException]
    """在执行 `Runnable` 期间发生的错误。

    此字段仅在 Runnable 抛出异常时可用。

    !!! version-added "在 `langchain-core` 1.0.0 中添加"
    """

    output: Any
    """生成此事件的 `Runnable` 的输出。

    输出仅在 *END* 事件时可用。

    对于大多数 Runnable 对象，此字段可以从 `chunk` 字段推断出来，
    但某些特殊的 Runnable（如聊天模型）可能返回更多信息。
    """

    chunk: Any
    """生成此事件的输出的流式块。

    块通常支持加法运算，将它们相加应该得到
    生成此事件的 Runnable 的完整输出。
    """


class BaseStreamEvent(TypedDict):
    """流式事件的基类。

    这是 `astream_events` 方法产生的流式事件的模式。

    使用示例:
        ```python
        from langchain_core.runnables import RunnableLambda

        async def reverse(s: str) -> str:
            return s[::-1]

        chain = RunnableLambda(func=reverse)

        events = [event async for event in chain.astream_events("hello")]

        # 将产生以下事件（省略了部分字段）:
        [
            {
                "data": {"input": "hello"},
                "event": "on_chain_start",
                "name": "reverse",
            },
            {
                "data": {"chunk": "olleh"},
                "event": "on_chain_stream",
                "name": "reverse",
            },
            {
                "data": {"output": "olleh"},
                "event": "on_chain_end",
                "name": "reverse",
            },
        ]
        ```
    """

    event: str
    """事件名称，格式为: `on_[runnable_type]_(start|stream|end)`。

    Runnable 类型包括:

    - **llm**: 非聊天模型使用
    - **chat_model**: 聊天模型使用
    - **prompt**: 提示模板（如 ChatPromptTemplate）
    - **tool**: 通过 @tool 装饰器定义或继承自 Tool/BaseTool 的工具
    - **chain**: 大多数 Runnable 对象属于此类型

    事件阶段:

    - **start**: Runnable 开始执行时触发
    - **stream**: Runnable 流式输出时触发
    - **end**: Runnable 结束执行时触发

    不同阶段的 `data` 负载略有不同。
    详见 `EventData` 的文档。
    """

    run_id: str
    """随机生成的 ID，用于追踪 Runnable 的执行。

    作为父 Runnable 执行一部分被调用的每个子 Runnable
    都会被分配自己独特的 ID。
    """

    tags: NotRequired[list[str]]
    """与生成此事件的 `Runnable` 关联的标签。

    标签总是从父 Runnable 对象继承。

    标签可以通过 `.with_config({"tags": ["hello"]})` 绑定到 Runnable，
    或者在运行时通过 `.astream_events(..., {"tags": ["hello"]})` 传递。
    """

    metadata: NotRequired[dict[str, Any]]
    """与生成此事件的 `Runnable` 关联的元数据。

    元数据可以通过以下方式绑定到 Runnable:

        `.with_config({"metadata": {"foo": "bar"}})`

    或者在运行时传递:

        `.astream_events(..., {"metadata": {"foo": "bar"}})`
    """

    parent_ids: Sequence[str]
    """与此事件关联的父 ID 列表。

    根事件的列表为空。

    例如，如果 Runnable A 调用 Runnable B，
    则 Runnable B 生成的事件的 `parent_ids` 中会包含 Runnable A 的 ID。

    父 ID 的顺序是从根父级到直接父级。

    仅在 astream_events API v2 中支持。v1 将返回空列表。
    """


class StandardStreamEvent(BaseStreamEvent):
    """遵循 LangChain 事件数据约定的标准流式事件。"""

    data: EventData
    """事件数据。

    事件数据的内容取决于事件类型。
    """

    name: str
    """生成此事件的 `Runnable` 的名称。"""


class CustomStreamEvent(BaseStreamEvent):
    """用户创建的自定义流式事件。

    可以通过 `adispatch_custom_event` 方法创建自定义事件。
    """

    # 覆盖 event 字段使其更具体
    event: Literal["on_custom_event"]  # type: ignore[misc]
    """事件类型，固定为 "on_custom_event"。"""

    name: str
    """用户定义的事件名称。"""

    data: Any
    """与事件关联的数据。自由格式，可以是任何内容。"""


# 流式事件类型别名
StreamEvent = StandardStreamEvent | CustomStreamEvent
"""流式事件的联合类型。

可以是 StandardStreamEvent（标准事件）或 CustomStreamEvent（自定义事件）。
"""
