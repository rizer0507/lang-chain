"""消息提示模板基类模块。

本模块定义了 `BaseMessagePromptTemplate`，这是所有消息提示模板的抽象基类。
消息提示模板用于在 ChatPromptTemplate 中生成聊天消息。

核心概念:
---------
在 LangChain 的聊天模型中，消息分为几种类型：
- SystemMessage: 系统消息，用于设定 AI 的行为
- HumanMessage: 用户消息
- AIMessage: AI 的回复
- ChatMessage: 自定义角色的消息

消息提示模板的作用是将模板字符串转换为这些消息类型。

类层次结构:
---------
BaseMessagePromptTemplate (本类)
├── MessagesPlaceholder          (动态消息占位符)
├── BaseStringMessagePromptTemplate
│   ├── HumanMessagePromptTemplate
│   ├── AIMessagePromptTemplate
│   ├── SystemMessagePromptTemplate
│   └── ChatMessagePromptTemplate
└── ...

使用示例:
---------
>>> from langchain_core.prompts.chat import (
...     ChatPromptTemplate,
...     HumanMessagePromptTemplate,
...     SystemMessagePromptTemplate,
... )
>>>
>>> # 创建消息模板
>>> system = SystemMessagePromptTemplate.from_template("你是一个{role}")
>>> human = HumanMessagePromptTemplate.from_template("{input}")
>>>
>>> # 组合成 ChatPromptTemplate
>>> chat_prompt = ChatPromptTemplate.from_messages([system, human])
>>> chat_prompt.format_messages(role="助手", input="你好")
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from langchain_core.load import Serializable
from langchain_core.utils.interactive_env import is_interactive_env

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage
    from langchain_core.prompts.chat import ChatPromptTemplate


class BaseMessagePromptTemplate(Serializable, ABC):
    """消息提示模板的基类。

    这是一个抽象基类，定义了消息提示模板的标准接口。
    所有消息提示模板都必须实现 `format_messages()` 方法和 `input_variables` 属性。

    与 StringPromptTemplate 的区别:
    - StringPromptTemplate.format() 返回 str
    - BaseMessagePromptTemplate.format_messages() 返回 list[BaseMessage]

    核心方法:
    ---------
    - format_messages(**kwargs): 格式化并返回消息列表
    - aformat_messages(**kwargs): 异步版本

    核心属性:
    ---------
    - input_variables: 此模板需要的输入变量列表

    使用示例:
    ---------
    >>> from langchain_core.prompts.chat import HumanMessagePromptTemplate
    >>>
    >>> template = HumanMessagePromptTemplate.from_template("你好，{name}！")
    >>> messages = template.format_messages(name="世界")
    >>> messages[0].content
    "你好，世界！"
    """

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """返回 True，表示此类可序列化。

        这使得消息模板可以被保存和加载。
        """
        return True

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """获取 LangChain 对象的命名空间。

        Returns:
            `["langchain", "prompts", "chat"]`
        """
        return ["langchain", "prompts", "chat"]

    @abstractmethod
    def format_messages(self, **kwargs: Any) -> list[BaseMessage]:
        """从关键字参数格式化消息，返回 BaseMessage 对象列表。

        这是消息模板的核心方法，必须由子类实现。

        Args:
            **kwargs: 用于格式化的关键字参数。

        Returns:
            BaseMessage 对象列表。

        示例:
            >>> template = HumanMessagePromptTemplate.from_template("{question}")
            >>> messages = template.format_messages(question="什么是 AI？")
            >>> len(messages)
            1
            >>> messages[0].content
            "什么是 AI？"
        """

    async def aformat_messages(self, **kwargs: Any) -> list[BaseMessage]:
        """异步格式化消息。

        默认实现调用同步版本。子类可以覆盖此方法以提供真正的异步实现。

        Args:
            **kwargs: 用于格式化的关键字参数。

        Returns:
            BaseMessage 对象列表。
        """
        return self.format_messages(**kwargs)

    @property
    @abstractmethod
    def input_variables(self) -> list[str]:
        """此提示模板的输入变量。

        返回此模板需要的所有输入变量名称列表。
        这用于验证输入和自动推断 ChatPromptTemplate 的输入变量。

        Returns:
            输入变量名称列表。
        """

    def pretty_repr(
        self,
        html: bool = False,  # noqa: FBT001,FBT002
    ) -> str:
        """获取人类可读的表示。

        Args:
            html: 是否格式化为 HTML。

        Returns:
            人类可读的字符串表示。
        """
        raise NotImplementedError

    def pretty_print(self) -> None:
        """打印人类可读的表示。

        在交互式环境（如 Jupyter）中会使用 HTML 格式。
        """
        print(self.pretty_repr(html=is_interactive_env()))  # noqa: T201

    def __add__(self, other: Any) -> ChatPromptTemplate:
        """组合两个提示模板。

        使用 + 运算符可以将消息模板与其他元素组合成 ChatPromptTemplate。

        Args:
            other: 另一个提示模板或消息。

        Returns:
            组合后的 ChatPromptTemplate。

        示例:
            >>> from langchain_core.prompts.chat import (
            ...     SystemMessagePromptTemplate,
            ...     HumanMessagePromptTemplate,
            ... )
            >>> system = SystemMessagePromptTemplate.from_template("你是助手")
            >>> human = HumanMessagePromptTemplate.from_template("{input}")
            >>> combined = system + human  # 自动创建 ChatPromptTemplate
        """
        # 本地导入以避免循环导入
        from langchain_core.prompts.chat import ChatPromptTemplate  # noqa: PLC0415

        # 先创建一个包含 self 的 ChatPromptTemplate
        prompt = ChatPromptTemplate(messages=[self])
        # 然后使用 ChatPromptTemplate 的 __add__ 方法组合
        return prompt.__add__(other)
