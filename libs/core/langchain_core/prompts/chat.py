"""聊天提示模板模块。

本模块提供了与聊天模型交互的提示模板类。
这是 LangChain 中最常用的模块之一，用于构建发送给 ChatGPT、Claude 等聊天模型的消息。

核心类:
--------
1. **ChatPromptTemplate**: 聊天提示模板（最常用）
2. **MessagesPlaceholder**: 消息列表占位符（用于对话历史）
3. **HumanMessagePromptTemplate**: 用户消息模板
4. **AIMessagePromptTemplate**: AI 消息模板
5. **SystemMessagePromptTemplate**: 系统消息模板

消息类型:
---------
聊天模型使用不同类型的消息:
- **SystemMessage**: 系统消息，设定 AI 的行为和上下文
- **HumanMessage**: 用户消息
- **AIMessage**: AI 的回复
- **ChatMessage**: 自定义角色的消息

使用示例:
---------
>>> from langchain_core.prompts import ChatPromptTemplate
>>>
>>> # 方式1: 使用元组列表（推荐）
>>> prompt = ChatPromptTemplate.from_messages([
...     ("system", "你是一个专业的{role}"),
...     ("human", "{input}"),
... ])
>>> messages = prompt.format_messages(role="翻译", input="你好世界")
>>>
>>> # 方式2: 使用消息类
>>> from langchain_core.prompts import (
...     SystemMessagePromptTemplate,
...     HumanMessagePromptTemplate,
... )
>>> prompt = ChatPromptTemplate.from_messages([
...     SystemMessagePromptTemplate.from_template("你是助手"),
...     HumanMessagePromptTemplate.from_template("{input}"),
... ])
>>>
>>> # 方式3: 使用消息占位符（用于对话历史）
>>> from langchain_core.prompts import MessagesPlaceholder
>>> prompt = ChatPromptTemplate.from_messages([
...     ("system", "你是助手"),
...     MessagesPlaceholder("history"),  # 动态插入历史消息
...     ("human", "{input}"),
... ])

与 LCEL 链结合使用:
---------
>>> from langchain_openai import ChatOpenAI
>>>
>>> llm = ChatOpenAI()
>>> prompt = ChatPromptTemplate.from_template("讲一个关于{topic}的笑话")
>>> chain = prompt | llm
>>> chain.invoke({"topic": "程序员"})
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import (
    Annotated,
    Any,
    TypedDict,
    TypeVar,
    cast,
    overload,
)

from pydantic import (
    Field,
    PositiveInt,
    SkipValidation,
    model_validator,
)
from typing_extensions import Self, override

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
    convert_to_messages,
)
from langchain_core.messages.base import get_msg_title_repr
from langchain_core.prompt_values import ChatPromptValue, ImageURL
from langchain_core.prompts.base import BasePromptTemplate
from langchain_core.prompts.dict import DictPromptTemplate
from langchain_core.prompts.image import ImagePromptTemplate
from langchain_core.prompts.message import (
    BaseMessagePromptTemplate,
)
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.prompts.string import (
    PromptTemplateFormat,
    StringPromptTemplate,
    get_template_variables,
)
from langchain_core.utils import get_colored_text
from langchain_core.utils.interactive_env import is_interactive_env


class MessagesPlaceholder(BaseMessagePromptTemplate):
    """消息列表占位符。

    用于在 ChatPromptTemplate 中插入消息列表的占位符。
    这在需要动态传入对话历史时非常有用。

    属性:
    -----
    variable_name : str
        变量名称，用于接收消息列表
    optional : bool
        是否可选。如果为 True，可以不提供此变量
    n_messages : int | None
        最大消息数量。如果设置，只保留最后 n 条消息

    使用示例 - 直接使用:
    ---------
    >>> from langchain_core.prompts import MessagesPlaceholder
    >>>
    >>> # 必需占位符
    >>> placeholder = MessagesPlaceholder("history")
    >>> placeholder.format_messages()  # 抛出 KeyError
    >>>
    >>> # 可选占位符
    >>> placeholder = MessagesPlaceholder("history", optional=True)
    >>> placeholder.format_messages()  # 返回空列表 []
    >>>
    >>> # 带消息格式化
    >>> placeholder.format_messages(
    ...     history=[
    ...         ("system", "你是 AI 助手"),
    ...         ("human", "你好！"),
    ...     ]
    ... )
    # -> [SystemMessage(content="你是 AI 助手"), HumanMessage(content="你好！")]

    使用示例 - 在 ChatPromptTemplate 中使用:
    ---------
    >>> from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    >>>
    >>> prompt = ChatPromptTemplate.from_messages([
    ...     ("system", "你是一个有用的助手。"),
    ...     MessagesPlaceholder("history"),
    ...     ("human", "{question}"),
    ... ])
    >>>
    >>> prompt.invoke({
    ...     "history": [("human", "5+2是多少"), ("ai", "7")],
    ...     "question": "乘以4是多少",
    ... })
    # -> ChatPromptValue(messages=[
    #     SystemMessage(content="你是一个有用的助手。"),
    #     HumanMessage(content="5+2是多少"),
    #     AIMessage(content="7"),
    #     HumanMessage(content="乘以4是多少"),
    # ])

    使用示例 - 限制消息数量:
    ---------
    >>> placeholder = MessagesPlaceholder("history", n_messages=1)
    >>>
    >>> placeholder.format_messages(
    ...     history=[
    ...         ("system", "你是 AI 助手"),
    ...         ("human", "你好！"),
    ...     ]
    ... )
    # -> [HumanMessage(content="你好！")]  # 只保留最后1条
    """

    variable_name: str
    """用作消息的变量名称。"""

    optional: bool = False
    """是否为可选。

    如果为 True，可以不传入此变量，format_messages 会返回空列表。
    如果为 False（默认），必须传入此变量，即使是空列表。
    """

    n_messages: PositiveInt | None = None
    """要包含的最大消息数量。

    如果为 None，包含所有消息。
    如果设置了值，只保留最后 n 条消息。
    """

    def __init__(
        self, variable_name: str, *, optional: bool = False, **kwargs: Any
    ) -> None:
        """创建消息占位符。

        Args:
            variable_name: 变量名称。
            optional: 是否可选。
            **kwargs: 其他参数。
        """
        super().__init__(variable_name=variable_name, optional=optional, **kwargs)

    def format_messages(self, **kwargs: Any) -> list[BaseMessage]:
        """从关键字参数格式化消息。

        Args:
            **kwargs: 包含变量值的关键字参数。

        Returns:
            BaseMessage 列表。

        Raises:
            ValueError: 如果变量不是消息列表。
        """
        # 如果是可选的且未提供，返回空列表
        value = (
            kwargs.get(self.variable_name, [])
            if self.optional
            else kwargs[self.variable_name]
        )
        if not isinstance(value, list):
            msg = (
                f"变量 {self.variable_name} 应该是 BaseMessage 列表，"
                f"但收到了 {type(value)} 类型的 {value}"
            )
            raise ValueError(msg)
        # 转换为消息对象
        value = convert_to_messages(value)
        # 如果设置了限制，只保留最后 n 条
        if self.n_messages:
            value = value[-self.n_messages :]
        return value

    @property
    def input_variables(self) -> list[str]:
        """此模板的输入变量。"""
        return [self.variable_name] if not self.optional else []

    @override
    def pretty_repr(self, html: bool = False) -> str:
        """人类可读的表示。"""
        var = "{" + self.variable_name + "}"
        if html:
            title = get_msg_title_repr("Messages Placeholder", bold=True)
            var = get_colored_text(var, "yellow")
        else:
            title = get_msg_title_repr("Messages Placeholder")
        return f"{title}\n\n{var}"


# 消息模板的类型变量
MessagePromptTemplateT = TypeVar(
    "MessagePromptTemplateT", bound="BaseStringMessagePromptTemplate"
)


class BaseStringMessagePromptTemplate(BaseMessagePromptTemplate, ABC):
    """使用字符串提示模板的消息模板基类。

    这是所有单消息模板的基类，包含一个 StringPromptTemplate。
    """

    prompt: StringPromptTemplate
    """字符串提示模板。"""

    additional_kwargs: dict = Field(default_factory=dict)
    """传递给消息的额外关键字参数。"""

    @classmethod
    def from_template(
        cls,
        template: str,
        template_format: PromptTemplateFormat = "f-string",
        partial_variables: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Self:
        """从字符串模板创建实例。

        Args:
            template: 模板字符串。
            template_format: 模板格式。
            partial_variables: 部分变量字典。
            **kwargs: 传递给构造函数的参数。

        Returns:
            此类的新实例。
        """
        prompt = PromptTemplate.from_template(
            template,
            template_format=template_format,
            partial_variables=partial_variables,
        )
        return cls(prompt=prompt, **kwargs)

    @classmethod
    def from_template_file(
        cls,
        template_file: str | Path,
        **kwargs: Any,
    ) -> Self:
        """从模板文件创建实例。

        Args:
            template_file: 模板文件路径。
            **kwargs: 传递给构造函数的参数。

        Returns:
            此类的新实例。
        """
        prompt = PromptTemplate.from_file(template_file)
        return cls(prompt=prompt, **kwargs)

    @abstractmethod
    def format(self, **kwargs: Any) -> BaseMessage:
        """格式化提示模板。

        Args:
            **kwargs: 格式化参数。

        Returns:
            格式化后的消息。
        """

    async def aformat(self, **kwargs: Any) -> BaseMessage:
        """异步格式化提示模板。"""
        return self.format(**kwargs)

    def format_messages(self, **kwargs: Any) -> list[BaseMessage]:
        """从参数格式化消息。返回单个消息的列表。"""
        return [self.format(**kwargs)]

    async def aformat_messages(self, **kwargs: Any) -> list[BaseMessage]:
        """异步格式化消息。"""
        return [await self.aformat(**kwargs)]

    @property
    def input_variables(self) -> list[str]:
        """此模板的输入变量。"""
        return self.prompt.input_variables

    @override
    def pretty_repr(self, html: bool = False) -> str:
        """人类可读的表示。"""
        title = self.__class__.__name__.replace("MessagePromptTemplate", " Message")
        title = get_msg_title_repr(title, bold=html)
        return f"{title}\n\n{self.prompt.pretty_repr(html=html)}"


class ChatMessagePromptTemplate(BaseStringMessagePromptTemplate):
    """自定义角色的聊天消息模板。"""

    role: str
    """消息的角色。"""

    def format(self, **kwargs: Any) -> BaseMessage:
        """格式化提示模板。"""
        text = self.prompt.format(**kwargs)
        return ChatMessage(
            content=text, role=self.role, additional_kwargs=self.additional_kwargs
        )

    async def aformat(self, **kwargs: Any) -> BaseMessage:
        """异步格式化提示模板。"""
        text = await self.prompt.aformat(**kwargs)
        return ChatMessage(
            content=text, role=self.role, additional_kwargs=self.additional_kwargs
        )


# 用于模板参数类型定义的 TypedDict
class _TextTemplateParam(TypedDict, total=False):
    text: str | dict


class _ImageTemplateParam(TypedDict, total=False):
    image_url: str | dict


class _StringImageMessagePromptTemplate(BaseMessagePromptTemplate):
    """支持字符串和图像的消息提示模板。

    这是 HumanMessagePromptTemplate、AIMessagePromptTemplate 和
    SystemMessagePromptTemplate 的基类。

    支持两种模板格式:
    1. 纯字符串模板: "你好，{name}"
    2. 多模态内容列表: [{"type": "text", "text": "..."}, {"type": "image_url", ...}]
    """

    prompt: (
        StringPromptTemplate
        | list[StringPromptTemplate | ImagePromptTemplate | DictPromptTemplate]
    )
    """提示模板，可以是单个字符串模板或多模态内容列表。"""

    additional_kwargs: dict = Field(default_factory=dict)
    """传递给消息的额外参数。"""

    _msg_class: type[BaseMessage]
    """要创建的消息类。"""

    @classmethod
    def from_template(
        cls: type[Self],
        template: str
        | list[str | _TextTemplateParam | _ImageTemplateParam | dict[str, Any]],
        template_format: PromptTemplateFormat = "f-string",
        *,
        partial_variables: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Self:
        """从字符串模板创建实例。

        支持两种模板格式:
        1. 单个字符串: "你好，{name}"
        2. 内容列表（多模态）:
           [
               "文本内容",
               {"type": "text", "text": "更多文本"},
               {"type": "image_url", "image_url": {"url": "..."}},
           ]

        Args:
            template: 模板字符串或内容列表。
            template_format: 模板格式（f-string、mustache、jinja2）。
            partial_variables: 部分变量。
            **kwargs: 其他参数。

        Returns:
            新实例。

        Raises:
            ValueError: 如果模板格式无效。
        """
        if isinstance(template, str):
            # 纯字符串模板
            prompt: StringPromptTemplate | list = PromptTemplate.from_template(
                template,
                template_format=template_format,
                partial_variables=partial_variables,
            )
            return cls(prompt=prompt, **kwargs)

        if isinstance(template, list):
            # 多模态内容列表
            if (partial_variables is not None) and len(partial_variables) > 0:
                msg = "列表模板不支持部分变量。"
                raise ValueError(msg)

            prompt = []
            for tmpl in template:
                if isinstance(tmpl, str) or (
                    isinstance(tmpl, dict)
                    and "text" in tmpl
                    and set(tmpl.keys()) <= {"type", "text"}
                ):
                    # 文本内容
                    if isinstance(tmpl, str):
                        text: str = tmpl
                    else:
                        text = cast("_TextTemplateParam", tmpl)["text"]
                    prompt.append(
                        PromptTemplate.from_template(
                            text, template_format=template_format
                        )
                    )
                elif (
                    isinstance(tmpl, dict)
                    and "image_url" in tmpl
                    and set(tmpl.keys()) <= {"type", "image_url"}
                ):
                    # 图像内容
                    img_template = cast("_ImageTemplateParam", tmpl)["image_url"]
                    input_variables = []
                    if isinstance(img_template, str):
                        variables = get_template_variables(
                            img_template, template_format
                        )
                        if variables:
                            if len(variables) > 1:
                                msg = (
                                    "每个图像模板只允许一个格式变量。"
                                    f"\n得到: {variables}"
                                    f"\n来自: {tmpl}"
                                )
                                raise ValueError(msg)
                            input_variables = [variables[0]]
                        img_template = {"url": img_template}
                        img_template_obj = ImagePromptTemplate(
                            input_variables=input_variables,
                            template=img_template,
                            template_format=template_format,
                        )
                    elif isinstance(img_template, dict):
                        img_template = dict(img_template)
                        for key in ["url", "path", "detail"]:
                            if key in img_template:
                                input_variables.extend(
                                    get_template_variables(
                                        img_template[key], template_format
                                    )
                                )
                        img_template_obj = ImagePromptTemplate(
                            input_variables=input_variables,
                            template=img_template,
                            template_format=template_format,
                        )
                    else:
                        msg = f"无效的图像模板: {tmpl}"
                        raise ValueError(msg)
                    prompt.append(img_template_obj)
                elif isinstance(tmpl, dict):
                    # 字典模板
                    if template_format == "jinja2":
                        msg = (
                            "jinja2 不安全，不支持表示为字典的模板。"
                            "请使用 'f-string' 或 'mustache' 格式。"
                        )
                        raise ValueError(msg)
                    data_template_obj = DictPromptTemplate(
                        template=cast("dict[str, Any]", tmpl),
                        template_format=template_format,
                    )
                    prompt.append(data_template_obj)
                else:
                    msg = f"无效的模板: {tmpl}"
                    raise ValueError(msg)
            return cls(prompt=prompt, **kwargs)

        msg = f"无效的模板: {template}"
        raise ValueError(msg)

    @classmethod
    def from_template_file(
        cls: type[Self],
        template_file: str | Path,
        input_variables: list[str],
        **kwargs: Any,
    ) -> Self:
        """从模板文件创建实例。"""
        template = Path(template_file).read_text(encoding="utf-8")
        return cls.from_template(template, input_variables=input_variables, **kwargs)

    def format_messages(self, **kwargs: Any) -> list[BaseMessage]:
        """格式化消息。"""
        return [self.format(**kwargs)]

    async def aformat_messages(self, **kwargs: Any) -> list[BaseMessage]:
        """异步格式化消息。"""
        return [await self.aformat(**kwargs)]

    @property
    def input_variables(self) -> list[str]:
        """输入变量。"""
        prompts = self.prompt if isinstance(self.prompt, list) else [self.prompt]
        return [iv for prompt in prompts for iv in prompt.input_variables]

    def format(self, **kwargs: Any) -> BaseMessage:
        """格式化提示模板。

        可以处理纯字符串模板或多模态内容列表。
        """
        if isinstance(self.prompt, StringPromptTemplate):
            # 纯字符串模板
            text = self.prompt.format(**kwargs)
            return self._msg_class(
                content=text, additional_kwargs=self.additional_kwargs
            )

        # 多模态内容列表
        content: list = []
        for prompt in self.prompt:
            inputs = {var: kwargs[var] for var in prompt.input_variables}
            if isinstance(prompt, StringPromptTemplate):
                formatted_text: str = prompt.format(**inputs)
                if formatted_text != "":
                    content.append({"type": "text", "text": formatted_text})
            elif isinstance(prompt, ImagePromptTemplate):
                formatted_image: ImageURL = prompt.format(**inputs)
                content.append({"type": "image_url", "image_url": formatted_image})
            elif isinstance(prompt, DictPromptTemplate):
                formatted_dict: dict[str, Any] = prompt.format(**inputs)
                content.append(formatted_dict)

        return self._msg_class(
            content=content, additional_kwargs=self.additional_kwargs
        )

    async def aformat(self, **kwargs: Any) -> BaseMessage:
        """异步格式化提示模板。"""
        if isinstance(self.prompt, StringPromptTemplate):
            text = await self.prompt.aformat(**kwargs)
            return self._msg_class(
                content=text, additional_kwargs=self.additional_kwargs
            )

        content: list = []
        for prompt in self.prompt:
            inputs = {var: kwargs[var] for var in prompt.input_variables}
            if isinstance(prompt, StringPromptTemplate):
                formatted_text: str = await prompt.aformat(**inputs)
                if formatted_text != "":
                    content.append({"type": "text", "text": formatted_text})
            elif isinstance(prompt, ImagePromptTemplate):
                formatted_image: ImageURL = await prompt.aformat(**inputs)
                content.append({"type": "image_url", "image_url": formatted_image})
            elif isinstance(prompt, DictPromptTemplate):
                formatted_dict: dict[str, Any] = prompt.format(**inputs)
                content.append(formatted_dict)

        return self._msg_class(
            content=content, additional_kwargs=self.additional_kwargs
        )

    @override
    def pretty_repr(self, html: bool = False) -> str:
        """人类可读的表示。"""
        title = self.__class__.__name__.replace("MessagePromptTemplate", " Message")
        title = get_msg_title_repr(title, bold=html)
        prompts = self.prompt if isinstance(self.prompt, list) else [self.prompt]
        prompt_reprs = "\n\n".join(prompt.pretty_repr(html=html) for prompt in prompts)
        return f"{title}\n\n{prompt_reprs}"


class HumanMessagePromptTemplate(_StringImageMessagePromptTemplate):
    """用户消息提示模板。

    用于创建用户发送的消息。

    使用示例:
        >>> template = HumanMessagePromptTemplate.from_template("{input}")
        >>> template.format(input="你好")
        HumanMessage(content="你好")
    """

    _msg_class: type[BaseMessage] = HumanMessage


class AIMessagePromptTemplate(_StringImageMessagePromptTemplate):
    """AI 消息提示模板。

    用于创建 AI 的回复消息。
    """

    _msg_class: type[BaseMessage] = AIMessage


class SystemMessagePromptTemplate(_StringImageMessagePromptTemplate):
    """系统消息提示模板。

    系统消息用于设定 AI 的行为和上下文，不直接发送给用户。

    使用示例:
        >>> template = SystemMessagePromptTemplate.from_template("你是一个{role}")
        >>> template.format(role="翻译")
        SystemMessage(content="你是一个翻译")
    """

    _msg_class: type[BaseMessage] = SystemMessage


class BaseChatPromptTemplate(BasePromptTemplate, ABC):
    """聊天提示模板的基类。"""

    @property
    @override
    def lc_attributes(self) -> dict:
        return {"input_variables": self.input_variables}

    def format(self, **kwargs: Any) -> str:
        """将聊天模板格式化为字符串。"""
        return self.format_prompt(**kwargs).to_string()

    async def aformat(self, **kwargs: Any) -> str:
        """异步将聊天模板格式化为字符串。"""
        return (await self.aformat_prompt(**kwargs)).to_string()

    def format_prompt(self, **kwargs: Any) -> ChatPromptValue:
        """格式化提示，返回 ChatPromptValue。"""
        messages = self.format_messages(**kwargs)
        return ChatPromptValue(messages=messages)

    async def aformat_prompt(self, **kwargs: Any) -> ChatPromptValue:
        """异步格式化提示。"""
        messages = await self.aformat_messages(**kwargs)
        return ChatPromptValue(messages=messages)

    @abstractmethod
    def format_messages(self, **kwargs: Any) -> list[BaseMessage]:
        """将参数格式化为消息列表。"""

    async def aformat_messages(self, **kwargs: Any) -> list[BaseMessage]:
        """异步格式化消息。"""
        return self.format_messages(**kwargs)

    def pretty_repr(
        self,
        html: bool = False,
    ) -> str:
        """人类可读的表示。"""
        raise NotImplementedError

    def pretty_print(self) -> None:
        """打印人类可读的表示。"""
        print(self.pretty_repr(html=is_interactive_env()))


# 消息类型别名
MessageLike = BaseMessagePromptTemplate | BaseMessage | BaseChatPromptTemplate

# 消息表示类型（支持多种输入格式）
MessageLikeRepresentation = (
    MessageLike
    | tuple[str | type, str | Sequence[dict] | Sequence[object]]
    | str
    | dict[str, Any]
)


class ChatPromptTemplate(BaseChatPromptTemplate):
    """聊天模型的提示模板。

    这是 LangChain 中最常用的类之一，用于为聊天模型创建灵活的模板化提示。

    消息格式:
    ---------
    消息可以用以下格式表示:
    1. BaseMessagePromptTemplate 对象
    2. BaseMessage 对象
    3. 2-元组 (消息类型, 模板): 如 ("human", "{user_input}")
    4. 2-元组 (消息类, 模板)
    5. 字符串: 等同于 ("human", 字符串)

    支持的消息类型:
    - "system": SystemMessage
    - "human" / "user": HumanMessage
    - "ai" / "assistant": AIMessage
    - "placeholder": MessagesPlaceholder

    使用示例:
    ---------
    >>> from langchain_core.prompts import ChatPromptTemplate
    >>>
    >>> template = ChatPromptTemplate([
    ...     ("system", "你是一个有用的 AI 助手，名字是 {name}。"),
    ...     ("human", "你好，你怎么样？"),
    ...     ("ai", "我很好，谢谢！"),
    ...     ("human", "{user_input}"),
    ... ])
    >>>
    >>> prompt_value = template.invoke({
    ...     "name": "小助",
    ...     "user_input": "你叫什么名字？",
    ... })
    # 输出:
    # ChatPromptValue(messages=[
    #    SystemMessage(content='你是一个有用的 AI 助手，名字是 小助。'),
    #    HumanMessage(content='你好，你怎么样？'),
    #    AIMessage(content="我很好，谢谢！"),
    #    HumanMessage(content='你叫什么名字？')
    # ])

    使用消息占位符:
    ---------
    >>> template = ChatPromptTemplate([
    ...     ("system", "你是一个有用的 AI 助手。"),
    ...     ("placeholder", "{conversation}"),  # 动态消息列表
    ... ])
    >>>
    >>> prompt_value = template.invoke({
    ...     "conversation": [
    ...         ("human", "你好！"),
    ...         ("ai", "我能帮你什么？"),
    ...     ]
    ... })

    单变量模板:
    ---------
    如果模板只有一个输入变量，可以直接传入值而不是字典:

    >>> template = ChatPromptTemplate([
    ...     ("system", "你是一个有用的助手。"),
    ...     ("human", "{user_input}"),
    ... ])
    >>>
    >>> prompt_value = template.invoke("你好！")
    # 等同于: template.invoke({"user_input": "你好！"})
    """

    messages: Annotated[list[MessageLike], SkipValidation()]
    """消息列表，可以是消息模板或消息对象。"""

    validate_template: bool = False
    """是否验证模板。"""

    def __init__(
        self,
        messages: Sequence[MessageLikeRepresentation],
        *,
        template_format: PromptTemplateFormat = "f-string",
        **kwargs: Any,
    ) -> None:
        """从多种消息格式创建聊天提示模板。

        Args:
            messages: 消息序列，支持多种格式。
            template_format: 模板格式。
            **kwargs: 额外参数。
        """
        # 将所有消息转换为统一格式
        messages_ = [
            _convert_to_message_template(message, template_format)
            for message in messages
        ]

        # 自动推断输入变量
        input_vars: set[str] = set()
        optional_variables: set[str] = set()
        partial_vars: dict[str, Any] = {}

        for _message in messages_:
            if isinstance(_message, MessagesPlaceholder) and _message.optional:
                partial_vars[_message.variable_name] = []
                optional_variables.add(_message.variable_name)
            elif isinstance(
                _message, (BaseChatPromptTemplate, BaseMessagePromptTemplate)
            ):
                input_vars.update(_message.input_variables)

        kwargs = {
            "input_variables": sorted(input_vars),
            "optional_variables": sorted(optional_variables),
            "partial_variables": partial_vars,
            **kwargs,
        }
        cast("type[ChatPromptTemplate]", super()).__init__(messages=messages_, **kwargs)

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """获取命名空间。"""
        return ["langchain", "prompts", "chat"]

    def __add__(self, other: Any) -> ChatPromptTemplate:
        """使用 + 运算符组合提示模板。

        Args:
            other: 另一个模板或消息。

        Returns:
            组合后的 ChatPromptTemplate。
        """
        partials = {**self.partial_variables}

        if hasattr(other, "partial_variables") and other.partial_variables:
            partials.update(other.partial_variables)

        if isinstance(other, ChatPromptTemplate):
            return ChatPromptTemplate(messages=self.messages + other.messages).partial(
                **partials
            )
        if isinstance(
            other, (BaseMessagePromptTemplate, BaseMessage, BaseChatPromptTemplate)
        ):
            return ChatPromptTemplate(messages=[*self.messages, other]).partial(
                **partials
            )
        if isinstance(other, (list, tuple)):
            other_ = ChatPromptTemplate.from_messages(other)
            return ChatPromptTemplate(messages=self.messages + other_.messages).partial(
                **partials
            )
        if isinstance(other, str):
            prompt = HumanMessagePromptTemplate.from_template(other)
            return ChatPromptTemplate(messages=[*self.messages, prompt]).partial(
                **partials
            )
        msg = f"+ 运算符不支持的类型: {type(other)}"
        raise NotImplementedError(msg)

    @model_validator(mode="before")
    @classmethod
    def validate_input_variables(cls, values: dict) -> Any:
        """验证输入变量。"""
        messages = values["messages"]
        input_vars: set = set()
        optional_variables = set()
        input_types: dict[str, Any] = values.get("input_types", {})

        for message in messages:
            if isinstance(message, (BaseMessagePromptTemplate, BaseChatPromptTemplate)):
                input_vars.update(message.input_variables)
            if isinstance(message, MessagesPlaceholder):
                if "partial_variables" not in values:
                    values["partial_variables"] = {}
                if (
                    message.optional
                    and message.variable_name not in values["partial_variables"]
                ):
                    values["partial_variables"][message.variable_name] = []
                    optional_variables.add(message.variable_name)
                if message.variable_name not in input_types:
                    input_types[message.variable_name] = list[AnyMessage]

        if "partial_variables" in values:
            input_vars -= set(values["partial_variables"])
        if optional_variables:
            input_vars -= optional_variables

        if "input_variables" in values and values.get("validate_template"):
            if input_vars != set(values["input_variables"]):
                msg = (
                    f"input_variables 不匹配。"
                    f"期望: {input_vars}。"
                    f"得到: {values['input_variables']}"
                )
                raise ValueError(msg)
        else:
            values["input_variables"] = sorted(input_vars)

        if optional_variables:
            values["optional_variables"] = sorted(optional_variables)
        values["input_types"] = input_types
        return values

    @classmethod
    def from_template(cls, template: str, **kwargs: Any) -> ChatPromptTemplate:
        """从单个模板字符串创建聊天提示（假设为用户消息）。

        Args:
            template: 模板字符串。
            **kwargs: 额外参数。

        Returns:
            新的 ChatPromptTemplate。
        """
        prompt_template = PromptTemplate.from_template(template, **kwargs)
        message = HumanMessagePromptTemplate(prompt=prompt_template)
        return cls.from_messages([message])

    @classmethod
    def from_messages(
        cls,
        messages: Sequence[MessageLikeRepresentation],
        template_format: PromptTemplateFormat = "f-string",
    ) -> ChatPromptTemplate:
        """从消息序列创建聊天提示模板。

        这是创建 ChatPromptTemplate 的推荐方式。

        Args:
            messages: 消息序列，支持多种格式:
                1. BaseMessagePromptTemplate
                2. BaseMessage
                3. 2-元组 (消息类型, 模板): ("human", "{user_input}")
                4. 2-元组 (消息类, 模板)
                5. 字符串: 等同于 ("human", template)
            template_format: 模板格式。

        Returns:
            聊天提示模板。

        使用示例:
            >>> template = ChatPromptTemplate.from_messages([
            ...     ("system", "你是助手"),
            ...     ("human", "{input}"),
            ... ])
        """
        return cls(messages, template_format=template_format)

    def format_messages(self, **kwargs: Any) -> list[BaseMessage]:
        """将模板格式化为消息列表。

        Args:
            **kwargs: 模板变量的值。

        Returns:
            格式化后的消息列表。
        """
        kwargs = self._merge_partial_and_user_variables(**kwargs)
        result = []
        for message_template in self.messages:
            if isinstance(message_template, BaseMessage):
                result.extend([message_template])
            elif isinstance(
                message_template, (BaseMessagePromptTemplate, BaseChatPromptTemplate)
            ):
                message = message_template.format_messages(**kwargs)
                result.extend(message)
            else:
                msg = f"意外的输入: {message_template}"
                raise ValueError(msg)
        return result

    async def aformat_messages(self, **kwargs: Any) -> list[BaseMessage]:
        """异步格式化消息。"""
        kwargs = self._merge_partial_and_user_variables(**kwargs)
        result = []
        for message_template in self.messages:
            if isinstance(message_template, BaseMessage):
                result.extend([message_template])
            elif isinstance(
                message_template, (BaseMessagePromptTemplate, BaseChatPromptTemplate)
            ):
                message = await message_template.aformat_messages(**kwargs)
                result.extend(message)
            else:
                msg = f"意外的输入: {message_template}"
                raise ValueError(msg)
        return result

    def partial(self, **kwargs: Any) -> ChatPromptTemplate:
        """获取已填充部分变量的新 ChatPromptTemplate。

        Args:
            **kwargs: 要预填充的变量。

        Returns:
            新的 ChatPromptTemplate。

        使用示例:
            >>> template = ChatPromptTemplate.from_messages([
            ...     ("system", "你是 {name} 助手"),
            ...     ("human", "{input}"),
            ... ])
            >>> template2 = template.partial(name="小助")
            >>> template2.format_messages(input="你好")
        """
        prompt_dict = self.__dict__.copy()
        prompt_dict["input_variables"] = list(
            set(self.input_variables).difference(kwargs)
        )
        prompt_dict["partial_variables"] = {**self.partial_variables, **kwargs}
        return type(self)(**prompt_dict)

    def append(self, message: MessageLikeRepresentation) -> None:
        """向模板末尾添加消息。

        Args:
            message: 要添加的消息。
        """
        self.messages.append(_convert_to_message_template(message))

    def extend(self, messages: Sequence[MessageLikeRepresentation]) -> None:
        """向模板末尾添加多个消息。

        Args:
            messages: 要添加的消息序列。
        """
        self.messages.extend(
            [_convert_to_message_template(message) for message in messages]
        )

    @overload
    def __getitem__(self, index: int) -> MessageLike: ...

    @overload
    def __getitem__(self, index: slice) -> ChatPromptTemplate: ...

    def __getitem__(self, index: int | slice) -> MessageLike | ChatPromptTemplate:
        """索引访问消息。

        如果是整数索引，返回该位置的消息。
        如果是切片，返回包含切片消息的新 ChatPromptTemplate。
        """
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self.messages))
            messages = self.messages[start:stop:step]
            return ChatPromptTemplate.from_messages(messages)
        return self.messages[index]

    def __len__(self) -> int:
        """返回消息数量。"""
        return len(self.messages)

    @property
    def _prompt_type(self) -> str:
        """提示类型，用于序列化。"""
        return "chat"

    def save(self, file_path: Path | str) -> None:
        """保存提示到文件。"""
        raise NotImplementedError

    @override
    def pretty_repr(self, html: bool = False) -> str:
        """人类可读的表示。"""
        return "\n\n".join(msg.pretty_repr(html=html) for msg in self.messages)


def _create_template_from_message_type(
    message_type: str,
    template: str | list,
    template_format: PromptTemplateFormat = "f-string",
) -> BaseMessagePromptTemplate:
    """从消息类型和模板字符串创建消息模板。

    Args:
        message_type: 消息类型（human、ai、system、placeholder）。
        template: 模板字符串。
        template_format: 模板格式。

    Returns:
        对应类型的消息模板。

    Raises:
        ValueError: 如果消息类型无效。
    """
    if message_type in {"human", "user"}:
        message: BaseMessagePromptTemplate = HumanMessagePromptTemplate.from_template(
            template, template_format=template_format
        )
    elif message_type in {"ai", "assistant"}:
        message = AIMessagePromptTemplate.from_template(
            cast("str", template), template_format=template_format
        )
    elif message_type == "system":
        message = SystemMessagePromptTemplate.from_template(
            cast("str", template), template_format=template_format
        )
    elif message_type == "placeholder":
        if isinstance(template, str):
            if template[0] != "{" or template[-1] != "}":
                msg = (
                    f"无效的占位符模板: {template}。"
                    "期望变量名用花括号包围。"
                )
                raise ValueError(msg)
            var_name = template[1:-1]
            message = MessagesPlaceholder(variable_name=var_name, optional=True)
        else:
            try:
                var_name_wrapped, is_optional = template
            except ValueError as e:
                msg = (
                    "placeholder 消息类型的参数无效。"
                    "期望单个字符串变量名或 [变量名: str, 是否可选: bool] 列表。"
                    f"得到: {template}"
                )
                raise ValueError(msg) from e

            if not isinstance(is_optional, bool):
                msg = f"期望 is_optional 是布尔值，得到: {is_optional}"
                raise ValueError(msg)

            if not isinstance(var_name_wrapped, str):
                msg = f"期望变量名是字符串，得到: {var_name_wrapped}"
                raise ValueError(msg)
            if var_name_wrapped[0] != "{" or var_name_wrapped[-1] != "}":
                msg = (
                    f"无效的占位符模板: {var_name_wrapped}。"
                    "期望变量名用花括号包围。"
                )
                raise ValueError(msg)
            var_name = var_name_wrapped[1:-1]

            message = MessagesPlaceholder(variable_name=var_name, optional=is_optional)
    else:
        msg = (
            f"未知的消息类型: {message_type}。"
            f"请使用 'human'、'user'、'ai'、'assistant' 或 'system'。"
        )
        raise ValueError(msg)
    return message


def _convert_to_message_template(
    message: MessageLikeRepresentation,
    template_format: PromptTemplateFormat = "f-string",
) -> BaseMessage | BaseMessagePromptTemplate | BaseChatPromptTemplate:
    """将多种消息格式转换为消息模板。

    支持的格式:
    - BaseMessagePromptTemplate
    - BaseMessage
    - 2-元组 (角色字符串, 模板): ("human", "{user_input}")
    - 2-元组 (消息类, 模板)
    - 字符串: 等同于 ("human", 模板)

    Args:
        message: 消息的任意表示形式。
        template_format: 模板格式。

    Returns:
        消息或消息模板实例。

    Raises:
        ValueError: 如果消息格式无效。
    """
    if isinstance(message, (BaseMessagePromptTemplate, BaseChatPromptTemplate)):
        message_: BaseMessage | BaseMessagePromptTemplate | BaseChatPromptTemplate = (
            message
        )
    elif isinstance(message, BaseMessage):
        message_ = message
    elif isinstance(message, str):
        # 单个字符串默认为用户消息
        message_ = _create_template_from_message_type(
            "human", message, template_format=template_format
        )
    elif isinstance(message, (tuple, dict)):
        if isinstance(message, dict):
            if set(message.keys()) != {"content", "role"}:
                msg = (
                    f"期望字典有 'role' 和 'content' 键。"
                    f"得到: {message}"
                )
                raise ValueError(msg)
            message_type_str = message["role"]
            template = message["content"]
        else:
            if len(message) != 2:
                msg = f"期望 2-元组 (角色, 模板)，得到 {message}"
                raise ValueError(msg)
            message_type_str, template = message

        if isinstance(message_type_str, str):
            message_ = _create_template_from_message_type(
                message_type_str, template, template_format=template_format
            )
        elif (
            hasattr(message_type_str, "model_fields")
            and "type" in message_type_str.model_fields
        ):
            message_type = message_type_str.model_fields["type"].default
            message_ = _create_template_from_message_type(
                message_type, template, template_format=template_format
            )
        else:
            message_ = message_type_str(
                prompt=PromptTemplate.from_template(
                    cast("str", template), template_format=template_format
                )
            )
    else:
        msg = f"不支持的消息类型: {type(message)}"
        raise NotImplementedError(msg)

    return message_


# 向后兼容别名
_convert_to_message = _convert_to_message_template
