"""结构化输出提示模板模块。

本模块定义了 `StructuredPrompt`，用于与 LLM 的结构化输出功能配合使用。
这使得从语言模型获取符合预定义 schema 的输出变得简单。

核心概念:
---------
结构化输出是指让 LLM 返回符合特定格式（如 JSON Schema 或 Pydantic 模型）的数据，
而不是自由格式的文本。这对于:
- 数据提取
- 函数调用
- 工具使用
- API 响应生成
等场景非常有用。

工作原理:
---------
StructuredPrompt 继承自 ChatPromptTemplate，但重载了 `|` 和 `pipe` 方法。
当与语言模型连接时，它会自动调用模型的 `with_structured_output()` 方法。

使用示例:
---------
>>> from pydantic import BaseModel
>>> from langchain_core.prompts.structured import StructuredPrompt
>>> from langchain_openai import ChatOpenAI
>>>
>>> # 定义输出 schema
>>> class Person(BaseModel):
...     name: str
...     age: int
...     city: str
>>>
>>> # 创建结构化提示
>>> prompt = StructuredPrompt.from_messages_and_schema(
...     messages=[
...         ("system", "从文本中提取人物信息"),
...         ("human", "{text}")
...     ],
...     schema=Person
... )
>>>
>>> # 创建链（自动添加结构化输出）
>>> llm = ChatOpenAI(model="gpt-4")
>>> chain = prompt | llm
>>>
>>> # 调用
>>> result = chain.invoke({"text": "小明今年25岁，住在北京"})
>>> print(result)  # Person(name="小明", age=25, city="北京")

注意事项:
---------
1. StructuredPrompt 必须连接到支持 `with_structured_output()` 的模型
2. 这是一个 beta 功能，API 可能会变化
"""

from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from typing import (
    Any,
)

from pydantic import BaseModel, Field
from typing_extensions import override

from langchain_core._api.beta_decorator import beta
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    MessageLikeRepresentation,
)
from langchain_core.prompts.string import PromptTemplateFormat
from langchain_core.runnables.base import (
    Other,
    Runnable,
    RunnableSequence,
    RunnableSerializable,
)
from langchain_core.utils import get_pydantic_field_names


@beta()
class StructuredPrompt(ChatPromptTemplate):
    """用于语言模型的结构化输出提示模板。

    这是 ChatPromptTemplate 的特殊版本，专门用于结构化输出场景。
    当与语言模型通过管道连接时，它会自动调用模型的
    `with_structured_output()` 方法来启用结构化输出。

    属性:
    -----
    schema_ : dict | type
        输出的 schema 定义。可以是:
        - Pydantic 模型类
        - JSON Schema 字典
    structured_output_kwargs : dict[str, Any]
        传递给 `with_structured_output()` 的额外参数

    使用示例:
    ---------
    >>> from pydantic import BaseModel
    >>> from langchain_core.prompts import StructuredPrompt
    >>>
    >>> class OutputSchema(BaseModel):
    ...     answer: str
    ...     confidence: float
    >>>
    >>> prompt = StructuredPrompt(
    ...     messages=[("human", "回答问题: {question}")],
    ...     schema_=OutputSchema
    ... )
    >>>
    >>> # 与 LLM 连接时自动启用结构化输出
    >>> chain = prompt | llm
    """

    schema_: dict | type
    """结构化提示的 schema 定义。

    这定义了期望的输出结构。可以是:
    - Pydantic BaseModel 子类（推荐）
    - JSON Schema 字典

    示例:
        class MySchema(BaseModel):
            field1: str
            field2: int
    """

    structured_output_kwargs: dict[str, Any] = Field(default_factory=dict)
    """传递给 `with_structured_output()` 的额外参数。

    这些参数在将提示连接到语言模型时使用。
    常见参数包括:
    - method: 结构化输出的方法（如 "function_calling"）
    - include_raw: 是否包含原始响应
    """

    def __init__(
        self,
        messages: Sequence[MessageLikeRepresentation],
        schema_: dict | type[BaseModel] | None = None,
        *,
        structured_output_kwargs: dict[str, Any] | None = None,
        template_format: PromptTemplateFormat = "f-string",
        **kwargs: Any,
    ) -> None:
        """创建结构化提示模板。

        Args:
            messages: 消息序列，定义提示的结构。
            schema_: 输出 schema，可以是 Pydantic 模型或字典。
            structured_output_kwargs: 传递给 with_structured_output 的额外参数。
            template_format: 模板格式。
            **kwargs: 其他参数。不属于模型字段的参数会被
                添加到 structured_output_kwargs。

        Raises:
            ValueError: 如果没有提供 schema。
        """
        # 支持使用 "schema" 作为参数名（向后兼容）
        schema_ = schema_ or kwargs.pop("schema", None)
        if not schema_:
            err_msg = (
                "必须传入非空的结构化输出 schema。收到: "
                f"{schema_}"
            )
            raise ValueError(err_msg)

        structured_output_kwargs = structured_output_kwargs or {}

        # 将不属于模型字段的 kwargs 移到 structured_output_kwargs
        for k in set(kwargs).difference(get_pydantic_field_names(self.__class__)):
            structured_output_kwargs[k] = kwargs.pop(k)

        super().__init__(
            messages=messages,
            schema_=schema_,
            structured_output_kwargs=structured_output_kwargs,
            template_format=template_format,
            **kwargs,
        )

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """获取 LangChain 对象的命名空间。

        例如，如果类是 `langchain.llms.openai.OpenAI`，
        则命名空间是 `["langchain", "llms", "openai"]`。

        Returns:
            LangChain 对象的命名空间。
        """
        return cls.__module__.split(".")

    @classmethod
    def from_messages_and_schema(
        cls,
        messages: Sequence[MessageLikeRepresentation],
        schema: dict | type,
        **kwargs: Any,
    ) -> ChatPromptTemplate:
        """从消息和 schema 创建聊天提示模板。

        这是创建 StructuredPrompt 的推荐方式。

        Args:
            messages: 消息序列，可以是以下格式:
                1. `BaseMessagePromptTemplate`
                2. `BaseMessage`
                3. 2-元组 `(消息类型, 模板)`；如 `("human", "{user_input}")`
                4. 2-元组 `(消息类, 模板)`
                5. 字符串，等同于 `("human", 字符串)`
            schema: 字典形式的函数调用表示，或 Pydantic 模型。
            **kwargs: 传递给 `ChatModel.with_structured_output(schema, **kwargs)`
                的额外参数。

        Returns:
            结构化提示模板。

        使用示例:
            >>> from pydantic import BaseModel
            >>> from langchain_core.prompts import StructuredPrompt
            >>>
            >>> class OutputSchema(BaseModel):
            ...     name: str
            ...     value: int
            >>>
            >>> template = StructuredPrompt.from_messages_and_schema(
            ...     messages=[
            ...         ("human", "你好，你怎么样？"),
            ...         ("ai", "我很好，谢谢！"),
            ...         ("human", "很高兴听到这个。"),
            ...     ],
            ...     schema=OutputSchema,
            ... )
        """
        return cls(messages, schema, **kwargs)

    @override
    def __or__(
        self,
        other: Runnable[Any, Other]
        | Callable[[Iterator[Any]], Iterator[Other]]
        | Callable[[AsyncIterator[Any]], AsyncIterator[Other]]
        | Callable[[Any], Other]
        | Mapping[str, Runnable[Any, Other] | Callable[[Any], Other] | Any],
    ) -> RunnableSerializable[dict, Other]:
        """重载管道运算符 |。

        当使用 `prompt | llm` 语法时调用此方法。
        它会自动将结构化输出配置应用到语言模型。
        """
        return self.pipe(other)

    def pipe(
        self,
        *others: Runnable[Any, Other]
        | Callable[[Iterator[Any]], Iterator[Other]]
        | Callable[[AsyncIterator[Any]], AsyncIterator[Other]]
        | Callable[[Any], Other]
        | Mapping[str, Runnable[Any, Other] | Callable[[Any], Other] | Any],
        name: str | None = None,
    ) -> RunnableSerializable[dict, Other]:
        """将结构化提示连接到语言模型。

        这是 LCEL 管道连接的核心方法。当连接到语言模型时，
        它会自动调用模型的 `with_structured_output()` 方法。

        Args:
            others: 要连接的 Runnable 对象，第一个应该是语言模型。
            name: 管道的名称。

        Returns:
            RunnableSequence 对象。

        Raises:
            NotImplementedError: 如果 `others` 的第一个元素不是语言模型。

        工作原理:
            ```python
            prompt | llm
            ```
            等同于:
            ```python
            prompt | llm.with_structured_output(schema, **kwargs)
            ```
        """
        # 检查第一个元素是否是语言模型或有 with_structured_output 方法
        if (others and isinstance(others[0], BaseLanguageModel)) or hasattr(
            others[0], "with_structured_output"
        ):
            return RunnableSequence(
                self,
                # 自动应用结构化输出
                others[0].with_structured_output(
                    self.schema_, **self.structured_output_kwargs
                ),
                *others[1:],
                name=name,
            )
        msg = "结构化提示需要连接到语言模型。"
        raise NotImplementedError(msg)
