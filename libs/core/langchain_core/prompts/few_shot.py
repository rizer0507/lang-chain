"""包含 Few-shot 示例的提示模板模块。

本模块提供了 Few-shot（少样本）学习的提示模板实现。
Few-shot 学习是一种通过提供少量示例来引导模型行为的技术。

核心概念:
---------
Few-shot 提示的结构:
```
[前缀说明]
[示例 1]
[示例 2]
...
[后缀/用户输入]
```

核心类:
--------
1. **FewShotPromptTemplate**: 用于字符串类型的 few-shot 提示
2. **FewShotChatMessagePromptTemplate**: 用于聊天模型的 few-shot 提示

示例来源:
---------
示例可以通过两种方式提供:
1. **静态示例**: 直接提供示例列表
2. **动态选择**: 使用 ExampleSelector 根据输入动态选择相关示例

使用示例:
---------
>>> from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
>>>
>>> # 定义示例
>>> examples = [
...     {"input": "2+2", "output": "4"},
...     {"input": "3+3", "output": "6"},
... ]
>>>
>>> # 定义单个示例的格式化模板
>>> example_prompt = PromptTemplate.from_template(
...     "问题: {input}\\n答案: {output}"
... )
>>>
>>> # 创建 few-shot 提示
>>> few_shot = FewShotPromptTemplate(
...     examples=examples,
...     example_prompt=example_prompt,
...     prefix="请解答数学问题:",
...     suffix="问题: {input}\\n答案:",
...     input_variables=["input"]
... )
>>>
>>> print(few_shot.format(input="4+4"))
# 输出:
# 请解答数学问题:
#
# 问题: 2+2
# 答案: 4
#
# 问题: 3+3
# 答案: 6
#
# 问题: 4+4
# 答案:
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)
from typing_extensions import override

from langchain_core.example_selectors import BaseExampleSelector
from langchain_core.messages import BaseMessage, get_buffer_string
from langchain_core.prompts.chat import BaseChatPromptTemplate
from langchain_core.prompts.message import BaseMessagePromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.prompts.string import (
    DEFAULT_FORMATTER_MAPPING,
    StringPromptTemplate,
    check_valid_template,
    get_template_variables,
)

if TYPE_CHECKING:
    from pathlib import Path

    from typing_extensions import Self


class _FewShotPromptTemplateMixin(BaseModel):
    """包含 Few-shot 示例的提示模板混入类。

    这是一个 Mixin 类，提供 few-shot 示例的管理功能。
    它被 FewShotPromptTemplate 和 FewShotChatMessagePromptTemplate 继承。

    示例提供方式:
    - examples: 静态示例列表
    - example_selector: 动态示例选择器（二选一）
    """

    examples: list[dict] | None = None
    """要格式化到提示中的示例列表。

    必须提供 examples 或 example_selector 其中之一（但不能同时提供）。

    每个示例是一个字典，包含与 example_prompt 输入变量对应的键值对。

    示例:
        examples = [
            {"input": "你好", "output": "Hello"},
            {"input": "再见", "output": "Goodbye"},
        ]
    """

    example_selector: BaseExampleSelector | None = None
    """用于选择格式化到提示中的示例的 ExampleSelector。

    必须提供 examples 或 example_selector 其中之一（但不能同时提供）。

    ExampleSelector 允许根据用户输入动态选择最相关的示例。
    常见的选择器包括:
    - SemanticSimilarityExampleSelector: 基于语义相似度选择
    - LengthBasedExampleSelector: 基于长度选择
    - MaxMarginalRelevanceExampleSelector: MMR 选择
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,  # 允许任意类型（如 ExampleSelector）
        extra="forbid",  # 禁止额外字段
    )

    @model_validator(mode="before")
    @classmethod
    def check_examples_and_selector(cls, values: dict) -> Any:
        """检查必须且只能提供 examples 或 example_selector 其中之一。

        这是一个验证器，确保:
        1. 不能同时提供 examples 和 example_selector
        2. 必须提供其中之一

        Args:
            values: 要检查的值字典。

        Returns:
            验证通过后的值。

        Raises:
            ValueError: 如果同时提供了两者或都没提供。
        """
        examples = values.get("examples")
        example_selector = values.get("example_selector")

        # 检查是否同时提供了两者
        if examples and example_selector:
            msg = "只能提供 'examples' 和 'example_selector' 其中之一"
            raise ValueError(msg)

        # 检查是否都没提供
        if examples is None and example_selector is None:
            msg = "必须提供 'examples' 或 'example_selector' 其中之一"
            raise ValueError(msg)

        return values

    def _get_examples(self, **kwargs: Any) -> list[dict]:
        """获取用于格式化提示的示例。

        如果使用静态 examples，直接返回。
        如果使用 example_selector，调用其 select_examples 方法。

        Args:
            **kwargs: 传递给示例选择器的关键字参数。
                这通常包含用户输入，用于选择相关示例。

        Returns:
            示例列表。

        Raises:
            ValueError: 如果既没有 examples 也没有 example_selector。
        """
        if self.examples is not None:
            return self.examples
        if self.example_selector is not None:
            return self.example_selector.select_examples(kwargs)
        msg = "必须提供 'examples' 或 'example_selector' 其中之一"
        raise ValueError(msg)

    async def _aget_examples(self, **kwargs: Any) -> list[dict]:
        """异步获取用于格式化提示的示例。

        Args:
            **kwargs: 传递给示例选择器的关键字参数。

        Returns:
            示例列表。

        Raises:
            ValueError: 如果既没有 examples 也没有 example_selector。
        """
        if self.examples is not None:
            return self.examples
        if self.example_selector is not None:
            return await self.example_selector.aselect_examples(kwargs)
        msg = "必须提供 'examples' 或 'example_selector' 其中之一"
        raise ValueError(msg)


class FewShotPromptTemplate(_FewShotPromptTemplateMixin, StringPromptTemplate):
    """包含 Few-shot 示例的字符串提示模板。

    用于创建包含示例的字符串格式提示，结构为:
    [prefix] + [examples] + [suffix]

    属性:
    -----
    examples : list[dict] | None
        静态示例列表
    example_selector : BaseExampleSelector | None
        动态示例选择器
    example_prompt : PromptTemplate
        格式化单个示例的模板
    prefix : str
        示例前的说明文本
    suffix : str
        示例后的模板（通常包含用户输入变量）
    example_separator : str
        示例之间的分隔符
    template_format : str
        模板格式（f-string 或 jinja2）

    使用示例:
    ---------
    >>> from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
    >>>
    >>> examples = [
    ...     {"word": "happy", "antonym": "sad"},
    ...     {"word": "tall", "antonym": "short"},
    ... ]
    >>>
    >>> example_prompt = PromptTemplate.from_template(
    ...     "单词: {word}\\n反义词: {antonym}"
    ... )
    >>>
    >>> few_shot = FewShotPromptTemplate(
    ...     examples=examples,
    ...     example_prompt=example_prompt,
    ...     prefix="给出每个单词的反义词。",
    ...     suffix="单词: {word}\\n反义词:",
    ... )
    """

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """返回 False，因为此类不支持序列化。

        由于 example_selector 可能包含不可序列化的对象，
        所以 FewShotPromptTemplate 不支持完整的序列化。
        """
        return False

    validate_template: bool = False
    """是否验证模板。

    如果设为 True，会检查 prefix + suffix 的模板语法。
    """

    example_prompt: PromptTemplate
    """用于格式化单个示例的 PromptTemplate。

    这个模板定义了每个示例如何被格式化。
    其 input_variables 应该与示例字典中的键匹配。
    """

    suffix: str
    """放在示例之后的提示模板字符串。

    通常包含用户输入的占位符。
    例如: "问题: {question}\\n答案:"
    """

    example_separator: str = "\n\n"
    """用于连接 prefix、examples 和 suffix 的分隔符。

    默认为两个换行符。
    """

    prefix: str = ""
    """放在示例之前的提示模板字符串。

    通常用于提供任务说明或背景信息。
    """

    template_format: Literal["f-string", "jinja2"] = "f-string"
    """提示模板的格式。

    可选值:
    - "f-string": 使用 {variable} 语法（默认）
    - "jinja2": 使用 {{ variable }} 语法
    """

    def __init__(self, **kwargs: Any) -> None:
        """初始化 Few-shot 提示模板。

        如果未提供 input_variables，会从 example_prompt 推断。
        """
        # 如果没有提供 input_variables，从 example_prompt 推断
        if "input_variables" not in kwargs and "example_prompt" in kwargs:
            kwargs["input_variables"] = kwargs["example_prompt"].input_variables
        super().__init__(**kwargs)

    @model_validator(mode="after")
    def template_is_valid(self) -> Self:
        """检查 prefix、suffix 和输入变量的一致性。

        验证或推断模板中的输入变量。
        """
        if self.validate_template:
            # 如果启用验证，检查模板有效性
            check_valid_template(
                self.prefix + self.suffix,
                self.template_format,
                self.input_variables + list(self.partial_variables),
            )
        elif self.template_format:
            # 否则，自动推断输入变量
            self.input_variables = [
                var
                for var in get_template_variables(
                    self.prefix + self.suffix, self.template_format
                )
                if var not in self.partial_variables
            ]
        return self

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    def format(self, **kwargs: Any) -> str:
        """用输入参数格式化提示，生成字符串。

        这是核心格式化方法，按以下步骤工作:
        1. 获取示例（静态或动态选择）
        2. 使用 example_prompt 格式化每个示例
        3. 组装 prefix + examples + suffix
        4. 最终格式化整个模板

        Args:
            **kwargs: 用于格式化的关键字参数。

        Returns:
            格式化后的字符串。

        示例:
            >>> prompt.format(question="什么是 AI？")
        """
        # 合并部分变量和用户变量
        kwargs = self._merge_partial_and_user_variables(**kwargs)

        # 获取示例
        examples = self._get_examples(**kwargs)
        # 只保留 example_prompt 需要的字段
        examples = [
            {k: e[k] for k in self.example_prompt.input_variables} for e in examples
        ]

        # 格式化每个示例
        example_strings = [
            self.example_prompt.format(**example) for example in examples
        ]

        # 组装最终模板: prefix + examples + suffix
        pieces = [self.prefix, *example_strings, self.suffix]
        template = self.example_separator.join([piece for piece in pieces if piece])

        # 使用对应格式化器格式化最终模板
        return DEFAULT_FORMATTER_MAPPING[self.template_format](template, **kwargs)

    async def aformat(self, **kwargs: Any) -> str:
        """异步格式化提示，生成字符串。

        Args:
            **kwargs: 用于格式化的关键字参数。

        Returns:
            格式化后的字符串。
        """
        kwargs = self._merge_partial_and_user_variables(**kwargs)

        # 异步获取示例
        examples = await self._aget_examples(**kwargs)
        examples = [
            {k: e[k] for k in self.example_prompt.input_variables} for e in examples
        ]

        # 异步格式化每个示例
        example_strings = [
            await self.example_prompt.aformat(**example) for example in examples
        ]

        # 组装最终模板
        pieces = [self.prefix, *example_strings, self.suffix]
        template = self.example_separator.join([piece for piece in pieces if piece])

        return DEFAULT_FORMATTER_MAPPING[self.template_format](template, **kwargs)

    @property
    def _prompt_type(self) -> str:
        """返回提示类型标识。"""
        return "few_shot"

    def save(self, file_path: Path | str) -> None:
        """将提示模板保存到文件。

        Args:
            file_path: 保存路径。

        Raises:
            ValueError: 如果使用了 example_selector（不支持序列化）。
        """
        if self.example_selector:
            msg = "目前不支持保存使用 example_selector 的模板"
            raise ValueError(msg)
        return super().save(file_path)


class FewShotChatMessagePromptTemplate(
    BaseChatPromptTemplate, _FewShotPromptTemplateMixin
):
    """支持 Few-shot 示例的聊天提示模板。

    生成的提示结构是一个消息列表，包含:
    - 前缀消息
    - 示例消息
    - 后缀消息

    这种结构可以创建带有中间示例的对话，例如:

        System: 你是一个有用的 AI 助手
        Human: 2+2 等于多少？
        AI: 4
        Human: 3+3 等于多少？
        AI: 6
        Human: 4+4 等于多少？

    此模板可用于生成固定的示例列表，也可以基于输入动态选择示例。

    使用示例 - 固定示例:
    ---------
    ```python
    from langchain_core.prompts import (
        FewShotChatMessagePromptTemplate,
        ChatPromptTemplate,
    )

    # 定义示例
    examples = [
        {"input": "2+2", "output": "4"},
        {"input": "2+3", "output": "5"},
    ]

    # 定义单个示例的格式化模板
    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{input} 等于多少？"),
        ("ai", "{output}"),
    ])

    # 创建 few-shot 聊天模板
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
    )

    # 组合成最终提示
    final_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个有用的 AI 助手"),
        few_shot_prompt,
        ("human", "{input}"),
    ])

    # 格式化
    messages = final_prompt.format_messages(input="4+4 等于多少？")
    ```

    使用示例 - 动态选择示例:
    ---------
    ```python
    from langchain_core.example_selectors import SemanticSimilarityExampleSelector
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import Chroma

    # 定义示例
    examples = [
        {"input": "2+2", "output": "4"},
        {"input": "2+3", "output": "5"},
        {"input": "2+4", "output": "6"},
    ]

    # 创建向量存储和示例选择器
    to_vectorize = [" ".join(example.values()) for example in examples]
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=examples)
    example_selector = SemanticSimilarityExampleSelector(vectorstore=vectorstore)

    # 创建带动态选择的 few-shot 模板
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        input_variables=["input"],  # 传递给 example_selector 的变量
        example_selector=example_selector,
        example_prompt=example_prompt,
    )
    ```
    """

    input_variables: list[str] = Field(default_factory=list)
    """输入变量名称列表。

    这些变量将被传递给 example_selector（如果提供了的话）用于选择相关示例。
    """

    example_prompt: BaseMessagePromptTemplate | BaseChatPromptTemplate
    """用于格式化每个示例的模板。

    可以是:
    - BaseMessagePromptTemplate: 单个消息模板
    - BaseChatPromptTemplate: 多个消息模板（如一问一答对）
    """

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """返回 False，因为此类不支持序列化。"""
        return False

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    def format_messages(self, **kwargs: Any) -> list[BaseMessage]:
        """将参数格式化为消息列表。

        这是核心方法，将示例转换为消息序列。

        Args:
            **kwargs: 用于格式化的关键字参数。
                如果使用 example_selector，这些参数会被传递给它。

        Returns:
            格式化后的消息列表。

        示例:
            >>> messages = few_shot.format_messages(input="4+4")
            >>> for msg in messages:
            ...     print(f"{msg.type}: {msg.content}")
            # human: 2+2 等于多少？
            # ai: 4
            # human: 3+3 等于多少？
            # ai: 6
        """
        # 获取示例
        examples = self._get_examples(**kwargs)
        # 只保留 example_prompt 需要的字段
        examples = [
            {k: e[k] for k in self.example_prompt.input_variables} for e in examples
        ]

        # 将每个示例格式化为消息，然后展平列表
        return [
            message
            for example in examples
            for message in self.example_prompt.format_messages(**example)
        ]

    async def aformat_messages(self, **kwargs: Any) -> list[BaseMessage]:
        """异步将参数格式化为消息列表。

        Args:
            **kwargs: 用于格式化的关键字参数。

        Returns:
            格式化后的消息列表。
        """
        examples = await self._aget_examples(**kwargs)
        examples = [
            {k: e[k] for k in self.example_prompt.input_variables} for e in examples
        ]

        return [
            message
            for example in examples
            for message in await self.example_prompt.aformat_messages(**example)
        ]

    def format(self, **kwargs: Any) -> str:
        """将提示格式化为字符串。

        将聊天消息转换为字符串表示，适用于:
        - 基于字符串的补全模型
        - 调试目的

        Args:
            **kwargs: 用于格式化的关键字参数。

        Returns:
            提示的字符串表示。
        """
        messages = self.format_messages(**kwargs)
        return get_buffer_string(messages)

    async def aformat(self, **kwargs: Any) -> str:
        """异步将提示格式化为字符串。

        Args:
            **kwargs: 用于格式化的关键字参数。

        Returns:
            提示的字符串表示。
        """
        messages = await self.aformat_messages(**kwargs)
        return get_buffer_string(messages)

    @override
    def pretty_repr(self, html: bool = False) -> str:
        """返回提示模板的美化表示。

        Args:
            html: 是否返回 HTML 格式的字符串。

        Returns:
            提示模板的美化表示。
        """
        raise NotImplementedError
