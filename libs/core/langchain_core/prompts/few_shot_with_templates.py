"""带模板的 Few-shot 提示模块。

本模块定义了 `FewShotPromptWithTemplates`，它是 `FewShotPromptTemplate` 的增强版本，
允许 prefix 和 suffix 本身也是模板对象，而不仅仅是字符串。

与 FewShotPromptTemplate 的区别:
---------
| 特性 | FewShotPromptTemplate | FewShotPromptWithTemplates |
|------|----------------------|---------------------------|
| prefix | str | StringPromptTemplate |
| suffix | str | StringPromptTemplate |
| 灵活性 | 基础 | 更高 |

使用场景:
---------
当 prefix 或 suffix 本身需要复杂的模板处理时使用此类。
例如，prefix 可能需要从文件加载或有复杂的变量替换逻辑。

使用示例:
---------
>>> from langchain_core.prompts import PromptTemplate
>>> from langchain_core.prompts.few_shot_with_templates import FewShotPromptWithTemplates
>>>
>>> # 定义示例
>>> examples = [
...     {"word": "happy", "antonym": "sad"},
...     {"word": "tall", "antonym": "short"},
... ]
>>>
>>> # 定义各个模板
>>> example_prompt = PromptTemplate.from_template("单词: {word}\\n反义词: {antonym}")
>>> prefix = PromptTemplate.from_template("任务: 为以下单词提供反义词。\\n上下文: {context}")
>>> suffix = PromptTemplate.from_template("单词: {word}\\n反义词:")
>>>
>>> # 创建 few-shot 模板
>>> few_shot = FewShotPromptWithTemplates(
...     examples=examples,
...     example_prompt=example_prompt,
...     prefix=prefix,
...     suffix=suffix,
... )
>>>
>>> # 格式化
>>> result = few_shot.format(context="正式语境", word="beautiful")
"""

from pathlib import Path
from typing import Any

from pydantic import ConfigDict, model_validator
from typing_extensions import Self

from langchain_core.example_selectors import BaseExampleSelector
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.prompts.string import (
    DEFAULT_FORMATTER_MAPPING,
    PromptTemplateFormat,
    StringPromptTemplate,
)


class FewShotPromptWithTemplates(StringPromptTemplate):
    """包含 Few-shot 示例的提示模板（使用模板对象）。

    这是 FewShotPromptTemplate 的高级版本，它允许 prefix 和 suffix
    使用完整的 StringPromptTemplate 对象，而不仅仅是字符串。

    这提供了更大的灵活性，因为:
    - prefix/suffix 可以从文件加载
    - prefix/suffix 可以有自己的输入变量
    - prefix/suffix 可以使用不同的模板格式

    属性:
    -----
    examples : list[dict] | None
        静态示例列表
    example_selector : BaseExampleSelector | None
        动态示例选择器
    example_prompt : PromptTemplate
        格式化单个示例的模板
    prefix : StringPromptTemplate | None
        示例前的模板
    suffix : StringPromptTemplate
        示例后的模板
    example_separator : str
        示例之间的分隔符
    template_format : str
        最终模板的格式
    """

    examples: list[dict] | None = None
    """要格式化到提示中的示例列表。

    必须提供 examples 或 example_selector 其中之一（但不能同时提供）。
    """

    example_selector: BaseExampleSelector | None = None
    """用于选择格式化到提示中的示例的 ExampleSelector。

    必须提供 examples 或 example_selector 其中之一（但不能同时提供）。
    """

    example_prompt: PromptTemplate
    """用于格式化单个示例的 PromptTemplate。

    这个模板定义了每个示例如何被格式化为字符串。
    """

    suffix: StringPromptTemplate
    """放在示例之后的 PromptTemplate。

    注意：这是一个 StringPromptTemplate，不是普通字符串！
    这允许 suffix 有自己的输入变量和模板逻辑。
    """

    example_separator: str = "\n\n"
    """用于连接 prefix、examples 和 suffix 的分隔符。"""

    prefix: StringPromptTemplate | None = None
    """放在示例之前的 PromptTemplate。

    可选。如果不提供，提示将直接从示例开始。
    """

    template_format: PromptTemplateFormat = "f-string"
    """提示模板的格式。

    可选值: 'f-string'、'jinja2'、'mustache'。
    """

    validate_template: bool = False
    """是否验证模板。"""

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """获取 LangChain 对象的命名空间。

        Returns:
            `["langchain", "prompts", "few_shot_with_templates"]`
        """
        return ["langchain", "prompts", "few_shot_with_templates"]

    @model_validator(mode="before")
    @classmethod
    def check_examples_and_selector(cls, values: dict) -> Any:
        """检查必须且只能提供 examples 或 example_selector 其中之一。"""
        examples = values.get("examples")
        example_selector = values.get("example_selector")

        if examples and example_selector:
            msg = "只能提供 'examples' 和 'example_selector' 其中之一"
            raise ValueError(msg)

        if examples is None and example_selector is None:
            msg = "必须提供 'examples' 或 'example_selector' 其中之一"
            raise ValueError(msg)

        return values

    @model_validator(mode="after")
    def template_is_valid(self) -> Self:
        """检查 prefix、suffix 和输入变量的一致性。"""
        if self.validate_template:
            # 如果启用验证，检查变量一致性
            input_variables = self.input_variables
            expected_input_variables = set(self.suffix.input_variables)
            expected_input_variables |= set(self.partial_variables)
            if self.prefix is not None:
                expected_input_variables |= set(self.prefix.input_variables)
            missing_vars = expected_input_variables.difference(input_variables)
            if missing_vars:
                msg = (
                    f"得到 input_variables={input_variables}，但根据 "
                    f"prefix/suffix 期望的是 {expected_input_variables}"
                )
                raise ValueError(msg)
        else:
            # 自动推断输入变量
            self.input_variables = sorted(
                set(self.suffix.input_variables)
                | set(self.prefix.input_variables if self.prefix else [])
                - set(self.partial_variables)
            )
        return self

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    def _get_examples(self, **kwargs: Any) -> list[dict]:
        """获取示例列表。"""
        if self.examples is not None:
            return self.examples
        if self.example_selector is not None:
            return self.example_selector.select_examples(kwargs)
        raise ValueError

    async def _aget_examples(self, **kwargs: Any) -> list[dict]:
        """异步获取示例列表。"""
        if self.examples is not None:
            return self.examples
        if self.example_selector is not None:
            return await self.example_selector.aselect_examples(kwargs)
        raise ValueError

    def format(self, **kwargs: Any) -> str:
        """用输入参数格式化提示。

        工作流程:
        1. 合并部分变量和用户变量
        2. 获取示例并格式化
        3. 格式化 prefix（如果有）
        4. 格式化 suffix
        5. 组装并返回最终模板

        Args:
            **kwargs: 要传递给提示模板的参数。

        Returns:
            格式化后的字符串。

        示例:
            >>> prompt.format(context="正式", word="good")
        """
        kwargs = self._merge_partial_and_user_variables(**kwargs)

        # 获取并格式化示例
        examples = self._get_examples(**kwargs)
        example_strings = [
            self.example_prompt.format(**example) for example in examples
        ]

        # 格式化 prefix
        if self.prefix is None:
            prefix = ""
        else:
            # 只传递 prefix 需要的变量
            prefix_kwargs = {
                k: v for k, v in kwargs.items() if k in self.prefix.input_variables
            }
            # 从 kwargs 中移除已使用的变量
            for k in prefix_kwargs:
                kwargs.pop(k)
            prefix = self.prefix.format(**prefix_kwargs)

        # 格式化 suffix
        suffix_kwargs = {
            k: v for k, v in kwargs.items() if k in self.suffix.input_variables
        }
        for k in suffix_kwargs:
            kwargs.pop(k)
        suffix = self.suffix.format(**suffix_kwargs)

        # 组装最终模板
        pieces = [prefix, *example_strings, suffix]
        template = self.example_separator.join([piece for piece in pieces if piece])

        # 使用剩余的 kwargs 进行最终格式化
        return DEFAULT_FORMATTER_MAPPING[self.template_format](template, **kwargs)

    async def aformat(self, **kwargs: Any) -> str:
        """异步格式化提示。

        Args:
            **kwargs: 要传递给提示模板的参数。

        Returns:
            格式化后的字符串。
        """
        kwargs = self._merge_partial_and_user_variables(**kwargs)

        # 异步获取示例
        examples = await self._aget_examples(**kwargs)
        example_strings = [
            # 可以使用同步方法，因为 PromptTemplate 不会阻塞
            self.example_prompt.format(**example)
            for example in examples
        ]

        # 异步格式化 prefix
        if self.prefix is None:
            prefix = ""
        else:
            prefix_kwargs = {
                k: v for k, v in kwargs.items() if k in self.prefix.input_variables
            }
            for k in prefix_kwargs:
                kwargs.pop(k)
            prefix = await self.prefix.aformat(**prefix_kwargs)

        # 异步格式化 suffix
        suffix_kwargs = {
            k: v for k, v in kwargs.items() if k in self.suffix.input_variables
        }
        for k in suffix_kwargs:
            kwargs.pop(k)
        suffix = await self.suffix.aformat(**suffix_kwargs)

        # 组装最终模板
        pieces = [prefix, *example_strings, suffix]
        template = self.example_separator.join([piece for piece in pieces if piece])

        return DEFAULT_FORMATTER_MAPPING[self.template_format](template, **kwargs)

    @property
    def _prompt_type(self) -> str:
        """返回提示类型标识。"""
        return "few_shot_with_templates"

    def save(self, file_path: Path | str) -> None:
        """保存提示模板到文件。

        Args:
            file_path: 保存路径。

        Raises:
            ValueError: 如果使用了 example_selector（不支持序列化）。
        """
        if self.example_selector:
            msg = "目前不支持保存使用 example_selector 的模板"
            raise ValueError(msg)
        return super().save(file_path)
