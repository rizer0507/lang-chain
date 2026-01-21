"""Select examples based on length.

中文翻译:
根据长度选择示例。"""

import re
from collections.abc import Callable

from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self

from langchain_core.example_selectors.base import BaseExampleSelector
from langchain_core.prompts.prompt import PromptTemplate


def _get_length_based(text: str) -> int:
    return len(re.split(r"\n| ", text))


class LengthBasedExampleSelector(BaseExampleSelector, BaseModel):
    """Select examples based on length.

    中文翻译:
    根据长度选择示例。"""

    examples: list[dict]
    """A list of the examples that the prompt template expects.

    中文翻译:
    提示模板所需的示例列表。"""

    example_prompt: PromptTemplate
    """Prompt template used to format the examples.

    中文翻译:
    用于格式化示例的提示模板。"""

    get_text_length: Callable[[str], int] = _get_length_based
    """Function to measure prompt length. Defaults to word count.

    中文翻译:
    测量提示长度的功能。默认为字数统计。"""

    max_length: int = 2048
    """Max length for the prompt, beyond which examples are cut.

    中文翻译:
    提示的最大长度，超出该长度的示例将被删除。"""

    example_text_lengths: list[int] = Field(default_factory=list)
    """Length of each example.

    中文翻译:
    每个示例的长度。"""

    def add_example(self, example: dict[str, str]) -> None:
        """Add new example to list.

        Args:
            example: A dictionary with keys as input variables
                and values as their values.
        

        中文翻译:
        将新示例添加到列表中。
        参数：
            示例：以键作为输入变量的字典
                和价值观作为他们的价值观。"""
        self.examples.append(example)
        string_example = self.example_prompt.format(**example)
        self.example_text_lengths.append(self.get_text_length(string_example))

    async def aadd_example(self, example: dict[str, str]) -> None:
        """Async add new example to list.

        Args:
            example: A dictionary with keys as input variables
                and values as their values.
        

        中文翻译:
        异步添加新示例到列表。
        参数：
            示例：以键作为输入变量的字典
                和价值观作为他们的价值观。"""
        self.add_example(example)

    @model_validator(mode="after")
    def post_init(self) -> Self:
        """Validate that the examples are formatted correctly.

        中文翻译:
        验证示例的格式是否正确。"""
        if self.example_text_lengths:
            return self
        string_examples = [self.example_prompt.format(**eg) for eg in self.examples]
        self.example_text_lengths = [self.get_text_length(eg) for eg in string_examples]
        return self

    def select_examples(self, input_variables: dict[str, str]) -> list[dict]:
        """Select which examples to use based on the input lengths.

        Args:
            input_variables: A dictionary with keys as input variables
               and values as their values.

        Returns:
            A list of examples to include in the prompt.
        

        中文翻译:
        根据输入长度选择要使用的示例。
        参数：
            input_variables：以键作为输入变量的字典
               和价值观作为他们的价值观。
        返回：
            要包含在提示中的示例列表。"""
        inputs = " ".join(input_variables.values())
        remaining_length = self.max_length - self.get_text_length(inputs)
        i = 0
        examples = []
        while remaining_length > 0 and i < len(self.examples):
            new_length = remaining_length - self.example_text_lengths[i]
            if new_length < 0:
                break
            examples.append(self.examples[i])
            remaining_length = new_length
            i += 1
        return examples

    async def aselect_examples(self, input_variables: dict[str, str]) -> list[dict]:
        """Async select which examples to use based on the input lengths.

        Args:
            input_variables: A dictionary with keys as input variables
               and values as their values.

        Returns:
            A list of examples to include in the prompt.
        

        中文翻译:
        异步根据输入长度选择要使用的示例。
        参数：
            input_variables：以键作为输入变量的字典
               和价值观作为他们的价值观。
        返回：
            要包含在提示中的示例列表。"""
        return self.select_examples(input_variables)
