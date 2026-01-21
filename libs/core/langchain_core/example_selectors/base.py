"""Interface for selecting examples to include in prompts.

中文翻译:
用于选择要包含在提示中的示例的界面。"""

from abc import ABC, abstractmethod
from typing import Any

from langchain_core.runnables import run_in_executor


class BaseExampleSelector(ABC):
    """Interface for selecting examples to include in prompts.

    中文翻译:
    用于选择要包含在提示中的示例的界面。"""

    @abstractmethod
    def add_example(self, example: dict[str, str]) -> Any:
        """Add new example to store.

        Args:
            example: A dictionary with keys as input variables
                and values as their values.

        Returns:
            Any return value.
        

        中文翻译:
        添加新示例以存储。
        参数：
            示例：以键作为输入变量的字典
                和价值观作为他们的价值观。
        返回：
            任何返回值。"""

    async def aadd_example(self, example: dict[str, str]) -> Any:
        """Async add new example to store.

        Args:
            example: A dictionary with keys as input variables
                and values as their values.

        Returns:
            Any return value.
        

        中文翻译:
        异步添加新示例以存储。
        参数：
            示例：以键作为输入变量的字典
                和价值观作为他们的价值观。
        返回：
            任何返回值。"""
        return await run_in_executor(None, self.add_example, example)

    @abstractmethod
    def select_examples(self, input_variables: dict[str, str]) -> list[dict]:
        """Select which examples to use based on the inputs.

        Args:
            input_variables: A dictionary with keys as input variables
                and values as their values.

        Returns:
            A list of examples.
        

        中文翻译:
        根据输入选择要使用的示例。
        参数：
            input_variables：以键作为输入变量的字典
                和价值观作为他们的价值观。
        返回：
            示例列表。"""

    async def aselect_examples(self, input_variables: dict[str, str]) -> list[dict]:
        """Async select which examples to use based on the inputs.

        Args:
            input_variables: A dictionary with keys as input variables
                and values as their values.

        Returns:
            A list of examples.
        

        中文翻译:
        异步根据输入选择要使用的示例。
        参数：
            input_variables：以键作为输入变量的字典
                和价值观作为他们的价值观。
        返回：
            示例列表。"""
        return await run_in_executor(None, self.select_examples, input_variables)
