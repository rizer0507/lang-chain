"""Parsers for list output.

中文翻译:
列表输出的解析器。"""

from __future__ import annotations

import csv
import re
from abc import abstractmethod
from collections import deque
from io import StringIO
from typing import TYPE_CHECKING, TypeVar

from typing_extensions import override

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers.transform import BaseTransformOutputParser

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

T = TypeVar("T")


def droplastn(
    iter: Iterator[T],  # noqa: A002
    n: int,
) -> Iterator[T]:
    """Drop the last n elements of an iterator.

    Args:
        iter: The iterator to drop elements from.
        n: The number of elements to drop.

    Yields:
        The elements of the iterator, except the last n elements.
    

    中文翻译:
    删除迭代器的最后 n 个元素。
    参数：
        iter：从中删除元素的迭代器。
        n：要删除的元素数量。
    产量：
        迭代器的元素，最后 n 个元素除外。"""
    buffer: deque[T] = deque()
    for item in iter:
        buffer.append(item)
        if len(buffer) > n:
            yield buffer.popleft()


class ListOutputParser(BaseTransformOutputParser[list[str]]):
    """Parse the output of a model to a list.

    中文翻译:
    将模型的输出解析为列表。"""

    @property
    def _type(self) -> str:
        return "list"

    @abstractmethod
    def parse(self, text: str) -> list[str]:
        """Parse the output of an LLM call.

        Args:
            text: The output of an LLM call.

        Returns:
            A list of strings.
        

        中文翻译:
        解析 LLM 调用的输出。
        参数：
            文本：LLM 调用的输出。
        返回：
            字符串列表。"""

    def parse_iter(self, text: str) -> Iterator[re.Match]:
        """Parse the output of an LLM call.

        Args:
            text: The output of an LLM call.

        Yields:
            A match object for each part of the output.
        

        中文翻译:
        解析 LLM 调用的输出。
        参数：
            文本：LLM 调用的输出。
        产量：
            输出的每个部分的匹配对象。"""
        raise NotImplementedError

    @override
    def _transform(self, input: Iterator[str | BaseMessage]) -> Iterator[list[str]]:
        buffer = ""
        for chunk in input:
            if isinstance(chunk, BaseMessage):
                # Extract text
                # 中文: 提取文本
                chunk_content = chunk.content
                if not isinstance(chunk_content, str):
                    continue
                buffer += chunk_content
            else:
                # Add current chunk to buffer
                # 中文: 将当前块添加到缓冲区
                buffer += chunk
            # Parse buffer into a list of parts
            # 中文: 将缓冲区解析为零件列表
            try:
                done_idx = 0
                # Yield only complete parts
                # 中文: 只生产完整的零件
                for m in droplastn(self.parse_iter(buffer), 1):
                    done_idx = m.end()
                    yield [m.group(1)]
                buffer = buffer[done_idx:]
            except NotImplementedError:
                parts = self.parse(buffer)
                # Yield only complete parts
                # 中文: 只生产完整的零件
                if len(parts) > 1:
                    for part in parts[:-1]:
                        yield [part]
                    buffer = parts[-1]
        # Yield the last part
        # 中文: 产生最后一部分
        for part in self.parse(buffer):
            yield [part]

    @override
    async def _atransform(
        self, input: AsyncIterator[str | BaseMessage]
    ) -> AsyncIterator[list[str]]:
        buffer = ""
        async for chunk in input:
            if isinstance(chunk, BaseMessage):
                # Extract text
                # 中文: 提取文本
                chunk_content = chunk.content
                if not isinstance(chunk_content, str):
                    continue
                buffer += chunk_content
            else:
                # Add current chunk to buffer
                # 中文: 将当前块添加到缓冲区
                buffer += chunk
            # Parse buffer into a list of parts
            # 中文: 将缓冲区解析为零件列表
            try:
                done_idx = 0
                # Yield only complete parts
                # 中文: 只生产完整的零件
                for m in droplastn(self.parse_iter(buffer), 1):
                    done_idx = m.end()
                    yield [m.group(1)]
                buffer = buffer[done_idx:]
            except NotImplementedError:
                parts = self.parse(buffer)
                # Yield only complete parts
                # 中文: 只生产完整的零件
                if len(parts) > 1:
                    for part in parts[:-1]:
                        yield [part]
                    buffer = parts[-1]
        # Yield the last part
        # 中文: 产生最后一部分
        for part in self.parse(buffer):
            yield [part]


class CommaSeparatedListOutputParser(ListOutputParser):
    """Parse the output of a model to a comma-separated list.

    中文翻译:
    将模型的输出解析为逗号分隔的列表。"""

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return `True` as this class is serializable.

        中文翻译:
        返回“True”，因为此类是可序列化的。"""
        return True

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the LangChain object.

        Returns:
            `["langchain", "output_parsers", "list"]`
        

        中文翻译:
        获取LangChain对象的命名空间。
        返回：
            `[“langchain”，“output_parsers”，“列表”]`"""
        return ["langchain", "output_parsers", "list"]

    @override
    def get_format_instructions(self) -> str:
        """Return the format instructions for the comma-separated list output.

        中文翻译:
        返回逗号分隔列表输出的格式指令。"""
        return (
            "Your response should be a list of comma separated values, "
            "eg: `foo, bar, baz` or `foo,bar,baz`"
        )

    @override
    def parse(self, text: str) -> list[str]:
        """Parse the output of an LLM call.

        Args:
            text: The output of an LLM call.

        Returns:
            A list of strings.
        

        中文翻译:
        解析 LLM 调用的输出。
        参数：
            文本：LLM 调用的输出。
        返回：
            字符串列表。"""
        try:
            reader = csv.reader(
                StringIO(text), quotechar='"', delimiter=",", skipinitialspace=True
            )
            return [item for sublist in reader for item in sublist]
        except csv.Error:
            # Keep old logic for backup
            # 中文: 保留旧逻辑进行备份
            return [part.strip() for part in text.split(",")]

    @property
    def _type(self) -> str:
        return "comma-separated-list"


class NumberedListOutputParser(ListOutputParser):
    """Parse a numbered list.

    中文翻译:
    解析编号列表。"""

    pattern: str = r"\d+\.\s([^\n]+)"
    """The pattern to match a numbered list item.

    中文翻译:
    匹配编号列表项的模式。"""

    @override
    def get_format_instructions(self) -> str:
        return (
            "Your response should be a numbered list with each item on a new line. "
            "For example: \n\n1. foo\n\n2. bar\n\n3. baz"
        )

    def parse(self, text: str) -> list[str]:
        """Parse the output of an LLM call.

        Args:
            text: The output of an LLM call.

        Returns:
            A list of strings.
        

        中文翻译:
        解析 LLM 调用的输出。
        参数：
            文本：LLM 调用的输出。
        返回：
            字符串列表。"""
        return re.findall(self.pattern, text)

    @override
    def parse_iter(self, text: str) -> Iterator[re.Match]:
        return re.finditer(self.pattern, text)

    @property
    def _type(self) -> str:
        return "numbered-list"


class MarkdownListOutputParser(ListOutputParser):
    """Parse a Markdown list.

    中文翻译:
    Error 500 (Server Error)!!1500.That’s an error.There was an error. Please try again later.That’s all we know."""

    pattern: str = r"^\s*[-*]\s([^\n]+)$"
    """The pattern to match a Markdown list item.

    中文翻译:
    匹配 Markdown 列表项的模式。"""

    @override
    def get_format_instructions(self) -> str:
        """Return the format instructions for the Markdown list output.

        中文翻译:
        返回 Markdown 列表输出的格式说明。"""
        return "Your response should be a markdown list, eg: `- foo\n- bar\n- baz`"

    def parse(self, text: str) -> list[str]:
        """Parse the output of an LLM call.

        Args:
            text: The output of an LLM call.

        Returns:
            A list of strings.
        

        中文翻译:
        解析 LLM 调用的输出。
        参数：
            文本：LLM 调用的输出。
        返回：
            字符串列表。"""
        return re.findall(self.pattern, text, re.MULTILINE)

    @override
    def parse_iter(self, text: str) -> Iterator[re.Match]:
        return re.finditer(self.pattern, text, re.MULTILINE)

    @property
    def _type(self) -> str:
        return "markdown-list"
