"""Base parser for language model outputs.

中文翻译:
语言模型输出的基本解析器。"""

from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    TypeVar,
    cast,
)

from typing_extensions import override

from langchain_core.language_models import LanguageModelOutput
from langchain_core.messages import AnyMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, Generation
from langchain_core.runnables import Runnable, RunnableConfig, RunnableSerializable
from langchain_core.runnables.config import run_in_executor

if TYPE_CHECKING:
    from langchain_core.prompt_values import PromptValue

T = TypeVar("T")
OutputParserLike = Runnable[LanguageModelOutput, T]


class BaseLLMOutputParser(ABC, Generic[T]):
    """Abstract base class for parsing the outputs of a model.

    中文翻译:
    用于解析模型输出的抽象基类。"""

    @abstractmethod
    def parse_result(self, result: list[Generation], *, partial: bool = False) -> T:
        """Parse a list of candidate model `Generation` objects into a specific format.

        Args:
            result: A list of `Generation` to be parsed. The `Generation` objects are
                assumed to be different candidate outputs for a single model input.
            partial: Whether to parse the output as a partial result. This is useful
                for parsers that can parse partial results.

        Returns:
            Structured output.
        

        中文翻译:
        将候选模型“Generation”对象列表解析为特定格式。
        参数：
            结果：要解析的“Generation”列表。 `Generation` 对象是
                假设是单个模型输入的不同候选输出。
            partial：是否将输出解析为部分结果。这很有用
                用于可以解析部分结果的解析器。
        返回：
            结构化输出。"""

    async def aparse_result(
        self, result: list[Generation], *, partial: bool = False
    ) -> T:
        """Async parse a list of candidate model `Generation` objects into a specific format.

        Args:
            result: A list of `Generation` to be parsed. The Generations are assumed
                to be different candidate outputs for a single model input.
            partial: Whether to parse the output as a partial result. This is useful
                for parsers that can parse partial results.

        Returns:
            Structured output.
        

        中文翻译:
        异步将候选模型“Generation”对象列表解析为特定格式。
        参数：
            结果：要解析的“Generation”列表。假定世代
                成为单个模型输入的不同候选输出。
            partial：是否将输出解析为部分结果。这很有用
                用于可以解析部分结果的解析器。
        返回：
            结构化输出。"""  # noqa: E501
        return await run_in_executor(None, self.parse_result, result, partial=partial)


class BaseGenerationOutputParser(
    BaseLLMOutputParser, RunnableSerializable[LanguageModelOutput, T]
):
    """Base class to parse the output of an LLM call.

    中文翻译:
    用于解析 LLM 调用输出的基类。"""

    @property
    @override
    def InputType(self) -> Any:
        """Return the input type for the parser.

        中文翻译:
        返回解析器的输入类型。"""
        return str | AnyMessage

    @property
    @override
    def OutputType(self) -> type[T]:
        """Return the output type for the parser.

        中文翻译:
        返回解析器的输出类型。"""
        # even though mypy complains this isn't valid,
        # 中文: 尽管 mypy 抱怨这是无效的，
        # it is good enough for pydantic to build the schema from
        # 中文: pydantic 足以构建架构
        return cast("type[T]", T)  # type: ignore[misc]

    @override
    def invoke(
        self,
        input: str | BaseMessage,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> T:
        if isinstance(input, BaseMessage):
            return self._call_with_config(
                lambda inner_input: self.parse_result(
                    [ChatGeneration(message=inner_input)]
                ),
                input,
                config,
                run_type="parser",
            )
        return self._call_with_config(
            lambda inner_input: self.parse_result([Generation(text=inner_input)]),
            input,
            config,
            run_type="parser",
        )

    @override
    async def ainvoke(
        self,
        input: str | BaseMessage,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> T:
        if isinstance(input, BaseMessage):
            return await self._acall_with_config(
                lambda inner_input: self.aparse_result(
                    [ChatGeneration(message=inner_input)]
                ),
                input,
                config,
                run_type="parser",
            )
        return await self._acall_with_config(
            lambda inner_input: self.aparse_result([Generation(text=inner_input)]),
            input,
            config,
            run_type="parser",
        )


class BaseOutputParser(
    BaseLLMOutputParser, RunnableSerializable[LanguageModelOutput, T]
):
    """Base class to parse the output of an LLM call.

    Output parsers help structure language model responses.

    Example:
        ```python
        # Implement a simple boolean output parser
        # 中文: 实现一个简单的布尔输出解析器


        class BooleanOutputParser(BaseOutputParser[bool]):
            true_val: str = "YES"
            false_val: str = "NO"

            def parse(self, text: str) -> bool:
                cleaned_text = text.strip().upper()
                if cleaned_text not in (
                    self.true_val.upper(),
                    self.false_val.upper(),
                ):
                    raise OutputParserException(
                        f"BooleanOutputParser expected output value to either be "
                        f"{self.true_val} or {self.false_val} (case-insensitive). "
                        f"Received {cleaned_text}."
                    )
                return cleaned_text == self.true_val.upper()

            @property
            def _type(self) -> str:
                return "boolean_output_parser"
        ```
    

    中文翻译:
    用于解析 LLM 调用输出的基类。
    输出解析器帮助构建语言模型响应。
    示例：
        ````蟒蛇
        # 实现一个简单的布尔输出解析器
        类 BooleanOutputParser(BaseOutputParser[bool]):
            true_val: str = "是"
            false_val：str =“否”
            def parse(self, text: str) -> bool:
                clean_text = text.strip().upper()
                如果 clean_text 不在 (
                    self.true_val.upper(),
                    self.false_val.upper(),
                ）：
                    引发 OutputParserException(
                        f“BooleanOutputParser 预期输出值为”
                        f"{self.true_val} 或 {self.false_val}（不区分大小写）。"
                        f“收到{cleaned_text}。”
                    ）
                返回 clean_text == self.true_val.upper()
            @属性
            def _type(self) -> str:
                返回“布尔输出解析器”
        ````"""

    @property
    @override
    def InputType(self) -> Any:
        """Return the input type for the parser.

        中文翻译:
        返回解析器的输入类型。"""
        return str | AnyMessage

    @property
    @override
    def OutputType(self) -> type[T]:
        """Return the output type for the parser.

        This property is inferred from the first type argument of the class.

        Raises:
            TypeError: If the class doesn't have an inferable `OutputType`.
        

        中文翻译:
        返回解析器的输出类型。
        该属性是从类的第一个类型参数推断出来的。
        加薪：
            TypeError：如果该类没有可推断的“OutputType”。"""
        for base in self.__class__.mro():
            if hasattr(base, "__pydantic_generic_metadata__"):
                metadata = base.__pydantic_generic_metadata__
                if "args" in metadata and len(metadata["args"]) > 0:
                    return cast("type[T]", metadata["args"][0])

        msg = (
            f"Runnable {self.__class__.__name__} doesn't have an inferable OutputType. "
            "Override the OutputType property to specify the output type."
        )
        raise TypeError(msg)

    @override
    def invoke(
        self,
        input: str | BaseMessage,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> T:
        if isinstance(input, BaseMessage):
            return self._call_with_config(
                lambda inner_input: self.parse_result(
                    [ChatGeneration(message=inner_input)]
                ),
                input,
                config,
                run_type="parser",
            )
        return self._call_with_config(
            lambda inner_input: self.parse_result([Generation(text=inner_input)]),
            input,
            config,
            run_type="parser",
        )

    @override
    async def ainvoke(
        self,
        input: str | BaseMessage,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> T:
        if isinstance(input, BaseMessage):
            return await self._acall_with_config(
                lambda inner_input: self.aparse_result(
                    [ChatGeneration(message=inner_input)]
                ),
                input,
                config,
                run_type="parser",
            )
        return await self._acall_with_config(
            lambda inner_input: self.aparse_result([Generation(text=inner_input)]),
            input,
            config,
            run_type="parser",
        )

    @override
    def parse_result(self, result: list[Generation], *, partial: bool = False) -> T:
        """Parse a list of candidate model `Generation` objects into a specific format.

        The return value is parsed from only the first `Generation` in the result, which
            is assumed to be the highest-likelihood `Generation`.

        Args:
            result: A list of `Generation` to be parsed. The `Generation` objects are
                assumed to be different candidate outputs for a single model input.
            partial: Whether to parse the output as a partial result. This is useful
                for parsers that can parse partial results.

        Returns:
            Structured output.
        

        中文翻译:
        将候选模型“Generation”对象列表解析为特定格式。
        返回值仅从结果中的第一个“Generation”中解析，其中
            被假定为最高可能性的“一代”。
        参数：
            结果：要解析的“Generation”列表。 `Generation` 对象是
                假设是单个模型输入的不同候选输出。
            partial：是否将输出解析为部分结果。这很有用
                用于可以解析部分结果的解析器。
        返回：
            结构化输出。"""
        return self.parse(result[0].text)

    @abstractmethod
    def parse(self, text: str) -> T:
        """Parse a single string model output into some structure.

        Args:
            text: String output of a language model.

        Returns:
            Structured output.
        

        中文翻译:
        将单个字符串模型输出解析为某种结构。
        参数：
            text：语言模型的字符串输出。
        返回：
            结构化输出。"""

    async def aparse_result(
        self, result: list[Generation], *, partial: bool = False
    ) -> T:
        """Async parse a list of candidate model `Generation` objects into a specific format.

        The return value is parsed from only the first `Generation` in the result, which
            is assumed to be the highest-likelihood `Generation`.

        Args:
            result: A list of `Generation` to be parsed. The `Generation` objects are
                assumed to be different candidate outputs for a single model input.
            partial: Whether to parse the output as a partial result. This is useful
                for parsers that can parse partial results.

        Returns:
            Structured output.
        

        中文翻译:
        异步将候选模型“Generation”对象列表解析为特定格式。
        返回值仅从结果中的第一个“Generation”中解析，其中
            被假定为最高可能性的“一代”。
        参数：
            结果：要解析的“Generation”列表。 `Generation` 对象是
                假设是单个模型输入的不同候选输出。
            partial：是否将输出解析为部分结果。这很有用
                用于可以解析部分结果的解析器。
        返回：
            结构化输出。"""  # noqa: E501
        return await run_in_executor(None, self.parse_result, result, partial=partial)

    async def aparse(self, text: str) -> T:
        """Async parse a single string model output into some structure.

        Args:
            text: String output of a language model.

        Returns:
            Structured output.
        

        中文翻译:
        异步将单个字符串模型输出解析为某种结构。
        参数：
            text：语言模型的字符串输出。
        返回：
            结构化输出。"""
        return await run_in_executor(None, self.parse, text)

    # TODO: rename 'completion' -> 'text'.
    def parse_with_prompt(
        self,
        completion: str,
        prompt: PromptValue,  # noqa: ARG002
    ) -> Any:
        """Parse the output of an LLM call with the input prompt for context.

        The prompt is largely provided in the event the `OutputParser` wants
        to retry or fix the output in some way, and needs information from
        the prompt to do so.

        Args:
            completion: String output of a language model.
            prompt: Input `PromptValue`.

        Returns:
            Structured output.
        

        中文翻译:
        使用上下文的输入提示解析 LLM 调用的输出。
        提示主要是在“OutputParser”想要的情况下提供的
        以某种方式重试或修复输出，并且需要来自
        提示您这样做。
        参数：
            完成：语言模型的字符串输出。
            提示：输入“PromptValue”。
        返回：
            结构化输出。"""
        return self.parse(completion)

    def get_format_instructions(self) -> str:
        """Instructions on how the LLM output should be formatted.

        中文翻译:
        关于如何格式化 LLM 输出的说明。"""
        raise NotImplementedError

    @property
    def _type(self) -> str:
        """Return the output parser type for serialization.

        中文翻译:
        返回用于序列化的输出解析器类型。"""
        msg = (
            f"_type property is not implemented in class {self.__class__.__name__}."
            " This is required for serialization."
        )
        raise NotImplementedError(msg)

    def dict(self, **kwargs: Any) -> dict:
        """Return dictionary representation of output parser.

        中文翻译:
        返回输出解析器的字典表示。"""
        output_parser_dict = super().model_dump(**kwargs)
        with contextlib.suppress(NotImplementedError):
            output_parser_dict["_type"] = self._type
        return output_parser_dict
