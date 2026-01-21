"""Parsers for OpenAI functions output.

中文翻译:
OpenAI 函数输出的解析器。"""

import copy
import json
from types import GenericAlias
from typing import Any

import jsonpatch  # type: ignore[import-untyped]
from pydantic import BaseModel, model_validator
from pydantic.v1 import BaseModel as BaseModelV1
from typing_extensions import override

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import (
    BaseCumulativeTransformOutputParser,
    BaseGenerationOutputParser,
)
from langchain_core.output_parsers.json import parse_partial_json
from langchain_core.outputs import ChatGeneration, Generation


class OutputFunctionsParser(BaseGenerationOutputParser[Any]):
    """Parse an output that is one of sets of values.

    中文翻译:
    解析作为一组值之一的输出。"""

    args_only: bool = True
    """Whether to only return the arguments to the function call.

    中文翻译:
    是否仅返回函数调用的参数。"""

    @override
    def parse_result(self, result: list[Generation], *, partial: bool = False) -> Any:
        """Parse the result of an LLM call to a JSON object.

        Args:
            result: The result of the LLM call.
            partial: Whether to parse partial JSON objects.

        Returns:
            The parsed JSON object.

        Raises:
            OutputParserException: If the output is not valid JSON.
        

        中文翻译:
        解析对 JSON 对象的 LLM 调用的结果。
        参数：
            结果：LLM 调用的结果。
            partial：是否解析部分JSON对象。
        返回：
            解析后的 JSON 对象。
        加薪：
            OutputParserException：如果输出不是有效的 JSON。"""
        generation = result[0]
        if not isinstance(generation, ChatGeneration):
            msg = "This output parser can only be used with a chat generation."
            raise OutputParserException(msg)
        message = generation.message
        try:
            func_call = copy.deepcopy(message.additional_kwargs["function_call"])
        except KeyError as exc:
            msg = f"Could not parse function call: {exc}"
            raise OutputParserException(msg) from exc

        if self.args_only:
            return func_call["arguments"]
        return func_call


class JsonOutputFunctionsParser(BaseCumulativeTransformOutputParser[Any]):
    """Parse an output as the JSON object.

    中文翻译:
    将输出解析为 JSON 对象。"""

    strict: bool = False
    """Whether to allow non-JSON-compliant strings.

    See: https://docs.python.org/3/library/json.html#encoders-and-decoders

    Useful when the parsed output may include unicode characters or new lines.
    

    中文翻译:
    是否允许不符合 JSON 的字符串。
    请参阅：https://docs.python.org/3/library/json.html#encoders-and-decoders
    当解析的输出可能包含 unicode 字符或换行符时很有用。"""

    args_only: bool = True
    """Whether to only return the arguments to the function call.

    中文翻译:
    是否仅返回函数调用的参数。"""

    @property
    def _type(self) -> str:
        return "json_functions"

    @override
    def _diff(self, prev: Any | None, next: Any) -> Any:
        return jsonpatch.make_patch(prev, next).patch

    def parse_result(self, result: list[Generation], *, partial: bool = False) -> Any:
        """Parse the result of an LLM call to a JSON object.

        Args:
            result: The result of the LLM call.
            partial: Whether to parse partial JSON objects.

        Returns:
            The parsed JSON object.

        Raises:
            OutputParserException: If the output is not valid JSON.
        

        中文翻译:
        解析对 JSON 对象的 LLM 调用的结果。
        参数：
            结果：LLM 调用的结果。
            partial：是否解析部分JSON对象。
        返回：
            解析后的 JSON 对象。
        加薪：
            OutputParserException：如果输出不是有效的 JSON。"""
        if len(result) != 1:
            msg = f"Expected exactly one result, but got {len(result)}"
            raise OutputParserException(msg)
        generation = result[0]
        if not isinstance(generation, ChatGeneration):
            msg = "This output parser can only be used with a chat generation."
            raise OutputParserException(msg)
        message = generation.message
        try:
            function_call = message.additional_kwargs["function_call"]
        except KeyError as exc:
            if partial:
                return None
            msg = f"Could not parse function call: {exc}"
            raise OutputParserException(msg) from exc
        try:
            if partial:
                try:
                    if self.args_only:
                        return parse_partial_json(
                            function_call["arguments"], strict=self.strict
                        )
                    return {
                        **function_call,
                        "arguments": parse_partial_json(
                            function_call["arguments"], strict=self.strict
                        ),
                    }
                except json.JSONDecodeError:
                    return None
            elif self.args_only:
                try:
                    return json.loads(function_call["arguments"], strict=self.strict)
                except (json.JSONDecodeError, TypeError) as exc:
                    msg = f"Could not parse function call data: {exc}"
                    raise OutputParserException(msg) from exc
            else:
                try:
                    return {
                        **function_call,
                        "arguments": json.loads(
                            function_call["arguments"], strict=self.strict
                        ),
                    }
                except (json.JSONDecodeError, TypeError) as exc:
                    msg = f"Could not parse function call data: {exc}"
                    raise OutputParserException(msg) from exc
        except KeyError:
            return None

    # This method would be called by the default implementation of `parse_result`
    # 中文: 该方法将由“parse_result”的默认实现调用
    # but we're overriding that method so it's not needed.
    # 中文: 但我们将重写该方法，因此不需要它。
    def parse(self, text: str) -> Any:
        """Parse the output of an LLM call to a JSON object.

        Args:
            text: The output of the LLM call.

        Returns:
            The parsed JSON object.
        

        中文翻译:
        解析对 JSON 对象的 LLM 调用的输出。
        参数：
            文本：LLM 调用的输出。
        返回：
            解析后的 JSON 对象。"""
        raise NotImplementedError


class JsonKeyOutputFunctionsParser(JsonOutputFunctionsParser):
    """Parse an output as the element of the JSON object.

    中文翻译:
    将输出解析为 JSON 对象的元素。"""

    key_name: str
    """The name of the key to return.

    中文翻译:
    要返回的键的名称。"""

    def parse_result(self, result: list[Generation], *, partial: bool = False) -> Any:
        """Parse the result of an LLM call to a JSON object.

        Args:
            result: The result of the LLM call.
            partial: Whether to parse partial JSON objects.

        Returns:
            The parsed JSON object.
        

        中文翻译:
        解析对 JSON 对象的 LLM 调用的结果。
        参数：
            结果：LLM 调用的结果。
            partial：是否解析部分JSON对象。
        返回：
            解析后的 JSON 对象。"""
        res = super().parse_result(result, partial=partial)
        if partial and res is None:
            return None
        return res.get(self.key_name) if partial else res[self.key_name]


class PydanticOutputFunctionsParser(OutputFunctionsParser):
    """Parse an output as a Pydantic object.

    This parser is used to parse the output of a chat model that uses OpenAI function
    format to invoke functions.

    The parser extracts the function call invocation and matches them to the Pydantic
    schema provided.

    An exception will be raised if the function call does not match the provided schema.

    Example:
        ```python
        message = AIMessage(
            content="This is a test message",
            additional_kwargs={
                "function_call": {
                    "name": "cookie",
                    "arguments": json.dumps({"name": "value", "age": 10}),
                }
            },
        )
        chat_generation = ChatGeneration(message=message)


        class Cookie(BaseModel):
            name: str
            age: int


        class Dog(BaseModel):
            species: str


        # Full output
        # 中文: 满输出
        parser = PydanticOutputFunctionsParser(
            pydantic_schema={"cookie": Cookie, "dog": Dog}
        )
        result = parser.parse_result([chat_generation])
        ```

    

    中文翻译:
    将输出解析为 Pydantic 对象。
    该解析器用于解析使用 OpenAI 函数的聊天模型的输出
    调用函数的格式。
    解析器提取函数调用并将它们与 Pydantic 进行匹配
    提供的架构。
    如果函数调用与提供的模式不匹配，则会引发异常。
    示例：
        ````蟒蛇
        消息 = AI消息(
            content="这是一条测试消息",
            额外的_kwargs={
                “函数调用”：{
                    “名称”：“饼干”，
                    “参数”：json.dumps（{“名称”：“值”，“年龄”：10}），
                }
            },
        ）
        chat_ Generation = ChatGeneration（消息=消息）
        Cookie 类（基础模型）：
            名称：str
            年龄：整数
        狗类（基础模型）：
            物种：str
        # 完整输出
        解析器 = PydanticOutputFunctionsParser(
            pydantic_schema={"cookie": Cookie, "dog": 狗}
        ）
        结果 = parser.parse_result([chat_ Generation])
        ````"""

    pydantic_schema: type[BaseModel] | dict[str, type[BaseModel]]
    """The Pydantic schema to parse the output with.

    If multiple schemas are provided, then the function name will be used to
    determine which schema to use.
    

    中文翻译:
    用于解析输出的 Pydantic 模式。
    如果提供了多个模式，则函数名称将用于
    确定使用哪个模式。"""

    @model_validator(mode="before")
    @classmethod
    def validate_schema(cls, values: dict[str, Any]) -> Any:
        """Validate the Pydantic schema.

        Args:
            values: The values to validate.

        Returns:
            The validated values.

        Raises:
            ValueError: If the schema is not a Pydantic schema.
        

        中文翻译:
        验证 Pydantic 架构。
        参数：
            值：要验证的值。
        返回：
            验证值。
        加薪：
            ValueError：如果模式不是 Pydantic 模式。"""
        schema = values["pydantic_schema"]
        if "args_only" not in values:
            values["args_only"] = (
                isinstance(schema, type)
                and not isinstance(schema, GenericAlias)
                and issubclass(schema, BaseModel)
            )
        elif values["args_only"] and isinstance(schema, dict):
            msg = (
                "If multiple pydantic schemas are provided then args_only should be"
                " False."
            )
            raise ValueError(msg)
        return values

    @override
    def parse_result(self, result: list[Generation], *, partial: bool = False) -> Any:
        """Parse the result of an LLM call to a JSON object.

        Args:
            result: The result of the LLM call.
            partial: Whether to parse partial JSON objects.

        Raises:
            ValueError: If the Pydantic schema is not valid.

        Returns:
            The parsed JSON object.
        

        中文翻译:
        解析对 JSON 对象的 LLM 调用的结果。
        参数：
            结果：LLM 调用的结果。
            partial：是否解析部分JSON对象。
        加薪：
            ValueError：如果 Pydantic 架构无效。
        返回：
            解析后的 JSON 对象。"""
        result_ = super().parse_result(result)
        if self.args_only:
            if hasattr(self.pydantic_schema, "model_validate_json"):
                pydantic_args = self.pydantic_schema.model_validate_json(result_)
            else:
                pydantic_args = self.pydantic_schema.parse_raw(result_)  # type: ignore[attr-defined]
        else:
            fn_name = result_["name"]
            args = result_["arguments"]
            if isinstance(self.pydantic_schema, dict):
                pydantic_schema = self.pydantic_schema[fn_name]
            else:
                pydantic_schema = self.pydantic_schema
            if issubclass(pydantic_schema, BaseModel):
                pydantic_args = pydantic_schema.model_validate_json(args)
            elif issubclass(pydantic_schema, BaseModelV1):
                pydantic_args = pydantic_schema.parse_raw(args)
            else:
                msg = f"Unsupported Pydantic schema: {pydantic_schema}"
                raise ValueError(msg)
        return pydantic_args


class PydanticAttrOutputFunctionsParser(PydanticOutputFunctionsParser):
    """Parse an output as an attribute of a Pydantic object.

    中文翻译:
    将输出解析为 Pydantic 对象的属性。"""

    attr_name: str
    """The name of the attribute to return.

    中文翻译:
    要返回的属性的名称。"""

    @override
    def parse_result(self, result: list[Generation], *, partial: bool = False) -> Any:
        """Parse the result of an LLM call to a JSON object.

        Args:
            result: The result of the LLM call.
            partial: Whether to parse partial JSON objects.

        Returns:
            The parsed JSON object.
        

        中文翻译:
        解析对 JSON 对象的 LLM 调用的结果。
        参数：
            结果：LLM 调用的结果。
            partial：是否解析部分JSON对象。
        返回：
            解析后的 JSON 对象。"""
        result = super().parse_result(result)
        return getattr(result, self.attr_name)
