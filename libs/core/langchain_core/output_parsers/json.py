"""Parser for JSON output.

中文翻译:
JSON 输出的解析器。"""

from __future__ import annotations

import json
from json import JSONDecodeError
from typing import Annotated, Any, TypeVar

import jsonpatch  # type: ignore[import-untyped]
import pydantic
from pydantic import SkipValidation
from pydantic.v1 import BaseModel
from typing_extensions import override

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers.format_instructions import JSON_FORMAT_INSTRUCTIONS
from langchain_core.output_parsers.transform import BaseCumulativeTransformOutputParser
from langchain_core.outputs import Generation
from langchain_core.utils.json import (
    parse_and_check_json_markdown,
    parse_json_markdown,
    parse_partial_json,
)

# Union type needs to be last assignment to PydanticBaseModel to make mypy happy.
# 中文: 联合类型需要最后分配给 PydanticBaseModel 才能使 mypy 满意。
PydanticBaseModel = BaseModel | pydantic.BaseModel

TBaseModel = TypeVar("TBaseModel", bound=PydanticBaseModel)


class JsonOutputParser(BaseCumulativeTransformOutputParser[Any]):
    """Parse the output of an LLM call to a JSON object.

    Probably the most reliable output parser for getting structured data that does *not*
    use function calling.

    When used in streaming mode, it will yield partial JSON objects containing
    all the keys that have been returned so far.

    In streaming, if `diff` is set to `True`, yields JSONPatch operations describing the
    difference between the previous and the current object.
    

    中文翻译:
    解析对 JSON 对象的 LLM 调用的输出。
    可能是获取结构化数据最可靠的输出解析器，但*不*
    使用函数调用。
    当在流模式下使用时，它将生成包含以下内容的部分 JSON 对象：
    到目前为止已归还的所有钥匙。
    在流式传输中，如果“diff”设置为“True”，则会生成描述
    前一个对象和当前对象之间的差异。"""

    pydantic_object: Annotated[type[TBaseModel] | None, SkipValidation()] = None  # type: ignore[valid-type]
    """The Pydantic object to use for validation.
    If `None`, no validation is performed.

    中文翻译:
    用于验证的 Pydantic 对象。
    如果为“None”，则不执行验证。"""

    @override
    def _diff(self, prev: Any | None, next: Any) -> Any:
        return jsonpatch.make_patch(prev, next).patch

    @staticmethod
    def _get_schema(pydantic_object: type[TBaseModel]) -> dict[str, Any]:
        if issubclass(pydantic_object, pydantic.BaseModel):
            return pydantic_object.model_json_schema()
        return pydantic_object.schema()

    @override
    def parse_result(self, result: list[Generation], *, partial: bool = False) -> Any:
        """Parse the result of an LLM call to a JSON object.

        Args:
            result: The result of the LLM call.
            partial: Whether to parse partial JSON objects.
                If `True`, the output will be a JSON object containing
                all the keys that have been returned so far.
                If `False`, the output will be the full JSON object.

        Returns:
            The parsed JSON object.

        Raises:
            OutputParserException: If the output is not valid JSON.
        

        中文翻译:
        解析对 JSON 对象的 LLM 调用的结果。
        参数：
            结果：LLM 调用的结果。
            partial：是否解析部分JSON对象。
                如果为 True，输出将是一个 JSON 对象，其中包含
                到目前为止已归还的所有钥匙。
                如果为“False”，则输出将是完整的 JSON 对象。
        返回：
            解析后的 JSON 对象。
        加薪：
            OutputParserException：如果输出不是有效的 JSON。"""
        text = result[0].text
        text = text.strip()
        if partial:
            try:
                return parse_json_markdown(text)
            except JSONDecodeError:
                return None
        else:
            try:
                return parse_json_markdown(text)
            except JSONDecodeError as e:
                msg = f"Invalid json output: {text}"
                raise OutputParserException(msg, llm_output=text) from e

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
        return self.parse_result([Generation(text=text)])

    def get_format_instructions(self) -> str:
        """Return the format instructions for the JSON output.

        Returns:
            The format instructions for the JSON output.
        

        中文翻译:
        返回 JSON 输出的格式说明。
        返回：
            JSON 输出的格式说明。"""
        if self.pydantic_object is None:
            return "Return a JSON object."
        # Copy schema to avoid altering original Pydantic schema.
        # 中文: 复制架构以避免更改原始 Pydantic 架构。
        schema = dict(self._get_schema(self.pydantic_object).items())

        # Remove extraneous fields.
        # 中文: 删除无关字段。
        reduced_schema = schema
        if "title" in reduced_schema:
            del reduced_schema["title"]
        if "type" in reduced_schema:
            del reduced_schema["type"]
        # Ensure json in context is well-formed with double quotes.
        # 中文: 确保上下文中的 json 格式正确并带有双引号。
        schema_str = json.dumps(reduced_schema, ensure_ascii=False)
        return JSON_FORMAT_INSTRUCTIONS.format(schema=schema_str)

    @property
    def _type(self) -> str:
        return "simple_json_output_parser"


# For backwards compatibility
# 中文: 为了向后兼容
SimpleJsonOutputParser = JsonOutputParser


__all__ = [
    "JsonOutputParser",
    "SimpleJsonOutputParser",  # For backwards compatibility
    "parse_and_check_json_markdown",  # For backwards compatibility
    "parse_partial_json",  # For backwards compatibility
]
