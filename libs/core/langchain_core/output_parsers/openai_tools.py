"""Parse tools for OpenAI tools output.

中文翻译:
OpenAI 工具输出的解析工具。"""

import copy
import json
import logging
from json import JSONDecodeError
from typing import Annotated, Any

from pydantic import SkipValidation, ValidationError

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, InvalidToolCall
from langchain_core.messages.tool import invalid_tool_call
from langchain_core.messages.tool import tool_call as create_tool_call
from langchain_core.output_parsers.transform import BaseCumulativeTransformOutputParser
from langchain_core.outputs import ChatGeneration, Generation
from langchain_core.utils.json import parse_partial_json
from langchain_core.utils.pydantic import (
    TypeBaseModel,
    is_pydantic_v1_subclass,
    is_pydantic_v2_subclass,
)

logger = logging.getLogger(__name__)


def parse_tool_call(
    raw_tool_call: dict[str, Any],
    *,
    partial: bool = False,
    strict: bool = False,
    return_id: bool = True,
) -> dict[str, Any] | None:
    """Parse a single tool call.

    Args:
        raw_tool_call: The raw tool call to parse.
        partial: Whether to parse partial JSON.
        strict: Whether to allow non-JSON-compliant strings.
        return_id: Whether to return the tool call id.

    Returns:
        The parsed tool call.

    Raises:
        OutputParserException: If the tool call is not valid JSON.
    

    中文翻译:
    解析单个工具调用。
    参数：
        raw_tool_call：要解析的原始工具调用。
        partial：是否解析部分JSON。
        strict：是否允许不符合 JSON 的字符串。
        return_id：是否返回工具调用id。
    返回：
        解析的工具调用。
    加薪：
        OutputParserException：如果工具调用不是有效的 JSON。"""
    if "function" not in raw_tool_call:
        return None

    arguments = raw_tool_call["function"]["arguments"]

    if partial:
        try:
            function_args = parse_partial_json(arguments, strict=strict)
        except (JSONDecodeError, TypeError):  # None args raise TypeError
            return None
    # Handle None or empty string arguments for parameter-less tools
    # 中文: 处理无参数工具的 None 或空字符串参数
    elif not arguments:
        function_args = {}
    else:
        try:
            function_args = json.loads(arguments, strict=strict)
        except JSONDecodeError as e:
            msg = (
                f"Function {raw_tool_call['function']['name']} arguments:\n\n"
                f"{arguments}\n\nare not valid JSON. "
                f"Received JSONDecodeError {e}"
            )
            raise OutputParserException(msg) from e
    parsed = {
        "name": raw_tool_call["function"]["name"] or "",
        "args": function_args or {},
    }
    if return_id:
        parsed["id"] = raw_tool_call.get("id")
        parsed = create_tool_call(**parsed)  # type: ignore[assignment,arg-type]
    return parsed


def make_invalid_tool_call(
    raw_tool_call: dict[str, Any],
    error_msg: str | None,
) -> InvalidToolCall:
    """Create an InvalidToolCall from a raw tool call.

    Args:
        raw_tool_call: The raw tool call.
        error_msg: The error message.

    Returns:
        An InvalidToolCall instance with the error message.
    

    中文翻译:
    从原始工具调用创建 InvalidToolCall。
    参数：
        raw_tool_call：原始工具调用。
        error_msg：错误消息。
    返回：
        带有错误消息的 InvalidToolCall 实例。"""
    return invalid_tool_call(
        name=raw_tool_call["function"]["name"],
        args=raw_tool_call["function"]["arguments"],
        id=raw_tool_call.get("id"),
        error=error_msg,
    )


def parse_tool_calls(
    raw_tool_calls: list[dict],
    *,
    partial: bool = False,
    strict: bool = False,
    return_id: bool = True,
) -> list[dict[str, Any]]:
    """Parse a list of tool calls.

    Args:
        raw_tool_calls: The raw tool calls to parse.
        partial: Whether to parse partial JSON.
        strict: Whether to allow non-JSON-compliant strings.
        return_id: Whether to return the tool call id.

    Returns:
        The parsed tool calls.

    Raises:
        OutputParserException: If any of the tool calls are not valid JSON.
    

    中文翻译:
    解析工具调用列表。
    参数：
        raw_tool_calls：原始工具调用解析。
        partial：是否解析部分JSON。
        strict：是否允许不符合 JSON 的字符串。
        return_id：是否返回工具调用id。
    返回：
        解析的工具调用。
    加薪：
        OutputParserException：如果任何工具调用不是有效的 JSON。"""
    final_tools: list[dict[str, Any]] = []
    exceptions = []
    for tool_call in raw_tool_calls:
        try:
            parsed = parse_tool_call(
                tool_call, partial=partial, strict=strict, return_id=return_id
            )
            if parsed:
                final_tools.append(parsed)
        except OutputParserException as e:
            exceptions.append(str(e))
            continue
    if exceptions:
        raise OutputParserException("\n\n".join(exceptions))
    return final_tools


class JsonOutputToolsParser(BaseCumulativeTransformOutputParser[Any]):
    """Parse tools from OpenAI response.

    中文翻译:
    从 OpenAI 响应中解析工具。"""

    strict: bool = False
    """Whether to allow non-JSON-compliant strings.

    See: https://docs.python.org/3/library/json.html#encoders-and-decoders

    Useful when the parsed output may include unicode characters or new lines.
    

    中文翻译:
    是否允许不符合 JSON 的字符串。
    请参阅：https://docs.python.org/3/library/json.html#encoders-and-decoders
    当解析的输出可能包含 unicode 字符或换行符时很有用。"""
    return_id: bool = False
    """Whether to return the tool call id.

    中文翻译:
    是否返回工具调用id。"""
    first_tool_only: bool = False
    """Whether to return only the first tool call.

    If `False`, the result will be a list of tool calls, or an empty list
    if no tool calls are found.

    If true, and multiple tool calls are found, only the first one will be returned,
    and the other tool calls will be ignored.
    If no tool calls are found, None will be returned.
    

    中文翻译:
    是否仅返回第一个工具调用。
    如果为“False”，结果将是工具调用列表，或空列表
    如果没有找到工具调用。
    如果为 true，并且发现多个工具调用，则仅返回第一个，
    并且其他工具调用将被忽略。
    如果没有找到工具调用，则不会返回 None。"""

    def parse_result(self, result: list[Generation], *, partial: bool = False) -> Any:
        """Parse the result of an LLM call to a list of tool calls.

        Args:
            result: The result of the LLM call.
            partial: Whether to parse partial JSON.
                If `True`, the output will be a JSON object containing
                all the keys that have been returned so far.
                If `False`, the output will be the full JSON object.

        Returns:
            The parsed tool calls.

        Raises:
            OutputParserException: If the output is not valid JSON.
        

        中文翻译:
        将 LLM 调用的结果解析为工具调用列表。
        参数：
            结果：LLM 调用的结果。
            partial：是否解析部分JSON。
                如果为 True，输出将是一个 JSON 对象，其中包含
                到目前为止已归还的所有钥匙。
                如果为“False”，则输出将是完整的 JSON 对象。
        返回：
            解析的工具调用。
        加薪：
            OutputParserException：如果输出不是有效的 JSON。"""
        generation = result[0]
        if not isinstance(generation, ChatGeneration):
            msg = "This output parser can only be used with a chat generation."
            raise OutputParserException(msg)
        message = generation.message
        if isinstance(message, AIMessage) and message.tool_calls:
            tool_calls = [dict(tc) for tc in message.tool_calls]
            for tool_call in tool_calls:
                if not self.return_id:
                    _ = tool_call.pop("id")
        else:
            try:
                raw_tool_calls = copy.deepcopy(message.additional_kwargs["tool_calls"])
            except KeyError:
                return []
            tool_calls = parse_tool_calls(
                raw_tool_calls,
                partial=partial,
                strict=self.strict,
                return_id=self.return_id,
            )
        # for backwards compatibility
        # 中文: 为了向后兼容
        for tc in tool_calls:
            tc["type"] = tc.pop("name")

        if self.first_tool_only:
            return tool_calls[0] if tool_calls else None
        return tool_calls

    def parse(self, text: str) -> Any:
        """Parse the output of an LLM call to a list of tool calls.

        Args:
            text: The output of the LLM call.

        Returns:
            The parsed tool calls.
        

        中文翻译:
        将 LLM 调用的输出解析为工具调用列表。
        参数：
            文本：LLM 调用的输出。
        返回：
            解析的工具调用。"""
        raise NotImplementedError


class JsonOutputKeyToolsParser(JsonOutputToolsParser):
    """Parse tools from OpenAI response.

    中文翻译:
    从 OpenAI 响应中解析工具。"""

    key_name: str
    """The type of tools to return.

    中文翻译:
    要返回的工具类型。"""

    def parse_result(self, result: list[Generation], *, partial: bool = False) -> Any:
        """Parse the result of an LLM call to a list of tool calls.

        Args:
            result: The result of the LLM call.
            partial: Whether to parse partial JSON.
                If `True`, the output will be a JSON object containing
                    all the keys that have been returned so far.
                If `False`, the output will be the full JSON object.

        Raises:
            OutputParserException: If the generation is not a chat generation.

        Returns:
            The parsed tool calls.
        

        中文翻译:
        将 LLM 调用的结果解析为工具调用列表。
        参数：
            结果：LLM 调用的结果。
            partial：是否解析部分JSON。
                如果为 True，输出将是一个 JSON 对象，其中包含
                    到目前为止已归还的所有钥匙。
                如果为“False”，则输出将是完整的 JSON 对象。
        加薪：
            OutputParserException：如果生成不是聊天生成。
        返回：
            解析的工具调用。"""
        generation = result[0]
        if not isinstance(generation, ChatGeneration):
            msg = "This output parser can only be used with a chat generation."
            raise OutputParserException(msg)
        message = generation.message
        if isinstance(message, AIMessage) and message.tool_calls:
            parsed_tool_calls = [dict(tc) for tc in message.tool_calls]
            for tool_call in parsed_tool_calls:
                if not self.return_id:
                    _ = tool_call.pop("id")
        else:
            try:
                # This exists purely for backward compatibility / cached messages
                # 中文: 这纯粹是为了向后兼容/缓存消息而存在
                # All new messages should use `message.tool_calls`
                # 中文: 所有新消息都应使用“message.tool_calls”
                raw_tool_calls = copy.deepcopy(message.additional_kwargs["tool_calls"])
            except KeyError:
                if self.first_tool_only:
                    return None
                return []
            parsed_tool_calls = parse_tool_calls(
                raw_tool_calls,
                partial=partial,
                strict=self.strict,
                return_id=self.return_id,
            )
        # For backwards compatibility
        # 中文: 为了向后兼容
        for tc in parsed_tool_calls:
            tc["type"] = tc.pop("name")
        if self.first_tool_only:
            parsed_result = list(
                filter(lambda x: x["type"] == self.key_name, parsed_tool_calls)
            )
            single_result = (
                parsed_result[0]
                if parsed_result and parsed_result[0]["type"] == self.key_name
                else None
            )
            if self.return_id:
                return single_result
            if single_result:
                return single_result["args"]
            return None
        return (
            [res for res in parsed_tool_calls if res["type"] == self.key_name]
            if self.return_id
            else [
                res["args"] for res in parsed_tool_calls if res["type"] == self.key_name
            ]
        )


# Common cause of ValidationError is truncated output due to max_tokens.
# 中文: ValidationError 的常见原因是由于 max_tokens 导致输出被截断。
_MAX_TOKENS_ERROR = (
    "Output parser received a `max_tokens` stop reason. "
    "The output is likely incomplete—please increase `max_tokens` "
    "or shorten your prompt."
)


class PydanticToolsParser(JsonOutputToolsParser):
    """Parse tools from OpenAI response.

    中文翻译:
    从 OpenAI 响应中解析工具。"""

    tools: Annotated[list[TypeBaseModel], SkipValidation()]
    """The tools to parse.

    中文翻译:
    解析工具。"""

    # TODO: Support more granular streaming of objects. Currently only streams once all
    # Pydantic object fields are present.
    # 中文: Pydantic 对象场存在。
    def parse_result(self, result: list[Generation], *, partial: bool = False) -> Any:
        """Parse the result of an LLM call to a list of Pydantic objects.

        Args:
            result: The result of the LLM call.
            partial: Whether to parse partial JSON.
                If `True`, the output will be a JSON object containing
                    all the keys that have been returned so far.
                If `False`, the output will be the full JSON object.

        Returns:
            The parsed Pydantic objects.

        Raises:
            ValueError: If the tool call arguments are not a dict.
            ValidationError: If the tool call arguments do not conform
                to the Pydantic model.
        

        中文翻译:
        将 LLM 调用的结果解析为 Pydantic 对象列表。
        参数：
            结果：LLM 调用的结果。
            partial：是否解析部分JSON。
                如果为 True，输出将是一个 JSON 对象，其中包含
                    到目前为止已归还的所有钥匙。
                如果为“False”，则输出将是完整的 JSON 对象。
        返回：
            解析后的 Pydantic 对象。
        加薪：
            ValueError：如果工具调用参数不是字典。
            ValidationError：如果工具调用参数不符合
                到 Pydantic 模型。"""
        json_results = super().parse_result(result, partial=partial)
        if not json_results:
            return None if self.first_tool_only else []

        json_results = [json_results] if self.first_tool_only else json_results
        name_dict_v2: dict[str, TypeBaseModel] = {
            tool.model_config.get("title") or tool.__name__: tool
            for tool in self.tools
            if is_pydantic_v2_subclass(tool)
        }
        name_dict_v1: dict[str, TypeBaseModel] = {
            tool.__name__: tool for tool in self.tools if is_pydantic_v1_subclass(tool)
        }
        name_dict: dict[str, TypeBaseModel] = {**name_dict_v2, **name_dict_v1}
        pydantic_objects = []
        for res in json_results:
            if not isinstance(res["args"], dict):
                if partial:
                    continue
                msg = (
                    f"Tool arguments must be specified as a dict, received: "
                    f"{res['args']}"
                )
                raise ValueError(msg)
            try:
                pydantic_objects.append(name_dict[res["type"]](**res["args"]))
            except (ValidationError, ValueError):
                if partial:
                    continue
                has_max_tokens_stop_reason = any(
                    generation.message.response_metadata.get("stop_reason")
                    == "max_tokens"
                    for generation in result
                    if isinstance(generation, ChatGeneration)
                )
                if has_max_tokens_stop_reason:
                    logger.exception(_MAX_TOKENS_ERROR)
                raise
        if self.first_tool_only:
            return pydantic_objects[0] if pydantic_objects else None
        return pydantic_objects
