"""Agent 结构化输出策略模块。

本模块定义了 Agent 响应格式的类型和策略。

核心策略:
---------
**AutoStrategy**: 自动选择最佳策略（推荐）
**ToolStrategy**: 使用工具调用策略获取结构化输出
**ProviderStrategy**: 使用模型提供商原生结构化输出

支持的 Schema 类型:
-------------------
- Pydantic 模型
- dataclass
- TypedDict
- JSON Schema 字典

错误处理:
---------
- `StructuredOutputError`: 结构化输出基础错误
- `MultipleStructuredOutputsError`: 返回多个结构化输出时的错误
- `StructuredOutputValidationError`: 解析验证失败时的错误

使用示例:
---------
>>> from pydantic import BaseModel
>>> from langchain.agents import create_agent
>>> from langchain.agents.structured_output import ToolStrategy
>>>
>>> class WeatherResponse(BaseModel):
...     temperature: float
...     description: str
>>>
>>> # 使用 Pydantic 模型作为响应格式
>>> agent = create_agent(
...     model="openai:gpt-4o",
...     response_format=WeatherResponse,  # 自动使用 AutoStrategy
... )
>>>
>>> # 或显式使用 ToolStrategy
>>> agent = create_agent(
...     model="openai:gpt-4o",
...     response_format=ToolStrategy(WeatherResponse),
... )
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, is_dataclass
from types import UnionType
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, TypeAdapter
from typing_extensions import Self, is_typeddict

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from langchain_core.messages import AIMessage

# Supported schema types: Pydantic models, dataclasses, TypedDict, JSON schema dicts
# 中文: 支持的模式类型：Pydantic 模型、数据类、TypedDict、JSON 模式字典
SchemaT = TypeVar("SchemaT")

SchemaKind = Literal["pydantic", "dataclass", "typeddict", "json_schema"]


class StructuredOutputError(Exception):
    """Base class for structured output errors.

    中文翻译:
    结构化输出错误的基类。"""

    ai_message: AIMessage


class MultipleStructuredOutputsError(StructuredOutputError):
    """Raised when model returns multiple structured output tool calls when only one is expected.

    中文翻译:
    当模型返回多个结构化输出工具调用（而仅需要一个）时引发。"""

    def __init__(self, tool_names: list[str], ai_message: AIMessage) -> None:
        """Initialize `MultipleStructuredOutputsError`.

        Args:
            tool_names: The names of the tools called for structured output.
            ai_message: The AI message that contained the invalid multiple tool calls.
        

        中文翻译:
        初始化“MultipleStructuredOutputsError”。
        参数：
            tool_names：结构化输出调用的工具的名称。
            ai_message：包含无效的多个工具调用的 AI 消息。"""
        self.tool_names = tool_names
        self.ai_message = ai_message

        super().__init__(
            "Model incorrectly returned multiple structured responses "
            f"({', '.join(tool_names)}) when only one is expected."
        )


class StructuredOutputValidationError(StructuredOutputError):
    """Raised when structured output tool call arguments fail to parse according to the schema.

    中文翻译:
    当结构化输出工具调用参数无法根据架构进行解析时引发。"""

    def __init__(self, tool_name: str, source: Exception, ai_message: AIMessage) -> None:
        """Initialize `StructuredOutputValidationError`.

        Args:
            tool_name: The name of the tool that failed.
            source: The exception that occurred.
            ai_message: The AI message that contained the invalid structured output.
        

        中文翻译:
        初始化“StructuredOutputValidationError”。
        参数：
            tool_name：失败的工具的名称。
            来源：发生的异常。
            ai_message：包含无效结构化输出的 AI 消息。"""
        self.tool_name = tool_name
        self.source = source
        self.ai_message = ai_message
        super().__init__(f"Failed to parse structured output for tool '{tool_name}': {source}.")


def _parse_with_schema(
    schema: type[SchemaT] | dict, schema_kind: SchemaKind, data: dict[str, Any]
) -> Any:
    """Parse data using for any supported schema type.

    Args:
        schema: The schema type (Pydantic model, `dataclass`, or `TypedDict`)
        schema_kind: One of `"pydantic"`, `"dataclass"`, `"typeddict"`, or
            `"json_schema"`
        data: The data to parse

    Returns:
        The parsed instance according to the schema type

    Raises:
        ValueError: If parsing fails
    

    中文翻译:
    使用任何支持的架构类型解析数据。
    参数：
        schema：模式类型（Pydantic 模型、“dataclass”或“TypedDict”）
        schema_kind：“pydantic”、“dataclass”、“typeddict”之一，或
            `“json_schema”`
        data：要解析的数据
    返回：
        根据模式类型解析的实例
    加薪：
        ValueError：如果解析失败"""
    if schema_kind == "json_schema":
        return data
    try:
        adapter: TypeAdapter[SchemaT] = TypeAdapter(schema)
        return adapter.validate_python(data)
    except Exception as e:
        schema_name = getattr(schema, "__name__", str(schema))
        msg = f"Failed to parse data to {schema_name}: {e}"
        raise ValueError(msg) from e


@dataclass(init=False)
class _SchemaSpec(Generic[SchemaT]):
    """Describes a structured output schema.

    中文翻译:
    描述结构化输出模式。"""

    schema: type[SchemaT]
    """The schema for the response, can be a Pydantic model, `dataclass`, `TypedDict`,
    or JSON schema dict.

    中文翻译:
    响应的模式可以是 Pydantic 模型、`dataclass`、`TypedDict`、
    或 JSON 模式字典。"""

    name: str
    """Name of the schema, used for tool calling.

    If not provided, the name will be the model name or `"response_format"` if it's a
    JSON schema.
    

    中文翻译:
    模式名称，用于工具调用。
    如果未提供，名称将为模型名称或“response_format”（如果是）
    JSON 架构。"""

    description: str
    """Custom description of the schema.

    If not provided, provided will use the model's docstring.
    

    中文翻译:
    模式的自定义描述。
    如果未提供，则提供将使用模型的文档字符串。"""

    schema_kind: SchemaKind
    """The kind of schema.

    中文翻译:
    架构的类型。"""

    json_schema: dict[str, Any]
    """JSON schema associated with the schema.

    中文翻译:
    与架构关联的 JSON 架构。"""

    strict: bool | None = None
    """Whether to enforce strict validation of the schema.

    中文翻译:
    是否强制严格验证模式。"""

    def __init__(
        self,
        schema: type[SchemaT],
        *,
        name: str | None = None,
        description: str | None = None,
        strict: bool | None = None,
    ) -> None:
        """Initialize SchemaSpec with schema and optional parameters.

        中文翻译:
        使用架构和可选参数初始化 SchemaSpec。"""
        self.schema = schema

        if name:
            self.name = name
        elif isinstance(schema, dict):
            self.name = str(schema.get("title", f"response_format_{str(uuid.uuid4())[:4]}"))
        else:
            self.name = str(getattr(schema, "__name__", f"response_format_{str(uuid.uuid4())[:4]}"))

        self.description = description or (
            schema.get("description", "")
            if isinstance(schema, dict)
            else getattr(schema, "__doc__", None) or ""
        )

        self.strict = strict

        if isinstance(schema, dict):
            self.schema_kind = "json_schema"
            self.json_schema = schema
        elif isinstance(schema, type) and issubclass(schema, BaseModel):
            self.schema_kind = "pydantic"
            self.json_schema = schema.model_json_schema()
        elif is_dataclass(schema):
            self.schema_kind = "dataclass"
            self.json_schema = TypeAdapter(schema).json_schema()
        elif is_typeddict(schema):
            self.schema_kind = "typeddict"
            self.json_schema = TypeAdapter(schema).json_schema()
        else:
            msg = (
                f"Unsupported schema type: {type(schema)}. "
                f"Supported types: Pydantic models, dataclasses, TypedDicts, and JSON schema dicts."
            )
            raise ValueError(msg)


@dataclass(init=False)
class ToolStrategy(Generic[SchemaT]):
    """Use a tool calling strategy for model responses.

    中文翻译:
    使用工具调用策略进行模型响应。"""

    schema: type[SchemaT]
    """Schema for the tool calls.

    中文翻译:
    工具调用的架构。"""

    schema_specs: list[_SchemaSpec[SchemaT]]
    """Schema specs for the tool calls.

    中文翻译:
    工具调用的架构规范。"""

    tool_message_content: str | None
    """The content of the tool message to be returned when the model calls
    an artificial structured output tool.

    中文翻译:
    模型调用时返回的工具消息内容
    人工结构化输出工具。"""

    handle_errors: (
        bool | str | type[Exception] | tuple[type[Exception], ...] | Callable[[Exception], str]
    )
    """Error handling strategy for structured output via `ToolStrategy`.

    - `True`: Catch all errors with default error template
    - `str`: Catch all errors with this custom message
    - `type[Exception]`: Only catch this exception type with default message
    - `tuple[type[Exception], ...]`: Only catch these exception types with default
        message
    - `Callable[[Exception], str]`: Custom function that returns error message
    - `False`: No retry, let exceptions propagate
    

    中文翻译:
    通过“ToolStrategy”进行结构化输出的错误处理策略。
    - `True`：使用默认错误模板捕获所有错误
    - `str`：使用此自定义消息捕获所有错误
    - `type[Exception]`：仅捕获带有默认消息的异常类型
    - `tuple[type[Exception], ...]`：仅捕获这些默认的异常类型
        留言
    - `Callable[[Exception], str]`：返回错误消息的自定义函数
    - `False`：不重试，让异常传播"""

    def __init__(
        self,
        schema: type[SchemaT],
        *,
        tool_message_content: str | None = None,
        handle_errors: bool
        | str
        | type[Exception]
        | tuple[type[Exception], ...]
        | Callable[[Exception], str] = True,
    ) -> None:
        """Initialize `ToolStrategy`.

        Initialize `ToolStrategy` with schemas, tool message content, and error handling
        strategy.
        

        中文翻译:
        初始化“工具策略”。
        使用架构、工具消息内容和错误处理初始化“ToolStrategy”
        策略。"""
        self.schema = schema
        self.tool_message_content = tool_message_content
        self.handle_errors = handle_errors

        def _iter_variants(schema: Any) -> Iterable[Any]:
            """Yield leaf variants from Union and JSON Schema oneOf.

            中文翻译:
            产生来自 Union 和 JSON Schema oneOf 的叶变体。"""
            if get_origin(schema) in {UnionType, Union}:
                for arg in get_args(schema):
                    yield from _iter_variants(arg)
                return

            if isinstance(schema, dict) and "oneOf" in schema:
                for sub in schema.get("oneOf", []):
                    yield from _iter_variants(sub)
                return

            yield schema

        self.schema_specs = [_SchemaSpec(s) for s in _iter_variants(schema)]


@dataclass(init=False)
class ProviderStrategy(Generic[SchemaT]):
    """Use the model provider's native structured output method.

    中文翻译:
    使用模型提供者的本机结构化输出方法。"""

    schema: type[SchemaT]
    """Schema for native mode.

    中文翻译:
    本机模式的架构。"""

    schema_spec: _SchemaSpec[SchemaT]
    """Schema spec for native mode.

    中文翻译:
    本机模式的架构规范。"""

    def __init__(
        self,
        schema: type[SchemaT],
        *,
        strict: bool | None = None,
    ) -> None:
        """Initialize ProviderStrategy with schema.

        Args:
            schema: Schema to enforce via the provider's native structured output.
            strict: Whether to request strict provider-side schema enforcement.
        

        中文翻译:
        使用架构初始化 ProviderStrategy。
        参数：
            schema：通过提供者的本机结构化输出强制执行的模式。
            strict：是否要求严格的提供者端架构执行。"""
        self.schema = schema
        self.schema_spec = _SchemaSpec(schema, strict=strict)

    def to_model_kwargs(self) -> dict[str, Any]:
        """Convert to kwargs to bind to a model to force structured output.

        中文翻译:
        转换为 kwargs 以绑定到模型以强制结构化输出。"""
        # OpenAI:
        # 中文: 开放人工智能：
        # - see https://platform.openai.com/docs/guides/structured-outputs
        # 中文: - 请参阅 https://platform.openai.com/docs/guides/structed-outputs
        json_schema: dict[str, Any] = {
            "name": self.schema_spec.name,
            "schema": self.schema_spec.json_schema,
        }
        if self.schema_spec.strict:
            json_schema["strict"] = True

        response_format: dict[str, Any] = {
            "type": "json_schema",
            "json_schema": json_schema,
        }
        return {"response_format": response_format}


@dataclass
class OutputToolBinding(Generic[SchemaT]):
    """Information for tracking structured output tool metadata.

    This contains all necessary information to handle structured responses
    generated via tool calls, including the original schema, its type classification,
    and the corresponding tool implementation used by the tools strategy.
    

    中文翻译:
    用于跟踪结构化输出工具元数据的信息。
    这包含处理结构化响应的所有必要信息
    通过工具调用生成，包括原始模式、其类型分类、
    以及工具策略使用的相应工具实现。"""

    schema: type[SchemaT]
    """The original schema provided for structured output
    (Pydantic model, dataclass, TypedDict, or JSON schema dict).

    中文翻译:
    为结构化输出提供的原始模式
    （Pydantic 模型、数据类、TypedDict 或 JSON 模式字典）。"""

    schema_kind: SchemaKind
    """Classification of the schema type for proper response construction.

    中文翻译:
    对模式类型进行分类，以进行正确的响应构造。"""

    tool: BaseTool
    """LangChain tool instance created from the schema for model binding.

    中文翻译:
    根据模型绑定模式创建 LangChain 工具实例。"""

    @classmethod
    def from_schema_spec(cls, schema_spec: _SchemaSpec[SchemaT]) -> Self:
        """Create an `OutputToolBinding` instance from a `SchemaSpec`.

        Args:
            schema_spec: The `SchemaSpec` to convert

        Returns:
            An `OutputToolBinding` instance with the appropriate tool created
        

        中文翻译:
        从“SchemaSpec”创建“OutputToolBinding”实例。
        参数：
            schema_spec：要转换的“SchemaSpec”
        返回：
            创建了具有适当工具的“OutputToolBinding”实例"""
        return cls(
            schema=schema_spec.schema,
            schema_kind=schema_spec.schema_kind,
            tool=StructuredTool(
                args_schema=schema_spec.json_schema,
                name=schema_spec.name,
                description=schema_spec.description,
            ),
        )

    def parse(self, tool_args: dict[str, Any]) -> SchemaT:
        """Parse tool arguments according to the schema.

        Args:
            tool_args: The arguments from the tool call

        Returns:
            The parsed response according to the schema type

        Raises:
            ValueError: If parsing fails
        

        中文翻译:
        根据模式解析工具参数。
        参数：
            tool_args：工具调用的参数
        返回：
            根据模式类型解析的响应
        加薪：
            ValueError：如果解析失败"""
        return _parse_with_schema(self.schema, self.schema_kind, tool_args)


@dataclass
class ProviderStrategyBinding(Generic[SchemaT]):
    """Information for tracking native structured output metadata.

    This contains all necessary information to handle structured responses
    generated via native provider output, including the original schema,
    its type classification, and parsing logic for provider-enforced JSON.
    

    中文翻译:
    用于跟踪本机结构化输出元数据的信息。
    这包含处理结构化响应的所有必要信息
    通过本机提供程序输出生成，包括原始模式，
    它的类型分类以及提供者强制执行的 JSON 的解析逻辑。"""

    schema: type[SchemaT]
    """The original schema provided for structured output
    (Pydantic model, `dataclass`, `TypedDict`, or JSON schema dict).

    中文翻译:
    为结构化输出提供的原始模式
    （Pydantic 模型、`dataclass`、`TypedDict` 或 JSON 模式字典）。"""

    schema_kind: SchemaKind
    """Classification of the schema type for proper response construction.

    中文翻译:
    对模式类型进行分类，以进行正确的响应构造。"""

    @classmethod
    def from_schema_spec(cls, schema_spec: _SchemaSpec[SchemaT]) -> Self:
        """Create a `ProviderStrategyBinding` instance from a `SchemaSpec`.

        Args:
            schema_spec: The `SchemaSpec` to convert

        Returns:
            A `ProviderStrategyBinding` instance for parsing native structured output
        

        中文翻译:
        从“SchemaSpec”创建“ProviderStrategyBinding”实例。
        参数：
            schema_spec：要转换的“SchemaSpec”
        返回：
            用于解析本机结构化输出的“ProviderStrategyBinding”实例"""
        return cls(
            schema=schema_spec.schema,
            schema_kind=schema_spec.schema_kind,
        )

    def parse(self, response: AIMessage) -> SchemaT:
        """Parse `AIMessage` content according to the schema.

        Args:
            response: The `AIMessage` containing the structured output

        Returns:
            The parsed response according to the schema

        Raises:
            ValueError: If text extraction, JSON parsing or schema validation fails
        

        中文翻译:
        根据模式解析“AIMessage”内容。
        参数：
            响应：包含结构化输出的“AIMessage”
        返回：
            根据模式解析的响应
        加薪：
            ValueError：如果文本提取、JSON 解析或架构验证失败"""
        # Extract text content from AIMessage and parse as JSON
        # 中文: 从AIMessage中提取文本内容并解析为JSON
        raw_text = self._extract_text_content_from_message(response)

        try:
            data = json.loads(raw_text)
        except Exception as e:
            schema_name = getattr(self.schema, "__name__", "response_format")
            msg = (
                f"Native structured output expected valid JSON for {schema_name}, "
                f"but parsing failed: {e}."
            )
            raise ValueError(msg) from e

        # Parse according to schema
        # 中文: 根据模式解析
        return _parse_with_schema(self.schema, self.schema_kind, data)

    def _extract_text_content_from_message(self, message: AIMessage) -> str:
        """Extract text content from an AIMessage.

        Args:
            message: The AI message to extract text from

        Returns:
            The extracted text content
        

        中文翻译:
        从 AIMessage 中提取文本内容。
        参数：
            message：要从中提取文本的 AI 消息
        返回：
            提取的文本内容"""
        content = message.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for c in content:
                if isinstance(c, dict):
                    if c.get("type") == "text" and "text" in c:
                        parts.append(str(c["text"]))
                    elif "content" in c and isinstance(c["content"], str):
                        parts.append(c["content"])
                else:
                    parts.append(str(c))
            return "".join(parts)
        return str(content)


class AutoStrategy(Generic[SchemaT]):
    """Automatically select the best strategy for structured output.

    中文翻译:
    自动选择结构化输出的最佳策略。"""

    schema: type[SchemaT]
    """Schema for automatic mode.

    中文翻译:
    自动模式的架构。"""

    def __init__(
        self,
        schema: type[SchemaT],
    ) -> None:
        """Initialize AutoStrategy with schema.

        中文翻译:
        使用架构初始化 AutoStrategy。"""
        self.schema = schema


ResponseFormat = ToolStrategy[SchemaT] | ProviderStrategy[SchemaT] | AutoStrategy[SchemaT]
