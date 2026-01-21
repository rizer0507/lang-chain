"""Base classes and utilities for LangChain tools.

中文翻译:
LangChain 工具的基类和实用程序。"""

from __future__ import annotations

import functools
import inspect
import json
import typing
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable  # noqa: TC003
from inspect import signature
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Literal,
    TypeVar,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

import typing_extensions
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PydanticDeprecationWarning,
    SkipValidation,
    ValidationError,
    validate_arguments,
)
from pydantic.fields import FieldInfo
from pydantic.v1 import BaseModel as BaseModelV1
from pydantic.v1 import ValidationError as ValidationErrorV1
from pydantic.v1 import validate_arguments as validate_arguments_v1
from typing_extensions import override

from langchain_core.callbacks import (
    AsyncCallbackManager,
    CallbackManager,
    Callbacks,
)
from langchain_core.messages.tool import ToolCall, ToolMessage, ToolOutputMixin
from langchain_core.runnables import (
    RunnableConfig,
    RunnableSerializable,
    ensure_config,
    patch_config,
    run_in_executor,
)
from langchain_core.runnables.config import set_config_context
from langchain_core.runnables.utils import coro_with_context
from langchain_core.utils.function_calling import (
    _parse_google_docstring,
    _py_38_safe_origin,
)
from langchain_core.utils.pydantic import (
    TypeBaseModel,
    _create_subset_model,
    get_fields,
    is_basemodel_subclass,
    is_pydantic_v1_subclass,
    is_pydantic_v2_subclass,
)

if TYPE_CHECKING:
    import uuid
    from collections.abc import Sequence

FILTERED_ARGS = ("run_manager", "callbacks")
TOOL_MESSAGE_BLOCK_TYPES = (
    "text",
    "image_url",
    "image",
    "json",
    "search_result",
    "custom_tool_call_output",
    "document",
    "file",
)


class SchemaAnnotationError(TypeError):
    """Raised when args_schema is missing or has an incorrect type annotation.

    中文翻译:
    当 args_schema 丢失或类型注释不正确时引发。"""


def _is_annotated_type(typ: type[Any]) -> bool:
    """Check if a type is an Annotated type.

    Args:
        typ: The type to check.

    Returns:
        `True` if the type is an Annotated type, `False` otherwise.
    

    中文翻译:
    检查类型是否是带注释的类型。
    参数：
        type：要检查的类型。
    返回：
        如果类型是带注释的类型，则为“True”，否则为“False”。"""
    return get_origin(typ) in {typing.Annotated, typing_extensions.Annotated}


def _get_annotation_description(arg_type: type) -> str | None:
    """Extract description from an Annotated type.

    Checks for string annotations and `FieldInfo` objects with descriptions.

    Args:
        arg_type: The type to extract description from.

    Returns:
        The description string if found, None otherwise.
    

    中文翻译:
    从带注释的类型中提取描述。
    检查字符串注释和带有描述的“FieldInfo”对象。
    参数：
        arg_type：从中提取描述的类型。
    返回：
        如果找到则为描述字符串，否则为 None。"""
    if _is_annotated_type(arg_type):
        annotated_args = get_args(arg_type)
        for annotation in annotated_args[1:]:
            if isinstance(annotation, str):
                return annotation
            if isinstance(annotation, FieldInfo) and annotation.description:
                return annotation.description
    return None


def _get_filtered_args(
    inferred_model: type[BaseModel],
    func: Callable,
    *,
    filter_args: Sequence[str],
    include_injected: bool = True,
) -> dict:
    """Get filtered arguments from a function's signature.

    Args:
        inferred_model: The Pydantic model inferred from the function.
        func: The function to extract arguments from.
        filter_args: Arguments to exclude from the result.
        include_injected: Whether to include injected arguments.

    Returns:
        Dictionary of filtered arguments with their schema definitions.
    

    中文翻译:
    从函数的签名中获取过滤后的参数。
    参数：
        inferred_model：从函数推断的 Pydantic 模型。
        func：从中提取参数的函数。
        filter_args：要从结果中排除的参数。
        include_injected：是否包含注入的参数。
    返回：
        过滤参数及其模式定义的字典。"""
    schema = inferred_model.model_json_schema()["properties"]
    valid_keys = signature(func).parameters
    return {
        k: schema[k]
        for i, (k, param) in enumerate(valid_keys.items())
        if k not in filter_args
        and (i > 0 or param.name not in {"self", "cls"})
        and (include_injected or not _is_injected_arg_type(param.annotation))
    }


def _parse_python_function_docstring(
    function: Callable, annotations: dict, *, error_on_invalid_docstring: bool = False
) -> tuple[str, dict]:
    """Parse function and argument descriptions from a docstring.

    Assumes the function docstring follows Google Python style guide.

    Args:
        function: The function to parse the docstring from.
        annotations: Type annotations for the function parameters.
        error_on_invalid_docstring: Whether to raise an error on invalid docstring.

    Returns:
        A tuple containing the function description and argument descriptions.
    

    中文翻译:
    从文档字符串中解析函数和参数描述。
    假设函数文档字符串遵循 Google Python 风格指南。
    参数：
        function：解析文档字符串的函数。
        注释：为函数参数键入注释。
        error_on_invalid_docstring：是否针对无效文档字符串引发错误。
    返回：
        包含函数描述和参数描述的元组。"""
    docstring = inspect.getdoc(function)
    return _parse_google_docstring(
        docstring,
        list(annotations),
        error_on_invalid_docstring=error_on_invalid_docstring,
    )


def _validate_docstring_args_against_annotations(
    arg_descriptions: dict, annotations: dict
) -> None:
    """Validate that docstring arguments match function annotations.

    Args:
        arg_descriptions: Arguments described in the docstring.
        annotations: Type annotations from the function signature.

    Raises:
        ValueError: If a docstring argument is not found in function signature.
    

    中文翻译:
    验证文档字符串参数是否与函数注释匹配。
    参数：
        arg_descriptions：文档字符串中描述的参数。
        注释：从函数签名中键入注释。
    加薪：
        ValueError：如果在函数签名中找不到文档字符串参数。"""
    for docstring_arg in arg_descriptions:
        if docstring_arg not in annotations:
            msg = f"Arg {docstring_arg} in docstring not found in function signature."
            raise ValueError(msg)


def _infer_arg_descriptions(
    fn: Callable,
    *,
    parse_docstring: bool = False,
    error_on_invalid_docstring: bool = False,
) -> tuple[str, dict]:
    """Infer argument descriptions from function docstring and annotations.

    Args:
        fn: The function to infer descriptions from.
        parse_docstring: Whether to parse the docstring for descriptions.
        error_on_invalid_docstring: Whether to raise error on invalid docstring.

    Returns:
        A tuple containing the function description and argument descriptions.
    

    中文翻译:
    从函数文档字符串和注释推断参数描述。
    参数：
        fn：从中推断描述的函数。
        parse_docstring：是否解析文档字符串以获取描述。
        error_on_invalid_docstring：是否在无效文档字符串上引发错误。
    返回：
        包含函数描述和参数描述的元组。"""
    annotations = typing.get_type_hints(fn, include_extras=True)
    if parse_docstring:
        description, arg_descriptions = _parse_python_function_docstring(
            fn, annotations, error_on_invalid_docstring=error_on_invalid_docstring
        )
    else:
        description = inspect.getdoc(fn) or ""
        arg_descriptions = {}
    if parse_docstring:
        _validate_docstring_args_against_annotations(arg_descriptions, annotations)
    for arg, arg_type in annotations.items():
        if arg in arg_descriptions:
            continue
        if desc := _get_annotation_description(arg_type):
            arg_descriptions[arg] = desc
    return description, arg_descriptions


def _is_pydantic_annotation(annotation: Any, pydantic_version: str = "v2") -> bool:
    """Check if a type annotation is a Pydantic model.

    Args:
        annotation: The type annotation to check.
        pydantic_version: The Pydantic version to check against ("v1" or "v2").

    Returns:
        `True` if the annotation is a Pydantic model, `False` otherwise.
    

    中文翻译:
    检查类型注释是否是 Pydantic 模型。
    参数：
        注释：要检查的类型注释。
        pydantic_version：要检查的 Pydantic 版本（“v1”或“v2”）。
    返回：
        如果注释是 Pydantic 模型，则为“True”，否则为“False”。"""
    base_model_class = BaseModelV1 if pydantic_version == "v1" else BaseModel
    try:
        return issubclass(annotation, base_model_class)
    except TypeError:
        return False


def _function_annotations_are_pydantic_v1(
    signature: inspect.Signature, func: Callable
) -> bool:
    """Check if all Pydantic annotations in a function are from V1.

    Args:
        signature: The function signature to check.
        func: The function being checked.

    Returns:
        True if all Pydantic annotations are from V1, `False` otherwise.

    Raises:
        NotImplementedError: If the function contains mixed V1 and V2 annotations.
    

    中文翻译:
    检查函数中的所有 Pydantic 注释是否来自 V1。
    参数：
        签名：要检查的函数签名。
        func：正在检查的函数。
    返回：
        如果所有 Pydantic 注释都来自 V1，则为 True，否则为 False。
    加薪：
        NotImplementedError：如果函数包含混合的 V1 和 V2 注释。"""
    any_v1_annotations = any(
        _is_pydantic_annotation(parameter.annotation, pydantic_version="v1")
        for parameter in signature.parameters.values()
    )
    any_v2_annotations = any(
        _is_pydantic_annotation(parameter.annotation, pydantic_version="v2")
        for parameter in signature.parameters.values()
    )
    if any_v1_annotations and any_v2_annotations:
        msg = (
            f"Function {func} contains a mix of Pydantic v1 and v2 annotations. "
            "Only one version of Pydantic annotations per function is supported."
        )
        raise NotImplementedError(msg)
    return any_v1_annotations and not any_v2_annotations


class _SchemaConfig:
    """Configuration for Pydantic models generated from function signatures.

    中文翻译:
    从函数签名生成的 Pydantic 模型的配置。"""

    extra: str = "forbid"
    """Whether to allow extra fields in the model.

    中文翻译:
    是否允许模型中存在额外字段。"""
    arbitrary_types_allowed: bool = True
    """Whether to allow arbitrary types in the model.

    中文翻译:
    是否允许模型中的任意类型。"""


def create_schema_from_function(
    model_name: str,
    func: Callable,
    *,
    filter_args: Sequence[str] | None = None,
    parse_docstring: bool = False,
    error_on_invalid_docstring: bool = False,
    include_injected: bool = True,
) -> type[BaseModel]:
    """Create a Pydantic schema from a function's signature.

    Args:
        model_name: Name to assign to the generated Pydantic schema.
        func: Function to generate the schema from.
        filter_args: Optional list of arguments to exclude from the schema.
            Defaults to `FILTERED_ARGS`.
        parse_docstring: Whether to parse the function's docstring for descriptions
            for each argument.
        error_on_invalid_docstring: if `parse_docstring` is provided, configure
            whether to raise `ValueError` on invalid Google Style docstrings.
        include_injected: Whether to include injected arguments in the schema.
            Defaults to `True`, since we want to include them in the schema
            when *validating* tool inputs.

    Returns:
        A Pydantic model with the same arguments as the function.
    

    中文翻译:
    从函数的签名创建 Pydantic 模式。
    参数：
        model_name：分配给生成的 Pydantic 模式的名称。
        func：生成模式的函数。
        filter_args：要从架构中排除的可选参数列表。
            默认为“FILTERED_ARGS”。
        parse_docstring：是否解析函数的文档字符串以获取描述
            对于每个参数。
        error_on_invalid_docstring：如果提供了“parse_docstring”，请配置
            是否在无效的 Google 样式文档字符串上引发“ValueError”。
        include_injected：是否在架构中包含注入的参数。
            默认为“True”，因为我们希望将它们包含在架构中
            当*验证*工具输入时。
    返回：
        与函数具有相同参数的 Pydantic 模型。"""
    sig = inspect.signature(func)

    if _function_annotations_are_pydantic_v1(sig, func):
        validated = validate_arguments_v1(func, config=_SchemaConfig)  # type: ignore[call-overload]
    else:
        # https://docs.pydantic.dev/latest/usage/validation_decorator/
        # 中文: https://docs.pydantic.dev/latest/usage/validation_decorator/
        with warnings.catch_warnings():
            # We are using deprecated functionality here.
            # 中文: 我们在这里使用已弃用的功能。
            # This code should be re-written to simply construct a Pydantic model
            # 中文: 应该重写此代码以简单地构建 Pydantic 模型
            # using inspect.signature and create_model.
            # 中文: 使用inspect.signature和create_model。
            warnings.simplefilter("ignore", category=PydanticDeprecationWarning)
            validated = validate_arguments(func, config=_SchemaConfig)  # type: ignore[operator]

    # Let's ignore `self` and `cls` arguments for class and instance methods
    # 中文: 让我们忽略类和实例方法的“self”和“cls”参数
    # If qualified name has a ".", then it likely belongs in a class namespace
    # 中文: 如果限定名称有一个“.”，那么它可能属于类命名空间
    in_class = bool(func.__qualname__ and "." in func.__qualname__)

    has_args = False
    has_kwargs = False

    for param in sig.parameters.values():
        if param.kind == param.VAR_POSITIONAL:
            has_args = True
        elif param.kind == param.VAR_KEYWORD:
            has_kwargs = True

    inferred_model = validated.model

    if filter_args:
        filter_args_ = filter_args
    else:
        # Handle classmethods and instance methods
        # 中文: 处理类方法和实例方法
        existing_params: list[str] = list(sig.parameters.keys())
        if existing_params and existing_params[0] in {"self", "cls"} and in_class:
            filter_args_ = [existing_params[0], *list(FILTERED_ARGS)]
        else:
            filter_args_ = list(FILTERED_ARGS)

        for existing_param in existing_params:
            if not include_injected and _is_injected_arg_type(
                sig.parameters[existing_param].annotation
            ):
                filter_args_.append(existing_param)

    description, arg_descriptions = _infer_arg_descriptions(
        func,
        parse_docstring=parse_docstring,
        error_on_invalid_docstring=error_on_invalid_docstring,
    )
    # Pydantic adds placeholder virtual fields we need to strip
    # 中文: Pydantic 添加了我们需要剥离的占位符虚拟字段
    valid_properties = []
    for field in get_fields(inferred_model):
        if not has_args and field == "args":
            continue
        if not has_kwargs and field == "kwargs":
            continue

        if field == "v__duplicate_kwargs":  # Internal pydantic field
            continue

        if field not in filter_args_:
            valid_properties.append(field)

    return _create_subset_model(
        model_name,
        inferred_model,
        list(valid_properties),
        descriptions=arg_descriptions,
        fn_description=description,
    )


class ToolException(Exception):  # noqa: N818
    """Exception thrown when a tool execution error occurs.

    This exception allows tools to signal errors without stopping the agent.
    The error is handled according to the tool's handle_tool_error setting,
    and the result is returned as an observation to the agent.
    

    中文翻译:
    工具执行错误时抛出异常。
    此异常允许工具在不停止代理的情况下发出错误信号。
    根据工具的handle_tool_error设置处理错误，
    并将结果作为观察结果返回给代理。"""


ArgsSchema = TypeBaseModel | dict[str, Any]

_EMPTY_SET: frozenset[str] = frozenset()


class BaseTool(RunnableSerializable[str | dict | ToolCall, Any]):
    """Base class for all LangChain tools.

    This abstract class defines the interface that all LangChain tools must implement.

    Tools are components that can be called by agents to perform specific actions.
    

    中文翻译:
    所有 LangChain 工具的基类。
    这个抽象类定义了所有LangChain工具必须实现的接口。
    工具是代理可以调用​​以执行特定操作的组件。"""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Validate the tool class definition during subclass creation.

        Args:
            **kwargs: Additional keyword arguments passed to the parent class.

        Raises:
            SchemaAnnotationError: If `args_schema` has incorrect type annotation.
        

        中文翻译:
        在子类创建期间验证工具类定义。
        参数：
            **kwargs：传递给父类的附加关键字参数。
        加薪：
            SchemaAnnotationError：如果 `args_schema` 有不正确的类型注释。"""
        super().__init_subclass__(**kwargs)

        args_schema_type = cls.__annotations__.get("args_schema", None)

        if args_schema_type is not None and args_schema_type == BaseModel:
            # Throw errors for common mis-annotations.
            # 中文: 抛出常见错误注释的错误。
            # TODO: Use get_args / get_origin and fully
            # specify valid annotations.
            # 中文: 指定有效的注释。
            typehint_mandate = """
class ChildTool(BaseTool):
    ...
    args_schema: Type[BaseModel] = SchemaClass
    ...

 中文翻译:
 子工具类（基础工具）：
    ...
    args_schema：类型[BaseModel] = SchemaClass
    ..."""
            name = cls.__name__
            msg = (
                f"Tool definition for {name} must include valid type annotations"
                f" for argument 'args_schema' to behave as expected.\n"
                f"Expected annotation of 'Type[BaseModel]'"
                f" but got '{args_schema_type}'.\n"
                f"Expected class looks like:\n"
                f"{typehint_mandate}"
            )
            raise SchemaAnnotationError(msg)

    name: str
    """The unique name of the tool that clearly communicates its purpose.

    中文翻译:
    清楚传达其用途的工具的唯一名称。"""
    description: str
    """Used to tell the model how/when/why to use the tool.

    You can provide few-shot examples as a part of the description.
    

    中文翻译:
    用于告诉模型如何/何时/为何使用该工具。
    您可以提供少量示例作为描述的一部分。"""

    args_schema: Annotated[ArgsSchema | None, SkipValidation()] = Field(
        default=None, description="The tool schema."
    )
    """Pydantic model class to validate and parse the tool's input arguments.

    Args schema should be either:

    - A subclass of `pydantic.BaseModel`.
    - A subclass of `pydantic.v1.BaseModel` if accessing v1 namespace in pydantic 2
    - A JSON schema dict
    

    中文翻译:
    Pydantic 模型类用于验证和解析工具的输入参数。
    Args 模式应该是：
    - `pydantic.BaseModel` 的子类。
    - 如果访问 pydantic 2 中的 v1 命名空间，则为“pydantic.v1.BaseModel”的子类
    - JSON 模式字典"""
    return_direct: bool = False
    """Whether to return the tool's output directly.

    Setting this to `True` means that after the tool is called, the `AgentExecutor` will
    stop looping.
    

    中文翻译:
    是否直接返回工具的输出。
    设置为“True”意味着调用该工具后，“AgentExecutor”将
    停止循环。"""
    verbose: bool = False
    """Whether to log the tool's progress.

    中文翻译:
    是否记录工具的进度。"""

    callbacks: Callbacks = Field(default=None, exclude=True)
    """Callbacks to be called during tool execution.

    中文翻译:
    在工具执行期间调用的回调。"""

    tags: list[str] | None = None
    """Optional list of tags associated with the tool.

    These tags will be associated with each call to this tool,
    and passed as arguments to the handlers defined in `callbacks`.

    You can use these to, e.g., identify a specific instance of a tool with its use
    case.
    

    中文翻译:
    与该工具关联的可选标签列表。
    这些标签将与每次调用该工具相关联，
    并作为参数传递给“callbacks”中定义的处理程序。
    例如，您可以使用它们来识别工具的特定实例及其使用情况
    案例。"""
    metadata: dict[str, Any] | None = None
    """Optional metadata associated with the tool.

    This metadata will be associated with each call to this tool,
    and passed as arguments to the handlers defined in `callbacks`.

    You can use these to, e.g., identify a specific instance of a tool with its use
    case.
    

    中文翻译:
    与该工具关联的可选元数据。
    该元数据将与对该工具的每次调用相关联，
    并作为参数传递给“callbacks”中定义的处理程序。
    例如，您可以使用它们来识别工具的特定实例及其使用情况
    案例。"""

    handle_tool_error: bool | str | Callable[[ToolException], str] | None = False
    """Handle the content of the `ToolException` thrown.

    中文翻译:
    处理抛出的`ToolException`的内容。"""

    handle_validation_error: (
        bool | str | Callable[[ValidationError | ValidationErrorV1], str] | None
    ) = False
    """Handle the content of the `ValidationError` thrown.

    中文翻译:
    处理抛出的“ValidationError”的内容。"""

    response_format: Literal["content", "content_and_artifact"] = "content"
    """The tool response format.

    If `'content'` then the output of the tool is interpreted as the contents of a
    `ToolMessage`. If `'content_and_artifact'` then the output is expected to be a
    two-tuple corresponding to the `(content, artifact)` of a `ToolMessage`.
    

    中文翻译:
    工具响应格式。
    如果为“content”，则该工具的输出将被解释为
    `工具消息`。如果“content_and_artifact”那么输出预计是
    对应于“ToolMessage”的“(content,artifact)”的二元组。"""

    extras: dict[str, Any] | None = None
    """Optional provider-specific extra fields for the tool.

    This is used to pass provider-specific configuration that doesn't fit into
    standard tool fields.

    Example:
        Anthropic-specific fields like [`cache_control`](https://docs.langchain.com/oss/python/integrations/chat/anthropic#prompt-caching),
        [`defer_loading`](https://docs.langchain.com/oss/python/integrations/chat/anthropic#tool-search),
        or `input_examples`.

        ```python
        @tool(extras={"defer_loading": True, "cache_control": {"type": "ephemeral"}})
        def my_tool(x: str) -> str:
            return x
        ```
    

    中文翻译:
    该工具的可选特定于提供商的额外字段。
    这用于传递不适合的特定于提供者的配置
    标准工具字段。
    示例：
        人类特定的字段，例如 [`cache_control`](https://docs.langchain.com/oss/python/integrations/chat/anthropic#prompt-caching)，
        [`defer_loading`](https://docs.langchain.com/oss/python/integrations/chat/anthropic#tool-search),
        或“输入示例”。
        ````蟒蛇
        @tool(extras={"defer_loading": True, "cache_control": {"type": "ephemeral"}})
        def my_tool(x: str) -> str:
            返回x
        ````"""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the tool.

        Raises:
            TypeError: If `args_schema` is not a subclass of pydantic `BaseModel` or
                `dict`.
        

        中文翻译:
        初始化该工具。
        加薪：
            TypeError: 如果 `args_schema` 不是 pydantic `BaseModel` 的子类或者
                `字典`。"""
        if (
            "args_schema" in kwargs
            and kwargs["args_schema"] is not None
            and not is_basemodel_subclass(kwargs["args_schema"])
            and not isinstance(kwargs["args_schema"], dict)
        ):
            msg = (
                "args_schema must be a subclass of pydantic BaseModel or "
                f"a JSON schema dict. Got: {kwargs['args_schema']}."
            )
            raise TypeError(msg)
        super().__init__(**kwargs)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @property
    def is_single_input(self) -> bool:
        """Check if the tool accepts only a single input argument.

        Returns:
            `True` if the tool has only one input argument, `False` otherwise.
        

        中文翻译:
        检查该工具是否只接受单个输入参数。
        返回：
            如果工具只有一个输入参数，则为“True”，否则为“False”。"""
        keys = {k for k in self.args if k != "kwargs"}
        return len(keys) == 1

    @property
    def args(self) -> dict:
        """Get the tool's input arguments schema.

        Returns:
            `dict` containing the tool's argument properties.
        

        中文翻译:
        获取工具的输入参数架构。
        返回：
            包含工具参数属性的“dict”。"""
        if isinstance(self.args_schema, dict):
            json_schema = self.args_schema
        elif self.args_schema and issubclass(self.args_schema, BaseModelV1):
            json_schema = self.args_schema.schema()
        else:
            input_schema = self.tool_call_schema
            if isinstance(input_schema, dict):
                json_schema = input_schema
            else:
                json_schema = input_schema.model_json_schema()
        return cast("dict", json_schema["properties"])

    @property
    def tool_call_schema(self) -> ArgsSchema:
        """Get the schema for tool calls, excluding injected arguments.

        Returns:
            The schema that should be used for tool calls from language models.
        

        中文翻译:
        获取工具调用的架构，不包括注入的参数。
        返回：
            用于从语言模型调用工具的模式。"""
        if isinstance(self.args_schema, dict):
            if self.description:
                return {
                    **self.args_schema,
                    "description": self.description,
                }

            return self.args_schema

        full_schema = self.get_input_schema()
        fields = []
        for name, type_ in get_all_basemodel_annotations(full_schema).items():
            if not _is_injected_arg_type(type_):
                fields.append(name)
        return _create_subset_model(
            self.name, full_schema, fields, fn_description=self.description
        )

    @functools.cached_property
    def _injected_args_keys(self) -> frozenset[str]:
        # base implementation doesn't manage injected args
        # 中文: 基本实现不管理注入的参数
        return _EMPTY_SET

    # --- Runnable ---
    # 中文: --- 可运行 ---

    @override
    def get_input_schema(self, config: RunnableConfig | None = None) -> type[BaseModel]:
        """The tool's input schema.

        Args:
            config: The configuration for the tool.

        Returns:
            The input schema for the tool.
        

        中文翻译:
        该工具的输入架构。
        参数：
            config：工具的配置。
        返回：
            工具的输入架构。"""
        if self.args_schema is not None:
            if isinstance(self.args_schema, dict):
                return super().get_input_schema(config)
            return self.args_schema
        return create_schema_from_function(self.name, self._run)

    @override
    def invoke(
        self,
        input: str | dict | ToolCall,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        tool_input, kwargs = _prep_run_args(input, config, **kwargs)
        return self.run(tool_input, **kwargs)

    @override
    async def ainvoke(
        self,
        input: str | dict | ToolCall,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        tool_input, kwargs = _prep_run_args(input, config, **kwargs)
        return await self.arun(tool_input, **kwargs)

    # --- Tool ---
    # 中文: - - 工具  - -

    def _parse_input(
        self, tool_input: str | dict, tool_call_id: str | None
    ) -> str | dict[str, Any]:
        """Parse and validate tool input using the args schema.

        Args:
            tool_input: The raw input to the tool.
            tool_call_id: The ID of the tool call, if available.

        Returns:
            The parsed and validated input.

        Raises:
            ValueError: If `string` input is provided with JSON schema `args_schema`.
            ValueError: If `InjectedToolCallId` is required but `tool_call_id` is not
                provided.
            TypeError: If `args_schema` is not a Pydantic `BaseModel` or dict.
        

        中文翻译:
        使用 args 架构解析和验证工具输入。
        参数：
            tool_input：工具的原始输入。
            tool_call_id：工具调用的 ID（如果可用）。
        返回：
            已解析和验证的输入。
        加薪：
            ValueError：如果“string”输入提供了 JSON 架构“args_schema”。
            ValueError：如果需要“InjectedToolCallId”但不需要“tool_call_id”
                提供。
            TypeError: 如果 `args_schema` 不是 Pydantic `BaseModel` 或 dict。"""
        input_args = self.args_schema

        if isinstance(tool_input, str):
            if input_args is not None:
                if isinstance(input_args, dict):
                    msg = (
                        "String tool inputs are not allowed when "
                        "using tools with JSON schema args_schema."
                    )
                    raise ValueError(msg)
                key_ = next(iter(get_fields(input_args).keys()))
                if issubclass(input_args, BaseModel):
                    input_args.model_validate({key_: tool_input})
                elif issubclass(input_args, BaseModelV1):
                    input_args.parse_obj({key_: tool_input})
                else:
                    msg = f"args_schema must be a Pydantic BaseModel, got {input_args}"
                    raise TypeError(msg)
            return tool_input

        if input_args is not None:
            if isinstance(input_args, dict):
                return tool_input
            if issubclass(input_args, BaseModel):
                # Check args_schema for InjectedToolCallId
                # 中文: 检查 args_schema 中的 InjectedToolCallId
                for k, v in get_all_basemodel_annotations(input_args).items():
                    if _is_injected_arg_type(v, injected_type=InjectedToolCallId):
                        if tool_call_id is None:
                            msg = (
                                "When tool includes an InjectedToolCallId "
                                "argument, tool must always be invoked with a full "
                                "model ToolCall of the form: {'args': {...}, "
                                "'name': '...', 'type': 'tool_call', "
                                "'tool_call_id': '...'}"
                            )
                            raise ValueError(msg)
                        tool_input[k] = tool_call_id
                result = input_args.model_validate(tool_input)
                result_dict = result.model_dump()
            elif issubclass(input_args, BaseModelV1):
                # Check args_schema for InjectedToolCallId
                # 中文: 检查 args_schema 中的 InjectedToolCallId
                for k, v in get_all_basemodel_annotations(input_args).items():
                    if _is_injected_arg_type(v, injected_type=InjectedToolCallId):
                        if tool_call_id is None:
                            msg = (
                                "When tool includes an InjectedToolCallId "
                                "argument, tool must always be invoked with a full "
                                "model ToolCall of the form: {'args': {...}, "
                                "'name': '...', 'type': 'tool_call', "
                                "'tool_call_id': '...'}"
                            )
                            raise ValueError(msg)
                        tool_input[k] = tool_call_id
                result = input_args.parse_obj(tool_input)
                result_dict = result.dict()
            else:
                msg = (
                    f"args_schema must be a Pydantic BaseModel, got {self.args_schema}"
                )
                raise NotImplementedError(msg)

            # Include fields from tool_input, plus fields with explicit defaults.
            # 中文: 包括来自 tool_input 的字段，以及具有显式默认值的字段。
            # This applies Pydantic defaults (like Field(default=1)) while excluding
            # 中文: 这适用于 Pydantic 默认值（如 Field(default=1)），同时排除
            # synthetic "args"/"kwargs" fields that Pydantic creates for *args/**kwargs.
            # 中文: Pydantic 为 *args/**kwargs 创建的合成“args”/“kwargs”字段。
            field_info = get_fields(input_args)
            validated_input = {}
            for k in result_dict:
                if k in tool_input:
                    # Field was provided in input - include it (validated)
                    # 中文: 输入中提供了字段 - 包含它（已验证）
                    validated_input[k] = getattr(result, k)
                elif k in field_info and k not in ("args", "kwargs"):
                    # Check if field has an explicit default defined in the schema.
                    # 中文: 检查字段是否在架构中定义了显式默认值。
                    # Exclude "args"/"kwargs" as these are synthetic fields for variadic
                    # 中文: 排除“args”/“kwargs”，因为这些是可变参数的合成字段
                    # parameters that should not be passed as keyword arguments.
                    # 中文: 不应作为关键字参数传递的参数。
                    fi = field_info[k]
                    # Pydantic v2 uses is_required() method, v1 uses required attribute
                    # 中文: Pydantic v2 使用 is_required() 方法，v1 使用 required 属性
                    has_default = (
                        not fi.is_required()
                        if hasattr(fi, "is_required")
                        else not getattr(fi, "required", True)
                    )
                    if has_default:
                        validated_input[k] = getattr(result, k)

            for k in self._injected_args_keys:
                if k in tool_input:
                    validated_input[k] = tool_input[k]
                elif k == "tool_call_id":
                    if tool_call_id is None:
                        msg = (
                            "When tool includes an InjectedToolCallId "
                            "argument, tool must always be invoked with a full "
                            "model ToolCall of the form: {'args': {...}, "
                            "'name': '...', 'type': 'tool_call', "
                            "'tool_call_id': '...'}"
                        )
                        raise ValueError(msg)
                    validated_input[k] = tool_call_id

            return validated_input

        return tool_input

    @abstractmethod
    def _run(self, *args: Any, **kwargs: Any) -> Any:
        """Use the tool.

        Add `run_manager: CallbackManagerForToolRun | None = None` to child
        implementations to enable tracing.

        Returns:
            The result of the tool execution.
        

        中文翻译:
        使用该工具。
        添加 `run_manager: CallbackManagerForToolRun | None = None` 给孩子
        启用跟踪的实现。
        返回：
            工具执行的结果。"""

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        """Use the tool asynchronously.

        Add `run_manager: AsyncCallbackManagerForToolRun | None = None` to child
        implementations to enable tracing.

        Returns:
            The result of the tool execution.
        

        中文翻译:
        异步使用该工具。
        添加 `run_manager: AsyncCallbackManagerForToolRun | None = None` 给孩子
        启用跟踪的实现。
        返回：
            工具执行的结果。"""
        if kwargs.get("run_manager") and signature(self._run).parameters.get(
            "run_manager"
        ):
            kwargs["run_manager"] = kwargs["run_manager"].get_sync()
        return await run_in_executor(None, self._run, *args, **kwargs)

    def _filter_injected_args(self, tool_input: dict) -> dict:
        """Filter out injected tool arguments from the input dictionary.

        Injected arguments are those annotated with `InjectedToolArg` or its
        subclasses, or arguments in `FILTERED_ARGS` like `run_manager` and callbacks.

        Args:
            tool_input: The tool input dictionary to filter.

        Returns:
            A filtered dictionary with injected arguments removed.
        

        中文翻译:
        从输入字典中过滤掉注入的工具参数。
        注入的参数是那些用 `InjectedToolArg` 或其注释的参数
        子类，或“FILTERED_ARGS”中的参数，例如“run_manager”和回调。
        参数：
            tool_input：要过滤的工具输入字典。
        返回：
            删除了注入参数的过滤字典。"""
        # Start with filtered args from the constant
        # 中文: 从常量中过滤后的参数开始
        filtered_keys = set[str](FILTERED_ARGS)

        # If we have an args_schema, use it to identify injected args
        # 中文: 如果我们有 args_schema，用它来识别注入的参数
        if self.args_schema is not None:
            try:
                annotations = get_all_basemodel_annotations(self.args_schema)
                for field_name, field_type in annotations.items():
                    if _is_injected_arg_type(field_type):
                        filtered_keys.add(field_name)
            except Exception:  # noqa: S110
                # If we can't get annotations, just use FILTERED_ARGS
                # 中文: 如果我们无法获取注释，只需使用 FILTERED_ARGS
                pass

        # Filter out the injected keys from tool_input
        # 中文: 从tool_input中过滤掉注入的key
        return {k: v for k, v in tool_input.items() if k not in filtered_keys}

    def _to_args_and_kwargs(
        self, tool_input: str | dict, tool_call_id: str | None
    ) -> tuple[tuple, dict]:
        """Convert tool input to positional and keyword arguments.

        Args:
            tool_input: The input to the tool.
            tool_call_id: The ID of the tool call, if available.

        Returns:
            A tuple of `(positional_args, keyword_args)` for the tool.

        Raises:
            TypeError: If the tool input type is invalid.
        

        中文翻译:
        将工具输入转换为位置参数和关键字参数。
        参数：
            tool_input：工具的输入。
            tool_call_id：工具调用的 ID（如果可用）。
        返回：
            该工具的“(positional_args, keywords_args)”元组。
        加薪：
            TypeError：如果工具输入类型无效。"""
        if (
            self.args_schema is not None
            and isinstance(self.args_schema, type)
            and is_basemodel_subclass(self.args_schema)
            and not get_fields(self.args_schema)
        ):
            # StructuredTool with no args
            # 中文: 不带参数的 StructuredTool
            return (), {}
        tool_input = self._parse_input(tool_input, tool_call_id)
        # For backwards compatibility, if run_input is a string,
        # 中文: 为了向后兼容，如果 run_input 是一个字符串，
        # pass as a positional argument.
        # 中文: 作为位置参数传递。
        if isinstance(tool_input, str):
            return (tool_input,), {}
        if isinstance(tool_input, dict):
            # Make a shallow copy of the input to allow downstream code
            # 中文: 制作输入的浅表副本以允许下游代码
            # to modify the root level of the input without affecting the
            # 中文: 修改输入的根级别而不影响
            # original input.
            # 中文: 原始输入。
            # This is used by the tool to inject run time information like
            # 中文: 该工具使用它来注入运行时信息，例如
            # the callback manager.
            # 中文: 回调管理器。
            return (), tool_input.copy()
        # This code path is not expected to be reachable.
        # 中文: 该代码路径预计无法访问。
        msg = f"Invalid tool input type: {type(tool_input)}"
        raise TypeError(msg)

    def run(
        self,
        tool_input: str | dict[str, Any],
        verbose: bool | None = None,  # noqa: FBT001
        start_color: str | None = "green",
        color: str | None = "green",
        callbacks: Callbacks = None,
        *,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        run_name: str | None = None,
        run_id: uuid.UUID | None = None,
        config: RunnableConfig | None = None,
        tool_call_id: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run the tool.

        Args:
            tool_input: The input to the tool.
            verbose: Whether to log the tool's progress.
            start_color: The color to use when starting the tool.
            color: The color to use when ending the tool.
            callbacks: Callbacks to be called during tool execution.
            tags: Optional list of tags associated with the tool.
            metadata: Optional metadata associated with the tool.
            run_name: The name of the run.
            run_id: The id of the run.
            config: The configuration for the tool.
            tool_call_id: The id of the tool call.
            **kwargs: Keyword arguments to be passed to tool callbacks (event handler)

        Returns:
            The output of the tool.

        Raises:
            ToolException: If an error occurs during tool execution.
        

        中文翻译:
        运行该工具。
        参数：
            tool_input：工具的输入。
            verbose：是否记录工具的进度。
            start_color：启动工具时使用的颜色。
            color：结束工具时使用的颜色。
            回调：工具执行期间调用的回调。
            标签：与工具关联的可选标签列表。
            元数据：与工具关联的可选元数据。
            run_name：运行的名称。
            run_id：运行的 id。
            config：工具的配置。
            tool_call_id：工具调用的id。
            **kwargs：要传递给工具回调（事件处理程序）的关键字参数
        返回：
            工具的输出。
        加薪：
            ToolException：如果工具执行期间发生错误。"""
        callback_manager = CallbackManager.configure(
            callbacks,
            self.callbacks,
            self.verbose or bool(verbose),
            tags,
            self.tags,
            metadata,
            self.metadata,
        )

        # Filter out injected arguments from callback inputs
        # 中文: 从回调输入中过滤掉注入的参数
        filtered_tool_input = (
            self._filter_injected_args(tool_input)
            if isinstance(tool_input, dict)
            else None
        )

        # Use filtered inputs for the input_str parameter as well
        # 中文: 也对 input_str 参数使用过滤后的输入
        tool_input_str = (
            tool_input
            if isinstance(tool_input, str)
            else str(
                filtered_tool_input if filtered_tool_input is not None else tool_input
            )
        )

        run_manager = callback_manager.on_tool_start(
            {"name": self.name, "description": self.description},
            tool_input_str,
            color=start_color,
            name=run_name,
            run_id=run_id,
            inputs=filtered_tool_input,
            tool_call_id=tool_call_id,
            **kwargs,
        )

        content = None
        artifact = None
        status = "success"
        error_to_raise: Exception | KeyboardInterrupt | None = None
        try:
            child_config = patch_config(config, callbacks=run_manager.get_child())
            with set_config_context(child_config) as context:
                tool_args, tool_kwargs = self._to_args_and_kwargs(
                    tool_input, tool_call_id
                )
                if signature(self._run).parameters.get("run_manager"):
                    tool_kwargs |= {"run_manager": run_manager}
                if config_param := _get_runnable_config_param(self._run):
                    tool_kwargs |= {config_param: config}
                response = context.run(self._run, *tool_args, **tool_kwargs)
            if self.response_format == "content_and_artifact":
                msg = (
                    "Since response_format='content_and_artifact' "
                    "a two-tuple of the message content and raw tool output is "
                    f"expected. Instead, generated response is of type: "
                    f"{type(response)}."
                )
                if not isinstance(response, tuple):
                    error_to_raise = ValueError(msg)
                else:
                    try:
                        content, artifact = response
                    except ValueError:
                        error_to_raise = ValueError(msg)
            else:
                content = response
        except (ValidationError, ValidationErrorV1) as e:
            if not self.handle_validation_error:
                error_to_raise = e
            else:
                content = _handle_validation_error(e, flag=self.handle_validation_error)
                status = "error"
        except ToolException as e:
            if not self.handle_tool_error:
                error_to_raise = e
            else:
                content = _handle_tool_error(e, flag=self.handle_tool_error)
                status = "error"
        except (Exception, KeyboardInterrupt) as e:
            error_to_raise = e

        if error_to_raise:
            run_manager.on_tool_error(error_to_raise)
            raise error_to_raise
        output = _format_output(content, artifact, tool_call_id, self.name, status)
        run_manager.on_tool_end(output, color=color, name=self.name, **kwargs)
        return output

    async def arun(
        self,
        tool_input: str | dict,
        verbose: bool | None = None,  # noqa: FBT001
        start_color: str | None = "green",
        color: str | None = "green",
        callbacks: Callbacks = None,
        *,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        run_name: str | None = None,
        run_id: uuid.UUID | None = None,
        config: RunnableConfig | None = None,
        tool_call_id: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run the tool asynchronously.

        Args:
            tool_input: The input to the tool.
            verbose: Whether to log the tool's progress.
            start_color: The color to use when starting the tool.
            color: The color to use when ending the tool.
            callbacks: Callbacks to be called during tool execution.
            tags: Optional list of tags associated with the tool.
            metadata: Optional metadata associated with the tool.
            run_name: The name of the run.
            run_id: The id of the run.
            config: The configuration for the tool.
            tool_call_id: The id of the tool call.
            **kwargs: Keyword arguments to be passed to tool callbacks

        Returns:
            The output of the tool.

        Raises:
            ToolException: If an error occurs during tool execution.
        

        中文翻译:
        异步运行该工具。
        参数：
            tool_input：工具的输入。
            verbose：是否记录工具的进度。
            start_color：启动工具时使用的颜色。
            color：结束工具时使用的颜色。
            回调：工具执行期间调用的回调。
            标签：与工具关联的可选标签列表。
            元数据：与工具关联的可选元数据。
            run_name：运行的名称。
            run_id：运行的 id。
            config：工具的配置。
            tool_call_id：工具调用的id。
            **kwargs：要传递给工具回调的关键字参数
        返回：
            工具的输出。
        加薪：
            ToolException：如果工具执行期间发生错误。"""
        callback_manager = AsyncCallbackManager.configure(
            callbacks,
            self.callbacks,
            self.verbose or bool(verbose),
            tags,
            self.tags,
            metadata,
            self.metadata,
        )

        # Filter out injected arguments from callback inputs
        # 中文: 从回调输入中过滤掉注入的参数
        filtered_tool_input = (
            self._filter_injected_args(tool_input)
            if isinstance(tool_input, dict)
            else None
        )

        # Use filtered inputs for the input_str parameter as well
        # 中文: 也对 input_str 参数使用过滤后的输入
        tool_input_str = (
            tool_input
            if isinstance(tool_input, str)
            else str(
                filtered_tool_input if filtered_tool_input is not None else tool_input
            )
        )

        run_manager = await callback_manager.on_tool_start(
            {"name": self.name, "description": self.description},
            tool_input_str,
            color=start_color,
            name=run_name,
            run_id=run_id,
            inputs=filtered_tool_input,
            tool_call_id=tool_call_id,
            **kwargs,
        )
        content = None
        artifact = None
        status = "success"
        error_to_raise: Exception | KeyboardInterrupt | None = None
        try:
            tool_args, tool_kwargs = self._to_args_and_kwargs(tool_input, tool_call_id)
            child_config = patch_config(config, callbacks=run_manager.get_child())
            with set_config_context(child_config) as context:
                func_to_check = (
                    self._run if self.__class__._arun is BaseTool._arun else self._arun  # noqa: SLF001
                )
                if signature(func_to_check).parameters.get("run_manager"):
                    tool_kwargs["run_manager"] = run_manager
                if config_param := _get_runnable_config_param(func_to_check):
                    tool_kwargs[config_param] = config

                coro = self._arun(*tool_args, **tool_kwargs)
                response = await coro_with_context(coro, context)
            if self.response_format == "content_and_artifact":
                msg = (
                    "Since response_format='content_and_artifact' "
                    "a two-tuple of the message content and raw tool output is "
                    f"expected. Instead, generated response is of type: "
                    f"{type(response)}."
                )
                if not isinstance(response, tuple):
                    error_to_raise = ValueError(msg)
                else:
                    try:
                        content, artifact = response
                    except ValueError:
                        error_to_raise = ValueError(msg)
            else:
                content = response
        except ValidationError as e:
            if not self.handle_validation_error:
                error_to_raise = e
            else:
                content = _handle_validation_error(e, flag=self.handle_validation_error)
                status = "error"
        except ToolException as e:
            if not self.handle_tool_error:
                error_to_raise = e
            else:
                content = _handle_tool_error(e, flag=self.handle_tool_error)
                status = "error"
        except (Exception, KeyboardInterrupt) as e:
            error_to_raise = e

        if error_to_raise:
            await run_manager.on_tool_error(error_to_raise)
            raise error_to_raise

        output = _format_output(content, artifact, tool_call_id, self.name, status)
        await run_manager.on_tool_end(output, color=color, name=self.name, **kwargs)
        return output


def _is_tool_call(x: Any) -> bool:
    """Check if the input is a tool call dictionary.

    Args:
        x: The input to check.

    Returns:
        `True` if the input is a tool call, `False` otherwise.
    

    中文翻译:
    检查输入是否是工具调用字典。
    参数：
        x：要检查的输入。
    返回：
        如果输入是工具调用，则为“True”，否则为“False”。"""
    return isinstance(x, dict) and x.get("type") == "tool_call"


def _handle_validation_error(
    e: ValidationError | ValidationErrorV1,
    *,
    flag: Literal[True] | str | Callable[[ValidationError | ValidationErrorV1], str],
) -> str:
    """Handle validation errors based on the configured flag.

    Args:
        e: The validation error that occurred.
        flag: How to handle the error (`bool`, `str`, or `Callable`).

    Returns:
        The error message to return.

    Raises:
        ValueError: If the flag type is unexpected.
    

    中文翻译:
    根据配置的标志处理验证错误。
    参数：
        e：发生的验证错误。
        flag：如何处理错误（“bool”、“str”或“Callable”）。
    返回：
        要返回的错误消息。
    加薪：
        ValueError：如果标志类型是意外的。"""
    if isinstance(flag, bool):
        content = "Tool input validation error"
    elif isinstance(flag, str):
        content = flag
    elif callable(flag):
        content = flag(e)
    else:
        msg = (
            f"Got unexpected type of `handle_validation_error`. Expected bool, "
            f"str or callable. Received: {flag}"
        )
        raise ValueError(msg)  # noqa: TRY004
    return content


def _handle_tool_error(
    e: ToolException,
    *,
    flag: Literal[True] | str | Callable[[ToolException], str] | None,
) -> str:
    """Handle tool execution errors based on the configured flag.

    Args:
        e: The tool exception that occurred.
        flag: How to handle the error (`bool`, `str`, or `Callable`).

    Returns:
        The error message to return.

    Raises:
        ValueError: If the flag type is unexpected.
    

    中文翻译:
    根据配置的标志处理工具执行错误。
    参数：
        e：发生的工具异常。
        flag：如何处理错误（“bool”、“str”或“Callable”）。
    返回：
        要返回的错误消息。
    加薪：
        ValueError：如果标志类型是意外的。"""
    if isinstance(flag, bool):
        content = e.args[0] if e.args else "Tool execution error"
    elif isinstance(flag, str):
        content = flag
    elif callable(flag):
        content = flag(e)
    else:
        msg = (
            f"Got unexpected type of `handle_tool_error`. Expected bool, str "
            f"or callable. Received: {flag}"
        )
        raise ValueError(msg)  # noqa: TRY004
    return content


def _prep_run_args(
    value: str | dict | ToolCall,
    config: RunnableConfig | None,
    **kwargs: Any,
) -> tuple[str | dict, dict]:
    """Prepare arguments for tool execution.

    Args:
        value: The input value (`str`, `dict`, or `ToolCall`).
        config: The runnable configuration.
        **kwargs: Additional keyword arguments.

    Returns:
        A tuple of `(tool_input, run_kwargs)`.
    

    中文翻译:
    准备工具执行的参数。
    参数：
        value：输入值（“str”、“dict”或“ToolCall”）。
        config：可运行的配置。
        **kwargs：附加关键字参数。
    返回：
        “(tool_input, run_kwargs)”的元组。"""
    config = ensure_config(config)
    if _is_tool_call(value):
        tool_call_id: str | None = cast("ToolCall", value)["id"]
        tool_input: str | dict = cast("ToolCall", value)["args"].copy()
    else:
        tool_call_id = None
        tool_input = cast("str | dict", value)
    return (
        tool_input,
        dict(
            callbacks=config.get("callbacks"),
            tags=config.get("tags"),
            metadata=config.get("metadata"),
            run_name=config.get("run_name"),
            run_id=config.pop("run_id", None),
            config=config,
            tool_call_id=tool_call_id,
            **kwargs,
        ),
    )


def _format_output(
    content: Any,
    artifact: Any,
    tool_call_id: str | None,
    name: str,
    status: str,
) -> ToolOutputMixin | Any:
    """Format tool output as a `ToolMessage` if appropriate.

    Args:
        content: The main content of the tool output.
        artifact: Any artifact data from the tool.
        tool_call_id: The ID of the tool call.
        name: The name of the tool.
        status: The execution status.

    Returns:
        The formatted output, either as a `ToolMessage` or the original content.
    

    中文翻译:
    如果适用，将工具输出格式化为“ToolMessage”。
    参数：
        content：工具输出的主要内容。
        工件：来自工具的任何工件数据。
        tool_call_id：工具调用的 ID。
        名称：工具的名称。
        状态：执行状态。
    返回：
        格式化输出，作为“ToolMessage”或原始内容。"""
    if isinstance(content, ToolOutputMixin) or tool_call_id is None:
        return content
    if not _is_message_content_type(content):
        content = _stringify(content)
    return ToolMessage(
        content,
        artifact=artifact,
        tool_call_id=tool_call_id,
        name=name,
        status=status,
    )


def _is_message_content_type(obj: Any) -> bool:
    """Check if object is valid message content format.

    Validates content for OpenAI or Anthropic format tool messages.

    Args:
        obj: The object to check.

    Returns:
        `True` if the object is valid message content, `False` otherwise.
    

    中文翻译:
    检查对象是否是有效的消息内容格式。
    验证 OpenAI 或 Anthropic 格式工具消息的内容。
    参数：
        obj：要检查的对象。
    返回：
        如果对象是有效的消息内容，则为“True”，否则为“False”。"""
    return isinstance(obj, str) or (
        isinstance(obj, list) and all(_is_message_content_block(e) for e in obj)
    )


def _is_message_content_block(obj: Any) -> bool:
    """Check if object is a valid message content block.

    Validates content blocks for OpenAI or Anthropic format.

    Args:
        obj: The object to check.

    Returns:
        `True` if the object is a valid content block, `False` otherwise.
    

    中文翻译:
    检查对象是否是有效的消息内容块。
    验证 OpenAI 或 Anthropic 格式的内容块。
    参数：
        obj：要检查的对象。
    返回：
        如果对象是有效内容块，则为“True”，否则为“False”。"""
    if isinstance(obj, str):
        return True
    if isinstance(obj, dict):
        return obj.get("type", None) in TOOL_MESSAGE_BLOCK_TYPES
    return False


def _stringify(content: Any) -> str:
    """Convert content to string, preferring JSON format.

    Args:
        content: The content to stringify.

    Returns:
        String representation of the content.
    

    中文翻译:
    将内容转换为字符串，首选 JSON 格式。
    参数：
        content：要字符串化的内容。
    返回：
        内容的字符串表示形式。"""
    try:
        return json.dumps(content, ensure_ascii=False)
    except Exception:
        return str(content)


def _get_type_hints(func: Callable) -> dict[str, type] | None:
    """Get type hints from a function, handling partial functions.

    Args:
        func: The function to get type hints from.

    Returns:
        `dict` of type hints, or `None` if extraction fails.
    

    中文翻译:
    从函数获取类型提示，处理部分函数。
    参数：
        func：从中获取类型提示的函数。
    返回：
        类型提示的“dict”，如果提取失败则为“None”。"""
    if isinstance(func, functools.partial):
        func = func.func
    try:
        return get_type_hints(func)
    except Exception:
        return None


def _get_runnable_config_param(func: Callable) -> str | None:
    """Find the parameter name for `RunnableConfig` in a function.

    Args:
        func: The function to check.

    Returns:
        The parameter name for `RunnableConfig`, or `None` if not found.
    

    中文翻译:
    在函数中查找“RunnableConfig”的参数名称。
    参数：
        func：要检查的函数。
    返回：
        “RunnableConfig”的参数名称，如果未找到则为“None”。"""
    type_hints = _get_type_hints(func)
    if not type_hints:
        return None
    for name, type_ in type_hints.items():
        if type_ is RunnableConfig:
            return name
    return None


class InjectedToolArg:
    """Annotation for tool arguments that are injected at runtime.

    Tool arguments annotated with this class are not included in the tool
    schema sent to language models and are instead injected during execution.
    

    中文翻译:
    在运行时注入的工具参数的注释。
    用此类注释的工具参数不包含在工具中
    架构发送到语言模型，并在执行期间注入。"""


class _DirectlyInjectedToolArg:
    """Annotation for tool arguments that are injected at runtime.

    Injected via direct type annotation, rather than annotated metadata.

    For example, `ToolRuntime` is a directly injected argument.

    Note the direct annotation rather than the verbose alternative:
    `Annotated[ToolRuntime, InjectedRuntime]`

    ```python
    from langchain_core.tools import tool, ToolRuntime


    @tool
    def foo(x: int, runtime: ToolRuntime) -> str:
        # use runtime.state, runtime.context, runtime.store, etc.
        # 中文: 使用runtime.state、runtime.context、runtime.store等。
        ...
    ```
    

    中文翻译:
    在运行时注入的工具参数的注释。
    通过直接类型注释注入，而不是带注释的元数据。
    例如，“ToolRuntime”是直接注入的参数。
    请注意直接注释而不是冗长的替代方案：
    `带注释的[ToolRuntime，InjectedRuntime]`
    ````蟒蛇
    从langchain_core.tools导入工具，ToolRuntime
    @工具
    def foo(x: int, 运行时: ToolRuntime) -> str:
        # 使用runtime.state、runtime.context、runtime.store等。
        ...
    ````"""


class InjectedToolCallId(InjectedToolArg):
    """Annotation for injecting the tool call ID.

    This annotation is used to mark a tool parameter that should receive
    the tool call ID at runtime.

    ```python
    from typing import Annotated
    from langchain_core.messages import ToolMessage
    from langchain_core.tools import tool, InjectedToolCallId

    @tool
    def foo(
        x: int, tool_call_id: Annotated[str, InjectedToolCallId]
    ) -> ToolMessage:
        \"\"\"Return x.\"\"\"
        return ToolMessage(
            str(x),
            artifact=x,
            name="foo",
            tool_call_id=tool_call_id
        )

    ```
    

    中文翻译:
    用于注入工具调用 ID 的注释。
    该注释用于标记应接收的工具参数
    运行时的工具调用 ID。
    ````蟒蛇
    从输入导入注释
    从 langchain_core.messages 导入 ToolMessage
    从langchain_core.tools导入工具，InjectedToolCallId
    @工具
    def foo(
        x：int，tool_call_id：注释[str，InjectedToolCallId]
    ) -> 工具消息:
        \"\"\"返回 x。\"\"\"
        返回工具消息(
            str(x),
            工件=x，
            名称=“富”，
            tool_call_id=tool_call_id
        ）
    ````"""


def _is_directly_injected_arg_type(type_: Any) -> bool:
    """Check if a type annotation indicates a directly injected argument.

    This is currently only used for `ToolRuntime`.
    Checks if either the annotation itself is a subclass of `_DirectlyInjectedToolArg`
    or the origin of the annotation is a subclass of `_DirectlyInjectedToolArg`.

    Ex: `ToolRuntime` or `ToolRuntime[ContextT, StateT]` would both return `True`.
    

    中文翻译:
    检查类型注释是否指示直接注入的参数。
    目前仅用于“ToolRuntime”。
    检查注释本身是否是“_DirectlyInjectedToolArg”的子类
    或者注释的来源是`_DirectlyInjectedToolArg`的子类。
    例如：“ToolRuntime”或“ToolRuntime[ContextT，StateT]”都会返回“True”。"""
    return (
        isinstance(type_, type) and issubclass(type_, _DirectlyInjectedToolArg)
    ) or (
        (origin := get_origin(type_)) is not None
        and isinstance(origin, type)
        and issubclass(origin, _DirectlyInjectedToolArg)
    )


def _is_injected_arg_type(
    type_: type | TypeVar, injected_type: type[InjectedToolArg] | None = None
) -> bool:
    """Check if a type annotation indicates an injected argument.

    Args:
        type_: The type annotation to check.
        injected_type: The specific injected type to check for.

    Returns:
        `True` if the type is an injected argument, `False` otherwise.
    

    中文翻译:
    检查类型注释是否指示注入的参数。
    参数：
        type_：要检查的类型注释。
        Injected_type：要检查的特定注入类型。
    返回：
        如果类型是注入参数，则为“True”，否则为“False”。"""
    if injected_type is None:
        # if no injected type is specified,
        # 中文: 如果没有指定注入类型，
        # check if the type is a directly injected argument
        # 中文: 检查类型是否是直接注入的参数
        if _is_directly_injected_arg_type(type_):
            return True
        injected_type = InjectedToolArg

    # if the type is an Annotated type, check if annotated metadata
    # 中文: 如果类型是带注释的类型，请检查是否带注释的元数据
    # is an intance or subclass of the injected type
    # 中文: 是注入类型的实例或子类
    return any(
        isinstance(arg, injected_type)
        or (isinstance(arg, type) and issubclass(arg, injected_type))
        for arg in get_args(type_)[1:]
    )


def get_all_basemodel_annotations(
    cls: TypeBaseModel | Any, *, default_to_bound: bool = True
) -> dict[str, type | TypeVar]:
    """Get all annotations from a Pydantic `BaseModel` and its parents.

    Args:
        cls: The Pydantic `BaseModel` class.
        default_to_bound: Whether to default to the bound of a `TypeVar` if it exists.

    Returns:
        `dict` of field names to their type annotations.
    

    中文翻译:
    从 Pydantic `BaseModel` 及其父级获取所有注释。
    参数：
        cls：Pydantic `BaseModel` 类。
        default_to_bound：是否默认为“TypeVar”的边界（如果存在）。
    返回：
        字段名称的“dict”到其类型注释。"""
    # cls has no subscript: cls = FooBar
    # 中文: cls 没有下标：cls = FooBar
    if isinstance(cls, type):
        fields = get_fields(cls)
        alias_map = {field.alias: name for name, field in fields.items() if field.alias}

        annotations: dict[str, type | TypeVar] = {}
        for name, param in inspect.signature(cls).parameters.items():
            # Exclude hidden init args added by pydantic Config. For example if
            # 中文: 排除 pydantic Config 添加的隐藏初始化参数。例如如果
            # BaseModel(extra="allow") then "extra_data" will part of init sig.
            # 中文: BaseModel(extra="allow") 那么“extra_data”将是 init sig 的一部分。
            if fields and name not in fields and name not in alias_map:
                continue
            field_name = alias_map.get(name, name)
            annotations[field_name] = param.annotation
        orig_bases: tuple = getattr(cls, "__orig_bases__", ())
    # cls has subscript: cls = FooBar[int]
    # 中文: cls 有下标： cls = FooBar[int]
    else:
        annotations = get_all_basemodel_annotations(
            get_origin(cls), default_to_bound=False
        )
        orig_bases = (cls,)

    # Pydantic v2 automatically resolves inherited generics, Pydantic v1 does not.
    # 中文: Pydantic v2 自动解析继承的泛型，Pydantic v1 则不会。
    if not (isinstance(cls, type) and is_pydantic_v2_subclass(cls)):
        # if cls = FooBar inherits from Baz[str], orig_bases will contain Baz[str]
        # 中文: 如果 cls = FooBar 继承自 Baz[str]，则 orig_bases 将包含 Baz[str]
        # if cls = FooBar inherits from Baz, orig_bases will contain Baz
        # 中文: 如果 cls = FooBar 继承自 Baz，则 orig_bases 将包含 Baz
        # if cls = FooBar[int], orig_bases will contain FooBar[int]
        # 中文: 如果 cls = FooBar[int]，orig_bases 将包含 FooBar[int]
        for parent in orig_bases:
            # if class = FooBar inherits from Baz, parent = Baz
            # 中文: 如果 class = FooBar 继承自 Baz，parent = Baz
            if isinstance(parent, type) and is_pydantic_v1_subclass(parent):
                annotations.update(
                    get_all_basemodel_annotations(parent, default_to_bound=False)
                )
                continue

            parent_origin = get_origin(parent)

            # if class = FooBar inherits from non-pydantic class
            # 中文: if class = FooBar 继承自非 pydantic 类
            if not parent_origin:
                continue

            # if class = FooBar inherits from Baz[str]:
            # 中文: 如果 class = FooBar 继承自 Baz[str]：
            # parent = class Baz[str],
            # 中文: 父 = 类 Baz[str],
            # parent_origin = class Baz,
            # 中文: Parent_origin = 类 Baz,
            # generic_type_vars = (type vars in Baz)
            # 中文: generic_type_vars =（Baz 中的类型变量）
            # generic_map = {type var in Baz: str}
            # 中文: generic_map = {Baz 中的类型 var: str}
            generic_type_vars: tuple = getattr(parent_origin, "__parameters__", ())
            generic_map = dict(zip(generic_type_vars, get_args(parent), strict=False))
            for field in getattr(parent_origin, "__annotations__", {}):
                annotations[field] = _replace_type_vars(
                    annotations[field], generic_map, default_to_bound=default_to_bound
                )

    return {
        k: _replace_type_vars(v, default_to_bound=default_to_bound)
        for k, v in annotations.items()
    }


def _replace_type_vars(
    type_: type | TypeVar,
    generic_map: dict[TypeVar, type] | None = None,
    *,
    default_to_bound: bool = True,
) -> type | TypeVar:
    """Replace `TypeVar`s in a type annotation with concrete types.

    Args:
        type_: The type annotation to process.
        generic_map: Mapping of `TypeVar`s to concrete types.
        default_to_bound: Whether to use `TypeVar` bounds as defaults.

    Returns:
        The type with `TypeVar`s replaced.
    

    中文翻译:
    将类型注释中的“TypeVar”替换为具体类型。
    参数：
        type_：要处理的类型注释。
        generic_map：将 `TypeVar` 映射到具体类型。
        default_to_bound：是否使用“TypeVar”边界作为默认值。
    返回：
        替换为“TypeVar”的类型。"""
    generic_map = generic_map or {}
    if isinstance(type_, TypeVar):
        if type_ in generic_map:
            return generic_map[type_]
        if default_to_bound:
            return type_.__bound__ if type_.__bound__ is not None else Any
        return type_
    if (origin := get_origin(type_)) and (args := get_args(type_)):
        new_args = tuple(
            _replace_type_vars(arg, generic_map, default_to_bound=default_to_bound)
            for arg in args
        )
        return cast("type", _py_38_safe_origin(origin)[new_args])  # type: ignore[index]
    return type_


class BaseToolkit(BaseModel, ABC):
    """Base class for toolkits containing related tools.

    A toolkit is a collection of related tools that can be used together
    to accomplish a specific task or work with a particular system.
    

    中文翻译:
    包含相关工具的工具包的基类。
    工具包是可以一起使用的相关工具的集合
    完成特定任务或使用特定系统。"""

    @abstractmethod
    def get_tools(self) -> list[BaseTool]:
        """Get all tools in the toolkit.

        Returns:
            List of tools contained in this toolkit.
        

        中文翻译:
        获取工具包中的所有工具。
        返回：
            该工具包中包含的工具列表。"""
