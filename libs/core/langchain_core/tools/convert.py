"""Convert functions and runnables to tools.

中文翻译:
将函数和可运行对象转换为工具。"""

import inspect
from collections.abc import Callable
from typing import Any, Literal, cast, get_type_hints, overload

from pydantic import BaseModel, Field, create_model

from langchain_core.callbacks import Callbacks
from langchain_core.runnables import Runnable
from langchain_core.tools.base import ArgsSchema, BaseTool
from langchain_core.tools.simple import Tool
from langchain_core.tools.structured import StructuredTool


@overload
def tool(
    *,
    description: str | None = None,
    return_direct: bool = False,
    args_schema: ArgsSchema | None = None,
    infer_schema: bool = True,
    response_format: Literal["content", "content_and_artifact"] = "content",
    parse_docstring: bool = False,
    error_on_invalid_docstring: bool = True,
    extras: dict[str, Any] | None = None,
) -> Callable[[Callable | Runnable], BaseTool]: ...


@overload
def tool(
    name_or_callable: str,
    runnable: Runnable,
    *,
    description: str | None = None,
    return_direct: bool = False,
    args_schema: ArgsSchema | None = None,
    infer_schema: bool = True,
    response_format: Literal["content", "content_and_artifact"] = "content",
    parse_docstring: bool = False,
    error_on_invalid_docstring: bool = True,
    extras: dict[str, Any] | None = None,
) -> BaseTool: ...


@overload
def tool(
    name_or_callable: Callable,
    *,
    description: str | None = None,
    return_direct: bool = False,
    args_schema: ArgsSchema | None = None,
    infer_schema: bool = True,
    response_format: Literal["content", "content_and_artifact"] = "content",
    parse_docstring: bool = False,
    error_on_invalid_docstring: bool = True,
    extras: dict[str, Any] | None = None,
) -> BaseTool: ...


@overload
def tool(
    name_or_callable: str,
    *,
    description: str | None = None,
    return_direct: bool = False,
    args_schema: ArgsSchema | None = None,
    infer_schema: bool = True,
    response_format: Literal["content", "content_and_artifact"] = "content",
    parse_docstring: bool = False,
    error_on_invalid_docstring: bool = True,
    extras: dict[str, Any] | None = None,
) -> Callable[[Callable | Runnable], BaseTool]: ...


def tool(
    name_or_callable: str | Callable | None = None,
    runnable: Runnable | None = None,
    *args: Any,
    description: str | None = None,
    return_direct: bool = False,
    args_schema: ArgsSchema | None = None,
    infer_schema: bool = True,
    response_format: Literal["content", "content_and_artifact"] = "content",
    parse_docstring: bool = False,
    error_on_invalid_docstring: bool = True,
    extras: dict[str, Any] | None = None,
) -> BaseTool | Callable[[Callable | Runnable], BaseTool]:
    """Convert Python functions and `Runnables` to LangChain tools.

    Can be used as a decorator with or without arguments to create tools from functions.

    Functions can have any signature - the tool will automatically infer input schemas
    unless disabled.

    !!! note "Requirements"
        - Functions must have type hints for proper schema inference
        - When `infer_schema=False`, functions must be `(str) -> str` and have
            docstrings
        - When using with `Runnable`, a string name must be provided

    Args:
        name_or_callable: Optional name of the tool or the `Callable` to be
            converted to a tool. Overrides the function's name.

            Must be provided as a positional argument.
        runnable: Optional `Runnable` to convert to a tool.

            Must be provided as a positional argument.
        description: Optional description for the tool.

            Precedence for the tool description value is as follows:

            - This `description` argument
                (used even if docstring and/or `args_schema` are provided)
            - Tool function docstring
                (used even if `args_schema` is provided)
            - `args_schema` description
                (used only if `description` and docstring are not provided)
        *args: Extra positional arguments. Must be empty.
        return_direct: Whether to return directly from the tool rather than continuing
            the agent loop.
        args_schema: Optional argument schema for user to specify.
        infer_schema: Whether to infer the schema of the arguments from the function's
            signature. This also makes the resultant tool accept a dictionary input to
            its `run()` function.
        response_format: The tool response format.

            If `'content'`, then the output of the tool is interpreted as the contents
            of a `ToolMessage`.

            If `'content_and_artifact'`, then the output is expected to be a two-tuple
            corresponding to the `(content, artifact)` of a `ToolMessage`.
        parse_docstring: If `infer_schema` and `parse_docstring`, will attempt to
            parse parameter descriptions from Google Style function docstrings.
        error_on_invalid_docstring: If `parse_docstring` is provided, configure
            whether to raise `ValueError` on invalid Google Style docstrings.
        extras: Optional provider-specific extra fields for the tool.

            Used to pass configuration that doesn't fit into standard tool fields.
            Chat models should process known extras when constructing model payloads.

            !!! example

                For example, Anthropic-specific fields like `cache_control`,
                `defer_loading`, or `input_examples`.

    Raises:
        ValueError: If too many positional arguments are provided (e.g. violating the
            `*args` constraint).
        ValueError: If a `Runnable` is provided without a string name. When using `tool`
            with a `Runnable`, a `str` name must be provided as the `name_or_callable`.
        ValueError: If the first argument is not a string or callable with
            a `__name__` attribute.
        ValueError: If the function does not have a docstring and description
            is not provided and `infer_schema` is `False`.
        ValueError: If `parse_docstring` is `True` and the function has an invalid
            Google-style docstring and `error_on_invalid_docstring` is True.
        ValueError: If a `Runnable` is provided that does not have an object schema.

    Returns:
        The tool.

    Examples:
        ```python
        @tool
        def search_api(query: str) -> str:
            # Searches the API for the query.
            # 中文: 在 API 中搜索查询。
            return


        @tool("search", return_direct=True)
        def search_api(query: str) -> str:
            # Searches the API for the query.
            # 中文: 在 API 中搜索查询。
            return


        @tool(response_format="content_and_artifact")
        def search_api(query: str) -> tuple[str, dict]:
            return "partial json of results", {"full": "object of results"}
        ```

        Parse Google-style docstrings:

        ```python
        @tool(parse_docstring=True)
        def foo(bar: str, baz: int) -> str:
            \"\"\"The foo.

            Args:
                bar: The bar.
                baz: The baz.
            \"\"\"
            return bar

        foo.args_schema.model_json_schema()
        ```

        ```python
        {
            "title": "foo",
            "description": "The foo.",
            "type": "object",
            "properties": {
                "bar": {
                    "title": "Bar",
                    "description": "The bar.",
                    "type": "string",
                },
                "baz": {
                    "title": "Baz",
                    "description": "The baz.",
                    "type": "integer",
                },
            },
            "required": ["bar", "baz"],
        }
        ```

        Note that parsing by default will raise `ValueError` if the docstring
        is considered invalid. A docstring is considered invalid if it contains
        arguments not in the function signature, or is unable to be parsed into
        a summary and `"Args:"` blocks. Examples below:

        ```python
        # No args section
        # 中文: 没有参数部分
        def invalid_docstring_1(bar: str, baz: int) -> str:
            \"\"\"The foo.\"\"\"
            return bar

        # Improper whitespace between summary and args section
        # 中文: 摘要和参数部分之间的空格不正确
        def invalid_docstring_2(bar: str, baz: int) -> str:
            \"\"\"The foo.
            Args:
                bar: The bar.
                baz: The baz.
            \"\"\"
            return bar

        # Documented args absent from function signature
        # 中文: 函数签名中缺少记录的参数
        def invalid_docstring_3(bar: str, baz: int) -> str:
            \"\"\"The foo.

            Args:
                banana: The bar.
                monkey: The baz.
            \"\"\"
            return bar

        ```
    

    中文翻译:
    将 Python 函数和 Runnables 转换为 LangChain 工具。
    可以用作带或不带参数的装饰器，以从函数创建工具。
    函数可以有任何签名 - 该工具将自动推断输入模式
    除非禁用。
    ！！！注意“要求”
        - 函数必须具有类型提示才能进行正确的模式推断
        - 当 `infer_schema=False` 时，函数必须是 `(str) -> str` 并且具有
            文档字符串
        - 与“Runnable”一起使用时，必须提供字符串名称
    参数：
        name_or_callable：工具或“Callable”的可选名称
            转换为工具。覆盖函数的名称。
            必须作为位置参数提供。
        runnable：可选的“Runnable”，可转换为工具。
            必须作为位置参数提供。
        描述：工具的可选描述。
            工具描述值的优先级如下：
            - 这个“描述”参数
                （即使提供了文档字符串和/或 `args_schema` 也可以使用）
            - 工具函数文档字符串
                （即使提供了“args_schema”也可以使用）
            - `args_schema` 描述
                （仅在未提供“描述”和文档字符串时使用）
        *args：额外的位置参数。必须是空的。
        return_direct：是否直接从工具返回而不是继续
            代理循环。
        args_schema：供用户指定的可选参数架构。
        infer_schema：是否从函数的参数推断模式
            签名。这也使得生成的工具接受字典输入
            它的“run()”函数。
        response_format：工具响应格式。
            如果为“内容”，则该工具的输出将被解释为内容
            “ToolMessage”的。
            如果是“content_and_artifact”，那么输出预计是一个二元组
            对应于“ToolMessage”的“(content,artifact)”。
        parse_docstring：如果 `infer_schema` 和 `parse_docstring`，将尝试
            从 Google Style 函数文档字符串中解析参数描述。
        error_on_invalid_docstring：如果提供了“parse_docstring”，请配置
            是否在无效的 Google 样式文档字符串上引发“ValueError”。
        extras：工具的可选特定于提供商的额外字段。
            用于传递不适合标准工具字段的配置。
            在构建模型有效负载时，聊天模型应该处理已知的额外内容。
            ！！！例子
                例如，人类特定的字段，如“cache_control”，
                `defer_loading` 或 `input_examples`。
    加薪：
        ValueError：如果提供了太多位置参数（例如违反了
            `*args` 约束）。
        ValueError：如果提供的“Runnable”没有字符串名称。使用“工具”时
            对于“Runnable”，必须提供“str”名称作为“name_or_callable”。
        ValueError：如果第一个参数不是字符串或可调用
            一个 `__name__` 属性。
        ValueError：如果函数没有文档字符串和描述
            未提供且“infer_schema”为“False”。
        ValueError：如果“parse_docstring”为“True”并且该函数具有无效值
            Google 风格的文档字符串和 `error_on_invalid_docstring` 为 True。
        ValueError：如果提供的“Runnable”没有对象模式。
    返回：
        工具。
    示例：
        ````蟒蛇
        @工具
        def search_api(查询: str) -> str:
            # 在 API 中搜索查询。
            返回
        @tool("搜索", return_direct=True)
        def search_api(查询: str) -> str:
            # 在 API 中搜索查询。
            返回
        @工具（response_format =“content_and_artifact”）
        def search_api(query: str) -> tuple[str, dict]:
            return "结果的部分 json", {"full": "结果的对象"}
        ````
        解析 Google 风格的文档字符串：
        ````蟒蛇
        @工具（parse_docstring = True）
        def foo(bar: str, baz: int) -> str:
            \"\"\"foo.
            参数：
                酒吧：酒吧。
                巴兹：巴兹。
            \"\"\"
            返回栏
        foo.args_schema.model_json_schema()
        ````
        ````蟒蛇
        {“标题”：“富”，
            "description": "foo。",
            “类型”：“对象”，
            “属性”：{
                “酒吧”：{
                    "title": "酒吧",
                    "description": "酒吧。",
                    “类型”：“字符串”，
                },
                “巴兹”：{
                    “标题”：“巴兹”，
                    "description": "巴兹。",
                    “类型”：“整数”，
                },
            },
            “必需”：[“酒吧”，“巴兹”]，
        }
        ````
        请注意，如果文档字符串默认解析将引发“ValueError”
        被视为无效。如果文档字符串包含以下内容，则该文档字符串被视为无效
        参数不在函数签名中，或者无法解析为
        摘要和“Args:”块。示例如下：
        ````蟒蛇
        # 没有参数部分
        def invalid_docstring_1(bar: str, baz: int) -> str:
            \"\"\"foo。\"\"\"
            返回栏
        # 摘要和参数部分之间的空格不正确
        def invalid_docstring_2(bar: str, baz: int) -> str:
            \"\"\"foo.
            参数：
                酒吧：酒吧。
                巴兹：巴兹。
            \"\"\"
            返回栏
        # 函数签名中缺少记录的参数
        def invalid_docstring_3(bar: str, baz: int) -> str:
            \"\"\"foo.
            参数：
                香蕉：酒吧。
                猴子：巴兹。
            \"\"\"
            返回栏
        ````"""  # noqa: D214, D410, D411  # We're intentionally showing bad formatting in examples

    def _create_tool_factory(
        tool_name: str,
    ) -> Callable[[Callable | Runnable], BaseTool]:
        """Create a decorator that takes a callable and returns a tool.

        Args:
            tool_name: The name that will be assigned to the tool.

        Returns:
            A function that takes a callable or Runnable and returns a tool.
        

        中文翻译:
        创建一个接受可调用对象并返回工具的装饰器。
        参数：
            tool_name：将分配给工具的名称。
        返回：
            接受可调用或可运行并返回工具的函数。"""

        def _tool_factory(dec_func: Callable | Runnable) -> BaseTool:
            tool_description = description
            if isinstance(dec_func, Runnable):
                runnable = dec_func

                if runnable.input_schema.model_json_schema().get("type") != "object":
                    msg = "Runnable must have an object schema."
                    raise ValueError(msg)

                async def ainvoke_wrapper(
                    callbacks: Callbacks | None = None, **kwargs: Any
                ) -> Any:
                    return await runnable.ainvoke(kwargs, {"callbacks": callbacks})

                def invoke_wrapper(
                    callbacks: Callbacks | None = None, **kwargs: Any
                ) -> Any:
                    return runnable.invoke(kwargs, {"callbacks": callbacks})

                coroutine = ainvoke_wrapper
                func = invoke_wrapper
                schema: ArgsSchema | None = runnable.input_schema
                tool_description = description or repr(runnable)
            elif inspect.iscoroutinefunction(dec_func):
                coroutine = dec_func
                func = None
                schema = args_schema
            else:
                coroutine = None
                func = dec_func
                schema = args_schema

            if infer_schema or args_schema is not None:
                return StructuredTool.from_function(
                    func,
                    coroutine,
                    name=tool_name,
                    description=tool_description,
                    return_direct=return_direct,
                    args_schema=schema,
                    infer_schema=infer_schema,
                    response_format=response_format,
                    parse_docstring=parse_docstring,
                    error_on_invalid_docstring=error_on_invalid_docstring,
                    extras=extras,
                )
            # If someone doesn't want a schema applied, we must treat it as
            # 中文: 如果有人不想应用模式，我们必须将其视为
            # a simple string->string function
            # 中文: 一个简单的字符串->字符串函数
            if dec_func.__doc__ is None:
                msg = (
                    "Function must have a docstring if "
                    "description not provided and infer_schema is False."
                )
                raise ValueError(msg)
            return Tool(
                name=tool_name,
                func=func,
                description=f"{tool_name} tool",
                return_direct=return_direct,
                coroutine=coroutine,
                response_format=response_format,
                extras=extras,
            )

        return _tool_factory

    if len(args) != 0:
        # Triggered if a user attempts to use positional arguments that
        # 中文: 如果用户尝试使用位置参数，则触发
        # do not exist in the function signature
        # 中文: 函数签名中不存在
        # e.g., @tool("name", runnable, "extra_arg")
        # 中文: 例如，@tool("name", runnable, "extra_arg")
        # Here, "extra_arg" is not a valid argument
        # 中文: 这里，“extra_arg”不是一个有效的参数
        msg = "Too many arguments for tool decorator. A decorator "
        raise ValueError(msg)

    if runnable is not None:
        # tool is used as a function
        # 中文: 工具被用作函数
        # for instance tool_from_runnable = tool("name", runnable)
        # 中文: 例如 tool_from_runnable = tool("name", runnable)
        if not name_or_callable:
            msg = "Runnable without name for tool constructor"
            raise ValueError(msg)
        if not isinstance(name_or_callable, str):
            msg = "Name must be a string for tool constructor"
            raise ValueError(msg)
        return _create_tool_factory(name_or_callable)(runnable)
    if name_or_callable is not None:
        if callable(name_or_callable) and hasattr(name_or_callable, "__name__"):
            # Used as a decorator without parameters
            # 中文: 用作不带参数的装饰器
            # @tool
            # 中文: @工具
            # def my_tool():
            # 中文: def my_tool():
            #    pass
            #    中文: 经过
            return _create_tool_factory(name_or_callable.__name__)(name_or_callable)
        if isinstance(name_or_callable, str):
            # Used with a new name for the tool
            # 中文: 与工具的新名称一起使用
            # @tool("search")
            # 中文: @工具（“搜索”）
            # def my_tool():
            # 中文: def my_tool():
            #    pass
            #    中文: 经过
            #
            # or
            #
            中文: ＃ 或者
            #
            # @tool("search", parse_docstring=True)
            #
            中文: # @tool("搜索", parse_docstring=True)
            # def my_tool():
            # 中文: def my_tool():
            #    pass
            #    中文: 经过
            return _create_tool_factory(name_or_callable)
        msg = (
            f"The first argument must be a string or a callable with a __name__ "
            f"for tool decorator. Got {type(name_or_callable)}"
        )
        raise ValueError(msg)

    # Tool is used as a decorator with parameters specified
    # 中文: 工具用作指定参数的装饰器
    # @tool(parse_docstring=True)
    # 中文: @工具（parse_docstring = True）
    # def my_tool():
    # 中文: def my_tool():
    #    pass
    #    中文: 经过
    def _partial(func: Callable | Runnable) -> BaseTool:
        """Partial function that takes a callable and returns a tool.

        中文翻译:
        接受可调用对象并返回工具的部分函数。"""
        name_ = func.get_name() if isinstance(func, Runnable) else func.__name__
        tool_factory = _create_tool_factory(name_)
        return tool_factory(func)

    return _partial


def _get_description_from_runnable(runnable: Runnable) -> str:
    """Generate a placeholder description of a runnable.

    中文翻译:
    生成可运行程序的占位符描述。"""
    input_schema = runnable.input_schema.model_json_schema()
    return f"Takes {input_schema}."


def _get_schema_from_runnable_and_arg_types(
    runnable: Runnable,
    name: str,
    arg_types: dict[str, type] | None = None,
) -> type[BaseModel]:
    """Infer args_schema for tool.

    中文翻译:
    推断工具的 args_schema。"""
    if arg_types is None:
        try:
            arg_types = get_type_hints(runnable.InputType)
        except TypeError as e:
            msg = (
                "Tool input must be str or dict. If dict, dict arguments must be "
                "typed. Either annotate types (e.g., with TypedDict) or pass "
                f"arg_types into `.as_tool` to specify. {e}"
            )
            raise TypeError(msg) from e
    fields = {key: (key_type, Field(...)) for key, key_type in arg_types.items()}
    return cast("type[BaseModel]", create_model(name, **fields))  # type: ignore[call-overload]


def convert_runnable_to_tool(
    runnable: Runnable,
    args_schema: type[BaseModel] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    arg_types: dict[str, type] | None = None,
) -> BaseTool:
    """Convert a Runnable into a BaseTool.

    Args:
        runnable: The runnable to convert.
        args_schema: The schema for the tool's input arguments.
        name: The name of the tool.
        description: The description of the tool.
        arg_types: The types of the arguments.

    Returns:
        The tool.
    

    中文翻译:
    将 Runnable 转换为 BaseTool。
    参数：
        runnable：要转换的runnable。
        args_schema：工具输入参数的架构。
        名称：工具的名称。
        描述：工具的描述。
        arg_types：参数的类型。
    返回：
        工具。"""
    if args_schema:
        runnable = runnable.with_types(input_type=args_schema)
    description = description or _get_description_from_runnable(runnable)
    name = name or runnable.get_name()

    schema = runnable.input_schema.model_json_schema()
    if schema.get("type") == "string":
        return Tool(
            name=name,
            func=runnable.invoke,
            coroutine=runnable.ainvoke,
            description=description,
        )

    async def ainvoke_wrapper(callbacks: Callbacks | None = None, **kwargs: Any) -> Any:
        return await runnable.ainvoke(kwargs, config={"callbacks": callbacks})

    def invoke_wrapper(callbacks: Callbacks | None = None, **kwargs: Any) -> Any:
        return runnable.invoke(kwargs, config={"callbacks": callbacks})

    if (
        arg_types is None
        and schema.get("type") == "object"
        and schema.get("properties")
    ):
        args_schema = runnable.input_schema
    else:
        args_schema = _get_schema_from_runnable_and_arg_types(
            runnable, name, arg_types=arg_types
        )

    return StructuredTool.from_function(
        name=name,
        func=invoke_wrapper,
        coroutine=ainvoke_wrapper,
        description=description,
        args_schema=args_schema,
    )
