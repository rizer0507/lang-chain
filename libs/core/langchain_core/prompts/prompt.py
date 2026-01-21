"""PromptTemplate 类定义模块。

本模块定义了 LangChain 中最常用的提示模板类 `PromptTemplate`。
这是用于语言模型的字符串提示模板的标准实现。

核心特性:
---------
1. **自动变量推断**: 从模板字符串自动提取输入变量
2. **多种模板格式**: 支持 f-string、mustache、jinja2
3. **模板组合**: 支持使用 + 运算符组合模板
4. **部分填充**: 支持预先填充部分变量

使用示例:
---------
>>> from langchain_core.prompts import PromptTemplate
>>>
>>> # 方式1: from_template（推荐，自动推断变量）
>>> prompt = PromptTemplate.from_template("请告诉我关于{topic}的内容")
>>> prompt.format(topic="人工智能")
"请告诉我关于人工智能的内容"
>>>
>>> # 方式2: 直接构造
>>> prompt = PromptTemplate(template="你好，{name}！")
>>>
>>> # 方式3: 从文件加载
>>> prompt = PromptTemplate.from_file("prompt.txt")
>>>
>>> # 方式4: 使用 mustache 格式
>>> prompt = PromptTemplate.from_template(
...     "你好，{{name}}！",
...     template_format="mustache"
... )

安全警告:
---------
**永远不要接受来自不可信来源的 jinja2 模板！**
推荐使用 f-string 或 mustache 格式。
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, model_validator
from typing_extensions import override

from langchain_core.prompts.string import (
    DEFAULT_FORMATTER_MAPPING,
    PromptTemplateFormat,
    StringPromptTemplate,
    check_valid_template,
    get_template_variables,
    mustache_schema,
)

if TYPE_CHECKING:
    from langchain_core.runnables.config import RunnableConfig


class PromptTemplate(StringPromptTemplate):
    """用于语言模型的提示模板。

    PromptTemplate 是 LangChain 中最基础、最常用的提示模板类。
    它由一个模板字符串组成，接受用户提供的参数来生成发送给语言模型的提示。

    模板可以使用以下三种语法格式：
    - **f-string** (默认): 使用 {variable} 语法
    - **jinja2**: 使用 {{ variable }} 语法（功能强大但需注意安全）
    - **mustache**: 使用 {{variable}} 语法（无逻辑模板）

    **安全警告**:
        推荐使用 `template_format="f-string"` 而不是 `template_format="jinja2"`。
        如果必须使用 jinja2，**绝对不要**接受来自不可信来源的模板，
        因为它们可能导致任意 Python 代码执行。

        从 LangChain 0.0.329 开始，Jinja2 模板默认使用 SandboxedEnvironment 渲染。
        这种沙箱应该被视为尽力而为的安全措施，而不是安全保证。

        尽管有沙箱保护，我们仍然建议永远不要使用来自不可信来源的 jinja2 模板。

    属性:
    -----
    template : str
        提示模板字符串
    template_format : str
        模板格式，可选 'f-string'、'mustache'、'jinja2'
    input_variables : list[str]
        模板需要的输入变量列表（自动推断）
    partial_variables : dict[str, Any]
        预填充的部分变量

    使用示例:
    ---------
    >>> from langchain_core.prompts import PromptTemplate
    >>>
    >>> # 推荐方式: 使用 from_template 方法
    >>> prompt = PromptTemplate.from_template("说{foo}")
    >>> prompt.format(foo="你好")
    "说你好"
    >>>
    >>> # 直接使用构造函数
    >>> prompt = PromptTemplate(template="说{foo}")
    """

    # ==================== 序列化属性 ====================

    @property
    @override
    def lc_attributes(self) -> dict[str, Any]:
        """用于序列化的 LangChain 属性。

        这些属性会被包含在序列化输出中。
        """
        return {
            "template_format": self.template_format,
        }

    @classmethod
    @override
    def get_lc_namespace(cls) -> list[str]:
        """获取 LangChain 对象的命名空间。

        Returns:
            `["langchain", "prompts", "prompt"]`
        """
        return ["langchain", "prompts", "prompt"]

    # ==================== 核心字段 ====================

    template: str
    """提示模板字符串。

    这是模板的核心，包含占位符变量。
    变量的语法取决于 template_format:
    - f-string: {variable}
    - mustache: {{variable}}
    - jinja2: {{ variable }}

    示例:
        "请用{language}回答以下问题: {question}"
    """

    template_format: PromptTemplateFormat = "f-string"
    """提示模板的格式。

    可选值:
    - 'f-string' (默认): Python f-string 语法，最安全
    - 'mustache': Mustache 模板语法，支持简单逻辑
    - 'jinja2': Jinja2 模板语法，功能强大但需注意安全
    """

    validate_template: bool = False
    """是否验证模板。

    如果设为 True，会在初始化时检查模板语法和变量一致性。
    注意: Mustache 模板不支持验证。
    """

    # ==================== 验证器 ====================

    @model_validator(mode="before")
    @classmethod
    def pre_init_validation(cls, values: dict) -> Any:
        """验证模板和输入变量的一致性。

        这个验证器在模型初始化之前运行，主要功能:
        1. 设置默认值
        2. 如果 validate_template=True，检查模板有效性
        3. 自动从模板中提取输入变量

        Args:
            values: 初始化参数字典

        Returns:
            处理后的参数字典
        """
        if values.get("template") is None:
            # 如果没有提供 template，让 pydantic 抛出 ValidationError
            return values

        # 设置默认值
        values.setdefault("template_format", "f-string")
        values.setdefault("partial_variables", {})

        # 如果启用了模板验证
        if values.get("validate_template"):
            if values["template_format"] == "mustache":
                msg = "Mustache 模板不支持验证。"
                raise ValueError(msg)

            if "input_variables" not in values:
                msg = "启用模板验证时必须提供 input_variables。"
                raise ValueError(msg)

            # 验证模板
            all_inputs = values["input_variables"] + list(values["partial_variables"])
            check_valid_template(
                values["template"], values["template_format"], all_inputs
            )

        # 自动从模板中提取输入变量
        # 这是 PromptTemplate 的核心便利功能！
        if values["template_format"]:
            values["input_variables"] = [
                var
                for var in get_template_variables(
                    values["template"], values["template_format"]
                )
                # 排除已经有部分值的变量
                if var not in values["partial_variables"]
            ]

        return values

    # ==================== Runnable 接口 ====================

    @override
    def get_input_schema(self, config: RunnableConfig | None = None) -> type[BaseModel]:
        """获取提示的输入模式。

        对于 mustache 模板，会生成支持嵌套结构的模式。

        Args:
            config: 运行配置。

        Returns:
            描述输入结构的 Pydantic 模型类。

        示例:
            >>> prompt = PromptTemplate.from_template("{{person.name}}", template_format="mustache")
            >>> schema = prompt.get_input_schema()
            >>> # 生成支持嵌套的模式: {"person": {"name": str}}
        """
        if self.template_format != "mustache":
            return super().get_input_schema(config)

        # Mustache 模板支持嵌套结构，需要特殊处理
        return mustache_schema(self.template)

    # ==================== 运算符重载 ====================

    def __add__(self, other: Any) -> PromptTemplate:
        """重载 + 运算符，允许组合提示模板。

        两个 PromptTemplate 可以使用 + 运算符组合成一个新的模板。

        Args:
            other: 另一个 PromptTemplate 或字符串

        Returns:
            组合后的新 PromptTemplate

        Raises:
            ValueError: 如果模板格式不同或部分变量冲突
            NotImplementedError: 如果 other 不是 PromptTemplate 或 str

        示例:
            >>> prompt1 = PromptTemplate.from_template("你好，{name}！")
            >>> prompt2 = PromptTemplate.from_template("今天是{day}。")
            >>> combined = prompt1 + prompt2
            >>> combined.format(name="世界", day="星期一")
            "你好，世界！今天是星期一。"
        """
        # 与另一个 PromptTemplate 组合
        if isinstance(other, PromptTemplate):
            # 检查模板格式是否一致
            if self.template_format != other.template_format:
                msg = "无法组合不同格式的模板"
                raise ValueError(msg)

            # 合并输入变量
            input_variables = list(
                set(self.input_variables) | set(other.input_variables)
            )
            # 拼接模板字符串
            template = self.template + other.template
            # 如果任一模板不需要验证，则不验证
            validate_template = self.validate_template and other.validate_template

            # 合并部分变量（检查冲突）
            partial_variables = dict(self.partial_variables.items())
            for k, v in other.partial_variables.items():
                if k in partial_variables:
                    msg = "同一个变量不能被部分填充两次。"
                    raise ValueError(msg)
                partial_variables[k] = v

            return PromptTemplate(
                template=template,
                input_variables=input_variables,
                partial_variables=partial_variables,
                template_format=self.template_format,
                validate_template=validate_template,
            )

        # 与字符串组合
        if isinstance(other, str):
            # 将字符串转换为相同格式的 PromptTemplate
            prompt = PromptTemplate.from_template(
                other,
                template_format=self.template_format,
            )
            return self + prompt

        msg = f"+ 运算符不支持的操作数类型: {type(other)}"
        raise NotImplementedError(msg)

    # ==================== 序列化 ====================

    @property
    def _prompt_type(self) -> str:
        """返回提示类型标识。

        用于序列化时标识这是一个 PromptTemplate。
        """
        return "prompt"

    # ==================== 核心格式化方法 ====================

    def format(self, **kwargs: Any) -> str:
        """用输入参数格式化提示模板。

        这是 PromptTemplate 的核心方法，将模板和变量组合成最终的字符串。

        Args:
            **kwargs: 要填充到模板中的变量值。

        Returns:
            格式化后的提示字符串。

        示例:
            >>> prompt = PromptTemplate.from_template("你好，{name}！今年{age}岁。")
            >>> prompt.format(name="小明", age=18)
            "你好，小明！今年18岁。"
        """
        # 首先合并部分变量和用户提供的变量
        kwargs = self._merge_partial_and_user_variables(**kwargs)
        # 使用对应格式的格式化器
        return DEFAULT_FORMATTER_MAPPING[self.template_format](self.template, **kwargs)

    # ==================== 工厂方法 ====================

    @classmethod
    def from_examples(
        cls,
        examples: list[str],
        suffix: str,
        input_variables: list[str],
        example_separator: str = "\n\n",
        prefix: str = "",
        **kwargs: Any,
    ) -> PromptTemplate:
        """从示例列表创建提示模板。

        这个方法用于动态创建包含多个示例的提示模板。
        最终模板结构: prefix + examples + suffix

        Args:
            examples: 示例字符串列表。
            suffix: 放在示例列表之后的字符串，通常用于设置用户输入。
            input_variables: 最终模板需要的输入变量列表。
            example_separator: 示例之间的分隔符，默认为两个换行。
            prefix: 放在示例之前的字符串，通常包含说明。
            **kwargs: 传递给构造函数的其他参数。

        Returns:
            生成的 PromptTemplate。

        示例:
            >>> examples = [
            ...     "输入: 2+2\\n输出: 4",
            ...     "输入: 3*3\\n输出: 9",
            ... ]
            >>> prompt = PromptTemplate.from_examples(
            ...     examples=examples,
            ...     suffix="输入: {question}\\n输出:",
            ...     input_variables=["question"],
            ...     prefix="请解答数学问题:"
            ... )
        """
        # 使用分隔符连接所有部分
        template = example_separator.join([prefix, *examples, suffix])
        return cls(input_variables=input_variables, template=template, **kwargs)

    @classmethod
    def from_file(
        cls,
        template_file: str | Path,
        encoding: str | None = None,
        **kwargs: Any,
    ) -> PromptTemplate:
        """从文件加载提示模板。

        这个方法允许将提示模板保存在单独的文件中，便于管理和版本控制。

        Args:
            template_file: 包含提示模板的文件路径。
            encoding: 打开文件时使用的编码。如果未提供，使用操作系统默认值。
            **kwargs: 传递给 from_template 的其他参数。

        Returns:
            从文件加载的 PromptTemplate。

        示例:
            >>> # prompts/greeting.txt 内容: "你好，{name}！"
            >>> prompt = PromptTemplate.from_file("prompts/greeting.txt")
            >>> prompt.format(name="世界")
            "你好，世界！"
        """
        # 读取文件内容
        template = Path(template_file).read_text(encoding=encoding)
        return cls.from_template(template=template, **kwargs)

    @classmethod
    def from_template(
        cls,
        template: str,
        *,
        template_format: PromptTemplateFormat = "f-string",
        partial_variables: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> PromptTemplate:
        """从模板字符串创建提示模板（推荐方式）。

        这是创建 PromptTemplate 的推荐方式，因为它会自动推断输入变量。

        **安全警告**:
            推荐使用 `template_format="f-string"` 而不是 `template_format="jinja2"`。
            如果必须使用 jinja2，**绝对不要**接受来自不可信来源的模板。

            从 LangChain 0.0.329 开始，Jinja2 模板默认使用 SandboxedEnvironment。
            尽管有沙箱保护，我们仍然建议永远不要使用来自不可信来源的 jinja2 模板。

        Args:
            template: 模板字符串。
            template_format: 模板格式。
                - 'f-string' (默认): 使用 {variable} 语法
                - 'mustache': 使用 {{variable}} 语法
                - 'jinja2': 使用 {{ variable }} 语法
            partial_variables: 部分变量字典。这些变量会被预先填充，
                不需要在调用 format() 时提供。
                例如模板 "{variable1} {variable2}"，如果 partial_variables
                是 {"variable1": "foo"}，则最终只需要提供 variable2。
            **kwargs: 传递给构造函数的其他参数。

        Returns:
            创建的 PromptTemplate。

        示例:
            >>> # 基本用法
            >>> prompt = PromptTemplate.from_template("你好，{name}！")
            >>> prompt.input_variables
            ['name']
            >>>
            >>> # 使用部分变量
            >>> prompt = PromptTemplate.from_template(
            ...     "系统: {system}\n用户: {user}",
            ...     partial_variables={"system": "你是一个助手"}
            ... )
            >>> prompt.input_variables  # 只需要 user
            ['user']
            >>> prompt.format(user="你好")
            "系统: 你是一个助手\n用户: 你好"
        """
        # 从模板中提取所有变量
        input_variables = get_template_variables(template, template_format)
        partial_variables_ = partial_variables or {}

        # 从输入变量中排除已经有部分值的变量
        if partial_variables_:
            input_variables = [
                var for var in input_variables if var not in partial_variables_
            ]

        return cls(
            input_variables=input_variables,
            template=template,
            template_format=template_format,
            partial_variables=partial_variables_,
            **kwargs,
        )
