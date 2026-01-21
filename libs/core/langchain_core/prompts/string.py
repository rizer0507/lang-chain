"""字符串提示模板的基础模块。

本模块提供了字符串类型提示模板的核心功能，包括：
1. 模板格式化器（f-string、mustache、jinja2）
2. 模板变量提取
3. StringPromptTemplate 抽象基类

模板格式对比:
------------
| 格式     | 语法示例          | 安全性 | 功能丰富度 |
|----------|-------------------|--------|------------|
| f-string | {variable}        | ✅ 高  | 基础       |
| mustache | {{variable}}      | ✅ 高  | 中等       |
| jinja2   | {{ variable }}    | ⚠️ 中  | 丰富       |

安全警告:
---------
Jinja2 模板虽然使用沙箱环境，但仍存在潜在风险。
**永远不要接受来自不可信来源的 Jinja2 模板！**

使用示例:
---------
>>> from langchain_core.prompts import PromptTemplate
>>> # 使用 f-string 格式（默认，推荐）
>>> prompt = PromptTemplate.from_template("你好，{name}！")
>>> prompt.format(name="世界")
"你好，世界！"

>>> # 使用 mustache 格式
>>> prompt = PromptTemplate.from_template(
...     "你好，{{name}}！",
...     template_format="mustache"
... )
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from string import Formatter
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import BaseModel, create_model
from typing_extensions import override

from langchain_core.prompt_values import PromptValue, StringPromptValue
from langchain_core.prompts.base import BasePromptTemplate
from langchain_core.utils import get_colored_text, mustache
from langchain_core.utils.formatting import formatter
from langchain_core.utils.interactive_env import is_interactive_env

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

# ==================== Jinja2 可选依赖检测 ====================
# Jinja2 不是必需依赖，只有在使用 jinja2 格式时才需要
try:
    from jinja2 import meta
    from jinja2.sandbox import SandboxedEnvironment

    _HAS_JINJA2 = True
except ImportError:
    _HAS_JINJA2 = False

# 定义支持的模板格式类型
# Literal 类型用于类型检查，确保只能使用这三种格式
PromptTemplateFormat = Literal["f-string", "mustache", "jinja2"]


# ==================== 格式化器函数 ====================

def jinja2_formatter(template: str, /, **kwargs: Any) -> str:
    """使用 Jinja2 格式化模板字符串。

    Jinja2 是一个功能强大的模板引擎，支持：
    - 变量替换: {{ variable }}
    - 过滤器: {{ name|upper }}
    - 条件语句: {% if condition %}...{% endif %}
    - 循环: {% for item in items %}...{% endfor %}

    **安全警告**:
        从 LangChain 0.0.329 开始，此方法默认使用 Jinja2 的
        SandboxedEnvironment（沙箱环境）。然而，这种沙箱应该被视为
        "尽力而为"的安全措施，而不是安全保证。

        **不要接受来自不可信来源的 jinja2 模板，因为它们可能导致
        任意 Python 代码执行！**

        参考: https://jinja.palletsprojects.com/en/3.1.x/sandbox/

    Args:
        template: 模板字符串。
        **kwargs: 用于格式化模板的变量。

    Returns:
        格式化后的字符串。

    Raises:
        ImportError: 如果未安装 jinja2。

    示例:
        >>> jinja2_formatter("你好，{{ name }}！", name="世界")
        "你好，世界！"

        >>> # 使用过滤器
        >>> jinja2_formatter("{{ name|upper }}", name="hello")
        "HELLO"
    """
    if not _HAS_JINJA2:
        msg = (
            "未安装 jinja2，但使用 jinja2_formatter 需要它。"
            "请使用 `pip install jinja2` 安装。"
            "请注意使用 jinja2 模板时要小心。"
            "不要使用未经验证的或用户控制的输入来展开 jinja2 模板，"
            "因为这可能导致任意 Python 代码执行。"
        )
        raise ImportError(msg)

    # 使用受限的沙箱环境，阻止所有属性/方法访问
    # 只允许简单的变量查找，如 {{variable}}
    # 属性访问如 {{variable.attr}} 或 {{variable.method()}} 会被阻止
    return SandboxedEnvironment().from_string(template).render(**kwargs)


def validate_jinja2(template: str, input_variables: list[str]) -> None:
    """验证输入变量对于模板是否有效。

    如果发现缺失或多余的变量，会发出警告。
    这是一个验证函数，用于帮助开发者发现模板和变量之间的不匹配。

    Args:
        template: 模板字符串。
        input_variables: 声明的输入变量列表。

    示例:
        >>> validate_jinja2("你好，{{ name }}！", ["name"])  # 正常，无警告
        >>> validate_jinja2("你好，{{ name }}！", [])  # 警告: 缺少变量 name
    """
    input_variables_set = set(input_variables)
    # 从模板中提取实际使用的变量
    valid_variables = _get_jinja2_variables_from_template(template)

    # 检查缺失的变量（模板中有，但声明中没有）
    missing_variables = valid_variables - input_variables_set
    # 检查多余的变量（声明中有，但模板中没有）
    extra_variables = input_variables_set - valid_variables

    warning_message = ""
    if missing_variables:
        warning_message += f"缺失的变量: {missing_variables} "

    if extra_variables:
        warning_message += f"多余的变量: {extra_variables}"

    if warning_message:
        warnings.warn(warning_message.strip(), stacklevel=7)


def _get_jinja2_variables_from_template(template: str) -> set[str]:
    """从 Jinja2 模板中提取未声明的变量。

    这是一个内部辅助函数，使用 Jinja2 的 AST 解析功能
    来找出模板中使用的所有变量。

    Args:
        template: Jinja2 模板字符串。

    Returns:
        模板中使用的变量名称集合。

    Raises:
        ImportError: 如果未安装 jinja2。
    """
    if not _HAS_JINJA2:
        msg = (
            "未安装 jinja2，但使用 jinja2_formatter 需要它。"
            "请使用 `pip install jinja2` 安装。"
        )
        raise ImportError(msg)
    # 使用沙箱环境解析模板
    env = SandboxedEnvironment()
    # 将模板解析为 AST（抽象语法树）
    ast = env.parse(template)
    # 使用 meta 模块找出所有未声明的变量
    return meta.find_undeclared_variables(ast)


def mustache_formatter(template: str, /, **kwargs: Any) -> str:
    """使用 Mustache 格式化模板字符串。

    Mustache 是一种"无逻辑"模板语言，语法简单：
    - 变量: {{variable}}
    - 区块: {{#items}}...{{/items}}
    - 反向区块: {{^items}}...{{/items}}
    - 注释: {{! 这是注释 }}

    Args:
        template: 模板字符串。
        **kwargs: 用于格式化模板的变量。

    Returns:
        格式化后的字符串。

    示例:
        >>> mustache_formatter("你好，{{name}}！", name="世界")
        "你好，世界！"
    """
    return mustache.render(template, kwargs)


def mustache_template_vars(
    template: str,
) -> set[str]:
    """从 Mustache 模板中获取顶级变量。

    对于嵌套变量如 `{{person.name}}`，只返回顶级键（`person`）。
    这是因为在提供输入时，只需要提供顶级对象。

    Args:
        template: 模板字符串。

    Returns:
        模板中的顶级变量集合。

    示例:
        >>> mustache_template_vars("{{name}} - {{address.city}}")
        {'name', 'address'}  # 注意: address.city 只返回 address
    """
    variables: set[str] = set()
    section_depth = 0  # 跟踪当前区块嵌套深度

    # 遍历模板中的所有标记
    for type_, key in mustache.tokenize(template):
        if type_ == "end":
            # 区块结束，深度减一
            section_depth -= 1
        elif (
            type_ in {"variable", "section", "inverted section", "no escape"}
            and key != "."  # 忽略当前上下文引用
            and section_depth == 0  # 只收集顶级变量
        ):
            # 只取点号前的部分（顶级键）
            variables.add(key.split(".")[0])
        if type_ in {"section", "inverted section"}:
            # 进入新区块，深度加一
            section_depth += 1
    return variables


# 用于递归定义嵌套字典类型的类型别名
Defs = dict[str, "Defs"]


def mustache_schema(template: str) -> type[BaseModel]:
    """从 Mustache 模板生成 Pydantic 模型。

    分析模板结构，自动创建对应的 Pydantic 数据模型。
    这对于验证输入数据结构非常有用。

    Args:
        template: 模板字符串。

    Returns:
        描述模板变量结构的 Pydantic 模型类。

    示例:
        >>> schema = mustache_schema("{{name}} lives in {{address.city}}")
        >>> # 生成类似于:
        >>> # class Address(BaseModel):
        >>> #     city: str
        >>> # class PromptInput(BaseModel):
        >>> #     name: str
        >>> #     address: Address
    """
    fields = {}
    prefix: tuple[str, ...] = ()  # 当前路径前缀
    section_stack: list[tuple[str, ...]] = []  # 区块栈，用于跟踪嵌套

    # 遍历模板标记，构建字段结构
    for type_, key in mustache.tokenize(template):
        if key == ".":
            continue
        if type_ == "end":
            # 区块结束，恢复之前的前缀
            if section_stack:
                prefix = section_stack.pop()
        elif type_ in {"section", "inverted section"}:
            # 进入新区块
            section_stack.append(prefix)
            prefix += tuple(key.split("."))
            fields[prefix] = False  # 区块本身不是叶子节点
        elif type_ in {"variable", "no escape"}:
            # 变量是叶子节点
            fields[prefix + tuple(key.split("."))] = True

    # 标记真正的叶子节点（没有子节点的节点）
    for fkey, fval in fields.items():
        fields[fkey] = fval and not any(
            is_subsequence(fkey, k) for k in fields if k != fkey
        )

    # 构建嵌套定义结构
    defs: Defs = {}
    while fields:
        field, is_leaf = fields.popitem()
        current = defs
        for part in field[:-1]:
            current = current.setdefault(part, {})
        current.setdefault(field[-1], "" if is_leaf else {})  # type: ignore[arg-type]

    return _create_model_recursive("PromptInput", defs)


def _create_model_recursive(name: str, defs: Defs) -> type[BaseModel]:
    """递归创建嵌套的 Pydantic 模型。

    这是一个内部辅助函数，用于将嵌套的字典定义转换为
    嵌套的 Pydantic 模型类。

    Args:
        name: 模型名称。
        defs: 字段定义字典。

    Returns:
        Pydantic 模型类。
    """
    return cast(
        "type[BaseModel]",
        create_model(  # type: ignore[call-overload]
            name,
            **{
                k: (_create_model_recursive(k, v), None) if v else (type(v), None)
                for k, v in defs.items()
            },
        ),
    )


# ==================== 格式化器和验证器映射 ====================

# 默认格式化器映射
# 根据 template_format 选择对应的格式化函数
DEFAULT_FORMATTER_MAPPING: dict[str, Callable[..., str]] = {
    "f-string": formatter.format,      # Python 标准 f-string 格式
    "mustache": mustache_formatter,    # Mustache 模板格式
    "jinja2": jinja2_formatter,        # Jinja2 模板格式
}

# 默认验证器映射
# 用于验证模板和输入变量的一致性
DEFAULT_VALIDATOR_MAPPING: dict[str, Callable] = {
    "f-string": formatter.validate_input_variables,
    "jinja2": validate_jinja2,
    # 注意: mustache 没有验证器，因为它的语法更灵活
}


# ==================== 模板验证函数 ====================

def check_valid_template(
    template: str, template_format: str, input_variables: list[str]
) -> None:
    """检查模板字符串是否有效。

    此函数验证模板语法正确，并且与声明的输入变量匹配。

    Args:
        template: 模板字符串。
        template_format: 模板格式，应为 "f-string"、"mustache" 或 "jinja2"。
        input_variables: 声明的输入变量列表。

    Raises:
        ValueError: 如果模板格式不支持。
        ValueError: 如果提示模式无效（变量不匹配）。

    示例:
        >>> check_valid_template("你好，{name}！", "f-string", ["name"])  # 正常
        >>> check_valid_template("你好，{name}！", "f-string", [])  # 抛出 ValueError
    """
    try:
        validator_func = DEFAULT_VALIDATOR_MAPPING[template_format]
    except KeyError as exc:
        msg = (
            f"无效的模板格式 {template_format!r}，应为以下之一: "
            f"{list(DEFAULT_FORMATTER_MAPPING)}。"
        )
        raise ValueError(msg) from exc
    try:
        validator_func(template, input_variables)
    except (KeyError, IndexError) as exc:
        msg = (
            f"无效的提示模式；请检查输入参数 {input_variables} 是否有不匹配或缺失。"
        )
        raise ValueError(msg) from exc


def get_template_variables(template: str, template_format: str) -> list[str]:
    """从模板中提取变量名称。

    这是一个核心函数，用于自动推断模板中需要的输入变量。
    `PromptTemplate.from_template()` 内部使用此函数来自动设置 input_variables。

    Args:
        template: 模板字符串。
        template_format: 模板格式，应为 "f-string"、"mustache" 或 "jinja2"。

    Returns:
        排序后的变量名列表。

    Raises:
        ValueError: 如果模板格式不支持。
        ValueError: 如果 f-string 变量名包含非法字符。

    示例:
        >>> get_template_variables("你好，{name}！今天是{day}。", "f-string")
        ['day', 'name']  # 按字母顺序排序

        >>> get_template_variables("{{greeting}} {{name}}", "mustache")
        ['greeting', 'name']
    """
    if template_format == "jinja2":
        # Jinja2: 使用 AST 解析提取变量
        input_variables = _get_jinja2_variables_from_template(template)
    elif template_format == "f-string":
        # f-string: 使用 Python 标准库的 Formatter 解析
        input_variables = {
            v for _, v, _, _ in Formatter().parse(template) if v is not None
        }
    elif template_format == "mustache":
        # Mustache: 使用专门的解析器
        input_variables = mustache_template_vars(template)
    else:
        msg = f"不支持的模板格式: {template_format}"
        raise ValueError(msg)

    # f-string 安全检查
    # 阻止属性访问和索引语法，防止模板注入攻击
    if template_format == "f-string":
        for var in input_variables:
            # Formatter().parse() 会返回带点号或方括号的字段名
            # 例如 "obj.attr" 或 "obj[0]" - 这些需要被阻止
            if "." in var or "[" in var or "]" in var:
                msg = (
                    f"f-string 模板中的变量名 {var!r} 无效。"
                    f"变量名不能包含属性访问 (.) 或索引 ([])。"
                )
                raise ValueError(msg)

            # 阻止全数字的变量名（如 "0", "100"）
            # 这些会被解释为位置参数，而不是关键字参数
            if var.isdigit():
                msg = (
                    f"f-string 模板中的变量名 {var!r} 无效。"
                    f"变量名不能全是数字，因为它们会被解释为位置参数。"
                )
                raise ValueError(msg)

    # 返回排序后的列表，保证确定性
    return sorted(input_variables)


# ==================== StringPromptTemplate 类 ====================

class StringPromptTemplate(BasePromptTemplate, ABC):
    """暴露 format 方法的字符串提示模板，返回一个提示。

    这是所有返回字符串的提示模板的抽象基类。
    与 `BaseChatPromptTemplate`（返回消息列表）不同，
    此类的 `format()` 方法返回一个格式化后的字符串。

    继承关系:
    ---------
    StringPromptTemplate (本类)
    ├── PromptTemplate          (最常用的字符串模板)
    ├── FewShotPromptTemplate   (Few-shot 学习模板)
    └── FewShotPromptWithTemplates

    核心方法:
    ---------
    - `format(**kwargs) -> str`: 格式化为字符串（抽象方法，子类必须实现）
    - `format_prompt(**kwargs) -> StringPromptValue`: 格式化为 PromptValue
    - `pretty_repr()`: 获取人类可读的表示
    - `pretty_print()`: 打印人类可读的表示

    使用示例:
    ---------
    >>> from langchain_core.prompts import PromptTemplate
    >>>
    >>> # PromptTemplate 是 StringPromptTemplate 的主要子类
    >>> prompt = PromptTemplate.from_template("你好，{name}！")
    >>>
    >>> # format() 返回字符串
    >>> prompt.format(name="世界")
    "你好，世界！"
    >>>
    >>> # format_prompt() 返回 StringPromptValue
    >>> prompt_value = prompt.format_prompt(name="世界")
    >>> prompt_value.to_string()
    "你好，世界！"
    """

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """获取 LangChain 对象的命名空间。

        命名空间用于序列化时识别类的来源位置。

        Returns:
            `["langchain", "prompts", "base"]`
        """
        return ["langchain", "prompts", "base"]

    def format_prompt(self, **kwargs: Any) -> PromptValue:
        """用输入参数格式化提示。

        此方法调用 `format()` 获取字符串，然后包装为 `StringPromptValue`。
        这使得字符串提示可以与聊天提示有统一的接口。

        Args:
            **kwargs: 要传递给提示模板的参数。

        Returns:
            包含格式化字符串的 StringPromptValue 对象。

        示例:
            >>> prompt = PromptTemplate.from_template("你好，{name}！")
            >>> value = prompt.format_prompt(name="世界")
            >>> value.to_string()
            "你好，世界！"
            >>> value.to_messages()  # 也可以转换为消息
            [HumanMessage(content="你好，世界！")]
        """
        return StringPromptValue(text=self.format(**kwargs))

    async def aformat_prompt(self, **kwargs: Any) -> PromptValue:
        """异步格式化提示。

        Args:
            **kwargs: 要传递给提示模板的参数。

        Returns:
            包含格式化字符串的 StringPromptValue 对象。
        """
        return StringPromptValue(text=await self.aformat(**kwargs))

    @override
    @abstractmethod
    def format(self, **kwargs: Any) -> str:
        """格式化提示模板。

        这是抽象方法，子类必须实现。

        Args:
            **kwargs: 用于填充模板变量的关键字参数。

        Returns:
            格式化后的字符串。
        """
        ...

    def pretty_repr(
        self,
        html: bool = False,  # noqa: FBT001,FBT002
    ) -> str:
        """获取提示的美化表示。

        这个方法生成一个人类可读的提示表示，变量名用占位符显示。
        在交互式环境中，占位符会用颜色高亮显示。

        Args:
            html: 是否返回 HTML 格式的字符串（用于在 Jupyter 中显示）。

        Returns:
            提示的美化字符串表示。

        示例:
            >>> prompt = PromptTemplate.from_template("你好，{name}！")
            >>> print(prompt.pretty_repr())
            你好，{name}！
        """
        # TODO: 处理部分变量
        # 创建虚拟变量，用变量名本身作为值
        dummy_vars = {
            input_var: "{" + f"{input_var}" + "}" for input_var in self.input_variables
        }
        # 如果需要 HTML 格式，添加颜色
        if html:
            dummy_vars = {
                k: get_colored_text(v, "yellow") for k, v in dummy_vars.items()
            }
        return self.format(**dummy_vars)

    def pretty_print(self) -> None:
        """打印提示的美化表示。

        在交互式环境（如 Jupyter）中会使用 HTML 格式显示。
        """
        print(self.pretty_repr(html=is_interactive_env()))  # noqa: T201


# ==================== 辅助函数 ====================

def is_subsequence(child: Sequence, parent: Sequence) -> bool:
    """检查 child 是否是 parent 的前缀子序列。

    这是一个内部辅助函数，用于 mustache_schema 中判断
    一个路径是否是另一个路径的前缀。

    Args:
        child: 子序列。
        parent: 父序列。

    Returns:
        如果 child 是 parent 的前缀，返回 True。

    示例:
        >>> is_subsequence(('a', 'b'), ('a', 'b', 'c'))
        True
        >>> is_subsequence(('a', 'b'), ('a', 'c'))
        False
    """
    if len(child) == 0 or len(parent) == 0:
        return False
    if len(parent) < len(child):
        return False
    return all(child[i] == parent[i] for i in range(len(child)))
