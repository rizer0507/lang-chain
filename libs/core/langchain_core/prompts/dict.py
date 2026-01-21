"""字典提示模板模块。

本模块定义了 `DictPromptTemplate`，用于处理字典格式的提示模板。
这在多模态场景（如处理图像 URL）中特别有用。

核心特性:
---------
1. **递归变量识别**: 自动识别嵌套字典中的模板变量
2. **递归格式化**: 递归处理嵌套结构中的所有模板字符串
3. **只识别值中的变量**: 不识别字典键（key）中的变量，只识别值（value）

支持的格式:
---------
- f-string: 使用 {variable} 语法
- mustache: 使用 {{variable}} 语法
- 注意: 不支持 jinja2（出于安全考虑）

使用场景:
---------
DictPromptTemplate 主要用于:
1. 多模态消息的内容部分（如包含文本和图像的消息）
2. 需要保持字典结构的场景
3. API 请求体的模板化

使用示例:
---------
>>> from langchain_core.prompts.dict import DictPromptTemplate
>>>
>>> template = DictPromptTemplate(
...     template={
...         "type": "text",
...         "text": "你好，{name}！",
...         "metadata": {
...             "author": "{author}"
...         }
...     },
...     template_format="f-string"
... )
>>>
>>> result = template.format(name="世界", author="小明")
>>> # result = {
>>> #     "type": "text",
>>> #     "text": "你好，世界！",
>>> #     "metadata": {"author": "小明"}
>>> # }
"""

import warnings
from functools import cached_property
from typing import Any, Literal, cast

from typing_extensions import override

from langchain_core.load import dumpd
from langchain_core.prompts.string import (
    DEFAULT_FORMATTER_MAPPING,
    get_template_variables,
)
from langchain_core.runnables import RunnableConfig, RunnableSerializable
from langchain_core.runnables.config import ensure_config


class DictPromptTemplate(RunnableSerializable[dict, dict]):
    """用字典表示的模板。

    识别 f-string 或 mustache 格式的字典值中的变量。
    **不识别**字典键（key）中的变量。递归应用于嵌套结构。

    与 PromptTemplate 的区别:
    - PromptTemplate: 输入 dict -> 输出 str
    - DictPromptTemplate: 输入 dict -> 输出 dict

    这使得 DictPromptTemplate 适合用于需要保持结构化数据的场景。

    属性:
    -----
    template : dict[str, Any]
        模板字典，其中的字符串值可以包含模板变量
    template_format : Literal["f-string", "mustache"]
        模板格式，支持 f-string 或 mustache

    使用示例:
    ---------
    >>> template = DictPromptTemplate(
    ...     template={"greeting": "你好，{name}！"},
    ...     template_format="f-string"
    ... )
    >>> template.format(name="世界")
    {"greeting": "你好，世界！"}
    >>>
    >>> # 在 LCEL 管道中使用
    >>> chain = template | some_runnable
    >>> chain.invoke({"name": "世界"})
    """

    template: dict[str, Any]
    """模板字典。

    字典中的字符串值可以包含模板变量。
    支持任意深度的嵌套。

    注意: 字典的键（key）中的变量不会被识别！
    """

    template_format: Literal["f-string", "mustache"]
    """模板格式。

    可选值:
    - "f-string": 使用 {variable} 语法
    - "mustache": 使用 {{variable}} 语法

    注意: 出于安全考虑，不支持 jinja2。
    """

    @property
    def input_variables(self) -> list[str]:
        """模板的输入变量列表。

        自动从模板字典中递归提取所有变量名。
        """
        return _get_input_variables(self.template, self.template_format)

    def format(self, **kwargs: Any) -> dict[str, Any]:
        """用输入参数格式化模板。

        递归处理模板字典中的所有字符串值。

        Args:
            **kwargs: 用于填充模板变量的参数。

        Returns:
            格式化后的字典。

        示例:
            >>> template.format(name="世界", age=18)
            {"greeting": "你好，世界！", "info": "年龄: 18"}
        """
        return _insert_input_variables(self.template, kwargs, self.template_format)

    async def aformat(self, **kwargs: Any) -> dict[str, Any]:
        """异步格式化模板。

        当前实现只是调用同步版本。

        Returns:
            格式化后的字典。
        """
        return self.format(**kwargs)

    @override
    def invoke(
        self, input: dict, config: RunnableConfig | None = None, **kwargs: Any
    ) -> dict:
        """调用字典模板。

        作为 Runnable 接口的一部分，可以在 LCEL 管道中使用。

        Args:
            input: 输入字典，包含模板变量的值。
            config: 运行配置。
            **kwargs: 额外参数。

        Returns:
            格式化后的字典。
        """
        return self._call_with_config(
            lambda x: self.format(**x),
            input,
            ensure_config(config),
            run_type="prompt",
            serialized=self._serialized,
            **kwargs,
        )

    @property
    def _prompt_type(self) -> str:
        """提示类型标识。"""
        return "dict-prompt"

    @cached_property
    def _serialized(self) -> dict[str, Any]:
        """缓存的序列化表示。

        用于追踪和日志记录。

        内部说明:
            self 在这种情况下总是一个 Serializable 对象，因此结果
            保证是一个字典，因为 dumpd 使用默认回调，它使用
            obj.to_json，该方法总是返回 TypedDict 子类。
        """
        return cast("dict[str, Any]", dumpd(self))

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """返回 True，表示此类可序列化。"""
        return True

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """获取 LangChain 对象的命名空间。

        Returns:
            `["langchain_core", "prompts", "dict"]`
        """
        return ["langchain_core", "prompts", "dict"]

    def pretty_repr(self, *, html: bool = False) -> str:
        """获取人类可读的表示。

        Args:
            html: 是否格式化为 HTML。

        Returns:
            人类可读的字符串表示。
        """
        raise NotImplementedError


# ==================== 辅助函数 ====================

def _get_input_variables(
    template: dict, template_format: Literal["f-string", "mustache"]
) -> list[str]:
    """递归提取字典模板中的所有输入变量。

    遍历字典的所有值（包括嵌套的字典和列表），
    从字符串值中提取模板变量。

    Args:
        template: 模板字典。
        template_format: 模板格式。

    Returns:
        去重后的输入变量列表。
    """
    input_variables = []

    for v in template.values():
        if isinstance(v, str):
            # 字符串值：提取模板变量
            input_variables += get_template_variables(v, template_format)
        elif isinstance(v, dict):
            # 嵌套字典：递归处理
            input_variables += _get_input_variables(v, template_format)
        elif isinstance(v, (list, tuple)):
            # 列表或元组：逐个处理元素
            for x in v:
                if isinstance(x, str):
                    input_variables += get_template_variables(x, template_format)
                elif isinstance(x, dict):
                    input_variables += _get_input_variables(x, template_format)

    # 去重并返回
    return list(set(input_variables))


def _insert_input_variables(
    template: dict[str, Any],
    inputs: dict[str, Any],
    template_format: Literal["f-string", "mustache"],
) -> dict[str, Any]:
    """递归地将输入变量插入到模板字典中。

    遍历模板字典，用提供的输入值替换模板变量。

    Args:
        template: 模板字典。
        inputs: 输入变量字典。
        template_format: 模板格式。

    Returns:
        格式化后的字典。
    """
    formatted: dict[str, Any] = {}
    # 获取对应格式的格式化器
    formatter = DEFAULT_FORMATTER_MAPPING[template_format]

    for k, v in template.items():
        if isinstance(v, str):
            # 字符串值：直接格式化
            formatted[k] = formatter(v, **inputs)
        elif isinstance(v, dict):
            # 安全检查：阻止通过文件路径指定图像
            # 这是一个安全措施，防止路径注入攻击
            if k == "image_url" and "path" in v:
                msg = (
                    "在可能接收用户输入路径的环境中，通过文件路径指定图像输入"
                    "是一个安全漏洞。出于谨慎考虑，此功能已被移除以防止"
                    "可能的滥用。"
                )
                warnings.warn(msg, stacklevel=2)
            # 递归处理嵌套字典
            formatted[k] = _insert_input_variables(v, inputs, template_format)
        elif isinstance(v, (list, tuple)):
            # 列表或元组：逐个处理元素
            formatted_v: list[str | dict[str, Any]] = []
            for x in v:
                if isinstance(x, str):
                    formatted_v.append(formatter(x, **inputs))
                elif isinstance(x, dict):
                    formatted_v.append(
                        _insert_input_variables(x, inputs, template_format)
                    )
            # 保持原始类型（list 或 tuple）
            formatted[k] = type(v)(formatted_v)
        else:
            # 其他类型：直接保留
            formatted[k] = v

    return formatted
