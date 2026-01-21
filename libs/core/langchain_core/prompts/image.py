"""多模态模型的图像提示模板模块。

本模块定义了 `ImagePromptTemplate`，用于在多模态模型（如 GPT-4V、Claude 3）中
处理图像输入。

核心概念:
---------
多模态模型可以同时处理文本和图像输入。ImagePromptTemplate 用于创建
包含图像 URL 的提示，这些提示会被转换为模型可以理解的格式。

安全注意:
---------
从 LangChain 0.3.15 开始，**不再支持通过文件路径加载图像**。
这是为了防止在接受用户输入时可能发生的路径注入攻击。
请始终使用 URL（包括 data: URL）来指定图像。

使用示例:
---------
>>> from langchain_core.prompts.image import ImagePromptTemplate
>>>
>>> # 创建图像模板
>>> image_template = ImagePromptTemplate(
...     template={"url": "{image_url}"},
...     input_variables=["image_url"]
... )
>>>
>>> # 格式化
>>> result = image_template.format(image_url="https://example.com/image.jpg")
>>> # result = {"url": "https://example.com/image.jpg"}
>>>
>>> # 使用 base64 编码的图像
>>> image_template.format(
...     image_url="data:image/jpeg;base64,/9j/4AAQSkZJRg..."
... )

与 ChatPromptTemplate 结合使用:
---------
>>> from langchain_core.prompts import ChatPromptTemplate
>>> from langchain_core.messages import HumanMessage
>>>
>>> # 多模态消息模板
>>> prompt = ChatPromptTemplate.from_messages([
...     ("system", "你是一个图像分析助手"),
...     ("human", [
...         {"type": "text", "text": "请描述这张图片"},
...         {"type": "image_url", "image_url": {"url": "{image_url}"}}
...     ])
... ])
"""

from typing import Any, Literal, cast

from pydantic import Field

from langchain_core.prompt_values import ImagePromptValue, ImageURL, PromptValue
from langchain_core.prompts.base import BasePromptTemplate
from langchain_core.prompts.string import (
    DEFAULT_FORMATTER_MAPPING,
    PromptTemplateFormat,
)
from langchain_core.runnables import run_in_executor


class ImagePromptTemplate(BasePromptTemplate[ImageURL]):
    """多模态模型的图像提示模板。

    用于创建包含图像 URL 的提示。输出类型为 ImageURL，
    这是一个包含 'url' 和可选 'detail' 字段的 TypedDict。

    属性:
    -----
    template : dict
        模板字典，可以包含 'url' 和 'detail' 字段，支持模板变量
    template_format : str
        模板格式，支持 'f-string'、'mustache'、'jinja2'

    输出格式:
    --------
    格式化后返回 ImageURL 类型:
    {
        "url": str,           # 必需，图像的 URL
        "detail": str | None  # 可选，图像细节级别: 'auto', 'low', 'high'
    }

    安全限制:
    --------
    不支持 'path' 字段。所有图像必须通过 URL 指定。
    这包括:
    - HTTP/HTTPS URL: https://example.com/image.jpg
    - Data URL: data:image/jpeg;base64,/9j/4AAQSkZJRg...

    使用示例:
    ---------
    >>> template = ImagePromptTemplate(
    ...     template={"url": "{image_url}", "detail": "high"},
    ... )
    >>> template.format(image_url="https://example.com/cat.jpg")
    {"url": "https://example.com/cat.jpg", "detail": "high"}
    """

    template: dict = Field(default_factory=dict)
    """图像提示模板。

    一个字典，可以包含以下字段:
    - url: 图像的 URL（必需，可以是模板变量）
    - detail: 图像细节级别（可选）: 'auto', 'low', 'high'

    示例:
        {"url": "{image_url}"}
        {"url": "{image_url}", "detail": "high"}
    """

    template_format: PromptTemplateFormat = "f-string"
    """提示模板的格式。

    可选值:
    - 'f-string' (默认): 使用 {variable} 语法
    - 'mustache': 使用 {{variable}} 语法
    - 'jinja2': 使用 {{ variable }} 语法
    """

    def __init__(self, **kwargs: Any) -> None:
        """创建图像提示模板。

        Args:
            **kwargs: 传递给父类的参数。

        Raises:
            ValueError: 如果 input_variables 包含 'url'、'path' 或 'detail'。
                这些是保留字段名，不能作为输入变量。
        """
        if "input_variables" not in kwargs:
            kwargs["input_variables"] = []

        # 检查是否使用了保留字段名
        overlap = set(kwargs["input_variables"]) & {"url", "path", "detail"}
        if overlap:
            msg = (
                "图像模板的 input_variables 不能包含"
                "'url'、'path' 或 'detail' 中的任何一个。"
                f"发现: {overlap}"
            )
            raise ValueError(msg)
        super().__init__(**kwargs)

    @property
    def _prompt_type(self) -> str:
        """返回提示类型标识。"""
        return "image-prompt"

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """获取 LangChain 对象的命名空间。

        Returns:
            `["langchain", "prompts", "image"]`
        """
        return ["langchain", "prompts", "image"]

    def format_prompt(self, **kwargs: Any) -> PromptValue:
        """用输入参数格式化提示。

        Args:
            **kwargs: 要传递给提示模板的参数。

        Returns:
            包含图像 URL 信息的 ImagePromptValue 对象。
        """
        return ImagePromptValue(image_url=self.format(**kwargs))

    async def aformat_prompt(self, **kwargs: Any) -> PromptValue:
        """异步格式化提示。

        Args:
            **kwargs: 要传递给提示模板的参数。

        Returns:
            包含图像 URL 信息的 ImagePromptValue 对象。
        """
        return ImagePromptValue(image_url=await self.aformat(**kwargs))

    def format(
        self,
        **kwargs: Any,
    ) -> ImageURL:
        """用输入参数格式化图像提示。

        处理模板中的变量，生成最终的 ImageURL 字典。

        Args:
            **kwargs: 要传递给提示模板的参数。
                可以直接传入 'url' 和 'detail' 参数覆盖模板值。

        Returns:
            格式化后的 ImageURL 字典。

        Raises:
            ValueError: 如果没有提供 url。
            ValueError: 如果 url 不是字符串。
            ValueError: 如果尝试使用 'path' 指定图像（已移除此功能）。

        示例:
            >>> template = ImagePromptTemplate(template={"url": "{img}"})
            >>> template.format(img="https://example.com/image.jpg")
            {"url": "https://example.com/image.jpg"}
        """
        # 格式化模板中的字符串值
        formatted = {}
        for k, v in self.template.items():
            if isinstance(v, str):
                formatted[k] = DEFAULT_FORMATTER_MAPPING[self.template_format](
                    v, **kwargs
                )
            else:
                formatted[k] = v

        # 获取 url（kwargs 优先，其次是格式化后的模板值）
        url = kwargs.get("url") or formatted.get("url")

        # 安全检查：禁止使用 path
        if kwargs.get("path") or formatted.get("path"):
            msg = (
                "从 0.3.15 版本开始，出于安全原因，已移除通过 'path' 加载图像的功能。"
                "请使用 'url' 指定图像。"
            )
            raise ValueError(msg)

        # 获取 detail（可选）
        detail = kwargs.get("detail") or formatted.get("detail")

        # 验证 url
        if not url:
            msg = "必须提供 url。"
            raise ValueError(msg)
        if not isinstance(url, str):
            msg = "url 必须是字符串。"
            raise ValueError(msg)  # noqa: TRY004

        # 构建输出
        output: ImageURL = {"url": url}
        if detail:
            # 不在这里检查 detail 的具体值，让 API 来检查
            output["detail"] = cast("Literal['auto', 'low', 'high']", detail)

        return output

    async def aformat(self, **kwargs: Any) -> ImageURL:
        """异步格式化图像提示。

        Args:
            **kwargs: 要传递给提示模板的参数。

        Returns:
            格式化后的 ImageURL 字典。
        """
        return await run_in_executor(None, self.format, **kwargs)

    def pretty_repr(
        self,
        html: bool = False,  # noqa: FBT001,FBT002
    ) -> str:
        """返回提示的美化表示。

        Args:
            html: 是否返回 HTML 格式的字符串。

        Returns:
            提示的美化字符串表示。
        """
        raise NotImplementedError
