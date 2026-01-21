"""提示模板的基类模块。

本模块定义了 LangChain 中所有提示模板的基础抽象类 `BasePromptTemplate`。
提示模板是 LangChain 框架的核心组件之一，用于动态生成发送给语言模型的提示。

核心概念:
---------
1. 提示模板 (Prompt Template): 包含占位符变量的模板字符串，可以在运行时填充
2. 输入变量 (Input Variables): 需要用户在调用时提供的变量
3. 部分变量 (Partial Variables): 预先填充的变量，不需要每次调用时都提供
4. PromptValue: 格式化后的提示值，可以是字符串或聊天消息列表

继承关系:
---------
BasePromptTemplate
├── StringPromptTemplate  (字符串类型的提示模板)
│   ├── PromptTemplate
│   ├── FewShotPromptTemplate
│   └── ...
├── BaseChatPromptTemplate  (聊天类型的提示模板)
│   ├── ChatPromptTemplate
│   └── ...
└── ImagePromptTemplate  (图像类型的提示模板)

使用示例:
---------
>>> from langchain_core.prompts import PromptTemplate
>>> # 创建一个简单的提示模板
>>> prompt = PromptTemplate.from_template("告诉我关于{topic}的信息")
>>> # 格式化提示
>>> result = prompt.format(topic="人工智能")
>>> print(result)  # 输出: "告诉我关于人工智能的信息"

>>> # 使用 invoke 方法（LCEL 接口）
>>> prompt_value = prompt.invoke({"topic": "人工智能"})
>>> print(prompt_value.to_string())
"""

from __future__ import annotations

import contextlib
import json
import typing
from abc import ABC, abstractmethod
from collections.abc import Mapping  # noqa: TC003
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self, override

from langchain_core.exceptions import ErrorCode, create_message
from langchain_core.load import dumpd
from langchain_core.output_parsers.base import BaseOutputParser  # noqa: TC001
from langchain_core.prompt_values import (
    ChatPromptValueConcrete,
    PromptValue,
    StringPromptValue,
)
from langchain_core.runnables import RunnableConfig, RunnableSerializable
from langchain_core.runnables.config import ensure_config
from langchain_core.utils.pydantic import create_model_v2

if TYPE_CHECKING:
    from collections.abc import Callable

    from langchain_core.documents import Document


# 定义格式化输出的类型变量，用于泛型支持
# 不同的模板子类可能返回不同类型：字符串、字典等
FormatOutputType = TypeVar("FormatOutputType")


class BasePromptTemplate(
    RunnableSerializable[dict, PromptValue], ABC, Generic[FormatOutputType]
):
    """所有提示模板的基类，返回一个提示。

    这是 LangChain 中所有提示模板的抽象基类。它继承自 `RunnableSerializable`，
    这意味着它可以作为 LCEL（LangChain Expression Language）管道的一部分使用。

    关键特性:
    ---------
    1. **Runnable 接口**: 支持 `invoke()`, `ainvoke()`, `batch()` 等方法
    2. **序列化**: 可以保存到文件和从文件加载
    3. **变量验证**: 自动验证输入变量的完整性
    4. **部分填充**: 支持预先填充部分变量

    LCEL 管道示例:
    -------------
    ```python
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    # 创建一个 LCEL 管道
    prompt = ChatPromptTemplate.from_template("翻译成英文: {text}")
    llm = ChatOpenAI()
    chain = prompt | llm  # 使用管道操作符连接

    # 调用管道
    result = chain.invoke({"text": "你好世界"})
    ```

    类型参数:
    --------
    FormatOutputType: format() 方法返回的类型
        - 对于 StringPromptTemplate: str
        - 对于 ChatPromptTemplate: list[BaseMessage]
    """

    # ==================== 核心配置字段 ====================

    input_variables: list[str]
    """必需输入变量的名称列表。

    这些变量必须在调用 `format()` 或 `invoke()` 时提供值。
    如果缺少任何一个，将抛出 KeyError。

    示例:
        template = "你好，{name}！今天是{day}。"
        input_variables = ["name", "day"]  # 必须提供这两个变量
    """

    optional_variables: list[str] = Field(default=[])
    """可选变量的名称列表，用于占位符或 `MessagesPlaceholder`。

    这些变量会从提示模板中自动推断出来，用户不必提供它们。
    如果不提供，这些变量会使用默认值（通常是空列表或 None）。

    典型用例:
        - MessagesPlaceholder 中的历史消息
        - 可选的上下文信息
    """

    input_types: typing.Dict[str, Any] = Field(default_factory=dict, exclude=True)  # noqa: UP006
    """提示模板期望的变量类型字典。

    如果未提供，所有变量都假定为字符串类型。
    这用于生成 Pydantic 输入模式，帮助进行类型验证。

    示例:
        input_types = {
            "name": str,
            "age": int,
            "messages": list[BaseMessage]
        }
    """

    output_parser: BaseOutputParser | None = None
    """输出解析器，用于解析在此格式化提示上调用 LLM 的输出。

    虽然可以在提示模板中设置，但通常在链中单独配置更常见。
    这个字段主要用于某些旧版 API 的兼容性。
    """

    partial_variables: Mapping[str, Any] = Field(default_factory=dict)
    """提示模板携带的部分变量字典。

    部分变量会预先填充到模板中，这样你就不需要在每次调用提示时都传入它们。
    这对于需要在多次调用之间保持不变的值非常有用。

    示例:
        ```python
        # 创建一个带有部分变量的模板
        prompt = PromptTemplate(
            template="系统时间: {time}\n用户: {user_input}",
            input_variables=["user_input"],  # 只需要用户输入
            partial_variables={"time": "2024-01-01"}  # time 已预填充
        )

        # 或者使用 partial() 方法动态创建
        prompt2 = prompt.partial(time=lambda: datetime.now().isoformat())
        ```

    注意:
        部分变量可以是普通值，也可以是无参数的可调用对象（函数）。
        如果是可调用对象，每次格式化时都会调用它获取最新值。
    """

    metadata: typing.Dict[str, Any] | None = None  # noqa: UP006
    """用于追踪的元数据。

    这些元数据会被添加到 LangSmith 追踪中，帮助调试和监控。

    示例:
        metadata = {
            "version": "1.0",
            "author": "team_a",
            "purpose": "customer_support"
        }
    """

    tags: list[str] | None = None
    """用于追踪的标签列表。

    标签会被添加到 LangSmith 追踪中，可用于过滤和搜索追踪记录。

    示例:
        tags = ["production", "v2", "customer-facing"]
    """

    # ==================== 验证器 ====================

    @model_validator(mode="after")
    def validate_variable_names(self) -> Self:
        """验证变量名称不包含受限名称。

        这个验证器在模型初始化后自动运行，确保:
        1. 'stop' 不能作为输入变量名（因为它在内部用于控制生成）
        2. 'stop' 不能作为部分变量名
        3. 输入变量和部分变量之间没有重叠

        Raises:
            ValueError: 如果包含受限变量名或变量名重叠
        """
        # 检查 'stop' 是否被用作输入变量名
        # 'stop' 是 LLM 调用中的保留参数，用于指定停止序列
        if "stop" in self.input_variables:
            msg = (
                "不能使用 'stop' 作为输入变量名，因为它在内部被使用，"
                "请重命名。"
            )
            raise ValueError(
                create_message(message=msg, error_code=ErrorCode.INVALID_PROMPT_INPUT)
            )

        # 检查 'stop' 是否被用作部分变量名
        if "stop" in self.partial_variables:
            msg = (
                "不能使用 'stop' 作为部分变量名，因为它在内部被使用，"
                "请重命名。"
            )
            raise ValueError(
                create_message(message=msg, error_code=ErrorCode.INVALID_PROMPT_INPUT)
            )

        # 检查输入变量和部分变量是否有重叠
        # 如果一个变量同时出现在两个地方，会导致歧义
        overall = set(self.input_variables).intersection(self.partial_variables)
        if overall:
            msg = f"发现输入变量和部分变量重叠: {overall}"
            raise ValueError(
                create_message(message=msg, error_code=ErrorCode.INVALID_PROMPT_INPUT)
            )
        return self

    # ==================== 序列化相关方法 ====================

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """获取 LangChain 对象的命名空间。

        命名空间用于序列化和反序列化时识别类的位置。

        Returns:
            `["langchain", "schema", "prompt_template"]`
        """
        return ["langchain", "schema", "prompt_template"]

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """返回 `True`，表示此类是可序列化的。

        这意味着可以使用 `dumpd()` 和 `load()` 函数进行序列化和反序列化。
        """
        return True

    # Pydantic 模型配置
    model_config = ConfigDict(
        arbitrary_types_allowed=True,  # 允许任意类型，因为我们可能有复杂的自定义类型
    )

    @cached_property
    def _serialized(self) -> dict[str, Any]:
        """缓存的序列化表示。

        使用 cached_property 确保只序列化一次，提高性能。
        在 invoke() 调用中使用，用于追踪。

        内部说明:
            self 在这种情况下总是一个 Serializable 对象，因此结果
            保证是一个字典，因为 dumpd 使用默认回调，它使用
            obj.to_json，该方法总是返回 TypedDict 子类。
        """
        return cast("dict[str, Any]", dumpd(self))

    # ==================== Runnable 接口实现 ====================

    @property
    @override
    def OutputType(self) -> Any:
        """返回提示的输出类型。

        提示模板的输出可以是:
        - StringPromptValue: 简单字符串提示
        - ChatPromptValueConcrete: 聊天消息列表提示
        """
        return StringPromptValue | ChatPromptValueConcrete

    @override
    def get_input_schema(self, config: RunnableConfig | None = None) -> type[BaseModel]:
        """获取提示的输入模式。

        动态生成一个 Pydantic 模型来描述这个提示模板需要的输入。
        这对于自动生成 API 文档、验证输入等非常有用。

        Args:
            config: 提示的配置（目前未使用，保留用于未来扩展）。

        Returns:
            描述提示输入的 Pydantic BaseModel 类。

        示例:
            ```python
            prompt = PromptTemplate.from_template("你好 {name}，今年 {age} 岁")
            schema = prompt.get_input_schema()
            # schema 类似于:
            # class PromptInput(BaseModel):
            #     name: str
            #     age: str
            ```
        """
        # 为必需变量创建字段定义
        # 元组形式: (类型, 默认值)，... 表示必需字段
        required_input_variables = {
            k: (self.input_types.get(k, str), ...) for k in self.input_variables
        }
        # 为可选变量创建字段定义
        # None 作为默认值表示可选
        optional_input_variables = {
            k: (self.input_types.get(k, str), None) for k in self.optional_variables
        }
        return create_model_v2(
            "PromptInput",
            field_definitions={**required_input_variables, **optional_input_variables},
        )

    def _validate_input(self, inner_input: Any) -> dict:
        """验证输入并转换为字典格式。

        这是一个内部方法，用于在格式化之前验证用户输入。

        验证规则:
        1. 如果输入不是字典且只有一个输入变量，自动包装为字典
        2. 如果输入不是字典且有多个输入变量，抛出 TypeError
        3. 检查是否缺少必需的输入变量

        Args:
            inner_input: 用户提供的输入。

        Returns:
            验证后的输入字典。

        Raises:
            TypeError: 如果输入类型不正确。
            KeyError: 如果缺少必需的输入变量。

        示例:
            ```python
            # 单变量模板支持直接传值
            prompt = PromptTemplate.from_template("{name}")
            prompt.invoke("Alice")  # 自动转换为 {"name": "Alice"}

            # 多变量模板必须传字典
            prompt = PromptTemplate.from_template("{name} is {age}")
            prompt.invoke({"name": "Alice", "age": "20"})
            ```
        """
        if not isinstance(inner_input, dict):
            # 如果只有一个输入变量，允许直接传值而不是字典
            if len(self.input_variables) == 1:
                var_name = self.input_variables[0]
                inner_input_ = {var_name: inner_input}

            else:
                # 多个变量时必须传字典
                msg = (
                    f"期望 {self.__class__.__name__} 的输入类型为映射类型（字典）。"
                    f"收到的类型为 {type(inner_input)}。"
                )
                raise TypeError(
                    create_message(
                        message=msg, error_code=ErrorCode.INVALID_PROMPT_INPUT
                    )
                )
        else:
            inner_input_ = inner_input

        # 检查是否缺少必需的变量
        missing = set(self.input_variables).difference(inner_input_)
        if missing:
            msg = (
                f"{self.__class__.__name__} 的输入缺少变量 {missing}。"
                f"期望的变量: {self.input_variables}"
                f"收到的变量: {list(inner_input_.keys())}"
            )
            # 提供有用的提示：如果用户想要字面量的花括号，需要转义
            example_key = missing.pop()
            msg += (
                f"\n提示: 如果你想让 {{{example_key}}} 成为字符串的一部分"
                "而不是变量，请使用双花括号进行转义，如: "
                f"'{{{{{example_key}}}}}'。"
            )
            raise KeyError(
                create_message(message=msg, error_code=ErrorCode.INVALID_PROMPT_INPUT)
            )
        return inner_input_

    def _format_prompt_with_error_handling(self, inner_input: dict) -> PromptValue:
        """带错误处理的提示格式化方法。

        这是 invoke() 内部使用的方法，先验证输入再格式化。

        Args:
            inner_input: 输入字典。

        Returns:
            格式化后的 PromptValue。
        """
        inner_input_ = self._validate_input(inner_input)
        return self.format_prompt(**inner_input_)

    async def _aformat_prompt_with_error_handling(
        self, inner_input: dict
    ) -> PromptValue:
        """异步版本的带错误处理的提示格式化方法。

        Args:
            inner_input: 输入字典。

        Returns:
            格式化后的 PromptValue。
        """
        inner_input_ = self._validate_input(inner_input)
        return await self.aformat_prompt(**inner_input_)

    @override
    def invoke(
        self, input: dict, config: RunnableConfig | None = None, **kwargs: Any
    ) -> PromptValue:
        """调用提示模板。

        这是 Runnable 接口的核心方法，用于执行提示格式化。
        它会自动进行输入验证、添加追踪元数据，并返回格式化结果。

        Args:
            input: 提示的输入字典。
            config: 可选的运行配置，包含回调、标签等。
            **kwargs: 额外的关键字参数（目前未使用）。

        Returns:
            格式化后的 PromptValue 对象。

        使用示例:
            ```python
            prompt = PromptTemplate.from_template("你好，{name}！")

            # 基本调用
            result = prompt.invoke({"name": "世界"})
            print(result.to_string())  # "你好，世界！"

            # 在 LCEL 管道中使用
            chain = prompt | llm | output_parser
            chain.invoke({"name": "世界"})
            ```
        """
        # 确保配置对象存在
        config = ensure_config(config)

        # 合并元数据（如果有）
        if self.metadata:
            config["metadata"] = {**config["metadata"], **self.metadata}

        # 添加标签（如果有）
        if self.tags:
            config["tags"] += self.tags

        # 使用基类的 _call_with_config 方法执行，这会自动处理追踪
        return self._call_with_config(
            self._format_prompt_with_error_handling,
            input,
            config,
            run_type="prompt",  # 标识这是一个提示类型的运行
            serialized=self._serialized,  # 用于追踪的序列化表示
        )

    @override
    async def ainvoke(
        self, input: dict, config: RunnableConfig | None = None, **kwargs: Any
    ) -> PromptValue:
        """异步调用提示模板。

        这是 invoke() 的异步版本，用于在异步上下文中执行。

        Args:
            input: 提示的输入字典。
            config: 可选的运行配置。
            **kwargs: 额外的关键字参数。

        Returns:
            格式化后的 PromptValue 对象。

        使用示例:
            ```python
            prompt = PromptTemplate.from_template("你好，{name}！")
            result = await prompt.ainvoke({"name": "世界"})
            ```
        """
        config = ensure_config(config)
        if self.metadata:
            config["metadata"].update(self.metadata)
        if self.tags:
            config["tags"].extend(self.tags)
        return await self._acall_with_config(
            self._aformat_prompt_with_error_handling,
            input,
            config,
            run_type="prompt",
            serialized=self._serialized,
        )

    # ==================== 抽象方法和格式化方法 ====================

    @abstractmethod
    def format_prompt(self, **kwargs: Any) -> PromptValue:
        """创建 PromptValue 对象。

        这是一个抽象方法，必须由子类实现。不同的子类返回不同类型的 PromptValue:
        - StringPromptTemplate -> StringPromptValue
        - BaseChatPromptTemplate -> ChatPromptValue

        Args:
            **kwargs: 要传递给提示模板的参数。

        Returns:
            格式化后的 PromptValue 对象。
        """

    async def aformat_prompt(self, **kwargs: Any) -> PromptValue:
        """异步创建 PromptValue 对象。

        默认实现直接调用同步版本。子类可以覆盖此方法以提供真正的异步实现。

        Args:
            **kwargs: 要传递给提示模板的参数。

        Returns:
            格式化后的 PromptValue 对象。
        """
        return self.format_prompt(**kwargs)

    def partial(self, **kwargs: str | Callable[[], str]) -> BasePromptTemplate:
        """返回提示模板的部分填充版本。

        部分填充允许你预先设置模板中的某些变量，创建一个新的模板，
        该模板需要更少的输入变量。

        Args:
            **kwargs: 要设置的部分变量。可以是:
                - 普通值（字符串、数字等）
                - 无参可调用对象（每次格式化时调用获取值）

        Returns:
            一个新的提示模板，其中指定的变量已被部分填充。

        使用示例:
            ```python
            from datetime import datetime

            # 原始模板
            prompt = PromptTemplate.from_template(
                "日期: {date}\n问题: {question}"
            )

            # 使用静态值部分填充
            prompt1 = prompt.partial(date="2024-01-01")
            prompt1.invoke({"question": "今天天气如何？"})

            # 使用动态函数部分填充
            prompt2 = prompt.partial(date=lambda: datetime.now().strftime("%Y-%m-%d"))
            prompt2.invoke({"question": "今天天气如何？"})  # 每次调用都获取当前日期
            ```
        """
        # 复制当前模板的属性
        prompt_dict = self.__dict__.copy()
        # 从输入变量中移除已部分填充的变量
        prompt_dict["input_variables"] = list(
            set(self.input_variables).difference(kwargs)
        )
        # 合并新的部分变量
        prompt_dict["partial_variables"] = {**self.partial_variables, **kwargs}
        # 创建相同类型的新实例
        return type(self)(**prompt_dict)

    def _merge_partial_and_user_variables(self, **kwargs: Any) -> dict[str, Any]:
        """合并部分变量和用户提供的变量。

        这是内部方法，在格式化时调用。它会:
        1. 获取所有部分变量的值（如果是可调用对象则调用它）
        2. 将用户变量与部分变量合并（用户变量优先）

        Args:
            **kwargs: 用户提供的变量。

        Returns:
            合并后的变量字典。
        """
        # 获取部分变量的值，如果是可调用对象则调用
        partial_kwargs = {
            k: v if not callable(v) else v() for k, v in self.partial_variables.items()
        }
        # 用户变量优先级更高，会覆盖同名的部分变量
        return {**partial_kwargs, **kwargs}

    @abstractmethod
    def format(self, **kwargs: Any) -> FormatOutputType:
        """用输入变量格式化提示。

        这是另一个抽象方法，必须由子类实现。
        与 format_prompt() 不同，此方法返回原始格式化结果，而不是 PromptValue 包装。

        Args:
            **kwargs: 要传递给提示模板的参数。

        Returns:
            格式化后的结果（通常是字符串）。

        使用示例:
            ```python
            prompt = PromptTemplate.from_template("你好，{name}！")
            result = prompt.format(name="世界")
            print(result)  # "你好，世界！"
            ```
        """

    async def aformat(self, **kwargs: Any) -> FormatOutputType:
        """异步格式化提示。

        默认实现直接调用同步版本。

        Args:
            **kwargs: 要传递给提示模板的参数。

        Returns:
            格式化后的结果。

        使用示例:
            ```python
            result = await prompt.aformat(name="世界")
            ```
        """
        return self.format(**kwargs)

    # ==================== 序列化和持久化 ====================

    @property
    def _prompt_type(self) -> str:
        """返回提示类型标识键。

        这个属性用于序列化，标识提示模板的类型，以便反序列化时能正确恢复。
        子类应该覆盖此方法返回唯一的类型标识。

        常见类型:
            - "prompt": PromptTemplate
            - "few_shot": FewShotPromptTemplate
            - "chat": ChatPromptTemplate
        """
        raise NotImplementedError

    def dict(self, **kwargs: Any) -> dict:
        """返回提示的字典表示。

        这个方法用于序列化提示模板。

        Args:
            **kwargs: 传递给 model_dump 的额外参数。

        Returns:
            提示的字典表示。
        """
        prompt_dict = super().model_dump(**kwargs)
        # 尝试添加类型标识，如果子类未实现则忽略
        with contextlib.suppress(NotImplementedError):
            prompt_dict["_type"] = self._prompt_type
        return prompt_dict

    def save(self, file_path: Path | str) -> None:
        """保存提示模板到文件。

        支持保存为 JSON 或 YAML 格式。

        Args:
            file_path: 保存提示的文件路径。支持 .json、.yaml、.yml 后缀。

        Raises:
            ValueError: 如果提示包含部分变量（部分变量不可序列化）。
            ValueError: 如果文件路径不是 json 或 yaml 格式。
            NotImplementedError: 如果提示类型不支持保存。

        使用示例:
            ```python
            prompt = PromptTemplate.from_template("你好，{name}！")

            # 保存为 YAML
            prompt.save("my_prompt.yaml")

            # 保存为 JSON
            prompt.save("my_prompt.json")

            # 之后可以使用 load_prompt 加载
            from langchain_core.prompts import load_prompt
            loaded_prompt = load_prompt("my_prompt.yaml")
            ```
        """
        # 检查是否有部分变量
        # 部分变量可能包含函数或复杂对象，无法序列化
        if self.partial_variables:
            msg = "无法保存包含部分变量的提示模板。"
            raise ValueError(msg)

        # 获取要保存的字典
        prompt_dict = self.dict()
        if "_type" not in prompt_dict:
            msg = f"提示 {self} 不支持保存。"
            raise NotImplementedError(msg)

        # 转换为 Path 对象
        save_path = Path(file_path)

        # 确保父目录存在
        directory_path = save_path.parent
        directory_path.mkdir(parents=True, exist_ok=True)

        # 根据文件后缀选择保存格式
        if save_path.suffix == ".json":
            with save_path.open("w", encoding="utf-8") as f:
                json.dump(prompt_dict, f, indent=4)
        elif save_path.suffix.endswith((".yaml", ".yml")):
            with save_path.open("w", encoding="utf-8") as f:
                yaml.dump(prompt_dict, f, default_flow_style=False)
        else:
            msg = f"{save_path} 必须是 json 或 yaml 格式"
            raise ValueError(msg)


# ==================== 辅助函数 ====================

def _get_document_info(doc: Document, prompt: BasePromptTemplate[str]) -> dict:
    """从文档中提取提示模板所需的信息。

    这是一个内部辅助函数，用于 format_document()。

    Args:
        doc: 要处理的文档。
        prompt: 用于格式化的提示模板。

    Returns:
        包含 page_content 和所需元数据的字典。

    Raises:
        ValueError: 如果文档缺少必需的元数据字段。
    """
    # 创建基础信息字典，包含页面内容和所有元数据
    base_info = {"page_content": doc.page_content, **doc.metadata}

    # 检查是否缺少模板所需的元数据
    missing_metadata = set(prompt.input_variables).difference(base_info)
    if len(missing_metadata) > 0:
        # 获取除 page_content 之外的必需元数据字段
        required_metadata = [
            iv for iv in prompt.input_variables if iv != "page_content"
        ]
        msg = (
            f"文档提示模板需要文档包含以下元数据变量: "
            f"{required_metadata}。收到的文档缺少以下元数据: "
            f"{list(missing_metadata)}。"
        )
        raise ValueError(
            create_message(message=msg, error_code=ErrorCode.INVALID_PROMPT_INPUT)
        )
    # 只返回模板需要的字段
    return {k: base_info[k] for k in prompt.input_variables}


def format_document(doc: Document, prompt: BasePromptTemplate[str]) -> str:
    """根据提示模板将文档格式化为字符串。

    这个函数是 RAG（检索增强生成）流程中的常用工具，用于将检索到的文档
    格式化为 LLM 可以理解的字符串格式。

    工作流程:
    ---------
    首先，从两个来源提取文档信息:

    1. `page_content`:
        从 `document.page_content` 获取信息，并将其分配给名为
        `page_content` 的变量。
    2. `metadata`:
        从 `document.metadata` 获取信息，并将其分配给同名的变量。

    然后，这些变量被传入 `prompt` 以生成格式化的字符串。

    Args:
        doc: Document 对象，其 `page_content` 和 `metadata` 将用于
            创建最终字符串。
        prompt: BasePromptTemplate 对象，用于将 `page_content` 和
            `metadata` 格式化为最终字符串。

    Returns:
        格式化后的文档字符串。

    使用示例:
        ```python
        from langchain_core.documents import Document
        from langchain_core.prompts import PromptTemplate

        # 创建一个文档
        doc = Document(
            page_content="这是一个笑话",
            metadata={"page": "1", "source": "笑话集"}
        )

        # 创建格式化模板
        prompt = PromptTemplate.from_template(
            "第 {page} 页 ({source}): {page_content}"
        )

        # 格式化文档
        result = format_document(doc, prompt)
        print(result)  # "第 1 页 (笑话集): 这是一个笑话"
        ```

    在 RAG 中的典型用法:
        ```python
        from langchain_core.prompts import ChatPromptTemplate

        # 创建文档格式化模板
        doc_prompt = PromptTemplate.from_template("{page_content}")

        # 检索文档
        docs = retriever.invoke(query)

        # 格式化所有文档
        context = "\n\n".join(format_document(doc, doc_prompt) for doc in docs)

        # 在最终提示中使用
        final_prompt = ChatPromptTemplate.from_template(
            "根据以下上下文回答问题:\n\n{context}\n\n问题: {question}"
        )
        ```
    """
    return prompt.format(**_get_document_info(doc, prompt))


async def aformat_document(doc: Document, prompt: BasePromptTemplate[str]) -> str:
    """异步版本的文档格式化函数。

    功能与 format_document() 相同，但支持异步执行。

    工作流程:
    ---------
    首先，从两个来源提取文档信息:

    1. `page_content`:
        从 `document.page_content` 获取信息，并将其分配给名为
        `page_content` 的变量。
    2. `metadata`:
        从 `document.metadata` 获取信息，并将其分配给同名的变量。

    然后，这些变量被传入 `prompt` 以生成格式化的字符串。

    Args:
        doc: Document 对象，其 `page_content` 和 `metadata` 将用于
            创建最终字符串。
        prompt: BasePromptTemplate 对象，用于将 `page_content` 和
            `metadata` 格式化为最终字符串。

    Returns:
        格式化后的文档字符串。

    使用示例:
        ```python
        result = await aformat_document(doc, prompt)
        ```
    """
    return await prompt.aformat(**_get_document_info(doc, prompt))
