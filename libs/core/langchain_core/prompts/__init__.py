"""LangChain prompts 模块入口。

**Prompt（提示）** 是发送给模型的输入。

提示通常由多个组件和提示值构建而成。本模块中的提示类和函数使构建和使用提示变得简单。

模块结构:
---------
本模块导出了所有提示相关的核心类和函数:

核心基类:
- BasePromptTemplate: 所有提示模板的基类
- StringPromptTemplate: 字符串提示模板基类
- BaseChatPromptTemplate: 聊天提示模板基类

常用模板类:
- PromptTemplate: 标准字符串提示模板
- ChatPromptTemplate: 聊天模型提示模板
- FewShotPromptTemplate: Few-shot 学习提示模板
- FewShotChatMessagePromptTemplate: 聊天 few-shot 模板
- DictPromptTemplate: 字典格式提示模板

消息模板类:
- HumanMessagePromptTemplate: 用户消息模板
- AIMessagePromptTemplate: AI 消息模板
- SystemMessagePromptTemplate: 系统消息模板
- ChatMessagePromptTemplate: 自定义角色消息模板
- MessagesPlaceholder: 消息列表占位符

工具函数:
- format_document: 将文档格式化为字符串
- load_prompt: 从文件加载提示模板
- get_template_variables: 从模板提取变量

快速开始:
---------
>>> from langchain_core.prompts import (
...     PromptTemplate,
...     ChatPromptTemplate,
...     MessagesPlaceholder,
... )
>>>
>>> # 创建简单的字符串提示
>>> prompt = PromptTemplate.from_template("告诉我关于{topic}的内容")
>>> prompt.format(topic="人工智能")
"告诉我关于人工智能的内容"
>>>
>>> # 创建聊天提示
>>> chat_prompt = ChatPromptTemplate.from_messages([
...     ("system", "你是一个有用的助手"),
...     ("human", "{input}"),
... ])
>>> chat_prompt.format_messages(input="你好")
[SystemMessage(content="你是一个有用的助手"), HumanMessage(content="你好")]
"""

from typing import TYPE_CHECKING

from langchain_core._import_utils import import_attr

# 类型检查时的导入（不会在运行时执行，只用于 IDE 类型提示）
if TYPE_CHECKING:
    from langchain_core.prompts.base import (
        BasePromptTemplate,
        aformat_document,
        format_document,
    )
    from langchain_core.prompts.chat import (
        AIMessagePromptTemplate,
        BaseChatPromptTemplate,
        ChatMessagePromptTemplate,
        ChatPromptTemplate,
        HumanMessagePromptTemplate,
        MessagesPlaceholder,
        SystemMessagePromptTemplate,
    )
    from langchain_core.prompts.dict import DictPromptTemplate
    from langchain_core.prompts.few_shot import (
        FewShotChatMessagePromptTemplate,
        FewShotPromptTemplate,
    )
    from langchain_core.prompts.few_shot_with_templates import (
        FewShotPromptWithTemplates,
    )
    from langchain_core.prompts.loading import load_prompt
    from langchain_core.prompts.prompt import PromptTemplate
    from langchain_core.prompts.string import (
        StringPromptTemplate,
        check_valid_template,
        get_template_variables,
        jinja2_formatter,
        validate_jinja2,
    )

# 公开导出的符号列表
__all__ = (
    # 消息模板
    "AIMessagePromptTemplate",
    "BaseChatPromptTemplate",
    "BasePromptTemplate",
    "ChatMessagePromptTemplate",
    "ChatPromptTemplate",
    # 其他模板
    "DictPromptTemplate",
    "FewShotChatMessagePromptTemplate",
    "FewShotPromptTemplate",
    "FewShotPromptWithTemplates",
    # 消息相关
    "HumanMessagePromptTemplate",
    "MessagesPlaceholder",
    "PromptTemplate",
    "StringPromptTemplate",
    "SystemMessagePromptTemplate",
    # 函数
    "aformat_document",
    "check_valid_template",
    "format_document",
    "get_template_variables",
    "jinja2_formatter",
    "load_prompt",
    "validate_jinja2",
)

# 动态导入映射
# 键是符号名称，值是模块名称
# 这种延迟加载方式可以提高启动速度，只在需要时才导入相应的模块
_dynamic_imports = {
    # 基础模块
    "BasePromptTemplate": "base",
    "format_document": "base",
    "aformat_document": "base",
    # 聊天模块
    "AIMessagePromptTemplate": "chat",
    "BaseChatPromptTemplate": "chat",
    "ChatMessagePromptTemplate": "chat",
    "ChatPromptTemplate": "chat",
    "DictPromptTemplate": "dict",
    "HumanMessagePromptTemplate": "chat",
    "MessagesPlaceholder": "chat",
    "SystemMessagePromptTemplate": "chat",
    # Few-shot 模块
    "FewShotChatMessagePromptTemplate": "few_shot",
    "FewShotPromptTemplate": "few_shot",
    "FewShotPromptWithTemplates": "few_shot_with_templates",
    # 加载模块
    "load_prompt": "loading",
    # 提示模块
    "PromptTemplate": "prompt",
    # 字符串模块
    "StringPromptTemplate": "string",
    "check_valid_template": "string",
    "get_template_variables": "string",
    "jinja2_formatter": "string",
    "validate_jinja2": "string",
}


def __getattr__(attr_name: str) -> object:
    """动态导入属性。

    当访问模块中未直接定义的属性时，Python 会调用此函数。
    这实现了延迟加载，可以显著提高模块导入速度。

    Args:
        attr_name: 要访问的属性名称。

    Returns:
        导入的属性（类或函数）。
    """
    module_name = _dynamic_imports.get(attr_name)
    result = import_attr(attr_name, module_name, __spec__.parent)
    # 缓存结果，避免重复导入
    globals()[attr_name] = result
    return result


def __dir__() -> list[str]:
    """返回模块中可用的公开属性列表。

    这用于支持 IDE 的自动补全功能。
    """
    return list(__all__)
