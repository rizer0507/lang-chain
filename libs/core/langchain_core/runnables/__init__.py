"""LangChain **Runnable** 和 **LangChain 表达式语言 (LCEL)** 模块。

LangChain 表达式语言 (LCEL) 提供了一种声明式方法来构建
利用 LLM 能力的生产级程序。

核心特性:
---------
使用 LCEL 和 LangChain Runnable 创建的程序天然支持：

- **同步执行**: 标准的同步调用
- **异步执行**: 支持 async/await，适合高并发服务器
- **批量处理**: 并行处理多个输入
- **流式输出**: 实时输出中间结果，提升用户体验

LCEL 基础:
---------
LCEL 的核心是使用 `|` 运算符将 Runnable 对象串联：

>>> from langchain_core.prompts import ChatPromptTemplate
>>> from langchain_openai import ChatOpenAI
>>> from langchain_core.output_parsers import StrOutputParser
>>>
>>> # 创建一个简单的链
>>> prompt = ChatPromptTemplate.from_template("讲一个关于{topic}的笑话")
>>> model = ChatOpenAI()
>>> parser = StrOutputParser()
>>>
>>> chain = prompt | model | parser
>>> chain.invoke({"topic": "程序员"})

核心类:
--------
基础类:
- **Runnable**: LCEL 的核心抽象，所有可组合组件的基类
- **RunnableSerializable**: 可序列化的 Runnable
- **RunnableLambda**: 将函数包装为 Runnable
- **RunnableSequence**: 串联多个 Runnable（即 A | B）
- **RunnableParallel**: 并行执行多个 Runnable

控制流:
- **RunnableBranch**: 条件分支
- **RunnablePassthrough**: 直接传递输入
- **RunnableAssign**: 添加新键到字典
- **RunnablePick**: 从字典中选择键

高级功能:
- **RunnableWithFallbacks**: 失败回退
- **RunnableWithMessageHistory**: 消息历史管理
- **RunnableBinding**: 绑定配置

配置:
- **RunnableConfig**: 运行配置（回调、元数据、标签等）
"""

from typing import TYPE_CHECKING

from langchain_core._import_utils import import_attr

if TYPE_CHECKING:
    from langchain_core.runnables.base import (
        Runnable,
        RunnableBinding,
        RunnableGenerator,
        RunnableLambda,
        RunnableMap,
        RunnableParallel,
        RunnableSequence,
        RunnableSerializable,
        chain,
    )
    from langchain_core.runnables.branch import RunnableBranch
    from langchain_core.runnables.config import (
        RunnableConfig,
        ensure_config,
        get_config_list,
        patch_config,
        run_in_executor,
    )
    from langchain_core.runnables.fallbacks import RunnableWithFallbacks
    from langchain_core.runnables.history import RunnableWithMessageHistory
    from langchain_core.runnables.passthrough import (
        RunnableAssign,
        RunnablePassthrough,
        RunnablePick,
    )
    from langchain_core.runnables.router import RouterInput, RouterRunnable
    from langchain_core.runnables.utils import (
        AddableDict,
        ConfigurableField,
        ConfigurableFieldMultiOption,
        ConfigurableFieldSingleOption,
        ConfigurableFieldSpec,
        aadd,
        add,
    )

# 公开导出的符号列表
__all__ = (
    # 工具类
    "AddableDict",
    "ConfigurableField",
    "ConfigurableFieldMultiOption",
    "ConfigurableFieldSingleOption",
    "ConfigurableFieldSpec",
    # 路由
    "RouterInput",
    "RouterRunnable",
    # 核心 Runnable 类
    "Runnable",
    "RunnableAssign",
    "RunnableBinding",
    "RunnableBranch",
    "RunnableConfig",
    "RunnableGenerator",
    "RunnableLambda",
    "RunnableMap",
    "RunnableParallel",
    "RunnablePassthrough",
    "RunnablePick",
    "RunnableSequence",
    "RunnableSerializable",
    "RunnableWithFallbacks",
    "RunnableWithMessageHistory",
    # 函数
    "aadd",
    "add",
    "chain",
    "ensure_config",
    "get_config_list",
    "patch_config",
    "run_in_executor",
)

# 动态导入映射
# 键是符号名称，值是模块名称
_dynamic_imports = {
    # base.py - 核心 Runnable 类
    "chain": "base",
    "Runnable": "base",
    "RunnableBinding": "base",
    "RunnableGenerator": "base",
    "RunnableLambda": "base",
    "RunnableMap": "base",
    "RunnableParallel": "base",
    "RunnableSequence": "base",
    "RunnableSerializable": "base",
    # branch.py - 条件分支
    "RunnableBranch": "branch",
    # config.py - 配置相关
    "RunnableConfig": "config",
    "ensure_config": "config",
    "get_config_list": "config",
    "patch_config": "config",
    "run_in_executor": "config",
    # fallbacks.py - 失败回退
    "RunnableWithFallbacks": "fallbacks",
    # history.py - 消息历史
    "RunnableWithMessageHistory": "history",
    # passthrough.py - 直通和赋值
    "RunnableAssign": "passthrough",
    "RunnablePassthrough": "passthrough",
    "RunnablePick": "passthrough",
    # router.py - 路由
    "RouterInput": "router",
    "RouterRunnable": "router",
    # utils.py - 工具函数
    "AddableDict": "utils",
    "ConfigurableField": "utils",
    "ConfigurableFieldMultiOption": "utils",
    "ConfigurableFieldSingleOption": "utils",
    "ConfigurableFieldSpec": "utils",
    "aadd": "utils",
    "add": "utils",
}


def __getattr__(attr_name: str) -> object:
    """动态导入属性。

    实现延迟加载，提高模块导入速度。
    """
    module_name = _dynamic_imports.get(attr_name)
    result = import_attr(attr_name, module_name, __spec__.parent)
    globals()[attr_name] = result
    return result


def __dir__() -> list[str]:
    """返回模块中可用的公开属性列表。"""
    return list(__all__)
