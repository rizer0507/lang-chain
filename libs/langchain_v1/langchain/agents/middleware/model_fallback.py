"""模型回退中间件模块。

本模块提供在模型调用失败时自动切换到备用模型的能力。

核心类:
--------
**ModelFallbackMiddleware**: 模型回退中间件

功能特性:
---------
- 主模型失败时自动尝试备用模型
- 按顺序尝试多个备用模型
- 支持字符串和 BaseChatModel 实例

使用示例:
---------
>>> from langchain.agents import create_agent
>>> from langchain.agents.middleware import ModelFallbackMiddleware
>>>
>>> fallback = ModelFallbackMiddleware(
...     "openai:gpt-4o-mini",      # 第一个备用
...     "anthropic:claude-sonnet-4-5-20250929",  # 第二个备用
... )
>>>
>>> agent = create_agent(
...     model="openai:gpt-4o",  # 主模型
...     middleware=[fallback],
... )
>>>
>>> # 如果主模型失败，依次尝试 gpt-4o-mini, claude-sonnet
>>> result = agent.invoke({"messages": [...]})
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)
from langchain.chat_models import init_chat_model

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain_core.language_models.chat_models import BaseChatModel


class ModelFallbackMiddleware(AgentMiddleware):
    """Automatic fallback to alternative models on errors.

    Retries failed model calls with alternative models in sequence until
    success or all models exhausted. Primary model specified in `create_agent`.

    Example:
        ```python
        from langchain.agents.middleware.model_fallback import ModelFallbackMiddleware
        from langchain.agents import create_agent

        fallback = ModelFallbackMiddleware(
            "openai:gpt-4o-mini",  # Try first on error
            "anthropic:claude-sonnet-4-5-20250929",  # Then this
        )

        agent = create_agent(
            model="openai:gpt-4o",  # Primary model
            middleware=[fallback],
        )

        # If primary fails: tries gpt-4o-mini, then claude-sonnet-4-5-20250929
        # 中文: 如果主要失败：尝试 gpt-4o-mini，然后尝试 claude-sonnet-4-5-20250929
        result = await agent.invoke({"messages": [HumanMessage("Hello")]})
        ```
    

    中文翻译:
    出现错误时自动回退到替代模型。
    按顺序使用替代模型重试失败的模型调用，直到
    成功或所有模型都用尽。在`create_agent`中指定的主要模型。
    示例：
        ````蟒蛇
        从 langchain.agents.middleware.model_fallback 导入 ModelFallbackMiddleware
        从 langchain.agents 导入 create_agent
        后备 = ModelFallbackMiddleware(
            "openai:gpt-4o-mini", # 出错时先尝试
            "anthropic:claude-sonnet-4-5-20250929", # 然后这个
        ）
        代理=创建_代理（
            model="openai:gpt-4o", # 主要模型
            中间件=[后备],
        ）
        # 如果主节点失败：尝试 gpt-4o-mini，然后是 claude-sonnet-4-5-20250929
        结果=等待agent.invoke({"messages": [HumanMessage("Hello")]})
        ````"""

    def __init__(
        self,
        first_model: str | BaseChatModel,
        *additional_models: str | BaseChatModel,
    ) -> None:
        """Initialize model fallback middleware.

        Args:
            first_model: First fallback model (string name or instance).
            *additional_models: Additional fallbacks in order.
        

        中文翻译:
        初始化模型后备中间件。
        参数：
            first_model：第一个后备模型（字符串名称或实例）。
            *additional_models：按顺序附加后备。"""
        super().__init__()

        # Initialize all fallback models
        # 中文: 初始化所有后备模型
        all_models = (first_model, *additional_models)
        self.models: list[BaseChatModel] = []
        for model in all_models:
            if isinstance(model, str):
                self.models.append(init_chat_model(model))
            else:
                self.models.append(model)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Try fallback models in sequence on errors.

        Args:
            request: Initial model request.
            handler: Callback to execute the model.

        Returns:
            AIMessage from successful model call.

        Raises:
            Exception: If all models fail, re-raises last exception.
        

        中文翻译:
        在出现错误时按顺序尝试后备模型。
        参数：
            请求：初始模型请求。
            handler：执行模型的回调。
        返回：
            来自成功模型调用的 AIMessage。
        加薪：
            异常：如果所有模型都失败，则重新引发最后一个异常。"""
        # Try primary model first
        # 中文: 首先尝试主要模型
        last_exception: Exception
        try:
            return handler(request)
        except Exception as e:
            last_exception = e

        # Try fallback models
        # 中文: 尝试后备模型
        for fallback_model in self.models:
            try:
                return handler(request.override(model=fallback_model))
            except Exception as e:
                last_exception = e
                continue

        raise last_exception

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Try fallback models in sequence on errors (async version).

        Args:
            request: Initial model request.
            handler: Async callback to execute the model.

        Returns:
            AIMessage from successful model call.

        Raises:
            Exception: If all models fail, re-raises last exception.
        

        中文翻译:
        在出现错误时按顺序尝试回退模型（异步版本）。
        参数：
            请求：初始模型请求。
            handler：执行模型的异步回调。
        返回：
            来自成功模型调用的 AIMessage。
        加薪：
            异常：如果所有模型都失败，则重新引发最后一个异常。"""
        # Try primary model first
        # 中文: 首先尝试主要模型
        last_exception: Exception
        try:
            return await handler(request)
        except Exception as e:
            last_exception = e

        # Try fallback models
        # 中文: 尝试后备模型
        for fallback_model in self.models:
            try:
                return await handler(request.override(model=fallback_model))
            except Exception as e:
                last_exception = e
                continue

        raise last_exception
