"""Base abstraction and in-memory implementation of rate limiters.

These rate limiters can be used to limit the rate of requests to an API.

The rate limiters can be used together with `BaseChatModel`.

中文翻译:
速率限制器的基本抽象和内存中实现。
这些速率限制器可用于限制对 API 的请求速率。
速率限制器可以与“BaseChatModel”一起使用。
"""

from langchain_core.rate_limiters import BaseRateLimiter, InMemoryRateLimiter

__all__ = [
    "BaseRateLimiter",
    "InMemoryRateLimiter",
]
