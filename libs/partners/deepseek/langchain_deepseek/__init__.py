"""LangChain DeepSeek integration.

中文翻译:
浪链DeepSeek集成。"""

from importlib import metadata

from langchain_deepseek.chat_models import ChatDeepSeek

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    # 中文: 包元数据不可用的情况。
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "ChatDeepSeek",
    "__version__",
]
