"""**Load** module helps with serialization and deserialization.

中文翻译:
**Load** 模块有助于序列化和反序列化。"""

from typing import TYPE_CHECKING

from langchain_core._import_utils import import_attr

if TYPE_CHECKING:
    from langchain_core.load.dump import dumpd, dumps
    from langchain_core.load.load import InitValidator, loads
    from langchain_core.load.serializable import Serializable

# Unfortunately, we have to eagerly import load from langchain_core/load/load.py
# 中文: 不幸的是，我们必须急切地从 langchain_core/load/load.py 导入负载
# eagerly to avoid a namespace conflict. We want users to still be able to use
# 中文: 急切地避免命名空间冲突。我们希望用户仍然能够使用
# `from langchain_core.load import load` to get the load function, but
# 中文: `from langchain_core.load import load` 来获取load函数，但是
# the `from langchain_core.load.load import load` absolute import should also work.
# 中文: `from langchain_core.load.load import load` 绝对导入也应该起作用。
from langchain_core.load.load import load

__all__ = (
    "InitValidator",
    "Serializable",
    "dumpd",
    "dumps",
    "load",
    "loads",
)

_dynamic_imports = {
    "dumpd": "dump",
    "dumps": "dump",
    "InitValidator": "load",
    "loads": "load",
    "Serializable": "serializable",
}


def __getattr__(attr_name: str) -> object:
    module_name = _dynamic_imports.get(attr_name)
    result = import_attr(attr_name, module_name, __spec__.parent)
    globals()[attr_name] = result
    return result


def __dir__() -> list[str]:
    return list(__all__)
