"""Serialize LangChain objects to JSON.

Provides `dumps` (to JSON string) and `dumpd` (to dict) for serializing
`Serializable` objects.

## Escaping
#中文: # 转义

During serialization, plain dicts (user data) that contain an `'lc'` key are escaped
by wrapping them: `{"__lc_escaped__": {...original...}}`. This prevents injection
attacks where malicious data could trick the deserializer into instantiating
arbitrary classes. The escape marker is removed during deserialization.

This is an allowlist approach: only dicts explicitly produced by
`Serializable.to_json()` are treated as LC objects; everything else is escaped if it
could be confused with the LC format.

中文翻译:
将 LangChain 对象序列化为 JSON。
提供用于序列化的“dumps”（到 JSON 字符串）和“dumpd”（到 dict）
“可序列化”对象。
## 逃脱
在序列化期间，包含“lc”键的普通字典（用户数据）被转义
通过包装它们：`{"__lc_escaped__": {...original...}}`。这可以防止注入
恶意数据可能诱骗解串器实例化的攻击
任意类。转义标记在反序列化期间被删除。
这是一种白名单方法：仅由以下人员明确生成的字典
`Serialized.to_json()` 被视为 LC 对象；如果是的话，其他一切都会被转义
可能会与 LC 格式混淆。
"""

import json
from typing import Any

from pydantic import BaseModel

from langchain_core.load._validation import _serialize_value
from langchain_core.load.serializable import Serializable, to_json_not_implemented
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration


def default(obj: Any) -> Any:
    """Return a default value for an object.

    Args:
        obj: The object to serialize to json if it is a Serializable object.

    Returns:
        A JSON serializable object or a SerializedNotImplemented object.
    

    中文翻译:
    返回对象的默认值。
    参数：
        obj：如果是可序列化对象，则要序列化为 json 的对象。
    返回：
        JSON 可序列化对象或 SerializedNotImplemented 对象。"""
    if isinstance(obj, Serializable):
        return obj.to_json()
    return to_json_not_implemented(obj)


def _dump_pydantic_models(obj: Any) -> Any:
    """Convert nested Pydantic models to dicts for JSON serialization.

    Handles the special case where a `ChatGeneration` contains an `AIMessage`
    with a parsed Pydantic model in `additional_kwargs["parsed"]`. Since
    Pydantic models aren't directly JSON serializable, this converts them to
    dicts.

    Args:
        obj: The object to process.

    Returns:
        A copy of the object with nested Pydantic models converted to dicts, or
            the original object unchanged if no conversion was needed.
    

    中文翻译:
    将嵌套 Pydantic 模型转换为字典以进行 JSON 序列化。
    处理“ChatGeneration”包含“AIMessage”的特殊情况
    在 `additional_kwargs["parsed"]` 中使用已解析的 Pydantic 模型。自从
    Pydantic 模型不能直接 JSON 序列化，这会将它们转换为
    听写。
    参数：
        obj：要处理的对象。
    返回：
        具有转换为字典的嵌套 Pydantic 模型的对象的副本，或
            如果不需要转换，则原始对象不变。"""
    if (
        isinstance(obj, ChatGeneration)
        and isinstance(obj.message, AIMessage)
        and (parsed := obj.message.additional_kwargs.get("parsed"))
        and isinstance(parsed, BaseModel)
    ):
        obj_copy = obj.model_copy(deep=True)
        obj_copy.message.additional_kwargs["parsed"] = parsed.model_dump()
        return obj_copy
    return obj


def dumps(obj: Any, *, pretty: bool = False, **kwargs: Any) -> str:
    """Return a JSON string representation of an object.

    Note:
        Plain dicts containing an `'lc'` key are automatically escaped to prevent
        confusion with LC serialization format. The escape marker is removed during
        deserialization.

    Args:
        obj: The object to dump.
        pretty: Whether to pretty print the json.

            If `True`, the json will be indented by either 2 spaces or the amount
            provided in the `indent` kwarg.
        **kwargs: Additional arguments to pass to `json.dumps`

    Returns:
        A JSON string representation of the object.

    Raises:
        ValueError: If `default` is passed as a kwarg.
    

    中文翻译:
    返回对象的 JSON 字符串表示形式。
    注意：
        包含“lc”键的普通字典会自动转义以防止
        与 LC 序列化格式混淆。转义标记在期间被删除
        反序列化。
    参数：
        obj：要转储的对象。
        Pretty: 是否漂亮地打印 json。
            如果为“True”，则 json 将缩进 2 个空格或数量
            在 `indent` kwarg 中提供。
        **kwargs：传递给 `json.dumps` 的附加参数
    返回：
        对象的 JSON 字符串表示形式。
    加薪：
        ValueError：如果“default”作为 kwarg 传递。"""
    if "default" in kwargs:
        msg = "`default` should not be passed to dumps"
        raise ValueError(msg)

    obj = _dump_pydantic_models(obj)
    serialized = _serialize_value(obj)

    if pretty:
        indent = kwargs.pop("indent", 2)
        return json.dumps(serialized, indent=indent, **kwargs)
    return json.dumps(serialized, **kwargs)


def dumpd(obj: Any) -> Any:
    """Return a dict representation of an object.

    Note:
        Plain dicts containing an `'lc'` key are automatically escaped to prevent
        confusion with LC serialization format. The escape marker is removed during
        deserialization.

    Args:
        obj: The object to dump.

    Returns:
        Dictionary that can be serialized to json using `json.dumps`.
    

    中文翻译:
    返回对象的字典表示。
    注意：
        包含“lc”键的普通字典会自动转义以防止
        与 LC 序列化格式混淆。转义标记在期间被删除
        反序列化。
    参数：
        obj：要转储的对象。
    返回：
        可以使用“json.dumps”序列化为 json 的字典。"""
    obj = _dump_pydantic_models(obj)
    return _serialize_value(obj)
