"""Validation utilities for LangChain serialization.

Provides escape-based protection against injection attacks in serialized objects. The
approach uses an allowlist design: only dicts explicitly produced by
`Serializable.to_json()` are treated as LC objects during deserialization.

## How escaping works
#中文: # 转义是如何进行的

During serialization, plain dicts (user data) that contain an `'lc'` key are wrapped:

```python
{"lc": 1, ...}  # user data that looks like LC object
# becomes:
# 中文: 变成：
{"__lc_escaped__": {"lc": 1, ...}}
```

During deserialization, escaped dicts are unwrapped and returned as plain dicts,
NOT instantiated as LC objects.

中文翻译:
LangChain 序列化的验证实用程序。
提供基于转义的保护，防止序列化对象中的注入攻击。的
方法使用白名单设计：仅由以下人员明确生成的字典
在反序列化过程中，`Serialized.to_json()` 被视为 LC 对象。
## 转义是如何进行的
在序列化期间，包含“lc”键的普通字典（用户数据）被包装：
````蟒蛇
{"lc": 1, ...} # 看起来像 LC 对象的用户数据
# 变为：
{“__lc_escaped__”：{“lc”：1，...}}
````
在反序列化期间，转义的字典被解开并作为普通字典返回，
未实例化为 LC 对象。
"""

from typing import Any

_LC_ESCAPED_KEY = "__lc_escaped__"
"""Sentinel key used to mark escaped user dicts during serialization.

When a plain dict contains 'lc' key (which could be confused with LC objects),
we wrap it as {"__lc_escaped__": {...original...}}.

中文翻译:
Sentinel key 用于在序列化期间标记转义的用户字典。
当普通字典包含“lc”键（可能与 LC 对象混淆）时，
我们将其包装为 {"__lc_escaped__": {...original...}}。
"""


def _needs_escaping(obj: dict[str, Any]) -> bool:
    """Check if a dict needs escaping to prevent confusion with LC objects.

    A dict needs escaping if:

    1. It has an `'lc'` key (could be confused with LC serialization format)
    2. It has only the escape key (would be mistaken for an escaped dict)
    

    中文翻译:
    检查字典是否需要转义以防止与 LC 对象混淆。
    如果出现以下情况，则需要转义字典：
    1.它有一个“lc”密钥（可能与 LC 序列化格式混淆）
    2.它只有转义键（会被误认为是转义字典）"""
    return "lc" in obj or (len(obj) == 1 and _LC_ESCAPED_KEY in obj)


def _escape_dict(obj: dict[str, Any]) -> dict[str, Any]:
    """Wrap a dict in the escape marker.

    Example:
        ```python
        {"key": "value"}  # becomes {"__lc_escaped__": {"key": "value"}}
        ```
    

    中文翻译:
    在转义标记中包含一个字典。
    示例：
        ````蟒蛇
        {"key": "value"} # 变为 {"__lc_escaped__": {"key": "value"}}
        ````"""
    return {_LC_ESCAPED_KEY: obj}


def _is_escaped_dict(obj: dict[str, Any]) -> bool:
    """Check if a dict is an escaped user dict.

    Example:
        ```python
        {"__lc_escaped__": {...}}  # is an escaped dict
        ```
    

    中文翻译:
    检查字典是否是转义的用户字典。
    示例：
        ````蟒蛇
        {"__lc_escaped__": {...}} # 是一个转义字典
        ````"""
    return len(obj) == 1 and _LC_ESCAPED_KEY in obj


def _serialize_value(obj: Any) -> Any:
    """Serialize a value with escaping of user dicts.

    Called recursively on kwarg values to escape any plain dicts that could be confused
    with LC objects.

    Args:
        obj: The value to serialize.

    Returns:
        The serialized value with user dicts escaped as needed.
    

    中文翻译:
    通过转义用户字典来序列化一个值。
    对 kwarg 值进行递归调用，以避免任何可能混淆的普通字典
    与 LC 对象。
    参数：
        obj：要序列化的值。
    返回：
        带有用户字典的序列化值根据需要进行转义。"""
    from langchain_core.load.serializable import (  # noqa: PLC0415
        Serializable,
        to_json_not_implemented,
    )

    if isinstance(obj, Serializable):
        # This is an LC object - serialize it properly (not escaped)
        # 中文: 这是一个 LC 对象 - 正确序列化它（未转义）
        return _serialize_lc_object(obj)
    if isinstance(obj, dict):
        if not all(isinstance(k, (str, int, float, bool, type(None))) for k in obj):
            # if keys are not json serializable
            # 中文: 如果键不是 json 可序列化的
            return to_json_not_implemented(obj)
        # Check if dict needs escaping BEFORE recursing into values.
        # 中文: 检查 dict 是否需要在递归到值之前转义。
        # If it needs escaping, wrap it as-is - the contents are user data that
        # 中文: 如果需要转义，请按原样包装它 - 内容是用户数据
        # will be returned as-is during deserialization (no instantiation).
        # 中文: 将在反序列化期间按原样返回（无实例化）。
        # This prevents re-escaping of already-escaped nested content.
        # 中文: 这可以防止重新转义已经转义的嵌套内容。
        if _needs_escaping(obj):
            return _escape_dict(obj)
        # Safe dict (no 'lc' key) - recurse into values
        # 中文: 安全字典（无“lc”键）-递归到值
        return {k: _serialize_value(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize_value(item) for item in obj]
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj

    # Non-JSON-serializable object (datetime, custom objects, etc.)
    # 中文: 非 JSON 可序列化对象（日期时间、自定义对象等）
    return to_json_not_implemented(obj)


def _is_lc_secret(obj: Any) -> bool:
    """Check if an object is a LangChain secret marker.

    中文翻译:
    检查一个对象是否是LangChain秘密标记。"""
    expected_num_keys = 3
    return (
        isinstance(obj, dict)
        and obj.get("lc") == 1
        and obj.get("type") == "secret"
        and "id" in obj
        and len(obj) == expected_num_keys
    )


def _serialize_lc_object(obj: Any) -> dict[str, Any]:
    """Serialize a `Serializable` object with escaping of user data in kwargs.

    Args:
        obj: The `Serializable` object to serialize.

    Returns:
        The serialized dict with user data in kwargs escaped as needed.

    Note:
        Kwargs values are processed with `_serialize_value` to escape user data (like
        metadata) that contains `'lc'` keys. Secret fields (from `lc_secrets`) are
        skipped because `to_json()` replaces their values with secret markers.
    

    中文翻译:
    通过在 kwargs 中转义用户数据来序列化“可序列化”对象。
    参数：
        obj：要序列化的“可序列化”对象。
    返回：
        带有 kwargs 格式的用户数据的序列化字典根据需要进行转义。
    注意：
        Kwargs 值使用 `_serialize_value` 进行处理以转义用户数据（例如
        元数据）包含“lc”键。秘密字段（来自“lc_secrets”）是
        跳过，因为 `to_json()` 用秘密标记替换它们的值。"""
    from langchain_core.load.serializable import Serializable  # noqa: PLC0415

    if not isinstance(obj, Serializable):
        msg = f"Expected Serializable, got {type(obj)}"
        raise TypeError(msg)

    serialized: dict[str, Any] = dict(obj.to_json())

    # Process kwargs to escape user data that could be confused with LC objects
    # 中文: 处理 kwargs 以转义可能与 LC 对象混淆的用户数据
    # Skip secret fields - to_json() already converted them to secret markers
    # 中文: 跳过秘密字段 - to_json() 已将它们转换为秘密标记
    if serialized.get("type") == "constructor" and "kwargs" in serialized:
        serialized["kwargs"] = {
            k: v if _is_lc_secret(v) else _serialize_value(v)
            for k, v in serialized["kwargs"].items()
        }

    return serialized


def _unescape_value(obj: Any) -> Any:
    """Unescape a value, processing escape markers in dict values and lists.

    When an escaped dict is encountered (`{"__lc_escaped__": ...}`), it's
    unwrapped and the contents are returned AS-IS (no further processing).
    The contents represent user data that should not be modified.

    For regular dicts and lists, we recurse to find any nested escape markers.

    Args:
        obj: The value to unescape.

    Returns:
        The unescaped value.
    

    中文翻译:
    对值进行取消转义，处理字典值和列表中的转义标记。
    当遇到转义字典时（`{"__lc_escaped__": ...}`），它是
    拆开包装，内容按原样返回（无需进一步处理）。
    这些内容代表不应修改的用户数据。
    对于常规字典和列表，我们递归查找任何嵌套的转义标记。
    参数：
        obj：要取消转义的值。
    返回：
        未转义的值。"""
    if isinstance(obj, dict):
        if _is_escaped_dict(obj):
            # Unwrap and return the user data as-is (no further unescaping).
            # 中文: 解开并按原样返回用户数据（不再进一步转义）。
            # The contents are user data that may contain more escape keys,
            # 中文: 内容是可能包含更多转义键的用户数据，
            # but those are part of the user's actual data.
            # 中文: 但这些是用户实际数据的一部分。
            return obj[_LC_ESCAPED_KEY]

        # Regular dict - recurse into values to find nested escape markers
        # 中文: 常规字典 - 递归到值以查找嵌套转义标记
        return {k: _unescape_value(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_unescape_value(item) for item in obj]
    return obj
