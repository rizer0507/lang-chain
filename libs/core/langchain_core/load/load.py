"""Load LangChain objects from JSON strings or objects.

## How it works
#中文: # 它是如何工作的

Each `Serializable` LangChain object has a unique identifier (its "class path"), which
is a list of strings representing the module path and class name. For example:

- `AIMessage` -> `["langchain_core", "messages", "ai", "AIMessage"]`
- `ChatPromptTemplate` -> `["langchain_core", "prompts", "chat", "ChatPromptTemplate"]`

When deserializing, the class path from the JSON `'id'` field is checked against an
allowlist. If the class is not in the allowlist, deserialization raises a `ValueError`.

## Security model
#中文: # 安全模型

The `allowed_objects` parameter controls which classes can be deserialized:

- **`'core'` (default)**: Allow classes defined in the serialization mappings for
    langchain_core.
- **`'all'`**: Allow classes defined in the serialization mappings. This
    includes core LangChain types (messages, prompts, documents, etc.) and trusted
    partner integrations. See `langchain_core.load.mapping` for the full list.
- **Explicit list of classes**: Only those specific classes are allowed.

For simple data types like messages and documents, the default allowlist is safe to use.
These classes do not perform side effects during initialization.

!!! note "Side effects in allowed classes"

    Deserialization calls `__init__` on allowed classes. If those classes perform side
    effects during initialization (network calls, file operations, etc.), those side
    effects will occur. The allowlist prevents instantiation of classes outside the
    allowlist, but does not sandbox the allowed classes themselves.

Import paths are also validated against trusted namespaces before any module is
imported.

### Injection protection (escape-based)
#中文: ## 注入保护（基于转义）

During serialization, plain dicts that contain an `'lc'` key are escaped by wrapping
them: `{"__lc_escaped__": {...}}`. During deserialization, escaped dicts are unwrapped
and returned as plain dicts, NOT instantiated as LC objects.

This is an allowlist approach: only dicts explicitly produced by
`Serializable.to_json()` (which are NOT escaped) are treated as LC objects;
everything else is user data.

Even if an attacker's payload includes `__lc_escaped__` wrappers, it will be unwrapped
to plain dicts and NOT instantiated as malicious objects.

## Examples
#中文: # 例子

```python
from langchain_core.load import load
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage

# Use default allowlist (classes from mappings) - recommended
# 中文: 使用默认允许列表（映射中的类） - 推荐
obj = load(data)

# Allow only specific classes (most restrictive)
# 中文: 仅允许特定类别（最严格）
obj = load(
    data,
    allowed_objects=[
        ChatPromptTemplate,
        AIMessage,
        HumanMessage,
    ],
)
```

中文翻译:
从 JSON 字符串或对象加载 LangChain 对象。
## 它是如何工作的
每个“可序列化”LangChain 对象都有一个唯一的标识符（其“类路径”），该标识符
是表示模块路径和类名的字符串列表。例如：
- `AIMessage` -> `["langchain_core", "messages", "ai", "AIMessage"]`
- `ChatPromptTemplate` -> `["langchain_core", "提示", "聊天", "ChatPromptTemplate"]`
反序列化时，将根据 JSON“id”字段检查类路径
允许名单。如果该类不在允许列表中，反序列化会引发“ValueError”。
## 安全模型
“allowed_objects”参数控制哪些类可以被反序列化：
- **``'core'`（默认）**：允许在序列化映射中定义的类
    langchain_core。
- **`'all'`**：允许在序列化映射中定义的类。这个
    包括浪链核心类型（消息、提示、文档等）和可信
    合作伙伴集成。有关完整列表，请参阅“langchain_core.load.mapping”。
- **明确的类列表**：仅允许那些特定的类。
对于消息和文档等简单数据类型，默认允许列表可以安全使用。
这些类在初始化期间不会产生副作用。
!!!注意“允许的类别中的副作用”
    反序列化在允许的类上调用“__init__”。如果这些课程执行侧面
    初始化期间的影响（网络调用、文件操作等），那些方面
    会发生影响。允许列表可防止实例化外部类
    允许列表，但不对允许的类本身进行沙箱处理。
在任何模块被导入之前，导入路径也会根据受信任的命名空间进行验证。
进口的。
### 注入保护（基于转义）
在序列化期间，包含“lc”键的普通字典通过包装进行转义
他们：`{"__lc_escaped__": {...}}`。在反序列化过程中，转义的字典被解开
并作为普通字典返回，而不是实例化为 LC 对象。
这是一种白名单方法：仅由以下人员明确生成的字典
`Serialized.to_json()`（未转义）被视为 LC 对象；
其他一切都是用户数据。
即使攻击者的有效负载包含`__lc_escaped__`包装器，它也会被解开
为普通字典，而不是实例化为恶意对象。
## 示例
````蟒蛇
从 langchain_core.load 导入加载
从 langchain_core.prompts 导入 ChatPromptTemplate
从 langchain_core.messages 导入 AIMessage、HumanMessage
# 使用默认白名单（映射中的类） - 推荐
obj = 加载（数据）
# 只允许特定的类（最严格的）
对象 = 负载（
    数据，
    允许的对象=[
        聊天提示模板，
        人工智能留言，
        人类讯息，
    ],
）
````
"""

import importlib
import json
import os
from collections.abc import Callable, Iterable
from typing import Any, Literal, cast

from langchain_core._api import beta
from langchain_core.load._validation import _is_escaped_dict, _unescape_value
from langchain_core.load.mapping import (
    _JS_SERIALIZABLE_MAPPING,
    _OG_SERIALIZABLE_MAPPING,
    OLD_CORE_NAMESPACES_MAPPING,
    SERIALIZABLE_MAPPING,
)
from langchain_core.load.serializable import Serializable

DEFAULT_NAMESPACES = [
    "langchain",
    "langchain_core",
    "langchain_community",
    "langchain_anthropic",
    "langchain_groq",
    "langchain_google_genai",
    "langchain_aws",
    "langchain_openai",
    "langchain_google_vertexai",
    "langchain_mistralai",
    "langchain_fireworks",
    "langchain_xai",
    "langchain_sambanova",
    "langchain_perplexity",
]
# Namespaces for which only deserializing via the SERIALIZABLE_MAPPING is allowed.
# 中文: 仅允许通过 SERIALIZABLE_MAPPING 反序列化的命名空间。
# Load by path is not allowed.
# 中文: 不允许按路径加载。
DISALLOW_LOAD_FROM_PATH = [
    "langchain_community",
    "langchain",
]

ALL_SERIALIZABLE_MAPPINGS = {
    **SERIALIZABLE_MAPPING,
    **OLD_CORE_NAMESPACES_MAPPING,
    **_OG_SERIALIZABLE_MAPPING,
    **_JS_SERIALIZABLE_MAPPING,
}

# Cache for the default allowed class paths computed from mappings
# 中文: 缓存根据映射计算的默认允许的类路径
# Maps mode ("all" or "core") to the cached set of paths
# 中文: 将模式（“全部”或“核心”）映射到缓存的路径集
_default_class_paths_cache: dict[str, set[tuple[str, ...]]] = {}


def _get_default_allowed_class_paths(
    allowed_object_mode: Literal["all", "core"],
) -> set[tuple[str, ...]]:
    """Get the default allowed class paths from the serialization mappings.

    This uses the mappings as the source of truth for what classes are allowed
    by default. Both the legacy paths (keys) and current paths (values) are included.

    Args:
        allowed_object_mode: either `'all'` or `'core'`.

    Returns:
        Set of class path tuples that are allowed by default.
    

    中文翻译:
    从序列化映射中获取默认允许的类路径。
    这使用映射作为允许哪些类的真实来源
    默认情况下。包括旧路径（键）和当前路径（值）。
    参数：
        allowed_object_mode: either `'all'` or `'core'`.
    返回：
        默认情况下允许的类路径元组集。"""
    if allowed_object_mode in _default_class_paths_cache:
        return _default_class_paths_cache[allowed_object_mode]

    allowed_paths: set[tuple[str, ...]] = set()
    for key, value in ALL_SERIALIZABLE_MAPPINGS.items():
        if allowed_object_mode == "core" and value[0] != "langchain_core":
            continue
        allowed_paths.add(key)
        allowed_paths.add(value)

    _default_class_paths_cache[allowed_object_mode] = allowed_paths
    return _default_class_paths_cache[allowed_object_mode]


def _block_jinja2_templates(
    class_path: tuple[str, ...],
    kwargs: dict[str, Any],
) -> None:
    """Block jinja2 templates during deserialization for security.

    Jinja2 templates can execute arbitrary code, so they are blocked by default when
    deserializing objects with `template_format='jinja2'`.

    Note:
        We intentionally do NOT check the `class_path` here to keep this simple and
        future-proof. If any new class is added that accepts `template_format='jinja2'`,
        it will be automatically blocked without needing to update this function.

    Args:
        class_path: The class path tuple being deserialized (unused).
        kwargs: The kwargs dict for the class constructor.

    Raises:
        ValueError: If `template_format` is `'jinja2'`.
    

    中文翻译:
    为了安全起见，在反序列化期间阻止 jinja2 模板。
    Jinja2 模板可以执行任意代码，因此默认情况下它们会被阻止
    使用“template_format='jinja2'”反序列化对象。
    注意：
        我们故意不检查“class_path”，以保持简单和
        面向未来。如果添加任何接受 `template_format='jinja2'` 的新类，
        无需更新此功能，就会自动屏蔽。
    参数：
        class_path：正在反序列化的类路径元组（未使用）。
        kwargs：类构造函数的 kwargs 字典。
    加薪：
        ValueError：如果“template_format”是“jinja2”。"""
    _ = class_path  # Unused - see docstring for rationale. Kept to satisfy signature.
    if kwargs.get("template_format") == "jinja2":
        msg = (
            "Jinja2 templates are not allowed during deserialization for security "
            "reasons. Use 'f-string' template format instead, or explicitly allow "
            "jinja2 by providing a custom init_validator."
        )
        raise ValueError(msg)


def default_init_validator(
    class_path: tuple[str, ...],
    kwargs: dict[str, Any],
) -> None:
    """Default init validator that blocks jinja2 templates.

    This is the default validator used by `load()` and `loads()` when no custom
    validator is provided.

    Args:
        class_path: The class path tuple being deserialized.
        kwargs: The kwargs dict for the class constructor.

    Raises:
        ValueError: If template_format is `'jinja2'`.
    

    中文翻译:
    默认初始化验证器会阻止 jinja2 模板。
    这是没有自定义时 `load()` 和 `loads()` 使用的默认验证器
    提供了验证器。
    参数：
        class_path：正在反序列化的类路径元组。
        kwargs：类构造函数的 kwargs 字典。
    加薪：
        ValueError：如果 template_format 是“jinja2”。"""
    _block_jinja2_templates(class_path, kwargs)


AllowedObject = type[Serializable]
"""Type alias for classes that can be included in the `allowed_objects` parameter.

Must be a `Serializable` subclass (the class itself, not an instance).

中文翻译:
可以包含在“allowed_objects”参数中的类的类型别名。
必须是“可序列化”子类（类本身，而不是实例）。
"""

InitValidator = Callable[[tuple[str, ...], dict[str, Any]], None]
"""Type alias for a callable that validates kwargs during deserialization.

The callable receives:

- `class_path`: A tuple of strings identifying the class being instantiated
    (e.g., `('langchain', 'schema', 'messages', 'AIMessage')`).
- `kwargs`: The kwargs dict that will be passed to the constructor.

The validator should raise an exception if the object should not be deserialized.

中文翻译:
为在反序列化期间验证 kwargs 的可调用类型键入别名。
可调用接收：
- `class_path`：标识正在实例化的类的字符串元组
    （例如，`('langchain', 'schema', 'messages', 'AIMessage')`）。
- `kwargs`：将传递给构造函数的 kwargs 字典。
如果不应反序列化对象，验证器应引发异常。
"""


def _compute_allowed_class_paths(
    allowed_objects: Iterable[AllowedObject],
    import_mappings: dict[tuple[str, ...], tuple[str, ...]],
) -> set[tuple[str, ...]]:
    """Return allowed class paths from an explicit list of classes.

    A class path is a tuple of strings identifying a serializable class, derived from
    `Serializable.lc_id()`. For example: `('langchain_core', 'messages', 'AIMessage')`.

    Args:
        allowed_objects: Iterable of `Serializable` subclasses to allow.
        import_mappings: Mapping of legacy class paths to current class paths.

    Returns:
        Set of allowed class paths.

    Example:
        ```python
        # Allow a specific class
        # 中文: 允许特定类别
        _compute_allowed_class_paths([MyPrompt], {}) ->
            {("langchain_core", "prompts", "MyPrompt")}

        # Include legacy paths that map to the same class
        # 中文: 包括映射到同一类的旧路径
        import_mappings = {("old", "Prompt"): ("langchain_core", "prompts", "MyPrompt")}
        _compute_allowed_class_paths([MyPrompt], import_mappings) ->
            {("langchain_core", "prompts", "MyPrompt"), ("old", "Prompt")}
        ```
    

    中文翻译:
    从显式类列表中返回允许的类路径。
    类路径是标识可序列化类的字符串元组，派生自
    `Serialized.lc_id()`。例如：`('langchain_core', 'messages', 'AIMessage')`。
    参数：
        allowed_objects：允许的可迭代的“可序列化”子类。
        import_mappings：遗留类路径到当前类路径的映射。
    返回：
        允许的类路径集。
    示例：
        ````蟒蛇
        # 允许特定的类
        _compute_allowed_class_paths([MyPrompt], {}) ->
            {("langchain_core", "提示", "MyPrompt")}
        # 包括映射到同一类的遗留路径
        import_mappings = {("旧", "提示"): ("langchain_core", "提示", "MyPrompt")}
        _compute_allowed_class_paths([MyPrompt], import_mappings) ->
            {("langchain_core", "提示", "MyPrompt"), ("旧", "提示")}
        ````"""
    allowed_objects_list = list(allowed_objects)

    allowed_class_paths: set[tuple[str, ...]] = set()
    for allowed_obj in allowed_objects_list:
        if not isinstance(allowed_obj, type) or not issubclass(
            allowed_obj, Serializable
        ):
            msg = "allowed_objects must contain Serializable subclasses."
            raise TypeError(msg)

        class_path = tuple(allowed_obj.lc_id())
        allowed_class_paths.add(class_path)
        # Add legacy paths that map to the same class.
        # 中文: 添加映射到同一类的旧路径。
        for mapping_key, mapping_value in import_mappings.items():
            if tuple(mapping_value) == class_path:
                allowed_class_paths.add(mapping_key)
    return allowed_class_paths


class Reviver:
    """Reviver for JSON objects.

    Used as the `object_hook` for `json.loads` to reconstruct LangChain objects from
    their serialized JSON representation.

    Only classes in the allowlist can be instantiated.
    

    中文翻译:
    JSON 对象的 Reviver。
    用作`json.loads`的`object_hook`来重建LangChain对象
    他们的序列化 JSON 表示。
    只有允许列表中的类才能被实例化。"""

    def __init__(
        self,
        allowed_objects: Iterable[AllowedObject] | Literal["all", "core"] = "core",
        secrets_map: dict[str, str] | None = None,
        valid_namespaces: list[str] | None = None,
        secrets_from_env: bool = False,  # noqa: FBT001,FBT002
        additional_import_mappings: dict[tuple[str, ...], tuple[str, ...]]
        | None = None,
        *,
        ignore_unserializable_fields: bool = False,
        init_validator: InitValidator | None = default_init_validator,
    ) -> None:
        """Initialize the reviver.

        Args:
            allowed_objects: Allowlist of classes that can be deserialized.
                - `'core'` (default): Allow classes defined in the serialization
                    mappings for `langchain_core`.
                - `'all'`: Allow classes defined in the serialization mappings.

                    This includes core LangChain types (messages, prompts, documents,
                    etc.) and trusted partner integrations. See
                    `langchain_core.load.mapping` for the full list.
                - Explicit list of classes: Only those specific classes are allowed.
            secrets_map: A map of secrets to load.
                If a secret is not found in the map, it will be loaded from the
                environment if `secrets_from_env` is `True`.
            valid_namespaces: Additional namespaces (modules) to allow during
                deserialization, beyond the default trusted namespaces.
            secrets_from_env: Whether to load secrets from the environment.
            additional_import_mappings: A dictionary of additional namespace mappings.

                You can use this to override default mappings or add new mappings.

                When `allowed_objects` is `None` (using defaults), paths from these
                mappings are also added to the allowed class paths.
            ignore_unserializable_fields: Whether to ignore unserializable fields.
            init_validator: Optional callable to validate kwargs before instantiation.

                If provided, this function is called with `(class_path, kwargs)` where
                `class_path` is the class path tuple and `kwargs` is the kwargs dict.
                The validator should raise an exception if the object should not be
                deserialized, otherwise return `None`.

                Defaults to `default_init_validator` which blocks jinja2 templates.
        

        中文翻译:
        初始化恢复器。
        参数：
            allowed_objects：可以反序列化的类的白名单。
                - `'core'`（默认）：允许在序列化中定义的类
                    “langchain_core”的映射。
                - `'all'`：允许在序列化映射中定义的类。
                    这包括核心 LangChain 类型（消息、提示、文档、
                    等）和值得信赖的合作伙伴集成。参见
                    完整列表请参见“langchain_core.load.mapping”。
                - 明确的类别列表：仅允许那些特定的类别。
            Secrets_map：要加载的秘密地图。
                如果在地图中找不到秘密，则会从
                如果“secrets_from_env”为“True”，则为环境。
            valid_namespaces：允许的附加命名空间（模块）
                反序列化，超出默认的受信任命名空间。
            Secrets_from_env：是否从环境中加载机密。
            extra_import_mappings：附加命名空间映射的字典。
                您可以使用它来覆盖默认映射或添加新映射。
                当“allowed_objects”为“None”（使用默认值）时，来自这些的路径
                映射也被添加到允许的类路径中。
            ignore_unserialized_fields：是否忽略不可序列化字段。
            init_validator：可选的可调用函数，用于在实例化之前验证 kwargs。
                如果提供，则使用“(class_path, kwargs)”调用此函数，其中
                `class_path` 是类路径元组，`kwargs` 是 kwargs 字典。
                如果对象不应该被验证，验证器应该引发异常
                反序列化，否则返回“None”。
                默认为“default_init_validator”，它会阻止 jinja2 模板。"""
        self.secrets_from_env = secrets_from_env
        self.secrets_map = secrets_map or {}
        # By default, only support langchain, but user can pass in additional namespaces
        # 中文: 默认情况下，仅支持langchain，但用户可以传入额外的命名空间
        self.valid_namespaces = (
            [*DEFAULT_NAMESPACES, *valid_namespaces]
            if valid_namespaces
            else DEFAULT_NAMESPACES
        )
        self.additional_import_mappings = additional_import_mappings or {}
        self.import_mappings = (
            {
                **ALL_SERIALIZABLE_MAPPINGS,
                **self.additional_import_mappings,
            }
            if self.additional_import_mappings
            else ALL_SERIALIZABLE_MAPPINGS
        )
        # Compute allowed class paths:
        # 中文: 计算允许的类路径：
        # - "all" -> use default paths from mappings (+ additional_import_mappings)
        # 中文: -“全部”->使用映射中的默认路径（+additional_import_mappings）
        # - Explicit list -> compute from those classes
        # 中文: - 显式列表 -> 从这些类中计算
        if allowed_objects in ("all", "core"):
            self.allowed_class_paths: set[tuple[str, ...]] | None = (
                _get_default_allowed_class_paths(
                    cast("Literal['all', 'core']", allowed_objects)
                ).copy()
            )
            # Add paths from additional_import_mappings to the defaults
            # 中文: 将additional_import_mappings 中的路径添加到默认值
            if self.additional_import_mappings:
                for key, value in self.additional_import_mappings.items():
                    self.allowed_class_paths.add(key)
                    self.allowed_class_paths.add(value)
        else:
            self.allowed_class_paths = _compute_allowed_class_paths(
                cast("Iterable[AllowedObject]", allowed_objects), self.import_mappings
            )
        self.ignore_unserializable_fields = ignore_unserializable_fields
        self.init_validator = init_validator

    def __call__(self, value: dict[str, Any]) -> Any:
        """Revive the value.

        Args:
            value: The value to revive.

        Returns:
            The revived value.

        Raises:
            ValueError: If the namespace is invalid.
            ValueError: If trying to deserialize something that cannot
                be deserialized in the current version of langchain-core.
            NotImplementedError: If the object is not implemented and
                `ignore_unserializable_fields` is False.
        

        中文翻译:
        重振价值。
        参数：
            value：复活的值。
        返回：
            复活的价值。
        加薪：
            ValueError：如果命名空间无效。
            ValueError：如果尝试反序列化无法实现的内容
                在当前版本的 langchain-core 中反序列化。
            NotImplementedError：如果该对象未实现并且
                `ignore_unserialized_fields` 是 False。"""
        if (
            value.get("lc") == 1
            and value.get("type") == "secret"
            and value.get("id") is not None
        ):
            [key] = value["id"]
            if key in self.secrets_map:
                return self.secrets_map[key]
            if self.secrets_from_env and key in os.environ and os.environ[key]:
                return os.environ[key]
            return None

        if (
            value.get("lc") == 1
            and value.get("type") == "not_implemented"
            and value.get("id") is not None
        ):
            if self.ignore_unserializable_fields:
                return None
            msg = (
                "Trying to load an object that doesn't implement "
                f"serialization: {value}"
            )
            raise NotImplementedError(msg)

        if (
            value.get("lc") == 1
            and value.get("type") == "constructor"
            and value.get("id") is not None
        ):
            [*namespace, name] = value["id"]
            mapping_key = tuple(value["id"])

            if (
                self.allowed_class_paths is not None
                and mapping_key not in self.allowed_class_paths
            ):
                msg = (
                    f"Deserialization of {mapping_key!r} is not allowed. "
                    "The default (allowed_objects='core') only permits core "
                    "langchain-core classes. To allow trusted partner integrations, "
                    "use allowed_objects='all'. Alternatively, pass an explicit list "
                    "of allowed classes via allowed_objects=[...]. "
                    "See langchain_core.load.mapping for the full allowlist."
                )
                raise ValueError(msg)

            if (
                namespace[0] not in self.valid_namespaces
                # The root namespace ["langchain"] is not a valid identifier.
                # 中文: 根命名空间 [“langchain”] 不是有效的标识符。
                or namespace == ["langchain"]
            ):
                msg = f"Invalid namespace: {value}"
                raise ValueError(msg)
            # Determine explicit import path
            # 中文: 确定显式导入路径
            if mapping_key in self.import_mappings:
                import_path = self.import_mappings[mapping_key]
                # Split into module and name
                # 中文: 分为模块和名称
                import_dir, name = import_path[:-1], import_path[-1]
            elif namespace[0] in DISALLOW_LOAD_FROM_PATH:
                msg = (
                    "Trying to deserialize something that cannot "
                    "be deserialized in current version of langchain-core: "
                    f"{mapping_key}."
                )
                raise ValueError(msg)
            else:
                # Otherwise, treat namespace as path.
                # 中文: 否则，将命名空间视为路径。
                import_dir = namespace

            # Validate import path is in trusted namespaces before importing
            # 中文: 导入之前验证导入路径是否位于受信任的命名空间中
            if import_dir[0] not in self.valid_namespaces:
                msg = f"Invalid namespace: {value}"
                raise ValueError(msg)

            mod = importlib.import_module(".".join(import_dir))

            cls = getattr(mod, name)

            # The class must be a subclass of Serializable.
            # 中文: 该类必须是可序列化的子类。
            if not issubclass(cls, Serializable):
                msg = f"Invalid namespace: {value}"
                raise ValueError(msg)

            # We don't need to recurse on kwargs
            # 中文: 我们不需要递归 kwargs
            # as json.loads will do that for us.
            # 中文: 因为 json.loads 会为我们做到这一点。
            kwargs = value.get("kwargs", {})

            if self.init_validator is not None:
                self.init_validator(mapping_key, kwargs)

            return cls(**kwargs)

        return value


@beta()
def loads(
    text: str,
    *,
    allowed_objects: Iterable[AllowedObject] | Literal["all", "core"] = "core",
    secrets_map: dict[str, str] | None = None,
    valid_namespaces: list[str] | None = None,
    secrets_from_env: bool = False,
    additional_import_mappings: dict[tuple[str, ...], tuple[str, ...]] | None = None,
    ignore_unserializable_fields: bool = False,
    init_validator: InitValidator | None = default_init_validator,
) -> Any:
    """Revive a LangChain class from a JSON string.

    Equivalent to `load(json.loads(text))`.

    Only classes in the allowlist can be instantiated. The default allowlist includes
    core LangChain types (messages, prompts, documents, etc.). See
    `langchain_core.load.mapping` for the full list.

    !!! warning "Beta feature"

        This is a beta feature. Please be wary of deploying experimental code to
        production unless you've taken appropriate precautions.

    Args:
        text: The string to load.
        allowed_objects: Allowlist of classes that can be deserialized.

            - `'core'` (default): Allow classes defined in the serialization mappings
                for `langchain_core`.
            - `'all'`: Allow classes defined in the serialization mappings.

                This includes core LangChain types (messages, prompts, documents, etc.)
                and trusted partner integrations. See `langchain_core.load.mapping` for
                the full list.

            - Explicit list of classes: Only those specific classes are allowed.
            - `[]`: Disallow all deserialization (will raise on any object).
        secrets_map: A map of secrets to load.

            If a secret is not found in the map, it will be loaded from the environment
            if `secrets_from_env` is `True`.
        valid_namespaces: Additional namespaces (modules) to allow during
            deserialization, beyond the default trusted namespaces.
        secrets_from_env: Whether to load secrets from the environment.
        additional_import_mappings: A dictionary of additional namespace mappings.

            You can use this to override default mappings or add new mappings.

            When `allowed_objects` is `None` (using defaults), paths from these
            mappings are also added to the allowed class paths.
        ignore_unserializable_fields: Whether to ignore unserializable fields.
        init_validator: Optional callable to validate kwargs before instantiation.

            If provided, this function is called with `(class_path, kwargs)` where
            `class_path` is the class path tuple and `kwargs` is the kwargs dict.
            The validator should raise an exception if the object should not be
            deserialized, otherwise return `None`.

            Defaults to `default_init_validator` which blocks jinja2 templates.

    Returns:
        Revived LangChain objects.

    Raises:
        ValueError: If an object's class path is not in the `allowed_objects` allowlist.
    

    中文翻译:
    从 JSON 字符串恢复 LangChain 类。
    相当于“load(json.loads(text))”。
    只有允许列表中的类才能被实例化。默认允许列表包括
    LangChain核心类型（消息、提示、文档等）。参见
    完整列表请参见“langchain_core.load.mapping”。
    ！！！警告“测试版功能”
        这是测试版功能。请谨慎部署实验代码
        除非您采取了适当的预防措施。
    参数：
        text：要加载的字符串。
        allowed_objects：可以反序列化的类的白名单。
            - `'core'`（默认）：允许在序列化映射中定义的类
                对于“langchain_core”。
            - `'all'`：允许在序列化映射中定义的类。
                这包括核心LangChain类型（消息、提示、文档等）
                和值得信赖的合作伙伴集成。请参阅“langchain_core.load.mapping”
                完整列表。
            - 明确的类别列表：仅允许那些特定的类别。
            - `[]`：禁止所有反序列化（将在任何对象上引发）。
        Secrets_map：要加载的秘密地图。
            如果在地图中找不到秘密，则会从环境中加载它
            如果“secrets_from_env”为“True”。
        valid_namespaces：允许的附加命名空间（模块）
            反序列化，超出默认的受信任命名空间。
        Secrets_from_env：是否从环境中加载机密。
        extra_import_mappings：附加命名空间映射的字典。
            您可以使用它来覆盖默认映射或添加新映射。
            当“allowed_objects”为“None”（使用默认值）时，来自这些的路径
            映射也被添加到允许的类路径中。
        ignore_unserialized_fields：是否忽略不可序列化字段。
        init_validator：可选的可调用函数，用于在实例化之前验证 kwargs。
            如果提供，则使用“(class_path, kwargs)”调用此函数，其中
            `class_path` 是类路径元组，`kwargs` 是 kwargs 字典。
            如果对象不应该被验证，验证器应该引发异常
            反序列化，否则返回“None”。
            默认为“default_init_validator”，它会阻止 jinja2 模板。
    返回：
        恢复LangChain对象。
    加薪：
        ValueError：如果对象的类路径不在“allowed_objects”白名单中。"""
    # Parse JSON and delegate to load() for proper escape handling
    # 中文: 解析 JSON 并委托给 load() 以进行正确的转义处理
    raw_obj = json.loads(text)
    return load(
        raw_obj,
        allowed_objects=allowed_objects,
        secrets_map=secrets_map,
        valid_namespaces=valid_namespaces,
        secrets_from_env=secrets_from_env,
        additional_import_mappings=additional_import_mappings,
        ignore_unserializable_fields=ignore_unserializable_fields,
        init_validator=init_validator,
    )


@beta()
def load(
    obj: Any,
    *,
    allowed_objects: Iterable[AllowedObject] | Literal["all", "core"] = "core",
    secrets_map: dict[str, str] | None = None,
    valid_namespaces: list[str] | None = None,
    secrets_from_env: bool = False,
    additional_import_mappings: dict[tuple[str, ...], tuple[str, ...]] | None = None,
    ignore_unserializable_fields: bool = False,
    init_validator: InitValidator | None = default_init_validator,
) -> Any:
    """Revive a LangChain class from a JSON object.

    Use this if you already have a parsed JSON object, eg. from `json.load` or
    `orjson.loads`.

    Only classes in the allowlist can be instantiated. The default allowlist includes
    core LangChain types (messages, prompts, documents, etc.). See
    `langchain_core.load.mapping` for the full list.

    !!! warning "Beta feature"

        This is a beta feature. Please be wary of deploying experimental code to
        production unless you've taken appropriate precautions.

    Args:
        obj: The object to load.
        allowed_objects: Allowlist of classes that can be deserialized.

            - `'core'` (default): Allow classes defined in the serialization mappings
                for `langchain_core`.
            - `'all'`: Allow classes defined in the serialization mappings.

                This includes core LangChain types (messages, prompts, documents, etc.)
                and trusted partner integrations. See `langchain_core.load.mapping` for
                the full list.

            - Explicit list of classes: Only those specific classes are allowed.
            - `[]`: Disallow all deserialization (will raise on any object).
        secrets_map: A map of secrets to load.

            If a secret is not found in the map, it will be loaded from the environment
            if `secrets_from_env` is `True`.
        valid_namespaces: Additional namespaces (modules) to allow during
            deserialization, beyond the default trusted namespaces.
        secrets_from_env: Whether to load secrets from the environment.
        additional_import_mappings: A dictionary of additional namespace mappings.

            You can use this to override default mappings or add new mappings.

            When `allowed_objects` is `None` (using defaults), paths from these
            mappings are also added to the allowed class paths.
        ignore_unserializable_fields: Whether to ignore unserializable fields.
        init_validator: Optional callable to validate kwargs before instantiation.

            If provided, this function is called with `(class_path, kwargs)` where
            `class_path` is the class path tuple and `kwargs` is the kwargs dict.
            The validator should raise an exception if the object should not be
            deserialized, otherwise return `None`.

            Defaults to `default_init_validator` which blocks jinja2 templates.

    Returns:
        Revived LangChain objects.

    Raises:
        ValueError: If an object's class path is not in the `allowed_objects` allowlist.

    Example:
        ```python
        from langchain_core.load import load, dumpd
        from langchain_core.messages import AIMessage

        msg = AIMessage(content="Hello")
        data = dumpd(msg)

        # Deserialize using default allowlist
        # 中文: 使用默认白名单反序列化
        loaded = load(data)

        # Or with explicit allowlist
        # 中文: 或者使用明确的白名单
        loaded = load(data, allowed_objects=[AIMessage])

        # Or extend defaults with additional mappings
        # 中文: 或者使用附加映射扩展默认值
        loaded = load(
            data,
            additional_import_mappings={
                ("my_pkg", "MyClass"): ("my_pkg", "module", "MyClass"),
            },
        )
        ```
    

    中文翻译:
    从 JSON 对象恢复 LangChain 类。
    如果您已经有一个已解析的 JSON 对象，请使用此选项，例如。来自 `json.load` 或
    `orjson.loads`。
    只有允许列表中的类才能被实例化。默认允许列表包括
    LangChain核心类型（消息、提示、文档等）。参见
    完整列表请参见“langchain_core.load.mapping”。
    !!!警告“测试版功能”
        这是测试版功能。请谨慎部署实验代码
        除非您采取了适当的预防措施。
    参数：
        obj：要加载的对象。
        allowed_objects：可以反序列化的类的白名单。
            - `'core'`（默认）：允许在序列化映射中定义的类
                对于“langchain_core”。
            - `'all'`：允许在序列化映射中定义的类。
                这包括核心LangChain类型（消息、提示、文档等）
                和值得信赖的合作伙伴集成。请参阅“langchain_core.load.mapping”
                完整列表。
            - 明确的类别列表：仅允许那些特定的类别。
            - `[]`：禁止所有反序列化（将在任何对象上引发）。
        Secrets_map：要加载的秘密地图。
            如果在地图中找不到秘密，则会从环境中加载它
            如果“secrets_from_env”为“True”。
        valid_namespaces：允许的附加命名空间（模块）
            反序列化，超出默认的受信任命名空间。
        Secrets_from_env：是否从环境中加载机密。
        extra_import_mappings：附加命名空间映射的字典。
            您可以使用它来覆盖默认映射或添加新映射。
            当“allowed_objects”为“None”（使用默认值）时，来自这些的路径
            映射也被添加到允许的类路径中。
        ignore_unserialized_fields：是否忽略不可序列化字段。
        init_validator：可选的可调用函数，用于在实例化之前验证 kwargs。
            如果提供，则使用“(class_path, kwargs)”调用此函数，其中
            `class_path` 是类路径元组，`kwargs` 是 kwargs 字典。
            如果对象不应该被验证，验证器应该引发异常
            反序列化，否则返回“None”。
            默认为“default_init_validator”，它会阻止 jinja2 模板。
    返回：
        恢复LangChain对象。
    加薪：
        ValueError：如果对象的类路径不在“allowed_objects”白名单中。
    示例：
        ````蟒蛇
        从 langchain_core.load 导入加载，转储
        从 langchain_core.messages 导入 AIMessage
        msg = AIMessage(内容=“你好”)
        数据 = dumpd(消息)
        # 使用默认白名单反序列化
        已加载=加载（数据）
        # 或者使用明确的白名单
        已加载=加载（数据，allowed_objects = [AIMessage]）
        # 或者使用额外的映射来扩展默认值
        已加载 = 加载（
            数据，
            额外的导入映射={
                （“my_pkg”，“MyClass”）：（“my_pkg”，“模块”，“MyClass”），
            },
        ）
        ````"""
    reviver = Reviver(
        allowed_objects,
        secrets_map,
        valid_namespaces,
        secrets_from_env,
        additional_import_mappings,
        ignore_unserializable_fields=ignore_unserializable_fields,
        init_validator=init_validator,
    )

    def _load(obj: Any) -> Any:
        if isinstance(obj, dict):
            # Check for escaped dict FIRST (before recursing).
            # 中文: 首先检查转义字典（在递归之前）。
            # Escaped dicts are user data that should NOT be processed as LC objects.
            # 中文: 转义字典是不应作为 LC 对象处理的用户数据。
            if _is_escaped_dict(obj):
                return _unescape_value(obj)

            # Not escaped - recurse into children then apply reviver
            # 中文: 未逃脱 - 递归到子级然后应用 reviver
            loaded_obj = {k: _load(v) for k, v in obj.items()}
            return reviver(loaded_obj)
        if isinstance(obj, list):
            return [_load(o) for o in obj]
        return obj

    return _load(obj)
