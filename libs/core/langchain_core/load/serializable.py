"""Serializable base class.

中文翻译:
可序列化的基类。"""

import contextlib
import logging
from abc import ABC
from typing import (
    Any,
    Literal,
    TypedDict,
    cast,
)

from pydantic import BaseModel, ConfigDict
from pydantic.fields import FieldInfo
from typing_extensions import NotRequired, override

logger = logging.getLogger(__name__)


class BaseSerialized(TypedDict):
    """Base class for serialized objects.

    中文翻译:
    序列化对象的基类。"""

    lc: int
    """The version of the serialization format.

    中文翻译:
    序列化格式的版本。"""
    id: list[str]
    """The unique identifier of the object.

    中文翻译:
    对象的唯一标识符。"""
    name: NotRequired[str]
    """The name of the object.

    中文翻译:
    对象的名称。"""
    graph: NotRequired[dict[str, Any]]
    """The graph of the object.

    中文翻译:
    对象的图形。"""


class SerializedConstructor(BaseSerialized):
    """Serialized constructor.

    中文翻译:
    序列化构造函数。"""

    type: Literal["constructor"]
    """The type of the object. Must be `'constructor'`.

    中文翻译:
    对象的类型。必须是“构造函数”。"""
    kwargs: dict[str, Any]
    """The constructor arguments.

    中文翻译:
    构造函数参数。"""


class SerializedSecret(BaseSerialized):
    """Serialized secret.

    中文翻译:
    连载秘密。"""

    type: Literal["secret"]
    """The type of the object. Must be `'secret'`.

    中文翻译:
    对象的类型。一定是“秘密”。"""


class SerializedNotImplemented(BaseSerialized):
    """Serialized not implemented.

    中文翻译:
    系列化未实施。"""

    type: Literal["not_implemented"]
    """The type of the object. Must be `'not_implemented'`.

    中文翻译:
    对象的类型。必须是“未实现”。"""
    repr: str | None
    """The representation of the object.

    中文翻译:
    对象的表示。"""


def try_neq_default(value: Any, key: str, model: BaseModel) -> bool:
    """Try to determine if a value is different from the default.

    Args:
        value: The value.
        key: The key.
        model: The Pydantic model.

    Returns:
        Whether the value is different from the default.
    

    中文翻译:
    尝试确定值是否与默认值不同。
    参数：
        价值：价值。
        钥匙：钥匙。
        模型：Pydantic 模型。
    返回：
        该值是否与默认值不同。"""
    field = type(model).model_fields[key]
    return _try_neq_default(value, field)


def _try_neq_default(value: Any, field: FieldInfo) -> bool:
    # Handle edge case: inequality of two objects does not evaluate to a bool (e.g. two
    # 中文: 处理边缘情况：两个对象的不等式不会计算为布尔值（例如两个
    # Pandas DataFrames).
    # 中文: 熊猫数据框）。
    try:
        return bool(field.get_default() != value)
    except Exception as _:
        try:
            return all(field.get_default() != value)
        except Exception as _:
            try:
                return value is not field.default
            except Exception as _:
                return False


class Serializable(BaseModel, ABC):
    """Serializable base class.

    This class is used to serialize objects to JSON.

    It relies on the following methods and properties:

    - [`is_lc_serializable`][langchain_core.load.serializable.Serializable.is_lc_serializable]: Is this class serializable?

        By design, even if a class inherits from `Serializable`, it is not serializable
        by default. This is to prevent accidental serialization of objects that should
        not be serialized.
    - [`get_lc_namespace`][langchain_core.load.serializable.Serializable.get_lc_namespace]: Get the namespace of the LangChain object.

        During deserialization, this namespace is used to identify
        the correct class to instantiate.

        Please see the `Reviver` class in `langchain_core.load.load` for more details.
        During deserialization an additional mapping is handle classes that have moved
        or been renamed across package versions.

    - [`lc_secrets`][langchain_core.load.serializable.Serializable.lc_secrets]: A map of constructor argument names to secret ids.
    - [`lc_attributes`][langchain_core.load.serializable.Serializable.lc_attributes]: List of additional attribute names that should be included
        as part of the serialized representation.
    

    中文翻译:
    可序列化的基类。
    此类用于将对象序列化为 JSON。
    它依赖于以下方法和属性：
    - [`is_lc_serialized`][langchain_core.load.serializable.Serializable.is_lc_serializable]：这个类可以序列化吗？
        根据设计，即使一个类继承自“Serialized”，它也不是可序列化的
        默认情况下。这是为了防止意外序列化应该
        不被序列化。
    - [`get_lc_namespace`][langchain_core.load.serialized.Serialized.get_lc_namespace]：获取LangChain对象的命名空间。
        在反序列化过程中，该命名空间用于标识
        要实例化的正确类。
        请参阅“langchain_core.load.load”中的“Reviver”类了解更多详细信息。
        在反序列化期间，附加映射是已移动的句柄类
        或跨软件包版本重命名。
    - [`lc_secrets`][langchain_core.load.serializable.Serialized.lc_secrets]：构造函数参数名称到秘密 ID 的映射。
    - [`lc_attributes`][langchain_core.load.serializable.Serialized.lc_attributes]：应包含的其他属性名称列表
        作为序列化表示的一部分。"""  # noqa: E501

    # Remove default BaseModel init docstring.
    # 中文: 删除默认的 BaseModel 初始化文档字符串。
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """"""  # noqa: D419  # Intentional blank docstring
        super().__init__(*args, **kwargs)

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Is this class serializable?

        By design, even if a class inherits from `Serializable`, it is not serializable
        by default. This is to prevent accidental serialization of objects that should
        not be serialized.

        Returns:
            Whether the class is serializable. Default is `False`.
        

        中文翻译:
        这个类可以序列化吗？
        根据设计，即使一个类继承自“Serialized”，它也不是可序列化的
        默认情况下。这是为了防止意外序列化应该
        不被序列化。
        返回：
            类是否可序列化。默认为“假”。"""
        return False

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the LangChain object.

        For example, if the class is [`langchain.llms.openai.OpenAI`][langchain_openai.OpenAI],
        then the namespace is `["langchain", "llms", "openai"]`

        Returns:
            The namespace.
        

        中文翻译:
        获取LangChain对象的命名空间。
        例如，如果类是 [`langchain.llms.openai.OpenAI`][langchain_openai.OpenAI]，
        那么命名空间是 `["langchain", "llms", "openai"]`
        返回：
            命名空间。"""  # noqa: E501
        return cls.__module__.split(".")

    @property
    def lc_secrets(self) -> dict[str, str]:
        """A map of constructor argument names to secret ids.

        For example, `{"openai_api_key": "OPENAI_API_KEY"}`
        

        中文翻译:
        构造函数参数名称到秘密 ID 的映射。
        例如，`{"openai_api_key": "OPENAI_API_KEY"}`"""
        return {}

    @property
    def lc_attributes(self) -> dict:
        """List of attribute names that should be included in the serialized kwargs.

        These attributes must be accepted by the constructor.

        Default is an empty dictionary.
        

        中文翻译:
        应包含在序列化 kwargs 中的属性名称列表。
        这些属性必须被构造函数接受。
        默认是一个空字典。"""
        return {}

    @classmethod
    def lc_id(cls) -> list[str]:
        """Return a unique identifier for this class for serialization purposes.

        The unique identifier is a list of strings that describes the path
        to the object.

        For example, for the class `langchain.llms.openai.OpenAI`, the id is
        `["langchain", "llms", "openai", "OpenAI"]`.
        

        中文翻译:
        返回此类的唯一标识符以用于序列化目的。
        唯一标识符是描述路径的字符串列表
        到对象。
        例如，对于“langchain.llms.openai.OpenAI”类，id 为
        `[“langchain”、“llms”、“openai”、“OpenAI”]`。"""
        # Pydantic generics change the class name. So we need to do the following
        # 中文: Pydantic 泛型更改了类名。所以我们需要做以下事情
        if (
            "origin" in cls.__pydantic_generic_metadata__
            and cls.__pydantic_generic_metadata__["origin"] is not None
        ):
            original_name = cls.__pydantic_generic_metadata__["origin"].__name__
        else:
            original_name = cls.__name__
        return [*cls.get_lc_namespace(), original_name]

    model_config = ConfigDict(
        extra="ignore",
    )

    @override
    def __repr_args__(self) -> Any:
        return [
            (k, v)
            for k, v in super().__repr_args__()
            if (k not in type(self).model_fields or try_neq_default(v, k, self))
        ]

    def to_json(self) -> SerializedConstructor | SerializedNotImplemented:
        """Serialize the object to JSON.

        Raises:
            ValueError: If the class has deprecated attributes.

        Returns:
            A JSON serializable object or a `SerializedNotImplemented` object.
        

        中文翻译:
        将对象序列化为 JSON。
        加薪：
            ValueError：如果类具有已弃用的属性。
        返回：
            JSON 可序列化对象或“SerializedNotImplemented”对象。"""
        if not self.is_lc_serializable():
            return self.to_json_not_implemented()

        model_fields = type(self).model_fields
        secrets = {}
        # Get latest values for kwargs if there is an attribute with same name
        # 中文: 如果存在同名属性，则获取 kwargs 的最新值
        lc_kwargs = {}
        for k, v in self:
            if not _is_field_useful(self, k, v):
                continue
            # Do nothing if the field is excluded
            # 中文: 如果该字段被排除，则不执行任何操作
            if k in model_fields and model_fields[k].exclude:
                continue

            lc_kwargs[k] = getattr(self, k, v)

        # Merge the lc_secrets and lc_attributes from every class in the MRO
        # 中文: 合并 MRO 中每个类的 lc_secrets 和 lc_attributes
        for cls in [None, *self.__class__.mro()]:
            # Once we get to Serializable, we're done
            # 中文: 一旦我们达到可序列化，我们就完成了
            if cls is Serializable:
                break

            if cls:
                deprecated_attributes = [
                    "lc_namespace",
                    "lc_serializable",
                ]

                for attr in deprecated_attributes:
                    if hasattr(cls, attr):
                        msg = (
                            f"Class {self.__class__} has a deprecated "
                            f"attribute {attr}. Please use the corresponding "
                            f"classmethod instead."
                        )
                        raise ValueError(msg)

            # Get a reference to self bound to each class in the MRO
            # 中文: 获取对 MRO 中每个类的自绑定的引用
            this = cast("Serializable", self if cls is None else super(cls, self))

            secrets.update(this.lc_secrets)
            # Now also add the aliases for the secrets
            # 中文: 现在还要添加秘密的别名
            # This ensures known secret aliases are hidden.
            # 中文: 这可确保隐藏已知的秘密别名。
            # Note: this does NOT hide any other extra kwargs
            # 中文: 注意：这不会隐藏任何其他额外的 kwargs
            # that are not present in the fields.
            # 中文: 字段中不存在的内容。
            for key in list(secrets):
                value = secrets[key]
                if (key in model_fields) and (
                    alias := model_fields[key].alias
                ) is not None:
                    secrets[alias] = value
            lc_kwargs.update(this.lc_attributes)

        # include all secrets, even if not specified in kwargs
        # 中文: 包括所有秘密，即使 kwargs 中未指定
        # as these secrets may be passed as an environment variable instead
        # 中文: 因为这些秘密可以作为环境变量传递
        for key in secrets:
            secret_value = getattr(self, key, None) or lc_kwargs.get(key)
            if secret_value is not None:
                lc_kwargs.update({key: secret_value})

        return {
            "lc": 1,
            "type": "constructor",
            "id": self.lc_id(),
            "kwargs": lc_kwargs
            if not secrets
            else _replace_secrets(lc_kwargs, secrets),
        }

    def to_json_not_implemented(self) -> SerializedNotImplemented:
        """Serialize a "not implemented" object.

        Returns:
            `SerializedNotImplemented`.
        

        中文翻译:
        序列化“未实现”的对象。
        返回：
            `序列化未实现`。"""
        return to_json_not_implemented(self)


def _is_field_useful(inst: Serializable, key: str, value: Any) -> bool:
    """Check if a field is useful as a constructor argument.

    Args:
        inst: The instance.
        key: The key.
        value: The value.

    Returns:
        Whether the field is useful. If the field is required, it is useful.
        If the field is not required, it is useful if the value is not `None`.
        If the field is not required and the value is `None`, it is useful if the
        default value is different from the value.
    

    中文翻译:
    检查某个字段是否可用作构造函数参数。
    参数：
        inst：实例。
        钥匙：钥匙。
        价值：价值。
    返回：
        该字段是否有用。如果需要该字段，它很有用。
        如果该字段不是必需的，则该值不是“None”时很有用。
        如果该字段不是必需的并且值为“None”，则如果
        默认值与实际值不同。"""
    field = type(inst).model_fields.get(key)
    if not field:
        return False

    if field.is_required():
        return True

    # Handle edge case: a value cannot be converted to a boolean (e.g. a
    # 中文: 处理边缘情况：值无法转换为布尔值（例如
    # Pandas DataFrame).
    # 中文: 熊猫数据框）。
    try:
        value_is_truthy = bool(value)
    except Exception as _:
        value_is_truthy = False

    if value_is_truthy:
        return True

    # Value is still falsy here!
    # 中文: 这里的值仍然是假的！
    if field.default_factory is dict and isinstance(value, dict):
        return False

    # Value is still falsy here!
    # 中文: 这里的值仍然是假的！
    if field.default_factory is list and isinstance(value, list):
        return False

    value_neq_default = _try_neq_default(value, field)

    # If value is falsy and does not match the default
    # 中文: 如果值是假的并且与默认值不匹配
    return value_is_truthy or value_neq_default


def _replace_secrets(
    root: dict[Any, Any], secrets_map: dict[str, str]
) -> dict[Any, Any]:
    result = root.copy()
    for path, secret_id in secrets_map.items():
        [*parts, last] = path.split(".")
        current = result
        for part in parts:
            if part not in current:
                break
            current[part] = current[part].copy()
            current = current[part]
        if last in current:
            current[last] = {
                "lc": 1,
                "type": "secret",
                "id": [secret_id],
            }
    return result


def to_json_not_implemented(obj: object) -> SerializedNotImplemented:
    """Serialize a "not implemented" object.

    Args:
        obj: Object to serialize.

    Returns:
        `SerializedNotImplemented`
    

    中文翻译:
    序列化“未实现”的对象。
    参数：
        obj：要序列化的对象。
    返回：
        `序列化未实现`"""
    id_: list[str] = []
    try:
        if hasattr(obj, "__name__"):
            id_ = [*obj.__module__.split("."), obj.__name__]
        elif hasattr(obj, "__class__"):
            id_ = [*obj.__class__.__module__.split("."), obj.__class__.__name__]
    except Exception:
        logger.debug("Failed to serialize object", exc_info=True)

    result: SerializedNotImplemented = {
        "lc": 1,
        "type": "not_implemented",
        "id": id_,
        "repr": None,
    }
    with contextlib.suppress(Exception):
        result["repr"] = repr(obj)
    return result
