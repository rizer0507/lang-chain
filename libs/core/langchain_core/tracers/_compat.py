"""Compatibility helpers for Pydantic v1/v2 with langsmith Run objects.

Note: The generic helpers (`pydantic_to_dict`, `pydantic_copy`) detect Pydantic
version based on the langsmith `Run` model. They're intended for langsmith objects
(`Run`, `Example`) which migrate together.

For general Pydantic v1/v2 handling, see `langchain_core.utils.pydantic`.

中文翻译:
Pydantic v1/v2 与 langsmith Run 对象的兼容性帮助程序。
注意：通用助手（`pydantic_to_dict`、`pydantic_copy`）检测 Pydantic
基于 langsmith `Run` 模型的版本。它们适用于朗史密斯物品
（“运行”、“示例”）一起迁移。
对于一般 Pydantic v1/v2 处理，请参阅“langchain_core.utils.pydantic”。
"""

from __future__ import annotations

from typing import Any, TypeVar

from langchain_core.tracers.schemas import Run

# Detect Pydantic version once at import time based on Run model
# 中文: 基于运行模型在导入时检测 Pydantic 版本一次
_RUN_IS_PYDANTIC_V2 = hasattr(Run, "model_dump")

T = TypeVar("T")


def run_to_dict(run: Run, **kwargs: Any) -> dict[str, Any]:
    """Convert run to dict, compatible with both Pydantic v1 and v2.

    Args:
        run: The run to convert.
        **kwargs: Additional arguments passed to model_dump/dict.

    Returns:
        Dictionary representation of the run.
    

    中文翻译:
    将 run 转换为 dict，兼容 Pydantic v1 和 v2。
    参数：
        run：要转换的运行。
        **kwargs：传递给 model_dump/dict 的附加参数。
    返回：
        运行的字典表示。"""
    if _RUN_IS_PYDANTIC_V2:
        return run.model_dump(**kwargs)
    return run.dict(**kwargs)  # type: ignore[deprecated]


def run_copy(run: Run, **kwargs: Any) -> Run:
    """Copy run, compatible with both Pydantic v1 and v2.

    Args:
        run: The run to copy.
        **kwargs: Additional arguments passed to model_copy/copy.

    Returns:
        A copy of the run.
    

    中文翻译:
    复制运行，兼容 Pydantic v1 和 v2。
    参数：
        运行：要复制的运行。
        **kwargs：传递给 model_copy/copy 的附加参数。
    返回：
        运行的副本。"""
    if _RUN_IS_PYDANTIC_V2:
        return run.model_copy(**kwargs)
    return run.copy(**kwargs)  # type: ignore[deprecated]


def run_construct(**kwargs: Any) -> Run:
    """Construct run without validation, compatible with both Pydantic v1 and v2.

    Args:
        **kwargs: Fields to set on the run.

    Returns:
        A new Run instance constructed without validation.
    

    中文翻译:
    无需验证即可构建运行，与 Pydantic v1 和 v2 兼容。
    参数：
        **kwargs：运行时设置的字段。
    返回：
        未经验证而构造的新 Run 实例。"""
    if _RUN_IS_PYDANTIC_V2:
        return Run.model_construct(**kwargs)
    return Run.construct(**kwargs)  # type: ignore[deprecated]


def pydantic_to_dict(obj: Any, **kwargs: Any) -> dict[str, Any]:
    """Convert any Pydantic model to dict, compatible with both v1 and v2.

    Args:
        obj: The Pydantic model to convert.
        **kwargs: Additional arguments passed to model_dump/dict.

    Returns:
        Dictionary representation of the model.
    

    中文翻译:
    将任何 Pydantic 模型转换为 dict，兼容 v1 和 v2。
    参数：
        obj：要转换的 Pydantic 模型。
        **kwargs：传递给 model_dump/dict 的附加参数。
    返回：
        模型的字典表示。"""
    if _RUN_IS_PYDANTIC_V2:
        return obj.model_dump(**kwargs)  # type: ignore[no-any-return]
    return obj.dict(**kwargs)  # type: ignore[no-any-return]


def pydantic_copy(obj: T, **kwargs: Any) -> T:
    """Copy any Pydantic model, compatible with both v1 and v2.

    Args:
        obj: The Pydantic model to copy.
        **kwargs: Additional arguments passed to model_copy/copy.

    Returns:
        A copy of the model.
    

    中文翻译:
    复制任何 Pydantic 模型，与 v1 和 v2 兼容。
    参数：
        obj：要复制的 Pydantic 模型。
        **kwargs：传递给 model_copy/copy 的附加参数。
    返回：
        模型的副本。"""
    if _RUN_IS_PYDANTIC_V2:
        return obj.model_copy(**kwargs)  # type: ignore[attr-defined,no-any-return]
    return obj.copy(**kwargs)  # type: ignore[attr-defined,no-any-return]
