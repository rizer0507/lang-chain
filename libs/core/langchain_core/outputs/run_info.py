"""RunInfo class.

中文翻译:
运行信息类。"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel


class RunInfo(BaseModel):
    """Class that contains metadata for a single execution of a Chain or model.

    Defined for backwards compatibility with older versions of langchain_core.

    This model will likely be deprecated in the future.

    Users can acquire the run_id information from callbacks or via run_id
    information present in the astream_event API (depending on the use case).
    

    中文翻译:
    包含链或模型单次执行的元数据的类。
    定义用于向后兼容旧版本的 langchain_core。
    该模型将来可能会被弃用。
    用户可以通过回调或者run_id获取run_id信息
    astream_event API 中存在的信息（取决于用例）。"""

    run_id: UUID
    """A unique identifier for the model or chain run.

    中文翻译:
    模型或链运行的唯一标识符。"""
