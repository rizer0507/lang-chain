"""Internal tracers used for stream_log and astream events implementations.

中文翻译:
用于stream_log 和astream 事件实现的内部跟踪器。"""

import typing
from collections.abc import AsyncIterator, Iterator
from uuid import UUID

T = typing.TypeVar("T")


# THIS IS USED IN LANGGRAPH.
# 中文: 这在 LANGRAPH 中使用。
@typing.runtime_checkable
class _StreamingCallbackHandler(typing.Protocol[T]):
    """Types for streaming callback handlers.

    This is a common mixin that the callback handlers
    for both astream events and astream log inherit from.

    The `tap_output_aiter` method is invoked in some contexts
    to produce callbacks for intermediate results.
    

    中文翻译:
    流回调处理程序的类型。
    这是回调处理程序的常见混入
    对于 astream 事件和 astream 日志都继承自。
    在某些上下文中会调用 `tap_output_aiter` 方法
    为中间结果生成回调。"""

    def tap_output_aiter(
        self, run_id: UUID, output: AsyncIterator[T]
    ) -> AsyncIterator[T]:
        """Used for internal astream_log and astream events implementations.

        中文翻译:
        用于内部 astream_log 和 astream 事件实现。"""

    def tap_output_iter(self, run_id: UUID, output: Iterator[T]) -> Iterator[T]:
        """Used for internal astream_log and astream events implementations.

        中文翻译:
        用于内部 astream_log 和 astream 事件实现。"""


__all__ = [
    "_StreamingCallbackHandler",
]
