"""Fake LLMs for testing purposes.

中文翻译:
用于测试目的的假法学硕士。"""

import asyncio
import time
from collections.abc import AsyncIterator, Iterator, Mapping
from typing import Any

from typing_extensions import override

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.llms import LLM
from langchain_core.runnables import RunnableConfig


class FakeListLLM(LLM):
    """Fake LLM for testing purposes.

    中文翻译:
    用于测试目的的假 LLM。"""

    responses: list[str]
    """List of responses to return in order.

    中文翻译:
    按顺序返回的响应列表。"""
    # This parameter should be removed from FakeListLLM since
    # 中文: 应从 FakeListLLM 中删除此参数，因为
    # it's only used by sub-classes.
    # 中文: 它仅由子类使用。
    sleep: float | None = None
    """Sleep time in seconds between responses.

    Ignored by FakeListLLM, but used by sub-classes.
    

    中文翻译:
    响应之间的睡眠时间（以秒为单位）。
    被 FakeListLLM 忽略，但被子类使用。"""
    i: int = 0
    """Internally incremented after every model invocation.

    Useful primarily for testing purposes.
    

    中文翻译:
    每次模型调用后内部递增。
    主要用于测试目的。"""

    @property
    @override
    def _llm_type(self) -> str:
        """Return type of llm.

        中文翻译:
        返回类型为 llm。"""
        return "fake-list"

    @override
    def _call(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        """Return next response.

        中文翻译:
        返回下一个响应。"""
        response = self.responses[self.i]
        if self.i < len(self.responses) - 1:
            self.i += 1
        else:
            self.i = 0
        return response

    @override
    async def _acall(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        """Return next response.

        中文翻译:
        返回下一个响应。"""
        response = self.responses[self.i]
        if self.i < len(self.responses) - 1:
            self.i += 1
        else:
            self.i = 0
        return response

    @property
    @override
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"responses": self.responses}


class FakeListLLMError(Exception):
    """Fake error for testing purposes.

    中文翻译:
    用于测试目的的假错误。"""


class FakeStreamingListLLM(FakeListLLM):
    """Fake streaming list LLM for testing purposes.

    An LLM that will return responses from a list in order.

    This model also supports optionally sleeping between successive
    chunks in a streaming implementation.
    

    中文翻译:
    用于测试目的的假流媒体列表 LLM。
    法学硕士将按顺序返回列表中的响应。
    该模型还支持在连续的之间选择性地睡眠
    流实现中的块。"""

    error_on_chunk_number: int | None = None
    """If set, will raise an exception on the specified chunk number.

    中文翻译:
    如果设置，将引发指定块编号的异常。"""

    @override
    def stream(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        result = self.invoke(input, config)
        for i_c, c in enumerate(result):
            if self.sleep is not None:
                time.sleep(self.sleep)

            if (
                self.error_on_chunk_number is not None
                and i_c == self.error_on_chunk_number
            ):
                raise FakeListLLMError
            yield c

    @override
    async def astream(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        result = await self.ainvoke(input, config)
        for i_c, c in enumerate(result):
            if self.sleep is not None:
                await asyncio.sleep(self.sleep)

            if (
                self.error_on_chunk_number is not None
                and i_c == self.error_on_chunk_number
            ):
                raise FakeListLLMError
            yield c
