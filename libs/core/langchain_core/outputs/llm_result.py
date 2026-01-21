"""LLMResult class.

中文翻译:
LLMResult 类。"""

from __future__ import annotations

from copy import deepcopy
from typing import Literal

from pydantic import BaseModel

from langchain_core.outputs.chat_generation import ChatGeneration, ChatGenerationChunk
from langchain_core.outputs.generation import Generation, GenerationChunk
from langchain_core.outputs.run_info import RunInfo


class LLMResult(BaseModel):
    """A container for results of an LLM call.

    Both chat models and LLMs generate an LLMResult object. This object contains the
    generated outputs and any additional information that the model provider wants to
    return.
    

    中文翻译:
    LLM 调用结果的容器。
    聊天模型和 LLM 都会生成 LLMResult 对象。该对象包含
    生成的输出以及模型提供者想要的任何附加信息
    返回。"""

    generations: list[
        list[Generation | ChatGeneration | GenerationChunk | ChatGenerationChunk]
    ]
    """Generated outputs.

    The first dimension of the list represents completions for different input prompts.

    The second dimension of the list represents different candidate generations for a
    given prompt.

    - When returned from **an LLM**, the type is `list[list[Generation]]`.
    - When returned from a **chat model**, the type is `list[list[ChatGeneration]]`.

    ChatGeneration is a subclass of Generation that has a field for a structured chat
    message.
    

    中文翻译:
    生成的输出。
    列表的第一维表示不同输入提示的完成。
    列表的第二个维度代表不同的候选代
    给出提示。
    - 当从 **LLM** 返回时，类型为 `list[list[Generation]]`。
    - 当从 **聊天模型** 返回时，类型为 `list[list[ChatGeneration]]`。
    ChatGeneration 是 Generation 的子类，具有用于结构化聊天的字段
    消息。"""
    llm_output: dict | None = None
    """For arbitrary LLM provider specific output.

    This dictionary is a free-form dictionary that can contain any information that the
    provider wants to return. It is not standardized and is provider-specific.

    Users should generally avoid relying on this field and instead rely on accessing
    relevant information from standardized fields present in AIMessage.
    

    中文翻译:
    对于任意 LLM 提供商的特定输出。
    该字典是一个自由格式的字典，可以包含该字典中的任何信息。
    提供者想要返回。它不是标准化的，并且是特定于提供商的。
    用户通常应避免依赖此字段，而应依赖访问
    AIMessage 中标准化字段的相关信息。"""
    run: list[RunInfo] | None = None
    """List of metadata info for model call for each input.

    See `langchain_core.outputs.run_info.RunInfo` for details.
    

    中文翻译:
    每个输入的模型调用的元数据信息列表。
    有关详细信息，请参阅“langchain_core.outputs.run_info.RunInfo”。"""

    type: Literal["LLMResult"] = "LLMResult"
    """Type is used exclusively for serialization purposes.

    中文翻译:
    类型专门用于序列化目的。"""

    def flatten(self) -> list[LLMResult]:
        """Flatten generations into a single list.

        Unpack list[list[Generation]] -> list[LLMResult] where each returned LLMResult
        contains only a single Generation. If token usage information is available,
        it is kept only for the LLMResult corresponding to the top-choice
        Generation, to avoid over-counting of token usage downstream.

        Returns:
            List of LLMResults where each returned LLMResult contains a single
                Generation.
        

        中文翻译:
        将几代人扁平化为一个列表。
        解压 list[list[Generation]] -> list[LLMResult] 其中每个返回 LLMResult
        仅包含一个世代。如果令牌使用信息可用，
        它仅为与首选选项对应的 LLMResult 保留
        生成，以避免下游代币使用的过度计数。
        返回：
            LLMResults 列表，其中每个返回的 LLMResult 包含一个
                一代。"""
        llm_results = []
        for i, gen_list in enumerate(self.generations):
            # Avoid double counting tokens in OpenAICallback
            # 中文: 避免 OpenAICallback 中重复计算令牌
            if i == 0:
                llm_results.append(
                    LLMResult(
                        generations=[gen_list],
                        llm_output=self.llm_output,
                    )
                )
            else:
                if self.llm_output is not None:
                    llm_output = deepcopy(self.llm_output)
                    llm_output["token_usage"] = {}
                else:
                    llm_output = None
                llm_results.append(
                    LLMResult(
                        generations=[gen_list],
                        llm_output=llm_output,
                    )
                )
        return llm_results

    def __eq__(self, other: object) -> bool:
        """Check for `LLMResult` equality by ignoring any metadata related to runs.

        Args:
            other: Another `LLMResult` object to compare against.

        Returns:
            `True` if the generations and `llm_output` are equal, `False` otherwise.
        

        中文翻译:
        通过忽略与运行相关的任何元数据来检查“LLMResult”是否相等。
        参数：
            other：要比较的另一个“LLMResult”对象。
        返回：
            如果世代和 llm_output 相等，则为“True”，否则为“False”。"""
        if not isinstance(other, LLMResult):
            return NotImplemented
        return (
            self.generations == other.generations
            and self.llm_output == other.llm_output
        )

    __hash__ = None  # type: ignore[assignment]
