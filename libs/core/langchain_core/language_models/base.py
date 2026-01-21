"""Base language models class.

中文翻译:
基础语言模型类。"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from functools import cache
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    TypeAlias,
    TypeVar,
    cast,
)

from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing_extensions import TypedDict, override

from langchain_core.caches import BaseCache  # noqa: TC001
from langchain_core.callbacks import Callbacks  # noqa: TC001
from langchain_core.globals import get_verbose
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    MessageLikeRepresentation,
    get_buffer_string,
)
from langchain_core.prompt_values import (
    ChatPromptValueConcrete,
    PromptValue,
    StringPromptValue,
)
from langchain_core.runnables import Runnable, RunnableSerializable

if TYPE_CHECKING:
    from langchain_core.outputs import LLMResult

try:
    from transformers import GPT2TokenizerFast  # type: ignore[import-not-found]

    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False


class LangSmithParams(TypedDict, total=False):
    """LangSmith parameters for tracing.

    中文翻译:
    用于跟踪的 LangSmith 参数。"""

    ls_provider: str
    """Provider of the model.

    中文翻译:
    模型的提供者。"""
    ls_model_name: str
    """Name of the model.

    中文翻译:
    型号名称。"""
    ls_model_type: Literal["chat", "llm"]
    """Type of the model. Should be 'chat' or 'llm'.

    中文翻译:
    模型的类型。应该是“聊天”或“llm”。"""
    ls_temperature: float | None
    """Temperature for generation.

    中文翻译:
    生成温度。"""
    ls_max_tokens: int | None
    """Max tokens for generation.

    中文翻译:
    生成的最大代币数。"""
    ls_stop: list[str] | None
    """Stop words for generation.

    中文翻译:
    一代的停用词。"""


@cache  # Cache the tokenizer
def get_tokenizer() -> Any:
    """Get a GPT-2 tokenizer instance.

    This function is cached to avoid re-loading the tokenizer every time it is called.

    Raises:
        ImportError: If the transformers package is not installed.

    Returns:
        The GPT-2 tokenizer instance.

    

    中文翻译:
    获取 GPT-2 分词器实例。
    该函数被缓存以避免每次调用时重新加载分词器。
    加薪：
        ImportError：如果未安装 Transformer 包。
    返回：
        GPT-2 分词器实例。"""
    if not _HAS_TRANSFORMERS:
        msg = (
            "Could not import transformers python package. "
            "This is needed in order to calculate get_token_ids. "
            "Please install it with `pip install transformers`."
        )
        raise ImportError(msg)
    # create a GPT-2 tokenizer instance
    # 中文: 创建 GPT-2 分词器实例
    return GPT2TokenizerFast.from_pretrained("gpt2")


_GPT2_TOKENIZER_WARNED = False


def _get_token_ids_default_method(text: str) -> list[int]:
    """Encode the text into token IDs using the fallback GPT-2 tokenizer.

    中文翻译:
    使用后备 GPT-2 标记生成器将文本编码为标记 ID。"""
    global _GPT2_TOKENIZER_WARNED  # noqa: PLW0603
    if not _GPT2_TOKENIZER_WARNED:
        warnings.warn(
            "Using fallback GPT-2 tokenizer for token counting. "
            "Token counts may be inaccurate for non-GPT-2 models. "
            "For accurate counts, use a model-specific method if available.",
            stacklevel=3,
        )
        _GPT2_TOKENIZER_WARNED = True

    tokenizer = get_tokenizer()

    # Pass verbose=False to suppress the "Token indices sequence length is longer than
    # 中文: 传递 verbose=False 来抑制“令牌索引序列长度长于
    # the specified maximum sequence length" warning from HuggingFace. This warning is
    # 中文: 指定的最大序列长度”来自 HuggingFace 的警告。此警告是
    # about GPT-2's 1024 token context limit, but we're only using the tokenizer for
    # 中文: 关于 GPT-2 的 1024 个令牌上下文限制，但我们仅将令牌生成器用于
    # counting, not for model input.
    # 中文: 计数，不适用于模型输入。
    return cast("list[int]", tokenizer.encode(text, verbose=False))


LanguageModelInput = PromptValue | str | Sequence[MessageLikeRepresentation]
"""Input to a language model.

中文翻译:
输入到语言模型。"""

LanguageModelOutput = BaseMessage | str
"""Output from a language model.

中文翻译:
语言模型的输出。"""

LanguageModelLike = Runnable[LanguageModelInput, LanguageModelOutput]
"""Input/output interface for a language model.

中文翻译:
语言模型的输入/输出接口。"""

LanguageModelOutputVar = TypeVar("LanguageModelOutputVar", AIMessage, str)
"""Type variable for the output of a language model.

中文翻译:
语言模型输出的类型变量。"""


def _get_verbosity() -> bool:
    return get_verbose()


class BaseLanguageModel(
    RunnableSerializable[LanguageModelInput, LanguageModelOutputVar], ABC
):
    """Abstract base class for interfacing with language models.

    All language model wrappers inherited from `BaseLanguageModel`.

    

    中文翻译:
    用于与语言模型交互的抽象基类。
    所有语言模型包装器继承自“BaseLanguageModel”。"""

    cache: BaseCache | bool | None = Field(default=None, exclude=True)
    """Whether to cache the response.

    * If `True`, will use the global cache.
    * If `False`, will not use a cache
    * If `None`, will use the global cache if it's set, otherwise no cache.
    * If instance of `BaseCache`, will use the provided cache.

    Caching is not currently supported for streaming methods of models.
    

    中文翻译:
    是否缓存响应。
    * 如果为“True”，将使用全局缓存。
    * 如果为“False”，则不会使用缓存
    * 如果为“None”，则将使用全局缓存（如果已设置），否则不使用缓存。
    * 如果是 `BaseCache` 实例，将使用提供的缓存。
    目前模型的流方法不支持缓存。"""

    verbose: bool = Field(default_factory=_get_verbosity, exclude=True, repr=False)
    """Whether to print out response text.

    中文翻译:
    是否打印响应文本。"""

    callbacks: Callbacks = Field(default=None, exclude=True)
    """Callbacks to add to the run trace.

    中文翻译:
    添加到运行跟踪的回调。"""

    tags: list[str] | None = Field(default=None, exclude=True)
    """Tags to add to the run trace.

    中文翻译:
    要添加到运行跟踪的标签。"""

    metadata: dict[str, Any] | None = Field(default=None, exclude=True)
    """Metadata to add to the run trace.

    中文翻译:
    要添加到运行跟踪的元数据。"""

    custom_get_token_ids: Callable[[str], list[int]] | None = Field(
        default=None, exclude=True
    )
    """Optional encoder to use for counting tokens.

    中文翻译:
    用于计算令牌的可选编码器。"""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @field_validator("verbose", mode="before")
    def set_verbose(cls, verbose: bool | None) -> bool:  # noqa: FBT001
        """If verbose is `None`, set it.

        This allows users to pass in `None` as verbose to access the global setting.

        Args:
            verbose: The verbosity setting to use.

        Returns:
            The verbosity setting to use.

        

        中文翻译:
        如果 verbose 为“None”，请设置它。
        这允许用户传入“None”作为详细信息来访问全局设置。
        参数：
            verbose：要使用的详细程度设置。
        返回：
            要使用的详细程度设置。"""
        if verbose is None:
            return _get_verbosity()
        return verbose

    @property
    @override
    def InputType(self) -> TypeAlias:
        """Get the input type for this `Runnable`.

        中文翻译:
        获取此“Runnable”的输入类型。"""
        # This is a version of LanguageModelInput which replaces the abstract
        # 中文: 这是 LanguageModelInput 的一个版本，它取代了抽象
        # base class BaseMessage with a union of its subclasses, which makes
        # 中文: 基类 BaseMessage 及其子类的联合，这使得
        # for a much better schema.
        # 中文: 为了更好的模式。
        return str | StringPromptValue | ChatPromptValueConcrete | list[AnyMessage]

    @abstractmethod
    def generate_prompt(
        self,
        prompts: list[PromptValue],
        stop: list[str] | None = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Pass a sequence of prompts to the model and return model generations.

        This method should make use of batched calls for models that expose a batched
        API.

        Use this method when you want to:

        1. Take advantage of batched calls,
        2. Need more output from the model than just the top generated value,
        3. Are building chains that are agnostic to the underlying language model
            type (e.g., pure text completion models vs chat models).

        Args:
            prompts: List of `PromptValue` objects.

                A `PromptValue` is an object that can be converted to match the format
                of any language model (string for pure text generation models and
                `BaseMessage` objects for chat models).
            stop: Stop words to use when generating.

                Model output is cut off at the first occurrence of any of these
                substrings.
            callbacks: `Callbacks` to pass through.

                Used for executing additional functionality, such as logging or
                streaming, throughout generation.
            **kwargs: Arbitrary additional keyword arguments.

                These are usually passed to the model provider API call.

        Returns:
            An `LLMResult`, which contains a list of candidate `Generation` objects for
                each input prompt and additional model provider-specific output.

        

        中文翻译:
        将一系列提示传递给模型并返回模型生成。
        此方法应该对公开批量的模型使用批量调用
        API。
        当您想要执行以下操作时，请使用此方法：
        1.利用批量调用的优势，
        2. 需要模型的更多输出而不仅仅是顶部生成的值，
        3.正在构建与底层语言模型无关的链
            类型（例如，纯文本完成模型与聊天模型）。
        参数：
            提示：“PromptValue”对象的列表。
                `PromptValue` 是一个可以转换以匹配格式的对象
                任何语言模型的（纯文本生成模型的字符串和
                用于聊天模型的“BaseMessage”对象）。
            stop：生成时使用的停止词。
                模型输出在第一次出现这些情况时被切断
                子串。
            回调：要传递的“回调”。
                用于执行附加功能，例如日志记录或
                流式传输，贯穿一代。
            **kwargs：任意附加关键字参数。
                这些通常会传递给模型提供者 API 调用。
        返回：
            一个“LLMResult”，其中包含候选“Generation”对象的列表
                每个输入提示和附加模型提供者特定的输出。"""

    @abstractmethod
    async def agenerate_prompt(
        self,
        prompts: list[PromptValue],
        stop: list[str] | None = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Asynchronously pass a sequence of prompts and return model generations.

        This method should make use of batched calls for models that expose a batched
        API.

        Use this method when you want to:

        1. Take advantage of batched calls,
        2. Need more output from the model than just the top generated value,
        3. Are building chains that are agnostic to the underlying language model
            type (e.g., pure text completion models vs chat models).

        Args:
            prompts: List of `PromptValue` objects.

                A `PromptValue` is an object that can be converted to match the format
                of any language model (string for pure text generation models and
                `BaseMessage` objects for chat models).
            stop: Stop words to use when generating.

                Model output is cut off at the first occurrence of any of these
                substrings.
            callbacks: `Callbacks` to pass through.

                Used for executing additional functionality, such as logging or
                streaming, throughout generation.
            **kwargs: Arbitrary additional keyword arguments.

                These are usually passed to the model provider API call.

        Returns:
            An `LLMResult`, which contains a list of candidate `Generation` objects for
                each input prompt and additional model provider-specific output.

        

        中文翻译:
        异步传递一系列提示并返回模型生成。
        此方法应该对公开批量的模型使用批量调用
        API。
        当您想要执行以下操作时，请使用此方法：
        1.利用批量调用的优势，
        2. 需要模型的更多输出而不仅仅是顶部生成的值，
        3.正在构建与底层语言模型无关的链
            类型（例如，纯文本完成模型与聊天模型）。
        参数：
            提示：“PromptValue”对象的列表。
                `PromptValue` 是一个可以转换以匹配格式的对象
                任何语言模型的（纯文本生成模型的字符串和
                用于聊天模型的“BaseMessage”对象）。
            stop：生成时使用的停止词。
                模型输出在第一次出现这些情况时被切断
                子串。
            回调：要传递的“回调”。
                用于执行附加功能，例如日志记录或
                流式传输，贯穿一代。
            **kwargs：任意附加关键字参数。
                这些通常会传递给模型提供者 API 调用。
        返回：
            一个“LLMResult”，其中包含候选“Generation”对象的列表
                每个输入提示和附加模型提供者特定的输出。"""

    def with_structured_output(
        self, schema: dict | type, **kwargs: Any
    ) -> Runnable[LanguageModelInput, dict | BaseModel]:
        """Not implemented on this class.

        中文翻译:
        未在此类上实现。"""
        # Implement this on child class if there is a way of steering the model to
        # 中文: 如果有一种方法可以引导模型，请在子类上实现此操作
        # generate responses that match a given schema.
        # 中文: 生成与给定模式匹配的响应。
        raise NotImplementedError

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters.

        中文翻译:
        获取识别参数。"""
        return self.lc_attributes

    def get_token_ids(self, text: str) -> list[int]:
        """Return the ordered IDs of the tokens in a text.

        Args:
            text: The string input to tokenize.

        Returns:
            A list of IDs corresponding to the tokens in the text, in order they occur
                in the text.
        

        中文翻译:
        返回文本中标记的有序 ID。
        参数：
            text：要标记化的字符串输入。
        返回：
            与文本中的标记相对应的 ID 列表（按出现顺序排列）
                在文字中。"""
        if self.custom_get_token_ids is not None:
            return self.custom_get_token_ids(text)
        return _get_token_ids_default_method(text)

    def get_num_tokens(self, text: str) -> int:
        """Get the number of tokens present in the text.

        Useful for checking if an input fits in a model's context window.

        This should be overridden by model-specific implementations to provide accurate
        token counts via model-specific tokenizers.

        Args:
            text: The string input to tokenize.

        Returns:
            The integer number of tokens in the text.

        

        中文翻译:
        获取文本中存在的标记数量。
        用于检查输入是否适合模型的上下文窗口。
        这应该被特定于模型的实现覆盖，以提供准确的
        通过特定于模型的标记器进行标记计数。
        参数：
            text：要标记化的字符串输入。
        返回：
            文本中标记的整数个数。"""
        return len(self.get_token_ids(text))

    def get_num_tokens_from_messages(
        self,
        messages: list[BaseMessage],
        tools: Sequence | None = None,
    ) -> int:
        """Get the number of tokens in the messages.

        Useful for checking if an input fits in a model's context window.

        This should be overridden by model-specific implementations to provide accurate
        token counts via model-specific tokenizers.

        !!! note

            * The base implementation of `get_num_tokens_from_messages` ignores tool
                schemas.
            * The base implementation of `get_num_tokens_from_messages` adds additional
                prefixes to messages in represent user roles, which will add to the
                overall token count. Model-specific implementations may choose to
                handle this differently.

        Args:
            messages: The message inputs to tokenize.
            tools: If provided, sequence of dict, `BaseModel`, function, or
                `BaseTool` objects to be converted to tool schemas.

        Returns:
            The sum of the number of tokens across the messages.

        

        中文翻译:
        获取消息中的令牌数量。
        用于检查输入是否适合模型的上下文窗口。
        这应该被特定于模型的实现覆盖，以提供准确的
        通过特定于模型的标记器进行标记计数。
        !!!注释
            * `get_num_tokens_from_messages` 的基本实现忽略工具
                模式。
            * `get_num_tokens_from_messages` 的基本实现添加了额外的
                代表用户角色的消息前缀，这将添加到
                令牌总数。特定于模型的实现可以选择
                以不同的方式处理这个问题。
        参数：
            messages：要标记化的消息输入。
            工具：如果提供，则为字典序列、“BaseModel”、函数或
                要转换为工具模式的“BaseTool”对象。
        返回：
            消息中令牌数量的总和。"""
        if tools is not None:
            warnings.warn(
                "Counting tokens in tool schemas is not yet supported. Ignoring tools.",
                stacklevel=2,
            )
        return sum(self.get_num_tokens(get_buffer_string([m])) for m in messages)
