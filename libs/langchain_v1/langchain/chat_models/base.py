"""Factory functions for chat models.

中文翻译:
聊天模型的工厂函数。"""

from __future__ import annotations

import functools
import importlib
import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    TypeAlias,
    cast,
    overload,
)

from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, AnyMessage
from langchain_core.prompt_values import ChatPromptValueConcrete, StringPromptValue
from langchain_core.runnables import Runnable, RunnableConfig, ensure_config
from typing_extensions import override

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterator, Sequence
    from types import ModuleType

    from langchain_core.runnables.schema import StreamEvent
    from langchain_core.tools import BaseTool
    from langchain_core.tracers import RunLog, RunLogPatch
    from pydantic import BaseModel


def _call(cls: type[BaseChatModel], **kwargs: Any) -> BaseChatModel:
    # TODO: replace with operator.call when lower bounding to Python 3.11
    return cls(**kwargs)


_SUPPORTED_PROVIDERS: dict[str, tuple[str, str, Callable[..., BaseChatModel]]] = {
    "anthropic": ("langchain_anthropic", "ChatAnthropic", _call),
    "azure_ai": ("langchain_azure_ai.chat_models", "AzureAIChatCompletionsModel", _call),
    "azure_openai": ("langchain_openai", "AzureChatOpenAI", _call),
    "bedrock": ("langchain_aws", "ChatBedrock", _call),
    "bedrock_converse": ("langchain_aws", "ChatBedrockConverse", _call),
    "cohere": ("langchain_cohere", "ChatCohere", _call),
    "deepseek": ("langchain_deepseek", "ChatDeepSeek", _call),
    "fireworks": ("langchain_fireworks", "ChatFireworks", _call),
    "google_anthropic_vertex": (
        "langchain_google_vertexai.model_garden",
        "ChatAnthropicVertex",
        _call,
    ),
    "google_genai": ("langchain_google_genai", "ChatGoogleGenerativeAI", _call),
    "google_vertexai": ("langchain_google_vertexai", "ChatVertexAI", _call),
    "groq": ("langchain_groq", "ChatGroq", _call),
    "huggingface": (
        "langchain_huggingface",
        "ChatHuggingFace",
        lambda cls, model, **kwargs: cls.from_model_id(model_id=model, **kwargs),
    ),
    "ibm": (
        "langchain_ibm",
        "ChatWatsonx",
        lambda cls, model, **kwargs: cls(model_id=model, **kwargs),
    ),
    "mistralai": ("langchain_mistralai", "ChatMistralAI", _call),
    "nvidia": ("langchain_nvidia_ai_endpoints", "ChatNVIDIA", _call),
    "ollama": ("langchain_ollama", "ChatOllama", _call),
    "openai": ("langchain_openai", "ChatOpenAI", _call),
    "perplexity": ("langchain_perplexity", "ChatPerplexity", _call),
    "together": ("langchain_together", "ChatTogether", _call),
    "upstage": ("langchain_upstage", "ChatUpstage", _call),
    "xai": ("langchain_xai", "ChatXAI", _call),
}
"""Registry mapping provider names to their import configuration.

Each entry maps a provider key to a tuple of:

- `module_path`: The Python module path containing the chat model class.

    This may be a submodule (e.g., `'langchain_azure_ai.chat_models'`) if the class is
    not exported from the package root.
- `class_name`: The name of the chat model class to import.
- `creator_func`: A callable that instantiates the class with provided kwargs.

中文翻译:
注册表将提供程序名称映射到其导入配置。
每个条目将提供者密钥映射到以下元组：
- `module_path`：包含聊天模型类的 Python 模块路径。
    如果该类是，这可能是一个子模块（例如“langchain_azure_ai.chat_models”）
    未从包根目录导出。
- `class_name`：要导入的聊天模型类的名称。
- `creator_func`：使用提供的 kwargs 实例化类的可调用对象。
"""


def _import_module(module: str) -> ModuleType:
    """Import a module by name.

    Args:
        module: The fully qualified module name to import (e.g., `'langchain_openai'`).

    Returns:
        The imported module.

    Raises:
        ImportError: If the module cannot be imported, with a message suggesting
            the pip package to install.
    

    中文翻译:
    按名称导入模块。
    参数：
        module：要导入的完全限定模块名称（例如“langchain_openai”）。
    返回：
        导入的模块。
    加薪：
        ImportError: 如果模块无法导入，有提示信息
            要安装的 pip 包。"""
    try:
        return importlib.import_module(module)
    except ImportError as e:
        # Extract package name from module path (e.g., "langchain_azure_ai.chat_models"
        # 中文: 从模块路径中提取包名称（例如“langchain_azure_ai.chat_models”
        # becomes "langchain-azure-ai")
        # 中文: 变为“langchain-azure-ai”）
        pkg = module.split(".", maxsplit=1)[0].replace("_", "-")
        msg = f"Could not import {pkg} python package. Please install it with `pip install {pkg}`"
        raise ImportError(msg) from e


@functools.lru_cache(maxsize=len(_SUPPORTED_PROVIDERS))
def _get_chat_model_creator(
    provider: str,
) -> Callable[..., BaseChatModel]:
    """Return a factory function that creates a chat model for the given provider.

    This function is cached to avoid repeated module imports.

    Args:
        provider: The name of the model provider (e.g., `'openai'`, `'anthropic'`).

            Must be a key in `_SUPPORTED_PROVIDERS`.

    Returns:
        A callable that accepts model kwargs and returns a `BaseChatModel` instance for
            the specified provider.

    Raises:
        ValueError: If the provider is not in `_SUPPORTED_PROVIDERS`.
        ImportError: If the provider's integration package is not installed.
    

    中文翻译:
    返回一个为给定提供者创建聊天模型的工厂函数。
    该函数被缓存以避免重复的模块导入。
    参数：
        提供者：模型提供者的名称（例如“openai”、“anthropic”）。
            必须是“_SUPPORTED_PROVIDERS”中的键。
    返回：
        接受模型 kwargs 并返回“BaseChatModel”实例的可调用对象
            指定的提供商。
    加薪：
        ValueError：如果提供程序不在“_SUPPORTED_PROVIDERS”中。
        ImportError：如果未安装提供商的集成包。"""
    if provider not in _SUPPORTED_PROVIDERS:
        supported = ", ".join(_SUPPORTED_PROVIDERS.keys())
        msg = f"Unsupported {provider=}.\n\nSupported model providers are: {supported}"
        raise ValueError(msg)

    pkg, class_name, creator_func = _SUPPORTED_PROVIDERS[provider]
    try:
        module = _import_module(pkg)
    except ImportError as e:
        if provider != "ollama":
            raise
        # For backwards compatibility
        # 中文: 为了向后兼容
        try:
            module = _import_module("langchain_community.chat_models")
        except ImportError:
            # If both langchain-ollama and langchain-community aren't available,
            # 中文: 如果 langchain-ollama 和 langchain-community 都不可用，
            # raise an error related to langchain-ollama
            # 中文: 引发与 langchain-ollama 相关的错误
            raise e from None

    cls = getattr(module, class_name)
    return functools.partial(creator_func, cls=cls)


@overload
def init_chat_model(
    model: str,
    *,
    model_provider: str | None = None,
    configurable_fields: None = None,
    config_prefix: str | None = None,
    **kwargs: Any,
) -> BaseChatModel: ...


@overload
def init_chat_model(
    model: None = None,
    *,
    model_provider: str | None = None,
    configurable_fields: None = None,
    config_prefix: str | None = None,
    **kwargs: Any,
) -> _ConfigurableModel: ...


@overload
def init_chat_model(
    model: str | None = None,
    *,
    model_provider: str | None = None,
    configurable_fields: Literal["any"] | list[str] | tuple[str, ...] = ...,
    config_prefix: str | None = None,
    **kwargs: Any,
) -> _ConfigurableModel: ...


# FOR CONTRIBUTORS: If adding support for a new provider, please append the provider
# 中文: 对于贡献者：如果添加对新提供商的支持，请附加该提供商
# name to the supported list in the docstring below. Do *not* change the order of the
# 中文: 将名称添加到下面的文档字符串中支持的列表中。不要*改变*的顺序
# existing providers.
# 中文: 现有的提供商。
def init_chat_model(
    model: str | None = None,
    *,
    model_provider: str | None = None,
    configurable_fields: Literal["any"] | list[str] | tuple[str, ...] | None = None,
    config_prefix: str | None = None,
    **kwargs: Any,
) -> BaseChatModel | _ConfigurableModel:
    """Initialize a chat model from any supported provider using a unified interface.

    **Two main use cases:**

    1. **Fixed model** – specify the model upfront and get a ready-to-use chat model.
    2. **Configurable model** – choose to specify parameters (including model name) at
        runtime via `config`. Makes it easy to switch between models/providers without
        changing your code

    !!! note "Installation requirements"
        Requires the integration package for the chosen model provider to be installed.

        See the `model_provider` parameter below for specific package names
        (e.g., `pip install langchain-openai`).

        Refer to the [provider integration's API reference](https://docs.langchain.com/oss/python/integrations/providers)
        for supported model parameters to use as `**kwargs`.

    Args:
        model: The model name, optionally prefixed with provider (e.g., `'openai:gpt-4o'`).

            Will attempt to infer `model_provider` from model if not specified.

            The following providers will be inferred based on these model prefixes:

            - `gpt-...` | `o1...` | `o3...`       -> `openai`
            - `claude...`                         -> `anthropic`
            - `amazon...`                         -> `bedrock`
            - `gemini...`                         -> `google_vertexai`
            - `command...`                        -> `cohere`
            - `accounts/fireworks...`             -> `fireworks`
            - `mistral...`                        -> `mistralai`
            - `deepseek...`                       -> `deepseek`
            - `grok...`                           -> `xai`
            - `sonar...`                          -> `perplexity`
            - `solar...`                          -> `upstage`
        model_provider: The model provider if not specified as part of the model arg
            (see above).

            Supported `model_provider` values and the corresponding integration package
            are:

            - `openai`                  -> [`langchain-openai`](https://docs.langchain.com/oss/python/integrations/providers/openai)
            - `anthropic`               -> [`langchain-anthropic`](https://docs.langchain.com/oss/python/integrations/providers/anthropic)
            - `azure_openai`            -> [`langchain-openai`](https://docs.langchain.com/oss/python/integrations/providers/openai)
            - `azure_ai`                -> [`langchain-azure-ai`](https://docs.langchain.com/oss/python/integrations/providers/microsoft)
            - `google_vertexai`         -> [`langchain-google-vertexai`](https://docs.langchain.com/oss/python/integrations/providers/google)
            - `google_genai`            -> [`langchain-google-genai`](https://docs.langchain.com/oss/python/integrations/providers/google)
            - `bedrock`                 -> [`langchain-aws`](https://docs.langchain.com/oss/python/integrations/providers/aws)
            - `bedrock_converse`        -> [`langchain-aws`](https://docs.langchain.com/oss/python/integrations/providers/aws)
            - `cohere`                  -> [`langchain-cohere`](https://docs.langchain.com/oss/python/integrations/providers/cohere)
            - `fireworks`               -> [`langchain-fireworks`](https://docs.langchain.com/oss/python/integrations/providers/fireworks)
            - `together`                -> [`langchain-together`](https://docs.langchain.com/oss/python/integrations/providers/together)
            - `mistralai`               -> [`langchain-mistralai`](https://docs.langchain.com/oss/python/integrations/providers/mistralai)
            - `huggingface`             -> [`langchain-huggingface`](https://docs.langchain.com/oss/python/integrations/providers/huggingface)
            - `groq`                    -> [`langchain-groq`](https://docs.langchain.com/oss/python/integrations/providers/groq)
            - `ollama`                  -> [`langchain-ollama`](https://docs.langchain.com/oss/python/integrations/providers/ollama)
            - `google_anthropic_vertex` -> [`langchain-google-vertexai`](https://docs.langchain.com/oss/python/integrations/providers/google)
            - `deepseek`                -> [`langchain-deepseek`](https://docs.langchain.com/oss/python/integrations/providers/deepseek)
            - `ibm`                     -> [`langchain-ibm`](https://docs.langchain.com/oss/python/integrations/providers/ibm)
            - `nvidia`                  -> [`langchain-nvidia-ai-endpoints`](https://docs.langchain.com/oss/python/integrations/providers/nvidia)
            - `xai`                     -> [`langchain-xai`](https://docs.langchain.com/oss/python/integrations/providers/xai)
            - `perplexity`              -> [`langchain-perplexity`](https://docs.langchain.com/oss/python/integrations/providers/perplexity)
            - `upstage`                 -> [`langchain-upstage`](https://docs.langchain.com/oss/python/integrations/providers/upstage)

        configurable_fields: Which model parameters are configurable at runtime:

            - `None`: No configurable fields (i.e., a fixed model).
            - `'any'`: All fields are configurable. **See security note below.**
            - `list[str] | Tuple[str, ...]`: Specified fields are configurable.

            Fields are assumed to have `config_prefix` stripped if a `config_prefix` is
            specified.

            If `model` is specified, then defaults to `None`.

            If `model` is not specified, then defaults to `("model", "model_provider")`.

            !!! warning "Security note"

                Setting `configurable_fields="any"` means fields like `api_key`,
                `base_url`, etc., can be altered at runtime, potentially redirecting
                model requests to a different service/user.

                Make sure that if you're accepting untrusted configurations that you
                enumerate the `configurable_fields=(...)` explicitly.

        config_prefix: Optional prefix for configuration keys.

            Useful when you have multiple configurable models in the same application.

            If `'config_prefix'` is a non-empty string then `model` will be configurable
            at runtime via the `config["configurable"]["{config_prefix}_{param}"]` keys.
            See examples below.

            If `'config_prefix'` is an empty string then model will be configurable via
            `config["configurable"]["{param}"]`.
        **kwargs: Additional model-specific keyword args to pass to the underlying
            chat model's `__init__` method. Common parameters include:

            - `temperature`: Model temperature for controlling randomness.
            - `max_tokens`: Maximum number of output tokens.
            - `timeout`: Maximum time (in seconds) to wait for a response.
            - `max_retries`: Maximum number of retry attempts for failed requests.
            - `base_url`: Custom API endpoint URL.
            - `rate_limiter`: A
                [`BaseRateLimiter`][langchain_core.rate_limiters.BaseRateLimiter]
                instance to control request rate.

            Refer to the specific model provider's
            [integration reference](https://reference.langchain.com/python/integrations/)
            for all available parameters.

    Returns:
        A `BaseChatModel` corresponding to the `model_name` and `model_provider`
            specified if configurability is inferred to be `False`. If configurable, a
            chat model emulator that initializes the underlying model at runtime once a
            config is passed in.

    Raises:
        ValueError: If `model_provider` cannot be inferred or isn't supported.
        ImportError: If the model provider integration package is not installed.

    ???+ example "Initialize a non-configurable model"

        ```python
        # pip install langchain langchain-openai langchain-anthropic langchain-google-vertexai
        # 中文: pip 安装 langchain langchain-openai langchain-anthropic langchain-google-vertexai

        from langchain.chat_models import init_chat_model

        o3_mini = init_chat_model("openai:o3-mini", temperature=0)
        claude_sonnet = init_chat_model("anthropic:claude-sonnet-4-5-20250929", temperature=0)
        gemini_2-5_flash = init_chat_model("google_vertexai:gemini-2.5-flash", temperature=0)

        o3_mini.invoke("what's your name")
        claude_sonnet.invoke("what's your name")
        gemini_2-5_flash.invoke("what's your name")
        ```

    ??? example "Partially configurable model with no default"

        ```python
        # pip install langchain langchain-openai langchain-anthropic
        # 中文: pip install langchain langchain-openai langchain-anthropic

        from langchain.chat_models import init_chat_model

        # (We don't need to specify configurable=True if a model isn't specified.)
        # 中文: （如果未指定模型，我们不需要指定configurable=True。）
        configurable_model = init_chat_model(temperature=0)

        configurable_model.invoke("what's your name", config={"configurable": {"model": "gpt-4o"}})
        # Use GPT-4o to generate the response
        # 中文: 使用 GPT-4o 生成响应

        configurable_model.invoke(
            "what's your name",
            config={"configurable": {"model": "claude-sonnet-4-5-20250929"}},
        )
        ```

    ??? example "Fully configurable model with a default"

        ```python
        # pip install langchain langchain-openai langchain-anthropic
        # 中文: pip install langchain langchain-openai langchain-anthropic

        from langchain.chat_models import init_chat_model

        configurable_model_with_default = init_chat_model(
            "openai:gpt-4o",
            configurable_fields="any",  # This allows us to configure other params like temperature, max_tokens, etc at runtime.
            config_prefix="foo",
            temperature=0,
        )

        configurable_model_with_default.invoke("what's your name")
        # GPT-4o response with temperature 0 (as set in default)
        # 中文: GPT-4o 响应温度为 0（默认设置）

        configurable_model_with_default.invoke(
            "what's your name",
            config={
                "configurable": {
                    "foo_model": "anthropic:claude-sonnet-4-5-20250929",
                    "foo_temperature": 0.6,
                }
            },
        )
        # Override default to use Sonnet 4.5 with temperature 0.6 to generate response
        # 中文: 覆盖默认值以使用 Sonnet 4.5 和温度 0.6 来生成响应
        ```

    ??? example "Bind tools to a configurable model"

        You can call any chat model declarative methods on a configurable model in the
        same way that you would with a normal model:

        ```python
        # pip install langchain langchain-openai langchain-anthropic
        # 中文: pip install langchain langchain-openai langchain-anthropic

        from langchain.chat_models import init_chat_model
        from pydantic import BaseModel, Field


        class GetWeather(BaseModel):
            '''Get the current weather in a given location'''

            location: str = Field(..., description="The city and state, e.g. San Francisco, CA")


        class GetPopulation(BaseModel):
            '''Get the current population in a given location'''

            location: str = Field(..., description="The city and state, e.g. San Francisco, CA")


        configurable_model = init_chat_model(
            "gpt-4o", configurable_fields=("model", "model_provider"), temperature=0
        )

        configurable_model_with_tools = configurable_model.bind_tools(
            [
                GetWeather,
                GetPopulation,
            ]
        )
        configurable_model_with_tools.invoke(
            "Which city is hotter today and which is bigger: LA or NY?"
        )
        # Use GPT-4o
        # 中文: 使用 GPT-4o

        configurable_model_with_tools.invoke(
            "Which city is hotter today and which is bigger: LA or NY?",
            config={"configurable": {"model": "claude-sonnet-4-5-20250929"}},
        )
        # Use Sonnet 4.5
        # 中文: 使用十四行诗 4.5
        ```

    

    中文翻译:
    使用统一界面从任何受支持的提供商处初始化聊天模型。
    **两个主要用例：**
    1. **固定模型** – 预先指定模型并获得现成的聊天模型。
    2. **可配置模型** – 选择在以下位置指定参数（包括模型名称）
        通过“config”运行时。可以轻松地在模型/提供商之间切换，而无需
        改变你的代码
    !!!注意“安装要求”
        需要安装所选模型提供程序的集成包。
        具体包名请参见下面的`model_provider`参数
        （例如，“pip install langchain-openai”）。
        参考【提供商集成的API参考】(https://docs.langchain.com/oss/python/integrations/providers)
        支持的模型参数用作“**kwargs”。
    参数：
        model：模型名称，可以选择以提供者为前缀（例如“openai:gpt-4o”）。
            如果未指定，将尝试从模型推断“model_provider”。
            将根据这些模型前缀推断出以下提供程序：
            - `gpt-...` | `o1...` | `o3...` -> `openai`
            - `克劳德...` -> `人类`
            - `亚马逊...` -> `基岩`
            - `gemini...` -> `google_vertexai`
            - `命令...` -> `cohere`
            - `帐户/烟花...` -> `烟花`
            - `米斯特拉尔...` -> `米斯特拉莱`
            - `deepseek...` -> `deepseek`
            - `grok...` -> `xai`
            - `声纳...` -> `困惑`
            - `太阳能...` -> `后台`
        model_provider：模型提供者（如果未指定为模型参数的一部分）
            （见上文）。
            支持的`model_provider`值和相应的集成包
            是：
            - `openai` -> [`langchain-openai`](https://docs.langchain.com/oss/python/integrations/providers/openai)
            - `anthropic` -> [`langchain-anthropic`](https://docs.langchain.com/oss/python/integrations/providers/anthropic)
            - `azure_openai` -> [`langchain-openai`](https://docs.langchain.com/oss/python/integrations/providers/openai)
            - `azure_ai` -> [`langchain-azure-ai`](https://docs.langchain.com/oss/python/integrations/providers/microsoft)
            - `google_vertexai` -> [`langchain-google-vertexai`](https://docs.langchain.com/oss/python/integrations/providers/google)
            - `google_genai` -> [`langchain-google-genai`](https://docs.langchain.com/oss/python/integrations/providers/google)
            - `bedrock` -> [`langchain-aws`](https://docs.langchain.com/oss/python/integrations/providers/aws)
            - `bedrock_converse` -> [`langchain-aws`](https://docs.langchain.com/oss/python/integrations/providers/aws)
            - `cohere` -> [`langchain-cohere`](https://docs.langchain.com/oss/python/integrations/providers/cohere)
            - `烟花` -> [`langchain-fireworks`](https://docs.langchain.com/oss/python/integrations/providers/fireworks)
            - `together` -> [`langchain-together`](https://docs.langchain.com/oss/python/integrations/providers/together)
            - `mistralai` -> [`langchain-mistralai`](https://docs.langchain.com/oss/python/integrations/providers/mistralai)
            - `huggingface` -> [`langchain-huggingface`](https://docs.langchain.com/oss/python/integrations/providers/huggingface)
            - `groq` -> [`langchain-groq`](https://docs.langchain.com/oss/python/integrations/providers/groq)
            - `ollama` -> [`langchain-ollama`](https://docs.langchain.com/oss/python/integrations/providers/ollama)
            - `google_anthropic_vertex` -> [`langchain-google-vertexai`](https://docs.langchain.com/oss/python/integrations/providers/google)
            - `deepseek` -> [`langchain-deepseek`](https://docs.langchain.com/oss/python/integrations/providers/deepseek)
            - `ibm` -> [`langchain-ibm`](https://docs.langchain.com/oss/python/integrations/providers/ibm)- `nvidia` -> [`langchain-nvidia-ai-endpoints`](https://docs.langchain.com/oss/python/integrations/providers/nvidia)
            - `xai` -> [`langchain-xai`](https://docs.langchain.com/oss/python/integrations/providers/xai)
            - `perplexity` -> [`langchain-perplexity`](https://docs.langchain.com/oss/python/integrations/providers/perplexity)
            - `upstage` -> [`langchain-upstage`](https://docs.langchain.com/oss/python/integrations/providers/upstage)
        可配置字段：哪些模型参数可以在运行时配置：
            - `None`：没有可配置字段（即固定模型）。
            - `'any'`：所有字段都是可配置的。 **请参阅下面的安全说明。**
            - `列表[str] | Tuple[str, ...]`：指定字段是可配置的。
            如果“config_prefix”是，则假定字段已删除“config_prefix”
            指定。
            如果指定了“model”，则默认为“None”。
            如果未指定`model`，则默认为`("model", "model_provider")`。
            ！！！警告“安全说明”
                设置 `configurable_fields="any"` 意味着像 `api_key` 这样的字段，
                `base_url` 等可以在运行时更改，可能会重定向
                对不同服务/用户的模型请求。
                确保如果您接受不受信任的配置，
                显式枚举“configurable_fields=(...)”。
        config_prefix：配置键的可选前缀。
            当同一应用程序中有多个可配置模型时非常有用。
            如果“config_prefix”是非空字符串，则“model”将是可配置的
            在运行时通过 `config["configurable"]["{config_prefix}_{param}"]` 键。
            请参阅下面的示例。
            如果“config_prefix”是空字符串，则模型将可通过以下方式配置
            `config["可配置"]["{param}"]`。
        **kwargs：传递给底层的附加特定于模型的关键字参数
            聊天模型的`__init__`方法。常用参数包括：
            - `温度`：用于控制随机性的模型温度。
            - `max_tokens`：输出令牌的最大数量。
            - `timeout`：等待响应的最长时间（以秒为单位）。
            - `max_retries`：失败请求的最大重试次数。
            - `base_url`：自定义 API 端点 URL。
            - `速率限制器`：A
                [`BaseRateLimiter`][langchain_core.rate_limiters.BaseRateLimiter]
                控制请求率的实例。
            具体型号请参考供应商的
            [集成参考](https://reference.langchain.com/python/integrations/)
            对于所有可用的参数。
    返回：
        对应于“model_name”和“model_provider”的“BaseChatModel”
            如果可配置性被推断为“False”，则指定。如果可配置，
            聊天模型模拟器，在运行时初始化底层模型一次
            配置已传入。
    加薪：
        ValueError：如果无法推断或不支持“model_provider”。
        ImportError：如果未安装模型提供程序集成包。
    ???+ 示例“初始化不可配置的模型”
        ````蟒蛇
        # pip 安装 langchain langchain-openai langchain-anthropic langchain-google-vertexai
        从 langchain.chat_models 导入 init_chat_model
        o3_mini = init_chat_model("openai:o3-mini", 温度=0)
        claude_sonnet = init_chat_model("人类：claude-sonnet-4-5-20250929"，温度=0)
        gemini_2-5_flash = init_chat_model("google_vertexai:gemini-2.5-flash", 温度=0)
        o3_mini.invoke("你叫什么名字")
        claude_sonnet.invoke("你叫什么名字")
        gemini_2-5_flash.invoke("你叫什么名字")
        ````
    ???示例“部分可配置模型，无默认值”
        ````蟒蛇
        # pip install langchain langchain-openai langchain-anthropic
        # 中文: pip install langchain langchain-openai langchain-anthropic
        从 langchain.chat_models 导入 init_chat_model
        # （如果没有指定模型，我们不需要指定configurable=True。）
        可配置模型 = init_chat_model(温度=0)
        configurable_model.invoke("你叫什么名字", config={"configurable": {"model": "gpt-4o"}})
        # 使用 GPT-4o 生成响应可配置_模型.调用(
            “你叫什么名字”，
            config={"可配置": {"型号": "claude-sonnet-4-5-20250929"}},
        ）
        ````
    ???示例“具有默认值的完全可配置模型”
        ````蟒蛇
        # pip install langchain langchain-openai langchain-anthropic
        # 中文: pip install langchain langchain-openai langchain-anthropic
        从 langchain.chat_models 导入 init_chat_model
        可配置_model_with_default = init_chat_model(
            “openai：gpt-4o”，
            configurable_fields="any", # 这允许我们在运行时配置其他参数，如温度、max_tokens 等。
            config_prefix =“foo”，
            温度=0，
        ）
        configurable_model_with_default.invoke(“你叫什么名字”)
        # GPT-4o 响应温度为 0（默认设置）
        可配置_model_with_default.invoke（
            “你叫什么名字”，
            配置={
                “可配置”：{
                    "foo_model": "人类:claude-sonnet-4-5-20250929",
                    “foo_温度”：0.6，
                }
            },
        ）
        # 覆盖默认值以使用 Sonnet 4.5 和温度 0.6 来生成响应
        ````
    ???示例“将工具绑定到可配置模型”
        您可以在可配置模型上调用任何聊天模型声明方法
        与使用普通模型的方式相同：
        ````蟒蛇
        # pip install langchain langchain-openai langchain-anthropic
        # 中文: pip install langchain langchain-openai langchain-anthropic
        从 langchain.chat_models 导入 init_chat_model
        从 pydantic 导入 BaseModel、Field
        类 GetWeather(BaseModel):
            '''获取给定位置的当前天气'''
            位置：str = Field（...，description =“城市和州，例如加利福尼亚州旧金山”）
        类 GetPopulation(BaseModel):
            '''获取给定位置的当前人口'''
            位置：str = Field（...，description =“城市和州，例如加利福尼亚州旧金山”）
        可配置模型 = init_chat_model(
            “gpt-4o”，configurable_fields =（“模型”，“model_provider”），温度= 0
        ）
        可配置模型与工具 = 可配置模型.bind_tools(
            [
                获取天气，
                获取人口，
            ]
        ）
        可配置_model_with_tools.invoke（
            “今天哪个城市更热，哪个城市更大：洛杉矶还是纽约？”
        ）
        # 使用 GPT-4o
        可配置_model_with_tools.invoke（
            “今天哪个城市更热，哪个城市更大：洛杉矶还是纽约？”,
            config={"可配置": {"型号": "claude-sonnet-4-5-20250929"}},
        ）
        # 使用十四行诗 4.5
        ````"""  # noqa: E501
    if not model and not configurable_fields:
        configurable_fields = ("model", "model_provider")
    config_prefix = config_prefix or ""
    if config_prefix and not configurable_fields:
        warnings.warn(
            f"{config_prefix=} has been set but no fields are configurable. Set "
            f"`configurable_fields=(...)` to specify the model params that are "
            f"configurable.",
            stacklevel=2,
        )

    if not configurable_fields:
        return _init_chat_model_helper(
            cast("str", model),
            model_provider=model_provider,
            **kwargs,
        )
    if model:
        kwargs["model"] = model
    if model_provider:
        kwargs["model_provider"] = model_provider
    return _ConfigurableModel(
        default_config=kwargs,
        config_prefix=config_prefix,
        configurable_fields=configurable_fields,
    )


def _init_chat_model_helper(
    model: str,
    *,
    model_provider: str | None = None,
    **kwargs: Any,
) -> BaseChatModel:
    model, model_provider = _parse_model(model, model_provider)
    creator_func = _get_chat_model_creator(model_provider)
    return creator_func(model=model, **kwargs)


def _attempt_infer_model_provider(model_name: str) -> str | None:
    """Attempt to infer model provider from model name.

    Args:
        model_name: The name of the model to infer provider for.

    Returns:
        The inferred provider name, or `None` if no provider could be inferred.
    

    中文翻译:
    尝试从模型名称推断模型提供者。
    参数：
        model_name：要推断提供者的模型的名称。
    返回：
        推断的提供者名称，如果无法推断提供者，则为“无”。"""
    model_lower = model_name.lower()

    # OpenAI models (including newer models and aliases)
    # 中文: OpenAI 模型（包括较新的模型和别名）
    if any(
        model_lower.startswith(pre)
        for pre in (
            "gpt-",
            "o1",
            "o3",
            "chatgpt",
            "text-davinci",
        )
    ):
        return "openai"

    # Anthropic models
    # 中文: 人择模型
    if model_lower.startswith("claude"):
        return "anthropic"

    # Cohere models
    # 中文: 连贯模型
    if model_lower.startswith("command"):
        return "cohere"

    # Fireworks models
    # 中文: 烟花模型
    if model_name.startswith("accounts/fireworks"):
        return "fireworks"

    # Google models
    # 中文: 谷歌模型
    if model_lower.startswith("gemini"):
        return "google_vertexai"

    # AWS Bedrock models
    # 中文: AWS 基岩模型
    if model_name.startswith("amazon.") or model_lower.startswith(("anthropic.", "meta.")):
        return "bedrock"

    # Mistral models
    # 中文: 米斯特拉尔型号
    if model_lower.startswith(("mistral", "mixtral")):
        return "mistralai"

    # DeepSeek models
    # 中文: DeepSeek 模型
    if model_lower.startswith("deepseek"):
        return "deepseek"

    # xAI models
    # 中文: xAI模型
    if model_lower.startswith("grok"):
        return "xai"

    # Perplexity models
    # 中文: 困惑度模型
    if model_lower.startswith("sonar"):
        return "perplexity"

    # Upstage models
    # 中文: 抢镜模特
    if model_lower.startswith("solar"):
        return "upstage"

    return None


def _parse_model(model: str, model_provider: str | None) -> tuple[str, str]:
    """Parse model name and provider, inferring provider if necessary.

    中文翻译:
    解析模型名称和提供程序，必要时推断提供程序。"""
    # Handle provider:model format
    # 中文: 句柄提供者：模型格式
    if (
        not model_provider
        and ":" in model
        and model.split(":", maxsplit=1)[0] in _SUPPORTED_PROVIDERS
    ):
        model_provider = model.split(":", maxsplit=1)[0]
        model = ":".join(model.split(":")[1:])

    # Attempt to infer provider if not specified
    # 中文: 如果未指定，则尝试推断提供者
    model_provider = model_provider or _attempt_infer_model_provider(model)

    if not model_provider:
        # Enhanced error message with suggestions
        # 中文: 增强的错误消息和建议
        supported_list = ", ".join(sorted(_SUPPORTED_PROVIDERS))
        msg = (
            f"Unable to infer model provider for {model=}. "
            f"Please specify 'model_provider' directly.\n\n"
            f"Supported providers: {supported_list}\n\n"
            f"For help with specific providers, see: "
            f"https://docs.langchain.com/oss/python/integrations/providers"
        )
        raise ValueError(msg)

    # Normalize provider name
    # 中文: 规范化提供商名称
    model_provider = model_provider.replace("-", "_").lower()
    return model, model_provider


def _remove_prefix(s: str, prefix: str) -> str:
    return s.removeprefix(prefix)


_DECLARATIVE_METHODS = ("bind_tools", "with_structured_output")


class _ConfigurableModel(Runnable[LanguageModelInput, Any]):
    def __init__(
        self,
        *,
        default_config: dict | None = None,
        configurable_fields: Literal["any"] | list[str] | tuple[str, ...] = "any",
        config_prefix: str = "",
        queued_declarative_operations: Sequence[tuple[str, tuple, dict]] = (),
    ) -> None:
        self._default_config: dict = default_config or {}
        self._configurable_fields: Literal["any"] | list[str] = (
            "any" if configurable_fields == "any" else list(configurable_fields)
        )
        self._config_prefix = (
            config_prefix + "_"
            if config_prefix and not config_prefix.endswith("_")
            else config_prefix
        )
        self._queued_declarative_operations: list[tuple[str, tuple, dict]] = list(
            queued_declarative_operations,
        )

    def __getattr__(self, name: str) -> Any:
        if name in _DECLARATIVE_METHODS:
            # Declarative operations that cannot be applied until after an actual model
            # 中文: 在实际模型之后才能应用的声明性操作
            # object is instantiated. So instead of returning the actual operation,
            # 中文: 对象被实例化。所以不是返回实际操作，
            # we record the operation and its arguments in a queue. This queue is
            # 中文: 我们将操作及其参数记录在队列中。这个队列是
            # then applied in order whenever we actually instantiate the model (in
            # 中文: 然后在我们实际实例化模型时按顺序应用（在
            # self._model()).
            # 中文: self._model())。
            def queue(*args: Any, **kwargs: Any) -> _ConfigurableModel:
                queued_declarative_operations = list(
                    self._queued_declarative_operations,
                )
                queued_declarative_operations.append((name, args, kwargs))
                return _ConfigurableModel(
                    default_config=dict(self._default_config),
                    configurable_fields=list(self._configurable_fields)
                    if isinstance(self._configurable_fields, list)
                    else self._configurable_fields,
                    config_prefix=self._config_prefix,
                    queued_declarative_operations=queued_declarative_operations,
                )

            return queue
        if self._default_config and (model := self._model()) and hasattr(model, name):
            return getattr(model, name)
        msg = f"{name} is not a BaseChatModel attribute"
        if self._default_config:
            msg += " and is not implemented on the default model"
        msg += "."
        raise AttributeError(msg)

    def _model(self, config: RunnableConfig | None = None) -> Runnable:
        params = {**self._default_config, **self._model_params(config)}
        model = _init_chat_model_helper(**params)
        for name, args, kwargs in self._queued_declarative_operations:
            model = getattr(model, name)(*args, **kwargs)
        return model

    def _model_params(self, config: RunnableConfig | None) -> dict:
        config = ensure_config(config)
        model_params = {
            _remove_prefix(k, self._config_prefix): v
            for k, v in config.get("configurable", {}).items()
            if k.startswith(self._config_prefix)
        }
        if self._configurable_fields != "any":
            model_params = {k: v for k, v in model_params.items() if k in self._configurable_fields}
        return model_params

    def with_config(
        self,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> _ConfigurableModel:
        """Bind config to a `Runnable`, returning a new `Runnable`.

        中文翻译:
        将配置绑定到“Runnable”，返回一个新的“Runnable”。"""
        config = RunnableConfig(**(config or {}), **cast("RunnableConfig", kwargs))
        # Ensure config is not None after creation
        # 中文: 创建后确保配置不是 None
        config = ensure_config(config)
        model_params = self._model_params(config)
        remaining_config = {k: v for k, v in config.items() if k != "configurable"}
        remaining_config["configurable"] = {
            k: v
            for k, v in config.get("configurable", {}).items()
            if _remove_prefix(k, self._config_prefix) not in model_params
        }
        queued_declarative_operations = list(self._queued_declarative_operations)
        if remaining_config:
            queued_declarative_operations.append(
                (
                    "with_config",
                    (),
                    {"config": remaining_config},
                ),
            )
        return _ConfigurableModel(
            default_config={**self._default_config, **model_params},
            configurable_fields=list(self._configurable_fields)
            if isinstance(self._configurable_fields, list)
            else self._configurable_fields,
            config_prefix=self._config_prefix,
            queued_declarative_operations=queued_declarative_operations,
        )

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

    @override
    def invoke(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        return self._model(config).invoke(input, config=config, **kwargs)

    @override
    async def ainvoke(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        return await self._model(config).ainvoke(input, config=config, **kwargs)

    @override
    def stream(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> Iterator[Any]:
        yield from self._model(config).stream(input, config=config, **kwargs)

    @override
    async def astream(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> AsyncIterator[Any]:
        async for x in self._model(config).astream(input, config=config, **kwargs):
            yield x

    def batch(
        self,
        inputs: list[LanguageModelInput],
        config: RunnableConfig | list[RunnableConfig] | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any | None,
    ) -> list[Any]:
        config = config or None
        # If <= 1 config use the underlying models batch implementation.
        # 中文: 如果 <= 1 配置使用底层模型批量实现。
        if config is None or isinstance(config, dict) or len(config) <= 1:
            if isinstance(config, list):
                config = config[0]
            return self._model(config).batch(
                inputs,
                config=config,
                return_exceptions=return_exceptions,
                **kwargs,
            )
        # If multiple configs default to Runnable.batch which uses executor to invoke
        # 中文: 如果多个配置默认为 Runnable.batch 使用执行器调用
        # in parallel.
        # 中文: 并联。
        return super().batch(
            inputs,
            config=config,
            return_exceptions=return_exceptions,
            **kwargs,
        )

    async def abatch(
        self,
        inputs: list[LanguageModelInput],
        config: RunnableConfig | list[RunnableConfig] | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any | None,
    ) -> list[Any]:
        config = config or None
        # If <= 1 config use the underlying models batch implementation.
        # 中文: 如果 <= 1 配置使用底层模型批量实现。
        if config is None or isinstance(config, dict) or len(config) <= 1:
            if isinstance(config, list):
                config = config[0]
            return await self._model(config).abatch(
                inputs,
                config=config,
                return_exceptions=return_exceptions,
                **kwargs,
            )
        # If multiple configs default to Runnable.batch which uses executor to invoke
        # 中文: 如果多个配置默认为 Runnable.batch 使用执行器调用
        # in parallel.
        # 中文: 并联。
        return await super().abatch(
            inputs,
            config=config,
            return_exceptions=return_exceptions,
            **kwargs,
        )

    def batch_as_completed(
        self,
        inputs: Sequence[LanguageModelInput],
        config: RunnableConfig | Sequence[RunnableConfig] | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any,
    ) -> Iterator[tuple[int, Any | Exception]]:
        config = config or None
        # If <= 1 config use the underlying models batch implementation.
        # 中文: 如果 <= 1 配置使用底层模型批量实现。
        if config is None or isinstance(config, dict) or len(config) <= 1:
            if isinstance(config, list):
                config = config[0]
            yield from self._model(cast("RunnableConfig", config)).batch_as_completed(  # type: ignore[call-overload]
                inputs,
                config=config,
                return_exceptions=return_exceptions,
                **kwargs,
            )
        # If multiple configs default to Runnable.batch which uses executor to invoke
        # 中文: 如果多个配置默认为 Runnable.batch 使用执行器调用
        # in parallel.
        # 中文: 并联。
        else:
            yield from super().batch_as_completed(  # type: ignore[call-overload]
                inputs,
                config=config,
                return_exceptions=return_exceptions,
                **kwargs,
            )

    async def abatch_as_completed(
        self,
        inputs: Sequence[LanguageModelInput],
        config: RunnableConfig | Sequence[RunnableConfig] | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any,
    ) -> AsyncIterator[tuple[int, Any]]:
        config = config or None
        # If <= 1 config use the underlying models batch implementation.
        # 中文: 如果 <= 1 配置使用底层模型批量实现。
        if config is None or isinstance(config, dict) or len(config) <= 1:
            if isinstance(config, list):
                config = config[0]
            async for x in self._model(
                cast("RunnableConfig", config),
            ).abatch_as_completed(  # type: ignore[call-overload]
                inputs,
                config=config,
                return_exceptions=return_exceptions,
                **kwargs,
            ):
                yield x
        # If multiple configs default to Runnable.batch which uses executor to invoke
        # 中文: 如果多个配置默认为 Runnable.batch 使用执行器调用
        # in parallel.
        # 中文: 并联。
        else:
            async for x in super().abatch_as_completed(  # type: ignore[call-overload]
                inputs,
                config=config,
                return_exceptions=return_exceptions,
                **kwargs,
            ):
                yield x

    @override
    def transform(
        self,
        input: Iterator[LanguageModelInput],
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> Iterator[Any]:
        yield from self._model(config).transform(input, config=config, **kwargs)

    @override
    async def atransform(
        self,
        input: AsyncIterator[LanguageModelInput],
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
    ) -> AsyncIterator[Any]:
        async for x in self._model(config).atransform(input, config=config, **kwargs):
            yield x

    @overload
    @override
    def astream_log(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        *,
        diff: Literal[True] = True,
        with_streamed_output_list: bool = True,
        include_names: Sequence[str] | None = None,
        include_types: Sequence[str] | None = None,
        include_tags: Sequence[str] | None = None,
        exclude_names: Sequence[str] | None = None,
        exclude_types: Sequence[str] | None = None,
        exclude_tags: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[RunLogPatch]: ...

    @overload
    @override
    def astream_log(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        *,
        diff: Literal[False],
        with_streamed_output_list: bool = True,
        include_names: Sequence[str] | None = None,
        include_types: Sequence[str] | None = None,
        include_tags: Sequence[str] | None = None,
        exclude_names: Sequence[str] | None = None,
        exclude_types: Sequence[str] | None = None,
        exclude_tags: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[RunLog]: ...

    @override
    async def astream_log(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        *,
        diff: bool = True,
        with_streamed_output_list: bool = True,
        include_names: Sequence[str] | None = None,
        include_types: Sequence[str] | None = None,
        include_tags: Sequence[str] | None = None,
        exclude_names: Sequence[str] | None = None,
        exclude_types: Sequence[str] | None = None,
        exclude_tags: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[RunLogPatch] | AsyncIterator[RunLog]:
        async for x in self._model(config).astream_log(  # type: ignore[call-overload, misc]
            input,
            config=config,
            diff=diff,
            with_streamed_output_list=with_streamed_output_list,
            include_names=include_names,
            include_types=include_types,
            include_tags=include_tags,
            exclude_tags=exclude_tags,
            exclude_types=exclude_types,
            exclude_names=exclude_names,
            **kwargs,
        ):
            yield x

    @override
    async def astream_events(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        *,
        version: Literal["v1", "v2"] = "v2",
        include_names: Sequence[str] | None = None,
        include_types: Sequence[str] | None = None,
        include_tags: Sequence[str] | None = None,
        exclude_names: Sequence[str] | None = None,
        exclude_types: Sequence[str] | None = None,
        exclude_tags: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        async for x in self._model(config).astream_events(
            input,
            config=config,
            version=version,
            include_names=include_names,
            include_types=include_types,
            include_tags=include_tags,
            exclude_tags=exclude_tags,
            exclude_types=exclude_types,
            exclude_names=exclude_names,
            **kwargs,
        ):
            yield x

    # Explicitly added to satisfy downstream linters.
    # 中文: 显式添加以满足下游 linter。
    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type[BaseModel] | Callable | BaseTool],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        return self.__getattr__("bind_tools")(tools, **kwargs)

    # Explicitly added to satisfy downstream linters.
    # 中文: 显式添加以满足下游 linter。
    def with_structured_output(
        self,
        schema: dict | type[BaseModel],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, dict | BaseModel]:
        return self.__getattr__("with_structured_output")(schema, **kwargs)
