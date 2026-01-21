"""管理聊天消息历史的 Runnable 模块。

本模块提供 `RunnableWithMessageHistory`，用于自动管理对话历史。

核心功能:
---------
自动读取、更新和持久化聊天消息历史，使 LLM 能够记住之前的对话内容。

常用场景:
---------
1. 构建具有记忆功能的聊天机器人
2. 多轮对话应用
3. 需要上下文感知的 AI 助手

关键概念:
---------
- session_id: 唯一标识一个对话会话
- BaseChatMessageHistory: 消息历史存储的抽象接口
- 支持同步和异步操作

使用示例:
---------
>>> from langchain_core.runnables.history import RunnableWithMessageHistory
>>> from langchain_core.chat_history import InMemoryChatMessageHistory
>>>
>>> # 定义获取会话历史的函数
>>> store = {}
>>> def get_session_history(session_id: str):
...     if session_id not in store:
...         store[session_id] = InMemoryChatMessageHistory()
...     return store[session_id]
>>>
>>> # 包装你的链
>>> with_history = RunnableWithMessageHistory(
...     runnable=your_chain,
...     get_session_history=get_session_history,
...     input_messages_key="question",
...     history_messages_key="history",
... )
>>>
>>> # 调用时指定 session_id
>>> with_history.invoke(
...     {"question": "你好！"},
...     config={"configurable": {"session_id": "user-123"}}
... )
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from types import GenericAlias
from typing import (
    TYPE_CHECKING,
    Any,
)

from pydantic import BaseModel
from typing_extensions import override

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.load.load import load
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables.base import Runnable, RunnableBindingBase, RunnableLambda
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_core.runnables.utils import (
    ConfigurableFieldSpec,
    Output,
    get_unique_config_specs,
)
from langchain_core.utils.pydantic import create_model_v2

if TYPE_CHECKING:
    from langchain_core.language_models.base import LanguageModelLike
    from langchain_core.runnables.config import RunnableConfig
    from langchain_core.tracers.schemas import Run


MessagesOrDictWithMessages = Sequence["BaseMessage"] | dict[str, Any]
GetSessionHistoryCallable = Callable[..., BaseChatMessageHistory]


class RunnableWithMessageHistory(RunnableBindingBase):  # type: ignore[no-redef]
    """为另一个 Runnable 管理聊天消息历史的 Runnable。

    聊天消息历史是代表对话的消息序列。

    `RunnableWithMessageHistory` 包装另一个 Runnable 并为其管理聊天消息历史；
    它负责读取和更新聊天消息历史。

    核心功能:
    ---------
    1. 自动加载历史消息并注入到输入中
    2. 自动保存新的输入和输出消息到历史
    3. 支持自定义 session_id 和多租户场景

    调用要求:
    ---------
    必须在调用时提供包含会话工厂参数的配置。
    默认情况下，需要一个名为 `session_id` 的字符串参数。

    调用示例:
        `with_history.invoke(..., config={"configurable": {"session_id": "bar"}})`

    输入格式:
    ---------
    1. BaseMessage 列表
    2. 包含消息键的字典
    3. 包含当前输入和历史消息分开键的字典

    输出格式:
    ---------
    1. 可作为 AIMessage 的字符串
    2. BaseMessage 或 BaseMessage 序列
    3. 包含 BaseMessage 的字典

    属性:
    -----
    get_session_history : Callable
        返回新的 BaseChatMessageHistory 的函数
    input_messages_key : str | None
        输入字典中包含消息的键
    output_messages_key : str | None
        输出字典中包含消息的键
    history_messages_key : str | None
        输入字典中放置历史消息的键
    history_factory_config : Sequence[ConfigurableFieldSpec]
        传递给历史工厂的配置字段

    基础示例（字典输入）:
        ```python
        from langchain_anthropic import ChatAnthropic
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        from langchain_core.runnables.history import RunnableWithMessageHistory

        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个擅长{ability}的助手"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ])

        chain = prompt | ChatAnthropic(model="claude-3-haiku")

        chain_with_history = RunnableWithMessageHistory(
            chain,
            get_by_session_id,  # 定义的会话历史获取函数
            input_messages_key="question",
            history_messages_key="history",
        )

        chain_with_history.invoke(
            {"ability": "数学", "question": "余弦是什么意思？"},
            config={"configurable": {"session_id": "user-123"}},
        )
        ```

    多键工厂示例（使用 user_id 和 conversation_id）:
        ```python
        def get_session_history(user_id: str, conversation_id: str):
            key = (user_id, conversation_id)
            if key not in store:
                store[key] = InMemoryHistory()
            return store[key]

        with_message_history = RunnableWithMessageHistory(
            chain,
            get_session_history=get_session_history,
            input_messages_key="question",
            history_messages_key="history",
            history_factory_config=[
                ConfigurableFieldSpec(
                    id="user_id",
                    annotation=str,
                    name="用户 ID",
                    description="用户的唯一标识符。",
                ),
                ConfigurableFieldSpec(
                    id="conversation_id",
                    annotation=str,
                    name="对话 ID",
                    description="对话的唯一标识符。",
                ),
            ],
        )

        with_message_history.invoke(
            {"ability": "数学", "question": "余弦是什么意思？"},
            config={"configurable": {"user_id": "123", "conversation_id": "1"}},
        )
        ```
    """

    get_session_history: GetSessionHistoryCallable
    """Function that returns a new `BaseChatMessageHistory`.

    This function should either take a single positional argument `session_id` of type
    string and return a corresponding chat message history instance
    """
    input_messages_key: str | None = None
    """Must be specified if the base `Runnable` accepts a `dict` as input.
    The key in the input `dict` that contains the messages.
    """
    output_messages_key: str | None = None
    """Must be specified if the base `Runnable` returns a `dict` as output.
    The key in the output `dict` that contains the messages.
    """
    history_messages_key: str | None = None
    """Must be specified if the base `Runnable` accepts a `dict` as input and expects a
    separate key for historical messages.
    """
    history_factory_config: Sequence[ConfigurableFieldSpec]
    """Configure fields that should be passed to the chat history factory.

    See `ConfigurableFieldSpec` for more details.
    """

    def __init__(
        self,
        runnable: Runnable[
            list[BaseMessage], str | BaseMessage | MessagesOrDictWithMessages
        ]
        | Runnable[dict[str, Any], str | BaseMessage | MessagesOrDictWithMessages]
        | LanguageModelLike,
        get_session_history: GetSessionHistoryCallable,
        *,
        input_messages_key: str | None = None,
        output_messages_key: str | None = None,
        history_messages_key: str | None = None,
        history_factory_config: Sequence[ConfigurableFieldSpec] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize `RunnableWithMessageHistory`.

        Args:
            runnable: The base `Runnable` to be wrapped.

                Must take as input one of:

                1. A list of `BaseMessage`
                2. A `dict` with one key for all messages
                3. A `dict` with one key for the current input string/message(s) and
                    a separate key for historical messages. If the input key points
                    to a string, it will be treated as a `HumanMessage` in history.

                Must return as output one of:

                1. A string which can be treated as an `AIMessage`
                2. A `BaseMessage` or sequence of `BaseMessage`
                3. A `dict` with a key for a `BaseMessage` or sequence of
                    `BaseMessage`

            get_session_history: Function that returns a new `BaseChatMessageHistory`.

                This function should either take a single positional argument
                `session_id` of type string and return a corresponding
                chat message history instance.

                ```python
                def get_session_history(
                    session_id: str, *, user_id: str | None = None
                ) -> BaseChatMessageHistory: ...
                ```

                Or it should take keyword arguments that match the keys of
                `session_history_config_specs` and return a corresponding
                chat message history instance.

                ```python
                def get_session_history(
                    *,
                    user_id: str,
                    thread_id: str,
                ) -> BaseChatMessageHistory: ...
                ```

            input_messages_key: Must be specified if the base runnable accepts a `dict`
                as input.
            output_messages_key: Must be specified if the base runnable returns a `dict`
                as output.
            history_messages_key: Must be specified if the base runnable accepts a
                `dict` as input and expects a separate key for historical messages.
            history_factory_config: Configure fields that should be passed to the
                chat history factory. See `ConfigurableFieldSpec` for more details.

                Specifying these allows you to pass multiple config keys into the
                `get_session_history` factory.
            **kwargs: Arbitrary additional kwargs to pass to parent class
                `RunnableBindingBase` init.

        """
        history_chain: Runnable = RunnableLambda(
            self._enter_history, self._aenter_history
        ).with_config(run_name="load_history")
        messages_key = history_messages_key or input_messages_key
        if messages_key:
            history_chain = RunnablePassthrough.assign(
                **{messages_key: history_chain}
            ).with_config(run_name="insert_history")

        runnable_sync: Runnable = runnable.with_listeners(on_end=self._exit_history)
        runnable_async: Runnable = runnable.with_alisteners(on_end=self._aexit_history)

        def _call_runnable_sync(_input: Any) -> Runnable:
            return runnable_sync

        async def _call_runnable_async(_input: Any) -> Runnable:
            return runnable_async

        bound: Runnable = (
            history_chain
            | RunnableLambda(
                _call_runnable_sync,
                _call_runnable_async,
            ).with_config(run_name="check_sync_or_async")
        ).with_config(run_name="RunnableWithMessageHistory")

        if history_factory_config:
            config_specs = history_factory_config
        else:
            # If not provided, then we'll use the default session_id field
            config_specs = [
                ConfigurableFieldSpec(
                    id="session_id",
                    annotation=str,
                    name="Session ID",
                    description="Unique identifier for a session.",
                    default="",
                    is_shared=True,
                ),
            ]

        super().__init__(
            get_session_history=get_session_history,
            input_messages_key=input_messages_key,
            output_messages_key=output_messages_key,
            bound=bound,
            history_messages_key=history_messages_key,
            history_factory_config=config_specs,
            **kwargs,
        )
        self._history_chain = history_chain

    @property
    @override
    def config_specs(self) -> list[ConfigurableFieldSpec]:
        """Get the configuration specs for the `RunnableWithMessageHistory`."""
        return get_unique_config_specs(
            super().config_specs + list(self.history_factory_config)
        )

    @override
    def get_input_schema(self, config: RunnableConfig | None = None) -> type[BaseModel]:
        fields: dict = {}
        if self.input_messages_key and self.history_messages_key:
            fields[self.input_messages_key] = (
                str | BaseMessage | Sequence[BaseMessage],
                ...,
            )
        elif self.input_messages_key:
            fields[self.input_messages_key] = (Sequence[BaseMessage], ...)
        else:
            return create_model_v2(
                "RunnableWithChatHistoryInput",
                module_name=self.__class__.__module__,
                root=(Sequence[BaseMessage], ...),
            )
        return create_model_v2(
            "RunnableWithChatHistoryInput",
            field_definitions=fields,
            module_name=self.__class__.__module__,
        )

    @property
    @override
    def OutputType(self) -> type[Output]:
        return self._history_chain.OutputType

    @override
    def get_output_schema(
        self, config: RunnableConfig | None = None
    ) -> type[BaseModel]:
        """Get a Pydantic model that can be used to validate output to the `Runnable`.

        `Runnable` objects that leverage the `configurable_fields` and
        `configurable_alternatives` methods will have a dynamic output schema that
        depends on which configuration the `Runnable` is invoked with.

        This method allows to get an output schema for a specific configuration.

        Args:
            config: A config to use when generating the schema.

        Returns:
            A Pydantic model that can be used to validate output.
        """
        root_type = self.OutputType

        if (
            inspect.isclass(root_type)
            and not isinstance(root_type, GenericAlias)
            and issubclass(root_type, BaseModel)
        ):
            return root_type

        return create_model_v2(
            "RunnableWithChatHistoryOutput",
            root=root_type,
            module_name=self.__class__.__module__,
        )

    def _get_input_messages(
        self, input_val: str | BaseMessage | Sequence[BaseMessage] | dict
    ) -> list[BaseMessage]:
        # If dictionary, try to pluck the single key representing messages
        if isinstance(input_val, dict):
            if self.input_messages_key:
                key = self.input_messages_key
            elif len(input_val) == 1:
                key = next(iter(input_val.keys()))
            else:
                key = "input"
            input_val = input_val[key]

        # If value is a string, convert to a human message
        if isinstance(input_val, str):
            return [HumanMessage(content=input_val)]
        # If value is a single message, convert to a list
        if isinstance(input_val, BaseMessage):
            return [input_val]
        # If value is a list or tuple...
        if isinstance(input_val, (list, tuple)):
            # Handle empty case
            if len(input_val) == 0:
                return list(input_val)
            # If is a list of list, then return the first value
            # This occurs for chat models - since we batch inputs
            if isinstance(input_val[0], list):
                if len(input_val) != 1:
                    msg = f"Expected a single list of messages. Got {input_val}."
                    raise ValueError(msg)
                return input_val[0]
            return list(input_val)
        msg = (
            f"Expected str, BaseMessage, list[BaseMessage], or tuple[BaseMessage]. "
            f"Got {input_val}."
        )
        raise ValueError(msg)

    def _get_output_messages(
        self, output_val: str | BaseMessage | Sequence[BaseMessage] | dict
    ) -> list[BaseMessage]:
        # If dictionary, try to pluck the single key representing messages
        if isinstance(output_val, dict):
            if self.output_messages_key:
                key = self.output_messages_key
            elif len(output_val) == 1:
                key = next(iter(output_val.keys()))
            else:
                key = "output"
            # If you are wrapping a chat model directly
            # The output is actually this weird generations object
            if key not in output_val and "generations" in output_val:
                output_val = output_val["generations"][0][0]["message"]
            else:
                output_val = output_val[key]

        if isinstance(output_val, str):
            return [AIMessage(content=output_val)]
        # If value is a single message, convert to a list
        if isinstance(output_val, BaseMessage):
            return [output_val]
        if isinstance(output_val, (list, tuple)):
            return list(output_val)
        msg = (
            f"Expected str, BaseMessage, list[BaseMessage], or tuple[BaseMessage]. "
            f"Got {output_val}."
        )
        raise ValueError(msg)

    def _enter_history(self, value: Any, config: RunnableConfig) -> list[BaseMessage]:
        hist: BaseChatMessageHistory = config["configurable"]["message_history"]
        messages = hist.messages.copy()

        if not self.history_messages_key:
            # return all messages
            input_val = (
                value if not self.input_messages_key else value[self.input_messages_key]
            )
            messages += self._get_input_messages(input_val)
        return messages

    async def _aenter_history(
        self, value: dict[str, Any], config: RunnableConfig
    ) -> list[BaseMessage]:
        hist: BaseChatMessageHistory = config["configurable"]["message_history"]
        messages = (await hist.aget_messages()).copy()

        if not self.history_messages_key:
            # return all messages
            input_val = (
                value if not self.input_messages_key else value[self.input_messages_key]
            )
            messages += self._get_input_messages(input_val)
        return messages

    def _exit_history(self, run: Run, config: RunnableConfig) -> None:
        hist: BaseChatMessageHistory = config["configurable"]["message_history"]

        # Get the input messages
        inputs = load(run.inputs, allowed_objects="all")
        input_messages = self._get_input_messages(inputs)
        # If historic messages were prepended to the input messages, remove them to
        # avoid adding duplicate messages to history.
        if not self.history_messages_key:
            historic_messages = config["configurable"]["message_history"].messages
            input_messages = input_messages[len(historic_messages) :]

        # Get the output messages
        output_val = load(run.outputs, allowed_objects="all")
        output_messages = self._get_output_messages(output_val)
        hist.add_messages(input_messages + output_messages)

    async def _aexit_history(self, run: Run, config: RunnableConfig) -> None:
        hist: BaseChatMessageHistory = config["configurable"]["message_history"]

        # Get the input messages
        inputs = load(run.inputs, allowed_objects="all")
        input_messages = self._get_input_messages(inputs)
        # If historic messages were prepended to the input messages, remove them to
        # avoid adding duplicate messages to history.
        if not self.history_messages_key:
            historic_messages = await hist.aget_messages()
            input_messages = input_messages[len(historic_messages) :]

        # Get the output messages
        output_val = load(run.outputs, allowed_objects="all")
        output_messages = self._get_output_messages(output_val)
        await hist.aadd_messages(input_messages + output_messages)

    def _merge_configs(self, *configs: RunnableConfig | None) -> RunnableConfig:
        config = super()._merge_configs(*configs)
        expected_keys = [field_spec.id for field_spec in self.history_factory_config]

        configurable = config.get("configurable", {})

        missing_keys = set(expected_keys) - set(configurable.keys())
        parameter_names = _get_parameter_names(self.get_session_history)

        if missing_keys and parameter_names:
            example_input = {self.input_messages_key: "foo"}
            example_configurable = dict.fromkeys(missing_keys, "[your-value-here]")
            example_config = {"configurable": example_configurable}
            msg = (
                f"Missing keys {sorted(missing_keys)} in config['configurable'] "
                f"Expected keys are {sorted(expected_keys)}."
                f"When using via .invoke() or .stream(), pass in a config; "
                f"e.g., chain.invoke({example_input}, {example_config})"
            )
            raise ValueError(msg)

        if len(expected_keys) == 1:
            if parameter_names:
                # If arity = 1, then invoke function by positional arguments
                message_history = self.get_session_history(
                    configurable[expected_keys[0]]
                )
            else:
                if not config:
                    config["configurable"] = {}
                message_history = self.get_session_history()
        else:
            # otherwise verify that names of keys patch and invoke by named arguments
            if set(expected_keys) != set(parameter_names):
                msg = (
                    f"Expected keys {sorted(expected_keys)} do not match parameter "
                    f"names {sorted(parameter_names)} of get_session_history."
                )
                raise ValueError(msg)

            message_history = self.get_session_history(
                **{key: configurable[key] for key in expected_keys}
            )
        config["configurable"]["message_history"] = message_history
        return config


def _get_parameter_names(callable_: GetSessionHistoryCallable) -> list[str]:
    """Get the parameter names of the `Callable`."""
    sig = inspect.signature(callable_)
    return list(sig.parameters.keys())
