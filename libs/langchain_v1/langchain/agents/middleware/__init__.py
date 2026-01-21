"""Agent 中间件系统入口模块。

本模块是使用中间件（Middleware）插件与 Agent 配合使用的入口点。

什么是中间件:
-------------
中间件是可以拦截和修改 Agent 行为的组件，它们可以：
- 在 Agent/模型调用前后执行自定义逻辑
- 包装模型调用以添加重试、回退等功能
- 包装工具调用以添加验证、限制等功能
- 修改状态以影响后续处理

内置中间件:
-----------
**执行控制:**
- `ModelCallLimitMiddleware`: 限制模型调用次数
- `ToolCallLimitMiddleware`: 限制工具调用次数
- `ModelRetryMiddleware`: 模型调用重试
- `ToolRetryMiddleware`: 工具调用重试
- `ModelFallbackMiddleware`: 模型回退

**人机交互:**
- `HumanInTheLoopMiddleware`: 人工介入审批

**数据处理:**
- `PIIMiddleware`: 个人身份信息检测
- `SummarizationMiddleware`: 对话摘要
- `ContextEditingMiddleware`: 上下文编辑

**工具相关:**
- `ShellToolMiddleware`: Shell 命令执行
- `LLMToolEmulator`: LLM 工具模拟
- `LLMToolSelectorMiddleware`: 工具选择
- `FilesystemFileSearchMiddleware`: 文件搜索
- `TodoListMiddleware`: 待办事项管理

核心类型:
---------
- `AgentMiddleware`: 中间件基类
- `AgentState`: Agent 状态类型
- `ModelRequest`: 模型请求
- `ModelResponse`: 模型响应

钩子装饰器:
-----------
- `@before_agent`: Agent 执行前
- `@after_agent`: Agent 执行后
- `@before_model`: 模型调用前
- `@after_model`: 模型调用后
- `@wrap_model_call`: 包装模型调用
- `@wrap_tool_call`: 包装工具调用
- `@dynamic_prompt`: 动态系统提示

更多信息请参阅: https://docs.langchain.com/oss/python/langchain/middleware
"""

from .context_editing import (
    ClearToolUsesEdit,
    ContextEditingMiddleware,
)
from .file_search import FilesystemFileSearchMiddleware
from .human_in_the_loop import (
    HumanInTheLoopMiddleware,
    InterruptOnConfig,
)
from .model_call_limit import ModelCallLimitMiddleware
from .model_fallback import ModelFallbackMiddleware
from .model_retry import ModelRetryMiddleware
from .pii import PIIDetectionError, PIIMiddleware
from .shell_tool import (
    CodexSandboxExecutionPolicy,
    DockerExecutionPolicy,
    HostExecutionPolicy,
    RedactionRule,
    ShellToolMiddleware,
)
from .summarization import SummarizationMiddleware
from .todo import TodoListMiddleware
from .tool_call_limit import ToolCallLimitMiddleware
from .tool_emulator import LLMToolEmulator
from .tool_retry import ToolRetryMiddleware
from .tool_selection import LLMToolSelectorMiddleware
from .types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
    after_agent,
    after_model,
    before_agent,
    before_model,
    dynamic_prompt,
    hook_config,
    wrap_model_call,
    wrap_tool_call,
)

__all__ = [
    "AgentMiddleware",
    "AgentState",
    "ClearToolUsesEdit",
    "CodexSandboxExecutionPolicy",
    "ContextEditingMiddleware",
    "DockerExecutionPolicy",
    "FilesystemFileSearchMiddleware",
    "HostExecutionPolicy",
    "HumanInTheLoopMiddleware",
    "InterruptOnConfig",
    "LLMToolEmulator",
    "LLMToolSelectorMiddleware",
    "ModelCallLimitMiddleware",
    "ModelFallbackMiddleware",
    "ModelRequest",
    "ModelResponse",
    "ModelRetryMiddleware",
    "PIIDetectionError",
    "PIIMiddleware",
    "RedactionRule",
    "ShellToolMiddleware",
    "SummarizationMiddleware",
    "TodoListMiddleware",
    "ToolCallLimitMiddleware",
    "ToolRetryMiddleware",
    "after_agent",
    "after_model",
    "before_agent",
    "before_model",
    "dynamic_prompt",
    "hook_config",
    "wrap_model_call",
    "wrap_tool_call",
]
