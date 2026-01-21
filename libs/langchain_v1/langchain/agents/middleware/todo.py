"""待办事项管理中间件模块。

本模块提供 Agent 任务规划和进度跟踪能力。

核心类:
--------
**TodoListMiddleware**: 待办事项中间件

提供的工具:
-----------
- `write_todos`: 创建和管理结构化任务列表

任务状态:
---------
- `pending`: 待处理
- `in_progress`: 进行中
- `completed`: 已完成

使用示例:
---------
>>> from langchain.agents import create_agent
>>> from langchain.agents.middleware import TodoListMiddleware
>>>
>>> todo = TodoListMiddleware()
>>>
>>> agent = create_agent(
...     model="openai:gpt-4o",
...     middleware=[todo],
... )
>>>
>>> # Agent 可以使用 write_todos 工具跟踪任务进度
>>> result = agent.invoke({"messages": [...]})
>>> print(result.get("todos"))
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal, cast

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.types import Command
from typing_extensions import NotRequired, TypedDict

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
    OmitFromInput,
)
from langchain.tools import InjectedToolCallId


class Todo(TypedDict):
    """A single todo item with content and status.

    中文翻译:
    包含内容和状态的单个待办事项。"""

    content: str
    """The content/description of the todo item.

    中文翻译:
    待办事项的内容/描述。"""

    status: Literal["pending", "in_progress", "completed"]
    """The current status of the todo item.

    中文翻译:
    待办事项的当前状态。"""


class PlanningState(AgentState):
    """State schema for the todo middleware.

    中文翻译:
    todo 中间件的状态模式。"""

    todos: Annotated[NotRequired[list[Todo]], OmitFromInput]
    """List of todo items for tracking task progress.

    中文翻译:
    用于跟踪任务进度的待办事项列表。"""


WRITE_TODOS_TOOL_DESCRIPTION = """Use this tool to create and manage a structured task list for your current work session. This helps you track progress, organize complex tasks, and demonstrate thoroughness to the user.

Only use this tool if you think it will be helpful in staying organized. If the user's request is trivial and takes less than 3 steps, it is better to NOT use this tool and just do the task directly.

## When to Use This Tool
#中文: # 何时使用此工具
Use this tool in these scenarios:

1. Complex multi-step tasks - When a task requires 3 or more distinct steps or actions
2. Non-trivial and complex tasks - Tasks that require careful planning or multiple operations
3. User explicitly requests todo list - When the user directly asks you to use the todo list
4. User provides multiple tasks - When users provide a list of things to be done (numbered or comma-separated)
5. The plan may need future revisions or updates based on results from the first few steps

## How to Use This Tool
#中文: # 如何使用这个工具
1. When you start working on a task - Mark it as in_progress BEFORE beginning work.
2. After completing a task - Mark it as completed and add any new follow-up tasks discovered during implementation.
3. You can also update future tasks, such as deleting them if they are no longer necessary, or adding new tasks that are necessary. Don't change previously completed tasks.
4. You can make several updates to the todo list at once. For example, when you complete a task, you can mark the next task you need to start as in_progress.

## When NOT to Use This Tool
#中文: # 何时不使用此工具
It is important to skip using this tool when:
1. There is only a single, straightforward task
2. The task is trivial and tracking it provides no benefit
3. The task can be completed in less than 3 trivial steps
4. The task is purely conversational or informational

## Task States and Management
#中文: # 任务状态和管理

1. **Task States**: Use these states to track progress:
   - pending: Task not yet started
   - in_progress: Currently working on (you can have multiple tasks in_progress at a time if they are not related to each other and can be run in parallel)
   - completed: Task finished successfully

2. **Task Management**:
   - Update task status in real-time as you work
   - Mark tasks complete IMMEDIATELY after finishing (don't batch completions)
   - Complete current tasks before starting new ones
   - Remove tasks that are no longer relevant from the list entirely
   - IMPORTANT: When you write this todo list, you should mark your first task (or tasks) as in_progress immediately!.
   - IMPORTANT: Unless all tasks are completed, you should always have at least one task in_progress to show the user that you are working on something.

3. **Task Completion Requirements**:
   - ONLY mark a task as completed when you have FULLY accomplished it
   - If you encounter errors, blockers, or cannot finish, keep the task as in_progress
   - When blocked, create a new task describing what needs to be resolved
   - Never mark a task as completed if:
     - There are unresolved issues or errors
     - Work is partial or incomplete
     - You encountered blockers that prevent completion
     - You couldn't find necessary resources or dependencies
     - Quality standards haven't been met

4. **Task Breakdown**:
   - Create specific, actionable items
   - Break complex tasks into smaller, manageable steps
   - Use clear, descriptive task names

Being proactive with task management demonstrates attentiveness and ensures you complete all requirements successfully
Remember: If you only need to make a few tool calls to complete a task, and it is clear what you need to do, it is better to just do the task directly and NOT call this tool at all.

 中文翻译:
 使用此工具为您当前的工作会话创建和管理结构化任务列表。这可以帮助您跟踪进度、组织复杂的任务并向用户展示彻底性。
仅当您认为此工具有助于保持井井有条时才使用此工具。如果用户的请求很简单并且需要的步骤少于 3 个，那么最好不要使用此工具，直接执行任务。
## 何时使用此工具
在以下场景中使用此工具：
1. 复杂的多步骤任务 - 当一项任务需要 3 个或更多不同的步骤或操作时
2. 不平凡且复杂的任务 - 需要仔细计划或多次操作的任务
3. 用户明确请求待办事项列表 - 当用户直接要求您使用待办事项列表时
4. 用户提供多项任务 - 当用户提供要完成的事情列表时（编号或逗号分隔）
5. 该计划未来可能需要根据前几个步骤的结果进行修改或更新
## 如何使用这个工具
1. 当您开始处理某项任务时 - 在开始工作之前将其标记为 in_progress。
2. 完成任务后 - 将其标记为已完成并添加在实施过程中发现的任何新的后续任务。
3. 您还可以更新未来的任务，例如删除不再需要的任务，或添加需要的新任务。不要更改以前完成的任务。
4. 您可以一次对待办事项列表进行多项更新。例如，当您完成一项任务时，您可以将需要启动的下一个任务标记为 in_progress。
## 何时不使用此工具
在以下情况下，请务必跳过使用此工具：
1.只有一个简单的任务
2. 任务很琐碎，跟踪它没有任何好处
3. 任务只需不到 3 个简单步骤即可完成
4.任务纯粹是对话性的或信息性的
## 任务状态和管理
1. **任务状态**：使用这些状态来跟踪进度：
   -待定：任务尚未开始
   - in_progress：当前正在处理（如果它们彼此不相关并且可以并行运行，则一次可以有多个任务 in_progress）
   - 完成：任务成功完成
2. **任务管理**：
   - 在工作时实时更新任务状态
   - 完成后立即标记任务完成（不要批量完成）
   - 在开始新任务之前完成当前任务
   - 从列表中完全删除不再相关的任务
   - 重要提示：当您编写此待办事项列表时，您应该立即将您的第一个任务（或多个任务）标记为 in_progress！
   - 重要提示：除非所有任务都已完成，否则您应该始终至少有一项任务正在进行中，以向用户表明您正在处理某件事。
3. **任务完成要求**：
   - 仅当您完全完成任务时才将其标记为已完成
   - 如果遇到错误、阻碍或无法完成，请将任务保持为进行中
   - 当被阻止时，创建一个新任务来描述需要解决的问题
   - 在以下情况下切勿将任务标记为已完成：
     - 存在未解决的问题或错误
     - 工作不完整或不完整
     - 您遇到了阻碍完成的障碍
     - 您找不到必要的资源或依赖项
     - 未达到质量标准
4. **任务分解**：
   - 创建具体的、可操作的项目
   - 将复杂的任务分解为更小的、可管理的步骤
   - 使用清晰、描述性的任务名称
积极主动地进行任务管理表明您的专注并确保您成功完成所有要求
请记住：如果您只需要调用几个工具即可完成任务，并且很清楚您需要做什么，那么最好直接执行任务，而根本不调用此工具。"""  # noqa: E501

WRITE_TODOS_SYSTEM_PROMPT = """## `write_todos`

You have access to the `write_todos` tool to help you manage and plan complex objectives.
Use this tool for complex objectives to ensure that you are tracking each necessary step and giving the user visibility into your progress.
This tool is very helpful for planning complex objectives, and for breaking down these larger complex objectives into smaller steps.

It is critical that you mark todos as completed as soon as you are done with a step. Do not batch up multiple steps before marking them as completed.
For simple objectives that only require a few steps, it is better to just complete the objective directly and NOT use this tool.
Writing todos takes time and tokens, use it when it is helpful for managing complex many-step problems! But not for simple few-step requests.

## Important To-Do List Usage Notes to Remember
#中文: # 需要记住的重要待办事项列表使用说明
- The `write_todos` tool should never be called multiple times in parallel.
- Don't be afraid to revise the To-Do list as you go. New information may reveal new tasks that need to be done, or old tasks that are irrelevant.

 中文翻译:
 ## `write_todos`
 #中文: # `write_todos`
您可以使用“write_todos”工具来帮助您管理和规划复杂的目标。
使用此工具来实现复杂的目标，以确保您跟踪每个必要的步骤并让用户了解您的进度。
该工具对于规划复杂的目标以及将这些较大的复杂目标分解为较小的步骤非常有帮助。
完成某个步骤后立即将待办事项标记为已完成非常重要。在将多个步骤标记为已完成之前，请勿将其批处理。
对于只需要几个步骤的简单目标，最好直接完成目标而不使用此工具。
编写待办事项需要时间和令牌，当它有助于管理复杂的多步骤问题时使用它！但不适用于简单的几步请求。
## 需要记住的重要待办事项列表使用说明
- `write_todos` 工具不应该被并行调用多次。
- 不要害怕随时修改待办事项列表。新信息可能揭示需要完成的新任务，或不相关的旧任务。"""  # noqa: E501


@tool(description=WRITE_TODOS_TOOL_DESCRIPTION)
def write_todos(todos: list[Todo], tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """Create and manage a structured task list for your current work session.

    中文翻译:
    为当前工作会话创建和管理结构化任务列表。"""
    return Command(
        update={
            "todos": todos,
            "messages": [ToolMessage(f"Updated todo list to {todos}", tool_call_id=tool_call_id)],
        }
    )


class TodoListMiddleware(AgentMiddleware):
    """Middleware that provides todo list management capabilities to agents.

    This middleware adds a `write_todos` tool that allows agents to create and manage
    structured task lists for complex multi-step operations. It's designed to help
    agents track progress, organize complex tasks, and provide users with visibility
    into task completion status.

    The middleware automatically injects system prompts that guide the agent on when
    and how to use the todo functionality effectively.

    Example:
        ```python
        from langchain.agents.middleware.todo import TodoListMiddleware
        from langchain.agents import create_agent

        agent = create_agent("openai:gpt-4o", middleware=[TodoListMiddleware()])

        # Agent now has access to write_todos tool and todo state tracking
        # 中文: 代理现在可以访问 write_todos 工具和 todo 状态跟踪
        result = await agent.invoke({"messages": [HumanMessage("Help me refactor my codebase")]})

        print(result["todos"])  # Array of todo items with status tracking
        ```
    

    中文翻译:
    为代理提供待办事项列表管理功能的中间件。
    该中间件添加了一个“write_todos”工具，允许代理创建和管理
    用于复杂的多步骤操作的结构化任务列表。它旨在帮助
    代理跟踪进度、组织复杂的任务并为用户提供可见性
    进入任务完成状态。
    中间件会自动注入系统提示，指导代理何时
    以及如何有效地使用待办事项功能。
    示例：
        ````蟒蛇
        从 langchain.agents.middleware.todo 导入 TodoListMiddleware
        从 langchain.agents 导入 create_agent
        代理 = create_agent("openai:gpt-4o", middleware=[TodoListMiddleware()])
        # 代理现在可以访问 write_todos 工具和 todo 状态跟踪
        result = wait agent.invoke({"messages": [HumanMessage("帮我重构我的代码库")]})
        print(result["todos"]) # 带有状态跟踪的待办事项数组
        ````"""

    state_schema = PlanningState

    def __init__(
        self,
        *,
        system_prompt: str = WRITE_TODOS_SYSTEM_PROMPT,
        tool_description: str = WRITE_TODOS_TOOL_DESCRIPTION,
    ) -> None:
        """Initialize the `TodoListMiddleware` with optional custom prompts.

        Args:
            system_prompt: Custom system prompt to guide the agent on using the todo
                tool.
            tool_description: Custom description for the `write_todos` tool.
        

        中文翻译:
        使用可选的自定义提示初始化“TodoListMiddleware”。
        参数：
            system_prompt：自定义系统提示，指导代理使用待办事项
                工具。
            tool_description：“write_todos”工具的自定义描述。"""
        super().__init__()
        self.system_prompt = system_prompt
        self.tool_description = tool_description

        # Dynamically create the write_todos tool with the custom description
        # 中文: 使用自定义描述动态创建 write_todos 工具
        @tool(description=self.tool_description)
        def write_todos(
            todos: list[Todo], tool_call_id: Annotated[str, InjectedToolCallId]
        ) -> Command:
            """Create and manage a structured task list for your current work session.

            中文翻译:
            为当前工作会话创建和管理结构化任务列表。"""
            return Command(
                update={
                    "todos": todos,
                    "messages": [
                        ToolMessage(f"Updated todo list to {todos}", tool_call_id=tool_call_id)
                    ],
                }
            )

        self.tools = [write_todos]

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Update the system message to include the todo system prompt.

        中文翻译:
        更新系统消息以包含待办事项系统提示。"""
        if request.system_message is not None:
            new_system_content = [
                *request.system_message.content_blocks,
                {"type": "text", "text": f"\n\n{self.system_prompt}"},
            ]
        else:
            new_system_content = [{"type": "text", "text": self.system_prompt}]
        new_system_message = SystemMessage(
            content=cast("list[str | dict[str, str]]", new_system_content)
        )
        return handler(request.override(system_message=new_system_message))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Update the system message to include the todo system prompt (async version).

        中文翻译:
        更新系统消息以包含待办事项系统提示（异步版本）。"""
        if request.system_message is not None:
            new_system_content = [
                *request.system_message.content_blocks,
                {"type": "text", "text": f"\n\n{self.system_prompt}"},
            ]
        else:
            new_system_content = [{"type": "text", "text": self.system_prompt}]
        new_system_message = SystemMessage(
            content=cast("list[str | dict[str, str]]", new_system_content)
        )
        return await handler(request.override(system_message=new_system_message))
