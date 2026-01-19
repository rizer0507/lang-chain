#!/usr/bin/env python3
"""Minimal LangChain agent that saves model's final answer to a timestamped file.

This example demonstrates using create_agent with vLLM's OpenAI-compatible
/v1/chat/completions endpoint to:
1. Generate a final answer to the user's question
2. Call the message_save tool with that answer
3. Return the answer to the user

The tool writes to /home/langchain_agent_logs/ with timestamped filenames.
"""

import os
from datetime import datetime
from pathlib import Path

from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


@tool
def message_save(prompt: str) -> str:
    """将模型的文本输出保存到本地文件。

    Args:
        prompt: 要保存的文本内容

    Returns:
        保存成功的文件路径或错误信息
    """
    try:
        # Get save directory from env or use default
        save_dir = os.environ.get("SAVE_DIR", "/home/langchain_agent_logs")
        save_path = Path(save_dir)

        # Create directory if it doesn't exist
        save_path.mkdir(parents=True, exist_ok=True)

        # Generate timestamp-based filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = save_path / f"answer_{timestamp}.txt"

        # Append to file
        with open(filename, "a", encoding="utf-8") as f:
            f.write(prompt + "\n")

        return f"已成功保存到文件: {filename}"

    except Exception as e:
        return f"保存失败: {type(e).__name__}: {e}"


@wrap_model_call
def force_tool_choice_middleware(request, handler):
    """强制设置 tool_choice 以避免 vLLM 的 'auto' 限制。

    vLLM 在未开启 --enable-auto-tool-choice 时不接受 tool_choice="auto"。
    这个 middleware 会根据当前对话状态动态设置 tool_choice：

    - 如果还没调用过 message_save：强制调用 message_save
    - 如果已经调用过 message_save：禁止工具调用（tool_choice="none"）

    这样保证整个 agent loop 只调用一次工具，然后返回最终回答。
    """
    # 检查是否已经执行过 message_save 工具
    has_saved = any(
        isinstance(msg, ToolMessage) and msg.name == "message_save"
        for msg in request.messages
    )

    if has_saved:
        # 已经保存过，不再调用工具
        new_request = request.override(tool_choice="none")
    else:
        # 还没保存，强制调用 message_save
        new_request = request.override(
            tool_choice={
                "type": "function",
                "function": {"name": "message_save"}
            }
        )

    return handler(new_request)


def main():
    """Run the agent with a sample input."""
    # Read configuration from environment variables
    vllm_base_url = os.environ.get("VLLM_BASE_URL", "http://192.168.1.253:8000/v1")
    vllm_model = os.environ.get("VLLM_MODEL", "Qwen2-VL-7B-Instruct")

    # Initialize ChatOpenAI with vLLM endpoint
    # Force chat completions API (not responses API)
    model = ChatOpenAI(
        base_url=vllm_base_url,
        api_key="EMPTY",  # Placeholder for vLLM without auth
        model=vllm_model,
        temperature=0,
        max_tokens=512,
        use_responses_api=False,  # Force /v1/chat/completions
        output_version="v0",  # Avoid responses/v1 default
    )

    # System prompt that enforces the behavior:
    # 1. Generate final answer and put it in message_save tool call
    # 2. After tool returns, output the same answer to user
    system_prompt = """你是一个有用的助手。你会通过 message_save 工具保存答案到文件。

工作流程：
1. 当你第一次回答用户问题时，生成完整的中文答案，并将其作为 prompt 参数传递给 message_save 工具
2. message_save 工具会将答案保存到文件并返回确认信息
3. 看到工具返回后，将同样的答案直接输出给用户

注意：
- 第一次输出时将完整答案放在 message_save 的 prompt 参数中
- 工具返回后，直接输出答案内容给用户（不要再次调用工具）"""

    # Create agent with the tool and middleware
    agent = create_agent(
        model=model,
        tools=[message_save],
        system_prompt=system_prompt,
        middleware=[force_tool_choice_middleware],
        debug=False,  # Set to True to see detailed execution logs
    )

    # Sample input: Ask a simple question
    user_question = "请用中文简单介绍一下人工智能的历史，控制在100字以内。"

    print(f"\n{'='*60}")
    print(f"用户问题: {user_question}")
    print(f"{'='*60}\n")

    # Invoke the agent
    result = agent.invoke({"messages": [HumanMessage(content=user_question)]})

    # Extract results
    messages = result["messages"]

    print("\n执行流程:")
    print("-" * 60)

    # Display the message sequence
    for i, msg in enumerate(messages, 1):
        msg_type = msg.__class__.__name__
        print(f"\n[{i}] {msg_type}:")

        if hasattr(msg, "content") and msg.content:
            print(f"  Content: {msg.content[:200]}...")

        if hasattr(msg, "tool_calls") and msg.tool_calls:
            print(f"  Tool Calls: {len(msg.tool_calls)}")
            for tc in msg.tool_calls:
                print(f"    - {tc['name']}(prompt={tc['args'].get('prompt', '')[:50]}...)")

        if msg_type == "ToolMessage":
            print(f"  Tool Result: {msg.content}")

    # Get final answer (last AIMessage)
    final_answer = None
    for msg in reversed(messages):
        if msg.__class__.__name__ == "AIMessage" and msg.content:
            final_answer = msg.content
            break

    print(f"\n{'='*60}")
    print("最终回答:")
    print(f"{'='*60}")
    print(final_answer)
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

