# vLLM Agent 最小示例：保存模型回答到文件

这个示例展示如何使用 LangChain 的最新 agent 架构（`langchain.agents.create_agent`）连接到 vLLM 部署的 Qwen2-VL-7B-Instruct 模型，实现一个"单次工具调用"场景：模型生成回答后自动保存到服务器本地文件。

## 前置要求

### 服务器端必须满足

1. **vLLM 服务运行中**：
   - 地址：`http://192.168.1.253:8000`
   - 模型：`Qwen2-VL-7B-Instruct`
   - 必须支持 `/v1/chat/completions` 工具调用（已通过您的测试验证）

2. **Python 环境**：Python 3.10+

3. **文件写入权限**：确保运行脚本的用户对 `/home` 目录有写入权限

## 安装步骤（在 Linux 服务器上执行）

**重要**：以下所有命令都必须在您的 **Linux 服务器**上执行，不要在 Windows 本机执行。

### 方式 A：使用 venv（Python 标准虚拟环境）

```bash
# 1. 创建隔离的虚拟环境（不污染系统 Python）
python3 -m venv /home/.venvs/lc_agent

# 2. 激活虚拟环境
source /home/.venvs/lc_agent/bin/activate

# 3. 安装依赖
pip install langchain langchain-openai langgraph

# 4. 验证安装
python -c "from langchain.agents import create_agent; print('OK')"
```

### 方式 B：使用 uv（更快的包管理器）

```bash
# 1. 安装 uv（如果尚未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 创建虚拟环境
uv venv /home/.venvs/lc_agent

# 3. 激活虚拟环境
source /home/.venvs/lc_agent/bin/activate

# 4. 安装依赖
uv pip install langchain langchain-openai langgraph

# 5. 验证安装
python -c "from langchain.agents import create_agent; print('OK')"
```

## 使用方法

### 1. 拷贝脚本到服务器

将 `run_agent.py` 拷贝到服务器的任意目录，例如：

```bash
# 在服务器上
mkdir -p /home/langchain_examples
cd /home/langchain_examples

# 然后通过 scp/sftp/rsync 等方式从您的 Windows 机器传输文件
# 例如在 Windows PowerShell 中：
# scp D:\work\langchain-master\examples\vllm_agent_save_answer\run_agent.py user@192.168.1.253:/home/langchain_examples/
```

### 2. 运行脚本（使用默认配置）

```bash
# 激活虚拟环境
source /home/.venvs/lc_agent/bin/activate

# 运行脚本
python run_agent.py
```

### 3. 使用自定义配置（可选）

通过环境变量覆盖默认值：

```bash
# 自定义 vLLM 地址
export VLLM_BASE_URL="http://192.168.1.253:8000/v1"

# 自定义模型名
export VLLM_MODEL="Qwen2-VL-7B-Instruct"

# 自定义保存目录
export SAVE_DIR="/home/my_custom_logs"

# 运行
python run_agent.py
```

### 4. 启用调试模式

如果需要查看详细的执行日志（包括每一步的 LLM 调用、工具调用等），修改 `run_agent.py` 中的：

```python
agent = create_agent(
    model=model,
    tools=[message_save],
    system_prompt=system_prompt,
    debug=True,  # 改为 True
)
```

## 预期输出

脚本成功运行后，您应该看到类似以下输出：

```
============================================================
用户问题: 请用中文简单介绍一下人工智能的历史，控制在100字以内。
============================================================

执行流程:
------------------------------------------------------------

[1] HumanMessage:
  Content: 请用中文简单介绍一下人工智能的历史，控制在100字以内。...

[2] AIMessage:
  Content: ...
  Tool Calls: 1
    - message_save(prompt=人工智能起源于1956年达特茅斯会议，经历了早期...)

[3] ToolMessage:
  Tool Result: 已成功保存到文件: /home/langchain_agent_logs/answer_20260113_175530.txt

[4] AIMessage:
  Content: 人工智能起源于1956年达特茅斯会议，经历了早期...

============================================================
最终回答:
============================================================
人工智能起源于1956年达特茅斯会议，经历了早期符号主义、知识工程、
机器学习等阶段。21世纪以来，深度学习推动AI飞速发展，在图像识别、
自然语言处理等领域取得突破性进展。
============================================================
```

## 验证清单

运行脚本后，请确认以下几点：

- [ ] **工具调用被触发**：在"执行流程"中看到 `AIMessage` 包含 `Tool Calls: 1`
- [ ] **文件成功创建**：检查 `/home/langchain_agent_logs/` 目录下是否生成了 `answer_<timestamp>.txt`
- [ ] **文件内容正确**：打开生成的文件，确认内容与"最终回答"一致
- [ ] **最终回答输出**：脚本最后打印出完整的中文回答

验证文件内容示例：

```bash
# 查看最新生成的文件
ls -lt /home/langchain_agent_logs/ | head -5

# 查看文件内容
cat /home/langchain_agent_logs/answer_20260113_175530.txt
```

## 故障排查

### 1. vLLM 返回 400 错误：`"auto" tool choice requires --enable-auto-tool-choice`

**原因**：您的 vLLM 启动时未开启 `--enable-auto-tool-choice` 和 `--tool-call-parser`，因此不接受 `tool_choice="auto"`。

**解决方案**：脚本已经通过 `force_tool_choice_middleware` 避免了这个问题，它会：
- 第一次调用时强制 `tool_choice` 指定为 `message_save`
- 工具执行后强制 `tool_choice="none"`

如果仍然遇到此错误，请确认：
- 脚本中的 middleware 已正确传入 `create_agent`
- vLLM 服务正常运行且支持 `/v1/chat/completions` 的工具调用

### 2. 工具没有被调用（看不到 Tool Calls）

**原因**：模型可能没有正确生成工具调用，或工具调用被过滤。

**解决方案**：
- 启用调试模式查看详细日志（`debug=True`）
- 检查模型返回的第一个 `AIMessage` 是否包含 `tool_calls`
- 尝试修改 system_prompt 使其更明确

### 3. 连接 vLLM 失败

**错误信息**：`Connection refused` 或 `timeout`

**解决方案**：
- 确认 vLLM 服务正在运行：`curl http://192.168.1.253:8000/v1/models`
- 检查防火墙/网络配置
- 确认 `VLLM_BASE_URL` 环境变量正确

### 4. 文件写入失败

**错误信息**：`Permission denied` 或 `保存失败`

**解决方案**：
- 检查目录权限：`ls -ld /home`
- 手动创建目录并设置权限：
  ```bash
  sudo mkdir -p /home/langchain_agent_logs
  sudo chown $(whoami):$(whoami) /home/langchain_agent_logs
  ```
- 或修改 `SAVE_DIR` 为您有权限的目录

### 5. 依赖包导入错误

**错误信息**：`ModuleNotFoundError: No module named 'langchain'`

**解决方案**：
- 确认虚拟环境已激活：`which python` 应指向 `/home/.venvs/lc_agent/bin/python`
- 重新安装依赖：`pip install --upgrade langchain langchain-openai langgraph`

## 工作原理

这个示例采用 **LangChain v1 的 agent 架构**（基于 LangGraph），执行流程如下：

1. **用户输入** → Agent
2. **Agent 调用模型（第一次）**：
   - Middleware 检测到还没调用 `message_save`，强制 `tool_choice` 指定为该工具
   - 模型根据 system_prompt 生成答案并调用工具
   - 返回 `AIMessage` 包含 `tool_calls`（调用 `message_save`，参数为生成的答案）
3. **Agent 执行工具**：
   - 调用 `message_save(prompt="...")`
   - 写入文件到 `/home/langchain_agent_logs/answer_<timestamp>.txt`
   - 返回 `ToolMessage`（保存成功信息）
4. **Agent 调用模型（第二次）**：
   - Middleware 检测到已有 `message_save` 的 ToolMessage，强制 `tool_choice="none"`
   - 模型看到工具返回成功，输出最终的用户可见回答（与工具参数中的答案内容一致）
5. **返回结果** → 用户

这是标准的 **ReAct 模式**（Reasoning + Acting）agent loop。

### 关于 tool_choice 的特殊处理

由于您的 vLLM 未开启 `--enable-auto-tool-choice`，脚本使用了 **middleware** 来避免 `tool_choice="auto"`：

- `force_tool_choice_middleware` 会在每次模型调用前动态设置 `tool_choice`
- 第一次强制调用 `message_save`，第二次强制 `"none"` 禁止工具调用
- 这样既满足了 vLLM 的限制，又保证了工具只被调用一次

## 扩展方向

完成这个最小示例后，您可以尝试：

1. **多轮对话**：使用 checkpointer 保存会话历史
2. **添加更多工具**：例如查询数据库、调用 API 等
3. **多模态输入**：利用 Qwen2-VL 的视觉能力（需要修改输入为图文混合）
4. **结构化输出**：使用 `response_format` 返回 JSON schema
5. **流式输出**：使用 `agent.stream()` 实现逐 token 输出

## 参考资料

- [LangChain Agents 文档](https://docs.langchain.com/oss/python/langchain/agents)
- [ChatOpenAI 集成文档](https://docs.langchain.com/oss/python/integrations/chat/openai)
- [vLLM OpenAI 兼容服务器](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)

