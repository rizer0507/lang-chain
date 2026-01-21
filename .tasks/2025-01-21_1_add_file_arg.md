# 背景
文件名：2025-01-21_1_add_file_arg
创建于：2025-01-21_11:55:00
创建者：Antigravity
主分支：main
任务分支：task/add_file_arg_2025-01-21_1
Yolo模式：Off

# 任务描述
给 `scripts/translate_comments.py` 脚本增加一个参数，以便能够指定翻译单个文件，而不是扫描整个目录。

# 项目概览
当前脚本只能通过 `--dir` 参数指定目录进行批量处理。用户希望增加对单个文件的支持。

⚠️ 警告：永远不要修改此部分 ⚠️
1.你是Claude 4.0，集成在Antigravity IDE中... (RIPER-5 协议摘要)
⚠️ 警告：永远不要修改此部分 ⚠️

# 分析
- `scripts/translate_comments.py` 使用 `argparse` 解析参数。
- 目前 `main` 函数强制要求 `--dir` 参数。
- `FileProcessor` 类有 `process_file` 方法，支持单个文件处理。

# 提议的解决方案
1. Modify `argparse` to include a mutually exclusive group for `--dir` and `--file`.
2. Implement logic to handle the `--file` argument by calling `process_file` directly.
3. Validate that the file exists before processing.

# 当前执行步骤： "3. 实施更改"

# 任务进度
2025-01-21 11:58
- 已修改：scripts/translate_comments.py
- 更改：实现了 --file 参数，支持单个文件翻译。
- 状态：成功

2025-01-21 12:05
- 已修改：scripts/translate_comments.py
- 更改：实现了长文本分段翻译逻辑，修复了 5000 字符限制导致的 API 错误。
- 状态：待验证

2025-01-21 12:07
- 验证：成功 (在 factory.py 上触发了分段翻译逻辑)




# 最终审查
2025-01-21 12:02
- 实施与计划完全匹配: 是
- 功能验证: 成功 (单文件参数正常工作)
- 代码提交: 已完成

