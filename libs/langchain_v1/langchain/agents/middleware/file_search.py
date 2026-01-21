"""文件搜索中间件模块。

本模块提供在文件系统中进行 Glob 和 Grep 搜索的能力。

核心类:
--------
**FilesystemFileSearchMiddleware**: 文件搜索中间件

提供的工具:
-----------
- `glob_search`: 快速文件模式匹配（如 `**/*.py`）
- `grep_search`: 内容搜索（使用 ripgrep 或 Python 回退）

使用示例:
---------
>>> from langchain.agents import create_agent
>>> from langchain.agents.middleware import FilesystemFileSearchMiddleware
>>>
>>> search = FilesystemFileSearchMiddleware(
...     root_path="/workspace",
...     use_ripgrep=True,
... )
>>>
>>> agent = create_agent(
...     model="openai:gpt-4o",
...     middleware=[search],
... )
"""

from __future__ import annotations

import fnmatch
import json
import re
import subprocess
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from langchain_core.tools import tool

from langchain.agents.middleware.types import AgentMiddleware


def _expand_include_patterns(pattern: str) -> list[str] | None:
    """Expand brace patterns like `*.{py,pyi}` into a list of globs.

    中文翻译:
    将像“*.{py,pyi}”这样的大括号模式展开为一个通配符列表。"""
    if "}" in pattern and "{" not in pattern:
        return None

    expanded: list[str] = []

    def _expand(current: str) -> None:
        start = current.find("{")
        if start == -1:
            expanded.append(current)
            return

        end = current.find("}", start)
        if end == -1:
            raise ValueError

        prefix = current[:start]
        suffix = current[end + 1 :]
        inner = current[start + 1 : end]
        if not inner:
            raise ValueError

        for option in inner.split(","):
            _expand(prefix + option + suffix)

    try:
        _expand(pattern)
    except ValueError:
        return None

    return expanded


def _is_valid_include_pattern(pattern: str) -> bool:
    """Validate glob pattern used for include filters.

    中文翻译:
    验证用于包含过滤器的 glob 模式。"""
    if not pattern:
        return False

    if any(char in pattern for char in ("\x00", "\n", "\r")):
        return False

    expanded = _expand_include_patterns(pattern)
    if expanded is None:
        return False

    try:
        for candidate in expanded:
            re.compile(fnmatch.translate(candidate))
    except re.error:
        return False

    return True


def _match_include_pattern(basename: str, pattern: str) -> bool:
    """Return True if the basename matches the include pattern.

    中文翻译:
    如果基本名称与包含模式匹配，则返回 True。"""
    expanded = _expand_include_patterns(pattern)
    if not expanded:
        return False

    return any(fnmatch.fnmatch(basename, candidate) for candidate in expanded)


class FilesystemFileSearchMiddleware(AgentMiddleware):
    """Provides Glob and Grep search over filesystem files.

    This middleware adds two tools that search through local filesystem:

    - Glob: Fast file pattern matching by file path
    - Grep: Fast content search using ripgrep or Python fallback

    Example:
        ```python
        from langchain.agents import create_agent
        from langchain.agents.middleware import (
            FilesystemFileSearchMiddleware,
        )

        agent = create_agent(
            model=model,
            tools=[],  # Add tools as needed
            middleware=[
                FilesystemFileSearchMiddleware(root_path="/workspace"),
            ],
        )
        ```
    

    中文翻译:
    提供对文件系统文件的 Glob 和 Grep 搜索。
    该中间件添加了两个搜索本地文件系统的工具：
    - Glob：通过文件路径快速匹配文件模式
    - Grep：使用 ripgrep 或 Python 后备进行快速内容搜索
    示例：
        ````蟒蛇
        从 langchain.agents 导入 create_agent
        从 langchain.agents.middleware 导入（
            文件系统文件搜索中间件，
        ）
        代理=创建_代理（
            型号=型号，
            tools=[], # 根据需要添加工具
            中间件=[
                文件系统文件搜索中间件（root_path =“/工作空间”），
            ],
        ）
        ````"""

    def __init__(
        self,
        *,
        root_path: str,
        use_ripgrep: bool = True,
        max_file_size_mb: int = 10,
    ) -> None:
        """Initialize the search middleware.

        Args:
            root_path: Root directory to search.
            use_ripgrep: Whether to use `ripgrep` for search.

                Falls back to Python if `ripgrep` unavailable.
            max_file_size_mb: Maximum file size to search in MB.
        

        中文翻译:
        初始化搜索中间件。
        参数：
            root_path：要搜索的根目录。
            use_ripgrep：是否使用“ripgrep”进行搜索。
                如果 `ripgrep` 不可用，则回退到 Python。
            max_file_size_mb：要搜索的最大文件大小（以 MB 为单位）。"""
        self.root_path = Path(root_path).resolve()
        self.use_ripgrep = use_ripgrep
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024

        # Create tool instances as closures that capture self
        # 中文: 创建工具实例作为捕获 self 的闭包
        @tool
        def glob_search(pattern: str, path: str = "/") -> str:
            """Fast file pattern matching tool that works with any codebase size.

            Supports glob patterns like `**/*.js` or `src/**/*.ts`.

            Returns matching file paths sorted by modification time.

            Use this tool when you need to find files by name patterns.

            Args:
                pattern: The glob pattern to match files against.
                path: The directory to search in. If not specified, searches from root.

            Returns:
                Newline-separated list of matching file paths, sorted by modification
                time (most recently modified first). Returns `'No files found'` if no
                matches.
            

            中文翻译:
            适用于任何代码库大小的快速文件模式匹配工具。
            支持 `**/*.js` 或 `src/**/*.ts` 等 glob 模式。
            返回按修改时间排序的匹配文件路径。
            当您需要按名称模式查找文件时，请使用此工具。
            参数：
                模式：匹配文件的全局模式。
                path：要搜索的目录。如果未指定，则从根目录开始搜索。
            返回：
                以换行符分隔的匹配文件路径列表，按修改排序
                时间（最近修改的在前）。如果没有则返回“未找到文件”
                匹配。"""
            try:
                base_full = self._validate_and_resolve_path(path)
            except ValueError:
                return "No files found"

            if not base_full.exists() or not base_full.is_dir():
                return "No files found"

            # Use pathlib glob
            # 中文: 使用路径库 glob
            matching: list[tuple[str, str]] = []
            for match in base_full.glob(pattern):
                if match.is_file():
                    # Convert to virtual path
                    # 中文: 转换为虚拟路径
                    virtual_path = "/" + str(match.relative_to(self.root_path))
                    stat = match.stat()
                    modified_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
                    matching.append((virtual_path, modified_at))

            if not matching:
                return "No files found"

            file_paths = [p for p, _ in matching]
            return "\n".join(file_paths)

        @tool
        def grep_search(
            pattern: str,
            path: str = "/",
            include: str | None = None,
            output_mode: Literal["files_with_matches", "content", "count"] = "files_with_matches",
        ) -> str:
            """Fast content search tool that works with any codebase size.

            Searches file contents using regular expressions. Supports full regex
            syntax and filters files by pattern with the include parameter.

            Args:
                pattern: The regular expression pattern to search for in file contents.
                path: The directory to search in. If not specified, searches from root.
                include: File pattern to filter (e.g., `'*.js'`, `'*.{ts,tsx}'`).
                output_mode: Output format:

                    - `'files_with_matches'`: Only file paths containing matches
                    - `'content'`: Matching lines with `file:line:content` format
                    - `'count'`: Count of matches per file

            Returns:
                Search results formatted according to `output_mode`.
                    Returns `'No matches found'` if no results.
            

            中文翻译:
            适用于任何代码库大小的快速内容搜索工具。
            使用正则表达式搜索文件内容。支持完整的正则表达式
            语法并使用 include 参数按模式过滤文件。
            参数：
                模式：要在文件内容中搜索的正则表达式模式。
                path：要搜索的目录。如果未指定，则从根目录开始搜索。
                include：要过滤的文件模式（例如“*.js”、“*.{ts,tsx}”）。
                输出模式：输出格式：
                    - `'files_with_matches'`：仅包含匹配的文件路径
                    - `'content'`：匹配具有 `file:line:content` 格式的行
                    - `'count'`：每个文件的匹配数
            返回：
                根据“output_mode”格式化的搜索结果。
                    如果没有结果，则返回“未找到匹配项”。"""
            # Compile regex pattern (for validation)
            # 中文: 编译正则表达式模式（用于验证）
            try:
                re.compile(pattern)
            except re.error as e:
                return f"Invalid regex pattern: {e}"

            if include and not _is_valid_include_pattern(include):
                return "Invalid include pattern"

            # Try ripgrep first if enabled
            # 中文: 如果启用，请先尝试 ripgrep
            results = None
            if self.use_ripgrep:
                with suppress(
                    FileNotFoundError,
                    subprocess.CalledProcessError,
                    subprocess.TimeoutExpired,
                ):
                    results = self._ripgrep_search(pattern, path, include)

            # Python fallback if ripgrep failed or is disabled
            # 中文: 如果 ripgrep 失败或被禁用，Python 会回退
            if results is None:
                results = self._python_search(pattern, path, include)

            if not results:
                return "No matches found"

            # Format output based on mode
            # 中文: 根据模式格式化输出
            return self._format_grep_results(results, output_mode)

        self.glob_search = glob_search
        self.grep_search = grep_search
        self.tools = [glob_search, grep_search]

    def _validate_and_resolve_path(self, path: str) -> Path:
        """Validate and resolve a virtual path to filesystem path.

        中文翻译:
        验证并解析文件系统路径的虚拟路径。"""
        # Normalize path
        # 中文: 标准化路径
        if not path.startswith("/"):
            path = "/" + path

        # Check for path traversal
        # 中文: 检查路径遍历
        if ".." in path or "~" in path:
            msg = "Path traversal not allowed"
            raise ValueError(msg)

        # Convert virtual path to filesystem path
        # 中文: 将虚拟路径转换为文件系统路径
        relative = path.lstrip("/")
        full_path = (self.root_path / relative).resolve()

        # Ensure path is within root
        # 中文: 确保路径位于根目录内
        try:
            full_path.relative_to(self.root_path)
        except ValueError:
            msg = f"Path outside root directory: {path}"
            raise ValueError(msg) from None

        return full_path

    def _ripgrep_search(
        self, pattern: str, base_path: str, include: str | None
    ) -> dict[str, list[tuple[int, str]]]:
        """Search using ripgrep subprocess.

        中文翻译:
        使用 ripgrep 子进程进行搜索。"""
        try:
            base_full = self._validate_and_resolve_path(base_path)
        except ValueError:
            return {}

        if not base_full.exists():
            return {}

        # Build ripgrep command
        # 中文: 构建 ripgrep 命令
        cmd = ["rg", "--json"]

        if include:
            # Convert glob pattern to ripgrep glob
            # 中文: 将 glob 模式转换为 ripgrep glob
            cmd.extend(["--glob", include])

        cmd.extend(["--", pattern, str(base_full)])

        try:
            result = subprocess.run(  # noqa: S603
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # Fallback to Python search if ripgrep unavailable or times out
            # 中文: 如果 ripgrep 不可用或超时，则回退到 Python 搜索
            return self._python_search(pattern, base_path, include)

        # Parse ripgrep JSON output
        # 中文: 解析 ripgrep JSON 输出
        results: dict[str, list[tuple[int, str]]] = {}
        for line in result.stdout.splitlines():
            try:
                data = json.loads(line)
                if data["type"] == "match":
                    path = data["data"]["path"]["text"]
                    # Convert to virtual path
                    # 中文: 转换为虚拟路径
                    virtual_path = "/" + str(Path(path).relative_to(self.root_path))
                    line_num = data["data"]["line_number"]
                    line_text = data["data"]["lines"]["text"].rstrip("\n")

                    if virtual_path not in results:
                        results[virtual_path] = []
                    results[virtual_path].append((line_num, line_text))
            except (json.JSONDecodeError, KeyError):
                continue

        return results

    def _python_search(
        self, pattern: str, base_path: str, include: str | None
    ) -> dict[str, list[tuple[int, str]]]:
        """Search using Python regex (fallback).

        中文翻译:
        使用 Python 正则表达式进行搜索（后备）。"""
        try:
            base_full = self._validate_and_resolve_path(base_path)
        except ValueError:
            return {}

        if not base_full.exists():
            return {}

        regex = re.compile(pattern)
        results: dict[str, list[tuple[int, str]]] = {}

        # Walk directory tree
        # 中文: 遍历目录树
        for file_path in base_full.rglob("*"):
            if not file_path.is_file():
                continue

            # Check include filter
            # 中文: 检查包含过滤器
            if include and not _match_include_pattern(file_path.name, include):
                continue

            # Skip files that are too large
            # 中文: 跳过太大的文件
            if file_path.stat().st_size > self.max_file_size_bytes:
                continue

            try:
                content = file_path.read_text()
            except (UnicodeDecodeError, PermissionError):
                continue

            # Search content
            # 中文: 搜索内容
            for line_num, line in enumerate(content.splitlines(), 1):
                if regex.search(line):
                    virtual_path = "/" + str(file_path.relative_to(self.root_path))
                    if virtual_path not in results:
                        results[virtual_path] = []
                    results[virtual_path].append((line_num, line))

        return results

    def _format_grep_results(
        self,
        results: dict[str, list[tuple[int, str]]],
        output_mode: str,
    ) -> str:
        """Format grep results based on output mode.

        中文翻译:
        根据输出模式格式化 grep 结果。"""
        if output_mode == "files_with_matches":
            # Just return file paths
            # 中文: 只返回文件路径
            return "\n".join(sorted(results.keys()))

        if output_mode == "content":
            # Return file:line:content format
            # 中文: 返回文件:行:内容格式
            lines = []
            for file_path in sorted(results.keys()):
                for line_num, line in results[file_path]:
                    lines.append(f"{file_path}:{line_num}:{line}")
            return "\n".join(lines)

        if output_mode == "count":
            # Return file:count format
            # 中文: 返回文件：计数格式
            lines = []
            for file_path in sorted(results.keys()):
                count = len(results[file_path])
                lines.append(f"{file_path}:{count}")
            return "\n".join(lines)

        # Default to files_with_matches
        # 中文: 默认为 files_with_matches
        return "\n".join(sorted(results.keys()))


__all__ = [
    "FilesystemFileSearchMiddleware",
]
