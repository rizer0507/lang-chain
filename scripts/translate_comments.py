#!/usr/bin/env python3
"""
Python代码注释翻译脚本

使用Google Translate (通过deep-translator库) 将Python代码中的英文注释翻译为中文。

功能特性:
- 处理 # 单行注释
- 处理 ''' ''' 和 \"\"\" \"\"\" 多行文档字符串
- 保留原英文注释，在下方添加中文翻译
- 跳过 >>> 代码示例
- 批量处理指定目录下的所有 .py 文件
- 免费使用，无需API Key

使用方法:
    python translate_comments.py --dir <目标目录>

示例:
    python translate_comments.py --dir ./libs/langchain_v1
    python translate_comments.py --dir ./libs --dry-run
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

try:
    from deep_translator import GoogleTranslator
except ImportError:
    print("错误: 请先安装 deep-translator 库")
    print("运行: pip install deep-translator")
    sys.exit(1)


class GoogleTransTranslator:
    """Google翻译器封装类"""

    def __init__(self, target_lang: str = "zh-CN"):
        """
        初始化翻译器

        Args:
            target_lang: 目标语言代码，默认为简体中文 (zh-CN)
        """
        self.translator = GoogleTranslator(source='auto', target=target_lang)
        self._request_count = 0
        self._last_request_time = 0
        self._cache = {} # 简单内存缓存

    def translate(self, text: str) -> str:
        """
        翻译文本

        Args:
            text: 要翻译的英文文本

        Returns:
            翻译后的中文文本
        """
        if not text or not text.strip():
            return text

        # 检查缓存
        if text in self._cache:
            return self._cache[text]

        # 限流
        self._rate_limit()

        try:
            # Google Translate API有长度限制（约5000字符）
            # 我们设定一个较安全的限制，例如4500字符
            MAX_LENGTH = 4500

            if len(text) <= MAX_LENGTH:
                result = self.translator.translate(text)
                self._cache[text] = result
                return result

            # 长文本处理：按行切分
            print(f"    [提示] 文本过长 ({len(text)} 字符)，正在分段翻译...")
            lines = text.splitlines(keepends=True)
            parts = []
            current_chunk = []
            current_length = 0

            for line in lines:
                # 如果当前块加上新行超过限制，先翻译当前块
                if current_length + len(line) > MAX_LENGTH:
                    if current_chunk:
                        chunk_text = "".join(current_chunk)
                        part_result = self.translator.translate(chunk_text)
                        parts.append(part_result)
                        # 每次翻译后也稍微限流一下
                        self._rate_limit()

                    current_chunk = [line]
                    current_length = len(line)
                else:
                    current_chunk.append(line)
                    current_length += len(line)

            # 处理剩余块
            if current_chunk:
                chunk_text = "".join(current_chunk)
                parts.append(self.translator.translate(chunk_text))

            # 合并结果
            result = "".join(parts)
            self._cache[text] = result
            return result

        except Exception as e:
            print(f"翻译错误: {e}")
            return text

    def _rate_limit(self):
        """速率限制，避免API调用过快"""
        self._request_count += 1
        current_time = time.time()

        # 每1个请求等待0.2秒，避免太快
        elapsed = current_time - self._last_request_time
        if elapsed < 0.2:
            time.sleep(0.2 - elapsed)

        self._last_request_time = time.time()


class CommentExtractor:
    """注释提取与处理类"""

    # 匹配多行文档字符串
    DOCSTRING_PATTERN = re.compile(
        r'([ \t]*)(\"\"\"|\'\'\')(.*?)\2',
        re.DOTALL
    )

    # 匹配单行注释
    SINGLE_COMMENT_PATTERN = re.compile(
        r'^([ \t]*)(#\s*)(.+)$',
        re.MULTILINE
    )

    # 中文字符检测
    CHINESE_PATTERN = re.compile(r'[\u4e00-\u9fff]')

    # 代码示例模式
    CODE_EXAMPLE_PATTERN = re.compile(r'^\s*>>>')

    def __init__(self, translator: GoogleTransTranslator):
        """
        初始化提取器

        Args:
            translator: 翻译器实例
        """
        self.translator = translator

    def has_chinese(self, text: str) -> bool:
        """检查文本是否包含中文"""
        return bool(self.CHINESE_PATTERN.search(text))

    def is_code_example(self, text: str) -> bool:
        """检查是否为代码示例行"""
        return bool(self.CODE_EXAMPLE_PATTERN.match(text))

    def process_docstring(self, match: re.Match) -> str:
        """
        处理多行文档字符串

        Args:
            match: 正则匹配对象

        Returns:
            处理后的文档字符串
        """
        indent = match.group(1)
        quote = match.group(2)
        content = match.group(3)

        # 如果已包含中文，跳过
        if self.has_chinese(content):
            return match.group(0)

        # 分离代码示例和普通文本
        lines = content.split('\n')
        text_parts = []
        example_ranges = []

        i = 0
        while i < len(lines):
            line = lines[i]
            if self.is_code_example(line):
                # 标记代码示例范围
                start = i
                while i < len(lines) and (
                    self.is_code_example(lines[i]) or
                    (lines[i].strip() and not lines[i].strip().startswith(('Args:', 'Returns:', 'Raises:', 'Example:', 'Note:')))
                ):
                    # 检查是否是代码示例的延续行
                    if not self.is_code_example(lines[i]) and lines[i].strip():
                        if not lines[i].startswith('    ...') and not lines[i].strip().startswith('...'):
                            break
                    i += 1
                example_ranges.append((start, i))
            else:
                if line.strip():
                    text_parts.append((i, line))
                i += 1

        # 收集需要翻译的文本（排除代码示例）
        translatable_lines = []
        for line_idx, line in text_parts:
            in_example = any(start <= line_idx < end for start, end in example_ranges)
            if not in_example:
                translatable_lines.append(line)

        if not translatable_lines:
            return match.group(0)

        # 翻译收集的文本
        text_to_translate = '\n'.join(translatable_lines)
        if not text_to_translate.strip():
            return match.group(0)

        print(f"  正在翻译多行注释 ({len(text_to_translate)} 字符)...")
        translated = self.translator.translate(text_to_translate)

        # 构建新的docstring
        if content.endswith('\n'):
            new_content = f"{content}\n{indent}中文翻译:\n{indent}{translated}\n{indent}"
        else:
            new_content = f"{content}\n\n{indent}中文翻译:\n{indent}{translated}"

        return f"{indent}{quote}{new_content}{quote}"

    def process_single_comment(self, match: re.Match) -> str:
        """
        处理单行注释

        Args:
            match: 正则匹配对象

        Returns:
            处理后的注释
        """
        indent = match.group(1)
        prefix = match.group(2)
        content = match.group(3)
        stripped_content = content.strip()

        # 如果已包含中文，跳过
        if self.has_chinese(content):
            return match.group(0)

        # 跳过特殊注释 (type hints, noqa, etc.)
        skip_patterns = ['type:', 'noqa', 'pylint:', 'pragma:', '...', 'fmt:', 'TODO', 'FIXME', 'XXX', 'NOTE']
        if any(pattern in content for pattern in skip_patterns):
            return match.group(0)

        # 跳过过短的注释（通常无需翻译）
        if len(stripped_content) < 2:
            return match.group(0)

        # 翻译注释
        # print(f"  翻译单行: {stripped_content[:30]}...")
        translated = self.translator.translate(stripped_content)

        # 返回原注释 + 翻译
        return f"{indent}{prefix}{content}\n{indent}{prefix}中文: {translated}"


class FileProcessor:
    """文件处理类"""

    def __init__(self, extractor: CommentExtractor, dry_run: bool = False, backup: bool = False):
        """
        初始化文件处理器

        Args:
            extractor: 注释提取器实例
            dry_run: 是否仅预览不修改
            backup: 是否备份原文件
        """
        self.extractor = extractor
        self.dry_run = dry_run
        self.backup = backup
        self.processed_count = 0
        self.modified_count = 0

    def process_file(self, filepath: Path) -> bool:
        """
        处理单个Python文件

        Args:
            filepath: 文件路径

        Returns:
            是否成功处理
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                original_content = f.read()
        except UnicodeDecodeError:
            print(f"跳过 (编码错误): {filepath}")
            return False
        except Exception as e:
            print(f"读取失败 {filepath}: {e}")
            return False

        self.processed_count += 1

        print(f"正在处理: {filepath}")

        # 1. 先处理docstrings (因为它们包含多行，且不用#开头，容易从整体提取)
        # 注意：这里有个策略问题。如果先处理#，可能会影响"""内部的内容吗？通常不会，因为"""里不应该有#开头的独立行（除非是代码示例）。
        # 最好的方式是分别处理，但要注意重叠。
        # 简单起见，按顺序处理。Docstring通常是结构化的，先处理比较安全。

        new_content = self.extractor.DOCSTRING_PATTERN.sub(
            self.extractor.process_docstring,
            original_content
        )

        # 2. 处理单行注释
        new_content = self.extractor.SINGLE_COMMENT_PATTERN.sub(
            self.extractor.process_single_comment,
            new_content
        )

        # 检查是否有变化
        if new_content == original_content:
            return True

        if self.dry_run:
            print(f"  [预览] 检测到可翻译内容")
            # 简单展示点差异
            # diff = '\n'.join(list(difflib.unified_diff(original_content.splitlines(), new_content.splitlines())))
            # print(diff[:500] + '...')
            return True

        # 备份原文件
        if self.backup:
            backup_path = filepath.with_suffix('.py.bak')
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(original_content)

        # 写入新内容
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            self.modified_count += 1
            print(f"  => 已保存修改")
            return True
        except Exception as e:
            print(f"  => 写入失败: {e}")
            return False

    def process_directory(self, directory: Path, exclude: Optional[list] = None):
        """
        递归处理目录中的所有Python文件

        Args:
            directory: 目标目录
            exclude: 排除的文件/目录模式列表
        """
        exclude = exclude or []

        files = list(directory.rglob('*.py'))
        print(f"找到 {len(files)} 个Python文件")

        for filepath in files:
            # 检查是否在排除列表中
            relative = filepath.relative_to(directory)
            skip = False
            for pattern in exclude:
                if pattern in str(relative):
                    skip = True
                    break
            if skip:
                # print(f"跳过 (排除): {filepath}")
                continue

            self.process_file(filepath)

    def print_summary(self):
        """打印处理摘要"""
        print("\n" + "=" * 50)
        print("处理完成!")
        print(f"扫描文件数: {self.processed_count}")
        print(f"修改文件数: {self.modified_count}")
        print("=" * 50)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="将Python代码中的英文注释翻译为中文 (使用Google Translate免费版)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python translate_comments.py --dir ./src
  python translate_comments.py --file ./script.py
  python translate_comments.py --dir ./libs --dry-run
  python translate_comments.py --dir ./project --backup
        """
    )

    # 创建互斥组，必须指定 --dir 或 --file 其中之一
    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        '--dir', '-d',
        help='目标目录路径'
    )

    group.add_argument(
        '--file', '-f',
        help='目标文件路径'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='仅预览将要修改的文件，不实际修改'
    )

    parser.add_argument(
        '--backup',
        action='store_true',
        help='修改前备份原文件 (.py.bak)'
    )

    parser.add_argument(
        '--exclude', '-e',
        nargs='+',
        default=[],
        help='排除的文件或目录模式'
    )

    args = parser.parse_args()

    # 初始化组件
    print(f"初始化 Google 翻译器...")
    translator = GoogleTransTranslator(target_lang="zh-CN")

    extractor = CommentExtractor(translator)
    processor = FileProcessor(extractor, dry_run=args.dry_run, backup=args.backup)

    # 根据参数类型执行处理
    if args.dir:
        target_dir = Path(args.dir)
        if not target_dir.exists():
            print(f"错误: 目录不存在: {target_dir}")
            sys.exit(1)
        if not target_dir.is_dir():
            print(f"错误: 不是目录: {target_dir}")
            sys.exit(1)

        print(f"\n开始处理目录: {target_dir}")
        if args.dry_run:
            print("[预览模式 - 不会实际修改文件]")
        if args.backup:
            print("[备份模式 - 原文件将保存为 .py.bak]")
        print("-" * 50)

        processor.process_directory(target_dir, exclude=args.exclude)

    elif args.file:
        target_file = Path(args.file)
        if not target_file.exists():
            print(f"错误: 文件不存在: {target_file}")
            sys.exit(1)
        if not target_file.is_file():
            print(f"错误: 不是文件: {target_file}")
            sys.exit(1)

        print(f"\n开始处理文件: {target_file}")
        if args.dry_run:
            print("[预览模式 - 不会实际修改文件]")
        if args.backup:
            print("[备份模式 - 原文件将保存为 .py.bak]")
        print("-" * 50)

        processor.process_file(target_file)

    processor.print_summary()


if __name__ == "__main__":
    main()
