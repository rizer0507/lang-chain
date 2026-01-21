"""Check imports script.

Quickly verify that a list of Python files can be loaded by the Python interpreter
without raising any errors. Ran before running more expensive tests. Useful in
Makefiles.

If loading a file fails, the script prints the problematic filename and the detailed
error traceback.

中文翻译:
检查导入脚本。
快速验证Python解释器是否可以加载Python文件列表
不会引发任何错误。在运行更昂贵的测试之前运行。有用于
生成文件。
如果加载文件失败，脚本会打印有问题的文件名和详细信息
错误回溯。
"""

import random
import string
import sys
import traceback
from importlib.machinery import SourceFileLoader

if __name__ == "__main__":
    files = sys.argv[1:]
    has_failure = False
    for file in files:
        try:
            module_name = "".join(
                random.choice(string.ascii_letters)  # noqa: S311
                for _ in range(20)
            )
            SourceFileLoader(module_name, file).load_module()
        except Exception:
            has_failure = True
            print(file)  # noqa: T201
            traceback.print_exc()
            print()  # noqa: T201

    sys.exit(1 if has_failure else 0)
