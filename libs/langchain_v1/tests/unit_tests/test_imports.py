import importlib
import warnings
from pathlib import Path

# Attempt to recursively import all modules in langchain
# 中文: 尝试递归导入langchain中的所有模块
PKG_ROOT = Path(__file__).parent.parent.parent


def test_import_all() -> None:
    """Generate the public API for this package.

    中文翻译:
    为此包生成公共 API。"""
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", category=UserWarning)
        library_code = PKG_ROOT / "langchain"
        for path in library_code.rglob("*.py"):
            # Calculate the relative path to the module
            # 中文: 计算模块的相对路径
            module_name = path.relative_to(PKG_ROOT).with_suffix("").as_posix().replace("/", ".")
            if module_name.endswith("__init__"):
                # Without init
                # 中文: 没有初始化
                module_name = module_name.rsplit(".", 1)[0]

            mod = importlib.import_module(module_name)

            all_attrs = getattr(mod, "__all__", [])

            for name in all_attrs:
                # Attempt to import the name from the module
                # 中文: 尝试从模块导入名称
                try:
                    obj = getattr(mod, name)
                    assert obj is not None
                except Exception as e:
                    msg = f"Could not import {module_name}.{name}"
                    raise AssertionError(msg) from e


def test_import_all_using_dir() -> None:
    """Generate the public API for this package.

    中文翻译:
    为此包生成公共 API。"""
    library_code = PKG_ROOT / "langchain"
    for path in library_code.rglob("*.py"):
        # Calculate the relative path to the module
        # 中文: 计算模块的相对路径
        module_name = path.relative_to(PKG_ROOT).with_suffix("").as_posix().replace("/", ".")
        if module_name.endswith("__init__"):
            # Without init
            # 中文: 没有初始化
            module_name = module_name.rsplit(".", 1)[0]

        try:
            mod = importlib.import_module(module_name)
        except ModuleNotFoundError as e:
            msg = f"Could not import {module_name}"
            raise ModuleNotFoundError(msg) from e
        attributes = dir(mod)

        for name in attributes:
            if name.strip().startswith("_"):
                continue
            # Attempt to import the name from the module
            # 中文: 尝试从模块导入名称
            getattr(mod, name)
