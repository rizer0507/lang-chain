"""A unit test meant to catch accidental introduction of non-optional dependencies.

中文翻译:
单元测试旨在捕获意外引入的非可选依赖项。"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import toml
from packaging.requirements import Requirement

HERE = Path(__file__).parent

PYPROJECT_TOML = HERE / "../../pyproject.toml"


@pytest.fixture
def uv_conf() -> dict[str, Any]:
    """Load the pyproject.toml file.

    中文翻译:
    加载 pyproject.toml 文件。"""
    with PYPROJECT_TOML.open() as f:
        return toml.load(f)


def test_required_dependencies(uv_conf: Mapping[str, Any]) -> None:
    """A test that checks if a new non-optional dependency is being introduced.

    If this test is triggered, it means that a contributor is trying to introduce a new
    required dependency. This should be avoided in most situations.
    

    中文翻译:
    检查是否引入新的非可选依赖项的测试。
    如果触发此测试，则意味着贡献者正在尝试引入新的
    所需的依赖。在大多数情况下应该避免这种情况。"""
    # Get the dependencies from the [tool.poetry.dependencies] section
    # 中文: 从 [tool.poetry.dependencies] 部分获取依赖项
    dependencies = uv_conf["project"]["dependencies"]
    required_dependencies = {Requirement(dep).name for dep in dependencies}

    assert sorted(required_dependencies) == sorted(
        [
            "langchain-core",
            "langgraph",
            "pydantic",
        ]
    )
