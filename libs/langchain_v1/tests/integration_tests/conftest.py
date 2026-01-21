from pathlib import Path

import pytest
from dotenv import load_dotenv

# Getting the absolute path of the current file's directory
# 中文: 获取当前文件所在目录的绝对路径
ABS_PATH = Path(__file__).resolve().parent

# Getting the absolute path of the project's root directory
# 中文: 获取项目根目录的绝对路径
PROJECT_DIR = ABS_PATH.parent.parent


# Loading the .env file if it exists
# 中文: 加载 .env 文件（如果存在）
def _load_env() -> None:
    dotenv_path = PROJECT_DIR / "tests" / "integration_tests" / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path)


_load_env()


@pytest.fixture(scope="module")
def test_dir() -> Path:
    return PROJECT_DIR / "tests" / "integration_tests"


# This fixture returns a string containing the path to the cassette directory for the
# 中文: 该装置返回一个字符串，其中包含磁带目录的路径
# current module
# 中文: 当前模块
@pytest.fixture(scope="module")
def vcr_cassette_dir(request: pytest.FixtureRequest) -> str:
    module = Path(request.module.__file__)
    return str(module.parent / "cassettes" / module.stem)
