"""Configuration for unit tests.

中文翻译:
单元测试的配置。"""

from collections.abc import Sequence
from importlib import util
from typing import Any

import pytest
from langchain_tests.conftest import CustomPersister, CustomSerializer, base_vcr_config
from vcr import VCR

_EXTRA_HEADERS = [
    ("openai-organization", "PLACEHOLDER"),
    ("user-agent", "PLACEHOLDER"),
    ("x-openai-client-user-agent", "PLACEHOLDER"),
]


def remove_request_headers(request: Any) -> Any:
    """Remove sensitive headers from the request.

    中文翻译:
    从请求中删除敏感标头。"""
    for k in request.headers:
        request.headers[k] = "**REDACTED**"
    request.uri = "**REDACTED**"
    return request


def remove_response_headers(response: dict) -> dict:
    """Remove sensitive headers from the response.

    中文翻译:
    从响应中删除敏感标头。"""
    for k in response["headers"]:
        response["headers"][k] = "**REDACTED**"
    return response


@pytest.fixture(scope="session")
def vcr_config() -> dict:
    """Extend the default configuration coming from langchain_tests.

    中文翻译:
    扩展来自 langchain_tests 的默认配置。"""
    config = base_vcr_config()
    config.setdefault("filter_headers", []).extend(_EXTRA_HEADERS)
    config["before_record_request"] = remove_request_headers
    config["before_record_response"] = remove_response_headers
    config["serializer"] = "yaml.gz"
    config["path_transformer"] = VCR.ensure_suffix(".yaml.gz")
    return config


def pytest_recording_configure(config: dict, vcr: VCR) -> None:  # noqa: ARG001
    vcr.register_persister(CustomPersister())
    vcr.register_serializer("yaml.gz", CustomSerializer())


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add custom command line options to pytest.

    中文翻译:
    将自定义命令行选项添加到 pytest.txt"""
    parser.addoption(
        "--only-extended",
        action="store_true",
        help="Only run extended tests. Does not allow skipping any extended tests.",
    )
    parser.addoption(
        "--only-core",
        action="store_true",
        help="Only run core tests. Never runs any extended tests.",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: Sequence[pytest.Function]) -> None:
    """Add implementations for handling custom markers.

    At the moment, this adds support for a custom `requires` marker.

    The `requires` marker is used to denote tests that require one or more packages
    to be installed to run. If the package is not installed, the test is skipped.

    The `requires` marker syntax is:

    ```python
    @pytest.mark.requires("package1", "package2")
    def test_something(): ...
    ```
    

    中文翻译:
    添加处理自定义标记的实现。
    目前，这增加了对自定义“requires”标记的支持。
    “requires”标记用于表示需要一个或多个包的测试
    才能安装运行。如果未安装该包，则跳过测试。
    `requires` 标记语法是：
    ````蟒蛇
    @pytest.mark.requires("package1", "package2")
    def test_something(): ...
    ````"""
    # Mapping from the name of a package to whether it is installed or not.
    # 中文: 从包的名称到它是否已安装的映射。
    # Used to avoid repeated calls to `util.find_spec`
    # 中文: 用于避免重复调用“util.find_spec”
    required_pkgs_info: dict[str, bool] = {}

    only_extended = config.getoption("--only-extended", default=False)
    only_core = config.getoption("--only-core", default=False)

    if only_extended and only_core:
        msg = "Cannot specify both `--only-extended` and `--only-core`."
        raise ValueError(msg)

    for item in items:
        requires_marker = item.get_closest_marker("requires")
        if requires_marker is not None:
            if only_core:
                item.add_marker(pytest.mark.skip(reason="Skipping not a core test."))
                continue

            # Iterate through the list of required packages
            # 中文: 遍历所需包的列表
            required_pkgs = requires_marker.args
            for pkg in required_pkgs:
                # If we haven't yet checked whether the pkg is installed
                # 中文: 如果我们还没有检查pkg是否安装
                # let's check it and store the result.
                # 中文: 让我们检查一下并存储结果。
                if pkg not in required_pkgs_info:
                    try:
                        installed = util.find_spec(pkg) is not None
                    except Exception:
                        installed = False
                    required_pkgs_info[pkg] = installed

                if not required_pkgs_info[pkg]:
                    if only_extended:
                        pytest.fail(
                            f"Package `{pkg}` is not installed but is required for "
                            f"extended tests. Please install the given package and "
                            f"try again.",
                        )

                    else:
                        # If the package is not installed, we immediately break
                        # 中文: 如果包没有安装，我们立即破解
                        # and mark the test as skipped.
                        # 中文: 并将测试标记为已跳过。
                        item.add_marker(
                            pytest.mark.skip(reason=f"Requires pkg: `{pkg}`"),
                        )
                        break
        elif only_extended:
            item.add_marker(
                pytest.mark.skip(reason="Skipping not an extended test."),
            )
