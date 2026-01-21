import pytest
import pytest_socket
import requests


def test_socket_disabled() -> None:
    """This test should fail.

    中文翻译:
    这个测试应该会失败。"""
    with pytest.raises(pytest_socket.SocketBlockedError):
        requests.get("https://www.example.com", timeout=1)
