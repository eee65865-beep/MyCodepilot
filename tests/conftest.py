"""为每次测试提供独立临时目录，避免 Windows 共享目录权限冲突。"""

import tempfile
import socket
import ipaddress

import pytest


TEMP_DIRECTORY = pytest.StashKey[tempfile.TemporaryDirectory]()


def pytest_addoption(parser) -> None:
    parser.addoption(
        "--run-integration", action="store_true", default=False,
        help="显式运行 integration 标记的真实 API 测试（可能消耗额度）",
    )


def pytest_collection_modifyitems(config, items) -> None:
    if not config.getoption("--run-integration"):
        skip = pytest.mark.skip(reason="真实 API 测试默认关闭，使用 --run-integration 开启")
        for item in items:
            if item.get_closest_marker("integration"):
                item.add_marker(skip)


@pytest.fixture(autouse=True)
def offline_unit_tests(request, monkeypatch):
    """普通测试不使用真实密钥，并阻止意外网络连接。"""
    if request.node.get_closest_marker("integration"):
        return
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-placeholder")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-chat")

    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def is_local(address):
        if not isinstance(address, tuple):
            return False
        try:
            return address[0] == "localhost" or ipaddress.ip_address(address[0]).is_loopback
        except ValueError:
            return False

    def blocked(connection, address):
        if is_local(address):
            return original_connect(connection, address)
        raise AssertionError("基础单元测试禁止外部网络连接，请使用 mock。")

    def blocked_ex(connection, address):
        if is_local(address):
            return original_connect_ex(connection, address)
        raise AssertionError("基础单元测试禁止外部网络连接，请使用 mock。")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked_ex)


def pytest_configure(config: pytest.Config) -> None:
    # 尊重调用者显式指定的 --basetemp。
    if config.option.basetemp is None:
        directory = tempfile.TemporaryDirectory(prefix="codepilot-pytest-")
        config.stash[TEMP_DIRECTORY] = directory
        config.option.basetemp = directory.name


def pytest_unconfigure(config: pytest.Config) -> None:
    directory = config.stash.get(TEMP_DIRECTORY, None)
    if directory is not None:
        directory.cleanup()
