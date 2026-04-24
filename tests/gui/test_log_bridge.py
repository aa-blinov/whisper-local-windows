"""Tests for the QtLogBridge that forwards logging records into a Qt signal."""

import logging


def test_bridge_emits_signal_per_record(qtbot):
    from app.gui.log_bridge import QtLogBridge

    bridge = QtLogBridge()
    logger = logging.getLogger("test.bridge.emit")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(bridge.handler())
    try:
        with qtbot.waitSignal(bridge.line_received, timeout=1000) as blocker:
            logger.info("hello world")
        assert "hello world" in blocker.args[0]
    finally:
        logger.removeHandler(bridge.handler())


def test_bridge_install_and_uninstall_on_logger(qtbot):
    from app.gui.log_bridge import QtLogBridge

    bridge = QtLogBridge()
    logger = logging.getLogger("test.bridge.install")
    logger.setLevel(logging.DEBUG)

    received: list[str] = []
    bridge.line_received.connect(received.append)

    bridge.install(logger)
    logger.info("one")
    bridge.uninstall(logger)
    logger.info("two")

    assert any("one" in line for line in received)
    assert not any("two" in line for line in received)


def test_bridge_format_includes_level(qtbot):
    from app.gui.log_bridge import QtLogBridge

    bridge = QtLogBridge()
    logger = logging.getLogger("test.bridge.level")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(bridge.handler())
    try:
        with qtbot.waitSignal(bridge.line_received, timeout=1000) as blocker:
            logger.warning("careful")
        assert "WARNING" in blocker.args[0]
    finally:
        logger.removeHandler(bridge.handler())


def test_bridge_respects_handler_level_filter(qtbot):
    from app.gui.log_bridge import QtLogBridge

    bridge = QtLogBridge()
    bridge.handler().setLevel(logging.WARNING)
    logger = logging.getLogger("test.bridge.filter")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(bridge.handler())

    received: list[str] = []
    bridge.line_received.connect(received.append)
    try:
        logger.debug("debug-should-drop")
        logger.info("info-should-drop")
        logger.warning("warn-should-pass")
    finally:
        logger.removeHandler(bridge.handler())

    assert any("warn-should-pass" in line for line in received)
    assert not any("debug-should-drop" in line for line in received)
    assert not any("info-should-drop" in line for line in received)
