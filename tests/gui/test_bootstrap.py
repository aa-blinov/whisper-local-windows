"""Sanity checks that pytest-qt bootstrap and the app.gui package are wired up."""


def test_pyside6_importable():
    from PySide6.QtWidgets import QApplication  # noqa: F401
    from PySide6.QtCore import Qt  # noqa: F401


def test_app_gui_package_importable():
    import app.gui  # noqa: F401


def test_qapplication_instantiates(qtbot):
    from PySide6.QtWidgets import QWidget

    widget = QWidget()
    qtbot.addWidget(widget)
    assert widget is not None
