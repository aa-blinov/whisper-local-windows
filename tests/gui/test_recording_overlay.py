import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication


def test_overlay_starts_hidden(qtbot):
    from app.gui.widgets.recording_overlay import RecordingOverlay

    overlay = RecordingOverlay()
    qtbot.addWidget(overlay)

    assert overlay.state() == "idle"
    assert not overlay.isVisible()
    assert overlay.testAttribute(Qt.WA_TranslucentBackground)
    if sys.platform == "darwin":
        assert overlay.testAttribute(Qt.WA_MacAlwaysShowToolWindow)


def test_overlay_shows_recording_state(qtbot):
    from app.gui.widgets.recording_overlay import RecordingOverlay

    overlay = RecordingOverlay()
    qtbot.addWidget(overlay)

    overlay.set_state("recording")

    assert overlay.isVisible()
    assert overlay.state() == "recording"
    assert overlay._title.text() == "Recording"
    assert overlay._body.text() == "Speak now"
    assert overlay._surface.isVisible()


def test_overlay_shows_processing_state(qtbot):
    from app.gui.widgets.recording_overlay import RecordingOverlay

    overlay = RecordingOverlay()
    qtbot.addWidget(overlay)

    overlay.set_state("processing")

    assert overlay.isVisible()
    assert overlay.state() == "processing"
    assert overlay._title.text() == "Processing"
    assert overlay._body.text() == "Transcribing speech"
    assert overlay._dot.property("state") == "processing"


def test_overlay_hides_on_idle(qtbot):
    from app.gui.widgets.recording_overlay import RecordingOverlay

    overlay = RecordingOverlay()
    qtbot.addWidget(overlay)
    overlay.set_state("recording")

    overlay.set_state("idle")

    assert overlay.state() == "idle"
    assert not overlay.isVisible()


def test_overlay_centers_on_primary_screen(qtbot):
    from app.gui.widgets.recording_overlay import RecordingOverlay

    overlay = RecordingOverlay()
    qtbot.addWidget(overlay)
    overlay.set_state("recording")

    screen = QApplication.primaryScreen()
    rect = screen.availableGeometry()
    expected_center = rect.x() + rect.width() // 2
    actual_center = overlay.frameGeometry().center().x()

    assert abs(actual_center - expected_center) <= 2
