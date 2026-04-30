"""Shared fixtures for the Qt GUI test tree.

Top-level ``conftest.py`` already pins the offscreen Qt platform.
This file adds an ``autouse`` fixture that stubs the modal-dialog
helpers (``confirm`` / ``confirm_three_way`` / ``notify``) at every
callsite where the production code imports them.

Why an autouse stub
-------------------
``QMessageBox.exec()`` is a blocking modal — pytest can't dismiss
it, so any test that triggers a code path which shows a dialog
hangs forever. The previous code monkey-patched
``QMessageBox.question`` / ``information`` static methods directly;
those are still the names tests historically used, but the
production code now goes through ``app.gui.widgets.dialogs.confirm``
/ ``notify`` which build a ``QMessageBox`` instance. Stubbing the
helpers here keeps those tests passing without each one needing
its own monkey-patch boilerplate.

Tests that want to assert a particular dialog answer (e.g.
"Cancel doesn't delete") simply override the relevant attribute
with their own ``monkeypatch.setattr(...)`` — pytest applies
overrides on top of fixtures.
"""

from __future__ import annotations

import pytest


_HELPER_TARGETS = (
    # Each entry: ``module.attr`` and a default-stub return value.
    # Stubs accept ``*a, **kw`` so the helpers' optional
    # ``yes_label`` / ``default_yes`` etc. don't trip them up.
    ("app.gui.controllers._history_mixin.confirm", True),
    ("app.gui.controllers._history_mixin.notify", None),
    ("app.gui.controllers._storage_mixin.notify", None),
    ("app.gui.controllers._storage_mixin.confirm_three_way", "no"),
    ("app.gui.controllers.app_controller.confirm", True),
    # The dialogs module itself, in case a future caller imports
    # the helper at module load and the per-callsite patches above
    # don't catch it.
    ("app.gui.widgets.dialogs.confirm", True),
    ("app.gui.widgets.dialogs.confirm_three_way", "no"),
    ("app.gui.widgets.dialogs.notify", None),
)


@pytest.fixture(autouse=True)
def stub_modal_dialogs(monkeypatch: pytest.MonkeyPatch):
    """Replace every modal-dialog helper with a non-blocking stub
    so tests don't hang on ``QMessageBox.exec()``.

    The default for ``confirm_three_way`` is ``"no"`` (leave-as-is)
    rather than ``"yes"`` because most code paths gate destructive
    work behind ``"yes"`` and a default of "yes" would silently
    perform actions tests didn't ask for. Tests that need a
    specific answer override this fixture's stub:

    .. code-block:: python

        monkeypatch.setattr(
            "app.gui.controllers._storage_mixin.confirm_three_way",
            lambda *a, **kw: "yes",
        )
    """
    for target, default in _HELPER_TARGETS:
        monkeypatch.setattr(
            target,
            lambda *_a, _default=default, **_kw: _default,
            raising=False,
        )
    yield
