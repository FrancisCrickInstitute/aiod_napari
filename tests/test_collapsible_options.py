"""
Tests for the reusable CollapsibleOptions widget.
"""

from qtpy.QtWidgets import QLabel, QVBoxLayout

from aiod_napari.widget_classes import CollapsibleOptions


def test_starts_collapsed(qtbot):
    options = CollapsibleOptions()
    qtbot.addWidget(options)

    assert not options.is_expanded()
    assert not options.content_widget.isVisible()
    assert "▶" in options.toggle_btn.text()


def test_toggle_shows_and_hides_content(qtbot):
    options = CollapsibleOptions(title="Advanced Options")
    qtbot.addWidget(options)
    options.show()

    options.toggle_btn.click()
    assert options.is_expanded()
    assert options.content_widget.isVisible()
    assert options.toggle_btn.text() == " ▼ Advanced Options"

    options.toggle_btn.click()
    assert not options.is_expanded()
    assert not options.content_widget.isVisible()
    assert options.toggle_btn.text() == " ▶ Advanced Options"


def test_set_expanded_matches_button_state(qtbot):
    options = CollapsibleOptions()
    qtbot.addWidget(options)

    options.set_expanded()
    assert options.toggle_btn.isChecked()
    assert options.is_expanded()

    options.set_expanded(False)
    assert not options.toggle_btn.isChecked()
    assert not options.is_expanded()


def test_content_widgets_are_hidden_until_expanded(qtbot):
    options = CollapsibleOptions(layout=QVBoxLayout)
    qtbot.addWidget(options)
    label = QLabel("hidden option")
    options.content_layout.addWidget(label)
    options.show()

    assert not label.isVisible()

    options.set_expanded()
    assert label.isVisible()
