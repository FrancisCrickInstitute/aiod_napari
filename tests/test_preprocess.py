"""Tests for the preprocessing subwidget.

These exercise the widget in isolation (``parent=None``), so the dimension
state is driven by calling ``_apply_dim_state`` directly rather than by
loading images. That is deliberate: the bugs these cover produce plausible
wrong *values* rather than errors, and several of them repair themselves as
soon as the widget is interacted with, so they are near-impossible to pin down
by hand.
"""

import pytest
from aiod_utils.preprocess import get_all_preprocess_methods

from aiod_napari.inference.preprocess import PreprocessWidget


@pytest.fixture
def preprocess_widget(make_napari_viewer_proxy):
    """A standalone PreprocessWidget with no parent plugin widget."""
    return PreprocessWidget(viewer=make_napari_viewer_proxy())


def _params(widget, method):
    return widget.preprocess_boxes[method]["params"]


def test_spinboxes_reject_values_the_backend_rejects(preprocess_widget):
    """Numeric params are bounded by their metadata, not by Qt's 0-99 default.

    A zero here is not merely odd: block_reduce refuses factors below 1, and a
    square/cube footprint of 0 is empty, which the rank filters fail an
    assertion on.
    """
    methods = get_all_preprocess_methods()

    size = _params(preprocess_widget, "Filter")["size"]
    assert size.minimum() == methods["Filter"]["params"]["size"]["min"] == 1
    assert size.maximum() >= 100

    # block_size is a list-backed param, so every subwidget takes the bounds
    for factor in _params(preprocess_widget, "Downsample")["block_size"]:
        assert factor.minimum() == 1
        assert factor.maximum() >= 1000

    for tile in _params(preprocess_widget, "CLAHE")["tileGridSize"]:
        assert tile.minimum() == 1


def test_spinboxes_accept_values_above_the_qt_default(preprocess_widget):
    """setValue past 99 must survive, not silently clamp.

    This is the path a config file takes through _load_options_into_ui, where
    a clamp would quietly rewrite a saved parameter.
    """
    size = _params(preprocess_widget, "Filter")["size"]
    size.setValue(100)
    assert size.value() == 100

    clip = _params(preprocess_widget, "CLAHE")["clipLimit"]
    clip.setValue(40.0)
    assert clip.value() == pytest.approx(40.0)


def test_defaults_survive_being_bounded(preprocess_widget):
    """Every widget still holds its declared default after configuration."""
    for method, method_def in get_all_preprocess_methods().items():
        for param_name, param_def in method_def["params"].items():
            widget = _params(preprocess_widget, method)[param_name]
            default = param_def["default"]
            if isinstance(widget, list):
                assert [w.value() for w in widget] == list(default)
            elif hasattr(widget, "value"):
                assert widget.value() == pytest.approx(default)
