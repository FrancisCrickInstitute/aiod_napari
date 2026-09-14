"""Tests for the preprocessing subwidget.

The widget is driven in isolation (``parent=None``). The bugs these cover
produce plausible wrong *values* rather than errors, and several repair
themselves once the widget is touched, so they resist checking by hand.
"""

import pytest
from aiod_utils.preprocess import (
    Downsample,
    get_all_preprocess_methods,
)

from aiod_napari.inference.preprocess import PreprocessWidget


@pytest.fixture
def preprocess_widget(make_napari_viewer_proxy):
    """A standalone PreprocessWidget with no parent plugin widget."""
    return PreprocessWidget(viewer=make_napari_viewer_proxy())


def _params(widget, method):
    return widget.preprocess_boxes[method]["params"]


def test_spinboxes_reject_values_the_backend_rejects(preprocess_widget):
    """Numeric params are bounded by their metadata, not by Qt's 0-99 default.

    Zero is invalid nearly everywhere: block_reduce and square(0) both reject it.
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

    This is the path a config takes through _load_options_into_ui.
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


def test_short_block_size_is_read_the_way_the_backend_reads_it(preprocess_widget):
    """A 2-element block_size means (H, W), not (D, H).

    aiod_utils supplies the depth factor, so the UI must omit the same entry.
    """
    w = preprocess_widget

    w._load_options_into_ui(
        [{"name": "Downsample", "params": {"block_size": [4, 4], "method": "mean"}}]
    )

    loaded = tuple(w.extract_options()[0]["params"]["block_size"])
    backend = Downsample({"block_size": (4, 4), "method": "mean"}).kwarg_params[
        "block_size"
    ]
    assert loaded == backend == (1, 4, 4)


def test_unusable_list_length_warns_and_keeps_defaults(preprocess_widget, monkeypatch):
    """A length that cannot be lined up is reported, not quietly half-applied."""
    warned = []
    monkeypatch.setattr("aiod_napari.inference.preprocess.show_warning", warned.append)
    w = preprocess_widget

    w._load_options_into_ui(
        [{"name": "Downsample", "params": {"block_size": [9], "method": "mean"}}]
    )

    assert w.extract_options()[0]["params"]["block_size"] == [1, 2, 2]
    assert any("block_size" in str(m) for m in warned)
