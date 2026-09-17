"""
Test suite for Napari plugin widgets: inference, evaluation, postprocessing.
Uses pytest and make_napari_viewer_proxy fixture.
"""

import importlib
import types

import aiod_utils.preprocess
import aiod_utils.rle as aiod_rle
import napari.layers
import numpy as np
import pytest
from aiod_utils.io import MASK_SEPARATOR, get_image_id, get_mask_prefix
from bioio_base.dimensions import Dimensions

from aiod_napari.inference.preprocess import PreprocessWidget

# Try to import the widgets
INFERENCE_WIDGET_PATH = "aiod_napari.inference.inference_widget"
EVALUATION_WIDGET_PATH = "aiod_napari.evaluation.evaluation_widget"
POSTPROCESS_WIDGET_PATH = "aiod_napari.postprocessing.postprocess_widget"


@pytest.mark.parametrize(
    "module_path",
    [
        INFERENCE_WIDGET_PATH,
        EVALUATION_WIDGET_PATH,
        POSTPROCESS_WIDGET_PATH,
    ],
)
def test_widget_module_import(module_path):
    """Test that widget modules can be imported."""
    module = importlib.import_module(module_path)
    assert module is not None


@pytest.fixture
def inference_widget_minimal(make_napari_viewer_proxy, tmp_path, monkeypatch):
    """Create a minimal Inference widget with a temp mask directory."""
    viewer = make_napari_viewer_proxy()
    _, widget = viewer.window.add_plugin_dock_widget("aiod-napari", "Inference")
    monkeypatch.setattr(widget, "store_settings", lambda: None)
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    widget.subwidgets["nxf"].mask_dir_path = mask_dir
    # run_hash is required by _get_final_mask_name; set a dummy value
    widget.run_hash = "a" * 32
    return viewer, widget, tmp_path


def _make_img_mask_info(img_path: object, layer_name: str, prep_options: list | None):
    """Build a single-entry img_mask_info list, as get_img_mask_preps would."""
    image_id = get_image_id(img_path)
    prep_hash = (
        aiod_utils.preprocess.get_prep_hash(prep_options) if prep_options else None
    )
    return [
        {
            "img_path": img_path,
            "image_id": image_id,
            "prep_hash": prep_hash,
            "layer_name": layer_name,
            "mask_prefix": get_mask_prefix(image_id, prep_hash),
            "prep_set": prep_options,
            "preprocess_str": None,
        }
    ]


class TestCreateMaskLayersRegressions:
    """Regression tests for Inference.create_mask_layers."""

    def test_2d_bioio_image_with_preprocessing_no_value_error(
        self, inference_widget_minimal, monkeypatch
    ):
        """Regression: create_mask_layers must not raise ValueError for a 2D image
        whose bioio metadata has dims.order='TCZYX' (full standard order) while
        the actual data has shape (H, W) — causing the previous zip to pair T and C
        with the data axes and yield an empty spatial shape."""
        viewer, widget, tmp_path = inference_widget_minimal

        img_name = "test_2d"
        img_path = tmp_path / f"{img_name}.tif"
        img_path.touch()
        # A plain 2D image as napari would have after loading via bioio.
        # bioio stores dims.order="TCZYX" even for YX-only data.
        viewer.add_image(np.zeros((64, 64), dtype=np.uint8), name=img_name)
        viewer.layers[img_name].metadata = {
            # As prepare_bioio_as_napari_layer records it - the image layer is
            # resolved by path, not by name
            "path": img_path,
            "dimensions": Dimensions(dims="TCZYX", shape=(1, 1, 1, 64, 64)),
        }
        layer_name = f"{img_name}_masks_test"
        # Preprocessing with CLAHE only (no shape change) to trigger the
        # get_output_shape code path without a dimensionality mismatch.
        prep_options = [
            {"name": "CLAHE", "params": {"clipLimit": 3.0, "tileGridSize": [8, 8]}}
        ]

        monkeypatch.setattr(widget, "get_img_mask_preps", lambda *a, **kw: None)
        widget.img_mask_info = _make_img_mask_info(img_path, layer_name, prep_options)
        monkeypatch.setattr(
            widget, "_get_final_mask_name", lambda *a, **kw: "nonexistent_mask.rle"
        )

        # Must not raise ValueError("Shape must be a Stack…")
        widget.create_mask_layers()

        assert layer_name in viewer.layers
        assert viewer.layers[layer_name].data.shape == (64, 64)


class TestUpdateMasksSubstacks:
    """
    Regression tests for Inference.update_masks.

    create_mask_layers sizes the layer to the model's full output, and each
    substack fills a subset of it - as combine_stacks.py does with the same
    files. Sizing the layer from a substack instead left later substacks
    indexing past the end of it.
    """

    def _setup(self, viewer, widget, tmp_path, monkeypatch, img_shape, prep_options):
        img_name = "vol"
        img_path = tmp_path / f"{img_name}.tif"
        img_path.touch()
        viewer.add_image(np.zeros(img_shape, dtype=np.uint8), name=img_name)
        viewer.layers[img_name].metadata = {
            "path": img_path,
            # The full standard bioio order, as a real load gives it - the
            # present dims are filtered out of it downstream
            "dimensions": Dimensions(dims="TCZYX", shape=(1, 1, *img_shape)),
        }
        layer_name = f"{img_name}_masks_test"
        monkeypatch.setattr(widget, "get_img_mask_preps", lambda *a, **kw: None)
        monkeypatch.setattr(
            widget, "_get_final_mask_name", lambda *a, **kw: "nonexistent_mask.rle"
        )
        widget.img_mask_info = _make_img_mask_info(img_path, layer_name, prep_options)
        widget.mask_info_by_prefix = {i["mask_prefix"]: i for i in widget.img_mask_info}
        nxf = widget.subwidgets["nxf"]
        nxf.progress_dict = {i["image_id"]: 0 for i in widget.img_mask_info}
        # Needs a live tqdm bar, which only exists during a real run
        monkeypatch.setattr(nxf, "update_progress_bar", lambda: None)
        widget.create_mask_layers()
        return layer_name

    def _write_substack(self, widget, shape, z_range):
        """One substack .rle, named as Segment-Flow names it."""
        start_z, end_z = z_range
        sub = np.zeros((end_z - start_z, *shape[1:]), dtype=np.uint16)
        sub[:, :2, :2] = 1
        prefix = widget.img_mask_info[0]["mask_prefix"]
        fpath = (
            widget.subwidgets["nxf"].mask_dir_path
            / f"{prefix}{MASK_SEPARATOR}{widget.run_hash}"
            f"_x0-{shape[2]}_y0-{shape[1]}_z{start_z}-{end_z}.rle"
        )
        aiod_rle.save_encoding(aiod_rle.encode(sub, mask_type="instance"), fpath)
        return fpath

    def test_substacks_fill_the_full_layer(self, inference_widget_minimal, monkeypatch):
        viewer, widget, tmp_path = inference_widget_minimal
        img_shape = (52, 32, 32)
        layer_name = self._setup(viewer, widget, tmp_path, monkeypatch, img_shape, None)
        assert viewer.layers[layer_name].data.shape == img_shape

        files = [
            self._write_substack(widget, img_shape, z) for z in ((0, 26), (26, 52))
        ]
        widget.update_masks(files)

        layer = viewer.layers[layer_name]
        assert layer.data.shape == img_shape
        assert layer.data[:26].any()
        assert layer.data[26:].any()

    def test_downsampled_substacks_fill_the_full_layer(
        self, inference_widget_minimal, monkeypatch
    ):
        """
        Substack filenames carry the downsampled indices, so they address the
        downsampled layer directly and need no conversion.
        """
        viewer, widget, tmp_path = inference_widget_minimal
        img_shape = (52, 32, 32)
        mask_shape = (52, 16, 16)
        prep_options = [
            {
                "name": "Downsample",
                "params": {"block_size": [1, 2, 2], "method": "median"},
            }
        ]
        layer_name = self._setup(
            viewer, widget, tmp_path, monkeypatch, img_shape, prep_options
        )
        assert viewer.layers[layer_name].data.shape == mask_shape

        files = [
            self._write_substack(widget, mask_shape, z) for z in ((0, 26), (26, 52))
        ]
        widget.update_masks(files)

        layer = viewer.layers[layer_name]
        assert layer.data.shape == mask_shape
        assert layer.data[:26].any()
        assert layer.data[26:].any()

    def test_mismatched_substack_leaves_the_layer_alone(
        self, inference_widget_minimal, monkeypatch
    ):
        """
        A mask that does not fit its slot is skipped, not made to fit by
        resizing the layer - insert_final_masks corrects the layer at the end.
        """
        viewer, widget, tmp_path = inference_widget_minimal
        img_shape = (52, 32, 32)
        layer_name = self._setup(viewer, widget, tmp_path, monkeypatch, img_shape, None)
        # Claims z0-26 but holds 25 slices
        bad = self._write_substack(widget, img_shape, (0, 25))
        bad = bad.rename(bad.with_name(bad.name.replace("z0-25", "z0-26")))

        widget.update_masks([bad])

        assert viewer.layers[layer_name].data.shape == img_shape
        assert not viewer.layers[layer_name].data.any()


class TestSubstackCount:
    """
    The pre-run job count is only an estimate - Segment-Flow caps substacks
    using config (model_max_substack, substack_scale, memory_per_job) that the
    plugin cannot see, so the same image can split differently there.
    refresh_substack_total swaps in the count the pipeline publishes.
    """

    ESTIMATE = 3

    def _nxf(self, inference_widget_minimal, contents):
        viewer, widget, tmp_path = inference_widget_minimal
        nxf = widget.subwidgets["nxf"]
        nxf.total_substacks = self.ESTIMATE
        nxf.init_progress_bar()
        splits = tmp_path / "splits" / "substacks_abc.csv"
        splits.parent.mkdir(parents=True)
        splits.write_text(contents)
        nxf.splits_csv_path = splits
        return nxf

    def test_published_count_replaces_the_estimate(self, inference_widget_minimal):
        rows = "image_id,stack_idx\n" + "".join(f"vol,{i}\n" for i in range(4))
        nxf = self._nxf(inference_widget_minimal, rows)

        assert nxf.refresh_substack_total() is True
        assert nxf.total_substacks == 4
        assert nxf.substacks_exact
        # The bar has to be re-ranged, or progress past the estimate is clamped
        assert nxf.pbar.maximum() == 4
        assert nxf.tqdm_pbar.total == 4

    def test_estimate_stands_until_the_file_appears(self, inference_widget_minimal):
        nxf = self._nxf(inference_widget_minimal, "image_id,stack_idx\nvol,0\n")
        nxf.splits_csv_path = nxf.splits_csv_path.with_name("not_yet.csv")

        assert nxf.refresh_substack_total() is False
        assert nxf.total_substacks == self.ESTIMATE
        assert not nxf.substacks_exact

    def test_header_only_file_is_not_trusted(self, inference_widget_minimal):
        # publishDir copies rather than renames, so the file can be read mid-copy
        nxf = self._nxf(inference_widget_minimal, "image_id,stack_idx\n")

        assert nxf.refresh_substack_total() is False
        assert nxf.total_substacks == self.ESTIMATE

    def test_estimate_is_only_flagged_when_progress_runs_against_it(
        self, inference_widget_minimal
    ):
        rows = "image_id,stack_idx\n" + "".join(f"vol,{i}\n" for i in range(4))
        nxf = self._nxf(inference_widget_minimal, rows)

        # An empty bar shows nothing worth qualifying, which is the normal case:
        # the count is published before the first mask, so it is exact by the
        # time anything is drawn
        assert "est." not in nxf._progress_prefix(0)
        # Only a bar advancing against an unverified total needs the caveat
        assert "est." in nxf._progress_prefix(1)

        nxf.refresh_substack_total()
        assert "est." not in nxf._progress_prefix(1)


class TestSelectedLayerPreprocessDisplay:
    """PreprocessWidget's selected-layer preprocessing text box reads
    layer.metadata["preprocess_str"] directly - no re-deriving/re-hashing.

    Exercises the bound method against a duck-typed stand-in for the widget
    plus bare (never-added-to-a-viewer) layer objects, rather than a full
    make_napari_viewer_proxy - _update_selected_layer_prep only ever touches
    self.viewer.layers.selection.active and self.selected_layer_prep, so a
    real Qt/vispy canvas isn't needed and would just be a slower, GL-dependent
    way to exercise the same logic.
    """

    class _FakeLineEdit:
        def __init__(self):
            self._text = ""

        def setText(self, text):
            self._text = text

        def text(self):
            return self._text

    def _make_widget(self, active_layer):
        widget = types.SimpleNamespace()
        widget.selected_layer_prep = self._FakeLineEdit()
        widget.viewer = types.SimpleNamespace(
            layers=types.SimpleNamespace(
                selection=types.SimpleNamespace(active=active_layer)
            )
        )
        return widget

    def _update(self, widget):
        PreprocessWidget._update_selected_layer_prep(widget)
        return widget.selected_layer_prep.text()

    def test_no_layer_selected(self):
        widget = self._make_widget(active_layer=None)
        assert self._update(widget) == ""

    def test_non_labels_layer_selected(self):
        layer = napari.layers.Image(np.zeros((4, 4)))
        widget = self._make_widget(active_layer=layer)
        assert self._update(widget) == ""

    def test_labels_layer_without_preprocess_metadata(self):
        layer = napari.layers.Labels(np.zeros((4, 4), dtype=np.uint16))
        widget = self._make_widget(active_layer=layer)
        assert self._update(widget) == ""

    def test_labels_layer_with_preprocessing(self):
        recipe = "Downsample-block_size=1,1,1-method=median"
        layer = napari.layers.Labels(
            np.zeros((4, 4), dtype=np.uint16), metadata={"preprocess_str": recipe}
        )
        widget = self._make_widget(active_layer=layer)
        assert self._update(widget) == recipe

    def test_labels_layer_with_no_op_preprocessing(self):
        layer = napari.layers.Labels(
            np.zeros((4, 4), dtype=np.uint16), metadata={"preprocess_str": None}
        )
        widget = self._make_widget(active_layer=layer)
        assert self._update(widget) == "No preprocessing"


def test_about_window_renders_links_in_link_colour(qtbot):
    """
    Render the About text and check the pixels, as the colour is easily lost:
    napari's app-wide stylesheet overrides the QPalette.Link role, so a link can
    silently fall back to Qt's default blue.
    """
    from napari._qt.qt_resources import get_stylesheet
    from qtpy.QtCore import Qt
    from qtpy.QtGui import QColor, QImage, QPainter
    from qtpy.QtWidgets import QApplication, QLabel

    from aiod_napari.utils import LINK_COLOUR, AboutWindow, html_link

    app = QApplication.instance()
    previous_stylesheet = app.styleSheet()
    app.setStyleSheet(get_stylesheet("dark"))
    try:
        window = AboutWindow(
            title="About", content=html_link("https://example.com", "documentation")
        )
        qtbot.addWidget(window)
        label = window.findChild(QLabel)
        label.setFixedSize(400, 60)

        image = QImage(label.size(), QImage.Format_ARGB32)
        image.fill(Qt.transparent)
        painter = QPainter(image)
        label.render(painter)
        painter.end()
    finally:
        app.setStyleSheet(previous_stylesheet)

    expected = QColor(LINK_COLOUR)
    # Anti-aliasing blends the text edges, so only the glyph cores match exactly
    matches = sum(
        QColor(image.pixel(x, y)) == expected
        for y in range(image.height())
        for x in range(image.width())
    )
    assert matches > 0


class TestInfoWindow:
    def test_content_is_not_interpreted_as_markup(self, qtbot):
        """
        A YAML dump of pipeline params can contain markup-shaped values, and its
        indentation is meaningful, so it must be shown verbatim rather than left
        to Qt's rich-text sniffing.
        """
        from aiod_napari.utils import InfoWindow

        content = "model_config: <not-a-tag>\nparams:\n    diameter: 30"
        window = InfoWindow(title="Info", content=content)
        qtbot.addWidget(window)

        assert window.info_label.toPlainText() == content
