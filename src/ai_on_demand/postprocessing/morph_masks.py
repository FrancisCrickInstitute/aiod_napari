from typing import Optional

from aiod_utils.preprocess import Filter
import dask.array as da
import dask_image.ndmeasure as dask_ndi
import dask_image.ndmorph as dask_morph
import napari
from napari.utils.notifications import show_error
import numpy as np
import qtpy.QtCore
from qtpy.QtWidgets import (
    QGridLayout,
    QWidget,
    QLayout,
    QLabel,
    QComboBox,
    QCheckBox,
    QPushButton,
    QSpinBox,
)
import skimage.morphology
import scipy.ndimage as ndi

from ai_on_demand.widget_classes import SubWidget
from ai_on_demand.utils import format_tooltip


class MorphMasks(SubWidget):
    """
    1. Dilate, erode, open, close
    2. Fill holes
    3. Binarize
    4. Label

    For 1), have a spin box where 0 is "All", but otherwise is label that the operation will apply to.
    We should have the eyedropper update this box.
    """

    _name = "morph"

    def __init__(
        self,
        viewer: napari.Viewer,
        parent: Optional[QWidget] = None,
        layout: QLayout = QGridLayout,
        **kwargs,
    ):
        super().__init__(
            viewer=viewer,
            title="Morph",
            parent=parent,
            layout=layout,
            tooltip="""
Morph masks using various methods. Each function works on the currently selected Labels layer only, and modifies in-place.
""",
            **kwargs,
        )

    def create_box(self):
        self.morph_ops_box = self._make_groupbox("Morphological Ops")
        layout = self.morph_ops_box.layout()
        self.morph_ops_lbl = QLabel("Operation to apply:")
        self.morph_ops_dropdown = QComboBox()
        self.morph_ops_dropdown.addItems(
            [
                "Dilation",
                "Erosion",
                "Opening",
                "Closing",
            ]
        )
        # Box to select structuring element
        # NOTE: Ignore args, we just want its dicts
        self.aiod_filter = Filter({"footprint": "disk", "method": "mean"})
        self.morph_ops_footprint_lbl = QLabel("Structure:")
        self.morph_ops_footprint = QComboBox()
        self.morph_ops_footprint.addItems(self.aiod_filter.filters.keys())
        self.morph_ops_footprint.setToolTip(
            format_tooltip(self.aiod_filter.params["footprint"]["tooltip"])
        )
        self.morph_ops_radius = QSpinBox()
        self.morph_ops_radius.setSuffix(" px")
        self.morph_ops_radius.setRange(1, 1000)
        self.morph_ops_radius.setAlignment(qtpy.QtCore.Qt.AlignCenter)
        self.morph_ops_radius.setValue(5)
        self.morph_ops_radius.setToolTip(
            format_tooltip(self.aiod_filter.params["size"]["tooltip"])
        )

        # Add box for which label to apply to
        self.morph_ops_label_lbl = QLabel("Label to apply to:")
        self.morph_ops_label = QSpinBox()
        self.morph_ops_label.setRange(0, 10000)
        self.morph_ops_label.setAlignment(qtpy.QtCore.Qt.AlignCenter)
        self.morph_ops_label.setSpecialValueText("All")
        self.morph_ops_btn = QPushButton("Apply")
        self.morph_ops_btn.clicked.connect(self.morph_masks)
        self.morph_ops_cb = QCheckBox("In-place?")
        self.morph_ops_cb.setToolTip(
            format_tooltip(
                "If checked, the operation will be applied in-place to the currently selected Labels layer."
            )
        )
        self.morph_ops_cb.setChecked(True)
        layout.addWidget(self.morph_ops_lbl, 0, 0, 1, 2)
        layout.addWidget(self.morph_ops_dropdown, 0, 2, 1, 2)
        layout.addWidget(self.morph_ops_footprint_lbl, 1, 0, 1, 1)
        layout.addWidget(self.morph_ops_footprint, 1, 1, 1, 1)
        layout.addWidget(self.morph_ops_radius, 1, 2, 1, 2)
        layout.addWidget(self.morph_ops_label_lbl, 2, 0, 1, 2)
        layout.addWidget(self.morph_ops_label, 2, 2, 1, 2)
        layout.addWidget(self.morph_ops_btn, 3, 0, 1, 3)
        layout.addWidget(self.morph_ops_cb, 3, 3, 1, 1)
        self.inner_layout.addWidget(self.morph_ops_box, 0, 0)

        # Fill holes
        self.fill_hole_box = self._make_groupbox("Fill Holes")
        layout = self.fill_hole_box.layout()
        self.fill_hole_lbl = QLabel("Max hole size to fill:")
        self.fill_hole_size = QSpinBox()
        self.fill_hole_size.setSuffix(" px")
        self.fill_hole_size.setRange(0, 10000)
        self.fill_hole_size.setValue(64)
        self.fill_hole_size.setAlignment(qtpy.QtCore.Qt.AlignCenter)
        self.fill_hole_btn = QPushButton("Fill holes")
        self.fill_hole_btn.clicked.connect(self.fill_holes)
        self.fill_hole_cb = QCheckBox("In-place?")
        self.fill_hole_cb.setToolTip(
            format_tooltip(
                "If checked, the operation will be applied in-place to the currently selected Labels layer."
            )
        )
        self.fill_hole_cb.setChecked(True)
        layout.addWidget(self.fill_hole_lbl, 0, 0, 1, 2)
        layout.addWidget(self.fill_hole_size, 0, 2, 1, 2)
        layout.addWidget(self.fill_hole_btn, 1, 0, 1, 3)
        layout.addWidget(self.fill_hole_cb, 1, 3, 1, 1)
        self.inner_layout.addWidget(self.fill_hole_box, 1, 0)

        # Binarize
        self.binarize_box = self._make_groupbox("Binarize Masks")
        layout = self.binarize_box.layout()
        self.binarize_btn = QPushButton("Binarize")
        self.binarize_btn.clicked.connect(self.binarize_masks)
        self.binarize_cb = QCheckBox("In-place?")
        self.binarize_cb.setToolTip(
            format_tooltip(
                "If checked, the operation will be applied in-place to the currently selected Labels layer."
            )
        )
        self.binarize_cb.setChecked(True)
        layout.addWidget(self.binarize_btn, 0, 0, 1, 3)
        layout.addWidget(self.binarize_cb, 0, 3, 1, 1)
        self.inner_layout.addWidget(self.binarize_box, 2, 0)

        # Label
        self.label_box = self._make_groupbox("Label Masks")
        layout = self.label_box.layout()
        self.label_dilation = QCheckBox("Label across skipped slices?")
        self.label_dilation.setToolTip(format_tooltip("""
If checked, a dilation in the z-axis will first be applied, allowing for objects that may disappear
for a single frame/slice to still be labelled the same.
            """))
        # TODO: Give some control over footprint/structure used
        self.label_btn = QPushButton("Label")
        self.label_btn.clicked.connect(self.label_masks)
        self.label_cb = QCheckBox("In-place?")
        self.label_cb.setToolTip(
            format_tooltip(
                "If checked, the operation will be applied in-place to the currently selected Labels layer."
            )
        )
        self.label_cb.setChecked(True)
        layout.addWidget(self.label_dilation, 0, 0, 1, 4)
        layout.addWidget(self.label_btn, 1, 0, 1, 3)
        layout.addWidget(self.label_cb, 1, 3, 1, 1)
        self.inner_layout.addWidget(self.label_box, 3, 0)

    def morph_masks(self):
        # Get the currently selected layer
        layers = self.parent._get_selected_layers()
        if len(layers) > 1:
            show_error(
                "Please select only one Labels layer to apply the operation to."
            )
            return
        layer = layers[0]
        footprint_txt = self.morph_ops_footprint.currentText()
        # Check if correct footprint is selected for given layer
        # FIXME: They can apply 2D to 3D if they want!
        if layer.data.ndim == 2:
            if footprint_txt in ["cube", "ball"]:
                show_error(
                    f"Structure {footprint_txt} is not valid for 2D data. Please select a 2D structure (square/disk)."
                )
                return
        # Flag whether we need to loop over slices and apply 2D structure
        apply_2d_in_3d = (
            True
            if footprint_txt in ["square", "disk"] and layer.data.ndim == 3
            else False
        )
        # Get the operation to apply
        footprint_func = self.aiod_filter.filters[footprint_txt]
        footprint = footprint_func(self.morph_ops_radius.value())
        operation = self.morph_ops_dropdown.currentText().lower()
        label = self.morph_ops_label.value()
        if self.morph_ops_cb.isChecked():
            data = layer.data
        else:
            data = layer.data.copy()
        if label != 0:
            masks = data == label
            # Ensure the faster binary operation is used
            operation = "binary_" + operation
        else:
            masks = data
        # Apply 2D structure to each slice
        if apply_2d_in_3d:
            for i in range(masks.shape[0]):
                masks[i] = getattr(skimage.morphology, operation)(
                    masks[i], footprint
                )
        # Otherwise 2D to 2D or 3D to 3D in one go
        else:
            masks = getattr(skimage.morphology, operation)(
                masks, footprint, out=masks
            )
        # If we adjusted a single label, reinsert into other labels
        if label != 0:
            # Multiply by label as it'll be binary
            data[masks] = label
        else:
            data = masks
        # If in-place is not checked, create a new layer
        if not self.morph_ops_cb.isChecked():
            self.viewer.add_labels(
                data=data,
                name=f"{layer.name}_{operation}",
                metadata=layer.metadata,
            )
        else:
            # In-place, so we need to remove the old layer
            layer.data = data

    def fill_holes(self):
        # Get the currently selected layer
        layers = self.parent._get_selected_layers()
        if len(layers) > 1:
            show_error(
                "Please select only one Labels layer to apply the operation to."
            )
            return
        layer = layers[0]
        if self.fill_hole_cb.isChecked():
            data = layer.data
        else:
            data = layer.data.copy()
        # Apply the filling after binarization
        data = data > 0
        # NOTE: area_threshold will be replaced by max_size in the future
        data = skimage.morphology.remove_small_holes(
            data, area_threshold=int(self.fill_hole_size.value()), out=data
        )
        # Check if original data was binary
        if not self.parent._check_layers_binary([layer]):
            # If so, we need to relabel
            data = ndi.label(data)[0]
        # If in-place is not checked, create a new layer
        if not self.fill_hole_cb.isChecked():
            self.viewer.add_labels(
                data=data,
                name=f"{layer.name}_filled",
                metadata=layer.metadata,
            )
        else:
            # In-place, so we need to remove the old layer
            layer.data = data

    def binarize_masks(self):
        # Get the currently selected layer
        layers = self.parent._get_selected_layers()
        if len(layers) > 1:
            show_error(
                "Please select only one Labels layer to apply the operation to."
            )
            return
        layer = layers[0]
        if self.binarize_cb.isChecked():
            data = layer.data
        else:
            data = layer.data.copy()
        # Apply the binarization
        data = (data > 0).astype(np.uint8)
        # If in-place is not checked, create a new layer
        if not self.binarize_cb.isChecked():
            self.viewer.add_labels(
                data=data,
                name=f"{layer.name}_binarized",
                metadata=layer.metadata,
            )
        else:
            # In-place, so we need to remove the old layer
            layer.data = data

    def label_masks(self):
        # Get the currently selected layer
        layers = self.parent._get_selected_layers()
        if len(layers) > 1:
            show_error(
                "Please select only one Labels layer to apply the operation to."
            )
            return
        layer = layers[0]
        if self.label_cb.isChecked():
            data = layer.data
        else:
            data = layer.data.copy()

        if not isinstance(data, da.Array):
            orig = "numpy" if isinstance(data, np.ndarray) else "dask"
            data = da.from_array(data)

        def _simple_label(data):
            return dask_ndi.label(
                data, structure=ndi.generate_binary_structure(3, 1)
            )[0]

        def _dilate_label(data):
            # Create a structure that dilates to the next slice/frame only
            dilation_structure = np.zeros((3, 3, 3), dtype=bool)
            dilation_structure[1:, 1, 1] = True
            dilated = dask_morph.binary_dilation(
                data, structure=dilation_structure
            )
            # Now label the dilated data
            dilated = dask_ndi.label(
                dilated, structure=ndi.generate_binary_structure(3, 1)
            )[0]
            # Remove what was dilated by setting to 0 where original was 0
            dilated[data == 0] = 0
            return dilated

        # Apply the labeling
        if self.label_dilation.isChecked():
            data = _dilate_label(data)
        else:
            data = _simple_label(data)
        # If it's small data and was in memory anyway, Napari works faster with numpy so just convert back
        if orig == "numpy":
            data = data.compute()
        # If in-place is not checked, create a new layer
        if not self.label_cb.isChecked():
            self.viewer.add_labels(
                data=data,
                name=f"{layer.name}_labeled",
                metadata=layer.metadata,
            )
        else:
            # In-place, so we need to replace the underlying data and trigger a refresh
            layer.data = data

    def contour_fill(self):
        pass
