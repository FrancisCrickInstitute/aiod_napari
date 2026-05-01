import operator
from typing import Optional

from app_model.types import KeyCode
import napari
from napari.utils.notifications import show_error
from napari.layers import Labels
import numpy as np
import qtpy.QtCore
from qtpy.QtWidgets import (
    QGridLayout,
    QWidget,
    QLayout,
    QLabel,
    QComboBox,
    QLineEdit,
    QCheckBox,
    QPushButton,
    QSpinBox,
)
from skimage.measure._regionprops import COL_DTYPES
from skimage.measure import regionprops

from ai_on_demand.widget_classes import SubWidget
from ai_on_demand.utils import format_tooltip


class FilterMasks(SubWidget):
    """
    1. Filter by anything in regionprops
    2. Filter by label (i.e. delete)
    3. Filter boundary labels
    """

    _name = "filter"
    operators = {
        "==": operator.eq,
        "!=": operator.ne,
        "<": operator.lt,
        "<=": operator.le,
        ">": operator.gt,
        ">=": operator.ge,
    }

    def __init__(
        self,
        viewer: napari.Viewer,
        parent: Optional[QWidget] = None,
        layout: QLayout = QGridLayout,
        **kwargs,
    ):
        super().__init__(
            viewer=viewer,
            title="Filter",
            parent=parent,
            layout=layout,
            tooltip="""
Filter masks using various methods. Each function works on the currently selected Labels layer only, and modifies in-place.
""",
            **kwargs,
        )

        self.viewer.layers.events.inserted.connect(self.add_layer)

        # Check for any existing labels layers & connect to our event
        if self.viewer.layers:
            for layer in self.viewer.layers:
                if isinstance(layer, Labels):
                    layer.events.selected_label.connect(
                        self._update_selected_label
                    )
                    layer.bind_key(KeyCode.Delete, self.filter_label)

    def add_layer(self, event):
        if isinstance(event.value, Labels):
            # If a labels layer is added, connected it to our selected label event
            event.value.events.selected_label.connect(
                self._update_selected_label
            )
            # Add a shortcut for deleting the currently selected label
            event.value.bind_key(KeyCode.Delete, self.filter_label)

    def _update_selected_label(self, event):
        # NOTE: The vars within the event do not seem what we want, so grab directly
        self.filter_label_entry.setValue(event._sources[0].selected_label)

    def create_box(self):
        # TODO: Also connect this to delete shortcut for user speed
        self.filter_label_box = self._make_groupbox("Filter Label")
        layout = self.filter_label_box.layout()
        self.filter_label_lbl = QLabel("Label to remove:")
        self.filter_label_entry = QSpinBox()
        self.filter_label_entry.setRange(0, 10000)
        self.filter_label_entry.setAlignment(qtpy.QtCore.Qt.AlignCenter)
        self.filter_label_entry.setToolTip(
            format_tooltip(
                "Set the label to remove. This will remove that label from the currently selected Labels layer."
            )
        )
        self.filter_label_btn = QPushButton("Remove label")
        self.filter_label_btn.clicked.connect(self.filter_label)
        self.filter_label_cb = QCheckBox("In-place?")
        self.filter_label_cb.setToolTip(
            format_tooltip(
                "If checked, will remove the label from the currently selected Labels layer. Otherwise, will add a new Labels layer with the label removed."
            )
        )
        self.filter_label_cb.setChecked(True)
        layout.addWidget(self.filter_label_lbl, 0, 0, 1, 2)
        layout.addWidget(self.filter_label_entry, 0, 2, 1, 2)
        layout.addWidget(self.filter_label_btn, 1, 0, 1, 3)
        layout.addWidget(self.filter_label_cb, 1, 3, 1, 1)
        self.inner_layout.addWidget(self.filter_label_box, 0, 0)

        self.regionprops_box = self._make_groupbox("Regionprops Filters")
        layout = self.regionprops_box.layout()
        self.regionprops_prop_label = QLabel("Property:")
        self.regionprops_dropdown = QComboBox()
        # Best is to manually set which properties are valid for filtering
        self.regionprops_dropdown.addItems(
            [
                "area",
                "area_bbox",
                "area_convex",
                "area_filled",
                "axis_major_length",
                "axis_minor_length",
                "eccentricity",
                "equivalent_diameter_area",
                "euler_number",
                "extent",
                "num_pixels",
                "orientation",
                "perimeter",
                "perimeter_crofton",
                "solidity",
            ]
        )
        self.regionprops_dropdown.setCurrentIndex(0)
        self.regionprops_dropdown.setToolTip(
            format_tooltip(
                "Select a region property to filter by. This will filter the currently selected Labels layer."
            )
        )
        self.regionprops_value_label = QLabel("Threshold:")
        self.regionprops_value = QLineEdit()
        self.regionprops_value.setPlaceholderText("0")
        self.regionprops_value.setToolTip(
            format_tooltip(
                "Set the threshold for the selected region property. This will filter the currently selected Labels layer."
            )
        )
        self.regionprops_ops = QComboBox()
        self.regionprops_ops.addItems(list(self.operators.keys()))
        self.regionprops_ops.setCurrentIndex(5)
        self.regionprops_ops.setToolTip(
            format_tooltip(
                "Select the operator to use with the given value for filtering."
            )
        )
        self.regionprops_btn = QPushButton("Filter labels")
        self.regionprops_btn.clicked.connect(self.filter_regionprops)
        self.regionprops_cb = QCheckBox("In-place?")
        layout.addWidget(self.regionprops_prop_label, 0, 0, 1, 2)
        layout.addWidget(self.regionprops_dropdown, 0, 2, 1, 2)
        layout.addWidget(self.regionprops_value_label, 1, 0, 1, 2)
        layout.addWidget(self.regionprops_ops, 1, 2, 1, 1)
        layout.addWidget(self.regionprops_value, 1, 3, 1, 1)
        layout.addWidget(self.regionprops_btn, 2, 0, 1, 3)
        layout.addWidget(self.regionprops_cb, 2, 3, 1, 1)
        # TODO: Would be nice to have a pop-out table to show values of these properties for each label
        self.inner_layout.addWidget(self.regionprops_box, 1, 0)

    def filter_label(self, layer: Optional[Labels] = None):
        layers = self.parent._get_selected_layers()
        if len(layers) > 1:
            show_error("Only one label layer can be selected for filtering!")
            return
        # Get the selected labels layer
        if self.filter_label_cb.isChecked():
            labels = layers[0].data
        else:
            labels = layers[0].data.copy()
        label_to_remove = self.filter_label_entry.value()
        if label_to_remove == 0:
            show_error("Label to remove must be greater than 0!")
            return
        # Remove the label from the labels layer
        labels[labels == label_to_remove] = 0
        if self.filter_label_cb.isChecked():
            # In-place, so we need to remove the old layer
            layers[0].data = labels
        else:
            self.viewer.add_labels(labels, name=f"{layers[0].name}_filtered")

    def filter_boundary(self):
        layers = self.parent._get_selected_layers()
        data = layers[0].data

        """
        Mask arg makes buffer_size get ignored.
        So if we want to apply to 3D, and take into account tiling, we need to create the mask manually.
        Just need to decide how to implement interface for this info, do we just copy the tiling options from before?
        The ideal would be taking it directly from what created that labels layer...but that's a pain. Could use metadata, and just default to auto if not available. Maybe that's v2...
        """

        if data.ndim == 2:
            mask = None
        # Ensure that the first and last frame are not considered boundary
        # We just care about XY boundaries
        elif data.ndim == 3:
            mask = np.ones_like(data)

    def filter_regionprops(self):
        layers = self.parent._get_selected_layers()
        if len(layers) > 1:
            show_error("Only one label layer can be selected for filtering!")
            return
        # Get the selected labels layer
        labels = layers[0].data
        props = regionprops(labels)

        op = self.operators[self.regionprops_ops.currentText()]

        selected_prop = self.regionprops_dropdown.currentText()
        threshold = COL_DTYPES[selected_prop](self.regionprops_value.text())
        matching_labels = [
            prop.label
            for prop in props
            if op(getattr(prop, selected_prop), threshold)
        ]
        if self.regionprops_cb.isChecked():
            # In-place, so we need to update and reinsert the layer to trigger update events
            labels[~np.isin(labels, matching_labels)] = 0
            layers[0].data = labels
        else:
            # Not in-place, so ensure that a copy is modified
            labels = labels.copy()
            labels[~np.isin(labels, matching_labels)] = 0
            # Add the new layer
            self.viewer.add_labels(labels, name=f"{layers[0].name}_filtered")
