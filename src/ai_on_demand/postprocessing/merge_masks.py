from collections import defaultdict
from typing import Optional

import glasbey
import napari
from napari.utils.notifications import show_warning, show_error
from napari._qt.qt_resources import QColoredSVGIcon
import numpy as np
import qtpy.QtCore
from qtpy.QtWidgets import (
    QPushButton,
    QGridLayout,
    QWidget,
    QLayout,
    QLabel,
    QSpinBox,
    QDialog,
    QCheckBox,
)
from qtpy.QtGui import QIcon
import pandas as pd
import scipy.ndimage as ndi

from ai_on_demand.widget_classes import SubWidget
from ai_on_demand.utils import format_tooltip


class MergeMasks(SubWidget):
    # NOTE: This needs a Dask rewrite after all basic functionality is implemented

    _name = "merge"

    def __init__(
        self,
        viewer: napari.Viewer,
        parent: Optional[QWidget] = None,
        layout: QLayout = QGridLayout,
        **kwargs,
    ):
        super().__init__(
            viewer=viewer,
            title="Merge",
            parent=parent,
            layout=layout,
            tooltip="""
Merge masks using various methods. Note that all buttons will use whatever Labels layers are currently selected.
""",
            **kwargs,
        )

        self.visualize_dict = None

    def create_box(self):
        # Union merge
        self.union_box = self._make_groupbox("Mask Union")
        layout = self.union_box.layout()
        self.mask_union_btn = QPushButton("Combine Masks")
        self.mask_union_btn.clicked.connect(self.mask_union)
        self.mask_union_btn.setToolTip(
            format_tooltip("Combine (and re-label) the selected mask sets.")
        )
        layout.addWidget(self.mask_union_btn, 0, 0)
        self.inner_layout.addWidget(self.union_box, 0, 0)

        # Vote merge
        self.union_vote_box = self._make_groupbox("Mask Union (Vote)")
        layout = self.union_vote_box.layout()
        self.mask_union_vote_lbl = QLabel("Vote threshold:")
        self.mask_union_vote_num = QSpinBox()
        self.mask_union_vote_num.setAlignment(qtpy.QtCore.Qt.AlignCenter)
        self.mask_union_vote_num.setMinimum(1)
        self.mask_union_vote_num.setValue(2)
        self.mask_union_vote_btn = QPushButton("Combine Masks (Vote)")
        self.mask_union_vote_btn.clicked.connect(self.mask_vote)
        self.mask_union_vote_btn.setToolTip(
            format_tooltip(
                "Combine (and re-label) the selected mask sets using a voting mechanism."
            )
        )
        layout.addWidget(self.mask_union_vote_lbl, 0, 0, 1, 1)
        layout.addWidget(self.mask_union_vote_num, 0, 1, 1, 1)
        # TODO: Move button to separate layout to make it half width instead to be less imposing
        layout.addWidget(self.mask_union_vote_btn, 1, 0, 1, 2)
        self.inner_layout.addWidget(self.union_vote_box, 1, 0)

        self.visualize_box = self._make_groupbox("Visualize Overlaps")
        layout = self.visualize_box.layout()
        self.visualize_sets_btn = QPushButton("Visualize")
        self.visualize_sets_btn.clicked.connect(self.visualize_sets)
        self.visualize_sets_btn.setToolTip(
            format_tooltip(
                "Visualize the differences between the selected mask sets."
            )
        )
        self.visualize_icon_btn = QPushButton("")
        self.visualize_icon_btn.setIcon(
            QColoredSVGIcon.from_resources("help").colored(theme="dark")
        )
        self.visualize_icon_btn.setFixedSize(25, 25)
        self.visualize_icon_btn.setIconSize(
            self.visualize_icon_btn.size() * 0.65
        )
        self.visualize_icon_btn.setToolTip(
            format_tooltip("Check colour legend for layer sources.")
        )
        self.visualize_icon_btn.clicked.connect(self.show_visualize_legend)
        layout.addWidget(self.visualize_sets_btn, 0, 0, 1, 5)
        layout.addWidget(self.visualize_icon_btn, 0, 5, 1, 1)
        self.inner_layout.addWidget(self.visualize_box, 2, 0)

        # Specific label merge
        self.merge_lbls_box = self._make_groupbox("Merge Labels")
        layout = self.merge_lbls_box.layout()
        self.mask_merge_lbl_1 = QLabel("Base label:")
        self.mask_merge_input_1 = QSpinBox()
        self.mask_merge_input_1.setRange(1, 10000)
        self.mask_merge_input_1.setValue(1)
        self.mask_merge_input_1.setAlignment(qtpy.QtCore.Qt.AlignCenter)
        self.mask_merge_lbl_2 = QLabel("Label to merge:")
        self.mask_merge_input_2 = QSpinBox()
        self.mask_merge_input_2.setRange(1, 10000)
        self.mask_merge_input_2.setValue(2)
        self.mask_merge_input_2.setAlignment(qtpy.QtCore.Qt.AlignCenter)
        self.mask_merge_btn = QPushButton("Merge Labels")
        self.mask_merge_btn.clicked.connect(self.merge_specific)
        self.mask_merge_btn.setToolTip(
            format_tooltip(
                "Merge two specific labels from the selected mask sets. The second label will be merged into the first."
            )
        )
        layout.addWidget(self.mask_merge_lbl_1, 0, 0, 1, 1)
        layout.addWidget(self.mask_merge_input_1, 0, 1, 1, 1)
        layout.addWidget(self.mask_merge_lbl_2, 1, 0, 1, 1)
        layout.addWidget(self.mask_merge_input_2, 1, 1, 1, 1)
        layout.addWidget(self.mask_merge_btn, 2, 0, 1, 2)
        self.inner_layout.addWidget(self.merge_lbls_box, 3, 0)

        # Overlap merge
        self.merge_overlap_box = self._make_groupbox("Merge 2D to 3D")
        layout = self.merge_overlap_box.layout()
        # self.mask_merge_overlap_lbl = QLabel("Overlap (IoU):")
        # self.mask_merge_overlap_input = QSpinBox()
        # self.mask_merge_overlap_input.setAlignment(qtpy.QtCore.Qt.AlignCenter)
        self.merge_overlap_cb = QCheckBox("Overwrite?")
        self.merge_overlap_cb.setToolTip(
            format_tooltip(
                "If checked, will overwrite the selected layer with the merged masks. Otherwise will create a new layer."
            )
        )
        self.merge_overlap_cb.setChecked(True)
        self.mask_merge_overlap_btn = QPushButton("Merge All (Overlap)")
        self.mask_merge_overlap_btn.clicked.connect(self.merge_overlap)
        self.mask_merge_overlap_btn.setToolTip(
            format_tooltip(
                "Merge labels from the selected mask sets based on overlap threshold (in 3D)."
            )
        )
        # layout.addWidget(self.mask_merge_overlap_lbl, 0, 0, 1, 1)
        # layout.addWidget(self.mask_merge_overlap_input, 0, 1, 1, 1)
        layout.addWidget(self.mask_merge_overlap_btn, 0, 0, 1, 3)
        layout.addWidget(self.merge_overlap_cb, 0, 3, 1, 1)
        self.inner_layout.addWidget(self.merge_overlap_box, 4, 0)

    def mask_union(self):
        layers = self.parent._get_selected_layers()
        if len(layers) == 1:
            show_error(
                "Only one label layer selected. Please select at least two Labels layers."
            )
            return
        # Get the union of the masks
        # NOTE: Technically special case of mask_vote, but logical_or should be faster so maybe worth separating
        union = self.parent._binarize_mask(layers[0])
        for layer in layers[1:]:
            union = np.logical_or(union, self.parent._binarize_mask(layer))
        # Check if the original layers were binary or not
        # Re-label unless both layers are binary
        if not self.parent._check_layers_binary(layers):
            # Not, so we need to re-label the union
            # FIXME: Use dask-image instead
            union = ndi.label(union)[0]
        # Create a new layer with the union
        self.viewer.add_labels(
            union,
            name="Mask Union",
        )

    def mask_vote(self):
        layers = self.parent._get_selected_layers()
        if len(layers) == 1:
            show_error(
                "Only one label layer selected. Please select at least two Labels layers."
            )
            return
        # Get the vote threshold
        threshold = self.mask_union_vote_num.value()
        if threshold < 1:
            show_error("Vote threshold must be at least 1.")
            return
        elif threshold > len(layers):
            show_warning(
                f"Vote threshold larger than number of selected layers. Setting to maximum ({len(layers)})."
            )
            threshold = len(layers)
        # Get the vote count
        # NOTE: Threshold=1 is equivalent to a union, maybe link
        vote = np.sum(
            [
                self.parent._binarize_mask(layer).astype(np.uint8)
                for layer in layers
            ],
            axis=0,
        )
        # Threshold the vote
        vote = vote >= threshold
        # Check if the original layers were binary or not
        if not self.parent._check_layers_binary(layers):
            vote = ndi.label(vote)[0]
        # Create a new layer with the vote
        self.viewer.add_labels(
            vote,
            name=f"Mask Vote (Threshold: {threshold})",
        )

    def merge_specific(self):
        # TODO: Probably move this to the Points layer approach that empanada uses
        # Get labels of all points in layer, use minimum as base, then merge all into that
        # With option to apply in 3d or not
        layers = self.parent._get_selected_layers()
        if len(layers) > 1:
            show_error("You can only merge two labels on one layer at a time.")
            return
        # Get the labels to merge
        base_label = int(self.mask_merge_input_1.value())
        merge_label = int(self.mask_merge_input_2.value())
        # Check if the labels exist
        all_labels = np.unique(layers[0].data)
        if base_label not in all_labels:
            show_error(f"Base label {base_label} not found in the layer.")
            return
        if merge_label not in all_labels:
            show_error(f"Merge label {merge_label} not found in the layer.")
            return
        # Merge the labels
        arr = layers[0].data
        arr[arr == merge_label] = base_label
        layers[0].data = arr

    def visualize_sets(self):
        # Grab selected layers
        selected_layers = self.parent._get_selected_layers()
        num_layers = len(selected_layers)
        if len(selected_layers) >= 5:
            # Create a warning message
            show_warning(
                "Too many layers selected, visualization will be convoluted!"
            )
        elif len(selected_layers) == 0:
            show_error(
                "No label layers selected. Please select at least two Labels layers."
            )
            return
        elif len(selected_layers) == 1:
            show_error(
                "Only one label layer selected. Please select at least two Labels layers."
            )
            return
        # Remove existing mask overlaps layer if it exists
        # TODO: Parameterize name?
        if "Mask Overlaps" in self.viewer.layers:
            self.viewer.layers.remove("Mask Overlaps")
        # Calc number of colours needed (0 is background)
        num_colours = 2**num_layers - 1
        # Now get the multipliers for each layer to ensure unique values
        multipliers = [1 << i for i in range(num_layers)]
        layer_values = {
            i: layer for i, layer in zip(multipliers, selected_layers)
        }
        # Binarize each mask, multiply by its relevant power of 2, and sum to get combined mask
        res = np.sum(
            [
                self.parent._binarize_mask(layer).astype(np.uint8) * multi
                for multi, layer in layer_values.items()
            ],
            axis=0,
        )
        # Create a colormap for the merged masks
        # Arg tips from https://napari.org/dev/gallery/glasbey-colormap.html
        # NOTE: Their documentation is wrong, so we need to adjust the output
        cmap = np.asarray(
            glasbey.create_palette(
                palette_size=num_colours,
                lightness_bounds=(20, 60),
                chroma_bounds=(40, 50),
                colorblind_safe=True,
                as_hex=False,
            ),
        )
        # Reset visualization dict now we are proceeding
        self.visualize_dict = {}
        # Now build the features table to construct every combination
        features = defaultdict(list)
        # Add the background
        features["label"].append(0)
        features["source"].append("Background")
        # NOTE: Easier to loop over all numbers rather than find which ones we have (esp for larger arrays)
        for label in range(1, num_colours + 1):
            features["label"].append(label)
            # Extract each layer that contributes to this value
            contrib_layers = [i for i in range(num_layers) if label & (1 << i)]
            # Construct the source string
            if len(contrib_layers) == 1:
                sources = selected_layers[contrib_layers[0]].name
            else:
                sources = " + ".join(
                    [selected_layers[i].name for i in contrib_layers]
                )
            features["source"].append(sources)
            self.visualize_dict[label] = {
                "layers": sources,
                "colour": cmap[label - 1],
            }
        # Convert features to a DataFrame for napari
        features = pd.DataFrame(features)
        # Add alpha channel to colourmap
        cmap = np.column_stack((cmap, np.ones((num_colours,))))
        # Convert to a label colour mapping
        cmap = {k: v for k, v in zip(range(1, num_colours + 1), cmap)}
        # Add values for 0 (background) and None (missing)
        cmap[0] = np.array([0, 0, 0, 0])
        cmap[None] = np.array([0, 0, 0, 1])
        # Create a new layer with the merged masks
        self.viewer.add_labels(
            res,
            name="Mask Overlaps",
            colormap=cmap,
            features=features,
            blending="opaque",  # Avoids blending issues
        )

    def show_visualize_legend(self):
        self.legend_window = VisualizeLegend(
            self, vis_dict=self.visualize_dict
        )
        self.legend_window.show()

    def merge_overlap(self):
        # Grab selected layers
        selected_layers = self.parent._get_selected_layers()
        num_layers = len(selected_layers)
        if num_layers >= 2:
            show_error(
                "Please select only one Labels layer to merge objects across Z."
            )
            return
        # Now label the selected layer
        # NOTE: scipy creates the structure we want by default
        res = ndi.label(
            selected_layers[0].data,
        )[0]
        if self.merge_overlap_cb.isChecked():
            # Overwrite the original layer
            selected_layers[0].data = res
        else:
            self.viewer.add_labels(
                res,
                name=f"{selected_layers[0].name}_merge-overlap",
            )


class VisualizeLegend(QDialog):
    def __init__(self, parent=None, vis_dict: Optional[dict] = None):
        super().__init__(parent)

        self.setWindowTitle("Mask Overlap Legend")
        self.setGeometry(100, 100, 300, 200)
        self.layout = QGridLayout()
        self.setLayout(self.layout)

        if vis_dict is None or len(vis_dict) == 0:
            self.layout.addWidget(
                QLabel("No mask overlaps to visualize!"), 0, 0, 1, 2
            )
            return

        for idx, (label, d) in enumerate(vis_dict.items()):
            # Mask label
            mask_lbl = QLabel(f"{label}:")
            mask_lbl.setAlignment(
                qtpy.QtCore.Qt.AlignVCenter | qtpy.QtCore.Qt.AlignRight
            )
            mask_lbl.setWordWrap(True)
            r, g, b = (d["colour"] * 255).astype(np.uint8)
            # Create a box with our colour
            colour_box = QWidget()
            colour_box.setStyleSheet(f"background-color: rgb({r}, {g}, {b});")
            colour_box.setFixedSize(20, 20)
            # Create a label with the text
            # For nicer presentation, add a newline after each +
            sources = d["layers"].replace(" + ", " +\n")
            source_text = QLabel(f"{sources}")
            source_text.setAlignment(
                qtpy.QtCore.Qt.AlignVCenter | qtpy.QtCore.Qt.AlignLeft
            )
            source_text.setWordWrap(True)
            # Add the box and label to the layout
            self.layout.addWidget(mask_lbl, idx, 0, 1, 1)
            self.layout.addWidget(colour_box, idx, 1, 1, 1)
            self.layout.addWidget(source_text, idx, 2, 1, 6)
