from pathlib import Path
from typing import Optional

import aiod_utils.rle as aiod_rle
import napari
from napari.utils.notifications import show_info
import numpy as np
import qtpy.QtCore
from qtpy.QtWidgets import (
    QWidget,
    QLayout,
    QGridLayout,
    QPushButton,
    QComboBox,
    QCheckBox,
    QFileDialog,
    QLabel,
)
import skimage.io

from ai_on_demand.utils import format_tooltip
from ai_on_demand.widget_classes import SubWidget


class ExportWidget(SubWidget):
    _name = "export"

    def __init__(
        self,
        viewer: napari.Viewer,
        parent: Optional[QWidget] = None,
        layout: QLayout = QGridLayout,
        **kwargs,
    ):
        super().__init__(
            viewer=viewer,
            title="Mask Export",
            parent=parent,
            layout=layout,
            tooltip="Export masks to other formats.",
            **kwargs,
        )

        # Initialise the selected mask layers list
        self.selected_mask_layers = []
        # Connect viewer to callbacks on events
        self.viewer.layers.selection.events.changed.connect(
            self.on_select_change
        )

    def create_box(self):
        self.export_masks_btn = QPushButton("Export all masks")
        self.export_masks_btn.clicked.connect(self.on_click_export)
        self.export_masks_btn.setToolTip(
            format_tooltip(
                "Export the output segmentation masks to a directory. Exports all masks (Labels layers) by default, or only the selected masks (if any)."
            )
        )
        self.export_masks_btn.setEnabled(True)
        self.inner_layout.addWidget(self.export_masks_btn, 1, 0, 1, 3)

        export_label = QLabel("Export format:")
        self.inner_layout.addWidget(export_label, 2, 0, 1, 1)

        self.export_format_dropdown = QComboBox()
        export_options = [
            (
                "RLE (.rle)",
                "Compressed RLE format. Binarise for most efficient storage and loading, though not recommended for SAM.",
            ),
            (
                "NumPy (.npy)",
                "NumPy format. Easiest to use elsewhere in Python without AIoD.",
            ),
            (
                "TIFF (.tiff)",
                "TIFF format. Most generic format for other imaging software.",
            ),
        ]
        for i, (fmt, desc) in enumerate(export_options):
            self.export_format_dropdown.addItem(fmt)
            self.export_format_dropdown.setItemData(
                i, desc, qtpy.QtCore.Qt.ToolTipRole
            )
        self.export_format_dropdown.setToolTip(
            format_tooltip(
                "Select the format to export the masks in. Hover over each item for a description."
            )
        )
        self.inner_layout.addWidget(self.export_format_dropdown, 2, 1, 1, 2)

        self.export_binary_check = QCheckBox("Binarise masks?")
        self.export_binary_check.setToolTip(
            format_tooltip(
                "Binarise the masks before exporting (i.e. black background, white masks)."
            )
        )
        self.inner_layout.addWidget(self.export_binary_check, 3, 0, 1, 3)

    def _binarise_mask(self, mask_layer):
        """
        Binarises the given mask layer.
        """
        return (mask_layer.data).astype(bool).astype(np.uint8) * 255

    def on_select_change(self, event):
        layers_selected = event.source
        # If nothing selected, reset the mask layers
        if len(layers_selected) == 0:
            # Filter mask layers to ensure they are Labels layers
            self.selected_mask_layers = self.get_mask_layers()
            # Reset text on export button
            self.export_masks_btn.setText("Export all masks")
            # Reset tile size label if nothing selected
            if "nxf" in self.parent.subwidgets:
                self.parent.subwidgets["nxf"].update_tile_size(
                    val=None, clear_label=True
                )
        else:
            # Update the tile size label based on the selected layers
            if "nxf" in self.parent.subwidgets:
                self.parent.subwidgets["nxf"].update_tile_size(
                    val=None, clear_label=False
                )
            # Filter mask layers to ensure they are from AIoD outputs and not external
            self.selected_mask_layers = self.get_mask_layers(layers_selected)
            num_selected = len(self.selected_mask_layers)
            # In case non-Labels layers are selected, reset
            if num_selected == 0:
                self.selected_mask_layers = self.get_mask_layers()
                self.export_masks_btn.setText("Export all masks")
            else:
                self.export_masks_btn.setText(
                    f"Export {num_selected} mask{'s' if num_selected > 1 else ''}"
                )
        return

    def get_mask_layers(
        self, layer_list: Optional[napari.components.LayerList] = None
    ) -> list[napari.layers.Labels]:
        """
        Return all the mask layers in the viewer.

        This is used to get the masks to evaluate against.
        """
        # If no layer list given, use all layers in the Napari viewer
        if layer_list is None:
            layer_list = self.viewer.layers
        # Select only the Labels layers
        valid_mask_layers = [
            layer
            for layer in layer_list
            if isinstance(layer, napari.layers.Labels)
        ]
        return valid_mask_layers

    def on_click_export(self):
        """
        Callback for when the export button is clicked. Opens a dialog to select a directory to save the masks to.
        """
        # Extract the data from each of the selected layers, and save the result in the given folder
        if self.selected_mask_layers:
            export_dir = QFileDialog.getExistingDirectory(
                self, caption="Select directory to save masks", directory=None
            )
            if not export_dir or export_dir is None:
                show_info("No directory selected!")
                return
            count = 0
            for mask_layer in self.selected_mask_layers:
                # Get the name of the mask layer as root for the filename
                fname = f"{mask_layer.name}"
                # Check if we are binarising
                if self.export_binary_check.isChecked():
                    mask_data = self._binarise_mask(mask_layer)
                    fname += "_binarised"
                else:
                    mask_data = mask_layer.data
                # Get the extension & add to fname
                ext = (
                    self.export_format_dropdown.currentText()
                    .split(".")[-1]
                    .replace(")", "")
                )
                fname += f".{ext}"
                fpath = Path(export_dir) / fname
                if ext == "npy":
                    np.save(
                        fpath,
                        mask_data,
                    )
                elif ext == "tiff":
                    skimage.io.imsave(
                        fpath,
                        mask_data,
                        plugin="tifffile",
                    )
                elif ext == "rle":
                    encoded_mask = aiod_rle.encode(
                        mask_data,
                        mask_type=(
                            "binary"
                            if self.export_binary_check.isChecked()
                            else "instance"
                        ),
                        metadata=mask_layer.metadata,
                    )
                    aiod_rle.save_encoding(fpath=fpath, rle=encoded_mask)
                count += 1
            show_info(f"Exported {count} mask files to {export_dir}!")
        else:
            show_info("No mask layers found!")
