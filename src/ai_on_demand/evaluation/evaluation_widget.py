from collections import defaultdict
from functools import partial
from typing import Optional

import napari
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info
from napari.layers import Image, Labels
import pandas as pd
from qtpy.QtWidgets import (
    QWidget,
    QLayout,
    QGridLayout,
    QComboBox,
    QLabel,
    QCheckBox,
    QPushButton,
    QTextBrowser,
    QFileDialog,
)
from qtpy import QtGui
import skimage.measure

from ai_on_demand.widget_classes import MainWidget, SubWidget
from ai_on_demand.utils import format_tooltip
import ai_on_demand.evaluation.metrics as aiod_metrics


class Evaluation(MainWidget):
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__(
            napari_viewer=napari_viewer,
            title="Evaluation",
            tooltip="""
Calculate various evaluation metrics (with or without ground truth) on selected masks
""",
        )

        # Register any subwidgets here
        self.register_widget(
            EvalWidget(viewer=self.viewer, parent=self, expanded=True)
        )

    def get_run_hash(self):
        # NOTE: Currently no need to hash anything for evaluation, as Nextflow is not used
        pass


class EvalWidget(SubWidget):
    _name = "eval"

    def __init__(
        self,
        viewer: napari.Viewer,
        parent: Optional[QWidget] = None,
        layout: QLayout = QGridLayout,
        **kwargs,
    ):
        super().__init__(
            viewer=viewer,
            title="Evaluation",
            parent=parent,
            layout=layout,
            tooltip=parent.tooltip,
            **kwargs,
        )
        # Unsure if changed means inserted/removed, or includes moved which we don't care about
        self.viewer.layers.events.inserted.connect(self.add_layer)
        self.viewer.layers.events.removed.connect(self.remove_layer)

    def create_box(self):
        # Mask layer selection
        self.mask_layer_label = QLabel("Mask layer:")
        # Get all labels layers that were already present when the widget was created
        init_mask_layers = [
            layer.name
            for layer in self.viewer.layers
            if isinstance(layer, Labels)
        ]
        self.mask_layer_dropdown = QComboBox()
        self.mask_layer_dropdown.setToolTip(
            format_tooltip("Select Labels layer containing masks to evaluate")
        )
        self.mask_layer_dropdown.addItems(init_mask_layers)
        self.inner_layout.addWidget(self.mask_layer_label, 0, 0, 1, 4)
        self.inner_layout.addWidget(self.mask_layer_dropdown, 1, 0, 1, 4)
        # Ground truth layer selection
        self.gt_layer_label = QLabel("True/other mask layer:")
        self.gt_layer_dropdown = QComboBox()
        self.gt_layer_dropdown.setToolTip(
            format_tooltip(
                """
    Select Labels layer containing ground truth (or another set of masks) to compare against
    """
            )
        )
        self.gt_layer_dropdown.addItems(init_mask_layers)
        self.gt_selected = QCheckBox("Ground truth selected?")
        self.gt_selected.setToolTip(
            format_tooltip(
                "Specify whether the other mask layer contains ground truth"
            )
        )
        # Define behaviour on checking/unchecking box
        self.gt_selected.stateChanged.connect(self.on_gt_select)
        self.gt_selected.setChecked(False)
        self.inner_layout.addWidget(self.gt_layer_label, 2, 0, 1, 2)
        self.inner_layout.addWidget(self.gt_selected, 2, 2, 1, 2)
        self.inner_layout.addWidget(self.gt_layer_dropdown, 3, 0, 1, 4)
        # Image selection
        # Get all image layers that were already present when the widget was created
        init_img_layers = [
            layer.name
            for layer in self.viewer.layers
            if isinstance(layer, Image)
        ]
        self.image_layer_label = QLabel("Image layer:")
        self.image_layer_dropdown = QComboBox()
        self.image_layer_dropdown.setToolTip(
            format_tooltip("Select image layer to include in analysis")
        )
        self.image_layer_dropdown.addItems(init_img_layers)
        self.inner_layout.addWidget(self.image_layer_label, 4, 0, 1, 4)
        self.inner_layout.addWidget(self.image_layer_dropdown, 5, 0, 1, 4)
        # Metric selection
        row = self.define_metrics(start_row=self.inner_layout.rowCount())
        # Calculate button
        self.calculate_btn = QPushButton("Calculate!")
        self.calculate_btn.clicked.connect(self.calculate_metrics)
        self.calculate_btn.setEnabled(True)
        self.inner_layout.addWidget(self.calculate_btn, row, 0, 1, 4)
        # Output box
        self.output_box = QTextBrowser()
        font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        # NOTE: Can reduce to 12 if more metrics added, but it auto-scrolls so not a big deal
        font.setPointSize(14)
        self.output_box.setFont(font)
        self.output_box.setWordWrapMode(QtGui.QTextOption.NoWrap)
        self.inner_layout.addWidget(self.output_box, row + 1, 0, 1, 4)
        # Export button
        self.export_btn = QPushButton("Export results")
        self.export_btn.clicked.connect(self.export_results)
        self.export_btn.setToolTip(format_tooltip("Export results to CSV"))
        self.inner_layout.addWidget(self.export_btn, row + 2, 0, 1, 2)
        # Append to existing results button
        self.append_btn = QPushButton("Append to existing results")
        self.append_btn.clicked.connect(self.append_results)
        self.append_btn.setToolTip(
            format_tooltip("Export and append results to previous export")
        )
        self.inner_layout.addWidget(self.append_btn, row + 2, 2, 1, 2)

    def define_metrics(self, start_row: int = 3, width: int = 4):
        """
        Define all metrics to be calculated, and their parameters
        """
        self.base_metrics = {
            "Dice": aiod_metrics.dice,
            "IoU": aiod_metrics.iou,
            "Hausdorff": aiod_metrics.hausdorff_dist,
            "Hausdorff (modified)": partial(
                aiod_metrics.hausdorff_dist, method="modified"
            ),
        }

        self.inner_layout.addWidget(
            QLabel("Unsupervised metrics:"), start_row, 0, 1, width
        )
        row = start_row + 1
        self.base_metric_widgets = {}

        for i, (name, func) in enumerate(self.base_metrics.items()):
            # Create a checkbox for each metric
            checkbox = QCheckBox(name)
            checkbox.setChecked(True)
            checkbox.setToolTip(format_tooltip(func.__doc__))
            self.base_metric_widgets[name] = checkbox
            self.inner_layout.addWidget(
                checkbox, row + (i // width), i % width, 1, 1
            )

        # Update the row and increment for GT metrics
        row += (i // width) + 1

        self.inner_layout.addWidget(
            QLabel("Supervised metrics:"), row, 0, 1, width
        )
        row += 1

        # Define additional metrics if ground truth available
        self.gt_metrics = {
            "Precision": aiod_metrics.precision,
            "Recall": aiod_metrics.recall,
        }
        self.gt_metric_widgets = {}
        for i, (name, func) in enumerate(self.gt_metrics.items()):
            # Create a checkbox for each metric
            checkbox = QCheckBox(name)
            if self.gt_selected.isChecked():
                checkbox.setEnabled(True)
                checkbox.setChecked(True)
            else:
                checkbox.setEnabled(False)
                checkbox.setChecked(False)
            checkbox.setToolTip(format_tooltip(func.__doc__))
            self.gt_metric_widgets[name] = checkbox
            self.inner_layout.addWidget(
                checkbox, row + (i // width), i % width, 1, 1
            )

        # TODO:
        # - Use table version?
        # - Any other metrics to add?
        # - Do we really need the image layer and metrics?
        img_metrics = {
            "Region Props": skimage.measure.regionprops,
        }

        return row + (i // width) + 1

    def on_gt_select(self):
        if self.gt_selected.isChecked():
            for checkbox in self.gt_metric_widgets.values():
                checkbox.setEnabled(True)
                checkbox.setChecked(True)
        else:
            for checkbox in self.gt_metric_widgets.values():
                checkbox.setEnabled(False)
                checkbox.setChecked(False)

    def add_layer(self, event):
        if isinstance(event.value, Labels):
            # Update ground truth and mask layer lists
            self.gt_layer_dropdown.insertItem(0, event.value.name)
            self.mask_layer_dropdown.insertItem(0, event.value.name)
        elif isinstance(event.value, Image):
            # Update image list
            self.image_layer_dropdown.insertItem(0, event.value.name)

    def remove_layer(self, event):
        if isinstance(event.value, Labels):
            # Update ground truth and mask layer lists
            self.gt_layer_dropdown.removeItem(
                self.gt_layer_dropdown.findText(event.value.name)
            )
            self.mask_layer_dropdown.removeItem(
                self.mask_layer_dropdown.findText(event.value.name)
            )
        elif isinstance(event.value, Image):
            # Update image list
            self.image_layer_dropdown.removeItem(
                self.image_layer_dropdown.findText(event.value.name)
            )

    def calculate_metrics(self):
        # Check that we have layers selected!
        # Get the mask layer
        self.mask_layer_name = self.mask_layer_dropdown.currentText()
        if self.mask_layer_name == "":
            show_info("No mask layer selected!")
            return
        else:
            mask_layer = self.viewer.layers[self.mask_layer_name]
        # Get the other/ground truth layer
        self.other_layer_name = self.gt_layer_dropdown.currentText()
        # NOTE: This is probably impossible to reach due to auto-population of dropdown
        if self.other_layer_name == "":
            show_info("No other layer selected!")
            return
        else:
            other_layer = self.viewer.layers[self.other_layer_name]
        # Disable the calculate button
        self.calculate_btn.setText("Calculating...")
        self.calculate_btn.setEnabled(False)
        # Identify all selected metrics
        selected_metrics = []
        for name, checkbox in self.base_metric_widgets.items():
            if checkbox.isChecked():
                selected_metrics.append((name, self.base_metrics[name]))
        for name, checkbox in self.gt_metric_widgets.items():
            if checkbox.isChecked():
                selected_metrics.append((name, self.gt_metrics[name]))
        # Extract mask layer data
        masks1 = mask_layer.data
        masks1_bin = aiod_metrics.labelled_to_binary(masks1)
        masks2 = other_layer.data
        masks2_bin = aiod_metrics.labelled_to_binary(masks2)

        # Calculate metrics
        @thread_worker(connect={"returned": self.display_results})
        def _calc_metrics(self, selected_metrics, masks1_bin, masks2_bin):
            results = defaultdict(list)
            for name, func in selected_metrics:
                results[name].append(func(masks1_bin, masks2_bin))
            return results

        _calc_metrics(self, selected_metrics, masks1_bin, masks2_bin)

        # Reset the calculate button
        self.reset_calculate_btn()

    def display_results(self, results: dict):
        # Convert results into a DataFrame for easier display and export
        self.df_results = pd.DataFrame(results)
        # Remove any previous results
        self.output_box.clear()
        # Convert results to markdown table and display
        res = self.df_results.T.copy()
        res.index.name = "Metric"
        res.columns = ["Value"]
        self.output_box.append(
            res.to_markdown(index=True, tablefmt="simple", floatfmt=".4f")
        )
        # Insert mask layer name & ground truth layer name
        self.df_results.loc[:, "mask_layer"] = self.mask_layer_name
        self.df_results.loc[:, "other_layer"] = self.other_layer_name

    def export_results(self):
        # Prompt dialog for save location
        fname, _ = QFileDialog.getSaveFileName(
            self.parent,
            "Save results to file",
            None,
        )
        if fname == "":
            show_info("No file selected!")
            return
        # Export results to CSV at location
        self.df_results.to_csv(fname, index=False)
        # Pop-up
        show_info(f"Results exported to {fname}!")

    def append_results(self):
        # Prompt dialog for existing results
        fname, _ = QFileDialog.getOpenFileName(
            self.parent,
            "Append results to file",
            None,
        )
        if fname == "":
            show_info("No file selected!")
            return
        # Load previous results
        prev_results = pd.read_csv(fname)
        # Append
        new_results = pd.concat(
            [prev_results, self.df_results], axis=0
        ).reset_index(drop=True)
        # Save in same location
        new_results.to_csv(fname, index=False)
        # Pop-up
        show_info(f"Results appended to {fname}!")

    def reset_calculate_btn(self):
        self.calculate_btn.setText("Calculate!")
        self.calculate_btn.setEnabled(True)


"""
The big question is whether this needs Nextflow or not...If memory is an issue, that's a bit independent of Nextflow and more that HPC is needed.

In that case, we will need to move over to using Dask for computing over chunks.
"""
