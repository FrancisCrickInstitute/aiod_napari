from os import environ
import shlex
import subprocess
from pathlib import Path
from typing import Optional, Union
import re

import aiod_utils.preprocess
import napari
import pandas as pd
import qtpy.QtCore
import tqdm
import yaml
from aiod_registry import TASK_NAMES
from aiod_utils.stacks import Stack, calc_num_stacks, generate_stack_indices
from napari.utils.notifications import show_info
from qtpy.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QGridLayout,
    QLabel,
    QLayout,
    QSpinBox,
    QVBoxLayout,
    QPushButton,
    QWidget,
)

from ai_on_demand.utils import (
    InfoWindow,
    format_tooltip,
    get_img_dims,
    sanitise_name,
)
from ai_on_demand.widget_classes import SubWidget
import aiod_utils.preprocess
from aiod_utils.stacks import generate_stack_indices, calc_num_stacks, Stack
from aiod_utils.io import image_paths_to_csv
from ai_on_demand.utils import format_tooltip, get_img_dims, sanitise_name
from ai_on_demand.widget_classes import BaseNxfWidget


class InferenceNxfWidget(BaseNxfWidget):
    """
    Nextflow sub-widget for the Inference pipeline.

    Extends BaseNxfWidget with inference-specific UI (overwrite checkbox,
    advanced tiling/overlap options) and the inference pipeline logic.
    """

    _name = "nxf"
    pipeline_finished = qtpy.QtCore.Signal()
    pipeline_failed = qtpy.QtCore.Signal()

    def __init__(
        self,
        viewer: napari.Viewer,
        parent: Optional[QWidget] = None,
        layout: QLayout = QGridLayout,
        **kwargs,
    ):
        # Whether all images have been loaded – needed to extract metadata
        self.all_loaded = False
        # Dictionary to monitor per-image progress
        self.progress_dict = {}

        super().__init__(
            viewer=viewer,
            parent=parent,
            layout=layout,
            **kwargs,
        )

    def _create_variant_ui(self):
        """Inject the overwrite checkbox and Advanced Options into the pipeline group."""
        # Overwrite existing results checkbox
        self.overwrite_btn = QCheckBox("Overwrite existing results")
        self.overwrite_btn.setToolTip(
            format_tooltip(
                """
Select/enable to overwrite any previous results.

Exactly what is overwritten will depend on the pipeline selected. By default, any previous results matching the current setup will be loaded if possible. This can be disabled by ticking this box.
        """
            )
        )
        self.pipeline_layout.addWidget(self.overwrite_btn, 1, 0, 1, 1)

        # Advanced options collapsible section
        self.options_widget = QWidget()
        self.options_layout = QVBoxLayout()
        self.advanced_box = QPushButton(" ▶ Advanced Options")
        self.advanced_box.setCheckable(True)
        self.advanced_box.setStyleSheet(
            f"QPushButton {{ text-align: left; }} QPushButton:checked {{background-color: {self.parent.subwidgets['model'].colour_selected}}}"
        )
        self.advanced_box.toggled.connect(self.on_toggle_advanced)
        self.advanced_box.setToolTip(
            format_tooltip(
                """
        Show/hide advanced options for the Nextflow pipeline. These options define how to split an image into separate jobs in Nextflow. The underlying models will likely do their own splitting internally into patches, but this controls the trade-off between the number and size of each job.
"""
            )
        )
        self.advanced_widget = QWidget()
        self.advanced_layout = QGridLayout()

        self._add_advanced_options()

        self.advanced_widget.setLayout(self.advanced_layout)
        self.advanced_widget.setVisible(False)
        self.options_layout.addWidget(self.advanced_box)
        self.options_layout.addWidget(self.advanced_widget)
        self.options_layout.setContentsMargins(0, 0, 0, 0)
        self.advanced_layout.setContentsMargins(4, 0, 4, 0)
        self.options_widget.setLayout(self.options_layout)
        self.pipeline_layout.addWidget(self.options_widget, 3, 0, 1, 2)

        # Dialog button to view pipeline parameters for selected hash
        self.display_params_button = QPushButton("Pipeline Parameters")
        self.display_params_button.setToolTip(
            format_tooltip(
                """
View the parameters used for the currently selected output.
"""
            )
        )
        self.display_params_button.setEnabled(False)
        # Check if run hash available whenever selection changes
        self.viewer.layers.selection.events.changed.connect(
            lambda: self.display_params_button.setEnabled(
                bool(self.get_selected_layer_hash())
            )
        )
        self.display_params_button.clicked.connect(self.on_display_params)
        self.config_ready.connect(
            lambda: self.display_params_button.setEnabled(True)
        )
        self.inner_layout.addWidget(self.display_params_button, 5, 1, 1, 1)

    def _add_advanced_options(self):
        self.tile_x_label = QLabel("Number X tiles:")
        self.tile_x_label.setToolTip(
            format_tooltip(
                """
Number of tiles to split the image into in the X dimension. 'auto' allows Nextflow to decide an appropriate split.
"""
            )
        )
        self.tile_x = QSpinBox(minimum=0, maximum=100, value=0)
        self.tile_x.setSpecialValueText("auto")
        self.tile_x.setAlignment(qtpy.QtCore.Qt.AlignCenter)

        self.tile_y_label = QLabel("Number Y tiles:")
        self.tile_y_label.setToolTip(
            format_tooltip(
                """
Number of tiles to split the image into in the Y dimension. 'auto' allows Nextflow to decide an appropriate split.
"""
            )
        )
        self.tile_y = QSpinBox(minimum=0, maximum=100, value=0)
        self.tile_y.setSpecialValueText("auto")
        self.tile_y.setAlignment(qtpy.QtCore.Qt.AlignCenter)

        self.tile_z_label = QLabel("Number Z tiles:")
        self.tile_z_label.setToolTip(
            format_tooltip(
                """
Number of tiles to split the image into in the Z dimension. 'auto' allows Nextflow to decide an appropriate split.
"""
            )
        )
        self.tile_z = QSpinBox(minimum=0, maximum=100, value=0)
        self.tile_z.setSpecialValueText("auto")
        self.tile_z.setAlignment(qtpy.QtCore.Qt.AlignCenter)

        self.advanced_layout.addWidget(self.tile_x_label, 0, 0, 1, 1)
        self.advanced_layout.addWidget(self.tile_x, 0, 1, 1, 1)
        self.advanced_layout.addWidget(self.tile_y_label, 1, 0, 1, 1)
        self.advanced_layout.addWidget(self.tile_y, 1, 1, 1, 1)
        self.advanced_layout.addWidget(self.tile_z_label, 2, 0, 1, 1)
        self.advanced_layout.addWidget(self.tile_z, 2, 1, 1, 1)

        self.overlap_x_label = QLabel("Overlap X:")
        self.overlap_x_label.setToolTip(
            format_tooltip(
                "Fraction of overlap between tiles in the X dimension."
            )
        )
        self.overlap_x = QDoubleSpinBox(minimum=0.0, maximum=0.5, value=0.0)
        self.overlap_x.setSingleStep(0.05)
        self.overlap_x.setAlignment(qtpy.QtCore.Qt.AlignCenter)

        self.overlap_y_label = QLabel("Overlap Y:")
        self.overlap_y_label.setToolTip(
            format_tooltip(
                "Fraction of overlap between tiles in the Y dimension."
            )
        )
        self.overlap_y = QDoubleSpinBox(minimum=0.0, maximum=0.5, value=0.0)
        self.overlap_y.setSingleStep(0.05)
        self.overlap_y.setAlignment(qtpy.QtCore.Qt.AlignCenter)

        self.overlap_z_label = QLabel("Overlap Z:")
        self.overlap_z_label.setToolTip(
            format_tooltip(
                "Fraction of overlap between tiles in the Z dimension."
            )
        )
        self.overlap_z = QDoubleSpinBox(minimum=0.0, maximum=0.5, value=0.0)
        self.overlap_z.setSingleStep(0.05)
        self.overlap_z.setAlignment(qtpy.QtCore.Qt.AlignCenter)

        self.advanced_layout.addWidget(self.overlap_x_label, 3, 0, 1, 1)
        self.advanced_layout.addWidget(self.overlap_x, 3, 1, 1, 1)
        self.advanced_layout.addWidget(self.overlap_y_label, 4, 0, 1, 1)
        self.advanced_layout.addWidget(self.overlap_y, 4, 1, 1, 1)
        self.advanced_layout.addWidget(self.overlap_z_label, 5, 0, 1, 1)
        self.advanced_layout.addWidget(self.overlap_z, 5, 1, 1, 1)

        self.tile_x.valueChanged.connect(self.update_tile_size)
        self.tile_y.valueChanged.connect(self.update_tile_size)
        self.tile_z.valueChanged.connect(self.update_tile_size)
        self.overlap_x.valueChanged.connect(self.update_tile_size)
        self.overlap_y.valueChanged.connect(self.update_tile_size)
        self.overlap_z.valueChanged.connect(self.update_tile_size)

        self.tile_size_label = QLabel("No image layers found!")
        self.tile_size_label.setToolTip(
            "Tile size based on currently selected image and tile settings above."
        )
        self.advanced_layout.addWidget(self.tile_size_label, 6, 0, 1, 2)

        self.postprocess_btn = QCheckBox("Re-label output")
        self.postprocess_btn.setChecked(False)
        self.postprocess_btn.setToolTip(
            format_tooltip(
                """
If checked, the model output will be re-labelled using connected components to create consistency across slices.
        """
            )
        )
        self.advanced_layout.addWidget(self.postprocess_btn, 7, 0, 1, 2)

        self.iou_thresh_label = QLabel("IoU threshold (SAM only):")
        self.iou_thresh_label.setToolTip(
            format_tooltip(
                """
Threshold for the Intersection over Union (IoU) metric used in the SAM post-processing step.
        """
            )
        )
        self.iou_thresh = QDoubleSpinBox(minimum=0.0, maximum=1.0, value=0.8)
        self.iou_thresh.setSingleStep(0.01)
        self.iou_thresh.setAlignment(qtpy.QtCore.Qt.AlignCenter)
        self.advanced_layout.addWidget(self.iou_thresh_label, 8, 0, 1, 1)
        self.advanced_layout.addWidget(self.iou_thresh, 8, 1, 1, 1)

        self.update_tile_size(val=None, clear_label=False)

    def on_toggle_advanced(self):
        if self.advanced_box.isChecked():
            self.advanced_widget.setVisible(True)
            self.advanced_box.setText(" ▼ Advanced Options")
        else:
            self.advanced_widget.setVisible(False)
            self.advanced_box.setText(" ▶ Advanced Options")

    def get_config_params(self, params):
        config = super().get_config_params(params)
        config["advanced_options"] = {
            "num_substacks": params.get("num_substacks"),
            "overlap": params.get("overlap"),
            "iou_threshold": params.get("iou_threshold"),
        }
        return config

    def load_config(self, config):
        super().load_config(config)
        adv = config.get("advanced_options", {})

        num_substacks = adv.get("num_substacks")
        if num_substacks is not None:
            tile_boxes = [self.tile_x, self.tile_y, self.tile_z]
            for box, val in zip(tile_boxes, num_substacks.split(",")):
                if val == "auto":
                    box.setValue(-1)
                else:
                    box.setValue(int(val))

        overlap_str = adv.get("overlap")
        if overlap_str is not None:
            overlap = [float(i) for i in overlap_str.split(",")]
            self.overlap_x.setValue(overlap[0])
            self.overlap_y.setValue(overlap[1])
            self.overlap_z.setValue(overlap[2])

        iou = adv.get("iou_threshold")
        if iou is not None:
            self.iou_thresh.setValue(float(iou))

    def store_img_paths(self, img_paths: list):
        """
        Writes the provided image paths to a CSV file to pass into Nextflow.
        """
        dims = []
        self.progress_dict = {}
        total_substacks = 0

        stack_size = (
            (
                "auto"
                if self.tile_x.value() == self.tile_x.minimum()
                else self.tile_x.value()
            ),
            (
                "auto"
                if self.tile_y.value() == self.tile_y.minimum()
                else self.tile_y.value()
            ),
            (
                "auto"
                if self.tile_z.value() == self.tile_z.minimum()
                else self.tile_z.value()
            ),
        )
        stack_size = Stack(
            height=stack_size[0], width=stack_size[1], depth=stack_size[2]
        )
        overlap_frac = Stack(
            height=round(self.overlap_x.value(), 2),
            width=round(self.overlap_y.value(), 2),
            depth=round(self.overlap_z.value(), 2),
        )

        for img_path in img_paths:
            layer = self.parent.viewer.layers[img_path.stem]
            H, W, num_slices, channels = get_img_dims(layer, img_path)
            dims.append({"Z": num_slices, "Y": H, "X": W, "C": channels})
            self.progress_dict[img_path.stem] = 0

            relevant_runs = [
                i
                for i in self.parent.img_mask_info
                if i["img_path"].stem == img_path.stem
            ]
            for d in relevant_runs:
                if d["prep_set"] is None:
                    final_shape = Stack(
                        height=H, width=W, depth=num_slices, channels=channels
                    )
                else:
                    output_shape = aiod_utils.preprocess.get_output_shape(
                        d["prep_set"], input_shape=(num_slices, H, W)
                    )
                    final_shape = Stack(
                        height=output_shape[1],
                        width=output_shape[2],
                        depth=output_shape[0],
                        channels=channels,
                    )
                num_substacks, eff_shape = calc_num_stacks(
                    image_shape=final_shape,
                    req_stacks=stack_size,
                    overlap_fraction=overlap_frac,
                )
                _, num_substacks, _ = generate_stack_indices(
                    image_shape=final_shape,
                    num_substacks=num_substacks,
                    overlap_fraction=overlap_frac,
                    eff_shape=eff_shape,
                )
                total_substacks += num_substacks

        image_paths_to_csv(
            image_paths=img_paths,
            output_csv_path=self.img_list_fpath,
            dimensions=dims,
            overwrite=True,
            index=False,
        )
        self.total_substacks = total_substacks

    def check_pipeline(self):
        """Validate all required inputs for inference."""
        if self.parent.selected_task is None:
            raise ValueError("No task/organelle selected!")
        if self.parent.selected_model is None:
            raise ValueError("No model selected!")
        if "data" not in self.parent.subwidgets:
            raise ValueError("Cannot run pipeline without data widget!")
        if len(self.parent.subwidgets["data"].image_path_dict) == 0:
            raise ValueError("No data selected!")
        if self.all_loaded is False and not (
            len(self.image_path_dict) > 0
            and self.parent.subwidgets["data"].existing_loaded
        ):
            show_info("Not all images have loaded, please wait...")
            return

    def setup_pipeline(self):
        """Build the Nextflow command and params dict for inference."""
        self.image_path_dict = self.parent.subwidgets["data"].image_path_dict
        self.parent.executed_task = self.parent.selected_task
        self.parent.executed_model = self.parent.selected_model
        self.parent.executed_variant = self.parent.selected_variant

        nxf_cmd = (
            self.nxf_base_cmd + f"run {self.nxf_repo} -latest -entry inference"
        )

        parent = self.parent
        config_path = parent.subwidgets["model"].get_model_config()
        self.mask_dir_path = (
            self.nxf_store_dir
            / f"{parent.executed_model}"
            / f"{sanitise_name(parent.executed_variant)}_masks"
        )

        nxf_params = {}
        nxf_params["root_dir"] = str(self.nxf_base_dir)
        nxf_params["img_dir"] = str(self.img_list_fpath)
        nxf_params["model"] = parent.selected_model
        nxf_params["model_config"] = str(config_path)
        nxf_params["model_type"] = sanitise_name(parent.executed_variant)
        nxf_params["task"] = parent.executed_task
        num_substacks = []
        num_substacks.append(
            "auto"
            if self.tile_x.value() == self.tile_x.minimum()
            else self.tile_x.value()
        )
        num_substacks.append(
            "auto"
            if self.tile_y.value() == self.tile_y.minimum()
            else self.tile_y.value()
        )
        num_substacks.append(
            "auto"
            if self.tile_z.value() == self.tile_z.minimum()
            else self.tile_z.value()
        )
        nxf_params["num_substacks"] = ",".join(map(str, num_substacks))
        nxf_params["overlap"] = (
            f"{round(self.overlap_x.value(), 2)},{round(self.overlap_y.value(), 2)},{round(self.overlap_z.value(), 2)}"
        )
        nxf_params["iou_threshold"] = round(self.iou_thresh.value(), 2)
        nxf_params["preprocess"] = parent.subwidgets[
            "preprocess"
        ].get_all_options()

        parent.get_run_hash(nxf_params)

        if self.overwrite_btn.isChecked():
            proceed = True
            load_paths = []
            parent.get_img_mask_preps()
            all_layer_names = [i["layer_name"] for i in parent.img_mask_info]
            for layer_name in all_layer_names:
                if layer_name in self.viewer.layers:
                    self.viewer.layers.remove(layer_name)
            for mask_path in self.mask_dir_path.glob("*.rle"):
                for layer_name in all_layer_names:
                    if layer_name in mask_path.stem:
                        mask_path.unlink()
                        break
            img_paths = list(
                self.parent.subwidgets["data"].image_path_dict.values()
            )
        else:
            proceed, img_paths, load_paths = self.parent.check_masks()

        if load_paths:
            self.nxf_run_btn.setEnabled(False)
            self.nxf_run_btn.setText("Loading already-run masks...")
            self.parent.create_mask_layers(img_paths=load_paths)
        self.nxf_run_btn.setText("Run Pipeline!")
        self.nxf_run_btn.setEnabled(True)

        if not proceed:
            msg = f"Masks already exist for all files for segmenting {TASK_NAMES[parent.executed_task]} with {parent.executed_model} ({parent.executed_variant})!"
            if self.parent.run_hash is not None:
                msg += f" (Hash: {self.parent.run_hash[:8]})"
                self.display_params_button.setEnabled(True)
            show_info(msg)
            return nxf_cmd, nxf_params, proceed, img_paths
        else:
            self.parent.watch_mask_files()
            return nxf_cmd, nxf_params, proceed, img_paths

    def _pre_run_hook(self, nxf_params: dict, img_paths):
        """Store settings, image paths, and inject the postprocess flag."""
        self.parent.store_settings()
        self.store_img_paths(img_paths=img_paths)
        nxf_params["postprocess"] = self.postprocess_btn.isChecked()

    def _pipeline_start(self):
        show_info("Pipeline started!")
        self.nxf_run_btn.setEnabled(False)
        self.nxf_run_btn.setText("Running Pipeline...")
        self.init_progress_bar()
        self._add_cancel_btn(self.cancel_pipeline)

    def _pipeline_finish(self):
        show_info("Pipeline finished!")
        self._reset_btns()
        self.parent.insert_final_masks()
        self.pbar.setValue(self.total_substacks)
        self.pipeline_finished.emit()

    def _pipeline_fail(self, exc):
        show_info("Pipeline failed! See terminal for details")
        print(exc)
        self._reset_btns()
        if hasattr(self.parent, "watcher_enabled"):
            print("Deactivating watcher...")
            self.parent.watcher_enabled = False
        self.pipeline_failed.emit()

    def cancel_pipeline(self):
        self.process.send_signal(subprocess.signal.SIGTERM)
        self.reset_progress_bar()
        self.parent.remove_mask_layers()

    def init_progress_bar(self):
        self.pbar.setRange(0, self.total_substacks)
        self.pbar.setValue(0)
        self.tqdm_pbar = tqdm.tqdm(total=self.total_substacks)
        self.pbar_label.setText("Progress: [--:--]")

    def update_progress_bar(self):
        curr_slices = sum(self.progress_dict.values())
        self.pbar.setValue(curr_slices)
        self.tqdm_pbar.update(curr_slices - self.tqdm_pbar.n)
        elapsed = self.tqdm_pbar.format_dict["elapsed"]
        rate = (
            self.tqdm_pbar.format_dict["rate"]
            if self.tqdm_pbar.format_dict["rate"]
            else 1
        )
        remaining = (self.tqdm_pbar.total - self.tqdm_pbar.n) / rate
        self.pbar_label.setText(
            f"Progress: [{self.tqdm_pbar.format_interval(elapsed)}<{self.tqdm_pbar.format_interval(remaining)}]"
        )

    def update_tile_size(
        self, val: Union[int, float, None], clear_label: bool = False
    ):
        """Update the tile-size label whenever a tiling spinbox changes."""
        stack_size = (
            (
                "auto"
                if self.tile_x.value() == self.tile_x.minimum()
                else self.tile_x.value()
            ),
            (
                "auto"
                if self.tile_y.value() == self.tile_y.minimum()
                else self.tile_y.value()
            ),
            (
                "auto"
                if self.tile_z.value() == self.tile_z.minimum()
                else self.tile_z.value()
            ),
        )
        stack_size = Stack(
            height=stack_size[0], width=stack_size[1], depth=stack_size[2]
        )
        overlap_frac = Stack(
            height=round(self.overlap_x.value(), 2),
            width=round(self.overlap_y.value(), 2),
            depth=round(self.overlap_z.value(), 2),
        )

        if len(self.viewer.layers.selection) >= 1:
            layers = self.viewer.layers.selection
        else:
            layers = self.viewer.layers
        layers = [
            layer for layer in layers if isinstance(layer, napari.layers.Image)
        ]

        if len(layers) == 0 or clear_label:
            self.tile_size_label.setText("No image layers found!")
            return

        H, W, num_slices, channels = get_img_dims(layers[0], verbose=False)
        img_shape = Stack(
            height=H, width=W, depth=num_slices, channels=channels
        )
        num_substacks, eff_shape = calc_num_stacks(
            image_shape=img_shape,
            req_stacks=stack_size,
            overlap_fraction=overlap_frac,
        )
        _, num_substacks, stack_size_px = generate_stack_indices(
            image_shape=img_shape,
            num_substacks=num_substacks,
            overlap_fraction=overlap_frac,
            eff_shape=eff_shape,
        )
        self.tile_size_label.setText(
            format_tooltip(
                f"Substack size: {stack_size_px.depth} slice{'s' if stack_size_px.depth > 1 else ''}, {stack_size_px.height}px x {stack_size_px.width}px for each of the {num_substacks} jobs to submit (for the selected image).",
                width=40,
            )
        )

    def get_selected_layer_hash(self):
        if len(self.viewer.layers.selection) > 1:
            # raise NotImplementedError("Viewing hash config details for multiple output layers not supported yet")
            return ""
        # Get current layer name if it's a labels layer
        selected = [
            layer
            for layer in self.viewer.layers.selection
            if isinstance(layer, napari.layers.Labels)
        ]
        if not selected:
            return ""
        else:
            selected = selected[0]
            # Look for hash crumb pattern in layer name
            crumb = re.split(r"[\W_]", selected.name)[-1]
            file_matches = list(
                self.nxf_store_dir.glob(f"nxf_params_{crumb}*.yml")
            )
            if not file_matches:
                # No matches, layer is not an aiod output
                return ""
            elif len(file_matches) > 1:
                raise RuntimeError(
                    f"Could not find unique Nextflow params file for hash {crumb}!"
                )
            else:
                # Fetch the full hash
                full_hash = file_matches[0].stem.split("nxf_params_")[-1]
                assert full_hash.startswith(crumb)
                # Hooray, it's an aiod output
                return full_hash

    def on_display_params(self):
        params = None
        full_hash = self.get_selected_layer_hash()
        if not full_hash:
            # This should not happen: layer selection event connection should only enable this button if hash is available
            raise RuntimeError(
                "No valid output layer selected to get hash from!"
            )
        with open(
            self.nxf_store_dir / f"nxf_params_{full_hash}.yml", "r"
        ) as f:
            params = yaml.safe_load(f)

        if not params:
            info = f"Hash details for {full_hash[:8]} not found"
        else:
            # Replace "model_config" value with the contents of the YAML file
            model_config_path = params.get("model_config")
            if model_config_path and Path(model_config_path).exists():
                with open(model_config_path, "r") as f:
                    params["model_config"] = yaml.safe_load(f)
            info = yaml.dump(params)

        params_popup = InfoWindow(
            self,
            title="Pipeline parameters"
            + (f" ({params['param_hash'][:8]})" if params else ""),
            content=info,
        )
        params_popup.show()
