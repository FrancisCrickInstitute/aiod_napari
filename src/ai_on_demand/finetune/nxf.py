import subprocess
from pathlib import Path
from typing import Optional

import napari
import tqdm
from napari.utils.notifications import show_info
from qtpy.QtWidgets import (
    QGridLayout,
    QLayout,
    QWidget,
)

from ai_on_demand.utils import sanitise_name
from ai_on_demand.widget_classes import BaseNxfWidget


class FinetuneNxfWidget(BaseNxfWidget):
    """
    Nextflow sub-widget for the Finetuning pipeline.

    Extends BaseNxfWidget with finetuning-specific pipeline logic
    (check, setup, progress bar, cancellation).  No extra UI controls
    are added beyond the shared profile selector.
    """

    _name = "nxf"

    def __init__(
        self,
        viewer: napari.Viewer,
        parent: Optional[QWidget] = None,
        layout: QLayout = QGridLayout,
        **kwargs,
    ):
        self.max_epochs = 0
        self.current_epoch = 0

        super().__init__(
            viewer=viewer,
            parent=parent,
            layout=layout,
            **kwargs,
        )

    def _create_variant_ui(self):
        pass

    def check_pipeline(self):
        """Validate all required inputs for finetuning."""
        if self.parent.selected_task is None:
            raise ValueError("No task/organelle selected!")
        if self.parent.selected_model is None:
            raise ValueError("No model selected!")
        if "finetune_params" not in self.parent.subwidgets:
            raise ValueError(
                "Cannot run pipeline without finetune params widget"
            )
        if (
            len(
                self.parent.subwidgets["finetune_params"].train_dir_text.text()
            )
            == 0
        ):
            raise ValueError("No Train directory selected!")
        if (
            self.parent.subwidgets["finetune_params"].model_save_name.text()
            == ""
        ):
            raise ValueError("No save name given for finetuned model")
        if not (
            Path(
                self.parent.subwidgets["finetune_params"].train_dir_text.text()
            ).exists()
        ):
            raise FileNotFoundError("Training Directory not found")
        test_dir = self.parent.subwidgets[
            "finetune_params"
        ].test_dir_text.text()
        if test_dir != "" and not Path(test_dir).exists():
            raise FileNotFoundError("Testing Directory not found")

    def setup_pipeline(self):
        """Build the Nextflow command and params dict for finetuning."""
        self.image_path_dict = self.parent.subwidgets[
            "finetune_params"
        ].train_dir_text.text()

        img_paths = ""
        proceed = True

        finetune_config_fpath = str(Path(self.nxf_repo) / "finetune.config")
        nxf_cmd = (
            self.nxf_base_cmd
            + f"run {self.nxf_repo} -latest -entry finetune -c {finetune_config_fpath}"
        )

        self.parent.executed_task = self.parent.selected_task
        self.parent.executed_model = self.parent.selected_model
        self.parent.executed_variant = self.parent.selected_variant

        parent = self.parent
        config_path = parent.subwidgets["model"].get_model_config()

        nxf_params = {}
        nxf_params["root_dir"] = str(self.nxf_base_dir)
        nxf_params["model_save_dir"] = (
            str(self.nxf_base_dir) + "/aiod_cache/finetune_cache"
        )
        nxf_params["model"] = parent.selected_model
        nxf_params["model_config"] = str(config_path)
        nxf_params["model_type"] = sanitise_name(parent.executed_variant)
        nxf_params["task"] = parent.executed_task

        nxf_params["train_dir"] = parent.subwidgets[
            "finetune_params"
        ].train_dir_text.text()
        test_dir = parent.subwidgets["finetune_params"].test_dir_text.text()
        nxf_params["test_dir"] = test_dir
        nxf_params["epochs"] = parent.subwidgets[
            "finetune_params"
        ].epochs.value()
        self.max_epochs = nxf_params["epochs"]
        nxf_params["finetune_layers"] = parent.subwidgets[
            "finetune_params"
        ].finetune_layers.currentText()
        nxf_params["model_save_name"] = parent.subwidgets[
            "finetune_params"
        ].model_save_name.text()
        nxf_params["learning_rate"] = float(
            parent.subwidgets["finetune_params"].learning_rate.text()
        )
        nxf_params["weight_decay"] = float(
            parent.subwidgets["finetune_params"].weight_decay.text()
        )
        nxf_params["sdg"] = bool(
            parent.subwidgets["finetune_params"].use_sgd.isChecked()
        )
        nxf_params["momentum"] = float(
            parent.subwidgets["finetune_params"].momentum.text()
        )

        parent.get_run_hash(nxf_params)

        return nxf_cmd, nxf_params, proceed, img_paths

    def _pipeline_start(self):
        show_info("Pipeline started!")
        self.nxf_run_btn.setEnabled(False)
        self.nxf_run_btn.setText("Running Pipeline...")
        self.parent.subwidgets["finetune_params"].model_save_name.setDisabled(
            True
        )
        self.init_finetune_pbar(self.max_epochs)
        self._add_cancel_btn(self.cancel_pipeline)
        training_metrics_path = (
            self.nxf_params["model_save_dir"] + "/training_metrics.csv"
        )
        self.parent.watch_metrics_file(metric_path=training_metrics_path)

    def _pipeline_finish(self):
        show_info("Pipeline finished! - Save model to local model registry")
        self._reset_btns()
        self.finetuned_model_ready.emit(str(self.nxf_base_dir))
        self.parent.watch_enabled = False
        self.pbar.setValue(self.max_epochs)

    def _pipeline_fail(self, exc):
        show_info("Pipeline failed! See terminal for details")
        print(exc)
        self._reset_btns()
        self.parent.subwidgets["finetune_params"].model_save_name.setDisabled(
            False
        )
        self.parent.watch_enabled = False

    def cancel_pipeline(self):
        self.process.send_signal(subprocess.signal.SIGTERM)
        self.reset_progress_bar()

    def init_finetune_pbar(self, epochs):
        self.pbar.setRange(0, epochs)
        self.pbar.setValue(0)
        self.tqdm_pbar = tqdm.tqdm(total=epochs)
        self.pbar_label.setText("Progress: [--:--]")

    def update_finetune_pbar(self, current_epoch):
        self.pbar.setValue(current_epoch)
        self.tqdm_pbar.update(current_epoch - self.tqdm_pbar.n)
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
