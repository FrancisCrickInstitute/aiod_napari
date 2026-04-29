import time
import os
from pathlib import Path
from typing import Optional

import napari
from napari.qt.threading import thread_worker

from ai_on_demand.finetune.finetune_params import FinetuneParameters
from ai_on_demand.finetune.nxf import FinetuneNxfWidget
from ai_on_demand.inference import (
    ModelWidget,
    TaskWidget,
)
from ai_on_demand.utils import calc_param_hash
from ai_on_demand.widget_classes import MainWidget


class Finetune(MainWidget):
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__(
            napari_viewer,
            title="Finetuning",
            tooltip="""
            Finetune existing models
                         """,
        )
        self.selected_task = None
        self.selected_model = None
        self.selected_variant = None
        self.executed_task = None
        self.executed_model = None
        self.executed_variant = None
        self.run_hash = None
        self.watch_enabled = None

        # Create radio buttons for selecting task (i.e. organelle)
        self.register_widget(
            TaskWidget(viewer=self.viewer, parent=self, expanded=False)
        )

        # Create radio buttons for selecting the model to run
        # Functionality currently limited to Meta's Segment Anything Model
        self.register_widget(
            ModelWidget(
                viewer=self.viewer,
                parent=self,
                variant="finetune",
                expanded=False,
            )
        )

        self.register_widget(
            FinetuneParameters(viewer=self.viewer, parent=self)
        )

        # Add the button for running the Nextflow pipeline
        self.register_widget(
            FinetuneNxfWidget(
                viewer=self.viewer,
                parent=self,
                expanded=False,
            )
        )

        self.subwidgets["nxf"].finetuned_model_ready.connect(
            self.subwidgets["finetune_params"].enable_add_model
        )

    def update_epoch(self, epoch):
        self.subwidgets["nxf"].update_finetune_pbar(epoch)

    def watch_metrics_file(self, metric_path):
        """
        File watcher to watch for a change in the finetuning metrics file

        This is used to update the progress bar based on completed epochs
        """
        print("watcher has been called")
        # Clear previous metrics
        if os.path.exists(metric_path):
            with open(metric_path, "w+"):
                pass

        @thread_worker(connect={"yielded": self.update_epoch})
        def _watch_metrics_file():
            print(f"watching metrics file for finetuning {metric_path}")
            last_epoch = 0
            self.watch_enabled = True
            while self.watch_enabled:  # enable at start of finetuning pipeline
                if Path(metric_path).exists():
                    with open(metric_path, "r") as f:
                        lines = f.read().splitlines()
                        if len(lines) > 1:  # Skip header row
                            last_line = lines[-1]
                            parts = last_line.split(",")
                            epoch = int(parts[0])
                            train_loss = parts[1]
                            test_loss = parts[2] if len(parts) > 2 else "N/A"
                            if epoch > last_epoch:
                                print(
                                    f"epoch: {epoch}, train_loss: {train_loss}, test_loss: {test_loss}\n"
                                )
                                last_epoch = epoch
                                yield epoch
                time.sleep(2)

        _watch_metrics_file()

    def get_run_hash(self, nxf_params: Optional[dict] = None):
        """
        Gather all the parameters from the subwidgets to be used in obtaining a unique hash for a run.
        """
        nxf_params = nxf_params or {}
        hashed_params = {}
        # Add model details
        hashed_params["task"] = nxf_params.get("task", self.selected_task)
        hashed_params["model"] = nxf_params.get("model", self.selected_model)
        hashed_params["variant"] = nxf_params.get(
            "model_type", self.selected_variant
        )
        # Add the model dictionary (hashed)
        hashed_params["model_hash"] = self.subwidgets["model"].model_param_hash
        # and Nextflow parameters that affect the output
        self.run_hash = calc_param_hash(hashed_params)
