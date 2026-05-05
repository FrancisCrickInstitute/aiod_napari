import builtins
import napari
import yaml
from typing import Optional
from pathlib import Path

from napari.utils.notifications import show_info
from aiod_registry import add_model_local
from napari._qt.qt_resources import QColoredSVGIcon
from qtpy.QtWidgets import (
    QWidget,
    QGridLayout,
    QVBoxLayout,
    QLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QComboBox,
    QFileDialog,
    QMessageBox,
    QCheckBox,
)
from ai_on_demand.widget_classes import SubWidget, QGroupBox
from ai_on_demand.utils import format_tooltip, calc_param_hash, sanitise_name


class FinetuneParameters(SubWidget):
    _name = "finetune_params"

    def __init__(
        self,
        viewer: napari.Viewer,
        parent: Optional[QWidget] = None,
        layout: QLayout = QGridLayout,
    ):
        super().__init__(
            viewer=viewer,
            title="Finetune Parameters",
            parent=parent,
            layout=layout,
            tooltip=parent.tooltip,
        )

        self.finetuning_meta_data = None
        self.base_model = None
        # Track the param widgets so we can read them back when building the
        # finetune YAML config. Keyed by `arg_name`.
        self.finetune_param_widgets: dict = {}
        # The list[ModelParam] in use for the currently-selected model so we
        # can look up the dtype / arg_name when serialising values.
        self._current_param_list: Optional[list] = None
        self.finetune_param_hash: Optional[str] = None

    def create_box(self):
        self.finetune_box = QGroupBox("Finetune Model")

        self.finetune_layout = QGridLayout()
        self.finetune_box.setLayout(self.finetune_layout)

        self.train_dir = QLineEdit(placeholderText="Train directory")
        self.test_dir = QLineEdit(placeholderText="Test directory (optional)")

        self.epochs = QSpinBox()
        self.epochs.setRange(0, 1000)
        # TODO: would we ever do finetuning for more than 1000 epochs?! maybe someone like Jon wants to retrain the model should we prevent that
        self.epochs.setValue(5)
        self.model_save_name = QLineEdit(
            placeholderText="Name your fine-tuned model"
        )
        # TODO: maybe not a good idea to let users pick names can be automatic like {model_name}_{finetuned}_{#}?

        self.train_dir_label = QLabel("train directory:")
        train_dir_tooltip = "Select the directory where you have saved the training data with /images, /labels"
        self.train_dir_label.setToolTip(format_tooltip(train_dir_tooltip))
        self.train_dir_text = QLabel("")
        self.train_dir_text.setWordWrap(True)
        self.train_dir_text.setToolTip(
            format_tooltip("The selected train directory.")
        )
        self.train_dir_text.setMaximumWidth(400)
        # Button to change the base directory
        self.train_dir_btn = QPushButton("Locate Training Data")
        self.train_dir_btn.clicked.connect(self.on_click_train_dir)
        self.train_dir_btn.setToolTip(format_tooltip(train_dir_tooltip))
        self.train_dir_info = QPushButton("")
        self.train_dir_info.setIcon(
            QColoredSVGIcon.from_resources("help").colored(theme="dark")
        )
        self.train_dir_info.setFixedWidth(30)
        self.train_dir_info.setToolTip("Help I don't how to structure my data")
        self.train_dir_info.clicked.connect(self._show_train_dir_info)

        self.finetune_layout.addWidget(self.train_dir_label, 0, 0)
        self.finetune_layout.addWidget(self.train_dir_text, 0, 1, 1, 2)
        self.finetune_layout.addWidget(self.train_dir_btn, 1, 0, 1, 2)
        self.finetune_layout.addWidget(self.train_dir_info, 1, 2)

        self.test_dir_label = QLabel("test directory (optional):")
        test_dir_tooltip = "Select the directory where you have saved the testing data with /images, /labels"
        self.test_dir_label.setToolTip(format_tooltip(test_dir_tooltip))
        self.test_dir_text = QLabel("")
        self.test_dir_text.setWordWrap(True)
        self.test_dir_text.setToolTip(
            format_tooltip(
                "The selected test directory. If empty, training data will be used as test data."
            )
        )
        self.test_dir_text.setMaximumWidth(400)
        self.test_dir_btn = QPushButton("Locate Testing Data")
        self.test_dir_btn.clicked.connect(self.on_click_test_dir)
        self.test_dir_btn.setToolTip(format_tooltip(test_dir_tooltip))
        self.test_dir_info = QPushButton("")
        self.test_dir_info.setIcon(
            QColoredSVGIcon.from_resources("help").colored(theme="dark")
        )
        self.test_dir_info.setFixedWidth(30)
        self.test_dir_info.setToolTip("Help I don't how to structure my data")
        self.test_dir_info.clicked.connect(self._show_test_dir_info)

        self.finetune_layout.addWidget(self.test_dir_label, 2, 0)
        self.finetune_layout.addWidget(self.test_dir_text, 2, 1, 1, 2)
        self.finetune_layout.addWidget(self.test_dir_btn, 3, 0, 1, 2)
        self.finetune_layout.addWidget(self.test_dir_info, 3, 2)

        self.finetune_layout.addWidget(QLabel("Epochs: "), 6, 0)
        self.finetune_layout.addWidget(self.epochs, 6, 1, 1, 2)

        self.manifest_name = QLineEdit(placeholderText="e.g. empanada")
        self.add_model_btn = QPushButton("Add Model To Registry")
        self.add_model_btn.setDisabled(True)
        self.add_model_btn.setToolTip(
            format_tooltip(
                "Adding model becomes available after running pipeline once"
            )
        )
        # name task location, manifestname
        self.add_model_btn.clicked.connect(self.add_model_to_registry)

        self.create_finetune_params_widget()

        self.finetune_layout.addWidget(QLabel("Finetuned model name: "), 13, 0)
        self.finetune_layout.addWidget(self.model_save_name, 13, 1, 1, 2)

        self.finetune_layout.addWidget(self.add_model_btn, 14, 0, 1, 3)

        self.inner_layout.addWidget(self.finetune_box)

    def create_finetune_params_widget(self):
        """
        Create a container that holds the dynamic, model-specific finetune
        parameter widgets. Initially populated with a placeholder; the contents
        are swapped in when a model is selected via `update_finetune_param_widget`.
        """
        # Container + layout that we can clear and refill on model change
        self.finetune_param_container = QWidget()
        self.finetune_param_container_layout = QVBoxLayout()
        self.finetune_param_container_layout.setContentsMargins(0, 0, 0, 0)
        self.finetune_param_container.setLayout(
            self.finetune_param_container_layout
        )

        # Placeholder shown until a finetunable model is selected
        self.finetune_param_placeholder = QLabel("No model selected")
        self.finetune_param_container_layout.addWidget(
            self.finetune_param_placeholder
        )

        # Track the currently mounted dynamic widget (if any) so we can replace it
        self._current_finetune_param_widget = None
        self.finetune_param_widgets = {}

        # Place the container below the rest of the static finetune controls
        self.finetune_layout.addWidget(
            self.finetune_param_container, 12, 0, 1, 3
        )

    def update_finetune_param_widget(self, param_list: Optional[list]):
        """
        Rebuild the dynamic finetune parameter widget for the given list of
        ``ModelParam`` (typically the selected model's ``finetuning_meta_data``).
        """
        # Tear down whatever is currently in the container
        if self._current_finetune_param_widget is not None:
            self.finetune_param_container_layout.removeWidget(
                self._current_finetune_param_widget
            )
            self._current_finetune_param_widget.setParent(None)
            self._current_finetune_param_widget = None
        if self.finetune_param_placeholder is not None:
            self.finetune_param_container_layout.removeWidget(
                self.finetune_param_placeholder
            )
            self.finetune_param_placeholder.setParent(None)
            self.finetune_param_placeholder = None

        if param_list:
            new_widget = self._create_finetune_param_widget(param_list)
            self._current_param_list = param_list
        else:
            # No finetuning metadata for this model — show placeholder again
            new_widget = QLabel("No finetune parameters for this model")
            self.finetune_param_widgets = {}
            self._current_param_list = None

        self._current_finetune_param_widget = new_widget
        self.finetune_param_container_layout.addWidget(new_widget)

    def _create_finetune_param_widget(self, param_list: list):
        """
        Dynamically build the params UI input for fine-tuning a model from a
        list of ``ModelParam`` objects. Mirrors ``_create_model_params_widget``
        in the inference ``ModelWidget`` so the two flows behave consistently.

        Widget type is chosen by ``ModelParam.value``:
          * ``bool``                  -> ``QCheckBox``
          * ``list``                  -> ``QComboBox`` (uses ``default`` if set)
          * ``int``/``float``/``str``/``None`` -> ``QLineEdit``
        """
        finetune_param_widget = QWidget()
        finetune_param_layout = QGridLayout()
        self.finetune_param_widgets = {}

        for i, model_param in enumerate(param_list):
            param_label = QLabel(f"{model_param.name}:")
            if model_param.tooltip:
                param_label.setToolTip(format_tooltip(model_param.tooltip))
            finetune_param_layout.addWidget(param_label, i, 0)

            param_value = model_param.value
            if param_value is True or param_value is False:
                param_val_widget = QCheckBox()
                param_val_widget.setChecked(bool(param_value))
            elif isinstance(param_value, list):
                param_val_widget = QComboBox()
                param_val_widget.addItems([str(v) for v in param_value])
                if model_param.default is not None:
                    idx = param_value.index(model_param.default)
                    param_val_widget.setCurrentIndex(idx)
            elif (
                isinstance(param_value, (int, float, str))
                or param_value is None
            ):
                param_val_widget = QLineEdit()
                param_val_widget.setText(
                    str(param_value) if param_value is not None else "None"
                )
            else:
                raise ValueError(
                    f"Finetune parameter {model_param.name!r} has unsupported "
                    f"type {type(param_value)}"
                )

            if model_param.tooltip:
                param_val_widget.setToolTip(format_tooltip(model_param.tooltip))
            finetune_param_layout.addWidget(param_val_widget, i, 1)
            # Index by `arg_name` to match the YAML config keys consumed by the
            # downstream finetune scripts.
            self.finetune_param_widgets[model_param.arg_name] = {
                "label": param_label,
                "value": param_val_widget,
                "param": model_param,
            }

        finetune_param_layout.setContentsMargins(0, 0, 0, 0)
        finetune_param_widget.setLayout(finetune_param_layout)
        return finetune_param_widget

    def _read_param_value(self, model_param, widget):
        """Read the current value from a finetune param widget, casting back to
        the schema-declared dtype."""
        if isinstance(widget, QCheckBox):
            return bool(widget.isChecked())
        if isinstance(widget, QComboBox):
            raw = widget.currentText()
        elif isinstance(widget, QLineEdit):
            raw = widget.text()
        else:
            raise NotImplementedError(
                f"Unhandled finetune widget type {type(widget)} for "
                f"{model_param.name!r}"
            )

        # If schema default was None we need to honour an empty/"None" entry
        if model_param.value is None:
            if raw == "" or raw == "None":
                return None
            cast = getattr(builtins, model_param.dtype, None)
            return cast(raw) if cast is not None else raw

        # For lists, cast back to the dtype of the (default) element
        if isinstance(model_param.value, list):
            return model_param.dtype(raw)

        return model_param.dtype(raw)

    def get_finetune_config_dict(self) -> dict:
        """Return the current finetune parameter values as a plain dict keyed
        by ``arg_name``. Returns an empty dict if the selected model has no
        finetune parameters."""
        if not self._current_param_list:
            return {}
        result = {}
        for model_param in self._current_param_list:
            entry = self.finetune_param_widgets.get(model_param.arg_name)
            if entry is None:
                continue
            result[model_param.arg_name] = self._read_param_value(
                model_param, entry["value"]
            )
        return result

    def get_finetune_config(self) -> Optional[Path]:
        """Serialise the finetune parameters to a YAML config file alongside
        the model config (under ``<nxf_base_dir>/configs``) and return the
        path. Returns ``None`` if there are no parameters to save (in which
        case no config file is needed)."""
        config_dict = self.get_finetune_config_dict()
        # Track for run-hash reproducibility
        self.finetune_param_hash = (
            calc_param_hash(config_dict) if config_dict else None
        )
        if not config_dict:
            return None

        config_dir = self.parent.subwidgets["nxf"].nxf_base_dir / "configs"
        config_dir.mkdir(parents=True, exist_ok=True)

        task = self.parent.selected_task
        model = self.parent.selected_model
        version = sanitise_name(self.parent.selected_variant or "")
        fname = (
            f"{task}-{model}-{version}_finetune_config_"
            f"{self.finetune_param_hash}.yaml"
        )
        fpath = config_dir / fname
        with open(fpath, "w") as f:
            yaml.safe_dump(
                config_dict,
                f,
                sort_keys=False,
                default_flow_style=False,
            )
        return fpath

    def on_click_train_dir(self):
        """
        Callback for when the train directory button is clicked. Opens a dialog to select a directory to get the trianing data from.
        """
        train_dir = QFileDialog.getExistingDirectory(
            self,
            caption="Select directory where the training data is",
            directory=None,
        )
        # Skip if no directory selected
        if train_dir == "":
            return
        # Replace any spaces, makes everything else easier
        new_dir_name = Path(train_dir).name.replace(" ", "_")
        train_dir = Path(train_dir).parent / new_dir_name
        # Update the text
        self.train_dir_text.setText(str(train_dir))

    def _show_train_dir_info(self):
        QMessageBox.information(
            self,
            "Training Data Information",
            # TODO: make this dynamic for different models or  enforce the structure in the segment flow side
            (
                "Training data should be organised in to a single directory containing images and masks.\n"
                'Each image mask pair should have the same name but the mask should have the suffix "_seg".\n'
                "Example image mask pair: image1.tiff, image1_seg.tiff\n"
            ),
        )

    def on_click_test_dir(self):
        """
        Callback for when the test directory button is clicked. Opens a dialog to select a directory to get the testing data from.
        """
        test_dir = QFileDialog.getExistingDirectory(
            self,
            caption="Select directory where the testing data is",
            directory=None,
        )
        # Skip if no directory selected
        if test_dir == "":
            return
        # Replace any spaces, makes everything else easier
        new_dir_name = Path(test_dir).name.replace(" ", "_")
        test_dir = Path(test_dir).parent / new_dir_name
        # Update the text
        self.test_dir_text.setText(str(test_dir))

    def _show_test_dir_info(self):
        QMessageBox.information(
            self,
            "Testing Data Information",
            (
                "Testing data should be organised the same as Training data.\n"
                "This can be used to evaluate over fitting and under fitting"
            ),
        )

    def update_finetune_params_ui(self, task_model_verson):
        version_data = self.parent.subwidgets["model"].model_version_tasks[
            task_model_verson
        ]
        # Save the schema (list[ModelParam]) for later use when registering the
        # finetuned model in the local manifest.
        self.finetuning_meta_data = version_data.finetuning_meta_data
        # Rebuild the dynamic finetune param widget from the model's metadata
        self.update_finetune_param_widget(self.finetuning_meta_data)
        # Store the base model version for registry
        self.base_model = task_model_verson[2]

    def enable_add_model(self, nxf_base_dir: str):
        self.nxf_base_dir = nxf_base_dir
        self.add_model_btn.setDisabled(False)

    def add_model_to_registry(self):
        print("saving model to registry...")
        model_name = self.model_save_name.text()
        model_task = self.parent.selected_task
        model_save_fpath = (
            f"{self.nxf_base_dir}/aiod_cache/finetune_cache/{model_name}.pth"
        )
        # TODO: maybe better to save it directly to checkpoints dir that way it won't have to copy the file when the user tries to run the model
        manifest_name = self.parent.selected_model

        # Persist the finetune parameter schema (list[ModelParam]) so the
        # local manifest exactly matches the global registry's schema.
        if self.finetuning_meta_data is None:
            finetune_meta_serialised = None
        else:
            finetune_meta_serialised = []
            for p in self.finetuning_meta_data:
                dumped = p.model_dump(exclude_none=True)
                # `extract_arg_type` overwrites `dtype` with the actual Python
                # type object (e.g. `float`), but the field is declared as a
                # str. Coerce back to the type's name so the local manifest
                # round-trips through JSON cleanly.
                dtype = dumped.get("dtype")
                if isinstance(dtype, type):
                    dumped["dtype"] = dtype.__name__
                finetune_meta_serialised.append(dumped)

        add_model_local(
            model_name,
            model_task,
            model_save_fpath,
            manifest_name,
            finetune_meta_serialised,
            self.base_model,
            cache_dir=f"{self.nxf_base_dir}/aiod_cache",
        )

        self.parent.refresh_instances(
            instances_to_refresh=["Inference", "Finetuning"]
        )

        show_info(
            "Fine-tuned model has been saved to registry and is ready to use"
        )
        self.parent.subwidgets["finetune_params"].model_save_name.setDisabled(
            False
        )
