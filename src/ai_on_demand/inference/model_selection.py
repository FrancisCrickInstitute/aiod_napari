import builtins
from typing import Optional
from pathlib import Path

import napari
from napari.utils.notifications import show_error
from napari._qt.qt_resources import QColoredSVGIcon
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QWidget,
    QLayout,
    QGridLayout,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QFileDialog,
    QLabel,
    QLineEdit,
    QComboBox,
    QCheckBox,
    QDialog,
    QTextEdit,
)
import yaml

from ai_on_demand.widget_classes import SubWidget
from ai_on_demand.utils import (
    format_tooltip,
    sanitise_name,
    merge_dicts,
    calc_param_hash,
    load_config_file,
    InfoWindow,
)


class ModelWidget(SubWidget):
    _name = "model"

    def __init__(
        self,
        viewer: napari.Viewer,
        variant: Optional[str] = None,
        parent: Optional[QWidget] = None,
        layout: QLayout = QVBoxLayout,
        **kwargs,
    ):
        # Set selection colour
        # Needs to be done before super call
        self.colour_selected = "#F7AD6F"
        self.variant = variant
        super().__init__(
            viewer=viewer,
            title="Model Selection",
            parent=parent,
            layout=layout,
            tooltip="""
Select the model and model variant to use for inference.

Parameters can be modified if setup properly, otherwise a config file can be loaded in whatever format the model takes!
        """,
            **kwargs,
        )
        # Extract the model info from all manifests
        self.extract_model_info()

        # Track whether the model defaults have been changed
        self.changed_defaults = False

    def extract_model_info(self):
        # Initialise model-related attributes
        # Easy access to the display name for each model
        self.base_to_display = {}
        self.display_to_base = {}
        # Dict of available models, and model versions, for each task
        self.versions_per_task = {}
        # Dict of model params for each model version, specific to each task
        self.model_version_tasks = {}

        # Extract the model info from all manifests
        for model_manifest in self.parent.all_manifests.values():
            # Get the short and display names
            base_name = model_manifest.short_name
            self.base_to_display[base_name] = model_manifest.name
            self.display_to_base[model_manifest.name] = base_name
            # Get each version
            for version_name, version in model_manifest.versions.items():
                # Get the tasks for this version
                for task_name, task in version.tasks.items():
                    # filter out the models without finetuning available for the finetune widget
                    if (
                        getattr(task, "finetuning_meta_data", None) is None
                        and self.variant == "finetune"
                    ):
                        continue
                    # Add this task if not yet seen
                    if task_name not in self.versions_per_task:
                        self.versions_per_task[task_name] = {}
                    # Add this base model if not yet seen for this task
                    if base_name not in self.versions_per_task[task_name]:
                        self.versions_per_task[task_name][base_name] = []
                    # Add this version, for this base model, under this task
                    self.versions_per_task[task_name][base_name].append(
                        version_name
                    )
                    # Store this model-version-task for easy access to params and config
                    self.model_version_tasks[
                        (task_name, base_name, version_name)
                    ] = task

    def create_box(self):
        # TODO: This will have to become a variant for e.g. fine-tuning
        model_box_layout = QGridLayout()
        # Create a label for the dropdown
        model_label = QLabel("Select model:")
        # Dropdown of available models
        self.model_dropdown = QComboBox()
        # Store initial message to handle erroneous clicking
        self.model_name_init = "Select a task first!"
        self.model_name_unavail = "No models available!"
        self.model_dropdown.addItems([self.model_name_init])
        # Connect function when new model selected
        self.model_dropdown.activated.connect(self.on_model_select)
        model_box_layout.addWidget(model_label, 0, 0)
        model_box_layout.addWidget(self.model_dropdown, 0, 1, 1, 2)
        # Add label + dropdown for model variants/versions
        model_version_label = QLabel("Select version:")
        model_version_label.setToolTip(
            format_tooltip(
                """
        Select the model version to use. Versions can vary either by intended functionality, or are specifically for reproducibility.
        """
            )
        )
        self.model_version_dropdown = QComboBox()
        self.model_version_dropdown.addItems(["Select a model first!"])
        self.model_version_dropdown.activated.connect(
            self.on_model_version_select
        )
        model_box_layout.addWidget(model_version_label, 1, 0)
        model_box_layout.addWidget(self.model_version_dropdown, 1, 1, 1, 2)

        # Add an icon for information about the selected model
        self.model_info_icon = QPushButton("")
        self.model_info_icon.setIcon(
            QColoredSVGIcon.from_resources("help").colored(theme="dark")
        )
        # Fix the size, and set the icon as a percentage of this
        self.model_info_icon.setFixedSize(30, 30)
        self.model_info_icon.setIconSize(self.model_info_icon.size() * 0.65)
        self.model_info_icon.setToolTip(
            format_tooltip("Information about the selected model.")
        )
        self.model_info_icon.clicked.connect(self.on_model_info)
        model_box_layout.addWidget(self.model_info_icon, 0, 3, 2, 1)

        self.inner_layout.addLayout(model_box_layout, 0, 0)

        # Create container for switching between setting params and loading config
        self.params_config_widget = QWidget()
        self.params_config_layout = QHBoxLayout()
        # Create button for displaying model param options
        self.model_param_btn = QPushButton("Modify Parameters")
        self.model_param_btn.setToolTip(
            format_tooltip(
                "Open options for modifying model parameters directly in napari."
            )
        )
        self.model_param_btn.setCheckable(True)
        self.model_param_btn.setStyleSheet(
            f"QPushButton:checked {{background-color: {self.colour_selected}}}"
        )
        self.model_param_btn.clicked.connect(self.on_click_model_params)
        self.params_config_layout.addWidget(self.model_param_btn)
        # Create button for displaying model config options
        self.model_config_btn = QPushButton("Load Config File")
        self.model_config_btn.setToolTip(
            format_tooltip(
                "Open options for loading a config file to pass to the model."
            )
        )
        self.model_config_btn.setCheckable(True)
        self.model_config_btn.setStyleSheet(
            f"QPushButton:checked {{background-color: {self.colour_selected}}}"
        )
        self.model_config_btn.clicked.connect(self.on_click_model_config)
        self.params_config_layout.addWidget(self.model_config_btn)
        # Reduce unnecessary margins/spacing
        self.params_config_layout.setContentsMargins(0, 0, 0, 0)
        self.params_config_layout.setSpacing(5)
        self.params_config_widget.setLayout(self.params_config_layout)
        # Add the widgets to the overall container

        if self.variant == "inference":
            self.inner_layout.addWidget(self.params_config_widget, 1, 0)

        # Create widgets for the two options
        self.create_model_param_widget()
        self.create_model_config_widget()

    @property
    def config_dir(self) -> Path:
        return self.parent.subwidgets["nxf"].nxf_base_dir / "configs"

    def on_model_select(self):
        """
        Callback for when a model button is clicked.

        Updates model params & config widgets for selected model.
        """
        # Extract selected model
        model_name = self.model_dropdown.currentText()
        if model_name == self.model_name_init:
            self.clear_model_param_widget()
            self.set_model_param_widget("init")
            return
        elif model_name == self.model_name_unavail:
            self.clear_model_param_widget()
            self.set_model_param_widget("init")
            self.model_version_dropdown.clear()
            self.model_version_dropdown.addItems([self.model_name_unavail])
            return
        # Get the shorthand name from the model display name
        self.parent.selected_model = self.display_to_base[model_name]
        # Update the dropdown for the model variants
        self.model_version_dropdown.clear()
        # Extract the model versions for this model for this task
        model_versions = self.versions_per_task[self.parent.selected_task][
            self.parent.selected_model
        ]
        self.model_version_dropdown.addItems(model_versions)
        self.parent.selected_variant = (
            self.model_version_dropdown.currentText()
        )
        # Update the model params & config widgets for the selected model
        self.update_model_param_config(
            self.parent.selected_model, self.parent.selected_variant
        )
        if self.variant == "finetune":
            task = self.parent.selected_task
            model = self.parent.selected_model
            version = self.model_version_dropdown.currentText()
            task_model_version = (task, model, version)
            self.parent.subwidgets["finetune_params"].update_finetune_layers(
                task_model_version
            )

    def on_model_version_select(self):
        # Update tracker for selected model variant/version
        self.parent.selected_variant = (
            self.model_version_dropdown.currentText()
        )
        # Update the model params & config widgets for the selected model variant/version
        self.update_model_param_config(
            self.parent.selected_model, self.parent.selected_variant
        )
        if self.variant == "finetune":
            task = self.parent.selected_task
            model = self.parent.selected_model
            version = self.model_version_dropdown.currentText()
            task_model_version = (task, model, version)
            self.parent.subwidgets["finetune_params"].update_finetune_layers(
                task_model_version
            )

    def on_click_model_params(self):
        # Uncheck config button
        if self.model_config_btn.isChecked():
            self.model_config_btn.setChecked(False)
            self.model_config_widget.setVisible(False)
        # Change visibility depending on checked status
        if self.model_param_btn.isChecked():
            self.curr_model_param_widget.setVisible(True)
            self.model_param_widget.setVisible(True)
        else:
            self.curr_model_param_widget.setVisible(False)
            self.model_param_widget.setVisible(False)

    def on_click_model_config(self):
        # Uncheck param button
        if self.model_param_btn.isChecked():
            self.model_param_btn.setChecked(False)
            self.curr_model_param_widget.setVisible(False)
            self.model_param_widget.setVisible(False)
        # Change visibility depending on checked status
        if self.model_config_btn.isChecked():
            self.model_config_widget.setVisible(True)
        else:
            self.model_config_widget.setVisible(False)

    def create_model_config_widget(self):
        """
        Creates the widget for loading a model config file.
        """
        self.model_config_widget = QWidget()
        self.model_config_layout = QGridLayout()

        # Add the button for loading a config file
        self.model_config_load_btn = QPushButton("Select model config")
        self.model_config_load_btn.clicked.connect(self.select_model_config)
        self.model_config_load_btn.setToolTip(
            format_tooltip(
                """
            Select a config file to be used for the selected model.
            Note that no checking/validation is done on the config file, it is just given to the model.
        """
            )
        )
        self.model_config_layout.addWidget(
            self.model_config_load_btn, 0, 0, 1, 2
        )
        # Add a label to display the selected config file (if any)
        self.model_config_label = QLabel("No model config file selected.")
        self.model_config_label.setWordWrap(True)
        self.model_config_layout.addWidget(self.model_config_label, 1, 0, 1, 2)
        # Set the overall widget layout
        self.model_config_widget.setLayout(self.model_config_layout)

        self.model_config_widget.setVisible(False)
        self.inner_layout.addWidget(self.model_config_widget)

    def create_model_param_widget(self):
        """
        Creates the widget for modifying model parameters.
        """
        self.model_param_widget = QWidget()
        self.model_param_layout = QVBoxLayout()
        # Add a button for resetting the config / UI params to defaults
        self.model_config_clear_btn = QPushButton(
            "Reset parameters to defaults"
        )
        self.model_config_clear_btn.clicked.connect(self.reset_model_config)
        self.model_config_clear_btn.setToolTip(
            format_tooltip(
                "Reset all model parameters to their defaults and clear any loaded config file."
            )
        )
        self.model_param_layout.addWidget(self.model_config_clear_btn)
        # Create a container for the model parameters and containing widget
        # NOTE: Should be light on memory, but this does store for every model!
        self.model_param_dict = {}
        self.model_param_widgets_dict = {}
        # Create initial widget as no model has been selected
        init_widget = QWidget()
        init_widget.setLayout(QVBoxLayout())
        # Add an initial label to show that no model has been selected
        model_param_init_label = QLabel("No model selected.")
        # Add this to the collapsible box
        init_widget.layout().addWidget(model_param_init_label)
        self.curr_model_param_widget = init_widget
        # Store initial widget so we can revert state if model deselected
        self.model_param_widgets_dict["init"] = init_widget
        # Add the initial widget to the main model param widget
        self.model_param_layout.addWidget(init_widget)
        self.model_param_widget.setLayout(self.model_param_layout)
        # Add a widget for when there aren't parameters, and a config is needed
        no_param_widget = QWidget()
        no_param_widget.setLayout(QVBoxLayout())
        no_param_label = QLabel(
            "Cannot modify parameters for this model!\nPlease select a config file."
        )
        no_param_widget.layout().addWidget(no_param_label)
        self.model_param_widgets_dict["no_param"] = no_param_widget
        # Stores user-loaded config paths for models with no UI params,
        # keyed by (task, model, version) tuple
        self.user_config_path: dict[tuple, Path] = {}
        # Disable showing widget until selected to view
        self.model_param_widget.setVisible(False)
        self.inner_layout.addWidget(self.model_param_widget)

    def update_model_param_config(self, model_name: str, model_version: str):
        """
        Updates the model param and config widgets for a specific model.

        Currently only updates the model widget as the config is now constant.
        """
        # Update the model parameters
        self.update_model_param_widget(model_name, model_version)

    def update_model_param_widget(self, model_name: str, model_version: str):
        """
        Updates the model param widget for a specific model
        """
        # Skip if initial message is still showing and clicked
        # TODO: Figure out when this occurs, and why it isn't model_name == model_name_init, or model_name == model_name_unavail
        if model_name not in self.base_to_display:
            return
        # Remove the current model param widget
        self.clear_model_param_widget()
        # Construct the unique tuple for this widget
        task_model_version = (
            self.parent.selected_task,
            model_name,
            model_version,
        )
        # Extract the default parameters for this model-version-task
        param_list = self.model_version_tasks[task_model_version].params
        # NOTE: Likely to create a lot of redundant widgets, but should be light on memory
        # and is the most extendable
        # Retrieve the widget for this model if already created
        if task_model_version in self.model_param_widgets_dict:
            self.curr_model_param_widget = self.model_param_widgets_dict[
                task_model_version
            ]
        # If no parameters, use the no_param widget
        elif param_list is None:
            self.curr_model_param_widget = self.model_param_widgets_dict[
                "no_param"
            ]
        # Otherwise, construct it
        else:
            self.curr_model_param_widget = self._create_model_params_widget(
                task_model_version, param_list
            )
            self.model_param_widgets_dict[task_model_version] = (
                self.curr_model_param_widget
            )
        # Set the current model param widget
        self.set_model_param_widget()

    def _create_model_params_widget(
        self, task_model_version: tuple[str, str, str], param_list: list
    ):
        """
        Creates the widget for a specific model's parameters to swap in and out
        """
        # Create a widget for the model parameters
        model_widget = QWidget()
        model_layout = QGridLayout()
        # Create container for model parameters
        self.model_param_dict[task_model_version] = {}
        # Add the default model parameters
        for i, model_param in enumerate(param_list):
            # Create labels for each of the model parameters
            param_label = QLabel(f"{model_param.name}:")
            param_label.setToolTip(format_tooltip(model_param.tooltip))
            model_layout.addWidget(param_label, i, 0)
            # Add the model parameter(s)
            param_values = model_param.value
            # Widget added depends on the input
            # True/False -> Checkbox
            if param_values is True or param_values is False:
                param_val_widget = QCheckBox()
                # Checked if default param value is True, unchecked if False
                if param_values:
                    param_val_widget.setChecked(True)
                else:
                    param_val_widget.setChecked(False)
                param_val_widget.stateChanged.connect(self.on_param_changed)
            # List -> ComboBox
            elif isinstance(param_values, list):
                param_val_widget = QComboBox()
                param_val_widget.addItems([str(i) for i in param_values])
                if model_param.default is not None:
                    default_idx = param_values.index(model_param.default)
                    param_val_widget.setCurrentIndex(default_idx)
                param_val_widget.currentIndexChanged.connect(
                    self.on_param_changed
                )
            # Int/float/str/None -> LineEdit
            elif (
                isinstance(param_values, (int, float, str))
                or param_values is None
            ):
                param_val_widget = QLineEdit()
                param_val_widget.setText(str(param_values))
                param_val_widget.textChanged.connect(self.on_param_changed)
            else:
                # Should be handled on the Pydantic side
                raise ValueError(
                    f"Model parameter {model_param.name} has invalid type {type(param_values)}"
                )
            model_layout.addWidget(param_val_widget, i, 1)
            # Store for later retrieval when saving the config
            self.model_param_dict[task_model_version][model_param.name] = {
                "label": param_label,
                "value": param_val_widget,
            }
        # Tighten up margins and set layout
        model_layout.setContentsMargins(0, 0, 0, 0)
        model_widget.setLayout(model_layout)
        return model_widget

    def on_param_changed(self):
        self.changed_defaults = True

    def clear_model_param_widget(self):
        # Remove the current model param widget
        self.model_param_layout.removeWidget(self.curr_model_param_widget)
        self.curr_model_param_widget.setParent(None)

    def set_model_param_widget(self, task_model_version: tuple | None = None):
        if task_model_version is not None:
            self.curr_model_param_widget = self.model_param_widgets_dict[
                task_model_version
            ]
        # Set the collapsible box to contain the params for this model
        self.model_param_layout.addWidget(self.curr_model_param_widget)
        # Ensure it's visible if the params button is pressed
        if self.model_param_btn.isChecked():
            self.curr_model_param_widget.setVisible(True)
        else:
            self.curr_model_param_widget.setVisible(False)

    def update_model_box(self, task_name: str):
        """The model box updates according to what's defined for each task."""
        # Clear and set available models in dropdown
        self.model_dropdown.clear()
        # Check that there is a model available for this task
        if task_name in self.versions_per_task:
            model_names = sorted(
                [
                    self.base_to_display[i]
                    for i in self.versions_per_task[task_name]
                ]
            )
        else:
            model_names = [self.model_name_unavail]
        self.model_dropdown.addItems(model_names)
        # Technically the first model in the list is now selected
        # So update everything accordingly
        self.on_model_select()

    def select_model_config(self):
        """
        Opens a file dialog for selecting the model config.

        Filters can be applied to restrict selection, which may help if restricting processing of those files.
        """
        fname, _ = QFileDialog.getOpenFileName(
            self,
            "Select a model config",
            str(self.config_dir),
            "Configs (*.yaml *.yml *.json)",
        )  # type: ignore
        # Reset if dialog cancelled
        if fname == "":
            return
        # Load the config and populate the UI widgets with the values from the config
        self.load_model_config(Path(fname))

    def load_model_config(self, config_path: Path):
        # If the model has no UI params, store the config path to pass directly
        # to the pipeline and skip UI population entirely.
        task_model_version = self.get_task_model_variant(executed=False)
        if all(task_model_version):
            param_list = self.model_version_tasks[task_model_version].params
            if param_list is None:
                self.user_config_path[task_model_version] = config_path
                self.model_config_label.setText(
                    f"Config '{config_path.name}' will be passed directly to the pipeline."
                )
                return
        # Parse the config and populate the UI widgets
        try:
            config = load_config_file(config_path)
            self.fill_ui_from_config(config)
            self.model_config_label.setText(
                f"Config '{config_path.name}' loaded into UI parameters."
            )
        except (KeyError, ValueError):
            # Let's assume they loaded a project config and try to extract the model config from the project config
            try:
                # Need to first load the project config
                project_config = load_config_file(config_path)
                # Then get the path to the model config contained therein
                model_config_path = Path(
                    project_config["model"]["model_config"]
                )
                # Now try to load *that* config and populate the UI widgets
                config = load_config_file(model_config_path)
                self.fill_ui_from_config(config)
                self.model_config_label.setText(
                    f"Config '{model_config_path.name}' loaded into UI parameters."
                )
            except KeyError as e:
                show_error(
                    f"Failed to load config '{config_path.name}': assumed input was a project config but no {e} key found!"
                )
                self.model_config_label.setText("No model config file loaded.")
                return
        except UserWarning as e:
            show_error(f"Failed to load config '{config_path.name}': {e}")
            self.model_config_label.setText("No model config file loaded.")
            return

    def reset_model_config(self):
        """Reset model parameter widgets to their schema defaults."""
        self.model_config_label.setText("No model config file loaded.")

        task_model_version = self.get_task_model_variant(executed=False)
        # Nothing selected yet — nothing to reset
        if not all(task_model_version):
            return
        param_list = self.model_version_tasks[task_model_version].params
        if param_list is None:
            # Clear any user-loaded config path for no-params models
            self.user_config_path.pop(task_model_version, None)
            return
        # Guard: widget dict may not exist if model was never fully rendered
        if task_model_version not in self.model_param_dict:
            return

        widget_dict = self.model_param_dict[task_model_version]
        for param in param_list:
            widget = widget_dict[param.name]["value"]
            default = param.value
            if isinstance(widget, QCheckBox):
                widget.setChecked(bool(default))
            elif isinstance(widget, QComboBox):
                # Default is the first item in the list
                widget.setCurrentIndex(0)
            elif isinstance(widget, QLineEdit):
                widget.setText(str(default) if default is not None else "None")
        self.changed_defaults = False

    def get_model_config(self) -> Path:
        """
        Construct the final model configuration from the UI.

        If the model has a schema config path, it is loaded as the base and UI
        parameter values are merged on top. Since param.value defaults are
        sourced from the same schema, the merge is always safe: unchanged widgets
        produce identical values to the config file, and any keys in config_path
        not covered by params pass through untouched.

        If there is no schema config, the config is built entirely from the UI
        parameter widgets.
        """
        task_model_version = self.get_task_model_variant(executed=True)
        model_version = self.model_version_tasks[task_model_version]

        if model_version.config_path is not None:
            base_dict = load_config_file(Path(model_version.config_path))
            if model_version.params is not None:
                model_dict = self.fill_config_from_ui(
                    base_dict, task_model_version=task_model_version
                )
            else:
                # No UI params: prefer user-loaded config, fall back to schema config
                user_path = self.user_config_path.get(task_model_version)
                model_dict = (
                    load_config_file(user_path) if user_path else base_dict
                )
        else:
            if model_version.params is None:
                # No schema and no UI params: must have a user-loaded config
                user_path = self.user_config_path.get(task_model_version)
                model_dict = load_config_file(user_path) if user_path else {}
            else:
                # No schema config: UI widgets are the sole source of truth
                model_dict = self.fill_config_from_ui(
                    {}, task_model_version=task_model_version
                )
        # Extract hash for model configs for tracking/reproducibility
        self.model_param_hash = calc_param_hash(model_dict)
        return self.save_model_config(model_dict)

    def save_model_config(self, model_dict: dict) -> Path:
        # Define save path for the model config
        self.config_dir.mkdir(parents=True, exist_ok=True)

        task_model_variant_name = self.get_task_model_variant_name()

        # Name the config using the same task-model-version convention as the masks
        model_config_fpath = (
            self.config_dir
            / f"{task_model_variant_name}_config_{self.model_param_hash}.yaml"
        )
        # Save the yaml config
        with open(model_config_fpath, "w") as f:
            yaml.dump(model_dict, f)
        return model_config_fpath

    def create_config_params(
        self, task_model_version: tuple | None = None
    ) -> dict:
        """
        Construct the model config from the parameter widgets.

        If no task_model_version is given, the executed model, variant, and task are used.
        """
        # Get the current dictionary of widgets for selected model
        if task_model_version is None:
            task_model_version = self.get_task_model_variant(executed=True)
        model_dict_orig = self.model_param_dict[task_model_version]
        # Get the relevant default params for this model
        default_params = self.model_version_tasks[task_model_version].params
        # Reformat the dict to pipe into downstream model run scripts
        model_dict = {}
        # Extract params from model param widgets
        for orig_param, (param_name, sub_dict) in zip(
            default_params, model_dict_orig.items()
        ):
            # Dicts maintain insertion order, so this should be fine, but double-check
            assert orig_param.name == param_name
            if isinstance(sub_dict["value"], QLineEdit):
                param_value = sub_dict["value"].text()
            elif isinstance(sub_dict["value"], QComboBox):
                param_value = sub_dict["value"].currentText()
            elif isinstance(sub_dict["value"], QCheckBox):
                param_value = sub_dict["value"].isChecked()
            else:
                raise NotImplementedError
            # Extract the original/intended dtype and cast what's in the box
            # If None, get the provided dtype from the schema and cast
            if orig_param.value is None:
                # Thanks textbox conversions
                # Could check NoneType compatible?
                if param_value is None or param_value == "None":
                    model_dict[orig_param.arg_name] = None
                else:
                    model_dict[orig_param.arg_name] = getattr(
                        builtins, orig_param.dtype
                    )(param_value)
            # Otherwise cast to the default value's dtype
            else:
                model_dict[orig_param.arg_name] = orig_param.dtype(param_value)
        return model_dict

    def fill_ui_from_config(self, config: dict):
        """
        Populate model parameter widgets from a config dict.

        Keys in the config that are not present in the UI raise a ValueError.
        Keys missing from the config are left at their current/default values.
        """
        task_model_version = self.get_task_model_variant(executed=False)
        if not all(task_model_version):
            raise UserWarning("No model/task/version selected.")

        param_list = self.model_version_tasks[task_model_version].params
        if param_list is None:
            raise ValueError(
                "Selected model has no UI parameters; use a config file directly."
            )

        # Build mapping from arg_name -> display name (widget dict key)
        arg_to_name = {p.arg_name: p.name for p in param_list}

        # Error on any key not represented in the UI
        unknown = set(config.keys()) - set(arg_to_name.keys())
        if unknown:
            raise ValueError(
                f"Config contains keys not present in the UI: {sorted(unknown)}"
            )

        widget_dict = self.model_param_dict[task_model_version]
        for arg_name, value in config.items():
            param_name = arg_to_name[arg_name]
            widget = widget_dict[param_name]["value"]
            if isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            elif isinstance(widget, QComboBox):
                idx = widget.findText(str(value))
                if idx != -1:
                    widget.setCurrentIndex(idx)
            elif isinstance(widget, QLineEdit):
                widget.setText(str(value) if value is not None else "None")
            else:
                raise NotImplementedError(
                    f"Unknown widget type for parameter '{param_name}'"
                )
        self.changed_defaults = True

    def fill_config_from_ui(
        self, config: dict, task_model_version: tuple | None = None
    ) -> dict:
        """
        Return a copy of config overridden by the current UI parameter widgets.

        UI values take precedence over any values already in config.
        If task_model_version is not provided, the currently selected model is used.
        """
        if task_model_version is None:
            task_model_version = self.get_task_model_variant(executed=False)
        if not all(task_model_version):
            raise ValueError("Cannot read UI: no model/task/version selected.")
        gui_dict = self.create_config_params(
            task_model_version=task_model_version
        )
        return merge_dicts(config, gui_dict)

    def get_task_model_variant(
        self, executed: bool = True
    ) -> tuple[str, str, str]:
        if executed:
            task, model, version = (
                self.parent.executed_task,
                self.parent.executed_model,
                self.parent.executed_variant,
            )
        else:
            task, model, version = (
                self.parent.selected_task,
                self.parent.selected_model,
                self.parent.selected_variant,
            )
        return (task, model, version)

    def get_task_model_variant_name(self, executed: bool = True) -> str:
        task, model, version = self.get_task_model_variant(executed)
        return f"{task}-{model}-{sanitise_name(version)}"

    def load_config(self, config):
        model_name = config["name"]
        model_version = config["model_type"]
        model_config_path = config["model_config"]

        model_display_name = self.base_to_display.get(model_name, None)
        if model_display_name is None:
            raise ValueError(f"Model {model_name} not recognised.")
        model_index = self.model_dropdown.findText(model_display_name)
        if model_index == -1:
            raise ValueError(
                f"Model {model_name} not available for this task."
            )
        self.model_dropdown.setCurrentIndex(model_index)
        self.on_model_select()

        version_index = -1
        for i in range(self.model_version_dropdown.count()):
            item_text = self.model_version_dropdown.itemText(i)
            if sanitise_name(item_text) == sanitise_name(model_version):
                version_index = i
                break
        if version_index == -1:
            raise ValueError(
                f"Model version {model_version} not available for this model."
            )
        self.model_version_dropdown.setCurrentIndex(version_index)
        self.on_model_version_select()

        # Now load and apply model config
        self.load_model_config(Path(model_config_path))

    def get_config_params(self, params):
        widget_config = {
            "name": params.get("model"),
            "model_type": params.get("model_type"),
            "model_config": params.get("model_config"),
        }
        return widget_config

    def on_model_info(self):
        """
        Callback for when the model info button is clicked.

        Opens a dialog box with information about the selected model.
        """
        # Extract the model info from the manifest
        task_model_version = self.get_task_model_variant(executed=False)
        # Handle not all selections made
        if not all(task_model_version):
            show_error("Please select a task, model, and version first!")
            return
        model_version = self.model_version_tasks[task_model_version]
        # Extract info from the schema if present
        # Need to get the parent model this version came from
        full_manifest = self.parent.all_manifests[task_model_version[1]]
        # Construct the message
        model_info = f"""Model: {self.base_to_display[task_model_version[1]]}
Version: {task_model_version[2]}
{full_manifest.metadata}"""

        # Add a usage guide if it exists
        if full_manifest.usage_guide is not None:
            model_info += f"\nUsage Guide:\n{full_manifest.usage_guide}\n"

        # Add the parameters if they exist
        if model_version.params is not None:
            model_info += "\nParameters:"
            for param in model_version.params:
                model_info += f"\n- {param.name} (default={param.value})"
                if param.tooltip is not None and param.tooltip != "":
                    model_info += f"\n        {param.tooltip}"
            model_info += "\n"

        # Add the config path if it exists
        if model_version.config_path is not None:
            model_info += f"\nConfig path: {model_version.config_path}"

        self.model_window = InfoWindow(
            self, title="Model Information", content=model_info
        )
        self.model_window.show()

    def refresh_ui(self):
        print("refreshing widget")
        self.extract_model_info()
        self.update_model_box(self.parent.selected_task)


class ModelInfoWindow(QDialog):
    def __init__(self, parent=None, model_info: str = ""):
        super().__init__(parent)

        # Set the layout
        self.layout = QVBoxLayout()
        # Set the window title
        self.setWindowTitle("Model Information")
        # Add the info label
        self.info_label = QTextEdit()
        # Make the text selectable, but not editable
        self.info_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.info_label.setText(model_info)
        self.info_label.setMinimumSize(500, 500)

        self.layout.addWidget(self.info_label)
        self.setLayout(self.layout)
