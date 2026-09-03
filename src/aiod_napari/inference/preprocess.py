from functools import partial

import napari
import numpy as np
import qtpy.QtCore
from aiod_utils.preprocess import (
    get_all_preprocess_methods,
    get_downsample_factor,
    get_params_str,
    run_preprocess,
)
from napari.utils.notifications import show_error, show_info, show_warning
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from aiod_napari.utils import ConfirmDialog, format_tooltip
from aiod_napari.widget_classes import SubWidget


class PreprocessWidget(SubWidget):
    _name = "preprocess"

    def __init__(
        self,
        viewer: napari.Viewer,
        parent: QWidget | None = None,
        layout: QLayout = QVBoxLayout,
        **kwargs,
    ):
        # Load and extract all the available preprocessing options
        self.preprocess_methods = get_all_preprocess_methods()
        # Store the elements for later extraction
        self.preprocess_boxes = {}
        # Store the order of the preprocessing
        self.preprocess_order = None
        self.init_order = "None selected!"
        # Store the order as a list for easier manipulation
        self.order_list: None | list[str] = None
        # Container for multiple sets of preprocessing options
        self.preprocess_sets = []

        super().__init__(
            viewer=viewer,
            title="Preprocessing",
            parent=parent,
            layout=layout,
            tooltip="""
Select image preprocessing options. Note that all preprocessing is done on-the-fly in Nextflow.

Any preprocessing applied here is for visualization purposes only, only the original image will be used in the Nextflow pipeline.
""",
            **kwargs,
        )

    def create_box(self, variant: str | None = None):
        # Need to create these first as they are used in the callback
        self.order_label = QLabel("Preprocessing order:")
        self.preprocess_order = QLineEdit()
        self.preprocess_order.setReadOnly(True)
        self.preprocess_order.setText(self.init_order)

        _min_spin_width = "35px"
        # Go through each method, creating a box and populating the UI elements for each parameter
        for name, d in self.preprocess_methods.items():
            group_box = QGroupBox(name)
            self.preprocess_boxes[name] = {
                "box": group_box,
                "params": {},
            }
            if getattr(d["object"], "tooltip", None) is not None:
                group_box.setToolTip(format_tooltip(d["object"].tooltip))
            group_box.setCheckable(True)
            group_box.setChecked(False)
            group_box.clicked.connect(self.on_click_preprocess(name))
            group_layout = QVBoxLayout()
            group_box.setLayout(group_layout)

            # Loop through params
            for i, (param_name, param_dict) in enumerate(d["params"].items()):
                # Create the label
                label = QLabel(param_dict["name"])
                param_row = QHBoxLayout()
                param_row.addWidget(label)

                # Create the input based on type
                # If values key exist, multiple options to select
                if "values" in param_dict:
                    widget = QComboBox()
                    for value in param_dict["values"]:
                        widget.addItem(value)
                    # Set the default value
                    widget.setCurrentIndex(
                        param_dict["values"].index(param_dict["default"])
                    )
                    param_row.addWidget(widget)
                elif isinstance(param_dict["default"], bool):
                    widget = QCheckBox()
                    if param_dict["default"]:
                        widget.setChecked(True)
                    else:
                        widget.setChecked(False)
                    param_row.addWidget(widget)
                elif isinstance(param_dict["default"], (str)):
                    widget = QLineEdit()
                    widget.setText(str(param_dict["default"]))
                    param_row.addWidget(widget)
                elif isinstance(param_dict["default"], (int, float)):
                    widget = (
                        QSpinBox()
                        if isinstance(param_dict["default"], int)
                        else QDoubleSpinBox()
                    )
                    widget.setValue(param_dict["default"])
                    widget.setStyleSheet(f"min-width: {_min_spin_width}")
                    param_row.addWidget(widget)
                # Get cleaner representation of list/tuple (avoid () & [])
                elif isinstance(defaults := param_dict["default"], (list, tuple)):
                    subwidgets = []
                    for val in defaults:
                        if isinstance(val, (int, float)):
                            subwidget = (
                                QSpinBox()
                                if isinstance(val, int)
                                else QDoubleSpinBox()
                            )
                            subwidget.setValue(val)
                            subwidget.setStyleSheet(f"min-width: {_min_spin_width}")
                        elif isinstance(val, str):
                            subwidget = QLineEdit()
                            subwidget.setText(val)
                        else:
                            raise ValueError(
                                f"Parameter {param_name} of preprocess method {name} has an invalid type ({type(val)}) in the default list/tuple."
                            )
                        param_row.addWidget(subwidget)
                        subwidgets.append(subwidget)
                    widget = subwidgets
                else:
                    raise ValueError(
                        f"Parameter {param_name} of preprocess method {name} has an invalid type ({type(param_dict['default'])})."
                    )
                # Add tooltip if available
                if "tooltip" in param_dict:
                    tooltip = format_tooltip(param_dict["tooltip"])
                    for i in range(param_row.count()):
                        param_row.itemAt(i).widget().setToolTip(tooltip)
                # Build the row and add to group
                param_row.addStretch()
                group_layout.addLayout(param_row)
                # Store the widget to extract the value of later
                self.preprocess_boxes[name]["params"][param_name] = widget

            # Add the group box to the inner layout
            self.inner_layout.addWidget(group_box)
        # Create a layout for the order
        self.order_widget = QWidget()
        self.order_layout = QGridLayout()
        self.order_layout.setAlignment(qtpy.QtCore.Qt.AlignTop)
        # Add text box to show current order of preprocessing
        self.order_layout.addWidget(self.order_label, 0, 0, 1, 1)
        self.order_layout.addWidget(self.preprocess_order, 0, 1, 1, 3)
        self.order_widget.setLayout(self.order_layout)
        self.inner_layout.addWidget(self.order_widget)
        # Create separate layout for buttons to be cleaner
        self.btn_widget = QWidget()
        self.btn_layout = QGridLayout()
        # Add preview button
        self.preview_btn = QPushButton("Apply to Slice")
        self.preview_btn.clicked.connect(partial(self.on_click_run, run_on_slice=True))
        self.preview_btn.setToolTip(
            format_tooltip(
                "Apply the selected preprocessing options to the current slice of the currently selected image (or first image layer if none selected)."
            )
        )
        self.btn_layout.addWidget(self.preview_btn, 0, 0, 1, 1)
        # Add a run button to apply the preprocessing entirely
        self.prep_run_btn = QPushButton("Apply to Stack")
        self.prep_run_btn.clicked.connect(
            partial(self.on_click_run, run_on_slice=False)
        )
        self.prep_run_btn.setToolTip(
            format_tooltip("""
Apply the selected preprocessing options to the entire stack of the currently selected image (or first image layer if none selected).
NOTE: This will run the computation locally and return an array in-memory, so be careful with larger images and/or expensive preprocessing.
NOTE: The result is just for visualization! Only the original image will be used in the Nextflow pipeline.
                """)
        )
        self.btn_layout.addWidget(self.prep_run_btn, 0, 1, 1, 1)
        # Add some draft buttons for preprocessing sets
        self.save_set_btn = QPushButton("Save preprocessing set")
        self.save_set_btn.clicked.connect(self.on_click_preprocess_save)
        self.btn_layout.addWidget(self.save_set_btn, 1, 0, 1, 1)
        self.no_preprocess_btn = QPushButton("Add 'No preprocessing'")
        self.no_preprocess_btn.clicked.connect(self.on_click_no_preprocess)
        self.no_preprocess_btn.setToolTip(
            format_tooltip(
                "Add a 'No preprocessing' set to the saved sets, allowing the pipeline to run once with the raw image alongside any other saved preprocessing sets."
            )
        )
        self.btn_layout.addWidget(self.no_preprocess_btn, 1, 1, 1, 1)
        self.view_sets_btn = QPushButton("View saved sets (0)")
        self.view_sets_btn.clicked.connect(self.on_click_preprocess_view)
        self.btn_layout.addWidget(self.view_sets_btn, 2, 0, 1, 1)
        self.clear_sets_btn = QPushButton("Clear saved sets")
        self.clear_sets_btn.clicked.connect(self.on_click_preprocess_clear)
        self.btn_layout.addWidget(self.clear_sets_btn, 2, 1, 1, 1)
        # Set the layout for the widget
        self.btn_widget.setLayout(self.btn_layout)
        self.inner_layout.addWidget(self.btn_widget)

    def on_click_preprocess(self, name: str):
        # Callback for when a preprocess method is selected
        def cb():
            # Get the box to check if it is checked
            group_box = self.preprocess_boxes[name]["box"]
            checked = group_box.isChecked()
            order = self.preprocess_order.text()
            if order == self.init_order:
                order = name
                self.order_list = [name]
            else:
                self.order_list = order.split("->")
                # If checked, add to the start of the list
                if checked:
                    self.order_list.append(name)
                else:
                    self.order_list.remove(name)
                # Handle when all are unchecked
                if len(self.order_list) == 0:
                    order = self.init_order
                else:
                    order = "->".join(self.order_list)
            self.preprocess_order.setText(order)

        # Return the callback
        return cb

    def on_click_run(self, run_on_slice: bool = False):
        # Callback for when the preview button is clicked
        # First check if we are able to preview
        if self.preprocess_order.text() == self.init_order:
            show_error(
                "No preprocessing methods selected! Please select at least one preprocessing method to preview.",
            )
            return
        if len(self.viewer.layers) == 0:
            show_error(
                "No image layers available! Please load an image layer to preview the preprocessing effect on.",
            )
            return
        if not run_on_slice:
            confirm = ConfirmDialog(
                parent=self,
                title="Preview Stack",
                text="This will load the entire selected image stack into memory and apply preprocessing.",
                informative_text=(
                    "For large images, this may consume significant memory or "
                    "cause the application to become unresponsive.\n\n"
                    "Are you sure you want to continue?"
                ),
            )
            if not confirm.exec():
                return
        # Extract the options from the UI elements
        options = self.extract_options()
        # Get the selected image
        if isinstance(self.viewer.layers.selection.active, napari.layers.Image):
            layer = self.viewer.layers.selection.active
        else:
            # NOTE: Use -1 as that's top of the list?
            layer = [
                layer
                for layer in self.viewer.layers
                if isinstance(layer, napari.layers.Image)
            ][0]
        # Extract just the slice of the data
        data = layer.data
        if data.ndim == 3:
            # Get the current slice
            if run_on_slice:
                image = data[self.viewer.dims.current_step[0]]
                # As the preview is for 2D only, remap 3D-specific options to 2D if needed
                for option in options:
                    if option["name"] == "Filter":
                        footprint = option["params"]["footprint"]
                        if footprint == "cube":
                            option["params"]["footprint"] = "square"
                        elif footprint == "ball":
                            option["params"]["footprint"] = "disk"
                        # Show info if changed
                        if footprint != option["params"]["footprint"]:
                            show_info(
                                f"Changed Filter footprint to {option['params']['footprint']} from {footprint} for 2D preview."
                            )
            else:
                image = data
        else:
            # Convert to numpy?
            image = data
        # Extract blocksize for rescaling if downsampling used
        # This will be the corrected blocksize based on preview/run and input data shape
        blocksize = get_downsample_factor(options)
        # Apply the preprocessing and show the result
        # Convert to numpy array in case it's dask
        image = run_preprocess(np.array(image), options)
        prep_str = get_params_str(options)
        # Add metadata to skip file path checks in plugin
        self.viewer.add_image(
            data=image,
            name=f"{layer.name}_{prep_str}",
            metadata={
                "preprocess": True,
                "downsample_blocksize": blocksize,
            },
            scale=layer.scale[-image.ndim :],
        )
        # Switch focus back to the original layer
        self.viewer.layers.selection.active = layer

    def extract_options(self) -> None | list[dict]:
        # Shortcut for when no postprocessing has been done
        if self.order_list is None:
            return
        # Extract the options from the UI elements
        options = []
        # Loop over the specified order
        for name in self.order_list:
            # Get the method and the widget dict
            method_dict = self.preprocess_methods[name]
            widget_dict = self.preprocess_boxes[name]
            # Create the sub-dict for the method
            option_dict = {"name": name, "params": {}}
            for param_name, widget in widget_dict["params"].items():
                # Get the default dtype to cast the value back
                dtype = type(method_dict["params"][param_name]["default"])
                # Extract the value based on the widget type
                if isinstance(widget, list):
                    # Multiple spinboxes/lineedits for a list/tuple param
                    internal_dtype = type(method_dict["params"][param_name]["default"][0])
                    # NOTE: We always cast to list to avoid '!!python/tuple' pyyaml tag
                    # As this cannot be loaded by the yaml.safe_load function
                    option_dict["params"][param_name] = [
                        internal_dtype(w.value() if isinstance(w, (QSpinBox, QDoubleSpinBox)) else w.text())
                        for w in widget
                    ]
                elif isinstance(widget, QCheckBox):
                    option_dict["params"][param_name] = widget.isChecked()
                elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                    option_dict["params"][param_name] = dtype(widget.value())
                elif isinstance(widget, QLineEdit):
                    option_dict["params"][param_name] = dtype(widget.text())
                elif isinstance(widget, QComboBox):
                    option_dict["params"][param_name] = dtype(widget.currentText())
            # Add the method dict to the options list
            options.append(option_dict)
        return options

    def get_all_options(self):
        if len(self.preprocess_sets) > 0:
            # If every set is empty (no-op), treat as no preprocessing so the
            # pipeline skips preprocessImage entirely.
            if all(not s for s in self.preprocess_sets):
                return None
            res = self.preprocess_sets
            extras = self.extract_options()
            if extras is not None:
                show_warning(
                    "You've selected preprocessing options but not saved them while using sets; they will be ignored."
                )
        else:
            # Need to extract options and wrap into a list to align with sets above
            res = self.extract_options()
            res = [res] if res is not None and len(res) > 0 else None
        # Now check all images are compatible with the options
        self.check_all_images(prep_params=res)
        return res

    def check_all_images(self, prep_params):
        # Skip if no preprocessing
        if prep_params is None:
            return
        # Get all image layers
        img_layers = [
            i for i in self.viewer.layers if isinstance(i, napari.layers.Image)
        ]
        # Check each param set against each image layer;
        # empty sets (no preprocessing) short-circuit inside run_preprocess.
        for layer in img_layers:
            for d in prep_params:
                run_preprocess(img=layer.data, methods=d, only_check=True)

    def on_click_no_preprocess(self):
        """Save a 'No preprocessing' sentinel (empty list) as a preprocessing set."""
        if any(not s for s in self.preprocess_sets):
            show_warning("A 'No preprocessing' set already exists!")
            return
        self.preprocess_sets.append([])
        self._reset_preprocess()
        self._update_viewsets_btn()
        show_info("Added a 'No preprocessing' set!")

    def on_click_preprocess_save(self):
        current_options = self.extract_options()
        if current_options is None or len(current_options) == 0:
            show_error(
                "No preprocessing methods selected! Please select at least one preprocessing method to save a set.",
            )
            return
        # Add the current set to the list of sets
        self.preprocess_sets.append(current_options)
        # Reset the order list, order text, and all preprocessing options/checkboxes
        self._reset_preprocess()
        self._update_viewsets_btn()
        show_info("Saved the current preprocessing set!")

    def _reset_preprocess(self):
        self.preprocess_order.setText(self.init_order)
        self.order_list = None
        for name, widget_dict in self.preprocess_boxes.items():
            widget_dict["box"].setChecked(False)
            for param_name, widget in widget_dict["params"].items():
                param_dict = self.preprocess_methods[name]["params"][param_name]
                default = param_dict["default"]
                if isinstance(widget, list):
                    for w, val in zip(widget, default):
                        if isinstance(w, (QSpinBox, QDoubleSpinBox)):
                            w.setValue(val)
                        elif isinstance(w, QLineEdit):
                            w.setText(str(val))
                elif isinstance(widget, QCheckBox):
                    widget.setChecked(False)
                elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                    widget.setValue(default)
                elif isinstance(widget, QLineEdit):
                    widget.setText(str(default))
                elif isinstance(widget, QComboBox):
                    widget.setCurrentIndex(param_dict["values"].index(default))

    def _update_viewsets_btn(self):
        count = len(self.preprocess_sets)
        self.view_sets_btn.setText(f"View saved sets ({count})")

    def on_click_preprocess_view(self):
        display_text = ""
        if len(self.preprocess_sets) == 0:
            display_text = "No saved preprocessing sets!"
        else:
            for i, pp_set in enumerate(self.preprocess_sets):
                display_text += f"Set {i + 1}:\n"
                if not pp_set:
                    display_text += "  No preprocessing\n"
                else:
                    for pp in pp_set:
                        display_text += f"  {pp['name']}:\n"
                        for param, value in pp["params"].items():
                            display_text += f"    {param}: {value}\n"
                display_text += "\n"
        # Create a dialog to display the text
        self.preprocess_set_popout = PreprocessSetWindow(
            self, preprocess_txt=display_text
        )
        self.preprocess_set_popout.show()

    def on_click_preprocess_clear(self):
        self.preprocess_sets = []
        self._update_viewsets_btn()
        show_info("Cleared all saved preprocessing sets!")

    def get_config_params(self, params):
        preprocess_params = params.get("preprocess")
        if preprocess_params is not None:
            return preprocess_params
        return False

    def load_config(self, config):
        if config:
            self.preprocess_sets = config
        else:
            self.preprocess_sets = []
        self._update_viewsets_btn()
        self._reset_preprocess()
        if config and len(config) == 1:
            # Single non-empty set: reflect it in the UI so the user can see
            self._load_options_into_ui(config[0])
            self.preprocess_sets = []
            self._update_viewsets_btn()

    def _load_options_into_ui(self, options: list[dict]):
        """Populate the UI checkboxes and parameter widgets from a single preprocessing set."""
        self.order_list = []
        for step in options:
            name = step["name"]
            params = step["params"]
            self.preprocess_boxes[name]["box"].setChecked(True)
            for param_name, value in params.items():
                widget = self.preprocess_boxes[name]["params"][param_name]
                if isinstance(widget, list):
                    for w, val in zip(widget, value):
                        if isinstance(w, (QSpinBox, QDoubleSpinBox)):
                            w.setValue(val)
                        elif isinstance(w, QLineEdit):
                            w.setText(str(val))
                elif isinstance(widget, QCheckBox):
                    widget.setChecked(bool(value))
                elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                    widget.setValue(value)
                elif isinstance(widget, QComboBox):
                    idx = widget.findText(str(value))
                    if idx != -1:
                        widget.setCurrentIndex(idx)
                elif isinstance(widget, QLineEdit):
                    widget.setText(str(value))
            self.order_list.append(name)
        if self.order_list:
            self.preprocess_order.setText("->".join(self.order_list))


class PreprocessSetWindow(QDialog):
    def __init__(self, parent=None, preprocess_txt: str = ""):
        super().__init__(parent)

        # Set the layout
        self.layout = QVBoxLayout()
        # Set the window title
        self.setWindowTitle("Preprocess Sets")
        self.set_text = QPlainTextEdit()
        # Make the text selectable, but not editable
        self.set_text.setReadOnly(True)
        self.set_text.setPlainText(preprocess_txt)
        self.layout.addWidget(self.set_text)
        self.setLayout(self.layout)
