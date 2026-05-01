from typing import Optional

import napari
import yaml
from datetime import datetime
from qtpy.QtWidgets import (
    QWidget,
    QLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QLineEdit,
    QFileDialog,
)
from ai_on_demand.utils import format_tooltip, get_plugin_cache
from ai_on_demand.widget_classes import SubWidget


class ConfigWidget(SubWidget):
    _name = "config"

    def __init__(
        self,
        viewer: napari.Viewer,
        variant: Optional[str] = None,
        parent: Optional[QWidget] = None,
        layout: QLayout = QGridLayout,
        **kwargs,
    ):
        super().__init__(
            viewer=viewer,
            title="Project Configuration",
            parent=parent,
            layout=layout,
            tooltip="""
Load and save all parameters of the plugin to a config file, to easily reproduce results or share settings with others and easily work on multiple projects and across sessions.""",
            **kwargs,
        )

    def create_box(self):
        """
        Create the box for loading and saving configurations.
        """
        # Create box for the custom config settings
        self.config_box = QGroupBox("Config Settings")
        self.config_box.setToolTip(
            format_tooltip(
                "Save a config containing all current UI settings, to automatically fill in all options in the plugin when loaded."
            )
        )
        self.config_layout = QGridLayout()
        self.config_box.setLayout(self.config_layout)

        self.config_name_label = QLabel("Config name:")
        self.config_name_input = QLineEdit(
            placeholderText="e.g. project_config_YYYY-MM-DDTHH:MM"
        )

        nxf_base_dir = getattr(
            self.parent.subwidgets.get("nxf"), "nxf_base_dir", None
        )
        if nxf_base_dir:
            self.save_dir = nxf_base_dir / "project_configs"
        else:
            self.save_dir, _ = get_plugin_cache()
        self.save_dir_label = QLabel(f"Save directory:")
        self.save_dir_text = QLabel(str(self.save_dir))
        self.save_dir_text.setWordWrap(True)

        self.load_config_button = QPushButton("Load Config")
        self.load_config_button.clicked.connect(self.on_load_config)

        self.save_dir_button = QPushButton("Change")
        self.save_dir_button.clicked.connect(self.on_change_save_dir)

        self.save_config_button = QPushButton("Save Config")
        self.save_config_button.setDisabled(True)
        self.save_config_button.setToolTip(
            format_tooltip(
                "Saving becomes available after running pipeline once"
            )
        )
        self.save_config_button.clicked.connect(self.on_save_config)

        self.save_dir_label.setWordWrap(True)
        self.config_layout.addWidget(self.config_name_label, 0, 0, 1, 1)
        self.config_layout.addWidget(self.config_name_input, 0, 1, 1, 5)
        self.config_layout.addWidget(self.save_dir_label, 1, 0, 1, 2)
        self.config_layout.addWidget(self.save_dir_text, 1, 2, 1, 3)
        self.config_layout.addWidget(self.save_dir_button, 1, 5, 1, 1)
        self.config_layout.addWidget(self.save_config_button, 2, 0, 1, 3)
        self.config_layout.addWidget(self.load_config_button, 2, 3, 1, 3)

        self.inner_layout.addWidget(self.config_box)

    def on_load_config(self):
        config_path_and_filter = QFileDialog.getOpenFileName(
            self,
            "select a config file",
            str(self.save_dir),
            "YAML Files (*.yaml *.yml)",
        )
        config_path = config_path_and_filter[0]
        if config_path:
            with open(config_path, "r") as f:
                config_data = yaml.safe_load(f)

            if config_data:
                self.parent.load_config_file(config_data)

    def enable_save_config(self):
        self.save_config_button.setDisabled(False)

    def on_change_save_dir(self):
        self.save_dir = QFileDialog.getExistingDirectory(
            self,
            caption="Select directory to store project config",
            directory=None,
        )

        if self.save_dir == "":
            return

        self.save_dir_text.setText(str(self.save_dir))

    def on_save_config(self):
        config_name = self.config_name_input.text().strip()
        if not config_name:
            config_name = f"project_config_{datetime.now().isoformat(timespec='minutes')}"

        self.parent.store_config(self.save_dir, config_name)
