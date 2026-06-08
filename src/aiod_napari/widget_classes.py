from abc import abstractmethod
from pathlib import Path
import string
from typing import Optional
import yaml

import napari
from napari.utils.notifications import show_info
from npe2 import PluginManager
from qtpy.QtWidgets import (
    QWidget,
    QScrollArea,
    QLayout,
    QGridLayout,
    QLabel,
    QVBoxLayout,
    QFrame,
    QGroupBox,
)
from qtpy.QtGui import QPixmap
import qtpy.QtCore

from aiod_napari.qcollapsible import QCollapsible
from aiod_napari.utils import (
    format_tooltip,
    get_plugin_cache,
)


class MainWidget(QWidget):
    def __init__(
        self,
        napari_viewer: napari.Viewer,
        title: str,
        tooltip: Optional[str] = None,
    ):
        super().__init__()
        pm = PluginManager.instance()
        self.all_manifests = pm.commands.execute("aiod-napari.get_manifests")
        self.plugin_settings = pm.commands.execute("aiod-napari.get_settings")

        self.viewer = napari_viewer
        self.scroll = QScrollArea()

        # Set overall layout for the widget
        self.setLayout(QVBoxLayout())
        self.layout().setAlignment(qtpy.QtCore.Qt.AlignTop)

        # Dictionary to contain all subwidgets
        self.subwidgets = {}

        # A hash to uniquely identify a run
        # Only used to uniquely identify a Nextflow pipeline based on inputs
        self.run_hash = None

        # Add a Crick logo to the widget
        self.logo_label = QLabel()
        logo = QPixmap(
            str(
                Path(__file__).parent
                / "resources"
                / "CRICK_Brandmark_01_transparent.png"
            )
        ).scaledToHeight(100, mode=qtpy.QtCore.Qt.SmoothTransformation)
        self.logo_label.setPixmap(logo)
        self.logo_label.setAlignment(qtpy.QtCore.Qt.AlignCenter)
        self.layout().addWidget(self.logo_label)

        # Widget title to display
        self.title = QLabel(f"AI OnDemand: {title}")
        title_font = self.font()
        title_font.setPointSize(16)
        title_font.setBold(True)
        self.title.setFont(title_font)
        self.title.setAlignment(qtpy.QtCore.Qt.AlignCenter)
        if tooltip is not None:
            self.tooltip = tooltip
            self.title.setToolTip(format_tooltip(tooltip))
        self.layout().addWidget(self.title)

        # Create the widget that will be used to add subwidgets to
        # This is then the widget for the scroll area, to the logo/title is excluded from scrolling
        self.content_widget = QWidget()
        self.content_widget.setLayout(QVBoxLayout())
        self.scroll.setWidgetResizable(True)
        # This is needed to avoid unnecessary spacing when in the ScrollArea
        self.content_widget.setSizePolicy(
            qtpy.QtWidgets.QSizePolicy.Minimum,
            qtpy.QtWidgets.QSizePolicy.Fixed,
        )
        self.content_widget.layout().setAlignment(qtpy.QtCore.Qt.AlignTop)
        self.scroll.setWidget(self.content_widget)
        self.layout().addWidget(self.scroll)

    def register_widget(self, widget: "SubWidget"):
        self.subwidgets[widget._name] = widget

    def store_settings(self):
        # Extract settings for every subwidget that has implemented the get_settings method
        for k, subwidget in self.subwidgets.items():
            settings = subwidget.get_settings()
            if settings is not None:
                self.plugin_settings[k] = settings
        # TODO: Think/check if we want to store anything else
        # Save the settings to the cache
        # As we retrieve everything every time, we can just overwrite the file
        _, settings_path = get_plugin_cache()
        with open(settings_path, "w") as f:
            yaml.dump(self.plugin_settings, f)

    def store_config(self, save_dir, config_name):
        # get next flow config settings from the pipeline param
        nxfWidget = self.subwidgets.get("nxf")
        nxf_params = nxfWidget.nxf_params

        config_settings = {}
        for k, subwidget in self.subwidgets.items():
            if hasattr(subwidget, "get_config_params"):
                config_for_subwidget = subwidget.get_config_params(nxf_params)
                config_settings[k] = config_for_subwidget

        Path(save_dir).mkdir(parents=True, exist_ok=True)
        config_file_path = Path(save_dir) / f"{config_name}.yaml"
        with open(config_file_path, "w") as f:
            yaml.dump(config_settings, f)
        show_info(f"Config saved: {config_file_path}")

    @abstractmethod
    def get_run_hash(self):
        """
        Gather all the parameters from the subwidgets to be used in obtaining a unique hash for a run.
        """
        raise NotImplementedError

    def store_widget_settings(self):
        """
        Store the settings for the widget.
        """
        pass

    def store_subwidget_settings(self):
        """
        Store the settings for the subwidgets.
        """
        for widget in self.subwidgets.values():
            widget.store_settings()

    def load_config_file(self, config: dict):
        """
        Load a config file for the widget.
        """
        for subwidget in self.subwidgets.values():
            if subwidget._name in config:
                subwidget.load_config(config=config[subwidget._name])
        show_info("Configuration loaded successfully.")


class SubWidget(QCollapsible):
    # Define a shorthand name to be used to register the widget
    _name: str = None

    def __init__(
        self,
        viewer: napari.Viewer,
        title: str,
        parent: Optional[QWidget] = None,
        layout: QLayout = QVBoxLayout,
        tooltip: Optional[str] = None,
        **kwargs,
    ):
        """
        Custom widget for the AI OnDemand plugin.

        Controls the subwidgets/modules of the plugin which are used for different meta-plugins.
        Allows for easy changes of style, uniform layout, and better interoperability between other subwidgets.

        Parameters
        ----------
        viewer : napari.Viewer
            Napari viewer object.
        parent : QWidget, optional
            Parent widget, by default None. Allows for easy access to the parent widget and its attributes.
        title : str
            Title of the widget to be displayed.
        layout : QLayout, optional
            Layout to use for the widget. This is the default layout for the subwidget.
        tooltip : Optional[str], optional
            Tooltip to display for the widget, by default None.
        kwargs
            Additional keyword arguments to pass to the QCollapsible widget, such as margins, animation duration, etc.
        """
        super().__init__(
            title=string.capwords(title),
            layout=layout,
            collapsedIcon="▶",
            expandedIcon="▼",
            duration=200,
            margins=(5, 0, 5, 0),
            **kwargs,
        )
        self.viewer = viewer
        self.parent = parent
        self.title = title

        # Set the inner widgets (the things that get collapsed/expanded)
        self.inner_widget = QWidget()
        self.inner_layout = QGridLayout()
        self.inner_layout.setAlignment(qtpy.QtCore.Qt.AlignTop)
        self.inner_layout.setContentsMargins(0, 0, 0, 0)

        # Set the tooltip if given
        if tooltip is not None:
            self.setToolTip(format_tooltip(tooltip))
        # Create the initial widgets/elements
        self.create_box()
        self.inner_widget.setLayout(self.inner_layout)
        # Add the inner widget to the collapsible widget
        self.addWidget(self.inner_widget)
        # Add a divider line to better separate subwidgets
        # NOTE: Currently invisible, but just a spacer
        # btn_colour = self._toggle_btn.palette().button().color().name()  # Tries to get the button colour
        divider_line = QFrame()
        divider_line.setFrameShape(QFrame.HLine)
        # divider_line.setFrameShadow(QFrame.Sunken)
        divider_line.setStyleSheet(
            """
            QFrame[frameShape='4'] {
                border: none;
            }
        """
        )
        # Ensure minimal space taken
        divider_line.setMaximumHeight(1)
        self.content().layout().addWidget(divider_line)

        # If given a parent at creation, add this widget to the parent's layout
        if self.parent is not None:
            # Add to the content widget (i.e. scrollable area)
            self.parent.content_widget.layout().addWidget(self)

        if kwargs.get("expanded", False):
            self.expand(animate=False)

        # Load any previous settings for this widget if available
        self.load_settings()

    @abstractmethod
    def create_box(self, variant: Optional[str] = None):
        """
        Create the box for the subwidget, i.e. all UI elements.
        """
        raise NotImplementedError

    @abstractmethod
    def load_settings(self):
        """
        Load settings for the subwidget.
        """
        pass

    @abstractmethod
    def get_settings(self):
        """
        Get settings for the subwidget.
        """
        pass

    @abstractmethod
    def load_config(self, config: dict):
        """
        Load a specific config and apply to the subwidget.
        """
        pass

    @abstractmethod
    def get_config_params(self, params: dict):
        """
        Gets the config params for the widget
        """
        pass

    def _make_separator(self):
        """
        Create a thin separator line to better separate elements within a subwidget.
        """
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Raised)
        separator.setSizePolicy(
            qtpy.QtWidgets.QSizePolicy.Expanding,
            qtpy.QtWidgets.QSizePolicy.Minimum,
        )
        colour = napari.utils.theme.get_theme("dark").secondary.as_rgb()
        separator.setStyleSheet(
            f"border: 1px solid {colour}; background-color: {colour};"
        )
        return separator

    def _make_groupbox(self, title: str, tooltip: Optional[str] = None):
        group_box = QGroupBox(title)
        if tooltip is not None:
            group_box.setToolTip(format_tooltip(tooltip))
        group_box.setCheckable(False)
        group_layout = QGridLayout()
        group_layout.setAlignment(qtpy.QtCore.Qt.AlignTop)
        group_box.setLayout(group_layout)
        group_box.setContentsMargins(0, 0, 0, 0)
        return group_box
