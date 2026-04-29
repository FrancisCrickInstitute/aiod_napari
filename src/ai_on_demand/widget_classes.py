from os import environ
import shlex
import shutil
import string
import subprocess
from abc import abstractmethod
from pathlib import Path
from typing import Optional

import napari
import qtpy.QtCore
import yaml
from aiod_registry import load_manifests
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info
from npe2 import PluginManager
from qtpy.QtGui import QPixmap
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLayout,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ai_on_demand.qcollapsible import QCollapsible
from ai_on_demand.utils import format_tooltip, get_plugin_cache, load_settings


class MainWidget(QWidget):

    instances = {}

    def __init__(
        self,
        napari_viewer: napari.Viewer,
        title: str,
        tooltip: Optional[str] = None,
    ):
        super().__init__()
        pm = PluginManager.instance()
        self.all_manifests = load_manifests(filter_access=True)
        self.plugin_settings = load_settings()

        MainWidget.instances[title] = self

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

    @classmethod
    def refresh_instances(cls, instances_to_refresh):
        for instance in instances_to_refresh:
            if instance in cls.instances.keys():
                cls.instances[instance].get_manifests()

    def get_manifests(self):
        # Re-retrieve manifests, including cache directory if available
        cache_dir = None
        nxf_widget = self.subwidgets.get("nxf")
        if nxf_widget:
            cache_dir = getattr(nxf_widget, "nxf_store_dir", None)
        self.all_manifests = load_manifests(
            filter_access=True, cache_dir=cache_dir
        )
        self.subwidgets[
            "model"
        ].refresh_ui()  # can add further configuration of which subwidget to refresh

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
        variant: Optional[str] = None,
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
    def create_box(self):
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
    def get_settings(self) -> Optional[dict]:
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
    def get_config_params(self, params: dict) -> Optional[dict]:
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


class BaseNxfWidget(SubWidget):
    """
    Abstract base class for all Nextflow pipeline sub-widgets.

    Provides the shared UI (cache settings, profile selector, run button,
    progress bar) and shared behaviour (directory management, cache
    inspection/clearing, progress bar helpers, run orchestration).

    Concrete subclasses must implement the five abstract pipeline methods
    and the ``_create_variant_ui`` hook.
    """

    _name = "nxf"

    config_ready = qtpy.QtCore.Signal()
    finetuned_model_ready = qtpy.QtCore.Signal(str)

    def __init__(
        self,
        viewer: napari.Viewer,
        parent: Optional[QWidget] = None,
        layout: QLayout = QGridLayout,
        **kwargs,
    ):
        # Must be set before super().__init__ because create_box is called
        # inside SubWidget.__init__ via the constructor chain.
        self.nxf_repo = (
            str(Path(environ["AIOD_NXF_REPO"]))
            if "AIOD_NXF_REPO" in environ
            else "FrancisCrickInstitute/Segment-Flow"
        )
        self.setup_nxf_dir_cmd()

        super().__init__(
            viewer=viewer,
            title="Run Pipeline",
            parent=parent,
            layout=layout,
            tooltip="""
Allows for the computational pipeline to be triggered, with different additional options depending on the main widget selected.
The profile determines where the pipeline is run.
""",
            **kwargs,
        )

        self.nxf_cmd = None
        self.nxf_params = None

    @abstractmethod
    def check_pipeline(self):
        """Validate that all required inputs are present before running."""
        pass

    @abstractmethod
    def setup_pipeline(self) -> tuple:
        """
        Build the Nextflow command and params dict.

        Must return ``(nxf_cmd, nxf_params, proceed, img_paths)``.
        """
        pass

    @abstractmethod
    def _pipeline_start(self):
        """Called when the pipeline thread starts."""
        pass

    @abstractmethod
    def _pipeline_finish(self):
        """Called when the pipeline thread finishes successfully."""
        pass

    @abstractmethod
    def _pipeline_fail(self, exc):
        """Called when the pipeline thread raises an exception."""
        pass

    @abstractmethod
    def cancel_pipeline(self):
        """Cancel the currently running pipeline process."""
        pass

    def _create_variant_ui(self):
        """
        Optionally add variant-specific widgets into the pipeline settings
        group box.  Called at the end of ``create_box``, before the run
        button and progress bar are added.

        Override in subclasses that need extra controls.
        The default implementation adds nothing.
        """
        pass

    def load_settings(self):
        """Load profile and base-dir settings saved from a previous session."""
        if not self.parent.plugin_settings:
            return
        if "nxf" in self.parent.plugin_settings:
            settings = self.parent.plugin_settings["nxf"]
            if "profile" in settings:
                idx = self.nxf_profile_box.findText(settings["profile"])
                if idx != -1:
                    self.nxf_profile_box.setCurrentIndex(idx)
            if "base_dir" in settings:
                nxf_base_dir = Path(settings["base_dir"])
                self.nxf_dir_text.setText(str(nxf_base_dir))
                self.setup_nxf_dir_cmd(base_dir=Path(nxf_base_dir))

    def get_settings(self) -> dict:
        """Return profile and base-dir settings for persistence."""
        return {
            "base_dir": str(self.nxf_base_dir),
            "profile": self.nxf_profile_box.currentText(),
        }

    def get_config_params(self, params):
        """Return the shared config params; subclasses extend this."""
        return {
            "base_dir": str(self.nxf_base_dir),
            "profile": self.nxf_profile_box.currentText(),
        }

    def load_config(self, config):
        """Apply a saved config; subclasses extend this for extra keys."""
        profile_index = self.nxf_profile_box.findText(config["profile"])
        if profile_index != -1:
            self.nxf_profile_box.setCurrentIndex(profile_index)
        base_dir = config["base_dir"]
        if self.nxf_dir_text.text() != base_dir:
            self.nxf_dir_text.setText(base_dir)
            self.setup_nxf_dir_cmd(base_dir=Path(base_dir))

    def setup_nxf_dir_cmd(self, base_dir: Optional[Path] = None):
        if base_dir is not None:
            self.nxf_base_dir = base_dir
        else:
            self.nxf_base_dir = Path.home() / ".nextflow" / "aiod"
        self.nxf_base_dir.mkdir(parents=True, exist_ok=True)
        self.nxf_store_dir = self.nxf_base_dir / "aiod_cache"
        self.nxf_store_dir.mkdir(parents=True, exist_ok=True)
        self.nxf_base_cmd = (
            f"nextflow -log '{str(self.nxf_base_dir / 'nextflow.log')}' "
        )
        self.img_list_fpath = self.nxf_store_dir / "all_img_paths.csv"
        self.nxf_work_dir = self.nxf_base_dir / "work"
        self.nxf_work_dir.mkdir(parents=True, exist_ok=True)

    def create_box(self):
        # ---- Cache settings group ----
        self.cache_box = QGroupBox("Cache Settings")
        self.cache_box.setToolTip(
            format_tooltip(
                "Settings for the AIoD/Nextflow cache for storing models and results."
            )
        )
        self.cache_layout = QGridLayout()
        self.cache_box.setLayout(self.cache_layout)

        self.nxf_dir_label = QLabel("Base directory:")
        base_dir_tooltip = "Select the base directory to store the Nextflow cache (i.e. all models & results) in."
        self.nxf_dir_label.setToolTip(format_tooltip(base_dir_tooltip))
        self.nxf_dir_text = QLabel(str(self.nxf_base_dir))
        self.nxf_dir_text.setWordWrap(True)
        self.nxf_dir_text.setToolTip(
            format_tooltip("The selected base directory.")
        )
        self.nxf_dir_text.setMaximumWidth(400)

        self.nxf_dir_btn = QPushButton("Change")
        self.nxf_dir_btn.clicked.connect(self.on_click_base_dir)
        self.nxf_dir_btn.setToolTip(format_tooltip(base_dir_tooltip))

        self.nxf_dir_inspect_btn = QPushButton("Inspect cache")
        self.nxf_dir_inspect_btn.clicked.connect(self.on_click_inspect_cache)
        self.nxf_dir_inspect_btn.setToolTip(
            format_tooltip(
                """
Open the base directory in the file explorer to inspect the cache.

Note that 'opening' won't do anything, this is just to see what files are present.
"""
            )
        )

        self.nxf_dir_clear_btn = QPushButton("Clear cache")
        self.nxf_dir_clear_btn.clicked.connect(self.on_click_clear_cache)
        self.nxf_dir_clear_btn.setToolTip(
            format_tooltip(
                "Clear the cache of all models and results. WARNING: This will remove all models and results from the cache."
            )
        )

        self.cache_layout.addWidget(self.nxf_dir_label, 0, 0, 1, 2)
        self.cache_layout.addWidget(self.nxf_dir_text, 0, 2, 1, 3)
        self.cache_layout.addWidget(self.nxf_dir_btn, 0, 5, 1, 1)
        self.cache_layout.addWidget(self.nxf_dir_inspect_btn, 1, 0, 1, 3)
        self.cache_layout.addWidget(self.nxf_dir_clear_btn, 1, 3, 1, 3)

        self.inner_layout.addWidget(self.cache_box, 0, 0, 1, 2)

        # ---- Pipeline settings group ----
        self.pipeline_box = QGroupBox("Pipeline Settings")
        self.pipeline_box.setToolTip(
            format_tooltip("Settings for the Segment-Flow pipeline itself.")
        )
        self.pipeline_layout = QGridLayout()
        self.pipeline_box.setLayout(self.pipeline_layout)

        self.nxf_profile_label = QLabel("Execution profile:")
        self.nxf_profile_label.setToolTip(
            format_tooltip("Select the execution profile to use.")
        )
        self.nxf_profile_box = QComboBox()
        config_dir = Path(__file__).parent / "Segment-Flow" / "profiles"
        avail_confs = [str(i.stem) for i in config_dir.glob("*.conf")]
        avail_confs.sort()
        if len(avail_confs) == 0:
            raise FileNotFoundError(
                f"No Nextflow profiles found in {config_dir}!"
            )
        self.nxf_profile_box.addItems(avail_confs)
        self.nxf_profile_box.setFocusPolicy(
            qtpy.QtCore.Qt.FocusPolicy.StrongFocus
        )
        self.pipeline_layout.addWidget(self.nxf_profile_label, 0, 0)
        self.pipeline_layout.addWidget(self.nxf_profile_box, 0, 1)

        # Let each subclass inject its own controls into the pipeline group
        self._create_variant_ui()

        self.inner_layout.addWidget(self.pipeline_box, 1, 0, 1, 2)

        # ---- Run button ----
        self.nxf_run_btn = QPushButton("Run Pipeline!")
        self.nxf_run_btn.clicked.connect(self.run_pipeline)
        self.nxf_run_btn.setToolTip(
            format_tooltip(
                "Run the pipeline with the chosen organelle(s), model, and images."
            )
        )
        self.inner_layout.addWidget(self.nxf_run_btn, 2, 0, 1, 2)

        # ---- Progress bar ----
        pbar_layout = QHBoxLayout()
        self.pbar = QProgressBar()
        self.pbar_label = QLabel("Progress: [--:--]")
        self.pbar_label.setToolTip(
            format_tooltip("Shows [elapsed<remaining] time for current run.")
        )
        pbar_layout.addWidget(self.pbar_label)
        pbar_layout.addWidget(self.pbar)
        self.inner_layout.addLayout(pbar_layout, 5, 0, 1, 1)
        self.tqdm_pbar = None

    def run_pipeline(self):
        self.check_pipeline()
        pipeline_result = self.setup_pipeline()
        nxf_cmd, nxf_params, proceed, img_paths = pipeline_result

        if not proceed:
            return

        # Let the subclass do any pre-run work (e.g. store image paths,
        # inject postprocess flag, persist settings).
        self._pre_run_hook(nxf_params, img_paths)

        if self.nxf_work_dir is not None:
            nxf_cmd += f" -w {self.nxf_work_dir}"
        nxf_cmd += f" -profile {self.nxf_profile_box.currentText()}"
        nxf_params["param_hash"] = self.parent.run_hash
        nxf_params_fpath = (
            self.nxf_store_dir / f"nxf_params_{self.parent.run_hash}.yml"
        )
        with open(nxf_params_fpath, "w") as f:
            yaml.dump(nxf_params, f)
        nxf_cmd += f" -params-file {nxf_params_fpath}"

        @thread_worker(
            connect={
                "started": self._pipeline_start,
                "returned": self._pipeline_finish,
                "errored": self._pipeline_fail,
            }
        )
        def _run_pipeline(nxf_cmd: str):
            self.process = subprocess.Popen(
                ["/bin/sh", "-l", "-c"] + shlex.split(shlex.quote(nxf_cmd)),
                shell=False,
                cwd=Path.home(),
            )
            self.process.wait()
            if self.process.returncode != 0:
                raise RuntimeError

        _run_pipeline(nxf_cmd)
        self.config_ready.emit()
        self.nxf_params = nxf_params

    def _pre_run_hook(self, nxf_params: dict, img_paths):
        """
        Called just before the Nextflow command is finalised and dispatched.
        Override in subclasses to inject variant-specific pre-run steps.
        """
        pass

    def _reset_btns(self):
        self.nxf_run_btn.setText("Run Pipeline!")
        self.nxf_run_btn.setEnabled(True)
        self._remove_cancel_btn()

    def _add_cancel_btn(self, cancel_slot):
        """
        Split the run button in half and add a cancel button beside it.
        ``cancel_slot`` is the callable to connect to the cancel button.
        """
        idx = self.inner_widget.layout().indexOf(self.nxf_run_btn)
        row, col, rowspan, colspan = (
            self.inner_widget.layout().getItemPosition(idx)
        )
        self.orig_colspan = colspan
        self.cancel_btn = QPushButton("Cancel Pipeline")
        self.cancel_btn.clicked.connect(cancel_slot)
        self.cancel_btn.setToolTip("Cancel the currently running pipeline.")
        new_colspan = colspan // 2 if colspan > 1 else 1
        self.inner_widget.layout().addWidget(
            self.nxf_run_btn, row, col, rowspan, new_colspan
        )
        self.inner_widget.layout().addWidget(
            self.cancel_btn, row, col + new_colspan, rowspan, new_colspan
        )

    def _remove_cancel_btn(self):
        self.inner_widget.layout().removeWidget(self.cancel_btn)
        self.cancel_btn.setParent(None)
        idx = self.inner_widget.layout().indexOf(self.nxf_run_btn)
        row, col, rowspan, _ = self.inner_widget.layout().getItemPosition(idx)
        self.inner_widget.layout().addWidget(
            self.nxf_run_btn, row, col, rowspan, self.orig_colspan
        )

    def reset_progress_bar(self):
        self.pbar.setValue(0)
        if self.tqdm_pbar is not None:
            self.tqdm_pbar.close()
        self.pbar_label.setText("Progress: [--:--]")

    def on_click_base_dir(self):
        base_dir = QFileDialog.getExistingDirectory(
            self, caption="Select directory to store cache", directory=None
        )
        if base_dir == "":
            return
        new_dir_name = Path(base_dir).name.replace(" ", "_")
        base_dir = Path(base_dir).parent / new_dir_name
        self.nxf_dir_text.setText(str(base_dir))
        self.setup_nxf_dir_cmd(base_dir=base_dir)

    def on_click_inspect_cache(self):
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.AnyFile)
        dialog.setDirectory(str(self.nxf_base_dir))
        dialog.exec()

    def on_click_clear_cache(self):
        prompt_window = QMessageBox()
        prompt_window.setIcon(QMessageBox.Question)
        prompt_window.setText("Are you sure you want to clear the cache?")
        prompt_window.setInformativeText(
            "This will remove all models and results from the cache."
        )
        mask_dirs = [
            i
            for i in (self.nxf_base_dir / "aiod_cache").rglob("*")
            if i.is_dir() and i.name.endswith("_masks")
        ]
        num_masks = sum(
            len(list(mask_dir.glob("*.rle"))) for mask_dir in mask_dirs
        )
        num_configs = len(
            list((self.nxf_base_dir / "aiod_cache").glob("nxf_params_*.yml"))
        )
        chkpt_dirs = [
            i
            for i in (self.nxf_base_dir / "aiod_cache").rglob("*")
            if i.is_dir() and i.name == "checkpoints"
        ]
        num_chkpts = sum(
            len(list(chkpt_dir.glob("*"))) for chkpt_dir in chkpt_dirs
        )
        msg = (
            f"Your cache ({self.nxf_base_dir}) contains the following files:\n"
            + "\n".join(
                [
                    f"{num_masks} masks",
                    f"{num_chkpts} model checkpoints (or related files)",
                    f"{num_configs} Nextflow parameter files",
                ]
            )
        )
        prompt_window.setDetailedText(msg)
        prompt_window.setWindowTitle("Clear cache")
        clear_models = prompt_window.addButton(
            "Clear models only", QMessageBox.ButtonRole.ActionRole
        )
        clear_masks = prompt_window.addButton(
            "Clear masks only", QMessageBox.ButtonRole.ActionRole
        )
        clear_all = prompt_window.addButton(
            "Clear all", QMessageBox.ButtonRole.ActionRole
        )
        cancel = prompt_window.addButton(QMessageBox.StandardButton.Cancel)
        prompt_window.setDefaultButton(cancel)
        prompt_window.exec()
        clicked_btn = prompt_window.clickedButton()
        if (
            clicked_btn == QMessageBox.StandardButton.Close
            or clicked_btn == cancel
        ):
            return
        elif clicked_btn == clear_models:
            for chkpt_dir in chkpt_dirs:
                shutil.rmtree(chkpt_dir)
        elif clicked_btn == clear_masks:
            for mask_dir in mask_dirs:
                shutil.rmtree(mask_dir)
        elif clicked_btn == clear_all:
            shutil.rmtree(self.nxf_base_dir)
            self.setup_nxf_dir_cmd(base_dir=self.nxf_base_dir)
