import napari
from typing import Optional
from pathlib import Path

from napari.utils.notifications import show_info
from aiod_registry import add_model_local
from napari._qt.qt_resources import QColoredSVGIcon
from qtpy.QtWidgets import (
    QWidget,
    QGridLayout,
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
from ai_on_demand.utils import format_tooltip


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

    def create_box(self):
        self.finetune_box = QGroupBox("Finetune Model")

        self.finetune_layout = QGridLayout()
        self.finetune_box.setLayout(self.finetune_layout)

        self.train_dir = QLineEdit(placeholderText="Train directory")
        self.test_dir = QLineEdit(placeholderText="Test directory (optional)")

        self.finetune_layers = QComboBox()

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

        self.finetune_layout.addWidget(QLabel("Finetune layers: "), 5, 0)
        self.finetune_layout.addWidget(self.finetune_layers, 5, 1, 1, 2)

        self.finetune_layout.addWidget(QLabel("Epochs: "), 6, 0)
        self.finetune_layout.addWidget(self.epochs, 6, 1, 1, 2)

        # Training hyperparameters
        self.learning_rate_label = QLabel("Learning rate:")
        self.learning_rate_label.setToolTip(
            format_tooltip("Learning rate for finetuning optimizer")
        )
        self.learning_rate = QLineEdit(placeholderText="e.g., 0.001")
        self.learning_rate.setText("0.001")
        self.finetune_layout.addWidget(self.learning_rate_label, 8, 0)
        self.finetune_layout.addWidget(self.learning_rate, 8, 1, 1, 2)

        self.weight_decay_label = QLabel("Weight decay:")
        self.weight_decay_label.setToolTip(
            format_tooltip("Weight decay for regularization")
        )
        self.weight_decay = QLineEdit(placeholderText="e.g., 0.0001")
        self.weight_decay.setText("0.0001")
        self.finetune_layout.addWidget(self.weight_decay_label, 9, 0)
        self.finetune_layout.addWidget(self.weight_decay, 9, 1, 1, 2)

        self.use_sgd_label = QLabel("Use SGD optimizer:")
        self.use_sgd_label.setToolTip(
            format_tooltip(
                "Use SGD optimizer instead of default Adam optimizer"
            )
        )
        self.use_sgd = QCheckBox()
        self.use_sgd.setChecked(False)
        self.finetune_layout.addWidget(self.use_sgd_label, 10, 0)
        self.finetune_layout.addWidget(self.use_sgd, 10, 1)

        self.momentum_label = QLabel("Momentum (SGD only):")
        self.momentum_label.setToolTip(
            format_tooltip("Momentum parameter for SGD optimizer")
        )
        self.momentum = QLineEdit(placeholderText="e.g., 0.9")
        self.momentum.setText("0.9")
        self.finetune_layout.addWidget(self.momentum_label, 11, 0)
        self.finetune_layout.addWidget(self.momentum, 11, 1, 1, 2)

        self.finetune_layout.addWidget(QLabel("Finetuned model name: "), 12, 0)
        self.finetune_layout.addWidget(self.model_save_name, 12, 1, 1, 2)

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

        self.finetune_layout.addWidget(self.add_model_btn, 13, 0, 1, 3)

        self.inner_layout.addWidget(self.finetune_box)

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

    def update_finetune_layers(self, task_model_verson):
        self.finetune_layers.clear()
        version_data = self.parent.subwidgets["model"].model_version_tasks[
            task_model_verson
        ]
        # save for later use when saving the model
        self.finetuning_meta_data = dict(version_data.finetuning_meta_data)
        avail_layers = self.finetuning_meta_data["avail_layers"]
        self.finetune_layers.addItems(avail_layers)
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

        add_model_local(
            model_name,
            model_task,
            model_save_fpath,
            manifest_name,
            self.finetuning_meta_data,
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
