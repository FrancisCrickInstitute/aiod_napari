import napari
from aiod_registry import TASK_NAMES
from qtpy.QtWidgets import (
    QGridLayout,
    QLayout,
    QRadioButton,
    QWidget,
)

from aiod_napari.widget_classes import SubWidget


class TaskWidget(SubWidget):
    _name = "task"

    def __init__(
        self,
        viewer: napari.Viewer,
        parent: QWidget | None = None,
        layout: QLayout = QGridLayout,
        **kwargs,
    ):
        super().__init__(
            viewer=viewer,
            title="Segmentation Task",
            parent=parent,
            layout=layout,
            tooltip="Select the organelle you want to segment. The models available will change depending on the organelle selected.",
            **kwargs,
        )

    def create_box(self, variant: str | None = None):
        """
        Create the box for selecting the task (i.e. organelle) to segment.
        """
        # Define and set the buttons for the different tasks
        # With callbacks to change other options accoridngly
        self.task_buttons = {}
        for name, label in TASK_NAMES.items():
            btn = QRadioButton(label)
            btn.setEnabled(True)
            btn.setChecked(False)
            btn.clicked.connect(self.on_click_task)
            self.inner_layout.addWidget(btn)
            self.task_buttons[name] = btn

    def on_click_task(self):
        """
        Callback for when a task button is clicked.

        Updates the model box to show only the models available for the selected task.
        """
        # Find out which button was pressed
        for task_name, task_btn in self.task_buttons.items():
            if task_btn.isChecked():
                self.parent.selected_task = task_name
        # Update the model box for the selected task
        self.parent.subwidgets["model"].update_model_box(self.parent.selected_task)

    def load_config(self, config: str):
        task = config
        if task not in TASK_NAMES:
            raise ValueError(f"Task {task} not recognised.")
        for task_name, task_btn in self.task_buttons.items():
            if task_name == task:
                task_btn.setChecked(True)
                self.on_click_task()

    def get_config_params(self, params):
        return params.get("task")
