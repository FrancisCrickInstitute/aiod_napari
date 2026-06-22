from collections import Counter
from pathlib import Path

import aiod_utils.io as aiod_io
import napari
import pandas as pd
import qtpy.QtCore
from napari.layers import Image, Layer
from napari.qt.threading import thread_worker
from bioio_base.dimensions import Dimensions
from qtpy.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QLabel,
    QLayout,
    QPushButton,
    QLabel,
    QLineEdit,
    QFileDialog,
    QWidget,
)

from aiod_napari.io import prepare_bioio_as_napari_layer
from aiod_napari.utils import format_tooltip, get_image_layer_path
from aiod_napari.widget_classes import SubWidget


class DataWidget(SubWidget):
    _name = "data"

    images_loaded = qtpy.QtCore.Signal()

    def __init__(
        self,
        viewer: napari.Viewer,
        parent: QWidget | None = None,
        layout: QLayout = QGridLayout,
        **kwargs,
    ):
        super().__init__(
            viewer,
            "Data Selection",
            parent,
            layout,
            tooltip="""
Select data to be used as input to the model.

Images can also be opened, or dragged into napari as normal. The selection will be updated accordingly. The 'Reset selection' button can be used to clear all images.
""",
            **kwargs,
        )

        # Connect to the viewer to some callbacks
        self.viewer.layers.events.inserted.connect(self.on_layer_added)
        self.viewer.layers.events.removed.connect(self.on_layer_removed)

    def create_box(self, variant: str | None = None):
        # Create empty counter to show image load progress
        self.load_img_counter = 0
        # Create container for image paths
        self.image_path_dict = {}
        # Do a quick check to see if the user has added any images already
        counter = 0
        if self.viewer.layers:
            for img_layer in self.viewer.layers:
                if isinstance(img_layer, Image):
                    img_path = get_image_layer_path(img_layer)
                    try:
                        self.image_path_dict[img_path.stem] = img_path
                        counter += 1
                    # Will fail if no path found
                    except AttributeError:
                        continue

        # If all pre-existing image layers have been added, set loaded flag
        # Set to False if no images, to avoid overriding the all_loaded flag
        if len(self.image_path_dict) == 0:
            self.existing_loaded = False
        elif len(self.image_path_dict) == counter:
            self.existing_loaded = True
        else:
            self.existing_loaded = False
        # Create a button to select individual images from
        self.img_btn = QPushButton("Select image\nfiles")
        self.img_btn.clicked.connect(self.browse_imgs_files)
        self.img_btn.setToolTip(
            format_tooltip(
                "Select individual image files to use as input to the model."
            )
        )
        self.inner_layout.addWidget(self.img_btn, 0, 0)
        # Create a button to navigate to a directory to take images from
        self.dir_btn = QPushButton("Select image\ndirectory")
        self.dir_btn.clicked.connect(self.browse_imgs_dir)
        self.dir_btn.setToolTip(
            format_tooltip(
                "Select folder/directory of images to use as input to the model."
            )
        )
        self.inner_layout.addWidget(self.dir_btn, 0, 1)
        # Create a button to clear selected directory
        self.clear_dir_btn = QPushButton("Reset\nselection")
        self.clear_dir_btn.clicked.connect(self.clear_directory)
        self.clear_dir_btn.setToolTip(
            format_tooltip(
                "Reset selection of images (clears all images in the viewer)."
            )
        )
        self.inner_layout.addWidget(self.clear_dir_btn, 0, 2)
        # Add an output to show the counts
        self.init_file_msg = "No files selected or added to Napari."
        self.img_counts = QLabel(self.init_file_msg)
        self.img_counts.setWordWrap(True)
        self.inner_layout.addWidget(self.img_counts, 1, 0, 1, 3)

        # Axis override row
        axes_label = QLabel("Axes override:")
        axes_label.setToolTip(
            format_tooltip(
                "Manually specify the axis order of loaded images (e.g. ZCYX, CZYX, ZYX). "
                "Use this when the file has no axis metadata and the channel dropdown shows wrong values. "
                "Letters: T=time, Z=depth, C=channel, Y=height, X=width, S=RGB samples."
            )
        )
        self.axes_override_edit = QLineEdit()
        self.axes_override_edit.setPlaceholderText("e.g. ZCYX  (blank = auto)")
        self.axes_override_edit.setToolTip(
            format_tooltip(
                "Enter axis letters matching the number of image dimensions. "
                "Leave blank to let bioio determine axes automatically."
            )
        )
        self.axes_override_btn = QPushButton("Apply")
        self.axes_override_btn.setToolTip(
            format_tooltip(
                "Apply the axes override to all currently loaded image layers "
                "whose number of dimensions matches the length of the axes string."
            )
        )
        self.axes_override_btn.clicked.connect(self.on_apply_axes_override)
        self.inner_layout.addWidget(axes_label, 2, 0)
        self.inner_layout.addWidget(self.axes_override_edit, 2, 1)
        self.inner_layout.addWidget(self.axes_override_btn, 2, 2)
        # Feedback label showing detected dimensions after override
        self.axes_feedback_label = QLabel("")
        self.axes_feedback_label.setWordWrap(True)
        self.inner_layout.addWidget(self.axes_feedback_label, 3, 0, 1, 3)

        # Run the file counter if there are images already loaded
        if len(self.image_path_dict) > 0:
            self.update_file_count()

    def on_layer_added(self, event):
        """
        Triggered whenever there is a new layer added to the viewer.

        Checks if the layer is an image, and if so, adds it to the list of images to process.
        """
        if isinstance(event.value, Image):
            # Extract the underlying filepath of the image
            img_path = get_image_layer_path(event.value, self.image_path_dict)
            # Insert into the overall dict of images and their paths (if path is present)
            # This will be None when we are viewing arrays loaded separately from napari
            if img_path is not None:
                self.image_path_dict[img_path.stem] = img_path
            # Then update the counts of files (and their types) with the extra image
            self.update_file_count()
            # Switch flag to signify the image has been loaded
            # Adding via drag+drop blocks the UI, so it's fine to do here and not when adding begins
            self.parent.subwidgets["nxf"].all_loaded = True

    def on_layer_removed(self, event):
        """
        Triggered whenever a layer is removed from the viewer.

        Checks if the layer is an image, and if so, removes it from the list of images to process.
        """
        if isinstance(event.value, Image):
            # Extract the underlying filepath of the image
            img_path = event.value.source.path
            # Remove from the list of images
            if img_path is not None:
                if Path(img_path).stem in self.image_path_dict:
                    del self.image_path_dict[Path(img_path).stem]
            else:
                if event.value.name in self.image_path_dict:
                    del self.image_path_dict[event.value.name]
            # Update file count with image removed
            self.update_file_count()

    def browse_imgs_files(self):
        """
        Opens a dialog for selecting images to segment.
        """
        # TODO: Implement a cache that stores the last directory used?
        # Should this cache persist across sessions, or is that invasive?
        fnames, _ = QFileDialog.getOpenFileNames(
            self,
            "Select one or more images",
            str(Path.home()),
            "",
        )
        if fnames != []:
            self.update_file_count(paths=fnames)
            self.view_images(imgs_to_load=fnames)

    def browse_imgs_dir(self):
        """
        Opens a dialog for selecting a directory that contains images to segment.
        """
        # TODO: Load multiple directories - https://stackoverflow.com/a/28548773/9963224
        # Quite the pain, and potentially brittle if Qt backend changes
        result = QFileDialog.getExistingDirectory(
            self, caption="Select image directory", directory=None
        )
        if result != "":
            all_paths = list(Path(result).glob("*"))
            self.update_file_count(paths=all_paths)
            self.view_images(imgs_to_load=all_paths)

    def view_images(self, imgs_to_load: list[Path | str] | None = None):
        """
        Loads the selected images into napari for viewing (in separate threads).
        """
        # Return if there's nothing to show
        if len(self.image_path_dict) == 0:
            return
        if imgs_to_load is None:
            # Check if there are images to load that haven't been already
            viewer_imgs = [
                Path(i.name).stem for i in self.viewer.layers if isinstance(i, Image)
            ]
            imgs_to_load = [
                v for k, v in self.image_path_dict.items() if k not in viewer_imgs
            ]
        # If giving paths, double-check they aren't already loaded somehow
        elif imgs_to_load:
            remove_fnames = []
            for fname in imgs_to_load:
                if Path(fname).stem in self.viewer.layers:
                    remove_fnames.append(fname)
            imgs_to_load = [i for i in imgs_to_load if i not in remove_fnames]
        # Selecting no images will cause imgs_to_load=False, I think?
        else:
            return
        if len(imgs_to_load) == 0:
            return
        # Ensure Nextflow subwidget knows not all images are loaded until loading completes
        self.parent.subwidgets["nxf"].all_loaded = False
        # Reset counter
        self.load_img_counter = 0

        # Create separate thread worker to avoid blocking
        @thread_worker(
            connect={
                "returned": self._add_image,
                "finished": self._finished_loading,
            }
        )
        def _load_image(fpath):
            # can't directly use viewer.open with plugin outside of main thread :(
            return aiod_io.load_image(fpath), Path(fpath)

        # Load each image in a separate thread
        for fpath in imgs_to_load:
            _load_image(fpath)
        # NOTE: This does not work well for a directory of large images on a remote directory
        # But that would trigger loading GBs into memory over network, which is...undesirable
        self.loading_txt = f" (loading {len(imgs_to_load)} image{'s' if len(imgs_to_load) > 1 else ''}...)"
        self.img_counts.setText(self.img_counts.text() + self.loading_txt)

    def _add_image(self, res):
        """
        Adds an image to the viewer when loaded, using its filepath as the name.
        """
        bioio_img, fpath = res
        # Add the image to the overall dict
        self.image_path_dict[fpath.stem] = fpath
        layer_data = prepare_bioio_as_napari_layer(bioio_img, fpath)
        for i in layer_data:
            self.viewer.add_layer(Layer.create(*i))

    def _finished_loading(self):
        """Signify to user that all images have been loaded."""
        self.img_counts.setText(
            self.img_counts.text().replace(self.loading_txt, " (all images loaded).")
        )
        # Ensure Nextflow subwidget knows everything is loaded to extract metadata
        self.parent.subwidgets["nxf"].all_loaded = True
        # Also reset the viewer itself to ensure images are visible
        self.viewer.reset_view()
        # Signalling the images
        self.images_loaded.emit()
        # Apply any pending axes override once images are loaded
        if self.axes_override_edit.text().strip():
            self.on_apply_axes_override()

    def on_apply_axes_override(self):
        """Apply a manually specified axis order to all loaded image layers.

        Patches ``layer.metadata["dimensions"]`` so that channel/Z dropdowns
        and ``get_img_dims()`` use the correct axis mapping without reloading
        the image data.
        """
        axes = self.axes_override_edit.text().strip().upper()
        if not axes:
            return
        # Validate: only known bioio axis letters
        valid_letters = set("TCZYXS")
        invalid = set(axes) - valid_letters
        if invalid:
            from napari.utils.notifications import show_error

            show_error(
                f"Invalid axis letters: {', '.join(sorted(invalid))}. "
                "Use letters from T, C, Z, Y, X, S only."
            )
            return
        if len(axes) != len(set(axes)):
            from napari.utils.notifications import show_error

            show_error("Duplicate axis letters in override string.")
            return

        patched = 0
        feedback_dims = None
        for layer in self.viewer.layers:
            if not isinstance(layer, Image):
                continue
            if layer.data.ndim != len(axes):
                continue
            new_dims = Dimensions(axes, layer.data.shape)
            layer.metadata["dimensions"] = new_dims
            if feedback_dims is None:
                feedback_dims = new_dims
            patched += 1

        if patched == 0:
            from napari.utils.notifications import show_info

            show_info(
                f"No image layers with {len(axes)} dimensions found to apply '{axes}' to."
            )
            return

        # Show detected dimensions as feedback
        if feedback_dims is not None:
            parts = []
            for letter in axes:
                val = getattr(feedback_dims, letter, None)
                if val is not None:
                    parts.append(f"{letter}={val}")
            self.axes_feedback_label.setText(f"Detected: {', '.join(parts)}")

        # Refresh channel dropdowns in model widget (if it exists)
        if "model" in self.parent.subwidgets:
            self.parent.subwidgets["model"]._refresh_all_channel_dropdowns()

        from napari.utils.notifications import show_info

        show_info(f"Applied axes '{axes}' to {patched} image layer(s).")

    def update_file_count(self, paths: list[str | Path] | None = None):
        """
        Identify all the files in a given path, and return a count
        (broken down by extension)
        """
        # Reinitialise text
        txt = "Selected "
        # Add paths to the overall list if specific ones need adding
        if paths is not None:
            for img_path in paths:
                img_path = Path(img_path)
                self.image_path_dict[img_path.stem] = img_path
        # If no files remaining, reset message and return
        if len(self.image_path_dict) == 0:
            self.img_counts.setText(self.init_file_msg)
            return
        # Get all the extensions in the path
        extension_counts = Counter([i.suffix for i in self.image_path_dict.values()])
        # Sort by highest and get the suffixes and their counts
        ext_counts = extension_counts.most_common()
        if len(ext_counts) > 1:
            # Nicely format the list of files and their extensions
            for i, (ext, count) in enumerate(ext_counts):
                if i == (len(ext_counts) - 1):
                    txt += f"and {count} {ext}"
                else:
                    txt += f"{count} {ext}, "
        else:
            txt += f"{ext_counts[0][1]} {ext_counts[0][0]}"
        txt += f" file{'s' if sum(extension_counts.values()) > 1 else ''}"
        self.img_counts.setText(txt)

    def clear_directory(self):
        """
        Clears the selected directory and resets the image counts.
        """
        # Reset selected images and their paths
        self.image_path_dict = {}
        # Reset image count text
        self.img_counts.setText(self.init_file_msg)
        # Remove Image layers from napari viewer
        img_layers = [i for i in self.viewer.layers if isinstance(i, Image)]
        for layer in img_layers:
            self.viewer.layers.remove(layer)

    def get_config_params(self, params):
        img_dir = params.get("img_dir")
        if img_dir is None:
            return {"img_paths": []}
        try:
            df = pd.read_csv(img_dir)
            img_paths = df["img_path"].drop_duplicates().tolist()
        except Exception as e:
            raise RuntimeError(
                f"Failed to read image list from config (img_dir={img_dir})"
            ) from e
        return {"img_paths": img_paths}

    def load_config(self, config):
        img_paths = config["img_paths"]
        self.update_file_count(paths=img_paths)
        self.view_images(imgs_to_load=img_paths)

    def specify_url(self):
        """
        Allow user to specify a URL to e.g. a Zarr file to use.

        Considerations here are around how to handle the data once loaded.
        May require napari-ome-zarr plugin.
        """
        raise NotImplementedError
