from pathlib import Path
import time
from typing import Optional, Union

import napari
from napari.qt.threading import thread_worker
import numpy as np

from ai_on_demand.inference import (
    TaskWidget,
    DataWidget,
    ExportWidget,
    ModelWidget,
    InferenceNxfWidget,
    PreprocessWidget,
    ConfigWidget,
)
from ai_on_demand.widget_classes import MainWidget
from ai_on_demand.utils import calc_param_hash
import aiod_utils.preprocess
from aiod_utils.io import extract_idxs_from_fname
import aiod_utils.rle as aiod_rle


class Inference(MainWidget):
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__(
            napari_viewer=napari_viewer,
            title="Inference",
            tooltip="""
Run segmentation/inference on selected images using one of the available pre-trained models.
""",
        )
        # Handy attributes to check things
        # These get set in subwidgets, but are used across so set here for ease
        self.selected_task = None
        self.selected_model = None
        self.selected_variant = None
        self.executed_task = None
        self.executed_model = None
        self.executed_variant = None
        self.run_hash = None

        # Set selection colour
        self.colour_selected = "#F7AD6F"

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
                variant="inference",
                expanded=False,
            )
        )

        # Create the box for selecting the directory, showing img count etc.
        self.register_widget(
            DataWidget(viewer=self.viewer, parent=self, expanded=False)
        )

        # Add a box for preprocessing options
        self.register_widget(
            PreprocessWidget(viewer=self.viewer, parent=self, expanded=False)
        )

        # Add the button for running the Nextflow pipeline
        self.register_widget(
            InferenceNxfWidget(
                viewer=self.viewer,
                parent=self,
                expanded=False,
            )
        )

        self.register_widget(
            ConfigWidget(viewer=self.viewer, parent=self, expanded=False)
        )

        # Add box for exporting masks
        self.register_widget(
            ExportWidget(viewer=self.viewer, parent=self, expanded=False)
        )

        self.subwidgets["nxf"].config_ready.connect(
            self.subwidgets["config"].enable_save_config
        )

    def get_run_hash(self, nxf_params: dict):
        """
        Gather all the parameters from the subwidgets to be used in obtaining a unique hash for a run.
        """
        hashed_params = {}
        # Add model details
        hashed_params["task"] = nxf_params["task"]
        hashed_params["model"] = nxf_params["model"]
        hashed_params["variant"] = nxf_params["model_type"]
        # Add the model dictionary (hashed)
        hashed_params["model_hash"] = self.subwidgets["model"].model_param_hash
        # Get the advanced Nextflow parameters
        hashed_params.update(
            {
                k: v
                for k, v in nxf_params.items()
                if k in ["num_substacks", "overlap"]
            }
        )
        # Get the preprocessing parameters
        hashed_params["preprocess"] = nxf_params["preprocess"]
        # Though this only applies if post-processing is added (I think)
        if self.subwidgets["nxf"].postprocess_btn.isChecked():
            hashed_params["iou_threshold"] = nxf_params["iou_threshold"]
        # Calculate the overall hash for this run considering the model parameters
        # and Nextflow parameters that affect the output
        self.run_hash = calc_param_hash(hashed_params)

    def check_masks(self) -> tuple[bool, list, list]:
        """
        Function to check if masks are present for the current setup, either
        already imported or in the Nextflow output directory.

        If all are present, avoids running the Nextflow pipeline.
        """
        # List of booleans for whether masks exist for each image
        masks_exist = []
        # List of image paths to load masks for
        load_paths = []
        # List of image paths to pass to Nextflow
        img_paths = []
        # Get the image-mask-preprocess combos so we can check for relevant masks
        self.get_img_mask_preps()
        # Loop over each image-mask-preprocess combo and check if the mask exists
        for img_dict in self.img_mask_info:
            # Extract the save string from the preprocessing options
            preprocess_str = aiod_utils.preprocess.get_params_str(
                img_dict["prep_set"], to_save=True
            )
            mask_layer_name = self._get_mask_layer_name(
                img_dict["img_path"].stem,
                executed=True,
                preprocess_str=preprocess_str,
            )
            # Check if this mask has been imported already
            if mask_layer_name in self.viewer.layers:
                masks_exist.append(True)
            # Check if the mask exists from a previous run to load in
            elif (
                self.subwidgets["nxf"].mask_dir_path
                / self._get_mask_name(
                    img_dict["img_path"].stem,
                    executed=True,
                    truncate=False,
                    preprocess_str=preprocess_str,
                )
            ).exists():
                masks_exist.append(True)
                load_paths.append(img_dict["img_path"])
            # Otherwise, we need to run the pipeline
            else:
                masks_exist.append(False)
                img_paths.append(img_dict["img_path"])
        # Proceed to run the pipeline if any masks are missing
        proceed = not all(masks_exist)
        # If we aren't proceeding, there should be no images without masks!
        if not proceed:
            assert len(img_paths) == 0
        return proceed, img_paths, load_paths

    def create_mask_layers(self, img_paths=None):
        # In some cases this may be repetitive, but repeaet to ensure up-to-date
        self.get_img_mask_preps(img_paths)
        # Now loop over every image-mask-preprocess combo
        for img_dict in self.img_mask_info:
            fpath, layer_name, prep_options = (
                img_dict["img_path"],
                img_dict["layer_name"],
                img_dict["prep_set"],
            )
            preprocess_str = aiod_utils.preprocess.get_params_str(
                prep_options, to_save=True
            )
            # Check if the mask file already exists
            mask_fpath = self.subwidgets[
                "nxf"
            ].mask_dir_path / self._get_mask_name(
                img_dict["img_path"].stem,
                executed=True,
                truncate=False,
                preprocess_str=preprocess_str,
            )
            # If it does, load it
            if mask_fpath.exists():
                mask_data = aiod_rle.load_encoding(mask_fpath)
                mask_data, metadata = aiod_rle.decode(mask_data)
                # Check if the mask layer already exists
                if layer_name in self.viewer.layers:
                    # If so, update the data just to make sure & ensure visible
                    self.viewer.layers[layer_name].data = mask_data
                    self.viewer.layers[layer_name].visible = True
                # If not, add a Labels layer
                else:
                    # Add a Labels layer for this file
                    self.viewer.add_labels(
                        mask_data,
                        name=layer_name,
                        visible=True,
                        opacity=0.5,
                        metadata=metadata["metadata"],
                    )
            else:
                # If the associated image is present, use its shape
                # Get ndim of the layer (this accounts for RGB)
                img_layer = self.viewer.layers[f"{fpath.stem}"]
                ndim = img_layer.ndim
                metadata = img_layer.metadata
                # Channels (non-RGB) & Z
                # TODO: Switch to using utils.get_img_dims
                if ndim == 4:
                    # Channels should be first, don't care for labels so remove
                    img_shape = img_layer.data.shape[1:]
                elif ndim == 3:
                    # If we have a Z, no problem
                    if ("bioio_dims" in metadata) and (
                        metadata["bioio_dims"].Z > 1
                    ):
                        img_shape = self.viewer.layers[
                            f"{fpath.stem}"
                        ].data.shape
                    # Otherwise not loaded with bioio, so handle as Napari interprets
                    else:
                        # If RGB, then 2D RGB image
                        # NOTE: This does not handle multi-channel 2D images
                        if img_layer.rgb:
                            img_shape = self.viewer.layers[
                                f"{fpath.stem}"
                            ].data.shape[1:]
                        # Otherwise it's 3D single-channel image
                        else:
                            img_shape = self.viewer.layers[
                                f"{fpath.stem}"
                            ].data.shape
                # Otherwise take the 2D image shape
                # NOTE: [:ndim] is to handle RGB images as Napari interprets
                else:
                    img_shape = self.viewer.layers[f"{fpath.stem}"].data.shape[
                        :ndim
                    ]
                if prep_options is not None:
                    # Check if downsampling
                    metadata = {}
                    downsample_factor = (
                        aiod_utils.preprocess.get_downsample_factor(
                            prep_options
                        )
                    )
                    if downsample_factor is not None:
                        metadata["downsample_factor"] = downsample_factor
                    mask_shape = aiod_utils.preprocess.get_output_shape(
                        options=prep_options, input_shape=img_shape
                    )
                else:
                    mask_shape = img_shape
                # Add a Labels layer for this file
                self.viewer.add_labels(
                    np.zeros(mask_shape, dtype=np.uint16),
                    name=layer_name,
                    visible=False,
                    opacity=0.5,
                    metadata=metadata,
                )
            # Now move the new layer to be just above the image layer, ensuring they group together
            self.viewer.layers.move(
                self.viewer.layers.index(layer_name),
                self.viewer.layers.index(fpath.stem) + 1,
            )

    def get_img_mask_preps(self, img_paths: Optional[list] = None):
        if img_paths is None:
            img_paths = list(self.subwidgets["data"].image_path_dict.values())
        # Get the preprocessing options, if any
        options = self.subwidgets["preprocess"].get_all_options()
        # Store the info for later use in the watcher/final mask insertion
        self.img_mask_info = []
        # If no preprocessing, create None's to zip with img_paths
        # And just use the normal layer names
        if options is None:
            prep_options = [None] * len(img_paths)
            all_img_paths = img_paths
            all_layer_names = [
                self._get_mask_layer_name(Path(i).stem, executed=True)
                for i in img_paths
            ]
        else:
            # Containers for all the paths, layer names, and preprocessing options
            prep_options = []
            all_img_paths = []
            all_layer_names = []
            # Now modify the layer names to include the preprocessing options
            for i, img_path in enumerate(img_paths):
                for prep_set in options:
                    all_img_paths.append(img_path)
                    # Get the preprocess param string to add to the layer name
                    suffix = aiod_utils.preprocess.get_params_str(
                        prep_set, to_save=True
                    )
                    layer_name = self._get_mask_layer_name(
                        Path(img_paths[i]).stem,
                        executed=True,
                        preprocess_str=suffix,
                    )
                    all_layer_names.append(layer_name)
                    prep_options.append(prep_set)
        self.mask_prefixes = set(
            [i.split("_masks_")[0] for i in all_layer_names]
        )
        # Insert all info into structure for later use
        for fpath, layer_name, prep_options in zip(
            all_img_paths, all_layer_names, prep_options
        ):
            self.img_mask_info.append(
                {
                    "img_path": fpath,
                    "layer_name": layer_name,
                    "prep_set": prep_options,
                }
            )

    def remove_mask_layers(self, img_paths=None):
        if img_paths is None:
            img_paths = self.subwidgets["data"].image_path_dict.values()
        # Construct the mask layer names
        layer_names = [
            self._get_mask_layer_name(Path(i).stem, executed=True)
            for i in img_paths
        ]
        # Create the Labels layers for each image
        for layer_name in layer_names:
            # Check if the mask layer already exists
            if layer_name in self.viewer.layers:
                self.viewer.layers.remove(self.viewer.layers[layer_name])

    def watch_mask_files(self):
        """
        File watcher to watch for new mask files being created during the Nextflow run.

        This is used to update the napari Labels layers with the new masks.

        Currently expects that the slices are stored as .rle files. Deactivates
        when it sees each image has the expected number of slices completed.
        """
        # Wait for at least one image to load as layers if not present
        if not self.viewer.layers:
            time.sleep(1)
        # Create the Labels layers for each image
        self.create_mask_layers()

        # NOTE: Wrapper as self/class not available at runtime
        @thread_worker(
            connect={
                "yielded": self.update_masks,
                "returned": self._reset_viewer,
            }
        )
        def _watch_mask_files(self):
            # Enable the watcher
            print("Activating watcher...")
            self.watcher_enabled = True
            # Initialize empty container for storing mask filepaths
            self.mask_fpaths = []
            # Loop and yield any changes infinitely while enabled
            while self.watcher_enabled:
                # Get all files
                current_files = list(
                    self.subwidgets["nxf"].mask_dir_path.glob("*.rle")
                )
                # Filter out any _all files, can occur when process is too fast (i.e. single image)
                current_files = [
                    i for i in current_files if Path(i).stem[-4:] != "_all"
                ]
                # Filter out files we are not running on
                current_files = [
                    i
                    for i in current_files
                    if Path(i).stem.split("_masks_")[0] in self.mask_prefixes
                ]
                if set(self.mask_fpaths) != set(current_files):
                    # Get the new files only
                    new_files = [
                        i for i in current_files if i not in self.mask_fpaths
                    ]
                    # Update file list and yield the difference
                    self.mask_fpaths = current_files
                    if new_files:
                        yield new_files
                # Sleep until next check
                time.sleep(2)
                # If we have as many slices as the total, we are done
                if (
                    sum(self.subwidgets["nxf"].progress_dict.values())
                    == self.subwidgets["nxf"].total_substacks
                ):
                    print("Deactivating watcher...")
                    self.watcher_enabled = False

        # Call the nested function
        _watch_mask_files(self)

    def _get_mask_layer_name(
        self,
        stem: str,
        extension: Optional[str] = None,
        executed: bool = False,
        include_hash: bool = True,
        truncate: bool = True,
        preprocess_str: Optional[str] = None,
    ):
        # If executed, use the executed attributes in case the user has changed the selection since running the pipeline
        task_model_variant_name = self.subwidgets[
            "model"
        ].get_task_model_variant_name(executed)
        if preprocess_str is not None:
            fname = f"{stem}_{preprocess_str}_masks_{task_model_variant_name}"
        else:
            # Construct the mask layer name
            fname = f"{stem}_masks_{task_model_variant_name}"
        # Add the hash if requested
        if include_hash:
            if truncate:
                fname += f"-{self.run_hash[:8]}"
            else:
                fname += f"-{self.run_hash}"
        if extension is not None:
            fname += f".{extension}"
        return fname

    def _get_mask_name(
        self,
        stem: str,
        extension: str = "rle",
        executed=False,
        truncate=False,
        preprocess_str: Optional[str] = None,
    ):
        mask_root = self._get_mask_layer_name(
            stem=stem,
            executed=executed,
            truncate=truncate,
            preprocess_str=preprocess_str,
        )
        # Add the _all marker to signify all slices/completeness
        mask_root += "_all"
        # Add the extension
        return f"{mask_root}.{extension}"

    def _reset_viewer(self):
        """
        Should help alleviate rendering issue where masks are mis-aligned.

        Need to do it here as interacting with the viewer in the thread_worker causes issues.
        """
        self.viewer.dims.set_point(0, 0)

    def update_masks(self, new_files: list[Union[str, Path]]):
        """
        Update the masks in the napari Labels layers with the new masks found in the last scan.
        """
        # Iterate over each new files and add the mask to the appropriate image
        for f in new_files:
            # Load the numpy array
            try:
                mask_arr = aiod_rle.load_encoding(f)
                mask_arr, _ = aiod_rle.decode(mask_arr)
            # NOTE: This is a temporary fix, and only occurs with fast models and a good GPU
            except FileNotFoundError:
                print(
                    f"File {f} not found, may have already been deleted. Skipping..."
                )
                continue
            except ValueError as e:
                print(f)
                print(e)
                continue
            # Get indices from fname, modified if downsampled
            start_x, end_x, start_y, end_y, start_z, end_z = (
                extract_idxs_from_fname(fname=f)
            )
            # Need to get the prefix and then compare with expected layer names
            prefix, _ = f.stem.split("_masks_")
            # Extract the relevant Labels layer
            for d in self.img_mask_info:
                if prefix == d["layer_name"].split("_masks_")[0]:
                    mask_layer_name = d["layer_name"]
                    img_name = d["img_path"].stem
                    break
            label_layer = self.viewer.layers[mask_layer_name]
            # Insert mask data
            # Check if dims match
            if label_layer.ndim != mask_arr.ndim:
                mask_arr = np.squeeze(mask_arr)
                assert (
                    label_layer.ndim == mask_arr.ndim
                ), f"Mask appears to be {mask_arr.ndim}D (after squeezing), but layer is {label_layer.ndim}D"
                label_layer.data = mask_arr
            else:
                # TODO: Handle multi-channel images
                # TODO: Check DHW orientation? Does Napari enforce this?
                if label_layer.ndim == 3:
                    label_layer.data[
                        start_z:end_z, start_y:end_y, start_x:end_x
                    ] = mask_arr
                else:
                    label_layer.data[start_y:end_y, start_x:end_x] = mask_arr
            label_layer.visible = True
            # Try to rearrange the layers to get them on top
            idxs = []
            # Have to check due to possible delay in loading
            if img_name in self.viewer.layers:
                img_idx = self.viewer.layers.index(
                    self.viewer.layers[img_name]
                )
                idxs.append(img_idx)
            # We create the mask layer, so it will always exist
            label_idx = self.viewer.layers.index(label_layer)
            idxs.append(label_idx)
            self.viewer.layers.move_multiple(idxs, -1)
            # Switch viewer to latest slice
            self.viewer.dims.set_point(0, end_z - 1)
            # Insert the slice number into tracker for the progress bar
            self.subwidgets["nxf"].progress_dict[img_name] += 1
        # Now update the total progress bar
        self.subwidgets["nxf"].update_progress_bar()

    def insert_final_masks(self):
        """
        Insert the final masks into the napari Labels layers.

        This is used to update the napari Labels layers with the final masks
        after the Nextflow pipeline has completed.
        """
        # Loop over each image and insert the final mask
        for img_dict in self.img_mask_info:
            # Extract the save string from the preprocessing options
            preprocess_str = aiod_utils.preprocess.get_params_str(
                img_dict["prep_set"], to_save=True
            )
            # Get the mask layer name, considering any preprocessing
            mask_layer_name = self._get_mask_layer_name(
                img_dict["img_path"].stem,
                executed=True,
                preprocess_str=preprocess_str,
            )
            # Clear the current mask layer of data (to free up memory??)
            self.viewer.layers[mask_layer_name].data = np.zeros_like(
                self.viewer.layers[mask_layer_name].data
            )
            # Load the mask
            fpath = self.subwidgets["nxf"].mask_dir_path / self._get_mask_name(
                img_dict["img_path"].stem,
                executed=True,
                truncate=False,
                preprocess_str=preprocess_str,
            )
            mask_arr = aiod_rle.load_encoding(fpath)
            # NOTE: Mask metadata should be no different, so ignore
            mask_arr, _ = aiod_rle.decode(mask_arr)
            # Insert mask data
            self.viewer.layers[mask_layer_name].data = mask_arr
            self.viewer.layers[mask_layer_name].visible = True
        # Now we'll sort all the layers, grouping together the image and mask layers for each image
        # Get the image layer names
        image_layers = sorted(
            [
                i
                for i in self.viewer.layers
                if isinstance(i, napari.layers.Image)
            ],
            key=lambda x: x.name,
            reverse=True,  # Lowest alphabetically is at bottom of Napari layerlist
        )
        idx = 0
        for img_layer in image_layers:
            # First, move the current image layer to next position
            self.viewer.layers.move(self.viewer.layers.index(img_layer), idx)
            # Grab all relevant mask layers
            mask_layer_names = [
                i["layer_name"]
                for i in self.img_mask_info
                if i["img_path"].stem == img_layer.name
            ]
            # Sort the mask layers
            mask_layers = sorted(
                [i for i in self.viewer.layers if i.name in mask_layer_names],
                key=lambda x: x.name,
                reverse=True,
            )
            for mask_layer in mask_layers:
                idx += 1
                # Move the mask layer to the next position
                self.viewer.layers.move(
                    self.viewer.layers.index(mask_layer), idx
                )
            # Increment the index for next image layer
            idx += 1
