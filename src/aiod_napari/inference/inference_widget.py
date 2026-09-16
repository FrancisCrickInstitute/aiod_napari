import copy
import time
from collections import Counter
from pathlib import Path

import aiod_utils.preprocess
import aiod_utils.rle as aiod_rle
import napari
import numpy as np
import tifffile
from aiod_utils.io import (
    MASK_SEPARATOR,
    extract_idxs_from_fname,
    get_combined_mask_name,
    get_image_id,
    get_mask_name,
    get_mask_prefix,
    get_mask_prefix_from_name,
    is_combined_mask,
)
from aiod_utils.stacks import stack_to_shape
from napari.qt.threading import thread_worker

from aiod_napari.inference.config_widget import ConfigWidget
from aiod_napari.inference.data_selection import DataWidget
from aiod_napari.inference.mask_export import ExportWidget
from aiod_napari.inference.model_selection import ModelWidget
from aiod_napari.inference.nxf import NxfWidget
from aiod_napari.inference.preprocess import PreprocessWidget
from aiod_napari.utils import (
    calc_param_hash,
    find_image_layer,
    get_image_layer_path,
    require_image_layer,
    short_hash,
)
from aiod_napari.widget_classes import MainWidget


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

        # Create the box for selecting the task (organelle) and the model to run
        self.register_widget(
            ModelWidget(viewer=self.viewer, parent=self, expanded=True)
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
            NxfWidget(
                viewer=self.viewer,
                parent=self,
                pipeline="inference",
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
            {k: v for k, v in nxf_params.items() if k in ["num_substacks", "overlap"]}
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
            mask_layer_name = img_dict["layer_name"]
            # Check if this mask has been imported already
            if mask_layer_name in self.viewer.layers:
                masks_exist.append(True)
            # Check if the mask exists from a previous run to load in
            elif (
                self.subwidgets["nxf"].mask_dir_path
                / self._get_final_mask_name(
                    img_dict["image_id"],
                    img_dict["prep_hash"],
                    extension=self._get_output_format(),
                )
            ).exists():
                masks_exist.append(True)
                # Used in create_mask_layers, which handles preprocessing sets
                load_paths.append(img_dict["img_path"])
            # Otherwise, we need to run the pipeline
            else:
                masks_exist.append(False)
                # Used directly in img_list_fpath CSV, which does not account for multiple preprocessing sets
                if img_dict["img_path"] not in img_paths:
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
            fpath, layer_name, prep_options, preprocess_str = (
                img_dict["img_path"],
                img_dict["layer_name"],
                img_dict["prep_set"],
                img_dict["preprocess_str"],
            )
            # Check if the mask file already exists
            mask_fpath = self.subwidgets[
                "nxf"
            ].mask_dir_path / self._get_final_mask_name(
                img_dict["image_id"],
                img_dict["prep_hash"],
                extension=self._get_output_format(),
            )
            # Grab corresponding image layer to get info as needed
            img_layer = require_image_layer(self.viewer, fpath)
            img_metadata = (
                copy.deepcopy(img_layer.metadata)
                if img_layer.metadata is not None
                else {}
            )
            # If it does, load it
            if mask_fpath.exists():
                mask_data, metadata = self._load_mask_file(mask_fpath)
                metadata = metadata["metadata"]
                metadata["img_scale"] = img_layer.scale
                metadata["preprocess_str"] = preprocess_str
                # Read downsampling structured preprocessing options
                downsample_factor = (
                    aiod_utils.preprocess.get_downsample_factor(methods=prep_options)
                    if prep_options
                    else None
                )
                if downsample_factor is not None:
                    metadata["downsample_factor"] = downsample_factor[
                        -len(img_layer.scale) :
                    ]
                # Expand the mask to match the image's full ndim by inserting
                # singleton dims at non-spatial positions (e.g. ZYX → ZCYX
                # gives (Z,Y,X) → (Z,1,Y,X)) so napari aligns axes correctly.
                mask_data = self._expand_mask_to_img_dims(
                    mask_data, img_layer, img_metadata
                )
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
                        metadata=metadata,
                        scale=(
                            img_layer.scale
                            * (downsample_factor or [1.0])[-img_layer.scale.ndim :]
                        ),
                    )
            else:
                # If the associated image is present, use its shape
                # Get ndim of the layer (this accounts for RGB)
                ndim = img_layer.ndim
                # Fetch axes metadata early; used for both img_shape and
                # mask_scale so that any ordering (CZYX, ZCYX, ZYX, CYX, YX,
                # …) is handled correctly without positional assumptions.
                _dims = img_metadata.get("dimensions", None)
                _spatial = frozenset("ZYX")
                # _dims.order is the FULL standard bioio order (e.g. "TCZYX"),
                # but the image data was loaded with only the *present* dims
                # (those with size > 1, e.g. "YX" for a 2D image or "ZYX" for
                # a 3D one). Using the full order in zip() would pair leading
                # non-spatial dims (T, C) with the data axes, yielding an
                # empty img_shape for 2D images. Filter to present dims only.
                _data_order = (
                    "".join(d for d in _dims.order if getattr(_dims, d, 1) > 1)
                    if _dims is not None and hasattr(_dims, "order")
                    else None
                )
                # Determine the spatial-only shape for the mask placeholder.
                # Instance segmentation models return spatial-only outputs
                # even when the input has one or more channel dimensions.
                if _data_order is not None:
                    # Use the present-dims order to pick spatial dims exactly,
                    # regardless of where C (or T) sits in the array.
                    img_shape = tuple(
                        s
                        for d, s in zip(_data_order, img_layer.data.shape, strict=False)
                        if d in _spatial
                    )
                elif img_layer.rgb:
                    # Napari represents RGB images with the colour channel as
                    # the last axis (Y, X, 3) or (Z, Y, X, 3); drop it.
                    img_shape = img_layer.data.shape[:-1]
                elif ndim >= 4:
                    # No axes metadata and ≥4-D: strip the leading axis as a
                    # best-effort guess (most likely C or T at position 0).
                    img_shape = img_layer.data.shape[1:]
                elif ndim == 3:
                    # If we have a Z, no problem
                    if ("dimensions" in img_metadata) and (
                        img_metadata["dimensions"].Z > 1
                    ):
                        img_shape = img_layer.data.shape
                    # Otherwise not loaded with bioio, so handle as Napari interprets
                    else:
                        # If RGB, then 2D RGB image
                        # NOTE: This does not handle multi-channel 2D images
                        if img_layer.rgb:
                            img_shape = img_layer.data.shape[1:]
                        # Otherwise it's 3D single-channel image
                        else:
                            img_shape = img_layer.data.shape
                # Otherwise take the 2D image shape
                # NOTE: [:ndim] is to handle RGB images as Napari interprets
                else:
                    img_shape = img_layer.data.shape[:ndim]
                if prep_options is not None:
                    # Check if downsampling
                    downsample_factor = aiod_utils.preprocess.get_downsample_factor(
                        prep_options
                    )
                    if downsample_factor is not None:
                        img_metadata["downsample_factor"] = downsample_factor
                    mask_shape = stack_to_shape(
                        aiod_utils.preprocess.get_output_shape(
                            options=prep_options, input_shape=img_shape
                        ),
                        ndim=len(img_shape),
                    )
                else:
                    mask_shape = img_shape
                # Expand the spatial-only mask shape to match the image's full
                # ndim by inserting size-1 dims at non-spatial axis positions
                # (e.g. (Z,Y,X) → (Z,1,Y,X) for ZCYX), so napari aligns axes
                # correctly instead of padding from the left with trailing-dim logic.
                # Use _data_order (present dims only) to avoid inserting singletons
                # for absent dims (T, C with size 1) from the full bioio order.
                if _data_order is not None and not img_layer.rgb:
                    expanded = list(mask_shape)
                    for pos, dim_char in enumerate(_data_order):
                        if dim_char not in _spatial:
                            expanded.insert(pos, 1)
                    mask_shape = tuple(expanded)
                img_metadata["img_scale"] = img_layer.scale
                img_metadata["preprocess_str"] = preprocess_str
                # After expansion mask_shape has the same ndim as img_layer.scale.
                mask_scale = img_layer.scale
                # Add a Labels layer for this file
                self.viewer.add_labels(
                    np.zeros(mask_shape, dtype=np.uint16),
                    name=layer_name,
                    visible=False,
                    opacity=0.5,
                    metadata=img_metadata,
                    scale=mask_scale,
                )
            # Now move the new layer to be just above the image layer, ensuring they group together
            self.viewer.layers.move(
                self.viewer.layers.index(layer_name),
                self.viewer.layers.index(img_layer) + 1,
            )

    def _get_ambiguous_stems(self) -> frozenset[str]:
        """
        Stems shared by more than one selected image, so the layer name for
        those images has to carry the full image_id to stay unique.
        """
        stems = Counter(
            image_id.stem for image_id in self.subwidgets["data"].image_path_dict
        )
        return frozenset(stem for stem, count in stems.items() if count > 1)

    def get_img_mask_preps(self, img_paths: list | None = None):
        if img_paths is None:
            img_paths = list(self.subwidgets["data"].image_path_dict.values())
        # Derived from the whole selection rather than the img_paths subset
        # to try and create cleaner names if possible
        ambiguous_stems = self._get_ambiguous_stems()
        # Get the preprocessing options, if any
        # No preprocessing is just a single no-op set, so it needs no branch of
        # its own - the `not prep_set` check below gives it the unsuffixed names
        options = self.subwidgets["preprocess"].get_all_options() or [None]
        # Store the info for later use in the watcher/final mask insertion
        self.img_mask_info = []
        for img_path in img_paths:
            image_id = get_image_id(img_path)
            for prep_set in options:
                # Both are None for a no-op set, so the names match the mask
                # Nextflow creates from the original image filename
                suffix = aiod_utils.preprocess.get_params_str(prep_set, to_save=True)
                prep_hash = aiod_utils.preprocess.get_prep_hash(prep_set)
                self.img_mask_info.append(
                    {
                        "img_path": img_path,
                        # The identity every name below derives from, and what
                        # Segment-Flow's image_id column will hold
                        "image_id": image_id,
                        "prep_hash": prep_hash,
                        "layer_name": self._get_mask_layer_name(
                            image_id,
                            prep_hash,
                            executed=True,
                            ambiguous_stems=ambiguous_stems,
                        ),
                        "mask_prefix": get_mask_prefix(image_id, prep_hash),
                        "prep_set": prep_set if prep_set else None,
                        "preprocess_str": suffix,
                    }
                )
        # Indexed by the (image, preprocessing) identity, so a mask appearing on
        # disk is matched back to its record with one lookup - used by the
        # watcher to filter files and by update_masks to find the target layer
        self.mask_info_by_prefix = {i["mask_prefix"]: i for i in self.img_mask_info}

    def remove_mask_layers(self, img_paths=None):
        # Collate all image-mask-preprocess combos (handles preprocessing variants)
        self.get_img_mask_preps(img_paths)
        # Remove each mask layer if it exists
        for img_dict in self.img_mask_info:
            layer_name = img_dict["layer_name"]
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
                # NOTE: run_hash to get files with these params,
                # and mask prefix to be specific to this run's paths
                current_files = [
                    fpath
                    for fpath in self.subwidgets["nxf"].mask_dir_path.glob(
                        f"*{MASK_SEPARATOR}{self.run_hash}*.rle"
                    )
                    # Combined files can appear when a run is fast (i.e. single image)
                    if not is_combined_mask(fpath)
                    and get_mask_prefix_from_name(fpath) in self.mask_info_by_prefix
                ]
                if set(self.mask_fpaths) != set(current_files):
                    # Get the new files only
                    new_files = [i for i in current_files if i not in self.mask_fpaths]
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
        image_id,
        prep_hash: str | None = None,
        executed: bool = False,
        ambiguous_stems: frozenset[str] = frozenset(),
    ) -> str:
        # Readable display name for the napari layer
        # Shortened run hash and str of task-model-version
        # Preprocessing params are still hashed rather than raw
        # ...could change that in the future if less helpful than before?
        # Use ambiguous stems to choose simpler names if we can (i.e. removing the extension if not needed for uniqueness)
        name_id = image_id.value if image_id.stem in ambiguous_stems else image_id.stem
        task_model_variant_name = self.subwidgets["model"].get_task_model_variant_name(
            executed
        )
        # Same leading half as the real filenames, so the two stay recognisably
        # related if the prep suffix format ever changes
        return (
            f"{get_mask_prefix(name_id, prep_hash)}"
            f"{MASK_SEPARATOR}{task_model_variant_name}-{short_hash(self.run_hash)}"
        )

    def _get_output_format(self) -> str:
        """Return the output format used for the executed run.

        Reads from the stored nxf_params (set when the pipeline ran)
        Falls back to the current dropdown value when no run has been executed yet.
        """
        nxf_params = self.subwidgets["nxf"].nxf_params
        if nxf_params is not None and "output_format" in nxf_params:
            return nxf_params["output_format"]
        return self.subwidgets["nxf"].output_format_box.currentText()

    def _load_mask_file(self, fpath: Path):
        """Load a mask file, handling both .rle and .tiff formats.

        Returns (arr, metadata_dict).
        """
        if fpath.suffix == ".tiff":
            with tifffile.TiffFile(fpath) as tif:
                arr = tif.asarray()
                # imagej_metadata holds the dict written by tifffile.imwrite(..., imagej=True)
                # e.g. {"downsample_factor": 2}. Wrap it to match the rle metadata structure.
                imagej_meta = tif.imagej_metadata or {}
                # Strip tifffile bookkeeping keys that aren't part of our saved metadata
                imagej_meta = {
                    k: v for k, v in imagej_meta.items() if k not in ("axes",)
                }
            return arr, {"metadata": imagej_meta}
        else:
            encoding = aiod_rle.load_encoding(fpath)
            arr, metadata = aiod_rle.decode(encoding)
            return arr, metadata

    def _get_final_mask_name(
        self,
        image_id,
        prep_hash: str | None = None,
        extension: str = "rle",
    ) -> str:
        # The single, final combined mask, not per-substack files
        return get_combined_mask_name(
            get_mask_name(
                run_hash=self.run_hash, image_id=image_id, prep_hash=prep_hash
            ),
            extension,
        )

    def _expand_mask_to_img_dims(
        self,
        mask_arr: np.ndarray,
        img_layer,
        img_metadata: dict,
    ) -> np.ndarray:
        """Expand a spatial-only mask to match the full ndim of the image layer.

        Inserts singleton dimensions at the positions of non-spatial axes using
        the axes order recorded in ``img_metadata["dimensions"].order``.

        Example: ZYX mask (75, 75, 75) + ZCYX image → (75, 1, 75, 75), so that
        napari aligns the Z slider correctly rather than using trailing-dim padding
        which would map the mask's Z axis onto the image's C axis.

        The invariant that makes the sequential insertion work: when processing
        position ``pos`` in the axes order string, the array has exactly ``pos``
        axes (one contributed per position already handled, whether spatial or
        inserted-singleton), so ``np.expand_dims(arr, axis=pos)`` always places
        the new singleton at the correct absolute position in the final array.

        Returns the mask unchanged when axes metadata is unavailable, when the
        mask already matches the image ndim, or for RGB images.
        """
        _dims = img_metadata.get("dimensions")
        _spatial = frozenset("ZYX")
        if (
            img_layer.rgb
            or _dims is None
            or not hasattr(_dims, "order")
            or mask_arr.ndim >= img_layer.ndim
        ):
            return mask_arr
        result = mask_arr
        for pos, dim_char in enumerate(_dims.order):
            if dim_char not in _spatial:
                result = np.expand_dims(result, axis=pos)
        return result

    def _reset_viewer(self, return_value=None):
        """
        Should help alleviate rendering issue where masks are mis-aligned.

        Need to do it here as interacting with the viewer in the thread_worker causes issues.
        """
        self.viewer.dims.set_point(0, 0)

    def update_masks(self, new_files: list[str | Path]):
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
                print(f"File {f} not found, may have already been deleted. Skipping...")
                continue
            except ValueError as e:
                print(f)
                print(e)
                continue
            # Get indices from fname, modified if downsampled
            start_x, end_x, start_y, end_y, start_z, end_z = extract_idxs_from_fname(
                fname=f
            )
            # Match the file back to its record by the (image, preprocessing)
            # identity
            d = self.mask_info_by_prefix.get(get_mask_prefix_from_name(f))
            if d is None:
                print(f"No matching layer found for mask file {f}, skipping...")
                continue
            mask_layer_name = d["layer_name"]
            img_path = d["img_path"]
            image_id = d["image_id"]
            label_layer = self.viewer.layers[mask_layer_name]
            # Expand the spatial-only mask slice to match the image's full ndim
            # (e.g. (nz,ny,nx) → (nz,1,ny,nx) for a ZCYX image) so the
            # assignment index and the array shape are consistent.
            _img_layer_ref = find_image_layer(self.viewer, img_path)
            _img_layer_meta = (
                (_img_layer_ref.metadata or {}) if _img_layer_ref is not None else {}
            )
            if _img_layer_ref is not None:
                mask_arr = self._expand_mask_to_img_dims(
                    mask_arr, _img_layer_ref, _img_layer_meta
                )
            _dims_meta = _img_layer_meta.get("dimensions", None)
            # Insert mask data using the correct per-axis index tuple so that
            # non-spatial singleton dims are addressed with slice(None).
            if (
                _dims_meta is not None
                and hasattr(_dims_meta, "order")
                and label_layer.ndim == len(_dims_meta.order)
            ):
                idx = tuple(
                    slice(start_z, end_z)
                    if c == "Z"
                    else slice(start_y, end_y)
                    if c == "Y"
                    else slice(start_x, end_x)
                    if c == "X"
                    else slice(None)
                    for c in _dims_meta.order
                )
            elif label_layer.ndim >= 3:
                idx = (
                    slice(start_z, end_z),
                    slice(start_y, end_y),
                    slice(start_x, end_x),
                )
            else:
                idx = (slice(start_y, end_y), slice(start_x, end_x))
            # Insert the substack into the established slice region (idx)
            if label_layer.data[idx].shape == mask_arr.shape:
                label_layer.data[idx] = mask_arr
            # On a mismatch, we just skip the preview as insert_final_masks should fix all
            else:
                print(
                    f"Mask {Path(f).name} has shape {mask_arr.shape}, but "
                    f"{label_layer.data[idx].shape} was expected at its substack "
                    "indices. Skipping preview for it."
                )
            label_layer.visible = True
            # Apply scale for downsampled masks, accounting for pixel size scaling if present
            downsample_factor = label_layer.metadata.get("downsample_factor", None)
            if downsample_factor is not None:
                img_scale = label_layer.metadata["img_scale"]
                label_layer.scale = img_scale * downsample_factor[-len(img_scale) :]
            # Try to rearrange the layers to get them on top
            idxs = []
            # Have to check due to possible delay in loading
            if _img_layer_ref is not None:
                idxs.append(self.viewer.layers.index(_img_layer_ref))
            # We create the mask layer, so it will always exist
            label_idx = self.viewer.layers.index(label_layer)
            idxs.append(label_idx)
            self.viewer.layers.move_multiple(idxs, -1)
            # Switch viewer to latest Z slice, using the correct viewer dim for Z.
            if end_z > start_z:
                if (
                    _dims_meta is not None
                    and hasattr(_dims_meta, "order")
                    and "Z" in _dims_meta.order
                    and end_z - 1 > 0
                ):
                    self.viewer.dims.set_point(_dims_meta.order.index("Z"), end_z - 1)
                else:
                    self.viewer.dims.set_point(0, end_z - 1)
            # Insert the slice number into tracker for the progress bar
            self.subwidgets["nxf"].progress_dict[image_id] += 1
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
            mask_layer_name = img_dict["layer_name"]
            # Clear the current mask layer of data (to free up memory??)
            self.viewer.layers[mask_layer_name].data = np.zeros_like(
                self.viewer.layers[mask_layer_name].data
            )
            # Load the mask
            fpath = self.subwidgets["nxf"].mask_dir_path / self._get_final_mask_name(
                img_dict["image_id"],
                img_dict["prep_hash"],
                extension=self._get_output_format(),
            )
            mask_arr, _ = self._load_mask_file(fpath)
            # Expand to image ndim (e.g. ZYX → Z1YX for ZCYX) so napari aligns
            # the mask's Z axis with the image's Z axis rather than its C axis.
            _img_layer_ref = find_image_layer(self.viewer, img_dict["img_path"])
            _img_layer_meta = (
                (_img_layer_ref.metadata or {}) if _img_layer_ref is not None else {}
            )
            if _img_layer_ref is not None:
                mask_arr = self._expand_mask_to_img_dims(
                    mask_arr, _img_layer_ref, _img_layer_meta
                )
            # Insert mask data
            label_layer = self.viewer.layers[mask_layer_name]
            # Recreate the layer if shape changed after expansion.
            if label_layer.data.shape != mask_arr.shape:
                layer_idx = self.viewer.layers.index(label_layer)
                layer_meta = label_layer.metadata
                layer_name_local = label_layer.name
                self.viewer.layers.remove(label_layer)
                label_layer = self.viewer.add_labels(
                    np.zeros(mask_arr.shape, dtype=np.uint16),
                    name=layer_name_local,
                    visible=False,
                    opacity=0.5,
                    metadata=layer_meta,
                )
                self.viewer.layers.move(
                    self.viewer.layers.index(layer_name_local), layer_idx
                )
            label_layer.data = mask_arr
            label_layer.visible = True
            # Apply scale for downsampled masks, accounting for pixel size scaling if present
            downsample_factor = label_layer.metadata.get("downsample_factor", None)
            if downsample_factor is not None:
                img_scale = label_layer.metadata["img_scale"]
                label_layer.scale = img_scale * downsample_factor[-len(img_scale) :]
        # Now we'll sort all the layers, grouping together the image and mask layers for each image
        # Get the image layer names
        image_layers = sorted(
            [i for i in self.viewer.layers if isinstance(i, napari.layers.Image)],
            key=lambda x: x.name,
            reverse=True,  # Lowest alphabetically is at bottom of Napari layerlist
        )
        idx = 0
        for img_layer in image_layers:
            # First, move the current image layer to next position
            self.viewer.layers.move(self.viewer.layers.index(img_layer), idx)
            # Grab all relevant mask layers
            layer_path = get_image_layer_path(img_layer)
            mask_layer_names = [
                i["layer_name"]
                for i in self.img_mask_info
                if i["img_path"] == layer_path
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
                self.viewer.layers.move(self.viewer.layers.index(mask_layer), idx)
            # Increment the index for next image layer
            idx += 1
