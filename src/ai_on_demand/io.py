from pathlib import Path
from typing import Union
from functools import partial
from bioio_base.dimensions import DEFAULT_DIMENSION_ORDER_WITH_SAMPLES
from bioio import BioImage
from bioio_base.reader import Reader
from bioio_base.exceptions import UnsupportedFileFormatError

import aiod_utils.rle
import aiod_utils.io
import contextlib


def get_bioio_reader(path: Union[str, Path]):
    # Check if bioio can read this
    try:
        reader, plugin = aiod_utils.io._guess_reader(path), None
        if reader is None:
            # Run more exhaustive check for any compatible available reader
            plugin = BioImage.determine_plugin(path)
            if plugin is None:
                # No plugin can handle this file
                return None
        # INFO: this reduces redundancy, as BioImage.__init__() will call determine_plugin() again internally anyway, unless a specific reader is forwarded to BioImage later on.
        return partial(
            bioio_reader, bioio_reader_class=reader or plugin.metadata.get_reader()
        )
    except (
        AttributeError,
        FileNotFoundError,
        UnsupportedFileFormatError,
    ):
        return None


def bioio_reader(
    path: Union[str, Path], bioio_reader_class: Union[Reader, None]=None
):
    # Load the image with utils loader, keeping defaults
    path = Path(path)
    bioio_img = aiod_utils.io.load_image(
        path,
        reader=bioio_reader_class,
    )
    return prepare_bioio_as_napari_layer(bioio_img, path)


def prepare_bioio_as_napari_layer(bioio_img, path):
    """Return LaterData tuple"""
    dim_order = "".join(
        d
        for d in DEFAULT_DIMENSION_ORDER_WITH_SAMPLES
        if d in bioio_img.standard_metadata.dimensions_present
    )
    # Construct attributes and metadata for the layer object
    # Keys are valid napari Layer constructor arguments
    # on scale values: https://github.com/napari/napari/issues/6968
    layer_attributes = {
        "name": path.stem,
        "rgb": aiod_utils.io.guess_rgba(bioio_img),
        "scale": [getattr(bioio_img.scale, d) or 1 for d in dim_order if d!='S'],
        "metadata": {
            "path": path,
            "bioio_metadata": {
                "standard": bioio_img.standard_metadata,
                "ome": None,
                "reader": bioio_img.metadata,
            },
            "pixel_sizes": None,
            "dimensions": bioio_img.dims,
            "dtype": bioio_img.dtype,
        },
    }
    with contextlib.suppress(NotImplementedError):
        layer_attributes["metadata"]["bioio_metadata"]["ome"] = (
            bioio_img.ome_metadata
        )
    # NOTE: https://github.com/bioio-devs/bioio/issues/25 issue for adding units
    with contextlib.suppress(NotImplementedError):
        layer_attributes["metadata"]["pixel_sizes"] = (
            bioio_img.physical_pixel_sizes
        )
    # Load image in napari-friendly order (with RGB dimension last)
    layer_data = aiod_utils.io.load_image_data(
        bioio_img,
        as_dask=True,
        dim_order=dim_order,
        rgb_as_channels=False,
    )
    # Napari layer data tuple
    # TODO: multichannel images could be split into separate layers here
    return [
        (
            layer_data,
            layer_attributes,
            "image",
        )
    ]


def get_rle_reader(path: Union[str, list[str]]):
    # If the path is a list, take the first element to get the extension
    if isinstance(path, list):
        path = path[0]
    path = Path(path)
    # Return our reader if the extension is in the accepted extensions
    return rle_reader if path.suffix in aiod_utils.rle.EXTENSIONS else None


def rle_reader(paths: Union[str, list[str]]):
    if not isinstance(paths, list):
        paths = [paths]
    # Container for Napari layers
    layer_tuples = []
    # Loop over each given file
    for path in paths:
        # Load & decode the RLE
        encoded_mask = aiod_utils.rle.load_encoding(path)
        # NOTE: Only doing this to insert type as metadata
        mask_type = aiod_utils.rle.check_rle_type(encoded_mask)
        mask, metadata = aiod_utils.rle.decode(encoded_mask, mask_type)
        # Flatten metadata if needed
        if "metadata" in metadata:
            metadata = metadata["metadata"]
        # TODO: Handle scale metadata if given for downsampled masks
        layer_tuples.append(
            (
                mask,
                {
                    "name": Path(path).stem,
                    "metadata": {
                        "path": path,
                        "mask_type": mask_type,
                        **metadata,
                    },
                },
                "labels",
            )
        )
    return layer_tuples


def rle_writer(path: str, data, attributes: dict) -> list[str]:
    # Check the suffix and add if necessary
    # NOTE: Not sure if needed and what Napari handles
    suffix = Path(path).suffix
    if suffix == "":
        path = Path(path).with_suffix(".rle")
    elif suffix not in aiod_utils.rle.EXTENSIONS:
        return None
    # Encode the data
    # TODO: Get the attributes and insert as metadata into the RLE
    metadata = attributes.get("metadata", {})
    encoded_mask = aiod_utils.rle.encode(data, metadata=metadata)
    # Save the encoded mask
    aiod_utils.rle.save_encoding(encoded_mask, path)
    return [path]
