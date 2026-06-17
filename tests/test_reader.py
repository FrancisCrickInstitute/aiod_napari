import importlib.util
from pathlib import Path

import numpy as np
import pytest

from aiod_napari.io import (
    bioio_reader,
    get_bioio_reader,
    prepare_bioio_as_napari_layer,
)


# Helper functions for common assertions
def assert_valid_napari_layer_data(
    layer_data_list, expected_layer_type="image"
):
    """Validate that layer_data_list follows napari layer format."""
    assert isinstance(layer_data_list, list), "Should return a list"
    assert len(layer_data_list) > 0, "Should return at least one layer"

    data, metadata, layer_type = layer_data_list[0]
    assert isinstance(data, np.ndarray) or hasattr(data, "__array__"), (
        "Data should be array-like"
    )
    assert isinstance(metadata, dict), "Metadata should be a dictionary"
    assert "name" in metadata, "Metadata should contain 'name' field"
    assert layer_type == expected_layer_type, (
        f"Layer type should be '{expected_layer_type}'"
    )

    return data, metadata, layer_type


def create_test_image(tmp_path, filename, data, file_format="tiff"):
    """Create a test image file and return the path."""
    test_file = tmp_path / filename

    if file_format == "tiff":
        if not importlib.util.find_spec("tifffile"):
            pytest.skip("tifffile not available")
        import tifffile

        if data.ndim > 2:
            # Add metadata for multi-dimensional files
            axes = {3: "ZYX", 4: "CZYX", 5: "TCZYX"}.get(data.ndim)
            with tifffile.TiffWriter(str(test_file)) as writer:
                writer.write(data, metadata={"axes": axes} if axes else {})
        else:
            tifffile.imwrite(str(test_file), data)
    elif file_format in ["jpeg", "png"]:
        if not importlib.util.find_spec("PIL"):
            pytest.skip("PIL not available")
        from PIL import Image

        Image.fromarray(data).save(str(test_file))

    return test_file


@pytest.fixture
def example_data_path():
    """Fixture to provide path to example data if it exists."""
    from aiod_napari.utils import get_plugin_cache

    cache_dir, _ = get_plugin_cache()
    example_path = cache_dir / "em_20nm_z_40_145.tif"

    if example_path.exists():
        return example_path
    return None


# Parameterized tests for different file formats
@pytest.mark.parametrize(
    "file_format,extension,shape,dtype",
    [
        ("tiff", ".tif", (20, 20), np.uint8),
        ("tiff", ".tiff", (10, 10), np.uint16),
        ("jpeg", ".jpg", (50, 50, 3), np.uint8),
        ("png", ".png", (30, 30), np.uint8),
    ],
)
def test_get_bioio_reader_supported_formats(
    tmp_path, file_format, extension, shape, dtype
):
    """Test that get_bioio_reader returns a callable reader for supported formats."""
    test_data = np.random.randint(0, 255, size=shape, dtype=dtype)
    test_file = create_test_image(
        tmp_path, f"test{extension}", test_data, file_format
    )

    reader = get_bioio_reader(str(test_file))
    assert reader is not None, f"Should return a reader for {extension} files"
    assert callable(reader), "Reader should be callable"


@pytest.mark.parametrize(
    "filename",
    [
        "test.xyz",
        "test.unknown",
        "nonexistent.fake",
    ],
)
def test_get_bioio_reader_returns_none(filename):
    """Test that get_bioio_reader returns None for unsupported files."""
    reader = get_bioio_reader(filename)
    assert reader is None, f"Should return None for: {filename}"


def test_bioio_reader_returns_valid_layer_data(tmp_path):
    """Test that bioio_reader returns proper napari layer data format."""
    test_data = np.random.randint(0, 255, size=(10, 10), dtype=np.uint8)
    test_file = create_test_image(tmp_path, "test_layer.tif", test_data)

    layer_data_list = bioio_reader(str(test_file))
    assert_valid_napari_layer_data(layer_data_list)


@pytest.mark.parametrize(
    "original_data",
    [
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint8),
        np.random.randint(0, 255, size=(5, 10, 20), dtype=np.uint8),  # 3D
        np.random.randint(
            0, 65535, size=(2, 3, 32, 64), dtype=np.uint16
        ),  # 4D CZYX
    ],
)
def test_bioio_reader_preserves_data_values(tmp_path, original_data):
    """Test that bioio_reader preserves original data values."""
    test_file = create_test_image(tmp_path, "test_values.tif", original_data)

    layer_data_list = bioio_reader(str(test_file))
    read_data = layer_data_list[0][0]

    # Handle both eager numpy arrays and lazy dask arrays
    if hasattr(read_data, "compute"):  # type: ignore
        read_data = read_data.compute()  # type: ignore

    # Squeeze singleton dimensions that bioio may add
    read_data_squeezed = np.squeeze(read_data)
    original_squeezed = np.squeeze(original_data)

    np.testing.assert_array_equal(
        read_data_squeezed,
        original_squeezed,
        "Data values should be preserved",
    )


def test_bioio_reader_metadata_structure(tmp_path):
    """Test that bioio_reader includes expected metadata fields."""
    test_data = np.random.randint(0, 255, size=(5, 10, 10), dtype=np.uint8)
    test_file = create_test_image(tmp_path, "test_metadata.tif", test_data)

    layer_data_list = bioio_reader(str(test_file))
    data, metadata, layer_type = assert_valid_napari_layer_data(
        layer_data_list
    )

    assert metadata["name"] == "test_metadata", "Name should match file stem"

    # Check for bioio-specific metadata when BioImage object was used
    if "metadata" in metadata and isinstance(metadata["metadata"], dict):
        inner_metadata = metadata["metadata"]
        assert "bioio_dims" in inner_metadata, "Should have bioio_dims"
        assert "pixel_sizes" in inner_metadata, "Should have pixel_sizes"


def test_bioio_reader_with_example_data(example_data_path):
    """Test bioio_reader with the actual example EM data if available."""
    if example_data_path is None:
        pytest.skip("Example data not available")

    layer_data_list = bioio_reader(str(example_data_path))
    data, metadata, layer_type = assert_valid_napari_layer_data(
        layer_data_list
    )

    assert data.ndim >= 2, "Should have at least 2 dimensions"
    assert metadata["name"] == "em_20nm_z_40_145", (
        "Name should match example data"
    )


def test_prepare_bioio_as_napari_layer_with_numpy_array(tmp_path):
    """Test prepare_bioio_as_napari_layer handles numpy arrays correctly."""
    test_data = np.random.randint(0, 255, size=(50, 50), dtype=np.uint8)
    test_path = Path(tmp_path) / "test_array.png"

    layer_data_list = prepare_bioio_as_napari_layer(test_data, test_path)
    data, metadata, layer_type = assert_valid_napari_layer_data(
        layer_data_list
    )

    assert len(layer_data_list) == 1, (
        "Should return one layer for simple array"
    )
    np.testing.assert_array_equal(data, test_data, "Data should be unchanged")
    assert metadata["name"] == "test_array", "Name should match file stem"


# Edge case and exception handling tests
@pytest.mark.parametrize(
    "invalid_path",
    [
        "",  # Empty string
        "file_without_extension",
        ".hidden",
        "/path/that/does/not/exist/file.xyz",
    ],
)
def test_get_bioio_reader_invalid_paths(invalid_path):
    """Test that get_bioio_reader handles invalid path formats gracefully."""
    reader = get_bioio_reader(invalid_path)
    # Should not crash - either None or callable is acceptable
    assert reader is None or callable(reader), (
        f"Should handle gracefully: {invalid_path}"
    )


def test_get_bioio_reader_with_pathlib_path(tmp_path):
    """Test that get_bioio_reader works with pathlib.Path objects."""
    test_data = np.random.randint(0, 255, size=(10, 10), dtype=np.uint8)
    test_file = create_test_image(tmp_path, "test_pathlib.tif", test_data)

    # Pass as Path object, not string
    reader = get_bioio_reader(test_file)
    assert reader is not None, "Should handle pathlib.Path objects"
    assert callable(reader), "Should return callable reader"

    # Verify it can actually read
    layer_data = reader(test_file)
    assert_valid_napari_layer_data(layer_data)


def test_bioio_reader_file_deleted_after_getting_reader(tmp_path):
    """Test behavior when file is deleted between getting reader and calling it."""
    test_data = np.random.randint(0, 255, size=(10, 10), dtype=np.uint8)
    test_file = create_test_image(tmp_path, "test.tif", test_data)

    # Get the reader (should succeed)
    reader = get_bioio_reader(str(test_file))
    assert reader is not None, "Should get a reader for valid TIFF"

    # Delete the file
    test_file.unlink()

    # Attempting to read should raise an exception
    with pytest.raises((FileNotFoundError, OSError, Exception)):
        reader(str(test_file))
