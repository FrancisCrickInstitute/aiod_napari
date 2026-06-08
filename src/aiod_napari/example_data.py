from napari.utils.notifications import show_info
import requests

from aiod_utils.io import load_image_data
from aiod_napari.utils import get_plugin_cache


def load_example_data():
    cache_dir, _ = get_plugin_cache()
    # Construct path to example data
    example_data_path = cache_dir / "em_20nm_z_40_145.tif"
    # Check if the example data is already downloaded
    if not example_data_path.exists():
        # Download the example data
        try:
            req = requests.get(
                "https://zenodo.org/records/7936982/files/em_20nm_z_40_145.tif",
                stream=True,
            )
            req.raise_for_status()
            with open(example_data_path, "wb") as f:
                for chunk in req.iter_content(chunk_size=8192):
                    f.write(chunk)
        except:
            show_info(
                f"Failed to download example data to {example_data_path}. May be due to insufficient permissions, space, or network issues. Please try again later."
            )
            return
    # Load the example data
    img = load_image_data(example_data_path)
    # https://github.com/krentzd/napari-clemreg/blob/main/napari_clemreg/clemreg/sample_data.py#L24
    metadata = {
        "ImageDescription": "\nunit=micron\nspacing=0.02\n",
        "XResolution": 50,
        "YResolution": 50,
        "path": str(example_data_path),
    }
    return [(img, {"name": "em_20nm_z_40_145", "metadata": metadata})]
