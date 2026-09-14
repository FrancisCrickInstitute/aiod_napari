# AIoD Napari Plugin

Napari plugin part of [AI OnDemand (AIoD)](https://franciscrickinstitute.github.io/aiod_docs) to provide an accessible interface for running deep learning models on images via our [Nextflow pipeline](https://github.com/FrancisCrickInstitute/Segment-Flow).

## Installation
See our latest installation information in our [documentation](https://franciscrickinstitute.github.io/aiod_docs/sections/front_ends/napari_plugin/#installation)!

In general, you should see the [official Napari installation instructions](https://napari.org/stable/getting_started/installation.html#install-python-package) first to ensure Napari is installed with an appropriate Qt backend.

### `uv`
A `uv.lock` file is provided for smoother installation:

```bash
uv add aiod_napari
```

### Conda
We have also provided a conda environment file to install all the dependencies for this plugin. To install the environment, run the following command:

```bash
conda env create -f ai-od.yml
```

## Getting Started
For a step-by-step walkthrough, please see our [First Segmentation guide](https://franciscrickinstitute.github.io/aiod_docs/sections/getting_started/first_segmentation/).

## Usage
For general usage of the plugin, see the [documentation](https://franciscrickinstitute.github.io/aiod_docs/sections/front_ends/napari_plugin/).

## Contributing
Contributions are very welcome! Please see our [AIoD Developer Guide](https://franciscrickinstitute.github.io/aiod_docs/sections/contributing/developing/) for general advice and information.

### Development setup
Install with test dependencies using either:

```bash
uv sync                        # uv dependency group (dev)
pip install -e ".[testing]"    # pip/tox extra (testing)
```

Both install the same test dependencies (pytest, tox, `napari[qt]`, ...), but allow for either `uv` or plain `pip`/tox workflows. (Note that, at present, the CI uses the `testing` extra via tox.)

## Issues
If you encounter any problems, please raise an issue along with a detailed description!
