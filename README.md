# AIoD Napari Plugin

[![PyPI](https://img.shields.io/pypi/v/aiod-napari.svg)](https://pypi.org/project/aiod-napari/)
[![Python versions](https://img.shields.io/pypi/pyversions/aiod-napari.svg)](https://pypi.org/project/aiod-napari/)
[![Tests](https://github.com/FrancisCrickInstitute/aiod_napari/actions/workflows/test_ci.yml/badge.svg)](https://github.com/FrancisCrickInstitute/aiod_napari/actions/workflows/test_ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-aiod__docs-1f6feb.svg)](https://franciscrickinstitute.github.io/aiod_docs/sections/front_ends/napari_plugin/)

Napari plugin part of [AI OnDemand (AIoD)](https://franciscrickinstitute.github.io/aiod_docs) to provide an accessible interface for running deep learning models on images via our [Nextflow pipeline](https://github.com/FrancisCrickInstitute/Segment-Flow).

Point it at your images, pick a model and a task, and press go — the underlying pipeline handles splitting the data, running the model (locally or on HPC, depending on *profile*), caching the results, and loading the masks back into Napari. It contributes three widgets: **Inference**, **Postprocess**, and **Evaluation**.

## Requirements

- Python 3.11 or 3.12
- Napari with a working Qt backend — see the [official Napari installation instructions](https://napari.org/stable/getting_started/installation.html#install-python-package) first
- [Nextflow](https://www.nextflow.io/) and Conda, to actually run models — see the docs [Prerequisites](https://franciscrickinstitute.github.io/aiod_docs/sections/getting_started/#prerequisites)

## Installation

See our latest installation information in our [documentation](https://franciscrickinstitute.github.io/aiod_docs/sections/front_ends/napari_plugin/#installation), which covers local, Crick NEMO and non-Crick HPC setups.

### `uv`

```bash
uv add aiod_napari
```

### Conda

We have also provided a conda environment file to install all the dependencies for this plugin. To install the environment, run the following command:

```bash
conda env create -f ai-od.yml
```

## Quick start

```bash
uv venv aiod-env && source aiod-env/bin/activate
uv pip install "napari[all]" aiod_napari
napari
```

Then, in Napari: `Plugins → AI OnDemand → Inference`. If you don't have data to hand, `File → Open Sample → AI OnDemand → Example FIB SEM data` gives you a volume to try it on.

For the full step-by-step walkthrough, from install to exported masks, see our [First Segmentation guide](https://franciscrickinstitute.github.io/aiod_docs/sections/getting_started/first_segmentation/).

## Documentation

Full documentation for AIoD lives at **[franciscrickinstitute.github.io/aiod_docs](https://franciscrickinstitute.github.io/aiod_docs/)**.

| Topic | Link |
| --- | --- |
| Plugin overview and installation | [Napari Plugin](https://franciscrickinstitute.github.io/aiod_docs/sections/front_ends/napari_plugin/) |
| Running a segmentation | [Inference widget](https://franciscrickinstitute.github.io/aiod_docs/sections/front_ends/napari_plugin/inference/) |
| Filtering, merging and morphology | [Postprocessing widget](https://franciscrickinstitute.github.io/aiod_docs/sections/front_ends/napari_plugin/postprocess/) |
| Comparing against ground truth | [Evaluation widget](https://franciscrickinstitute.github.io/aiod_docs/sections/front_ends/napari_plugin/evaluation/) |
| Which models are available | [Available Models](https://franciscrickinstitute.github.io/aiod_docs/sections/model_registry/models/) |
| Caching, substacks, model families | [AIoD Concepts](https://franciscrickinstitute.github.io/aiod_docs/sections/concepts/) |
| Something went wrong | [Troubleshooting](https://franciscrickinstitute.github.io/aiod_docs/sections/support/troubleshooting/) |

## Contributing

Contributions are very welcome! Please see our [AIoD Developer Guide](https://franciscrickinstitute.github.io/aiod_docs/sections/contributing/developing/) for general advice and information, including how to run against a local checkout of Segment-Flow.

### Development setup

Install with test dependencies using either:

```bash
uv sync                        # uv dependency group (dev)
pip install -e ".[testing]"    # pip/tox extra (testing)
```

Both install the same test dependencies (pytest, tox, `napari[qt]`, ...), but allow for either `uv` or plain `pip`/tox workflows. (Note that, at present, the CI uses the `testing` extra via tox.)

Run the tests with:

```bash
uv run pytest
```

## Support

If you encounter any problems, please [open an issue](https://github.com/FrancisCrickInstitute/aiod_napari/issues) along with a detailed description! Check [Troubleshooting](https://franciscrickinstitute.github.io/aiod_docs/sections/support/troubleshooting/) first — it covers the most common failures at each stage of a run.

## License

MIT — see [LICENSE](LICENSE).
