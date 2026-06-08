import napari
from napari.layers import Labels
from napari.utils.notifications import show_error, show_warning
import numpy as np

from aiod_napari.inference import ExportWidget
from aiod_napari.postprocessing.merge_masks import MergeMasks
from aiod_napari.postprocessing.filter_masks import FilterMasks
from aiod_napari.postprocessing.morph_masks import MorphMasks
from aiod_napari.widget_classes import MainWidget


class Postprocess(MainWidget):
    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__(
            napari_viewer=napari_viewer,
            title="Postprocess Masks",
            tooltip="""
Postprocess masks using various methods. This includes merging, splitting, and filtering masks.
""",
        )

        self.register_widget(
            FilterMasks(viewer=self.viewer, parent=self, expanded=False)
        )

        self.register_widget(
            MergeMasks(viewer=self.viewer, parent=self, expanded=False)
        )

        self.register_widget(
            MorphMasks(viewer=self.viewer, parent=self, expanded=False)
        )

        self.register_widget(
            ExportWidget(viewer=self.viewer, parent=self, expanded=False)
        )

    def _get_selected_layers(self):
        # NOTE: We leave each function to error handle the length of the response (beyond 0)
        layers = [
            i for i in self.viewer.layers.selection if isinstance(i, Labels)
        ]
        layer_sizes = [layer.data.shape for layer in layers]
        if len(set(layer_sizes)) > 1:
            show_warning("Selected label layers are not the same shape — smaller layers will be resized.")
        if len(layers) == 0:
            show_error("No label layers selected!")
        return layers

    def _check_layers_binary(self, layers) -> bool:
        res = []
        # Most robust approach would be np.unique, but that's expensive
        for layer in layers:
            # We grab the max of the dtype as sometimes masks are e.g. [0, 255] for uint8
            dtype_max = np.iinfo(layer.data.dtype).max
            res.append((layer.data.max() in [1, dtype_max]))
        # If all layers are binary, return True
        return all(res)

    def _binarize_mask(self, layer):
        return layer.data > 0
