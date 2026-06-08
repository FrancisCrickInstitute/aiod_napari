from aiod_napari.inference.inference_widget import Inference
from aiod_napari.evaluation import Evaluation


def test_inference(make_napari_viewer):
    viewer = make_napari_viewer()

    inf_widget = Inference(viewer)

    assert inf_widget is not None


def test_evaluation(make_napari_viewer):
    viewer = make_napari_viewer()

    eval_widget = Evaluation(viewer)

    assert eval_widget is not None
