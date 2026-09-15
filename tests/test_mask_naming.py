"""
Locks the mask-filename contract between aiod_napari and Segment-Flow.

napari predicts mask filenames before they exist (file watcher, mask reload,
overwrite globbing) while Segment-Flow independently derives them from the
image_id column that add_image_ids.py writes. If the two ever disagree, the
watcher silently finds nothing and masks never appear - so pin the agreement
here rather than relying on both sides being edited together.
"""

import types
from pathlib import Path

import pytest
from aiod_utils.io import (
    get_combined_mask_name,
    get_image_id,
    get_mask_name,
    get_mask_prefix,
    validate_image_ids,
)

from aiod_napari.inference import inference_widget as inference_widget_module
from aiod_napari.inference.inference_widget import Inference

RUN_HASH = "1a2b3c4d5e6f7a8b"


def segment_flow_mask_name(img_paths, target, run_hash, prep_hash=""):
    """
    Reproduce the Segment-Flow side end to end for the given run.

    ``img_paths`` is the CSV napari hands over (add_image_ids.py fills in the
    image_id column); the return value mirrors getMaskName in main.nf.
    """
    # Key/lookup by Path rather than str: img_paths are POSIX-style literals,
    # but target arrives as whatever Path subclass the widget stored (e.g.
    # WindowsPath, whose str() uses backslashes) - str() would only agree on
    # POSIX.
    image_ids = {
        Path(p): i.value
        for p, i in zip(img_paths, validate_image_ids(img_paths), strict=True)
    }
    image_id = image_ids[Path(target)]
    prep_suffix = f"_{prep_hash}" if prep_hash else ""
    return f"{image_id}{prep_suffix}_masks_{run_hash}"


@pytest.fixture
def widget():
    """
    Enough of an Inference instance to exercise the naming helpers.

    Deliberately avoids a napari viewer - these helpers are pure functions of
    the path, the run hash and the model selection.
    """
    return types.SimpleNamespace(
        run_hash=RUN_HASH,
        subwidgets={
            "model": types.SimpleNamespace(
                get_task_model_variant_name=lambda executed: "mito-SAM-vit_h"
            )
        },
    )


@pytest.fixture
def real_widget(make_napari_viewer_proxy, monkeypatch):
    """The actual Inference widget, for tests that go through record construction."""
    viewer = make_napari_viewer_proxy()
    _, widget = viewer.window.add_plugin_dock_widget("aiod-napari", "Inference")
    monkeypatch.setattr(widget, "store_settings", lambda: None)
    monkeypatch.setattr(
        widget.subwidgets["model"],
        "get_task_model_variant_name",
        lambda executed=True: "mito-SAM-vit_h",
    )
    widget.run_hash = RUN_HASH
    return widget


def record_mask_stem(widget, record):
    """
    Every file this run writes for the record starts with this - what nxf.py
    globs against, built the same way it builds it.
    """
    return get_mask_name(
        run_hash=widget.run_hash,
        image_id=record["image_id"],
        prep_hash=record["prep_hash"],
    )


def select_and_build(widget, paths):
    """Select the given images and return the records built for them."""
    widget.subwidgets["data"].image_path_dict = {
        get_image_id(p): Path(p) for p in paths
    }
    widget.get_img_mask_preps()
    return widget.img_mask_info


class TestNapariMatchesSegmentFlow:
    """
    Goes through the real record construction rather than the name builders
    directly: the adaptation under test is napari resolving a path to an
    image_id one at a time, versus Segment-Flow resolving the whole CSV.
    """

    def test_single_image(self, real_widget):
        paths = ["/data/expA/cells.ome.tiff"]
        record = select_and_build(real_widget, paths)[0]
        assert (
            real_widget._get_final_mask_name(record["image_id"], record["prep_hash"])
            == f"{segment_flow_mask_name(paths, paths[0], RUN_HASH)}_all.rle"
        )

    def test_colliding_stems_in_one_run(self, real_widget):
        # The case that broke: same stem, different extension, one run
        paths = ["/data/expA/cells.tiff", "/data/expA/cells.png"]
        for record in select_and_build(real_widget, paths):
            expected = segment_flow_mask_name(paths, record["img_path"], RUN_HASH)
            assert (
                real_widget._get_final_mask_name(
                    record["image_id"], record["prep_hash"]
                )
                == f"{expected}_all.rle"
            )

    def test_colliding_stems_when_nextflow_only_gets_a_subset(self, real_widget):
        # napari predicts names over every loaded image, but only images
        # missing masks reach the CSV (see check_masks). The prediction must
        # not depend on which side saw which images.
        loaded = ["/data/expA/cells.tiff", "/data/expA/cells.png"]
        csv_subset = ["/data/expA/cells.png"]
        records = select_and_build(real_widget, loaded)
        (record,) = [r for r in records if r["img_path"] == Path(csv_subset[0])]
        expected = segment_flow_mask_name(csv_subset, csv_subset[0], RUN_HASH)
        assert (
            real_widget._get_final_mask_name(record["image_id"], record["prep_hash"])
            == f"{expected}_all.rle"
        )

    def test_with_preprocessing(self, real_widget, monkeypatch):
        paths = ["/data/expA/cells.tiff", "/data/expA/cells.png"]
        prep_set = [{"name": "CLAHE", "params": {"clipLimit": 3.0}}]
        monkeypatch.setattr(
            real_widget.subwidgets["preprocess"], "get_all_options", lambda: [prep_set]
        )
        for record in select_and_build(real_widget, paths):
            expected = segment_flow_mask_name(
                paths,
                record["img_path"],
                RUN_HASH,
                prep_hash=record["prep_hash"],
            )
            assert record["prep_hash"] is not None
            assert (
                real_widget._get_final_mask_name(
                    record["image_id"], record["prep_hash"]
                )
                == f"{expected}_all.rle"
            )


class TestMaskNamesAreUnique:
    def test_same_stem_different_extension_do_not_share_a_mask_file(self, real_widget):
        records = select_and_build(
            real_widget, ["/data/expA/cells.tiff", "/data/expA/cells.png"]
        )
        names = {
            real_widget._get_final_mask_name(r["image_id"], r["prep_hash"])
            for r in records
        }
        assert len(names) == 2

    def test_watcher_prefix_matches_the_real_filename(self, real_widget):
        # watch_mask_files filters on the prefix parsed off each file, so the
        # stored prefix has to be the same string
        path = "/data/expA/cells.ome.tiff"
        (record,) = select_and_build(real_widget, [path])
        written = (
            f"{segment_flow_mask_name([path], path, RUN_HASH)}_x0-64_y0-64_z0-1.rle"
        )
        assert written.rsplit("_masks_", 1)[0] == record["mask_prefix"]


class TestLayerNamesStayReadable:
    def test_layer_name_uses_the_bare_stem(self, widget):
        # The extension belongs in filenames, not in the napari layer list
        layer_name = Inference._get_mask_layer_name(
            widget, get_image_id("/data/expA/cells.ome.tiff"), executed=True
        )
        assert layer_name.startswith("cells_masks_")
        assert "ome_tiff" not in layer_name

    def test_colliding_stems_get_distinct_layer_names(self, widget):
        # napari suffixes a duplicate layer name and the un-suffixed lookup then
        # returns the wrong layer, so these must not collide
        names = {
            Inference._get_mask_layer_name(
                widget,
                get_image_id(p),
                executed=True,
                ambiguous_stems=frozenset({"cells"}),
            )
            for p in ("/data/expA/cells.tiff", "/data/expA/cells.png")
        }
        assert len(names) == 2
        assert all(
            n.startswith(("cells_tiff_masks_", "cells_png_masks_")) for n in names
        )

    def test_unambiguous_stems_are_untouched_by_the_fallback(self, widget):
        # Only the colliding stems pay the readability cost
        assert Inference._get_mask_layer_name(
            widget,
            get_image_id("/data/expA/nucleus.tiff"),
            executed=True,
            ambiguous_stems=frozenset({"cells"}),
        ).startswith("nucleus_masks_")

    def test_layer_name_reuses_the_shared_prefix_rule(self, widget):
        # The leading half is built by get_mask_prefix, so a change to how the
        # prep hash is appended reaches layer names too
        image_id = get_image_id("/data/expA/cells.tiff")
        layer_name = Inference._get_mask_layer_name(
            widget, image_id, prep_hash="deadbeef", executed=True
        )
        assert layer_name.startswith(f"{get_mask_prefix('cells', 'deadbeef')}_masks_")

    def test_layer_name_prefix_follows_the_ambiguity_choice(self, widget):
        # Whichever id form is chosen, the prefix rule is applied to it
        image_id = get_image_id("/data/expA/cells.tiff")
        ambiguous = Inference._get_mask_layer_name(
            widget, image_id, prep_hash="deadbeef", ambiguous_stems=frozenset({"cells"})
        )
        assert ambiguous.startswith(get_mask_prefix(image_id, "deadbeef"))

    def test_mask_filenames_are_unique_regardless_of_the_layer_name(self, widget):
        # Filenames never depend on the ambiguity check - always the full id
        names = {
            Inference._get_final_mask_name(widget, get_image_id(p))
            for p in ("/data/expA/cells.tiff", "/data/expA/cells.png")
        }
        assert len(names) == 2


class TestAmbiguousStems:
    def _widget_with(self, *paths):
        return types.SimpleNamespace(
            subwidgets={
                "data": types.SimpleNamespace(
                    image_path_dict={get_image_id(p): Path(p) for p in paths}
                )
            }
        )

    def test_no_collision_gives_an_empty_set(self):
        widget = self._widget_with("/data/cells.tiff", "/data/nucleus.tiff")
        assert Inference._get_ambiguous_stems(widget) == frozenset()

    def test_same_stem_different_extension_is_flagged(self):
        widget = self._widget_with("/data/cells.tiff", "/data/cells.png")
        assert Inference._get_ambiguous_stems(widget) == frozenset({"cells"})

    def test_only_the_colliding_stem_is_flagged(self):
        widget = self._widget_with(
            "/data/cells.tiff", "/data/cells.png", "/data/nucleus.tiff"
        )
        assert Inference._get_ambiguous_stems(widget) == frozenset({"cells"})

    def test_compound_extension_stem_collides_with_the_plain_one(self):
        # cells.ome.tiff and cells.tiff both reduce to the stem "cells"
        widget = self._widget_with("/data/cells.ome.tiff", "/data/cells.tiff")
        assert Inference._get_ambiguous_stems(widget) == frozenset({"cells"})


class TestImageIdIsPathLocal:
    def test_get_mask_name_agrees_with_get_image_id_alone(self):
        # The property the fix rests on: a lone get_image_id call is always
        # correct, so callers never need to know the rest of the run
        path = "/data/expA/cells.ome.tiff"
        assert get_mask_name(run_hash=RUN_HASH, image_id=get_image_id(path)) == (
            get_mask_name(run_hash=RUN_HASH, image_path=path)
        )


class TestGetImgMaskPrepsWiring:
    """
    get_img_mask_preps is the only producer of layer_name now (check_masks and
    insert_final_masks consume what it stored), so the uniqueness guarantee has
    to hold in the real widget, not just in the helper.
    """

    @pytest.fixture
    def widget(self, make_napari_viewer_proxy, monkeypatch):
        viewer = make_napari_viewer_proxy()
        _, widget = viewer.window.add_plugin_dock_widget("aiod-napari", "Inference")
        monkeypatch.setattr(widget, "store_settings", lambda: None)
        monkeypatch.setattr(
            widget.subwidgets["model"],
            "get_task_model_variant_name",
            lambda executed=True: "mito-SAM-vit_h",
        )
        widget.run_hash = RUN_HASH
        return widget

    def _select(self, widget, *paths):
        widget.subwidgets["data"].image_path_dict = {
            get_image_id(p): Path(p) for p in paths
        }

    def test_colliding_stems_produce_unique_layer_names(self, widget):
        # Previously both got "cells_masks_...", so check_masks reported the
        # second image's mask as already existing and it was dropped from the run
        self._select(widget, "/data/cells.tiff", "/data/cells.png")
        widget.get_img_mask_preps()
        names = [i["layer_name"] for i in widget.img_mask_info]
        assert len(set(names)) == len(names) == 2

    def test_layer_names_stay_clean_without_a_collision(self, widget):
        self._select(widget, "/data/cells.ome.tiff", "/data/nucleus.tiff")
        widget.get_img_mask_preps()
        assert {i["layer_name"].split("_masks_")[0] for i in widget.img_mask_info} == {
            "cells",
            "nucleus",
        }

    def test_mask_prefixes_are_unique_either_way(self, widget):
        self._select(widget, "/data/cells.tiff", "/data/cells.png")
        widget.get_img_mask_preps()
        prefixes = [i["mask_prefix"] for i in widget.img_mask_info]
        assert len(set(prefixes)) == len(prefixes) == 2

    def test_record_carries_the_image_id(self, widget):
        self._select(widget, "/data/cells.ome.tiff")
        widget.get_img_mask_preps()
        (record,) = widget.img_mask_info
        assert record["image_id"] == get_image_id(record["img_path"])
        # The value is what Segment-Flow's image_id column will hold, so the
        # mask filename must be built from it
        assert record["mask_prefix"] == record["image_id"].value

    def test_record_image_id_matches_the_progress_dict_key(self, widget):
        # update_masks increments progress_dict[image_id]; nxf.py keys it from
        # the path. Same key, or progress silently KeyErrors
        self._select(widget, "/data/cells.tiff", "/data/cells.png")
        widget.get_img_mask_preps()
        for record in widget.img_mask_info:
            assert record["image_id"] == get_image_id(record["img_path"])

    def test_no_preprocessing_yields_one_record_per_image(self, widget):
        # The collapsed loop must not multiply records when options is None
        self._select(widget, "/data/cells.tiff", "/data/nucleus.tiff")
        widget.get_img_mask_preps()
        assert len(widget.img_mask_info) == 2
        assert all(r["prep_set"] is None for r in widget.img_mask_info)
        assert all(r["preprocess_str"] is None for r in widget.img_mask_info)

    def test_prefix_index_covers_every_record(self, widget):
        # The watcher filters on this index, so a missing entry means masks for
        # that image are ignored for the whole run
        self._select(widget, "/data/cells.tiff", "/data/cells.png")
        widget.get_img_mask_preps()
        assert len(widget.mask_info_by_prefix) == len(widget.img_mask_info) == 2
        for record in widget.img_mask_info:
            assert widget.mask_info_by_prefix[record["mask_prefix"]] is record

    def test_mask_prefix_is_the_mask_stem_without_the_run_hash(self, widget):
        # The prefix is built independently of the stem, so pin that they agree
        # and that the stem carries the full run hash the pipeline was given
        self._select(widget, "/data/cells.ome.tiff")
        widget.get_img_mask_preps()
        (record,) = widget.img_mask_info
        stem = record_mask_stem(widget, record)
        assert stem == f"{record['mask_prefix']}_masks_{RUN_HASH}"

    def test_prefix_index_matches_a_real_substack_filename(self, widget):
        # What the watcher and update_masks actually do: parse a file off disk
        # and look it up
        self._select(widget, "/data/cells.tiff")
        widget.get_img_mask_preps()
        (record,) = widget.img_mask_info
        stem = record_mask_stem(widget, record)
        written = Path(f"{stem}_x0-64_y0-64_z0-1.rle")
        assert (
            widget.mask_info_by_prefix[written.stem.rsplit("_masks_", 1)[0]] is record
        )

    def test_prefix_survives_an_image_named_with_masks_in_it(self, widget):
        # rsplit, not split: the old code unpacked a 2-way split and would
        # ValueError on this filename rather than mis-parse it
        self._select(widget, "/data/foo_masks_bar.tiff")
        widget.get_img_mask_preps()
        (record,) = widget.img_mask_info
        assert record["mask_prefix"] == "foo_masks_bar_tiff"
        stem = record_mask_stem(widget, record)
        written = Path(f"{stem}_x0-64_y0-64_z0-1.rle")
        assert (
            widget.mask_info_by_prefix[written.stem.rsplit("_masks_", 1)[0]] is record
        )

    def test_preprocessing_yields_one_record_per_set(self, widget, monkeypatch):
        # The other half of the collapsed branch: a no-op set keeps the
        # unsuffixed names so they still match what Nextflow writes
        prep_sets = [
            [],
            [{"name": "CLAHE", "params": {"clipLimit": 3.0, "tileGridSize": [8, 8]}}],
        ]
        monkeypatch.setattr(
            widget.subwidgets["preprocess"], "get_all_options", lambda: prep_sets
        )
        self._select(widget, "/data/cells.tiff")
        widget.get_img_mask_preps()
        assert len(widget.img_mask_info) == 2
        noop, clahe = widget.img_mask_info
        assert noop["prep_set"] is None and noop["preprocess_str"] is None
        assert clahe["prep_set"] == prep_sets[1]
        assert clahe["preprocess_str"] is not None
        # Distinct masks per set, both still carrying the full image_id
        assert noop["mask_prefix"] != clahe["mask_prefix"]
        assert noop["layer_name"] != clahe["layer_name"]
        assert all(
            r["mask_prefix"].startswith("cells_tiff") for r in widget.img_mask_info
        )

    def test_layer_name_is_stable_when_called_with_a_subset(self, widget):
        # create_mask_layers/remove_mask_layers pass only some of the selection;
        # the name must not change or the layer can no longer be found
        self._select(widget, "/data/cells.tiff", "/data/cells.png")
        widget.get_img_mask_preps()
        full = {i["img_path"]: i["layer_name"] for i in widget.img_mask_info}
        widget.get_img_mask_preps(img_paths=[Path("/data/cells.png")])
        subset = {i["img_path"]: i["layer_name"] for i in widget.img_mask_info}
        assert subset[Path("/data/cells.png")] == full[Path("/data/cells.png")]


class TestWatcherFileScoping:
    """
    The mask directory is shared by every run of a model variant, and run_hash
    does not include the image paths - so filtering on the (image, prep) prefix
    alone matches substacks left behind by an earlier interrupted run.
    """

    @pytest.fixture
    def widget(self, real_widget, tmp_path, monkeypatch):
        real_widget.subwidgets["nxf"].mask_dir_path = tmp_path
        select_and_build(real_widget, ["/data/cells.tiff"])
        self._make_watcher_synchronous(real_widget, monkeypatch)
        return real_widget

    def _make_watcher_synchronous(self, widget, monkeypatch):
        """
        Run the watcher body in the calling thread for one pass.

        The file filtering lives inside the thread_worker-wrapped loop, so
        there is nothing else to call it through.
        """
        self.yielded = []

        def collect(connect=None):
            def decorator(func):
                return lambda *args: self.yielded.extend(func(*args))

            return decorator

        monkeypatch.setattr(inference_widget_module, "thread_worker", collect)
        monkeypatch.setattr(inference_widget_module.time, "sleep", lambda _: None)
        monkeypatch.setattr(widget, "create_mask_layers", lambda: None)
        # Stop the loop after the first pass
        widget.subwidgets["nxf"].progress_dict = {}
        widget.subwidgets["nxf"].total_substacks = 0

    def _watched_files(self, widget):
        widget.watch_mask_files()
        return [fpath for batch in self.yielded for fpath in batch]

    def _write(self, widget, name):
        path = widget.subwidgets["nxf"].mask_dir_path / name
        path.touch()
        return path

    def test_picks_up_this_run_substacks(self, widget):
        (record,) = widget.img_mask_info
        stem = record_mask_stem(widget, record)
        wanted = self._write(widget, f"{stem}_x0-64_y0-64_z0-1.rle")
        assert self._watched_files(widget) == [wanted]

    def test_ignores_another_runs_substacks(self, widget):
        # Same image and preprocessing, different params - so the same prefix
        # but a different run hash
        (record,) = widget.img_mask_info
        other = get_mask_name(
            run_hash="0" * 16,
            image_id=record["image_id"],
            prep_hash=record["prep_hash"],
        )
        self._write(widget, f"{other}_x0-64_y0-64_z0-1.rle")
        assert self._watched_files(widget) == []

    def test_ignores_the_combined_mask(self, widget):
        (record,) = widget.img_mask_info
        stem = record_mask_stem(widget, record)
        self._write(widget, get_combined_mask_name(stem, "rle"))
        assert self._watched_files(widget) == []

    def test_ignores_images_not_in_this_run(self, widget):
        # Same params (so same run hash), an image this run does not cover
        other = get_mask_name(
            run_hash=widget.run_hash, image_path="/data/elsewhere.tiff"
        )
        self._write(widget, f"{other}_x0-64_y0-64_z0-1.rle")
        assert self._watched_files(widget) == []
