"""
Tests for how NxfWidget locates Nextflow execution profiles.

Profiles are mirrored into the package from Segment-Flow by the sync_profiles
workflow, so the packaged-profiles path is the one users hit by default and the
only thing standing between a packaging regression and an unusable dropdown.
"""

from importlib.resources import files
from pathlib import Path

import pytest


def _make_nxf_widget(viewer):
    _, widget = viewer.window.add_plugin_dock_widget("aiod-napari", "Inference")
    return widget.subwidgets["nxf"]


def _dropdown_items(nxf):
    box = nxf.nxf_profile_box
    return [box.itemText(i) for i in range(box.count())]


class TestBundledProfiles:
    """Default path: no AIOD_NXF_REPO, profiles read from the installed package."""

    def test_profiles_dir_is_the_packaged_one(
        self, make_napari_viewer_proxy, monkeypatch
    ):
        monkeypatch.delenv("AIOD_NXF_REPO", raising=False)
        nxf = _make_nxf_widget(make_napari_viewer_proxy())

        expected = Path(files("aiod_napari").joinpath("nxf_profiles"))
        assert nxf.nxf_profiles_dir == expected
        assert nxf.nxf_profiles_dir.is_dir()

    def test_dropdown_matches_packaged_profiles(
        self, make_napari_viewer_proxy, monkeypatch
    ):
        monkeypatch.delenv("AIOD_NXF_REPO", raising=False)
        nxf = _make_nxf_widget(make_napari_viewer_proxy())

        on_disk = sorted(p.stem for p in nxf.nxf_profiles_dir.glob("*.conf"))
        assert on_disk, (
            f"No .conf files under {nxf.nxf_profiles_dir} — profiles were not packaged"
        )
        assert _dropdown_items(nxf) == on_disk

    def test_local_profile_is_available(self, make_napari_viewer_proxy, monkeypatch):
        """The plugin is unusable out of the box without a local profile, so this
        is worth pinning even though the profile set is otherwise synced in."""
        monkeypatch.delenv("AIOD_NXF_REPO", raising=False)
        nxf = _make_nxf_widget(make_napari_viewer_proxy())

        assert "local" in _dropdown_items(nxf)


class TestAiodNxfRepoOverride:
    """AIOD_NXF_REPO path: profiles read from an external Segment-Flow checkout."""

    def test_profiles_read_from_override_checkout(
        self, make_napari_viewer_proxy, monkeypatch, tmp_path
    ):
        profiles = tmp_path / "profiles"
        profiles.mkdir()
        for name in ("zeta", "alpha"):
            (profiles / f"{name}.conf").write_text("")
        # Must not be picked up — only *.conf are profiles
        (profiles / "notes.txt").write_text("")

        monkeypatch.setenv("AIOD_NXF_REPO", str(tmp_path))
        nxf = _make_nxf_widget(make_napari_viewer_proxy())

        assert nxf.nxf_repo == tmp_path
        assert nxf.nxf_profiles_dir == profiles
        assert _dropdown_items(nxf) == ["alpha", "zeta"]

    def test_empty_checkout_raises(
        self, make_napari_viewer_proxy, monkeypatch, tmp_path
    ):
        (tmp_path / "profiles").mkdir()
        monkeypatch.setenv("AIOD_NXF_REPO", str(tmp_path))

        with pytest.raises(
            FileNotFoundError, match=r"No Nextflow \(.conf\) profiles found"
        ):
            _make_nxf_widget(make_napari_viewer_proxy())
