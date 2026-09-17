"""
Tests for how NxfWidget selects a Segment-Flow revision to run.

Revision selection only applies to the GitHub-hosted pipeline (the default when
AIOD_NXF_REPO is unset) - it's a developer/deployment concern, not something users
pick per run, so there's deliberately no UI for it, only AIOD_NXF_REV.
"""

from aiod_napari.inference.nxf import DEFAULT_NXF_REV


def _make_nxf_widget(viewer):
    _, widget = viewer.window.add_plugin_dock_widget("aiod-napari", "Inference")
    return widget.subwidgets["nxf"]


class TestGithubRepoRevision:
    """No AIOD_NXF_REPO: running the GitHub-hosted pipeline, so -r applies."""

    def test_defaults_to_pinned_revision(self, make_napari_viewer_proxy, monkeypatch):
        monkeypatch.delenv("AIOD_NXF_REPO", raising=False)
        monkeypatch.delenv("AIOD_NXF_REV", raising=False)
        nxf = _make_nxf_widget(make_napari_viewer_proxy())

        assert nxf.nxf_rev == DEFAULT_NXF_REV
        nxf_cmd, _ = nxf.setup_inference(nxf_params={})
        assert nxf_cmd.endswith(f"run {nxf.nxf_repo} -r {DEFAULT_NXF_REV} -latest")

    def test_env_var_overrides_default(self, make_napari_viewer_proxy, monkeypatch):
        monkeypatch.delenv("AIOD_NXF_REPO", raising=False)
        monkeypatch.setenv("AIOD_NXF_REV", "dev")
        nxf = _make_nxf_widget(make_napari_viewer_proxy())

        assert nxf.nxf_rev == "dev"
        nxf_cmd, _ = nxf.setup_inference(nxf_params={})
        assert nxf_cmd.endswith(f"run {nxf.nxf_repo} -r dev -latest")

    def test_empty_env_vars_are_treated_as_unset(
        self, make_napari_viewer_proxy, monkeypatch
    ):
        """`AIOD_NXF_REPO=` (set but empty) must behave like an unset var, not like
        Path("") - which confusingly reports the cwd as an existing directory."""
        monkeypatch.setenv("AIOD_NXF_REPO", "")
        monkeypatch.setenv("AIOD_NXF_REV", "")
        nxf = _make_nxf_widget(make_napari_viewer_proxy())

        assert nxf.nxf_repo == "FrancisCrickInstitute/Segment-Flow"
        assert nxf.nxf_rev == DEFAULT_NXF_REV


class TestAiodNxfRepoOverride:
    """AIOD_NXF_REPO path: a local checkout, so revision selection doesn't apply."""

    def test_revision_is_unset(self, make_napari_viewer_proxy, monkeypatch, tmp_path):
        (tmp_path / "profiles").mkdir()
        (tmp_path / "profiles" / "local.conf").write_text("")
        monkeypatch.setenv("AIOD_NXF_REPO", str(tmp_path))
        monkeypatch.setenv("AIOD_NXF_REV", "dev")
        nxf = _make_nxf_widget(make_napari_viewer_proxy())

        assert nxf.nxf_rev is None
        nxf_cmd, _ = nxf.setup_inference(nxf_params={})
        assert " -r " not in nxf_cmd
        assert " -latest" not in nxf_cmd
