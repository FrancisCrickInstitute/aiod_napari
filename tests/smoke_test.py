"""
Small smoke test script to use in release workflow to check that the build
was fine and that the package/plugin can be imported without issues.

Can be run standalone::

    python tests/smoke_test.py

or discovered automatically by pytest.
"""


def run_smoke_tests() -> None:
    """Run all smoke-test assertions."""
    # Check that we can import things
    from aiod_napari.io import get_bioio_reader

    # Check that our bioio stuff is imported and working
    try:
        result = get_bioio_reader("test.completelymadeupextension")
        assert result is None, (
            "Expected get_bioio_reader to return None for unsupported file type"
        )
    except AssertionError:
        raise
    except Exception as e:
        raise AssertionError(
            f"get_bioio_reader raised an unexpected exception: {e}"
        ) from e

    # Check that our submodule has brought in some files
    from importlib.resources import files

    profiles_dir = files("aiod_napari").joinpath("Segment-Flow", "profiles")
    conf_files = [
        p for p in profiles_dir.iterdir() if p.name.endswith(".conf")
    ]
    assert len(conf_files) > 0, (
        f"No .conf profiles found in {profiles_dir} — Segment-Flow submodule was not bundled correctly"
    )


def test_smoke() -> None:
    """Pytest entry-point for the smoke tests."""
    run_smoke_tests()


if __name__ == "__main__":
    run_smoke_tests()
