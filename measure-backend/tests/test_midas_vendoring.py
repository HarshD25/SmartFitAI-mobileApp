"""
tests/test_midas_vendoring.py

Tests for the vendored-MiDaS detection logic in measure_server.py.
This deliberately does NOT test torch.hub.load() itself (that needs
real network/filesystem state and is exercised manually via
setup_midas.py - see that script's module docstring for the full
rationale on why a vendored local clone is necessary at all in
network environments that block the GitHub REST API).

What's tested here is just the routing logic: does
measure_server.py correctly detect a vendored clone when one exists,
and correctly report "not vendored" when it doesn't, so it knows
which load path to take.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_vendored_path_detected_when_hubconf_exists(monkeypatch, tmp_path):
    import measure_server as ms

    fake_vendor_root = tmp_path / "measure-backend"
    fake_midas_dir = fake_vendor_root / "vendor" / "MiDaS"
    fake_midas_dir.mkdir(parents=True)
    (fake_midas_dir / "hubconf.py").write_text("# stub hubconf for test purposes\n")

    monkeypatch.setattr(ms, "__file__", str(fake_vendor_root / "measure_server.py"))
    result = ms._vendored_midas_path()
    assert result == str(fake_midas_dir)


def test_no_vendored_path_when_directory_absent(monkeypatch, tmp_path):
    import measure_server as ms

    fake_vendor_root = tmp_path / "measure-backend-empty"
    fake_vendor_root.mkdir(parents=True)

    monkeypatch.setattr(ms, "__file__", str(fake_vendor_root / "measure_server.py"))
    result = ms._vendored_midas_path()
    assert result is None


def test_vendored_path_ignores_a_file_named_midas(monkeypatch, tmp_path):
    """
    A stray file (not a directory) at the expected vendor path - e.g.
    a leftover from an interrupted clone - should not be reported as
    a usable vendored clone, since torch.hub.load(source="local")
    requires an actual repo directory.
    """
    import measure_server as ms

    fake_vendor_root = tmp_path / "measure-backend-bad"
    vendor_dir = fake_vendor_root / "vendor"
    vendor_dir.mkdir(parents=True)
    (vendor_dir / "MiDaS").write_text("not a real clone")

    monkeypatch.setattr(ms, "__file__", str(fake_vendor_root / "measure_server.py"))
    result = ms._vendored_midas_path()
    assert result is None


def test_vendored_path_rejects_incomplete_clone(monkeypatch, tmp_path):
    """
    Regression test for a real failure: a `git clone` that fails or is
    interrupted partway through (e.g. a dropped connection) can leave
    behind a vendor/MiDaS directory that exists but has no hubconf.py
    inside it. The original version of this function only checked
    os.path.isdir(), so it reported this broken directory as a valid
    vendored clone - and torch.hub.load(source="local") would then
    fail deep inside torch's own loader with a confusing
    FileNotFoundError for hubconf.py, instead of measure_server.py
    cleanly falling back to the normal torch.hub download path.
    """
    import measure_server as ms

    fake_vendor_root = tmp_path / "measure-backend-incomplete"
    fake_midas_dir = fake_vendor_root / "vendor" / "MiDaS"
    fake_midas_dir.mkdir(parents=True)
    # Directory exists, has some files, but NOT hubconf.py - simulating
    # a clone that died partway through.
    (fake_midas_dir / "README.md").write_text("partial clone\n")

    monkeypatch.setattr(ms, "__file__", str(fake_vendor_root / "measure_server.py"))
    result = ms._vendored_midas_path()
    assert result is None
