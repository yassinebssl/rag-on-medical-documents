"""Tests for scripts/build_flash_drive.py: the hand-in folder layout and its checks."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import build_flash_drive  # noqa: E402


def test_files_go_to_the_booklet_names():
    dests = {dest for _, dest in build_flash_drive.FILES}
    assert {"01_REPORT/final_report.pdf", "02_DOCUMENTATION/technical_documentation.pdf",
            "02_DOCUMENTATION/architecture_diagram.png", "02_DOCUMENTATION/user_manual.pdf",
            "04_DEFENSE/presentation.pdf", "04_DEFENSE/presentation.pptx", "04_DEFENSE/demo_video.mp4",
            "05_DATASETS/dataset_links.md"} <= dests


def test_missing_lists_absent_sources_and_empty_folders(tmp_path):
    (tmp_path / "a.pdf").write_text("x")
    files = [("a.pdf", "01_REPORT/final_report.pdf"), ("b.mp4", "04_DEFENSE/demo_video.mp4")]
    folders = ["06_PEER_EVALUATION", "07_EXTRA/screenshots"]
    (tmp_path / "07_EXTRA" / "screenshots").mkdir(parents=True)
    (tmp_path / "07_EXTRA" / "screenshots" / "app.png").write_text("x")
    assert build_flash_drive.missing(str(tmp_path), files, folders) == [
        "04_DEFENSE/demo_video.mp4 (from b.mp4)", "06_PEER_EVALUATION/ (empty)"]


def test_wrong_extensions_flags_scripts_that_are_not_python(tmp_path):
    for name in ("src/app.py", "notebooks/01_x.ipynb", "build.js", "run.sh", "docs/report/main.tex"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "hook.sh").write_text("x")
    assert build_flash_drive.wrong_extensions(str(tmp_path)) == ["build.js", "run.sh"]


def test_listing_is_relative_and_sorted(tmp_path):
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "y.txt").write_text("x")
    (tmp_path / "a.txt").write_text("x")
    assert build_flash_drive.listing(str(tmp_path)) == ["a.txt", "b/y.txt"]
