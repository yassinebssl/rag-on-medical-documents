"""Assemble the hand-in folder of the booklet (section 4) and its .rar copy, then list what is missing.

Run from the repo root: .venv/bin/python scripts/build_flash_drive.py --group NN --out DIR

It writes DIR/PROJECT_<NN>_RAG_Medical/ with the seven numbered folders, copies the
documents under their booklet names, clones this repository (its checked-out branch,
with the full history) into 03_CODE/, and makes DIR/PROJECT_<NN>_RAG_Medical.rar with
the `rar` command. It then checks that the folder and the archive hold the same files,
and prints each required file that is still missing and each script in 03_CODE/ that is
not a .py file. It never deletes anything: DIR/PROJECT_<NN>_RAG_Medical must not exist.
"""
import argparse
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TITLE = "RAG_Medical"

# (source in the repo, destination in the project folder), with the booklet's file names.
FILES = [
    ("docs/report/main.pdf", "01_REPORT/final_report.pdf"),
    ("docs/technical/architecture_diagram.png", "01_REPORT/figures/architecture_diagram.png"),
    ("docs/technical/main.pdf", "02_DOCUMENTATION/technical_documentation.pdf"),
    ("docs/technical/architecture_diagram.png", "02_DOCUMENTATION/architecture_diagram.png"),
    ("docs/user_manual/main.pdf", "02_DOCUMENTATION/user_manual.pdf"),
    ("docs/defense/final/main.pdf", "04_DEFENSE/presentation.pdf"),
    ("docs/defense/final/presentation.pptx", "04_DEFENSE/presentation.pptx"),
    ("docs/defense/demo_video.mp4", "04_DEFENSE/demo_video.mp4"),
    ("docs/dataset_links.md", "05_DATASETS/dataset_links.md"),
]
# Folders the team fills by hand: the signed peer evaluations and the app screenshots.
FOLDERS = ["06_PEER_EVALUATION", "07_EXTRA/screenshots"]
SCRIPT_EXTENSIONS = {".js", ".ts", ".sh", ".bash", ".bat", ".ps1", ".r", ".rb", ".pl", ".java", ".c", ".cpp", ".m"}


def missing(repo, files=FILES, folders=FOLDERS, project=None):
    """Return the destinations whose source is absent from repo, and the hand-filled folders still empty.

    Folders are looked up in project (the assembled folder), or in repo when project is None.
    """
    out = [f"{dest} (from {src})" for src, dest in files if not os.path.exists(os.path.join(repo, src))]
    base = project or repo
    for folder in folders:
        path = os.path.join(base, folder)
        if not (os.path.isdir(path) and os.listdir(path)):
            out.append(f"{folder}/ (empty)")
    return out


def wrong_extensions(code_dir):
    """Return the files under code_dir, outside .git, that are scripts in another language than Python."""
    bad = []
    for dirpath, dirnames, filenames in os.walk(code_dir):
        dirnames[:] = sorted(d for d in dirnames if d != ".git")
        for name in sorted(filenames):
            if os.path.splitext(name)[1].lower() in SCRIPT_EXTENSIONS:
                bad.append(os.path.relpath(os.path.join(dirpath, name), code_dir))
    return bad


def listing(folder):
    """Return every file under folder as a sorted list of paths relative to it, with / separators."""
    files = []
    for dirpath, _, filenames in os.walk(folder):
        for name in filenames:
            files.append(os.path.relpath(os.path.join(dirpath, name), folder).replace(os.sep, "/"))
    return sorted(files)


def main():
    """Build the folder and the archive, compare them, and print what is missing."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--group", required=True, help="group number, as on the flash drive label")
    parser.add_argument("--out", required=True, help="directory to write into, e.g. the mounted flash drive")
    args = parser.parse_args()
    name = f"PROJECT_{args.group}_{TITLE}"
    project = os.path.join(os.path.abspath(args.out), name)
    archive = project + ".rar"
    if os.path.exists(project) or os.path.exists(archive):
        raise SystemExit(f"{project} or its .rar already exists: move it away and run again.")
    if shutil.which("rar") is None:
        raise SystemExit("the rar command is missing (the booklet asks for .rar, not zip or 7z): install rar.")

    for src, dest in FILES:
        if os.path.exists(os.path.join(ROOT, src)):
            os.makedirs(os.path.dirname(os.path.join(project, dest)), exist_ok=True)
            shutil.copy2(os.path.join(ROOT, src), os.path.join(project, dest))
    for folder in FOLDERS:
        os.makedirs(os.path.join(project, folder), exist_ok=True)
    subprocess.run(["git", "clone", "--quiet", "--no-hardlinks", ROOT, os.path.join(project, "03_CODE")], check=True)
    subprocess.run(["rar", "a", "-r", "-idq", "-ep1", archive, project], check=True)

    rar_files = subprocess.run(["unrar", "lb", archive], check=True, capture_output=True, text=True).stdout.split("\n")
    in_rar = sorted(f[len(name) + 1:].replace("\\", "/") for f in rar_files if f.startswith(name + "/"))
    in_rar = [f for f in in_rar if os.path.isfile(os.path.join(project, f))]
    same = in_rar == listing(project)
    print(f"wrote {project}\nwrote {archive}")
    print("folder and archive hold the same files" if same else "the folder and the archive DIFFER: rebuild them")
    for item in missing(ROOT, project=project):
        print(f"missing: {item}")
    for path in wrong_extensions(os.path.join(project, "03_CODE")):
        print(f"not .py or .ipynb (-5 points each): 03_CODE/{path}")
    print("Also: label the drive, print and sign the Annex A checklist, and open both copies on another computer.")
    sys.exit(0 if same else 1)


if __name__ == "__main__":
    main()
