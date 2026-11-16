"""Start the app on the Hugging Face Space: fetch the index, then run src/app.py.

The index (chroma_db/, built by src/index.py) lives in a private dataset repository, named by the
INDEX_REPO variable, because it holds the text of the documents. HF_TOKEN, a Space secret with read
access to it, downloads it at each start, so neither the Space files nor its image contain it.
"""
import os
import runpy
import sys

from huggingface_hub import snapshot_download

APP = os.path.dirname(os.path.abspath(__file__))
snapshot_download(repo_id=os.environ["INDEX_REPO"], repo_type="dataset", token=os.environ["HF_TOKEN"],
                  local_dir=os.path.join(APP, "chroma_db"))
os.environ["HF_HUB_OFFLINE"] = "1"  # the models are in the image; nothing else is downloaded
sys.path.insert(0, os.path.join(APP, "src"))
runpy.run_path(os.path.join(APP, "src", "app.py"), run_name="__main__")
