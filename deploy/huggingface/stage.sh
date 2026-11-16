#!/bin/sh
# Copy the files the Hugging Face Space repository holds into one folder: deploy/huggingface/stage.sh <out-dir>
# No document, index or model is copied: the image downloads the models, start.py the private index.
set -e
out=$1
root=$(cd "$(dirname "$0")/../.." && pwd)
rm -rf "$out"
mkdir -p "$out"
cp "$root/deploy/huggingface/Dockerfile" "$root/deploy/huggingface/README.md" "$root/deploy/huggingface/start.py" \
   "$root/requirements.txt" "$out/"
cp -r "$root/src" "$out/src"
find "$out/src" -name __pycache__ -prune -exec rm -rf {} +
