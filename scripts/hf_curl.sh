#!/usr/bin/env bash
# Resumable download of every file of a Hugging Face model repo with curl (no Python env needed).
# Usage: hf_curl.sh <org/name> <dest_dir>
set -euo pipefail
repo=$1; dest=$2; mkdir -p "$dest"
files=$(curl -sf "https://huggingface.co/api/models/$repo" | python3 -c "import json,sys; print(\"\n\".join(s[\"rfilename\"] for s in json.load(sys.stdin)[\"siblings\"]))")
for f in $files; do
  case "$f" in *.h5|*.msgpack|*.ot|*onnx*|*.mlmodel|*tf_model*|*flax_model*|*rust_model*|coreml/*) continue;; esac
  mkdir -p "$dest/$(dirname "$f")"
  curl -sfL -C - --retry 10 --retry-delay 3 -o "$dest/$f" "https://huggingface.co/$repo/resolve/main/$f" &
done
wait
echo "done $repo -> $dest"; ls -la "$dest"
