#!/usr/bin/env bash
# bootstrap_model.sh — extract a ViT5 LoRA adapter tarball into ./models/.
#
# Usage:
#   scripts/bootstrap_model.sh                     # uses ./vit5-news-v2.tar.gz
#   scripts/bootstrap_model.sh path/to/file.tar.gz
#
# The repo's web UI (`make api`) defaults to MODEL_PATH=./models/vit5-news-v2
# which is what this script populates. After running it once you can start
# the backend with no HF Hub credentials.
set -euo pipefail

TARBALL=${1:-./vit5-news-v2.tar.gz}
DEST=models

if [[ ! -f "$TARBALL" ]]; then
    echo "error: tarball '$TARBALL' not found." >&2
    echo "       drop the file you downloaded from training (Colab cell 7)" >&2
    echo "       in the repo root, or pass its path as the first argument." >&2
    exit 1
fi

mkdir -p "$DEST"

# Tarballs produced by the training notebook are layered as
#   models/vit5-news-v2/...
# so extract directly to the repo root and let the inner ``models/`` merge
# with our destination directory.
echo "extracting $TARBALL -> $DEST/"
tar -xzf "$TARBALL"

if [[ ! -f "$DEST/vit5-news-v2/adapter_config.json" ]]; then
    echo "error: extraction did not yield $DEST/vit5-news-v2/adapter_config.json" >&2
    echo "       inspect the tarball layout with 'tar -tzf $TARBALL | head'." >&2
    exit 2
fi

echo "ok — adapter ready at ./$DEST/vit5-news-v2/"
echo "    set MODEL_PATH=./$DEST/vit5-news-v2 in your .env (this is the default)."
