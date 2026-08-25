#!/usr/bin/env bash
# gather_movies.sh — copy every *.avi found in subfolders up to the top level
# so the (flat-only) Makefile discovery can pick them up.
#
# Usage: gather_movies.sh <root_dir> <out_dir>
#   <root_dir>  top-level folder to gather into (usually the Makefile's CURDIR)
#   <out_dir>   the outputs directory to EXCLUDE from the search (e.g. results)
#
# Collision-safe: a movie at  batch1/sampleA/YS.avi  is copied to the top
# level as  batch1__sampleA__YS.avi  — the subpath becomes the filename prefix,
# so two same-named movies in different folders never clobber each other, and
# the provenance survives in the output stem (batch1__sampleA__YS_spots.csv).
#
# Idempotent: existing top-level copies are left untouched (cp -n).
# Copies (does not move) — originals stay where they are. Note: AVIs are large;
# on a Dropbox folder this duplicates bytes. Use `make OUT_DIR=...` + process
# in place if you'd rather not duplicate, or delete the top-level copies after.

set -euo pipefail

ROOT="${1:?usage: gather_inputs.sh <root_dir> <out_dir>}"
OUT_DIR="${2:?usage: gather_inputs.sh <root_dir> <out_dir>}"

cd "$ROOT"

copied=0
found=0

# Depth >= 2 => only movies that live inside a subfolder. Exclude the bundled
# Fiji install, our own outputs, our overlay AVIs, and hidden dirs.
while IFS= read -r -d '' src; do
  found=$((found + 1))
  rel="${src#./}"                      # strip leading ./
  dest="${rel//\//__}"                 # subpath -> __-joined filename
  if [ -e "$dest" ]; then
    echo "  skip (exists): $dest"
    continue
  fi
  cp -n "$src" "$dest"
  echo "  copied: $rel -> $dest"
  copied=$((copied + 1))
done < <(
  find . -mindepth 2 -type f -iname '*.avi' \
    -not -iname '*_tracks.avi' \
    -not -path './Fiji/*' \
    -not -path './Fiji.app/*' \
    -not -path "./$OUT_DIR/*" \
    -not -path '*/.*' \
    -print0
)

if [ "$found" -eq 0 ]; then
  echo "  no *.avi found in any subfolder of $ROOT"
else
  echo "  gathered $copied new file(s) ($found found in subfolders)."
fi
