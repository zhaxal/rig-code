#!/bin/bash
# Fetch the NNArchive model files into models/.
#
# The models are versioned in this repo, so a normal `git clone`/`git pull`
# already brings them down. But large binaries are exactly what a flaky link
# (Wi-Fi on a Pi, for example) tends to drop, leaving models/ empty or partial.
# This script re-fetches them straight from GitHub so you can recover without
# re-cloning. It is idempotent: a file that is already present and the right
# size is skipped.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Pinned to a branch so the raw URLs keep tracking the latest committed models.
REF="${MODELS_REF:-main}"
BASE="https://raw.githubusercontent.com/zhaxal/rig-code/${REF}/models"

# "filename expected_bytes" — keep in sync with the files committed under models/.
MODELS=(
  "tomatoripenessdetectionnano.rvc2.tar.xz 4494792"
  "tomatoripenessdetectionsmall.rvc2.tar.xz 18047696"
)

mkdir -p models

file_size() {
  # Portable byte size (Linux/macOS).
  stat -c%s "$1" 2>/dev/null || stat -f%z "$1" 2>/dev/null || echo 0
}

download() {
  local name="$1" want="$2" dest="models/$1" url="$BASE/$1"
  if [ -f "$dest" ] && [ "$(file_size "$dest")" = "$want" ]; then
    echo "==> $name already present ($want bytes), skipping."
    return 0
  fi
  echo "==> Downloading $name ..."
  local delay=2 tmp
  tmp="$(mktemp "models/.${name}.XXXXXX")"
  for attempt in 1 2 3 4 5; do
    if curl -fL --retry 3 -o "$tmp" "$url"; then
      local got; got="$(file_size "$tmp")"
      if [ -z "$want" ] || [ "$got" = "$want" ]; then
        mv -f "$tmp" "$dest"
        echo "    Saved $dest ($got bytes)."
        return 0
      fi
      echo "    Size mismatch (got $got, want $want), retrying..."
    fi
    echo "    Attempt $attempt failed; retrying in ${delay}s..."
    sleep "$delay"
    delay=$((delay * 2))
  done
  rm -f "$tmp"
  echo "ERROR: could not download $name from $url" >&2
  return 1
}

rc=0
for entry in "${MODELS[@]}"; do
  # shellcheck disable=SC2086
  download $entry || rc=1
done

if [ "$rc" -ne 0 ]; then
  echo "" >&2
  echo "Some models failed to download. Re-run this script once you have a" >&2
  echo "stable connection:  ./download_models.sh" >&2
  exit 1
fi

echo ""
echo "All models present in models/."
