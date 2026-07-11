#!/usr/bin/env bash
# The repo's node_modules is installed from Windows, so it only contains the
# win32 native binaries for rollup, esbuild, and the Tauri CLI. Running
# `npm install` from WSL would prune those win32 packages and break the
# Windows build, so instead this script drops the matching Linux native
# packages into node_modules directly (npm pack + untar, no reify).
#
# Run once from WSL after any `npm install` done on Windows:
#   bash scripts/setup-linux-node-modules.sh
set -euo pipefail

cd "$(dirname "$0")/.."

ver() { node -p "require('./node_modules/$1/package.json').version"; }

PKGS=(
  "@rollup/rollup-linux-x64-gnu@$(ver rollup)"
  "@esbuild/linux-x64@$(ver esbuild)"
  "@tauri-apps/cli-linux-x64-gnu@$(ver @tauri-apps/cli)"
)

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

for spec in "${PKGS[@]}"; do
  name="${spec%@*}"
  dest="node_modules/$name"
  if [ -f "$dest/package.json" ]; then
    echo "already present: $spec"
    continue
  fi
  echo "installing $spec"
  tarball="$(cd "$TMP" && npm pack "$spec" --silent)"
  mkdir -p "$dest"
  tar -xzf "$TMP/$tarball" -C "$dest" --strip-components=1
done

echo "done — vite / tauri CLI are now runnable from Linux."
