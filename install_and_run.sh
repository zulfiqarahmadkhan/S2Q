#!/usr/bin/env bash
# Download (or update) S2Q from GitHub into ~/S2Q, then run its setup + launcher
# (install_and_run_local.sh), which installs conda if needed.
set -u
REPO_URL="https://github.com/zulfiqarahmadkhan/S2Q.git"
INSTALL_DIR="$HOME/S2Q"

command -v git >/dev/null 2>&1 || { echo "[ERROR] Git is not installed (macOS: run  xcode-select --install)."; exit 1; }
if [ -d "$INSTALL_DIR/.git" ]; then
    echo "[*] Updating S2Q..."
    git -C "$INSTALL_DIR" pull --ff-only || echo "[WARN] Could not pull updates, using the existing code."
else
    echo "[*] Downloading S2Q..."
    git clone "$REPO_URL" "$INSTALL_DIR" || { echo "[ERROR] Could not clone $REPO_URL"; exit 1; }
fi
exec bash "$INSTALL_DIR/install_and_run_local.sh"
