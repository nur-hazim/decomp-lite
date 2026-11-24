#!/usr/bin/env bash
# Install Ghidra into GHIDRA_INSTALL_DIR (default ~/opt/ghidra; no sudo).
# Reuse only an installation of the requested version.
#
# Usage:
#   bash scripts/install_ghidra.sh [GHIDRA_VERSION]
# Default version: 12.1.2 (PUBLIC release). Requires JDK 21+ on PATH.
#
# After install, add this to .env:
#   GHIDRA_INSTALL_DIR=$HOME/opt/ghidra

set -euo pipefail

VERSION="${1:-12.1.2}"
[[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "Invalid version: $VERSION" >&2; exit 1; }
RELEASE_TAG="Ghidra_${VERSION}_build"
# The actual zip is named with a build date, so we resolve via the GitHub
# redirect-to-asset endpoint.
DEST="${GHIDRA_INSTALL_DIR:-$HOME/opt/ghidra}"

# JDK check.
if ! command -v java >/dev/null 2>&1; then
  echo "ERROR: java not on PATH. Install JDK 21+ first (e.g. apt install openjdk-21-jdk)." >&2
  exit 1
fi
JVER=$(java -version 2>&1 | awk -F\" '/version/ {print $2}' | cut -d. -f1)
if [[ -z "$JVER" || "$JVER" -lt 21 ]]; then
  echo "ERROR: JDK 21+ required (have JDK $JVER). Install openjdk-21-jdk." >&2
  exit 1
fi

install_jython() {
  # 12.1.2 ships Jython as a bundled extension rather than a core feature.
  # Activate the bundled version so our @runtime Jython post-script can run.
  local extension
  for extension in "$DEST"/Extensions/Ghidra/*_Jython.zip; do
    [[ -f "$extension" ]] || continue
    mkdir -p "$DEST/Ghidra/Extensions"
    unzip -qo "$extension" -d "$DEST/Ghidra/Extensions"
  done
  if [[ ! -d "$DEST/Ghidra/Features/Jython" && ! -d "$DEST/Ghidra/Extensions/Jython" ]]; then
    echo "Ghidra $VERSION has no usable bundled Jython runtime." >&2
    exit 1
  fi
}

if [[ -e "$DEST" ]]; then
  INSTALLED=$(sed -n 's/^application.version=//p' "$DEST/Ghidra/application.properties" 2>/dev/null | tr -d '\r' || true)
  if [[ "$INSTALLED" != "$VERSION" || ! -x "$DEST/support/analyzeHeadless" ]]; then
    echo "Destination $DEST is not a working Ghidra $VERSION installation." >&2
    echo 'Set GHIDRA_INSTALL_DIR to a new version-specific directory.' >&2
    exit 1
  fi
  install_jython
  echo "Ghidra $VERSION already installed at $DEST"
  echo "GHIDRA_INSTALL_DIR=$DEST"
  exit 0
fi

mkdir -p "$(dirname "$DEST")"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

echo "Resolving Ghidra ${VERSION} release URL..."
# GitHub redirects /releases/<tag>/download/Ghidra_<v>_PUBLIC_<datestamp>.zip
# but the datestamp isn't predictable. We use the API to find the asset.
ASSET_URL=$(curl --retry 3 -fsSL "https://api.github.com/repos/NationalSecurityAgency/ghidra/releases/tags/${RELEASE_TAG}" \
  | python3 -c 'import json,sys; v=sys.argv[1]; r=json.load(sys.stdin); print(next((a["browser_download_url"] for a in r.get("assets",[]) if a["name"].startswith("ghidra_"+v+"_PUBLIC_") and a["name"].endswith(".zip")), ""))' "$VERSION")
if [[ -z "$ASSET_URL" ]]; then
  echo "Could not find a release asset for ${RELEASE_TAG}." >&2
  echo "Browse https://github.com/NationalSecurityAgency/ghidra/releases and pass the right version." >&2
  exit 1
fi
echo "Downloading: $ASSET_URL"
curl --retry 3 -fSL -o "$TMP/ghidra.zip" "$ASSET_URL"
echo "Extracting..."
unzip -q "$TMP/ghidra.zip" -d "$TMP"

# The archive extracts a directory like ghidra_12.1.2_PUBLIC.
EXTRACTED=$(find "$TMP" -maxdepth 1 -type d -name "ghidra_${VERSION}_PUBLIC")
if [[ -z "$EXTRACTED" ]]; then
  echo "Unexpected archive layout under $TMP" >&2
  ls -la "$TMP" >&2
  exit 1
fi
INSTALLED=$(sed -n 's/^application.version=//p' "$EXTRACTED/Ghidra/application.properties" | tr -d '\r')
[[ "$INSTALLED" == "$VERSION" ]] || { echo "Archive version mismatch: $INSTALLED" >&2; exit 1; }
mv "$EXTRACTED" "$DEST"
install_jython

if [[ ! -x "$DEST/support/analyzeHeadless" ]]; then
  echo "Install looks broken: $DEST/support/analyzeHeadless missing" >&2
  exit 1
fi

echo
echo "Ghidra ${VERSION} installed to $DEST"
echo "Add this to your .env (or shell rc):"
echo "  GHIDRA_INSTALL_DIR=$DEST"
