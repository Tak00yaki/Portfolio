#!/bin/sh
# Builds the F1 Race Simulator for the browser with pygbag and copies it into
# the website's public/f1/ folder, so it is served at khushikhan.com/f1/.
#
# Run from anywhere:  sh projects/scripts/build_f1_web.sh
# Then deploy the site as usual:  npm run deploy
set -e

PROJECTS="$(cd "$(dirname "$0")/.." && pwd)"
SITE_PUBLIC="$PROJECTS/../public/f1"
WORK="$(mktemp -d)"

# pygbag packages a folder whose entry point is main.py
mkdir "$WORK/f1"
cp "$PROJECTS/f1.py" "$WORK/f1/main.py"

"$PROJECTS/.venv/bin/python" -m pygbag --build --title "F1 Race Simulator" "$WORK/f1"

rm -rf "$SITE_PUBLIC"
mkdir -p "$SITE_PUBLIC"
cp -R "$WORK/f1/build/web/." "$SITE_PUBLIC/"
rm -rf "$WORK"

echo "Built F1 web version into $SITE_PUBLIC"
