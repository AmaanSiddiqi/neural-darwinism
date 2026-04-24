#!/usr/bin/env bash
# record_demo.sh — run Colony in mock mode and produce demo artifacts for GitHub
# Outputs: demo.gif, cortex_final.png, history.png, demo.cast (if asciinema available)
set -euo pipefail

GENERATIONS=${GENERATIONS:-20}
FRAMES_DIR="frames"
OUTPUT_GIF="demo.gif"
CAST_FILE="demo.cast"
VENV_DIR=".venv"

# Activate virtualenv if present
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
fi

echo "=== Colony Demo ==="
echo "Generations: $GENERATIONS"
echo ""

# Clean previous run
rm -rf "$FRAMES_DIR" cortex_final.png history.png

# --- Terminal recording (optional) ---
if command -v asciinema &> /dev/null; then
    echo "Recording terminal session with asciinema → $CAST_FILE"
    asciinema rec "$CAST_FILE" --overwrite --command \
        "python main.py --generations $GENERATIONS --no-frames"
    echo "Cast saved: $CAST_FILE"
    echo "  Upload with: asciinema upload $CAST_FILE"
    echo ""
fi

# --- Headless mock run with PNG frames ---
echo "Generating PNG frames..."
python main.py --generations "$GENERATIONS"

echo ""
echo "=== Assembling GIF ==="

if command -v ffmpeg &> /dev/null; then
    ffmpeg -y -loglevel warning \
        -framerate 4 \
        -i "${FRAMES_DIR}/gen_%04d.png" \
        -vf "scale=800:-2:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer" \
        "$OUTPUT_GIF"
    echo "GIF saved: $OUTPUT_GIF  ($(du -sh $OUTPUT_GIF | cut -f1))"
elif command -v convert &> /dev/null; then
    convert -delay 25 -loop 0 -layers optimize "${FRAMES_DIR}/gen_*.png" "$OUTPUT_GIF"
    echo "GIF saved: $OUTPUT_GIF  ($(du -sh $OUTPUT_GIF | cut -f1))"
else
    echo "ffmpeg/ImageMagick not found — frames saved in ${FRAMES_DIR}/ (use any GIF tool)"
fi

echo ""
echo "=== Demo artifacts ==="
for f in "$OUTPUT_GIF" cortex_final.png history.png; do
    [ -f "$f" ] && echo "  $f  ($(du -sh $f | cut -f1))"
done
[ -f "$CAST_FILE" ] && echo "  $CAST_FILE  ($(du -sh $CAST_FILE | cut -f1))"
echo ""
echo "Drop demo.gif and cortex_final.png/history.png into your GitHub README."
