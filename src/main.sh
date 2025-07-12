# Graphviz and ImageMagick are required for this script

GIF_VIEWER_PATH="/Applications/Lyn.app"  # OPTIONAL, image viewer binary

cd "$(dirname "$0")"/..
cargo run

for scenario in output/*; do
  if [ -d "$scenario" ]; then
    echo "Creating GIF for this scenario: $scenario"

    for f in "$scenario"/frame_*.dot; do
      dot -Tpng "$f" -o "${f%.dot}.png"
    done

    scenario_name=$(basename "$scenario")
    magick -delay 50 -loop 0 "$scenario"/frame_*.png "$scenario"/${scenario_name}.gif
    echo "Gif created: $scenario/${scenario_name}.gif"
    if [ -e "$GIF_VIEWER_PATH" ]; then
      open -a "$GIF_VIEWER_PATH" "$scenario"/${scenario_name}.gif
    else
      open "$scenario"/${scenario_name}.gif
    fi
  fi
done

