import pandas as pd
import numpy as np
from PIL import Image
import os

# -------------------- User Settings --------------------

CSV_FILE = "pixel_fft_structured_2.csv"

# Canvas size (set to your video resolution)
CANVAS_WIDTH = 1920
CANVAS_HEIGHT = 1080

# Which color(s) to render:
# "start", "middle", "end", or "all"
COLOR_MODE = "all"

# Pixel block sizes (1 = single pixel, 20 = 20x20 block, 100 = 100x100 block)
PIXEL_SIZES = [1, 20, 100]

# Output folder
OUTPUT_DIR = "visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Column names from your CSV
COORD_COL = "X:Y"
COLOR_COL = "Start_Avg_End_Color"

# -------------------- Load Data --------------------

df = pd.read_csv(CSV_FILE)
print(f"Loaded {len(df)} rows from {CSV_FILE}")

# -------------------- Helpers --------------------

def parse_xy(xy_str):
    """Parse 'X:Y' into (x, y) ints."""
    x_str, y_str = xy_str.split(":")
    return int(x_str), int(y_str)

def parse_colors(color_str):
    """
    Parse 'R_G_B:R_G_B:R_G_B' into
    (start_rgb, avg_rgb, end_rgb), each a tuple (r,g,b)
    """
    parts = color_str.split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid color format: {color_str}")

    def parse_one(p):
        r, g, b = p.split("_")
        return int(r), int(g), int(b)

    start_rgb = parse_one(parts[0])
    avg_rgb   = parse_one(parts[1])
    end_rgb   = parse_one(parts[2])

    return start_rgb, avg_rgb, end_rgb

# Pre-parse coordinates and colors for speed
coords = []
start_colors = []
avg_colors = []
end_colors = []

for _, row in df.iterrows():
    x, y = parse_xy(row[COORD_COL])
    sc, ac, ec = parse_colors(row[COLOR_COL])

    coords.append((x, y))
    start_colors.append(sc)
    avg_colors.append(ac)
    end_colors.append(ec)

coords = np.array(coords)

# -------------------- Decide which modes to render --------------------

if COLOR_MODE == "all":
    modes = ["start", "middle", "end"]
else:
    modes = [COLOR_MODE]

# Map mode name to actual color list
color_map = {
    "start": start_colors,
    "middle": avg_colors,
    "end": end_colors,
}

# -------------------- Rendering --------------------

def draw_blocks(canvas, coords, colors, block_size):
    """
    Draw colored squares of size block_size x block_size
    centered (top-left anchored) at each (x, y).
    """
    h, w, _ = canvas.shape

    for (x, y), (r, g, b) in zip(coords, colors):
        # Skip if completely out of bounds
        if x < 0 or y < 0 or x >= w or y >= h:
            continue

        # Compute block extents
        x0 = x
        y0 = y
        x1 = min(w, x0 + block_size)
        y1 = min(h, y0 + block_size)

        canvas[y0:y1, x0:x1, :] = [r, g, b]

# -------------------- Generate Images --------------------

for mode in modes:
    colors = color_map[mode]

    for block_size in PIXEL_SIZES:
        # Create white canvas
        canvas = np.ones((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8) * 255

        # Draw blocks
        draw_blocks(canvas, coords, colors, block_size)

        # Save image
        out_name = f"canvas_{mode}_block{block_size}.png"
        out_path = os.path.join(OUTPUT_DIR, out_name)

        Image.fromarray(canvas).save(out_path)
        print(f"Saved: {out_path}")

print("✅ Done generating canvases.")
