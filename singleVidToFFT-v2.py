import cv2
import numpy as np
import csv
import random
import os

# -------------------- CSV Metadata initialisation --------------------
# These correspond to the first 20 columns in the CSV.
# If a value is computed algorithmically later, we initialize it to "null" here.

TEST_NUMBER = 1
TREE_SPECIES = "null"                 # Unknown / not provided
SCRIPT_NAME = os.path.basename(__file__)
VIDEO_FILE = "MVI_0001.MP4"

# These are filled after loading the video
START_FRAME = "null"
END_FRAME = "null"

NUM_FREQUENCIES = 128                 # How many FFT bins we keep per set
MODE = "pixel"                        # "cluster" / "pixel" / "other"
NUM_CLUSTERS = "null"                 # Only meaningful if MODE == "cluster"

# Column 9: X:Y pixel location -> generated per row
# Column 10: Start:Avg:End color -> generated per row

DISTANCE_FROM_TREE = "null"           # External measurement, not computed here
WIND_SPEED = "null"                   # External measurement
MOISTURE_CONTENT = "null"             # External measurement
DEAD_OR_ALIVE = "null"                # "Alive" / "Dead" or unknown

# Columns 15–20 are reserved / unused for now
BLANK15 = "null"
BLANK16 = "null"
BLANK17 = "null"
BLANK18 = "null"
BLANK19 = "null"
BLANK20 = "null"

# --------- Reproducibility and Processing Parameters --------
# Fixed seeds make the random pixel sampling and any NumPy randomness reproducible
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

OUTPUT_CSV = "pixel_fft_structured_2.csv"

NUM_FRAMES_TO_PROCESS = 256           # Length of each time series
SECONDS_TO_SKIP = 0.5                 # Skip initial shaky frames
N_SAMPLES = 1000                      # How many random pixels to sample

# Batch size for FFT processing (how many pixel time series to FFT at once)
# Useful if you later want to control memory usage on very large datasets
BATCH_SIZE = 512/4

# Round FFT magnitudes to this many decimals (helps file size + reproducibility)
ROUND_DECIMALS = 10

# -------------------- Load Video --------------------
cap = cv2.VideoCapture(VIDEO_FILE)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open {VIDEO_FILE}")

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
skip_frames = int(fps * SECONDS_TO_SKIP)

# Jump to the starting frame after the skip
cap.set(cv2.CAP_PROP_POS_FRAMES, skip_frames)

frames = []
for _ in range(NUM_FRAMES_TO_PROCESS):
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)

cap.release()

if len(frames) == 0:
    raise RuntimeError("No frames were loaded after skipping.")

frames = np.array(frames)  # Shape: (F, H, W, 3)
num_frames, H, W, _ = frames.shape

# Now we can fill in the algorithmic metadata
START_FRAME = skip_frames
END_FRAME = skip_frames + num_frames - 1

print(f"Loaded {num_frames} frames of size {W}x{H}")

# -------------------- Random Pixel Sampling --------------------
# Reproducible because of the fixed seed above
coords = [(random.randint(0, W - 1), random.randint(0, H - 1)) for _ in range(N_SAMPLES)]

# -------------------- Prepare Frames --------------------
start_frame_img = frames[0]
middle_frame_img = frames[len(frames)//2]
end_frame_img = frames[-1]
avg_frame_img = np.mean(frames, axis=0).astype(np.uint8)

# Convert BGR -> grayscale (OpenCV uses BGR ordering)
gray_frames = np.dot(frames[..., :3], [0.1140, 0.5870, 0.2989]).astype(np.float32)  # (F, H, W)

def bgr_to_rgb_str_nocomma(frame, x, y):
    """
    Convert a BGR pixel to an RGB string with NO commas.
    Format: R_G_B
    This avoids CSV parsing issues later.
    """
    b, g, r = frame[y, x]
    return f"{r}_{g}_{b}"

# -------------------- Build Time Series Matrix (Batch) --------------------
# Each row = one pixel, each column = one frame in time
# Shape: (N_SAMPLES, num_frames)
ts_matrix = np.zeros((N_SAMPLES, num_frames), dtype=np.float32)

for i, (x, y) in enumerate(coords):
    ts_matrix[i, :] = gray_frames[:, y, x]

# Remove mean from each pixel time series (removes DC component)
ts_matrix = ts_matrix - np.mean(ts_matrix, axis=1, keepdims=True)

# Apply Blackman window to each time series (reduces spectral leakage)
window = np.blackman(num_frames).astype(np.float32)
ts_matrix = ts_matrix * window[None, :]

# -------------------- Batch FFT --------------------
# Compute FFTs in one go (or in chunks later if you use BATCH_SIZE)
fft_vals = np.fft.rfft(ts_matrix, axis=1)
fft_mag = np.abs(fft_vals)

# Ensure exactly NUM_FREQUENCIES bins per pixel
if fft_mag.shape[1] < NUM_FREQUENCIES:
    pad_width = NUM_FREQUENCIES - fft_mag.shape[1]
    fft_mag = np.pad(fft_mag, ((0, 0), (0, pad_width)))
else:
    fft_mag = fft_mag[:, :NUM_FREQUENCIES]

# Round for reproducibility / smaller files
fft_mag = np.round(fft_mag, ROUND_DECIMALS)

# -------------------- Write CSV --------------------
with open(OUTPUT_CSV, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)

    metadata_header = [
        "TestNumber", "TreeSpecies", "ScriptName", "VideoName",
        "StartFrame:EndFrame", "NumFrequencies", "Mode",
        "NumClusters", "X:Y", "Start_Avg_End_Color",
        "DistanceFromTree", "WindSpeed", "MoistureContent", "DeadOrAlive",
        "Blank15", "Blank16", "Blank17", "Blank18", "Blank19", "Blank20"
    ]

    freq_header = [f"Freq1_{i+1}" for i in range(NUM_FREQUENCIES)] + ["end frequency 1"]

    writer.writerow(metadata_header + freq_header)

    for i, (x, y) in enumerate(coords):
        start_color = bgr_to_rgb_str_nocomma(start_frame_img, x, y)
        avg_color = bgr_to_rgb_str_nocomma(avg_frame_img, x, y)
        end_color = bgr_to_rgb_str_nocomma(end_frame_img, x, y)

        color_triplet = f"{start_color}:{avg_color}:{end_color}"

        metadata_row = [
            TEST_NUMBER,
            TREE_SPECIES,
            SCRIPT_NAME,
            VIDEO_FILE,
            f"{START_FRAME}:{END_FRAME}",
            NUM_FREQUENCIES,
            MODE,
            NUM_CLUSTERS,
            f"{x}:{y}",
            color_triplet,
            DISTANCE_FROM_TREE,
            WIND_SPEED,
            MOISTURE_CONTENT,
            DEAD_OR_ALIVE,
            BLANK15, BLANK16, BLANK17, BLANK18, BLANK19, BLANK20
        ]

        freq_block = fft_mag[i].tolist() + ["end frequency 1"]

        writer.writerow(metadata_row + freq_block)

print(f"✅ Done — reproducible, Blackman-windowed, batch FFT CSV saved to '{OUTPUT_CSV}'")
