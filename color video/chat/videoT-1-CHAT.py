import cv2
import os
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import shutil

# ======================= CONFIG =======================

VIDEO_PATH = "input.mp4"   # <<< CHANGE THIS TO YOUR VIDEO FILE
OUTPUT_DIR = "sort_types"

FRAME_RATE = 30
FRAMES_TO_DROP = 60        # 2 seconds at 30fps
FRAMES_PER_CHUNK = 256
NUM_CHUNKS = 30 * 8        # 240

N_CLUSTERS = 8
MINIBATCH_SIZE = 2000
KMEANS_RUNS = 5

SORT_METHODS = ["mean", "max", "energy", "variance", "l2norm", "centroid_freq"]

EPS = 1e-9

# ======================= SETUP =======================

def prepare_output_dirs():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for m in SORT_METHODS:
        os.makedirs(os.path.join(OUTPUT_DIR, m), exist_ok=True)

# ======================= VIDEO =======================

def get_chunk(cap, start_frame, frames_per_chunk):
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    for _ in range(frames_per_chunk):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
    if len(frames) != frames_per_chunk:
        return None
    return np.array(frames, dtype=np.float32)

# ======================= FFT + KMEANS =======================

def compute_fft_stack(frames_array):
    data_centered = frames_array - np.mean(frames_array, axis=0, keepdims=True)
    window = np.blackman(FRAMES_PER_CHUNK)[:, None, None]
    data_windowed = data_centered * window
    ffted = np.abs(np.fft.rfft(data_windowed, axis=0))
    data_for_kmeans = ffted.reshape(ffted.shape[0], -1).T
    return data_for_kmeans

def run_kmeans(data):
    kmeans = MiniBatchKMeans(
        n_clusters=N_CLUSTERS,
        random_state=0,
        batch_size=MINIBATCH_SIZE,
        n_init=KMEANS_RUNS
    )
    kmeans.fit(data)
    return kmeans.cluster_centers_

# ======================= SORTING =======================

def sort_centers(centers, method):
    if method == "mean":
        key = lambda c: np.mean(c)
    elif method == "max":
        key = lambda c: np.max(c)
    elif method == "energy":
        key = lambda c: np.sum(c**2)
    elif method == "variance":
        key = lambda c: np.var(c)
    elif method == "l2norm":
        key = lambda c: np.linalg.norm(c)
    elif method == "centroid_freq":
        freqs = np.arange(len(centers[0]))
        key = lambda c: np.sum(freqs * c) / (np.sum(c) + EPS)
    else:
        raise ValueError("Unknown sort method")

    return sorted(centers, key=key, reverse=True)

# ======================= RENDER =======================

def render_spectrum(sorted_centers, out_path, height=256):
    spec = np.array(sorted_centers, dtype=np.float32)

    spec = np.log1p(spec)
    maxv = np.max(spec)
    if maxv < EPS:
        spec[:] = 0
    else:
        spec = spec / maxv

    img = np.uint8(spec * 255)
    img = cv2.applyColorMap(img, cv2.COLORMAP_TURBO)
    img = cv2.resize(img, (img.shape[1], height), interpolation=cv2.INTER_NEAREST)

    cv2.imwrite(out_path, img)

# ======================= VIDEO FROM IMAGES =======================

def images_to_video(img_folder, out_path, fps=30):
    files = sorted([f for f in os.listdir(img_folder) if f.endswith(".png")])
    if not files:
        print("No images in", img_folder)
        return

    first = cv2.imread(os.path.join(img_folder, files[0]))
    h, w, _ = first.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    for f in files:
        img = cv2.imread(os.path.join(img_folder, f))
        out.write(img)

    out.release()

# ======================= MAIN =======================

def main():
    print("Processing video:", VIDEO_PATH)

    if not os.path.exists(VIDEO_PATH):
        print("ERROR: Video file not found.")
        return

    prepare_output_dirs()

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Failed to open video.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Total frames:", total_frames)

    start0 = FRAMES_TO_DROP

    max_possible_chunks = total_frames - start0 - FRAMES_PER_CHUNK
    if max_possible_chunks <= 0:
        print("ERROR: Video too short for chosen parameters.")
        cap.release()
        return

    chunks_to_run = min(NUM_CHUNKS, max_possible_chunks)
    print("Chunks to process:", chunks_to_run)

    for i in range(chunks_to_run):
        start_frame = start0 + i
        print(f"Chunk {i+1}/{chunks_to_run} (start frame {start_frame})")

        frames = get_chunk(cap, start_frame, FRAMES_PER_CHUNK)
        if frames is None:
            print("  Skipping chunk (not enough frames)")
            continue

        data_for_kmeans = compute_fft_stack(frames)
        centers = run_kmeans(data_for_kmeans)

        for method in SORT_METHODS:
            sorted_centers = sort_centers(centers, method)
            out_img = os.path.join(OUTPUT_DIR, method, f"frame_{i:05d}.png")
            render_spectrum(sorted_centers, out_img)

    cap.release()

    print("Building videos...")

    for method in SORT_METHODS:
        folder = os.path.join(OUTPUT_DIR, method)
        out_vid = f"{method}_sorted.mp4"
        images_to_video(folder, out_vid, fps=30)
        print("Wrote", out_vid)

    # Save summary
    with open(os.path.join(OUTPUT_DIR, "summary.txt"), "w") as f:
        f.write(f"Input video: {VIDEO_PATH}\n")
        f.write(f"Total frames: {total_frames}\n")
        f.write(f"Frames dropped: {FRAMES_TO_DROP}\n")
        f.write(f"Frames per chunk: {FRAMES_PER_CHUNK}\n")
        f.write(f"Chunks processed: {chunks_to_run}\n")
        f.write(f"Clusters: {N_CLUSTERS}\n")
        f.write("Sorting methods:\n")
        for m in SORT_METHODS:
            f.write(f"  - {m}\n")

    print("DONE.")

if __name__ == "__main__":
    main()
