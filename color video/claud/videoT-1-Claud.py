#!/usr/bin/env python3
''' ================================================================================
VIDEO SORTING METHOD ANALYSIS - FINAL VERSION
================================================================================

Authors: 
SP25: ???????,????????,Tim Price
FA25: Isaiah Wilson, Tim Price
SP26: Isaiah Wilson, ???????,????????

Purpose:
Test 6 different sorting methods for K-means cluster centers on overlapping
video chunks to determine which method provides the most consistent ordering.

Sorting Methods:
1. mean       - Average of all frequencies (original method)
2. max        - Maximum peak value (tallest)
3. median     - Median value
4. energy     - Sum of squares (total energy)
5. l2_norm    - Euclidean magnitude
6. centroid   - Weighted centroid frequency

================================================================================
'''

''' ------------------------- IMPORTS ------------------------------------- '''
import cv2
import os
import sys
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from datetime import datetime
import subprocess
from pathlib import Path

''' ------------------------- CONFIGURATION -------------------------------- '''

# ========== CHANGE THIS TO YOUR VIDEO FILE NAME ==========
VIDEO_FILENAME = "your_video.mp4"  # <<< CHANGE THIS
# ==========================================================

# Output settings
OUTPUT_DIR = "sort_types"
LOG_FILE = "processing_log.txt"

# Video processing parameters
FRAME_RATE = 30
FRAMES_TO_DROP = 60           # First 2 seconds (30fps * 2)
FRAMES_PER_CHUNK = 256        # Number of frames per chunk
TOTAL_CHUNKS = 30 * 8         # 240 chunks total
CHUNK_STEP_SIZE = 1           # Move forward 1 frame each time (255 frame overlap)

# K-means clustering settings
N_CLUSTERS = 8
NUM_HIGH_CLUSTERS_DROP = 0    # Drop N highest clusters
NUM_LOW_CLUSTERS_DROP = 2     # Drop N lowest clusters
KMEANS_BATCH_SIZE = 2000
KMEANS_N_INIT = 8
KMEANS_RANDOM_STATE = 10      # For reproducibility

# Image rendering settings
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 256

# Video encoding settings
VIDEO_CODEC = 'libx264'
VIDEO_CRF = 18                # Quality (lower = better, 18 is visually lossless)
VIDEO_PRESET = 'medium'       # Encoding speed vs compression

# Script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

''' ------------------------- SORTING FUNCTIONS ---------------------------- '''

def sort_by_mean(centers):
    """Sort by average value across all frequencies"""
    return sorted(centers, key=lambda c: np.mean(c), reverse=True)

def sort_by_max(centers):
    """Sort by maximum peak value (tallest)"""
    return sorted(centers, key=lambda c: np.max(c), reverse=True)

def sort_by_median(centers):
    """Sort by median value"""
    return sorted(centers, key=lambda c: np.median(c), reverse=True)

def sort_by_energy(centers):
    """Sort by total energy (sum of squares)"""
    return sorted(centers, key=lambda c: np.sum(c**2), reverse=True)

def sort_by_l2_norm(centers):
    """Sort by L2 norm (Euclidean magnitude)"""
    return sorted(centers, key=lambda c: np.linalg.norm(c), reverse=True)

def sort_by_centroid(centers):
    """Sort by weighted centroid frequency"""
    def centroid_freq(c):
        freqs = np.arange(len(c))
        return np.sum(freqs * c) / (np.sum(c) + 1e-9)
    return sorted(centers, key=centroid_freq, reverse=True)

# Dictionary of all sorting methods
SORT_METHODS = {
    "mean": ("Average of all frequencies", sort_by_mean),
    "max": ("Maximum peak value (tallest)", sort_by_max),
    "median": ("Median value", sort_by_median),
    "energy": ("Sum of squares (total energy)", sort_by_energy),
    "l2_norm": ("Euclidean magnitude", sort_by_l2_norm),
    "centroid": ("Weighted centroid frequency", sort_by_centroid)
}

''' ------------------------- HELPER FUNCTIONS ---------------------------- '''

class Logger:
    """Simple logger to write to both console and file"""
    def __init__(self, log_file):
        self.log_file = log_file
        self.start_time = datetime.now()
        
    def log(self, message, to_file=True):
        """Log message to console and optionally to file"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        formatted = f"[{timestamp}] {message}"
        print(formatted)
        if to_file:
            with open(self.log_file, 'a') as f:
                f.write(formatted + '\n')
    
    def log_elapsed(self, message):
        """Log message with elapsed time"""
        elapsed = datetime.now() - self.start_time
        self.log(f"{message} (Elapsed: {elapsed})")

def setup_output_folders():
    """Create folder structure for each sorting method"""
    output_path = os.path.join(SCRIPT_DIR, OUTPUT_DIR)
    
    # Create main directory
    os.makedirs(output_path, exist_ok=True)
    
    # Create subdirectories for each sorting method
    method_folders = {}
    for method_name, (description, _) in SORT_METHODS.items():
        method_dir = os.path.join(output_path, method_name)
        os.makedirs(method_dir, exist_ok=True)
        method_folders[method_name] = method_dir
        
        # Save method description
        desc_file = os.path.join(method_dir, "README.txt")
        with open(desc_file, 'w') as f:
            f.write(f"Sorting Method: {method_name}\n")
            f.write(f"Description: {description}\n")
            f.write(f"\nParameters:\n")
            f.write(f"  Clusters: {N_CLUSTERS}\n")
            f.write(f"  High clusters dropped: {NUM_HIGH_CLUSTERS_DROP}\n")
            f.write(f"  Low clusters dropped: {NUM_LOW_CLUSTERS_DROP}\n")
            f.write(f"  Effective clusters used: {N_CLUSTERS - NUM_HIGH_CLUSTERS_DROP - NUM_LOW_CLUSTERS_DROP}\n")
    
    return output_path, method_folders

def get_video_info(video_path):
    """Get video properties"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cap.release()
    
    return total_frames, fps, width, height

def extract_chunk(video_path, start_frame, num_frames):
    """
    Extract a chunk of frames from video.
    Returns numpy array of grayscale frames or None if insufficient frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
    
    cap.release()
    
    if len(frames) != num_frames:
        return None
    
    return np.array(frames)

def create_spectrum_image(spectrum, width=IMAGE_WIDTH, height=IMAGE_HEIGHT):
    """
    Create a colorful spectrum visualization.
    Uses log scaling and turbo colormap for better visualization.
    """
    # Apply log scaling for better visualization of small values
    spectrum_log = np.log1p(spectrum)  # log(1 + x) to handle zeros
    
    # Normalize to 0-1 range
    spectrum_norm = (spectrum_log - spectrum_log.min()) / (spectrum_log.max() - spectrum_log.min() + 1e-9)
    
    # Convert to 0-255 uint8
    spectrum_uint8 = (spectrum_norm * 255).astype(np.uint8)
    
    # Reshape to 2D array (rows, 1) for colormap application
    spectrum_2d = spectrum_uint8.reshape(-1, 1)
    
    # Apply TURBO colormap (better than RAINBOW for perception)
    colored = cv2.applyColorMap(spectrum_2d, cv2.COLORMAP_TURBO)
    
    # Reshape to (1, num_values, 3)
    colored = colored.reshape(1, -1, 3)
    
    # Resize to target dimensions
    # Use INTER_NEAREST to maintain sharp color boundaries
    image = cv2.resize(colored, (width, height), interpolation=cv2.INTER_NEAREST)
    
    return image

''' ------------------------- MAIN PROCESSING ----------------------------- '''

def process_chunk(frames_array, chunk_index, method_folders, logger):
    """
    Process a single chunk with all sorting methods.
    Returns True if successful, False otherwise.
    """
    if frames_array is None or frames_array.size == 0:
        return False
    
    # Step 1: Center the data (remove DC component)
    data_centered = frames_array - np.mean(frames_array, axis=0, keepdims=True)
    
    # Step 2: Apply Blackman window to reduce spectral leakage
    window = np.blackman(FRAMES_PER_CHUNK)[:, None, None]
    data_windowed = data_centered * window
    
    # Step 3: Compute FFT
    fft_data = np.abs(np.fft.rfft(data_windowed, axis=0))
    
    # Step 4: Reshape for K-means (pixels as samples, frequencies as features)
    # Shape: (num_pixels, num_frequencies)
    data_for_kmeans = fft_data.reshape(fft_data.shape[0], -1).T
    
    # Step 5: Run K-means clustering
    kmeans = MiniBatchKMeans(
        n_clusters=N_CLUSTERS,
        random_state=KMEANS_RANDOM_STATE,
        batch_size=KMEANS_BATCH_SIZE,
        n_init=KMEANS_N_INIT,
        max_iter=100
    )
    kmeans.fit(data_for_kmeans)
    centers = kmeans.cluster_centers_
    
    # Step 6: Process with each sorting method
    for method_name, (description, sort_func) in SORT_METHODS.items():
        # Sort the cluster centers
        sorted_centers = sort_func(centers)
        
        # Drop high and low clusters AFTER sorting
        start_idx = NUM_HIGH_CLUSTERS_DROP
        end_idx = N_CLUSTERS - NUM_LOW_CLUSTERS_DROP
        selected_centers = sorted_centers[start_idx:end_idx]
        
        # Concatenate all selected centers into single spectrum
        spectrum = np.hstack(selected_centers)
        
        # Create visualization image
        image = create_spectrum_image(spectrum)
        
        # Save image
        image_filename = f"chunk_{chunk_index:04d}.png"
        image_path = os.path.join(method_folders[method_name], image_filename)
        cv2.imwrite(image_path, image)
    
    return True

def process_video(video_path, method_folders, logger):
    """Process entire video with overlapping chunks"""
    
    # Get video information
    total_frames, fps, width, height = get_video_info(video_path)
    
    logger.log(f"Video Information:")
    logger.log(f"  Total frames: {total_frames}")
    logger.log(f"  FPS: {fps}")
    logger.log(f"  Resolution: {width}x{height}")
    logger.log(f"  Duration: {total_frames/fps:.2f} seconds")
    logger.log("")
    logger.log(f"Processing Parameters:")
    logger.log(f"  Frames to skip: {FRAMES_TO_DROP} ({FRAMES_TO_DROP/FRAME_RATE:.2f} sec)")
    logger.log(f"  Frames per chunk: {FRAMES_PER_CHUNK}")
    logger.log(f"  Chunk step size: {CHUNK_STEP_SIZE} frame")
    logger.log(f"  Frame overlap: {FRAMES_PER_CHUNK - CHUNK_STEP_SIZE} frames")
    logger.log(f"  Total chunks to process: {TOTAL_CHUNKS}")
    logger.log("")
    
    # Calculate how many chunks we can actually process
    max_possible_chunks = (total_frames - FRAMES_TO_DROP - FRAMES_PER_CHUNK) // CHUNK_STEP_SIZE + 1
    chunks_to_process = min(TOTAL_CHUNKS, max_possible_chunks)
    
    if chunks_to_process < TOTAL_CHUNKS:
        logger.log(f"WARNING: Can only process {chunks_to_process} chunks (video too short)")
        logger.log("")
    
    # Process each chunk
    successful_chunks = 0
    start_frame = FRAMES_TO_DROP
    
    for chunk_idx in range(chunks_to_process):
        current_start = start_frame + (chunk_idx * CHUNK_STEP_SIZE)
        
        # Progress indicator
        if chunk_idx % 10 == 0 or chunk_idx == 0:
            logger.log(f"Processing chunk {chunk_idx + 1}/{chunks_to_process} (frame {current_start})")
        
        # Extract frames
        frames = extract_chunk(video_path, current_start, FRAMES_PER_CHUNK)
        
        if frames is None:
            logger.log(f"  WARNING: Chunk {chunk_idx} incomplete, skipping")
            continue
        
        # Process chunk
        if process_chunk(frames, chunk_idx, method_folders, logger):
            successful_chunks += 1
    
    logger.log("")
    logger.log(f"Successfully processed {successful_chunks}/{chunks_to_process} chunks")
    return successful_chunks

def create_videos(output_dir, method_folders, logger):
    """Create videos from image sequences using ffmpeg"""
    logger.log("")
    logger.log("Creating videos from images...")
    
    # Check if ffmpeg is available
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.log("ERROR: ffmpeg not found. Videos cannot be created.")
        logger.log("Install ffmpeg to enable video creation.")
        logger.log("Images are saved in individual folders.")
        return
    
    for method_name in SORT_METHODS.keys():
        method_folder = method_folders[method_name]
        output_video = os.path.join(output_dir, f"{method_name}_video.mp4")
        
        # Check if images exist
        images = sorted([f for f in os.listdir(method_folder) if f.endswith('.png')])
        if not images:
            logger.log(f"  WARNING: No images found for {method_name}")
            continue
        
        # Build ffmpeg command
        cmd = [
            'ffmpeg',
            '-y',                                    # Overwrite output
            '-framerate', str(FRAME_RATE),          # Input framerate
            '-i', os.path.join(method_folder, 'chunk_%04d.png'),  # Input pattern
            '-c:v', VIDEO_CODEC,                    # Video codec
            '-preset', VIDEO_PRESET,                # Encoding preset
            '-crf', str(VIDEO_CRF),                 # Quality
            '-pix_fmt', 'yuv420p',                  # Pixel format (for compatibility)
            output_video
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.log(f"  ✓ Created {method_name}_video.mp4")
        except subprocess.CalledProcessError as e:
            logger.log(f"  ✗ Error creating {method_name} video: {e.stderr}")

def create_summary(output_dir, video_path, chunks_processed, logger):
    """Create a summary file with processing details"""
    summary_file = os.path.join(output_dir, "SUMMARY.txt")
    
    total_frames, fps, width, height = get_video_info(video_path)
    
    with open(summary_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("VIDEO SORTING METHOD ANALYSIS - PROCESSING SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        f.write("INPUT VIDEO:\n")
        f.write(f"  File: {os.path.basename(video_path)}\n")
        f.write(f"  Total frames: {total_frames}\n")
        f.write(f"  FPS: {fps}\n")
        f.write(f"  Resolution: {width}x{height}\n")
        f.write(f"  Duration: {total_frames/fps:.2f} seconds\n\n")
        
        f.write("PROCESSING PARAMETERS:\n")
        f.write(f"  Frames dropped at start: {FRAMES_TO_DROP} ({FRAMES_TO_DROP/FRAME_RATE:.2f} sec)\n")
        f.write(f"  Frames per chunk: {FRAMES_PER_CHUNK}\n")
        f.write(f"  Chunk step size: {CHUNK_STEP_SIZE} frame\n")
        f.write(f"  Frame overlap: {FRAMES_PER_CHUNK - CHUNK_STEP_SIZE} frames\n")
        f.write(f"  Chunks processed: {chunks_processed}\n\n")
        
        f.write("CLUSTERING PARAMETERS:\n")
        f.write(f"  K-means clusters: {N_CLUSTERS}\n")
        f.write(f"  High clusters dropped: {NUM_HIGH_CLUSTERS_DROP}\n")
        f.write(f"  Low clusters dropped: {NUM_LOW_CLUSTERS_DROP}\n")
        f.write(f"  Effective clusters: {N_CLUSTERS - NUM_HIGH_CLUSTERS_DROP - NUM_LOW_CLUSTERS_DROP}\n")
        f.write(f"  Batch size: {KMEANS_BATCH_SIZE}\n")
        f.write(f"  N_init: {KMEANS_N_INIT}\n\n")
        
        f.write("SORTING METHODS:\n")
        f.write("-"*70 + "\n")
        for method_name, (description, _) in SORT_METHODS.items():
            f.write(f"  {method_name:12s} - {description}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("OUTPUT:\n")
        f.write(f"  Images: {OUTPUT_DIR}/<method_name>/chunk_XXXX.png\n")
        f.write(f"  Videos: {OUTPUT_DIR}/<method_name>_video.mp4\n")
        f.write("="*70 + "\n")
    
    logger.log(f"\nSummary saved to: {summary_file}")

''' ------------------------- MAIN EXECUTION ------------------------------ '''

def main():
    """Main execution function"""
    
    # Initialize logger
    log_path = os.path.join(SCRIPT_DIR, LOG_FILE)
    logger = Logger(log_path)
    
    logger.log("="*70)
    logger.log("VIDEO SORTING METHOD ANALYSIS")
    logger.log("="*70)
    logger.log("")
    
    # Find video file
    video_path = os.path.join(SCRIPT_DIR, VIDEO_FILENAME)
    if not os.path.exists(video_path):
        logger.log(f"ERROR: Video file not found: {VIDEO_FILENAME}")
        logger.log(f"Please ensure the video file exists in: {SCRIPT_DIR}")
        logger.log("")
        logger.log("To fix this:")
        logger.log(f"  1. Place your video file in {SCRIPT_DIR}")
        logger.log(f"  2. Edit VIDEO_FILENAME at the top of this script")
        return 1
    
    logger.log(f"Video file: {VIDEO_FILENAME}")
    logger.log("")
    
    # Setup output folders
    try:
        output_dir, method_folders = setup_output_folders()
        logger.log(f"Output directory: {output_dir}")
        logger.log(f"Created {len(SORT_METHODS)} method folders")
        logger.log("")
    except Exception as e:
        logger.log(f"ERROR setting up folders: {e}")
        return 1
    
    # Process video
    try:
        chunks_processed = process_video(video_path, method_folders, logger)
    except Exception as e:
        logger.log(f"ERROR processing video: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Create videos
    try:
        create_videos(output_dir, method_folders, logger)
    except Exception as e:
        logger.log(f"ERROR creating videos: {e}")
        # Don't return error - images are still useful
    
    # Create summary
    try:
        create_summary(output_dir, video_path, chunks_processed, logger)
    except Exception as e:
        logger.log(f"WARNING: Could not create summary: {e}")
    
    # Final message
    logger.log("")
    logger.log("="*70)
    logger.log_elapsed("PROCESSING COMPLETE!")
    logger.log("="*70)
    logger.log(f"Results: {output_dir}/")
    logger.log(f"Log file: {log_path}")
    logger.log("")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())