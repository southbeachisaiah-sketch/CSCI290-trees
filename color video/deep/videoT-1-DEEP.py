''' -------------------- comments and intro stuff ----------------------------
Authors 
SP25: ???????,????????,Tim Price
FA25: Isaiah Wilson, Tim Price
SP26: Isaiah Wilson, ???????,????????

Final improved version with fixes for all identified issues
Specify video file path at the top
'''

''' ------------------------- imports ------------------------------------- '''
import cv2
import os
import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans
from datetime import datetime
import shutil
import json

''' ------------------------- CONFIGURATION -------------------------------- '''

# ========== VIDEO FILE CONFIGURATION ==========
# SPECIFY YOUR VIDEO FILE PATH HERE:
VIDEO_FILE_PATH = "YOUR_VIDEO_FILE.mp4"  # <<< CHANGE THIS TO YOUR VIDEO FILE

# ========== PROCESSING PARAMETERS ==========
FRAME_RATE = 30
FRAMES_TO_DROP = 60                       # First 2 seconds at 30fps
FRAMES_PER_CHUNK = 256
NUM_CHUNKS_TO_PROCESS = 240              # 30 * 8 chunks
CHUNK_OVERLAP = 255                       # Chunks share 255 frames
CHUNK_STEP = 1                            # Move 1 frame each chunk

N_CLUSTERS = 8
NUM_HIGH_CLUSTERS_DROPPED = 0
NUM_LOW_CLUSTERS_DROPPED = 2

# K-means configuration
USE_KMEANS = False                        # True for KMeans, False for MiniBatch
MINIBATCH_SIZE = 2000
KMEANS_N_INIT = 8

# Output configuration
OUTPUT_BASE = "sort_types"
TEST_NAME = "video_sort_analysis"

# ========== SORTING METHODS ==========
SORTING_METHODS = {
    'mean_desc': 'Mean descending (original)',
    'mean_asc': 'Mean ascending',
    'max_desc': 'Maximum value descending', 
    'max_asc': 'Maximum value ascending',
    'energy_desc': 'Energy (sum of squares) descending',
    'energy_asc': 'Energy (sum of squares) ascending'
}

''' --------------------------- FUNCTIONS -------------------------------- '''

def log_time(message):
    """Log message with timestamp"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

def validate_video_file():
    """Check if video file exists and is readable"""
    if not os.path.exists(VIDEO_FILE_PATH):
        raise FileNotFoundError(f"Video file not found: {VIDEO_FILE_PATH}")
    
    cap = cv2.VideoCapture(VIDEO_FILE_PATH)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {VIDEO_FILE_PATH}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cap.release()
    
    log_time(f"Video info: {total_frames} frames, {fps:.1f} FPS, {width}x{height}")
    
    if total_frames < FRAMES_TO_DROP + FRAMES_PER_CHUNK + NUM_CHUNKS_TO_PROCESS:
        raise ValueError(f"Video too short. Needs at least {FRAMES_TO_DROP + FRAMES_PER_CHUNK + NUM_CHUNKS_TO_PROCESS} frames")
    
    return total_frames, fps, width, height

def setup_output_folders():
    """Create organized output folder structure"""
    if os.path.exists(OUTPUT_BASE):
        shutil.rmtree(OUTPUT_BASE)
    
    os.makedirs(OUTPUT_BASE)
    
    # Create method folders
    method_folders = {}
    for method in SORTING_METHODS:
        folder = os.path.join(OUTPUT_BASE, method)
        os.makedirs(folder)
        method_folders[method] = folder
    
    # Create logs folder
    log_dir = os.path.join(OUTPUT_BASE, "logs")
    os.makedirs(log_dir)
    
    return method_folders

def extract_chunk(video_path, start_frame, chunk_size):
    """Extract a chunk of frames from video"""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frames = []
    for _ in range(chunk_size):
        ret, frame = cap.read()
        if not ret:
            break
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
    
    cap.release()
    
    if len(frames) != chunk_size:
        return None
    
    return np.array(frames)

def process_fft(frames):
    """Apply FFT to frames with windowing"""
    # Center data
    data_centered = frames - np.mean(frames, axis=0, keepdims=True)
    
    # Apply Blackman window
    window = np.blackman(FRAMES_PER_CHUNK)[:, None, None]
    data_windowed = data_centered * window
    
    # FFT
    fft_data = np.abs(np.fft.rfft(data_windowed, axis=0))
    
    # Flatten for k-means: shape = (pixels, frequencies)
    n_frequencies = fft_data.shape[0]
    height, width = frames.shape[1], frames.shape[2]
    data_for_kmeans = fft_data.reshape(n_frequencies, -1).T
    
    return data_for_kmeans, n_frequencies, height, width

def perform_clustering(data):
    """Perform k-means clustering"""
    if USE_KMEANS:
        kmeans = KMeans(
            n_clusters=N_CLUSTERS,
            random_state=42,
            n_init=KMEANS_N_INIT
        )
    else:
        kmeans = MiniBatchKMeans(
            n_clusters=N_CLUSTERS,
            random_state=42,
            batch_size=MINIBATCH_SIZE,
            n_init=KMEANS_N_INIT
        )
    
    kmeans.fit(data)
    return kmeans.cluster_centers_, kmeans.labels_

def sort_centers_by_method(centers, method):
    """Sort cluster centers using specified method"""
    if method == 'mean_desc':
        return sorted(centers, key=lambda c: c.mean(), reverse=True)
    elif method == 'mean_asc':
        return sorted(centers, key=lambda c: c.mean(), reverse=False)
    elif method == 'max_desc':
        return sorted(centers, key=lambda c: c.max(), reverse=True)
    elif method == 'max_asc':
        return sorted(centers, key=lambda c: c.max(), reverse=False)
    elif method == 'energy_desc':
        return sorted(centers, key=lambda c: np.sum(c**2), reverse=True)
    elif method == 'energy_asc':
        return sorted(centers, key=lambda c: np.sum(c**2), reverse=False)
    else:
        raise ValueError(f"Unknown sorting method: {method}")

def create_spectrum_image(spectrum, chunk_idx, method):
    """Create a colorful spectrum visualization"""
    # Normalize spectrum
    if spectrum.max() > spectrum.min():
        normalized = (spectrum - spectrum.min()) / (spectrum.max() - spectrum.min())
    else:
        normalized = np.zeros_like(spectrum)
    
    # Create color gradient (rainbow spectrum)
    height = 200  # Height of spectrum image
    width = len(spectrum) * 10  # Width based on spectrum length
    
    # Create hue based on normalized values
    hues = (normalized * 179).astype(np.uint8)  # 0-179 for OpenCV HSV
    
    # Create full saturation and value
    saturation = np.full_like(hues, 255)
    value = np.full_like(hues, 255)
    
    # Create HSV image
    hsv = np.zeros((1, len(spectrum), 3), dtype=np.uint8)
    hsv[0, :, 0] = hues
    hsv[0, :, 1] = saturation
    hsv[0, :, 2] = value
    
    # Convert to BGR
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Resize to desired dimensions
    spectrum_img = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_NEAREST)
    
    # Add label
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(spectrum_img, f"{method} - Chunk {chunk_idx:04d}", 
                (10, 30), font, 0.7, (255, 255, 255), 2)
    
    return spectrum_img

def process_single_chunk(video_path, start_frame, chunk_idx, method_folders):
    """Process a single chunk with all sorting methods"""
    # Extract frames
    frames = extract_chunk(video_path, start_frame, FRAMES_PER_CHUNK)
    if frames is None:
        log_time(f"Chunk {chunk_idx}: Failed to extract frames")
        return False
    
    # Process FFT
    data_for_kmeans, n_freq, height, width = process_fft(frames)
    
    # Perform clustering
    centers, labels = perform_clustering(data_for_kmeans)
    
    # Process each sorting method
    for method_name in SORTING_METHODS:
        # Sort centers
        sorted_centers = sort_centers_by_method(centers, method_name)
        
        # Apply cluster dropping
        start_idx = NUM_HIGH_CLUSTERS_DROPPED
        end_idx = N_CLUSTERS - NUM_LOW_CLUSTERS_DROPPED
        relevant_centers = sorted_centers[start_idx:end_idx]
        
        # Combine centers into spectrum
        spectrum = np.hstack(relevant_centers)
        
        # Create and save spectrum image
        spectrum_img = create_spectrum_image(spectrum, chunk_idx, method_name)
        
        # Save image
        output_path = os.path.join(method_folders[method_name], f"chunk_{chunk_idx:04d}.png")
        cv2.imwrite(output_path, spectrum_img)
    
    return True

def create_videos_from_images(method_folders, fps):
    """Create videos from the generated images"""
    log_time("Creating videos from images...")
    
    for method_name, folder in method_folders.items():
        # Get sorted list of image files
        image_files = sorted([f for f in os.listdir(folder) 
                            if f.endswith('.png') and f.startswith('chunk_')])
        
        if not image_files:
            log_time(f"No images found for {method_name}")
            continue
        
        # Read first image to get dimensions
        first_image = cv2.imread(os.path.join(folder, image_files[0]))
        if first_image is None:
            continue
        
        height, width = first_image.shape[:2]
        
        # Create video writer
        video_path = os.path.join(OUTPUT_BASE, f"{method_name}_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        
        # Write all images to video
        for img_file in image_files:
            img_path = os.path.join(folder, img_file)
            frame = cv2.imread(img_path)
            if frame is not None:
                video.write(frame)
        
        video.release()
        log_time(f"Created video: {video_path}")

def save_processing_summary(video_path, total_frames, fps, width, height, method_folders):
    """Save processing summary and configuration"""
    summary = {
        'input_video': video_path,
        'video_info': {
            'total_frames': total_frames,
            'fps': fps,
            'resolution': f"{width}x{height}",
            'duration_seconds': total_frames / fps if fps > 0 else 0
        },
        'processing_parameters': {
            'frames_dropped_at_start': FRAMES_TO_DROP,
            'frames_per_chunk': FRAMES_PER_CHUNK,
            'total_chunks_processed': NUM_CHUNKS_TO_PROCESS,
            'chunk_overlap_frames': CHUNK_OVERLAP,
            'chunk_step_frames': CHUNK_STEP,
            'n_clusters': N_CLUSTERS,
            'high_clusters_dropped': NUM_HIGH_CLUSTERS_DROPPED,
            'low_clusters_dropped': NUM_LOW_CLUSTERS_DROPPED,
            'kmeans_algorithm': 'KMeans' if USE_KMEANS else 'MiniBatchKMeans'
        },
        'sorting_methods': SORTING_METHODS,
        'output_folders': list(method_folders.keys()),
        'processing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Save as JSON
    json_path = os.path.join(OUTPUT_BASE, "processing_summary.json")
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Also save as readable text
    txt_path = os.path.join(OUTPUT_BASE, "processing_summary.txt")
    with open(txt_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("VIDEO PROCESSING SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Input Video: {os.path.basename(video_path)}\n")
        f.write(f"Video Frames: {total_frames}\n")
        f.write(f"Video FPS: {fps:.1f}\n")
        f.write(f"Video Resolution: {width}x{height}\n")
        f.write(f"Video Duration: {total_frames/fps:.1f} seconds\n\n")
        
        f.write("Processing Parameters:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Frames dropped at start: {FRAMES_TO_DROP} ({FRAMES_TO_DROP/fps:.1f} seconds)\n")
        f.write(f"Frames per chunk: {FRAMES_PER_CHUNK}\n")
        f.write(f"Total chunks to process: {NUM_CHUNKS_TO_PROCESS}\n")
        f.write(f"Chunk overlap: {CHUNK_OVERLAP} frames\n")
        f.write(f"Chunk step size: {CHUNK_STEP} frame(s)\n")
        f.write(f"Number of clusters: {N_CLUSTERS}\n")
        f.write(f"High clusters dropped: {NUM_HIGH_CLUSTERS_DROPPED}\n")
        f.write(f"Low clusters dropped: {NUM_LOW_CLUSTERS_DROPPED}\n")
        f.write(f"Clustering algorithm: {'KMeans' if USE_KMEANS else 'MiniBatchKMeans'}\n\n")
        
        f.write("Sorting Methods:\n")
        f.write("-" * 40 + "\n")
        for method, description in SORTING_METHODS.items():
            f.write(f"{method}: {description}\n")
        
        f.write(f"\nOutput saved in: {OUTPUT_BASE}/\n")
        f.write(f"Images per method: {NUM_CHUNKS_TO_PROCESS}\n")
        f.write(f"Total images generated: {NUM_CHUNKS_TO_PROCESS * len(SORTING_METHODS)}\n")
    
    log_time(f"Summary saved to: {json_path} and {txt_path}")

def main():
    """Main processing function"""
    print("=" * 60)
    print("VIDEO SPECTRUM SORTING ANALYSIS")
    print("=" * 60)
    
    try:
        # Validate video file
        log_time(f"Validating video file: {VIDEO_FILE_PATH}")
        total_frames, fps, width, height = validate_video_file()
        
        # Setup output folders
        log_time("Setting up output folders...")
        method_folders = setup_output_folders()
        
        # Calculate processing parameters
        start_frame = FRAMES_TO_DROP
        max_possible_chunks = min(NUM_CHUNKS_TO_PROCESS, 
                                 total_frames - FRAMES_TO_DROP - FRAMES_PER_CHUNK)
        
        if max_possible_chunks < NUM_CHUNKS_TO_PROCESS:
            log_time(f"Warning: Video only allows {max_possible_chunks} chunks, not {NUM_CHUNKS_TO_PROCESS}")
        
        # Process chunks
        log_time(f"Processing {max_possible_chunks} chunks...")
        successful_chunks = 0
        
        for chunk_idx in range(max_possible_chunks):
            current_start = start_frame + (chunk_idx * CHUNK_STEP)
            
            if chunk_idx % 10 == 0:
                log_time(f"Processing chunk {chunk_idx+1}/{max_possible_chunks} (frame {current_start})")
            
            # Process chunk with all sorting methods
            success = process_single_chunk(
                VIDEO_FILE_PATH, 
                current_start, 
                chunk_idx, 
                method_folders
            )
            
            if success:
                successful_chunks += 1
        
        log_time(f"Processed {successful_chunks} chunks successfully")
        
        # Create videos
        if successful_chunks > 0:
            create_videos_from_images(method_folders, fps/2)  # Half speed for smoother playback
        
        # Save summary
        save_processing_summary(
            VIDEO_FILE_PATH, total_frames, fps, width, height, method_folders
        )
        
        print("\n" + "=" * 60)
        print("PROCESSING COMPLETE!")
        print("=" * 60)
        print(f"Output location: {OUTPUT_BASE}/")
        print(f"Chunks processed: {successful_chunks}")
        print(f"Sorting methods: {len(SORTING_METHODS)}")
        print(f"Total images generated: {successful_chunks * len(SORTING_METHODS)}")
        print(f"Videos created: {len(SORTING_METHODS)}")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    # Check if video file path is set
    if VIDEO_FILE_PATH == "YOUR_VIDEO_FILE.mp4":
        print("ERROR: Please set VIDEO_FILE_PATH at the top of the script")
        print("Change line 18 to point to your video file")
        exit(1)
    
    exit(main())