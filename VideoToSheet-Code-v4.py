''' -------------------- commets and into stuff----------------------------

authers 
SP25: ???????,????????,Tim Price
FA25: Isaiah Wilson, Tim Price
SP26: Isaiah Wilson, ???????,????????




## ideas that may help
1. wen you sort the order of centers the tallest one may be bettor then the average for this line of code
sortedCenters = sorted(centers, key=lambda c: c.mean(), reverse=True)

2. try differnet kmans algortithms

3. try different windowing algorithms

4. when you find the centers, also find varrence from that center this may reveal somthing interesting
    this would be very useful in determining the right number of clusters
    poke around untill you find a number that gets the differnet clusters but is not deviding oveous clusters

5. instead of running it in grayscale try running it in red blue green that may change somthing

6. find optimm number of clusters
    https://youtu.be/iNlZ3IU5Ffw?si=2e2fFNIXEVsMeXrO&t=609
    
    
    
move the commet out on longer runs to global vars

make adgasentcy matrix for distance between each node, with the distrance between them
as the one or 0


'''
''' ------------------------- imports ------------------------------------- '''
import cv2
import os
import numpy as np
from scipy.fft import rfft, rfftfreq
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from datetime import datetime
from scipy.spatial.distance import cdist # for node testing
import subprocess
import sys
import csv


''' ------------------------- global vars  -------------------------------- '''

## this makes the program work on some computers
os.environ['LOKY_MAX_CPU_COUNT'] = '20' # limit to CPU cores

# this will be the name of the output.csv file
testNameRunNumber = "test29NewOldFFT"
treeSpecies = "Oak"  # You'll need to set this per run or pass as parameter
pythonFileName = os.path.basename(__file__)

## stadard vars
frameRate = 30;
framesPerChunk = 256 # Number of frames per chunk
chunkStepSize = 256  # Number of frames to advance between chunks (set this to framesPerChunk for non-overlapping, half for 2x chunks)
framesToDropAtBeginingOfVid = 60

n_clusters = 8  # Number of clusters (side note increasing this increases y values on graph)

kMeansAlgorithm = "MiniBatch" 
#kMeansAlgorithm = "KMeans"

miniBatchBatchSize = 2000
kMeansNumOfRuns = 8         # this is the number of time it runs to try and 

numberOfFramechunksToDo = 3
chunk_index = 0

# Metadata defaults (you can modify these per run)
distanceFromTree = "N/A"
windSpeed = "N/A"
moistureContent = "N/A"
treeStatus = "N/A"  # Dead / Alive


# Directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Create output directory
output_dir = os.path.join(script_dir, f"DataFrom-{testNameRunNumber}")
os.makedirs(output_dir, exist_ok=True)

# gets all folders that start with "run" adjasent to this file
run_folders = [
    name for name in os.listdir(script_dir)
    if os.path.isdir(os.path.join(script_dir, name)) and name.lower().startswith("run")
]


''' --------------------------- functions  -------------------------------- '''

    # this function makes a clean way to consle log time
    # the place name is so you can do "line of code number"
def logTime(placeNameString):
    print ("         "+placeNameString+": " + datetime.now().strftime("%H:%M:%S.%f")[:-3])



##def getdata(capture, start_frame):
def getdata(file, start_frame):
    capture = cv2.VideoCapture(file)

    
    frames_array = []

    if not capture.isOpened():
        return None, None

    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for _ in range(framesPerChunk):
        ret, frame = capture.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) # <<added
        frames_array.append(frame)

    frames_array = np.array(frames_array)

    if frames_array.size == 0:  # not sure if this is working
        return None, None

##    print("Long frames shape_after: ",frames_array.shape) # optional
##    print(frames_array[:20,:2,:2])
          
    ##frames_subset = frames_array[:numberOfFramechunksToDo]
    ##print(type(frames_subset))
    ##print(type(frames_array))
    return frames_array

## end of getdata




    # this is the main function that does the prossesing of thr clips
    # frames_array is an array where each item in the array is a series of 256 frames
def process(frames_array, output_file, start_frame, end_frame, video_name):
    
        # This exits "process" if there was no input or if the array had 0 frames
    if frames_array is None or frames_array.size ==0: return 

        # This shifts the wave down by the aveage amplatude so that its centered
        # Then it This applies a windowing affect to the arrat
        # windowType, dataCentered, and dataPrimedForFFT are all "numpy.ndarray"
    dataCentered = frames_array - np.mean(frames_array, axis=0, keepdims=True)
    windowType = np.blackman(framesPerChunk)[:, None, None]
    dataPrimedForFFT = dataCentered * windowType
    
        # This compleates a FFT on all pixels of the chunk at once
    dataFFTed = np.abs(np.fft.rfft(dataPrimedForFFT, axis=0))

        # Flatten FFT results for clustering
        # this changes the shape from 3D (freq,X,Y) to 2D (freq,pixalNum) 
        # here pixelNum is a single number but it runs thought all pixels so its max value is X*Y
    dataPrimedForKMeans = dataFFTed.reshape(dataFFTed.shape[0], -1).T # alternate...



        # this just choses the kmeans algorithm, kmeans is likely better but minibatch is faster
    if kMeansAlgorithm == "KMeans": kmeans = KMeans(n_clusters=n_clusters, random_state=10, n_init=kMeansNumOfRuns)
    else:                           kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=10, batch_size= miniBatchBatchSize)

        # this runs the kmeans algorithm, and sets the centers to the center var
        # a note is that there are n_clusters (number of) centers
        # and the center represents the avarage frequencies that split the pixels into best clusters
    kmeans.fit(dataPrimedForKMeans)
    centers = kmeans.cluster_centers_

        # Removed sorting function - now using centers in original order
    
        # Get number of frequencies per cluster
    num_frequencies = len(centers[0])
    
        # Prepare metadata row (20 columns)
    metadata = [
        testNameRunNumber,           # Test number
        treeSpecies,                 # Tree species
        pythonFileName,              # Python file name
        video_name,                  # Video name
        f"{start_frame}:{end_frame}", # Frame range
        str(num_frequencies),         # Number of frequencies
        "cluster",                    # Cluster / pixel / other
        str(n_clusters),              # Number of clusters
        "N/A",                        # X:Y pixel location
        "N/A:N/A:N/A",                # Start color : Average color : End color
        distanceFromTree,             # Distance from tree
        windSpeed,                    # Wind speed
        moistureContent,              # Moisture content
        treeStatus,                   # Dead / Alive
        "", "", "", "", "", ""        # 6 blank columns
    ]
    
        # Prepare frequency data with "end frequency X" markers
    frequency_data = []
    for i, center in enumerate(centers):
        # Add the frequency values
        frequency_data.extend(center.tolist())
        # Add the end marker
        frequency_data.append(f"end frequency {i+1}")
    
        # Combine metadata and frequency data
    row_data = metadata + frequency_data

    logTime("code line: 193")
    
    
        # this saves the data, the 'a' is for append mode
    with open(output_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row_data)


## comment out for long runs-------------------
    #labels = kmeans.labels_
    #labels.flatten
    #image = labels.reshape(1080,1920).astype(np.uint8)
    #image = image*125
    #cv2.imshow('Frame', image) 
    #cv2.imwrite("cluster_treelong.png", image)
## comment out for long runs-------------------






## end of prosses







def fileProssesing(file, output_file, output_file_All):
    
    ## creates the object that has the video info
    capture = cv2.VideoCapture(file)
    
    ## gets total frames in vid, start frame and count
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    start_processing_frame = framesToDropAtBeginingOfVid
    count = 0
    
    video_name = os.path.basename(file)
    
    ## this runs "prosses" on each frame chunk using the step size
    for start_frame in range(start_processing_frame, total_frames - framesPerChunk + 1, chunkStepSize):
        frames_array = getdata(file, start_frame)
        ##frames_array = getdata(capture, start_frame, framesPerChunk)
        if frames_array is None:
            continue
        if frames_array.shape[0] != framesPerChunk:
            print(f"Warning: Got {frames_array.shape[0]} frames, expected {framesPerChunk}")
            continue
    
        end_frame = start_frame + frames_array.shape[0] - 1
    
        # payload
        process(frames_array, output_file, start_frame, end_frame, video_name)
        process(frames_array, output_file_All, start_frame, end_frame, video_name)
        
        
        count +=1
        print("count: " +  str(count))

# end of fileProssesing





# Main execution


# this is the txtDoc that records notes
# Create the log file name
notesFileName = os.path.join(output_dir, f"{testNameRunNumber}-Notes.md")

# Initialize the file with a header
with open(notesFileName, 'w') as f:
    f.write(f"Processing Notes for {testNameRunNumber}\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Python File: {pythonFileName}\n")
    f.write(f"Tree Species: {treeSpecies}\n")
    f.write(f"Frames per chunk: {framesPerChunk}\n")
    f.write(f"Chunk step size: {chunkStepSize}\n")
    f.write(f"Number of clusters: {n_clusters}\n")
    f.write(f"KMeans Algorithm: {kMeansAlgorithm}\n\n")
    f.write("Processing Log:\n")
    f.write("-" * 30 + "\n")

# Simple function to append a line to the notes file
def logNote(text):
    with open(notesFileName, 'a') as f:
        f.write(text + "       "+ datetime.now().strftime("%H:%M:%S.%f")[:-3]+"\n")
# end of log notes



## new MAIN
# --------------------------------------------------------------------
# Process each run folder
# --------------------------------------------------------------------
print("Found run folders:", run_folders)
print(f"Output directory: {output_dir}")
print(f"Chunk configuration: {framesPerChunk} frames per chunk, stepping by {chunkStepSize} frames")
print(f"This will create approximately {framesPerChunk/chunkStepSize:.1f}x more chunks than non-overlapping")


# Initialize the combined All file with a header
combined_all_file = os.path.join(output_dir, f"Data-all.csv")
# Write header if file doesn't exist
if not os.path.exists(combined_all_file):
    with open(combined_all_file, 'w', newline='') as f:
        writer = csv.writer(f)
        # Optional: Write a header row with column descriptions
        header = ["TestNum", "TreeSpecies", "PythonFile", "VideoName", "FrameRange", 
                  "NumFrequencies", "DataType", "NumClusters", "PixelLocation", 
                  "ColorRange", "Distance", "WindSpeed", "Moisture", "Status",
                  "Blank1", "Blank2", "Blank3", "Blank4", "Blank5", "Blank6"]
        writer.writerow(header)

chunkIncramentor = 0;
for folder in run_folders:

    
    logNote(folder + "   Started  " + str(chunkIncramentor)+ " ")
    folder_path = os.path.join(script_dir, folder)

    # Output CSV names in the new directory
    output_file = os.path.join(output_dir, f"Data-{folder}.csv")
    output_file_All = combined_all_file

    # Write header for individual run file if it doesn't exist
    if not os.path.exists(output_file):
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ["TestNum", "TreeSpecies", "PythonFile", "VideoName", "FrameRange", 
                      "NumFrequencies", "DataType", "NumClusters", "PixelLocation", 
                      "ColorRange", "Distance", "WindSpeed", "Moisture", "Status",
                      "Blank1", "Blank2", "Blank3", "Blank4", "Blank5", "Blank6"]
            writer.writerow(header)

    # Reset file list
    filesNAMES = []

    # find MOV / MP4 files
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        if os.path.isfile(file_path):
            if file_path.lower().endswith(".mov") or file_path.lower().endswith(".mp4"):
                filesNAMES.append(file_path)
            else:
                print("Skipping wrong file type:", file_path)

    print(f"\n=== Processing folder: {folder} ===")
    print("Files:", filesNAMES)

    # Run your processing on each file
    for file in filesNAMES:
        chunkIncramentor = chunkIncramentor + 1
        logNote("  " + str(chunkIncramentor) + "  file: "+ file +"     folder: " + folder)
        print(f"Processing file: {file}")
        fileProssesing(file, output_file, output_file_All)
        
    logNote(folder + "   ENDED  " + str(chunkIncramentor)+ " ")

print("\nALL RUN FOLDERS COMPLETED.")
print(f"Output files saved in: {output_dir}")
print(f"Total chunks processed: {chunkIncramentor}")
