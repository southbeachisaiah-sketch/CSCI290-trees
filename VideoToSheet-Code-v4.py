''' -------------------- commets and into stuff---------------------------- '''
'''
## original authers ???????,????????,???????
## re-written and commeted by Isaiah R Wilson for improved clarity and speed.
## ChatGPT was used to help understand original code

## major changes
## 1. changed from FFT for each pixal to all at once. 
##      its about a 99% speed reduction in FFT part
## 2. passed capture thought getdata so that the video file is not opened 2 times
##      on a simple test run went from 11.6 seconds to 11 seconds
## 3. potentually change the kMeans algorithm to MiniBatchKMeans from K-Means
##      https://scikit-learn.org/stable/auto_examples/cluster/plot_mini_batch_kmeans.html
##      in a simple test its a 18.38 seconds -> 2.397 seconds reduction 87% reduction
## 4. change the frames_array to float 32 instead of 64
##      this changes speed of hole program from 10.9 seconds to 10.3 seconds
##      currently disabled b/c does not seem to improve speed by much
##
## 5. set up global vars at top of file
## 6. Removed var FR that was passed throught all the Fn and made it a global var (framerate)
## 7. renamed varius vars to make them more explicilt labled :p
## 8. changed how cluster ordering and picking works


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


''' ------------------------- global vars  -------------------------------- '''

## this makes the program work on some computers
os.environ['LOKY_MAX_CPU_COUNT'] = '20' # limit to CPU cores

# this will be the name of the output.csv file
testNameRunNumber = "test1MultiRun"
outPutFileName = "treeDataTest29newOldFFT"


## stadard vars
frameRate = 30;
framesPerChunk = 32 #256  # Number of frames per chunk
framesToDropAtBeginingOfVid = 60
n_clusters = 8  # Number of clusters (side note increasing this increases y values on graph)
numHighClustersDroped = 0  # Number of high clusters to drop
numLowClustersDroped = 2  # Number of low clusters to drop

kMeansAlgorithm = "MiniBatch" 
#kMeansAlgorithm = "KMeans"

miniBatchBatchSize = 2000
kMeansNumOfRuns = 8         # this is the number of time it runs to try and 

numberOfFramechunksToDo = 3
chunk_index = 0



# Directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# gets all folders that start with "run" (Written with GPT 5)
run_folders = [
    name for name in os.listdir(script_dir)
    if os.path.isdir(os.path.join(script_dir, name)) and name.lower().startswith("run")
]


# this is the path to where the treevideos is
#folderPathTooTreeVids = ".\\treeVidTest"
#folderPathTooTreeVids = r"E:\OneDrive - FVCC\ForestCharacterization\LongVideo\Speaker"
#folderPathTooTreeVids = r"K:\class\24_FA\CSCI_290_01\shared\NewCamera\SDCard"
#folderPathTooTreeVids = r"K:\class\24_FA\CSCI_290_01\shared\NewCamera\SDCard\GreenAshe"
#folderPathTooTreeVids = r"K:\class\24_FA\CSCI_290_01\shared\NewCamera\SDCard\Maple"
#folderPathTooTreeVids = r"K:\class\24_FA\CSCI_290_01\shared\NewCamera\SDCard\MapleRed"
#folderPathTooTreeVids = r"K:\class\24_FA\CSCI_290_01\shared\NewCamera\SDCard\MtAshe"
#folderPathTooTreeVids = r"K:\class\24_FA\CSCI_290_01\shared\NewCamera\SDCard\Willow"
#folderPathTooTreeVids = r"K:\class\24_FA\CSCI_290_01\shared\._Final_Organization\LongVideos\LongDry\MtAshe"
#folderPathTooTreeVids = r"E:\OneDrive - FVCC\ForestCharacterization\LongVideo\Trees"
#folderPathTooTreeVids = r"E:\TreeData\LongVideos\"
#folderPathTooTreeVids = r"K:\class\25_SP\CSCI_290_01\shared\EvergreenSorted\DougFir"

# gets all files in folder
#files = os.listdir(folderPathTooTreeVids)
#filesNAMES = []



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
def process(frames_array, output_file, output_file_All):
    
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

    '''test
    # == NEW SECTION: generate and save adjacency (distance) matrix, one row at a time
    print(f"== Generating adjacency matrix for chunk {chunk_index} ...")
    data_for_dist = dataPrimedForKMeans

    # == Optional: sample only part of the pixels to keep memory low (uncomment if needed)
    # if data_for_dist.shape[0] > 1000:
    #     idx = np.random.choice(data_for_dist.shape[0], 1000, replace=False)
    #     data_for_dist = data_for_dist[idx, :]

    # == Open CSV file for streaming write
    out_path = f"chunk_{chunk_index}_adjacency.csv"
    with open(out_path, 'w') as f:
        for i in range(data_for_dist.shape[0]):
            # == Compute distances from pixel i to all others
            dists = np.linalg.norm(data_for_dist - data_for_dist[i], axis=1)
            # == Save as comma-separated line
            np.savetxt(f, [dists], delimiter=',', fmt='%.4f')
            # == Optional progress display
            if i % 100 == 0:
                print(f"   Row {i}/{data_for_dist.shape[0]} done...")
                logTime("code line: 193")
    print(f"== Saved adjacency matrix: {out_path}")
    # == END NEW SECTION
    test'''


        # this just choses the kmeans algorithm, kmeans is likely better but minibatch is faster
    if kMeansAlgorithm == "KMeans": kmeans = KMeans(n_clusters=n_clusters, random_state=10, n_init=kMeansNumOfRuns)
    else:                           kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=10, batch_size= miniBatchBatchSize)

        # this runs the kmeans algorithm, and sets the centers to the center var
        # a note is that there are n_clusters (number of) centers
        # and the center represents the avarage frequencies that split the pixels into best clusters
    kmeans.fit(dataPrimedForKMeans)
    centers = kmeans.cluster_centers_

        # This sorts the cluster, so that way they are ordered from lowest to highest average (ascending order)
        # It may make a difference sorting from highest value in the cluster center
    sortedCenters = sorted(centers, key=lambda c: c.mean(), reverse=True)
    
    
        # this sets up the first and last cluster used (removes lowest clusters if set in global vars)
    startCluster = numHighClustersDroped
    endCluster = n_clusters - numLowClustersDroped
    
        # this changes the data and concatinated the different centers togather makeing it one dimentional (for the CSV file)
        # np.hstack(shape[j:n]) is wierd it concatinates shapes oddly, it inclueds and starts
        # at j but goes up to and does not inclued n 
    spectrums = np.hstack(sortedCenters[startCluster:endCluster])

    logTime("code line: 193")
    
    
        # this saves the cluster centers data, the 'a' is for append mode
    file = open(output_file,'a')
    np.savetxt(file,[spectrums], delimiter=',', fmt='%.2f')
    file.close()
 
    file = open(output_file_All,'a')
    np.savetxt(file,[spectrums], delimiter=',', fmt='%.2f')
    file.close()
    


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
    
    
    ## this runs "prosses" on each frame chunk
    for start_frame in range(start_processing_frame, total_frames, framesPerChunk):
        frames_array = getdata(file, start_frame)
        ##frames_array = getdata(capture, start_frame, framesPerChunk)
        if frames_array is None:
            continue
        if frames_array.shape[0]!=framesPerChunk:
            print(frames_array.shape[0])
            continue
    
    
        # payload
        process(frames_array, output_file, output_file_All)
        
        
        count +=1
        print("count: " +  str(count))

# end of fileProssesing





# Main execution


# this is the txtDoc that records notes (Written with GPT 5)
# Create the log file name
notesFileName = f"{testNameRunNumber}-Notes.md"

# Initialize the file with a header
with open(notesFileName, 'w') as f:
    f.write("Beginning of file\n")
    f.write("=================\n\n")

# Simple function to append a line to the notes file
def logNote(text):
    with open(notesFileName, 'a') as f:
        f.write(text + "       "+ datetime.now().strftime("%H:%M:%S.%f")[:-3]+"\n")
# end of log notes




''' OLD MAIN
for file_name in files:
    file_path = os.path.join(folderPathTooTreeVids, file_name)
    ##logNote(file_name)
    
    if os.path.isfile(file_path):
        filesNAMES.append(file_path)

for file in filesNAMES:
    print(file)
    if file[-3:] != "MOV" and file[-3:] != "MP4":
        print("wrong File Type: " + file)
        continue
    #print(file)
    output_file = outPutFileName + ".csv"
    #output_file = file[:-4]+".csv"  # File to save spectra
    #output_file = folderPathTooTreeVids[51:]+".csv"  # File to save spectra
    #print(output_file)
    fileProssesing(file, output_file)
print("Completed............")
'''

## new MAIN (Written with heavy aid from GPT 5)
# --------------------------------------------------------------------
# Process each run folder
# --------------------------------------------------------------------
print("Found run folders:", run_folders)


chunkIncramentor = 0;
for folder in run_folders:

    
    logNote(folder + "   Started  " + str(chunkIncramentor)+ " ")
    folder_path = os.path.join(script_dir, folder)

    # Output CSV name: testNameRunNumber + folderName + ".csv"
    output_file = f"{testNameRunNumber}_{folder}.csv"
    output_file_All = f"{testNameRunNumber}_All.csv"

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




