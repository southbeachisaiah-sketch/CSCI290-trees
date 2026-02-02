import numpy as np
from sklearn import neighbors  # imports the entire "neighbors" module, requires neighbors....
from sklearn.metrics import accuracy_score,precision_score,recall_score,classification_report
from sklearn.model_selection import train_test_split
import joblib # for saving the model

fileName = "t5-300-0_All.csv"
X = np.genfromtxt(fileName,delimiter=',')

# for 1470 lines (Lp MEANS LINE PONDEROSA)
lDF = 153
lL = 101
lLP = 125
lP   = 624
lS   = 467

# for 3-1
#lS = 465
# for 8-2 Run Re-all
# for 12-1
#lP   = 621


#X1 = X[::,3:40:]
#X2 = X[::,110:125:]
#---better
'''
print ("this is shape")
X1 = X[::,3:40:]
print(X1.shape)
X2 = X[::,128+3:128+40:]
print(X1.shape)


X = np.hstack((X1,X2))
print ("end of this is shape")'''
#X = X[::,3:30:] # great
#X = 0 to 128
#X = X[::,2:59:]
#X = X[::,10:124:]
#X = X[::,0:128:]


'''
# Example: choose how many blocks you want
I = 5   # <--- set this to any number you need

blocks = []

for i in range(I):
    start = i * 128 + 0
    stop = i * 128 + 128
    Xi = X[:, start:stop]
    blocks.append(Xi)
    print(f"Block {i} shape:", Xi.shape)

# stack all blocks horizontally
X = np.hstack(blocks)'''














###---worse
##X1 = X[::,2:10:]
##print(X1.shape)
##X2 = X[::,131:140:]
##print(X1.shape)
##X = np.hstack((X1,X2))
##print(X.shape)
##print(X[0:1,::])

np.set_printoptions(threshold=np.inf)

# Supervised
##GA = 0*np.zeros(6,dtype=np.int8) #
##M   = 1*np.ones(6,dtype=np.int8)
##MA = 2*(np.ones(5,dtype=np.int8))
##P   = 3*(np.ones(9,dtype=np.int8))
##S   = 4*(np.ones(21,dtype=np.int8))
##W = 5*(np.ones(8,dtype=np.int8))
##y   = np.hstack((GA,M,MA,P,S,W))
##print(y)

DF = 0*np.zeros(lDF,dtype=np.int8) #
L  = 1*np.ones(lL,dtype=np.int8)
LP = 2*(np.ones(lLP,dtype=np.int8))
P   = 3*(np.ones(lP,dtype=np.int8))
S   = 4*(np.ones(lS,dtype=np.int8))
y   = np.hstack((DF,L,LP,P,S))
print(y)

# KNN
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=None, train_size=.8,random_state=True, shuffle=True)
#knn = neighbors.KNeighborsClassifier(n_neighbors=5) 
knn = neighbors.KNeighborsClassifier(algorithm='auto', leaf_size=40,
metric='minkowski', n_neighbors=10, p=2, weights='uniform') # 'jaccard' ,'taxicab', , 'hamming'  ...
print(knn)
knn.fit(X_train, y_train)

#X_test.reshape(1, -1) #?
y_pred = knn.predict(X_test)
print("actual:       ",y_test)
print("predicted: ",y_pred)

#  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
print('Accuracy Score: y_test, y_pred: {:.2f}'.format(accuracy_score(y_test, y_pred)))
print('Precision Score: y_test, y_pred: {:.2f}'.format(precision_score(y_test, y_pred,
                                                    average = "weighted",zero_division = 0)))
print('Recall Score: y_test, y_pred: {:.2f}'.format(recall_score(y_test, y_pred,average = "weighted")))

#Classification Report
print("kNN Classification Report:---------------------------------- ")
print(classification_report(y_test, y_pred, zero_division=1))

# Save the model to a file
joblib.dump(knn, 'Tree_knn_model.joblib')
print(fileName)

from sklearn.inspection import permutation_importance
import numpy as np



#'''
# ================================================================
#   PERMUTATION IMPORTANCE FOR MULTIPLE 128-COLUMN BLOCKS
#   + Runtime Estimator
# ================================================================
from sklearn.inspection import permutation_importance
import time

print("\n------------------------------------------------------------")
print("Estimating runtime for permutation importance...")
print("------------------------------------------------------------")

n_repeats = 10        # number of times to shuffle each column
n_test = len(X_test)
n_features = X_test.shape[1]

# ---- BENCHMARK SMALL SAMPLE FOR TIME ESTIMATE ----
test_feature = 0
X_copy = X_test.copy()

start = time.time()
np.random.shuffle(X_copy[:, test_feature])   # shuffle 1 column once
knn.predict(X_copy)                          # predict once
elapsed = time.time() - start

# Scale up time estimate
estimated_seconds = elapsed * n_features * n_repeats

print(f"\nEstimated total runtime: ~{estimated_seconds:.1f} seconds")
print(f"                     or: {estimated_seconds/60:.2f} minutes")
print(f"(Using {n_features} features × {n_repeats} repeats)\n")


# ================================================================
#   COMPUTE PER-COLUMN PERMUTATION IMPORTANCE
# ================================================================
print("------------------------------------------------------------")
print("Computing feature importances (per column)...")
print("------------------------------------------------------------")

result = permutation_importance(
    knn,
    X_test,
    y_test,
    n_repeats=n_repeats,
    random_state=42
)

importances = result.importances_mean  # mean importance per column


# ================================================================
#   PER-COLUMN SUMMARY
# ================================================================
indices = np.argsort(importances)[::-1]

print("\nTop 10 MOST influential columns:")
for i in indices[:10]:
    print(f"Column {i}: importance = {importances[i]:.4f}")

print("\nTop 10 MOST harmful columns:")
harm = np.argsort(importances)
for i in harm[:10]:
    print(f"Column {i}: importance = {importances[i]:.4f}")


# ================================================================
#   GROUP INTO 128-COLUMN BLOCKS
# ================================================================
block_size = 128
total_cols = X_train.shape[1]
num_blocks = total_cols // block_size

print(f"\nDetected {num_blocks} blocks (each {block_size} columns wide)\n")

block_importance = []

for b in range(num_blocks):
    start = b * block_size
    end   = start + block_size
    mean_imp = np.mean(importances[start:end])
    block_importance.append(mean_imp)
    print(f"Block {b}: mean importance = {mean_imp:.4f}")


# ================================================================
#   SORT BLOCKS BY IMPORTANCE
# ================================================================
sorted_blocks = np.argsort(block_importance)[::-1]

print("\nBlocks ranked MOST to LEAST important:")
for b in sorted_blocks:
    print(f"Block {b}: mean importance = {block_importance[b]:.4f}")


# ================================================================
#   IDEAL BLOCK SELECTION (positive importance)
# ================================================================
good_blocks = [b for b in range(num_blocks) if block_importance[b] > 0]

print("\nBlocks recommended to KEEP (positive importance):")
print(good_blocks)

# Convert blocks → column indices
good_features = []
for b in good_blocks:
    start = b * block_size
    end   = start + block_size
    good_features.extend(range(start, end))

print(f"\nKeeping {len(good_features)} out of {total_cols} total columns.")


# ================================================================
#   TEST ACCURACY USING ONLY POSITIVE-IMPORTANCE BLOCKS
# ================================================================
X_train_reduced = X_train[:, good_features]
X_test_reduced  = X_test[:,  good_features]

knn2 = neighbors.KNeighborsClassifier(
    algorithm='auto',
    leaf_size=40,
    metric='minkowski',
    n_neighbors=10,
    p=2,
    weights='uniform'
)

knn2.fit(X_train_reduced, y_train)
y_pred2 = knn2.predict(X_test_reduced)

print("\n------------------------------------------------------------")
print("Accuracy using ONLY good blocks (positive importance):")
print("Reduced Accuracy: {:.2f}".format(accuracy_score(y_test, y_pred2)))
print("------------------------------------------------------------\n")




#'''












'''
# =============================================================
#   MULTI-BLOCK PERMUTATION IMPORTANCE (BLOCK SIZE 128)
# =============================================================
print("\n=== MULTI-BLOCK FEATURE IMPORTANCE ===")

BLOCK_SIZE = 128
n_features = X_train.shape[1]
num_blocks = n_features // BLOCK_SIZE

print(f"Detected {num_blocks} blocks of size {BLOCK_SIZE} each.")

block_importances = []

base_accuracy = accuracy_score(y_test, y_pred)
print(f"Base accuracy: {base_accuracy:.4f}")

for b in range(num_blocks):
    print(f"\nTesting block {b}...")

    # Columns for this block
    start = b * BLOCK_SIZE
    stop = start + BLOCK_SIZE

    X_test_mod = X_test.copy()

    # Shuffle whole block together (keeps internal structure)
    for col in range(start, stop):
        shuffled = X_test_mod[:, col].copy()
        np.random.shuffle(shuffled)
        X_test_mod[:, col] = shuffled

    # Predict with the modified block
    y_pred_block = knn.predict(X_test_mod)
    acc = accuracy_score(y_test, y_pred_block)

    importance = base_accuracy - acc
    block_importances.append(importance)

    print(f"Block {b} importance = {importance:.4f}")

block_importances = np.array(block_importances)


# -------------------------------------------------------------
#   TOP BLOCKS (HELPFUL)
# -------------------------------------------------------------
print("\nTop 10 MOST influential (helpful) blocks:")
top_blocks = np.argsort(block_importances)[::-1]
for b in top_blocks[:10]:
    print(f"Block {b}: importance = {block_importances[b]:.4f}")


# -------------------------------------------------------------
#   WORST BLOCKS (HARMFUL)
# -------------------------------------------------------------
print("\nTop 10 MOST harmful blocks:")
worst_blocks = np.argsort(block_importances)  # ascending
for b in worst_blocks[:10]:
    print(f"Block {b}: importance = {block_importances[b]:.4f}")


# -------------------------------------------------------------
#   SELECT IDEAL BLOCK SET (positive contribution)
# -------------------------------------------------------------
good_blocks = [b for b in range(num_blocks) if block_importances[b] > 0]

print("\nRecommended blocks to KEEP (positive importance):")
print(good_blocks)

# Flatten into column indices
good_cols = []
for b in good_blocks:
    start = b * BLOCK_SIZE
    stop = start + BLOCK_SIZE
    good_cols.extend(range(start, stop))


# -------------------------------------------------------------
#   TEST MODEL USING ONLY IDEAL BLOCKS
# -------------------------------------------------------------
X_train_reduced = X_train[:, good_cols]
X_test_reduced = X_test[:, good_cols]

knn_blocks = neighbors.KNeighborsClassifier(
    algorithm='auto',
    leaf_size=40,
    metric='minkowski',
    n_neighbors=10,
    p=2,
    weights='uniform'
)

knn_blocks.fit(X_train_reduced, y_train)
y_pred_reduced = knn_blocks.predict(X_test_reduced)
final_acc = accuracy_score(y_test, y_pred_reduced)

print("\nAccuracy using ONLY ideal (positive importance) blocks:")
print(f"Reduced accuracy: {final_acc:.4f}")
#'''