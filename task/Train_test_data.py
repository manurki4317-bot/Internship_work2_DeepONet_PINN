# IMPORT LIBRARIES
import numpy as np
from scipy.io import loadmat# Functions for loading and saving MATLAB (.mat) file
from scipy.io import savemat
from sklearn.model_selection import train_test_split # Function used to randomly split a dataset into training and testing subsets

# FILE PATHS AND SETTINGS
INPUT_FILE = "dataset_raw.mat"
TRAIN_FILE = "train_bursts.mat"
TEST_FILE = "test_bursts.mat"

TEST_SIZE = 0.20
RANDOM_STATE = 42


# LOAD DATASET
print("Loading dataset...")
data = loadmat(INPUT_FILE)# The result is a dictionary where each variable stored in the .mat file becomes one dictionary entry

X = data["X"] # Load the input parameters (number_of_samples, number_of_parameters)
V = data["V"] # Load the voltage traces (number_of_samples, number_of_time_points)

time = data["time"].squeeze() # Load the common time vectors
labels = data["label"].squeeze() # Load burst labels

# Display dataset dimensions
print("X:", X.shape)
print("V:", V.shape)
print("Labels:", labels.shape)
print("Time:", time.shape)


# TRAIN / TEST SPLIT

# X_train, X_test = input parameters
# V_train, V_test = voltage traces
# y_train, y_test = class labels
X_train, X_test, \
V_train, V_test, \
y_train, y_test = train_test_split(
    X,
    V,
    labels,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=labels
)

# NORMALIZATION
mean = np.mean(V_train) # Compute the mean of the training data
std = np.std(V_train) # Compute the standard deviation of the training data

# After normalization:
# mean = 0
# standard deviation = 1
V_train = (V_train - mean) / std

# Normalize the testing signals using the same training statistics, ensuring that both datasets share exactly the same scaling
V_test = (V_test - mean) / std

# SAVE TRAINING DATASET
# Save the training dataset in MATLAB format

# X = input parameters
# V = normalized voltage traces
# time = common time vector
# label = class labels
# mean = normalization mean
# std = normalization standard deviation

savemat(
    TRAIN_FILE,
    {
        "X": X_train,
        "V": V_train,
        "time": time,
        "label": y_train,
        "mean": np.array([mean], dtype=np.float32),
        "std": np.array([std], dtype=np.float32),
    }
)

# SAVE TESTING DATASET
savemat(
    TEST_FILE,
    {
        "X": X_test,
        "V": V_test,
        "time": time,
        "label": y_test,
        "mean": np.array([mean], dtype=np.float32),
        "std": np.array([std], dtype=np.float32),
    }
)

# SUMMARY
print("\nDataset successfully split.\n")
print("Training samples :", X_train.shape[0]) # Number of training samples
print("Testing samples  :", X_test.shape[0])

print("\nTraining labels") # Display the class distribution in the testing dataset
print(np.bincount(y_train))

print("\nTesting labels") # Display the class distribution in the testing dataset
print(np.bincount(y_test))

print("\nMean :", mean) # Display the normalization statistics
print("Std  :", std)