# IMPORT LIBRARIES
import torch # PyTorch is used for tensor manipulation and deep learning


# SCALING FUNCTIONS
def scale_data(data, min_value, max_value):
    """
    Perform Min-Max normalization
    """
    data_min = torch.min(data)    # Compute the minimum value of the dataset
    data_max = torch.max(data)    # Compute the maximum value

    # Apply the linear transformation
    # x_scaled = (max-min)*(x-data_min)/(data_max-data_min) + min
    scaled_data = (
        (max_value - min_value)
        * (data - data_min)
        / (data_max - data_min)
        + min_value
    )
    return data_max, data_min, scaled_data


def scale_data_for_test(data, data_min, data_max, min_value, max_value):
    """
    Apply Min-Max normalization to a test dataset
    """
    scaled_data = (
        (max_value - min_value)
        * (data - data_min)
        / (data_max - data_min)
        + min_value
    )
    return scaled_data


def unscale_data(scaled_data, original_max, original_min):
    """
    Restore Min-Max normalized data to the original scale.
    """
    unscaled_data = (
        scaled_data
        * (original_max - original_min)
        + original_min
    )
    return unscaled_data

# GAUSSIAN (STANDARD) NORMALIZATION
def gaussian_scale(data, eps=1e-5):
    """
    Standardize the data using x_scaled = (x - mean) / std
    """
    mean = torch.mean(data) # Compute dataset mean
    std_dev = torch.std(data) # Compute standard deviation
    # Standardization
    scaled_data = (
        data - mean
    ) / (std_dev + eps)
    return mean, std_dev, scaled_data


def gaussian_scale_for_test(data, mean, std_dev, eps=1e-5):
    """
    Normalize the test dataset using the training mean and standard deviation
    """
    scaled_data = (
        data - mean
    ) / (std_dev + eps)
    return scaled_data

def inverse_gaussian_scale(
    scaled_data,
    original_mean,
    original_std_dev,
    eps=1e-5
):
    """
    Restore standardized data back to the original scale
    """
    unscaled_data = (
        scaled_data
        * (original_std_dev + eps)
        + original_mean
    )
    return unscaled_data

# DATASET DESCRIPTION
"""
Each MATLAB file contains
X
time
V
label
"""

# IMPORT MATLAB LOADER
from scipy.io import loadmat # Used to read MATLAB .mat files

# IMPORT LIBRARIES
import torch # PyTorch is used for tensor manipulation and deep learning


# SCALING FUNCTIONS
def scale_data(data, min_value, max_value):
    """
    Perform Min-Max normalization
    """
    data_min = torch.min(data)    # Compute the minimum value of the dataset
    data_max = torch.max(data)    # Compute the maximum value

    # Apply the linear transformation
    # x_scaled = (max-min)*(x-data_min)/(data_max-data_min) + min
    scaled_data = (
        (max_value - min_value)
        * (data - data_min)
        / (data_max - data_min)
        + min_value
    )
    return data_max, data_min, scaled_data


def scale_data_for_test(data, data_min, data_max, min_value, max_value):
    """
    Apply Min-Max normalization to a test dataset
    """
    scaled_data = (
        (max_value - min_value)
        * (data - data_min)
        / (data_max - data_min)
        + min_value
    )
    return scaled_data


def unscale_data(scaled_data, original_max, original_min):
    """
    Restore Min-Max normalized data to the original scale.
    """
    unscaled_data = (
        scaled_data
        * (original_max - original_min)
        + original_min
    )
    return unscaled_data

# GAUSSIAN (STANDARD) NORMALIZATION
def gaussian_scale(data, eps=1e-5):
    """
    Standardize the data using x_scaled = (x - mean) / std
    """
    mean = torch.mean(data) # Compute dataset mean
    std_dev = torch.std(data) # Compute standard deviation
    # Standardization
    scaled_data = (
        data - mean
    ) / (std_dev + eps)
    return mean, std_dev, scaled_data


def gaussian_scale_for_test(data, mean, std_dev, eps=1e-5):
    """
    Normalize the test dataset using the training mean and standard deviation
    """
    scaled_data = (
        data - mean
    ) / (std_dev + eps)
    return scaled_data

def inverse_gaussian_scale(
    scaled_data,
    original_mean,
    original_std_dev,
    eps=1e-5
):
    """
    Restore standardized data back to the original scale
    """
    unscaled_data = (
        scaled_data
        * (original_std_dev + eps)
        + original_mean
    )
    return unscaled_data

# DATASET DESCRIPTION
"""
Each MATLAB file contains
X
time
V
label
"""

# IMPORT MATLAB LOADER
from scipy.io import loadmat # Used to read MATLAB .mat files

# LOAD TRAINING DATASET
def load_single_train(dataname):
    """
    Load a single training dataset (.mat).
    """

    data = loadmat(dataname)

    X = data["X"]
    time = data["time"].squeeze()
    V = data["V"]
    labels = data["label"].squeeze()
    mean = float(data["mean"].squeeze())
    std = float(data["std"].squeeze())

    return X, time, V, labels, mean, std

# LOAD TEST DATASET
def load_single_test(dataname):
    """
    Load a single testing dataset (.mat).
    """

    data = loadmat(dataname)

    X = data["X"]
    time = data["time"].squeeze()
    V = data["V"]
    labels = data["label"].squeeze()
    mean = float(data["mean"].squeeze())
    std = float(data["std"].squeeze())

    return X, time, V, labels, mean, std
