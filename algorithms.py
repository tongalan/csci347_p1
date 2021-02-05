# Part 2 of the Project
# A package of all the function outlined in the project description part 2.

import numpy as np
import math
np.set_printoptions(suppress=True)


# Compute a multidimensional mean
def compute_mean(numpy_array):
    num_row, num_col = numpy_array.shape
    avg = []
    for value in range(num_col):
        test = numpy_array[:, value]
        average = sum(test)/num_row
        avg.append(average)
    return np.asarray(avg, dtype=np.float32)


# Compute the covariance between two numpy arrays
def compute_covariance(arr_one, arr_two):
    n = len(arr_one) if len(arr_two) == len(arr_one) else None
    arr_one_mean = np.mean(arr_one)
    arr_two_mean = np.mean(arr_two)
    total = 0
    for i in range(0, n):
        total += ((arr_one[i] - arr_one_mean) * (arr_two[i] - arr_two_mean))
    return round((total / (n-1)), 2)


# Computes the Pearson's correlation coefficient between two numpy arrays.
def correlation(vec1, vec2):
    n = len(vec1) if len(vec1) == len(vec2) else None
    variance = compute_covariance(vec1, vec2) * (n-1)
    vec1_mean = np.mean(vec1)
    vec2_mean = np.mean(vec2)
    total_one = 0
    total_two = 0
    for i in range(0,n):
        total_one += (vec1[i] - vec1_mean)**2
        total_two += (vec2[i] - vec2_mean)**2
    return variance / math.sqrt(total_one * total_two)


# Returns the column of a 2D list as a list.
def get_column(matrix, i):
    return [row[i] for row in matrix]


# Computes the variance of a single vector.
def get_variance(vector):
    variance = 0
    n = len(vector)
    vector_mean = np.mean(vector)
    for i in range(n):
        variance += (vector[i] - vector_mean)**2
    return round((variance / (n-1)), 2)


# Implementation of range normalization.
def range_normalize(vector):
    # min_value = min(vector)
    # max_value = max(vector)
    normalize = []
    row, col = vector.shape
    for column in range(col):
        normal = []
        attribute = vector[:, column]
        min_value = min(attribute)
        max_value = max(attribute)
        for value in attribute:
            normalized_value = (value - min_value) / (max_value - min_value)
            normal.append(round(normalized_value, 1))
        normalize.append(normal)
    range_normalized = []
    for i in range(row):
        range_normalized.append(get_column(normalize, i))
    return np.asarray(range_normalized)


# Implementation of standard (zscore) normalization.
def standard_normalization(vector):
    normalize = []
    row, col = vector.shape
    for column in range(col):
        normal = []
        attribute = vector[:, column]
        avg_attribute = attribute.mean()
        std = attribute.std(ddof=1)
        for value in attribute:
            normalized_value = (value - avg_attribute) / std
            normal.append(round(normalized_value, 1))
        normalize.append(normal)
    std_normalization = []
    for i in range(row):
        std_normalization.append(get_column(normalize, i))
    return np.asarray(std_normalization)


# Computes the covariance matrix based on a numpy array
def covariance_matrix(vector):
    row, col = vector.shape
    attributes = []
    covar_matrix = [[0 for i in range(col)] for j in range(col)]
    for i in range(col):
        attributes.append(get_column(vector, i))
    for x in range(col):
        for y in range(col):
            if x == y:
                covar_matrix[x][y] = get_variance(attributes[x])
            else:
                covar_matrix[x][y] = compute_covariance(attributes[x], attributes[y])
    return np.asarray(covar_matrix)

# Implementation of label encoding with multidimensional categorical attributes.
def label_encoding(categorical_vector):
    label_encoded = []
    for observation in categorical_vector:
        labels = {}
        encode = []
        unique_values = np.unique(observation)
        for index, value in enumerate(unique_values):
            labels[value] = index
        for i in range(len(observation)):
            if observation[i] in labels.keys():
                encode.append(labels.get(observation[i]))
        label_encoded.append(encode)
    return np.asarray(label_encoded)
