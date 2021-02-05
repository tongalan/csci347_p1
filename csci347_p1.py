import csv
import algorithms as algo
import numpy as np
import matplotlib.pyplot as plt

IRIS_DATA_FILE_NAME = "forestfires.csv"


# ALAN TONG
# Entry point for Project 3 - run main()
# Opens the data file.
def open_iris_data():
    with open(IRIS_DATA_FILE_NAME) as csvfile:
        data = list(csv.reader(csvfile))
    return data[:]

# Initial processing of data to clear missing values and check which values are numerical vs strings.
def preprocessing(dataset):
    new_data = []
    for observation in dataset:
        empty_observation = []
        for index, attribute in enumerate(observation):
            if isinstance(attribute, str):
                try:
                    value = float(attribute)
                except:
                    value = attribute
            empty_observation.append(value)
        new_data.append(empty_observation)

    return np.asarray(new_data)

# Convert data string format into floats for computation.
def convert_to_float(arr):
    convertArr = []
    for s in arr.ravel():
        try:
            value = float(s)
        except ValueError:
            value = s
        convertArr.append(value)
    return np.array(convertArr, dtype=object).reshape(arr.shape)


# Implementation of label encoding, with inputs of which index in the data are categorical data.
def run_label_encoding(data, categorical_index):
    n = len(data)
    categorical_data = []
    for i in categorical_index:
        categorical_data.append(data[:, i])
    categorical_data = np.asarray(categorical_data)
    encoded = algo.label_encoding(categorical_data)
    for i in range(n):
        for j in categorical_index:
            # TODO: ensure all data set will work, not just forest fire
            data[i][j] = encoded[j-2][i]
    return data

def multivariate_mean(dataset):
    return algo.compute_mean(dataset)

def covar_matrix(dataset):
    return algo.covariance_matrix(dataset)

# Implementation of ranged normalization.
def range_norm_attr(dataset):
    range_normalized = algo.range_normalize(dataset)
    row, col = range_normalized.shape
    n = len(range_normalized)
    ranged_norm_attributes = []
    for attributes in range(col):
        ranged_norm_attributes.append(range_normalized[:, attributes])
    max_sample_covariance = algo.compute_covariance(ranged_norm_attributes[0], ranged_norm_attributes[1])
    attribute_x = -1
    attribute_y = -1
    for x in range(len(ranged_norm_attributes)):
        for y in range(x + 1, len(ranged_norm_attributes)):
            current_variance = algo.compute_covariance(ranged_norm_attributes[x], ranged_norm_attributes[y])
            if current_variance >= max_sample_covariance:
                max_sample_covariance = current_variance
                attribute_x, attribute_y = x, y
    return attribute_x, attribute_y, max_sample_covariance, ranged_norm_attributes[attribute_x], ranged_norm_attributes[attribute_y]


# Implementation of zscore normalization and finds the highest correlation attributes based on normalized data.
# TODO: Reduce the duplicate code
def std_norm_attr_gcorr(dataset):
    std_normalization = algo.standard_normalization(dataset)
    row, col = std_normalization.shape
    n = len(std_normalization)
    std_normalization_attribute = []
    for attribute in range(col):
        std_normalization_attribute.append(std_normalization[:, attribute])
    max_corr = algo.correlation(std_normalization[0], std_normalization[1])
    gcorr_attr_x = -1
    gcorr_attr_y = -1
    for x in range(len(std_normalization_attribute)):
        for y in range(x + 1, len(std_normalization_attribute)):
            curr_corr = algo.correlation(std_normalization_attribute[x], std_normalization_attribute[y])
            if curr_corr >= max_corr:
                max_corr = curr_corr
                gcorr_attr_x, gcorr_attr_y = x, y
    return gcorr_attr_x, gcorr_attr_y, max_corr, std_normalization[gcorr_attr_x], std_normalization[gcorr_attr_y]


# Implementation of zscore normalization and finds the lowest correlation attributes based on normalized data.
# TODO: Reduce duplicate code from function above
def std_norm_attr_lcorr(dataset):
    std_normalization = algo.standard_normalization(dataset)
    row, col = std_normalization.shape
    n = len(std_normalization)
    std_normalization_attribute = []
    for attribute in range(col):
        std_normalization_attribute.append(std_normalization[:, attribute])
    min_corr = algo.correlation(std_normalization[0], std_normalization[1])
    lcorr_attr_x = -1
    lcorr_attr_y = -1
    for x in range(len(std_normalization_attribute)):
        for y in range(x + 1, len(std_normalization_attribute)):
            curr_corr = algo.correlation(std_normalization_attribute[x], std_normalization_attribute[y])
            if curr_corr <= min_corr:
                min_corr = curr_corr
                lcorr_attr_x, lcorr_attr_y = x, y
    return lcorr_attr_x, lcorr_attr_y, min_corr, std_normalization_attribute[lcorr_attr_x], std_normalization_attribute[lcorr_attr_y]


# Calculates all correlation coefficient between attributes that is greater than 0.5.
def greater_than_half(dataset):
    pairs = []
    attribute_matrix = []
    n = len(dataset)
    row, col = dataset.shape
    for attribute in range(col):
        attribute_matrix.append(dataset[:, attribute])
    for x in range(col):
        for y in range(x+1, col):
            if algo.correlation(attribute_matrix[x], attribute_matrix[y]) >= 0.5:
                pairs.append((x,y))
    return len(pairs)

# Calculates all negative sample covariance attributes
def neg_sample_covariance(dataset):
    neg_sample_covar = []
    attribute_matrix = []
    n = len(dataset)
    row, col = dataset.shape
    for attribute in range(col):
        attribute_matrix.append(dataset[:, attribute])
    for x in range(col):
        for y in range(x + 1, col):
            if algo.compute_covariance(attribute_matrix[x], attribute_matrix[y]) < 0.0:
                neg_sample_covar.append((x, y))
    return len(neg_sample_covar)


# Calculate the total variance of a matrix
def total_variance(dataset):
    attribute_matrix = []
    total_variance = 0
    row, col = dataset.shape
    for attribute in range(col):
        attribute_matrix.append(dataset[:, attribute])
    for x in range(col):
        total_variance += algo.get_variance(attribute_matrix[x])
    return total_variance


# Top 5 Variance
def total_variance_five(dataset):
    attribute_matrix = []
    total_variance = 0
    ordered_variance = {}
    top_five = []
    row, col = dataset.shape
    for attribute in range(col):
        attribute_matrix.append(dataset[:, attribute])

    for index, attr in enumerate(attribute_matrix):
        variance = algo.get_variance(attr)
        ordered_variance[variance] = index

    for i in sorted(ordered_variance):
        top_five.append((i, ordered_variance[i]))
    top_five = top_five[-5:]
    for attr in top_five:
        total_variance += attr[0]
    return total_variance

# Scatter Plots
# def plotting_scatter_plots(dataset):
#     attribute_matrix = []
#     row, col = dataset.shape
#     for attribute in range(col):
#         attribute_matrix.append(dataset[:, attribute])
#     x = attribute_matrix[7]
#     y = attribute_matrix[8]
#
#     plt.autoscale(enable=True)
#     plt.scatter(x, y)
#
#     plt.title('Relationship between Fire Spread and Temperature')
#     plt.xlabel('Initial Spread Index (Fire Weather Index)')
#     plt.ylabel('Temperature (in Celsius)')
#     plt.grid(True)
#     plt.savefig('1.5Pairs_FireSpread_Temp.png')
#     plt.show()

# def plotting_scatter_plots(dataset):
#     attribute_matrix = []
#     row, col = dataset.shape
#     for attribute in range(col):
#         attribute_matrix.append(dataset[:, attribute])
#     x = attribute_matrix[8]
#     y = attribute_matrix[10]
#
#     plt.autoscale(enable=True)
#     plt.scatter(x, y)
#
#     plt.title('Relationship between Wind Speed and Temperature')
#     plt.xlabel('Temperature (in Celsius)')
#     plt.ylabel('Wind (km/hr)')
#     plt.grid(True)
#     plt.savefig('2.5Pairs_Wind_Temp.png')
#     plt.show()
#
# def plotting_scatter_plots(dataset):
#     attribute_matrix = []
#     row, col = dataset.shape
#     for attribute in range(col):
#         attribute_matrix.append(dataset[:, attribute])
#     x = attribute_matrix[4]
#     y = attribute_matrix[5]
#
#     plt.autoscale(enable=True)
#     plt.scatter(x, y)
#
#     plt.title('Relationship between FFMC and DMC')
#     plt.xlabel('Fine Fuel Moisture Code (Moisture for Shaded Litter Fuel)')
#     plt.ylabel('Duff Moisture Code (Moisture for Soil )')
#     plt.grid(True)
#     plt.savefig('3.5Pairs_FFMC_DMC.png')
#     plt.show()

# def plotting_scatter_plots(dataset):
#     attribute_matrix = []
#     row, col = dataset.shape
#     for attribute in range(col):
#         attribute_matrix.append(dataset[:, attribute])
#     x = attribute_matrix[8]
#     y = attribute_matrix[9]
#
#     plt.autoscale(enable=True)
#     plt.scatter(x, y)
#
#     plt.title('Relationship between  Humidity and Temperature')
#     plt.xlabel('Temperature (in Celsius)')
#     plt.ylabel('Relative Humidity')
#     plt.grid(True)
#     plt.savefig('4.5Pairs_RH_Temp.png')
#     plt.show()

# def plotting_scatter_plots(dataset):
#     attribute_matrix = []
#     row, col = dataset.shape
#     for attribute in range(col):
#         attribute_matrix.append(dataset[:, attribute])
#     x = attribute_matrix[4]
#     y = attribute_matrix[6]
#
#     plt.autoscale(enable=True)
#     plt.scatter(x, y)
#
#     plt.title('Relationship between FFMC and DC')
#     plt.xlabel('Fine Fuel Moisture Code')
#     plt.ylabel('Drought Code')
#     plt.grid(True)
#     plt.savefig('5.5Pairs_FFMC_DC.png')
#     plt.show()
#
# def plotting_scatter_plots(dataset):
#     attribute_matrix = []
#     row, col = dataset.shape
#     for attribute in range(col):
#         attribute_matrix.append(dataset[:, attribute])
#     x = attribute_matrix[4]
#     y = attribute_matrix[6]
#
#     plt.autoscale(enable=True)
#     plt.scatter(x, y)
#
#     plt.title('Relationship between FFMC and DC')
#     plt.xlabel('Fine Fuel Moisture Code')
#     plt.ylabel('Drought Code')
#     plt.grid(True)
#     plt.savefig('asdf3.5Pairs_FFMC_DC.png')
#     plt.show()


def gcorr_graph(dataset):
    x, y, corr, d1, d2 = std_norm_attr_gcorr(dataset)
    x = d1
    y = d2

    plt.autoscale(enable=True)
    plt.scatter(x, y)

    plt.title('Greatest Correlation of Std. Normalization')
    plt.xlabel('Normalized DMC Data')
    plt.ylabel('Normalized DC Data')
    plt.grid(True)
    plt.savefig('gcorr_graph.png')
    plt.show()

def lcorr_graph(dataset):
    x, y, corr, d1, d2 = std_norm_attr_lcorr(dataset)
    x = d1
    y = d2

    plt.autoscale(enable=True)
    plt.scatter(x, y)

    plt.title('Smallest Correlation of Std. Normalization')
    plt.xlabel('Normalized Temperature Data')
    plt.ylabel('Normalized Relative Humidity Data')
    plt.grid(True)
    plt.savefig('lcorr_graph.png')
    plt.show()

def ranged_covar_graph(dataset):
    x, y, covar, d1, d2 = range_norm_attr(dataset)
    x = d1
    y = d2

    plt.autoscale(enable=True)
    plt.scatter(x, y)

    plt.title('Greatest Covariance of Ranged Normalization')
    plt.xlabel('Normalized DMC Data')
    plt.ylabel('Normalized DC Data')
    plt.grid(True)
    plt.savefig('ranged_covar_graph.png')
    plt.show()


def main():
    data = open_iris_data()
    data = preprocessing(data)
    data = convert_to_float(data)
    header, data_set = data[0], data[1:]
    data_set = run_label_encoding(data_set, [2,3])

    # Part 3 --------------------------------------------------
    print(multivariate_mean(data_set))
    print(covar_matrix(data_set))
    print(range_norm_attr(data_set))
    print(std_norm_attr_gcorr(data_set))
    print(std_norm_attr_lcorr(data_set))
    print(greater_than_half(data_set))
    print(neg_sample_covariance(data_set))
    print(total_variance(data_set))
    print(total_variance_five(data_set))

    # Plots ----------------------------------------------------
    # plotting_scatter_plots(data_set)
    gcorr_graph(data_set)
    lcorr_graph(data_set)
    ranged_covar_graph(data_set)

main()
