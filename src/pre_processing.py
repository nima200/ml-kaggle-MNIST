import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from skimage.filters import sobel
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import closing, square, watershed
from skimage.transform import resize


def get_splits():
    X = pd.read_csv('./data/train_x.csv', sep=',')
    y = pd.read_csv('./data/train_y.csv', sep=',')

    X = X.as_matrix().reshape(-1, 64, 64)
    y = y.as_matrix().reshape(-1, 1)

    # USe 10% of data for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.10)

    X_train_new = np.zeros(shape=(X_train.shape[0], 28, 28))
    X_test_new = np.zeros(shape=(X_test.shape[0], 28, 28))

    # Find the biggest digit in the image and separate it into a new matrix array
    for i, image in enumerate(X_train):
        clean_image = find_biggest_digit(image)
        X_train_new[i] = clean_image

    for i, image in enumerate(X_test):
        clean_image = find_biggest_digit(image)
        X_test_new[i] = clean_image

    # Each list entry in the final training data is a tuple with the first element being a 28x28 matrix for the image
    # pixels, and the second entry being a vectorized representation of the image digit classification
    X_train_new = [np.reshape(x, (x.size, 1)) for x in X_train_new]
    y_train = [vectorized_result(y) for y in y_train]
    training_data = list(zip(X_train_new, y_train))

    # Each list entry in the final test data is a tuple with the first element being a 28x28 matrix for the image
    # pixels, and the second entry being a scalar value for the image digit classification
    X_test_new = [np.reshape(x, (x.size, 1)) for x in X_test_new]
    test_data = list(zip(X_test_new, y_test))

    return training_data, test_data


def get_training():
    X = pd.read_csv('./data/train_x.csv', sep=',')
    y = pd.read_csv('./data/train_y.csv', sep=',')

    X = X.as_matrix().reshape(-1, 64, 64)
    y = y.as_matrix().reshape(-1, 1)

    X_train = np.zeros(shape=(X.shape[0], 28, 28))

    # Find the biggest digit in the image and separate it into a new matrix array
    for i, image in enumerate(X):
        clean_image = find_biggest_digit(image)
        X_train[i] = clean_image

    # Each list entry in the final training data is a tuple with the first element being a 28x28 matrix for the image
    # pixels, and the second entry being a vectorized representation of the image digit classification
    X_train = [np.reshape(x, (x.size, 1)) for x in X_train]
    y_train = [vectorized_result(y) for y in y]
    training_data = list(zip(X_train, y_train))
    return training_data


def get_test():
    X = pd.read_csv('./data/test_x.csv', header=None).as_matrix()

    X = X.reshape(-1, 64, 64)

    X_test = np.zeros(shape=(X.shape[0], 28, 28))

    # Find the biggest digit in the image and separate it into a new matrix array
    for i, image in enumerate(X):
        clean_image = find_biggest_digit(image)
        X_test[i] = clean_image

    # Each list entry in the final test data is a 28x28 matrix representing the image's pixel data
    X_test = [np.reshape(x, (x.size, 1)) for x in X_test]

    return X_test


def find_biggest_digit(image):
    # Create the elevation map using the sobel filter
    elevation_map = sobel(image)

    # Create markings from areas where image is white and where it's not for segmentation
    markers = np.zeros_like(image)
    markers[image != 255] = 1
    markers[image == 255] = 2
    # Apply segmentation on the elevation map using the markers
    segmentation = watershed(elevation_map, markers)

    # Calculate a threshold value for separating the regions in the image
    threshold = threshold_otsu(segmentation)
    bw = closing(segmentation > threshold, square(1))

    # Create an area labeling using the different labels from previous step
    label_image = label(bw)
    # Calculate different regions of the image from labels
    regions = regionprops(label_image)

    def get_area(x):
        """Calculates the area of a region's bounding box"""
        minr, minc, maxr, maxc = x.bbox
        width = maxc - minc
        height = maxr - minr
        return width * height

    # Sort regions based on max bounding box area
    regions.sort(key=lambda x: get_area(x), reverse=True)
    biggest_region = regions[0]
    final_img = biggest_region.image
    # Re-scale the image to a smaller size for computational efficiency
    final_img = resize(final_img, (28, 28), mode='constant')
    # Zero out pixels that are not white, set all white pixels to 255 in value
    final_img[final_img < 1] = 0
    final_img[final_img == 1] = 255

    return final_img


def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
