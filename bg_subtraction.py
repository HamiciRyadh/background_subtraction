import numpy as np


def frame_differencing(image, background, threshold):
    # The images have to be of type int16.
    result = np.abs(image - background)

    moving = result >= threshold
    result[moving] = 255
    result[np.logical_not(moving)] = 0

    return result


def mean_filter(previous_images, image, threshold):
    buffer = np.array(previous_images)
    mean = np.empty(image.shape, dtype=np.int16)
    np.mean(buffer, axis=0, out=mean)
    return frame_differencing(image, mean, threshold)


def median_filter(previous_images, image, threshold):
    buffer = np.array(previous_images)
    median = np.empty(image.shape, dtype=np.int16)
    np.median(buffer, axis=0, out=median)
    return frame_differencing(image, median, threshold)


def gaussian_unique(image, mean, variance, alpha, k):
    new_mean = alpha*image - (1-alpha)*mean
    absolute_difference = np.abs(image - new_mean)
    new_variance = alpha*(absolute_difference**2) + (1-alpha)*variance

    result = absolute_difference/np.sqrt(new_variance)
    foreground = result > k
    background = np.logical_not(foreground)
    result[foreground] = 255
    result[background] = 0

    return result.astype(np.uint8), new_mean, new_variance
