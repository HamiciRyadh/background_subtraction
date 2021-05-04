import cv2
import numpy as np
import bg_subtraction as bg


def find_name(idx):
    if idx < 10:
        name = "000" + str(idx)
    elif idx < 100:
        name = "00" + str(idx)
    elif idx < 1000:
        name = "0" + str(idx)
    else:
        name = "" + str(idx)
    return "in00" + name + ".jpg"


def frame_differencing(update_background=True):
    background = np.int16(cv2.imread(input_path + find_name(offset), cv2.IMREAD_GRAYSCALE))
    for i in range(offset + 1, 1100):
        name = find_name(i)
        print("1) Frame Differencing : Image :", name)

        current = np.int16(cv2.imread(input_path + name, cv2.IMREAD_GRAYSCALE))
        cv2.imwrite(output_path + "frame_differencing/" + name, bg.frame_differencing(current, background, threshold))

        if update_background:
            background = current


def mean():
    buffer = []
    for i in range(offset - buffer_length, 1100):
        name = find_name(i)
        print("2) Mean : Image :", name)

        current = np.int16(cv2.imread(input_path + name, cv2.IMREAD_GRAYSCALE))
        if len(buffer) == buffer_length:
            cv2.imwrite(output_path + "mean/" + name, bg.mean_filter(buffer, current, threshold))
            buffer.pop(0)

        buffer.append(current)


def median():
    buffer = []
    for i in range(offset - buffer_length, 1100):
        name = find_name(i)
        print("3) Median : Image :", name)

        current = np.int16(cv2.imread(input_path + name, cv2.IMREAD_GRAYSCALE))
        if len(buffer) == buffer_length:
            cv2.imwrite(output_path + "median/" + name, bg.median_filter(buffer, current, threshold))
            buffer.pop(0)

        buffer.append(current)


def gaussian_unique():
    image = np.int16(cv2.imread(input_path + find_name(1), cv2.IMREAD_GRAYSCALE))
    g_mean = np.copy(image).astype(np.float16)
    g_variance = np.empty(image.shape, dtype=np.float16)
    g_variance[:, :] = 0

    for i in range(offset - buffer_length, 1100):
        name = find_name(i)
        print("4) Gaussian Unique : Image :", name)

        current = np.int16(cv2.imread(input_path + name, cv2.IMREAD_GRAYSCALE))
        result, g_mean, g_variance = bg.gaussian_unique(current, g_mean, g_variance, 0.25, 1.10)

        if i >= offset:
            cv2.imwrite(output_path + "Gaussian_Unique/" + name, result)


if __name__ == "__main__":
    input_path = "resources/inputs/pedestrians/"
    output_path = "resources/outputs/pedestrians/"
    threshold = 40
    buffer_length = 100
    offset = 300

    frame_differencing()
    mean()
    median()
    gaussian_unique()
