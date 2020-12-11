import cv2
import numpy as np
from scipy.spatial import distance
from skimage import metrics


def contour_image(image):
    img = cv2.imread(image, 0)
    img = cv2.resize(img, (200, 100))
    # ret, th = cv2.threshold(img, 90, 255, cv2.THRESH_BINARY_INV)
    th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 1001, 40)

    contours, hierarchy = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_xy = np.array(contours, dtype=object)
    x_min, x_max = 0, 0
    value = list()
    for i in range(len(contours_xy)):
        for j in range(len(contours_xy[i])):
            value.append(contours_xy[i][j][0][0])  # 네번째 괄호가 0일때 x의 값
            x_min = min(value)
            x_max = max(value)
    y_min, y_max = 0, 0
    value = list()
    for i in range(len(contours_xy)):
        for j in range(len(contours_xy[i])):
            value.append(contours_xy[i][j][0][1])  # 네번째 괄호가 0일때 x의 값
            y_min = min(value)
            y_max = max(value)
    x = x_min
    y = y_min
    w = x_max - x_min
    h = y_max - y_min
    img_trim = th[y:y + h, x:x + w]
    resize_img = cv2.resize(img_trim, (200, 100))

    return resize_img


def cal_similarity(image1, image2):
    Aflat = np.hstack(image1)
    Bflat = np.hstack(image2)
    return distance.cosine(Aflat, Bflat)


def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err


if __name__ == '__main__':
    reference = contour_image('hello4.png')

    comparison1 = contour_image('hello1.jpg')
    comparison2 = contour_image('hello2.png')
    comparison3 = contour_image('hello3.png')

    cv2.imshow('reference', reference)
    cv2.imshow('comparison1', comparison1)
    cv2.imshow('comparison2', comparison2)
    cv2.imshow('comparison3', comparison3)

    print(cal_similarity(reference, comparison1))
    print(cal_similarity(reference, comparison2))
    print(cal_similarity(reference, comparison3))
    print()
    print(metrics.structural_similarity(reference, comparison1))
    print(metrics.structural_similarity(reference, comparison2))
    print(metrics.structural_similarity(reference, comparison3))

    cv2.waitKey(0)
    cv2.destroyAllWindows()
