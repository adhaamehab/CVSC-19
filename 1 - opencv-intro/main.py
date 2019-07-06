"""Basics of opencv image operations."""
import cv2
import numpy as np


def normalize_images(image_1, image_2):
    """Prevents `Sizes of input arguments do not match` by resizing the second image.

    :param image_1: `numpy.ndarray` image as a multi dimensional np array.
    :param image_2: `numpy.ndarray` image as a multi dimensional np array.
    :return: both images after resizing the second image with to first image size
    """
    return image_1, cv2.resize(image_2, image_1.shape[:2])


def read_image(image_name):
    """Read an image and return it as a numpy array.

    :param image_name: `str` the full path and name of the image.
    :return: `numpy.ndarray` image as a multi dimensional np array
    """
    return cv2.imread(image_name)


def save_image(image_name, image):
    """Save a numpy array as an image with the path and return it as a numpy array.

    :param image_name: `str` the full path and name of the image.
    :param image: `numpy.ndarray` image as a multi dimensional np array
    :return: None
    """
    cv2.imwrite(image_name, image)


def add_images(image_a, image_b):
    """Add every pixel together.

    :param image_a: `numpy.ndarray` image as a multi dimensional np array.
    :param image_b: `numpy.ndarray` image as a multi dimensional np array.
    :return: the addition result as a `numpy.ndarray`
    """
    return cv2.add(image_a, image_b)


def sub_images(image_a, image_b):
    """Sub every pixel from each other.

    :param image_a: `numpy.ndarray` image as a multi dimensional np array.
    :param image_b: `numpy.ndarray` image as a multi dimensional np array.
    :return: the substraction result as a `numpy.ndarray`
    """
    return cv2.subtract(image_a, image_b)


def mul_images(image_a, image_b):
    """Mul every pixel with each other.

    :param image_a: `numpy.ndarray` image as a multi dimensional np array.
    :param image_b: `numpy.ndarray` image as a multi dimensional np array.
    :return: the substraction result as a `numpy.ndarray`
    """
    return cv2.multiply(image_a, image_b)


def div_images(image_a, image_b):
    """Divide every pixel with each other.

    :param image_a: `numpy.ndarray` image as a multi dimensional np array.
    :param image_b: `numpy.ndarray` image as a multi dimensional np array.
    :return: the substraction result as a `numpy.ndarray`
    """
    return cv2.divide(image_a, image_b)


def xor_images(image_a, image_b):
    """XOR every pixel with each other.

    :param image_a: `numpy.ndarray` image as a multi dimensional np array.
    :param image_b: `numpy.ndarray` image as a multi dimensional np array.
    :return: the substraction result as a `numpy.ndarray`
    """
    return cv2.bitwise_xor(image_a, image_b)


def to_gray(image):
    """Convert image from RGB based to gray scale.

    :param image: `numpy.ndarray` image as a multi dimensional np array.
    :return: `numpy.ndarray` gray image as a multi dimensional np array.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def rotate_image(image, angle, scale=1.0):
    """Rotate image with angel with respect to the X-axis.

    :param image: `numpy.ndarray` image as a multi dimensional np array.
    :param array: `float` angle to calculate the rotation matrix.
    :param scale: `float` image scale ratio
    :return: `numpy.ndarray` gray image as a multi dimensional np array.
    """
    image_center = tuple(
        np.array(image.shape[1::-1]) / 2)  # image.shape = X, Y, Z => [1::-1] =  Y, X => /2 = center
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, scale)
    return cv2.warpAffine(
        image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)


if __name__ == "__main__":
    image_1 = read_image("image_1.jpg")
    image_2 = read_image("image_2.jpg")

    image_1, image_2 = normalize_images(image_1, image_2)

    res = rotate_image(image_1, 30, 2.3)

    save_image("result.jpg", res)
