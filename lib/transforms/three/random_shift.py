import numpy as np
import scipy.ndimage as ndimage


def transform_matrix_offset_center_3d(matrix, x, y, z):
    offset_matrix = np.array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])
    return ndimage.interpolation.affine_transform(matrix, offset_matrix)


def random_shift(img_numpy, label=None, max_percentage=0.2):
    """
    Args:
        img_numpy:
        label:
        max_percentage:

    Returns:

    """
    dim1, dim2, dim3 = img_numpy.shape
    m1, m2, m3 = max(int(dim1 * max_percentage / 2), 1), max(int(dim1 * max_percentage / 2), 1), max(int(dim1 * max_percentage / 2), 1)
    d1 = np.random.randint(-m1, m1)
    d2 = np.random.randint(-m2, m2)
    d3 = np.random.randint(-m3, m3)
    if label is not None:
        return transform_matrix_offset_center_3d(img_numpy, d1, d2, d3), \
               transform_matrix_offset_center_3d(label, d1, d2, d3)
    return transform_matrix_offset_center_3d(img_numpy, d1, d2, d3)


class RandomShift(object):
    def __init__(self, max_percentage=0.2):
        self.max_percentage = max_percentage

    def __call__(self, img_numpy, label=None):
        img_numpy, label = random_shift(img_numpy, label, self.max_percentage)
        return img_numpy, label
