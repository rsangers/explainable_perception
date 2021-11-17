import numpy as np
import cv2


class SubstractMean:

    def __init__(self,mean):
        self.mean = mean

    def __call__(self, image_arr):
        return image_arr - self.mean

class ToArray:

    def __call__(self, image):
        return np.asarray(image, np.float32)

class Resize:

    def __init__(self,size):
        self.size = size

    def __call__(self, image_arr):
        return cv2.resize(image_arr, self.size)

class ToTorchDims:

    def __call__(self, image_arr):
        return image_arr.transpose((2, 0, 1))