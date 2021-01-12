import os
import cv2
import glob
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def get_data(PATH):
    """ Read images and make a dataset.

    Args:
        PATH (str): path to image folder.

    Returns:
        X (numpy arr): List of image data.
    """
    X = []
    for img in glob.glob(PATH+"/*"):
        x = np.array(Image.open(img).convert('RGB'))
        X.append(x)
    return np.array(X)


def data_shuffle(X, Y=[]):
    """ Shuffle image data and label, keeping the correspondence.
    """
    np.random.seed(5)
    p = np.random.permutation(len(X))
    if len(Y)>=1: 
        assert len(X)==len(Y), 'length of X and Y is not matched.'
        return X[p], Y[p]
    return X[p]
