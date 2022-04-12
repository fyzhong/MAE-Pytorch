import math
import torch
import numpy as np

def mae_per_pixel(img, target):
    return np.mean(abs(img - target))
