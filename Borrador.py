import os
import torch
from collections import defaultdict
import pandas as pd
import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import random
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.transforms.functional as F
import torchvision.utils as vutils
import torch
from torchvision.utils import draw_bounding_boxes
# Borrador
def __countCategory__(csv): # Number of elements from each class
    accElements = defaultdict(int) # Dictionary with default values
    for _, row in csv.iterrows(): # For all the rows in the CSV file
        category = row[3]  # Read category
        accElements[category] += 1  # Increment the category counter
    # Turn into a regular dictionary
    accElements = dict(accElements)
    return(accElements)
PPE_frame = pd.read_csv("C:/Users/vgarc/Pictures/anotaciones.csv")
result=__countCategory__(PPE_frame)
print(result)