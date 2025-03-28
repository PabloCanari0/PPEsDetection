import os
import numpy as np
import albumentations as A
import torch
import torchvision.transforms as transforms
import csv
import cv2
from PIL import Image
from torchvision.transforms import v2
from torchvision import tv_tensors
#   This script allows data augmentation to be applied to a certain list of images of a certain class. The bounding boxes
# will also be modified and the CSV file extended with the new labels from the new images 
def dataAugmentation(n,m,clase,path):
    # n: nº of transforms
    # m: magnitude of transforms
    # images: of certain class to be transformed 
    # clase: class or type of images to extract
    # path: to CSV file 
    annotations=open('***********************************', mode='r') # Opens CSV files with labels of the images 
    reader = csv.reader(annotations) # Create reader to enable search
    transforms=[A.Compose([A.RandomCrop()], bbox_params=A.BboxParams(format='pascal_voc')), # Define a list of transforms, they will be chosen randomly
                A.Compose([]),
                A.Compose([]),
                A.Compose([]),
                A.Compose([]),
                A.Compose([]),
                ]
    images=classFinder()
    for I in images: # For all the images of a certain class (*.jpg files)
        image = Image.open(I) # Open JPEG file 
        actual_row=[] 
        for row in reader: # Go over all the rows
            if I in str(row[0]): actual_row.append(row[4:7])  #Until you find the name of the image jpg file and save the row with labels. Multiple rows possible (multiple boxes in the same image)
        bboxes=actual_row.reshape(-1,4) # Extract bounding box corners coordinates [xmin ymin xmax ymax]
        while n>0: # Repeat n times to image I
            image_=transform[randint(0,9)](image=image, bboxes=bboxes) # Apply random transform from the 10 available to image and box
            transformed_image = transformed['image'] #Get transformed image
            transformed_image = transformed_image.resize(***************************) # Normalize size
            transformed_bboxes = transformed['bboxes'] #Get transformed bounding boxes
            new_name = ''.join(random.choices(string.ascii_letters + string.digits, k=20)) # Create a random name for new image
            for i in bboxes.shape[0]: # For all the bounding boxes
                annotations.append(new_name,SIZE, )


def classFinder(): # Searches for images with certain labels in dataset 








