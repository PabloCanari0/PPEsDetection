import os
import numpy as np
import albumentations as A
import torch
import torchvision.transforms as transforms
import csv
from PIL import Image
from torchvision.transforms import v2
from torchvision import tv_tensors
#   This script allows data augmentation to be applied to a certain list of images of a certain class. The bounding boxes
# will also be modified and the CSV file extended with the new labels from the new images 

def dataAugmentation(n, m, classs, csv_path, image_path):
    # Load existing annotations CSV
    annotations = pd.read_csv(csv_path)
    
    # Define a list of transformations to apply
    transforms = [
        A.Compose([
            A.RandomCrop(width=200, height=200),  # Example: random crop
            A.HorizontalFlip(p=0.5)               # Example: horizontal flip
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category'])),
        A.Compose([], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category'])),  # Placeholder
        # Add more transforms if needed
    ]
    
    images = classFinder(classs, csv_path, image_path)  # Find all images of a certain class
    
    new_rows = []  # List to store new annotation rows

    for I in images:
        image_name = os.path.basename(I)
        image = np.array(Image.open(I).convert("RGB"))  # Convert image to NumPy array

        # Filter annotations for this specific image
        rows = annotations[annotations['filename'] == image_name]
        
        bboxes = []
        categories = []
        for _, row in rows.iterrows():
            bboxes.append([row['xmin'], row['ymin'], row['xmax'], row['ymax']])
            categories.append(row['label'])

        # Apply n random augmentations
        for _ in range(n):
            transform = random.choice(transforms)
            augmented = transform(image=image, bboxes=bboxes, category=categories)
            
            transformed_image = Image.fromarray(augmented['image'])  # Convert back to PIL image
            transformed_image = transformed_image.resize((640, 640))  # Resize to desired size
            transformed_bboxes = augmented['bboxes']
            transformed_categories = augmented['category']
            
            # Generate unique name for the augmented image
            new_name = f"{image_name.rsplit('.', 1)[0]}_{''.join(random.choices(string.ascii_lowercase + string.digits, k=6))}.jpg"
            new_path = os.path.join(image_path, new_name)
            transformed_image.save(new_path)

            # Add one row for each bbox in the transformed image
            for label, box in zip(transformed_categories, transformed_bboxes):
                new_rows.append({
                    'filename': new_name,
                    'width': 640,
                    'height': 640,
                    'label': label,
                    'xmin': box[0],
                    'ymin': box[1],
                    'xmax': box[2],
                    'ymax': box[3],
                })

    # Append new annotations and save to CSV
    if new_rows:
        annotations = pd.concat([annotations, pd.DataFrame(new_rows)], ignore_index=True)
        annotations.to_csv(csv_path, index=False)



def classFinder(classs, csv_path, image_path): # Searches for images with certain labels in dataset
    annotations = pd.read_csv(csv_path)
    images=[] # Will store all the images of a certain class 
    for _, row in annotations.iterrows(): # Iterate over CSV file
        if row['class']==classs: 
            new_entry=image_path.join(row['filename']) # Capture image path 
            if os.path_exists(new_entry):
                images.append(new_entry)
    return images










