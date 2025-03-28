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
# This script loads all the different datasets gathered using a custom dataset as a class 
# It also provides a visualizator of dataset images

# FUNCTIONS, CLASSES and TRANSFORMS
def showDatasetImage(image,bboxes,categories): # Shows image and it's bounding boxes and categories
    for boxes in bboxes: # For every bounding box
        image_tensor = F.to_tensor(image) * 255 # Convert image to tensor and re-escalate
        bboxes_tensor = torch.tensor(bboxes, dtype=torch.float32) # Convert bounding boxes to tensors
        image_with_boxes = draw_bounding_boxes(image_tensor.to(torch.uint8), bboxes_tensor, labels=categories, colors="red", width=2)
        plt.imshow(image_with_boxes.permute(1, 2, 0))  # Reorder channels for visualization (width, height, image)
        plt.axis('off')
        plt.show()

def Visualizator(dataset,index=None): # Reads dataset and calls showImage function
    Visualize_data=dataset # Choose a dataset to be shown
    if index is None :index=random.randint(0,len(Visualize_data.PPE_frame)) # Choose a random index from the dataset
    Sample=Visualize_data.__getitem__(index) # Get the image associated to this random index
    showDatasetImage(Sample['image'],Sample['bounding_boxes'],Sample['category']) # Call function to show image


class PPEsDataset(Dataset):
    # Personal Protection Equipment Dataset Class for defining different datasets

    def __init__(self, csv_file, root_dir, transform): # Initialization of the data set
        
        # Arguments:
            # csv_file (string): Path to the csv file with annotations.
            # root_dir (string): Directory with all the images.
            # transform (callable, optional): for resizing the dataset or other transforms
        self.PPE_frame = pd.read_csv(csv_file) # Converts CSV into dataframe
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self): # Return CSV length
        return len(self.PPE_frame)  

    def __countCategory__(self): # Number of elements from each class
        accElements = defaultdict(int) # Dictionary with default values
        for _, row in self.PPE_frame.iterrows(): # For all the rows in the CSV file
            category = row[3]  # Read category
            accElements[category] += 1  # Increment the category counter
        # Turn into a regular dictionary
        accElements = dict(accElements)
        return(accElements)

    def __getitem__(self, idx): # Returns certain image of the dataset
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_row=self.PPE_frame.iloc[idx,0] # Single row of annotations file 
        img_name = os.path.join(self.root_dir,image_row) # Full image path
        #image = io.imread(img_name) # Image
        image = plt.imread(img_name)
        # One image can have multiple bounding boxes and categories, thus it will have multiple entries on the CSV list
        bboxes=[] # Will store the bounding boxes
        categories=[] # Will store the different categories
        for _, row in self.PPE_frame.iterrows(): # For all the rows in the CSV file
            if image_row in str(row[0]): # Search for rows with common image
                bboxes.append(row[4:8])  # Append bounding boxes to the bounding box
                categories.append(row[3])  # Store only the category separately
        
        categories = np.array(categories).reshape(-1, 1)
        bboxes = np.array(bboxes, dtype=np.float32).reshape(-1, 4) # Turn into an array and separate the bounding boxes in groups of 4 [xmin ymin xmax ymax category]

        if self.transform:
            transformed = self.transform(image=image, bboxes=bboxes, category=categories) #Apply transform to image, bounding boxes and labels
            image = transformed['image']
            bboxes = transformed['bboxes']
            categories = transformed['category']
        sample = {'image': image, 'bounding_boxes': bboxes, 'category':categories}
        return sample

transformResize=A.Compose([A.Resize(height=640,width=640)], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category'])) # Resize transform for normalization

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# DATA-SETS:
backgroundpics=PPEsDataset(csv_file="C:/Users/vgarc/Desktop/TFG/DataSets/background/_annotations.csv",
                          root_dir="C:/Users/vgarc/Desktop/TFG/DataSets/background",
                          transform=transformResize)


# TEST sets
# 1. PPE Dataset for Workplace Safety Computer Vision Project (https://universe.roboflow.com/siabar/ppe-dataset-for-workplace-safety)
WorkplaceSafetyTEST=PPEsDataset(csv_file="C:/Users/vgarc/Desktop/TFG/DataSets/PPE Dataset for Workplace Safety/test/_annotations.csv",
                                root_dir="C:/Users/vgarc/Desktop/TFG/DataSets/PPE Dataset for Workplace Safety/test/",
                                transform=transformResize)


# 2. Worker-Safety Computer Vision Project (https://universe.roboflow.com/computer-vision/worker-safety). 
WorkerSafetyTEST=PPEsDataset(csv_file="C:/Users/vgarc/Desktop/TFG/DataSets/Worker-Safety.v1-workersafety.tensorflow/test/_annotations.csv",
                             root_dir="C:/Users/vgarc/Desktop/TFG/DataSets/Worker-Safety.v1-workersafety.tensorflow/test/",
                             transform=transformResize)


# 3. PPE Detection Computer Vision Project (https://universe.roboflow.com/pram/ppe-detection-z3v2w)
PPEDetectionTEST=PPEsDataset(csv_file="C:/Users/vgarc/Desktop/TFG/DataSets/PPE Detection.v2-ppedetpramv2.tensorflow/test/_annotations.csv",
                             root_dir="C:/Users/vgarc/Desktop/TFG/DataSets/PPE Detection.v2-ppedetpramv2.tensorflow/test",
                             transform=transformResize) 


# 4. TallerYOLO Computer Vision Project (https://universe.roboflow.com/gonzalo-8ifbf/talleryolo-mc28j)
TallerYOLOTEST=PPEsDataset(csv_file="C:/Users/vgarc/Desktop/TFG/DataSets/TallerYOLO.v3i.tensorflow/test/_annotations.csv",
                           root_dir="C:/Users/vgarc/Desktop/TFG/DataSets/TallerYOLO.v3i.tensorflow/test",
                           transform=transformResize)


# 5. PPE2 Computer Vision Project (https://universe.roboflow.com/bangga/ppe-2-ynh14)
PPE2TEST=PPEsDataset(csv_file="C:/Users/vgarc/Desktop/TFG/DataSets/PPE 2.v2i.tensorflow/test/_annotations.csv",
                     root_dir="C:/Users/vgarc/Desktop/TFG/DataSets/PPE 2.v2i.tensorflow/test",
                     transform=transformResize)


# 6. Heavy_Equipment Computer Vision Project (https://universe.roboflow.com/kfu-ye4kz/heavy_equipment-ifaqm)
Heavy_EquipmentTEST=PPEsDataset(csv_file="C:/Users/vgarc/Desktop/TFG/DataSets/Heavy_Equipment.v2i.tensorflow/test/_annotations.csv",
                                root_dir="C:/Users/vgarc/Desktop/TFG/DataSets/Heavy_Equipment.v2i.tensorflow/test",
                                transform=transformResize)


# 7. RAVEN - Loader Computer Vision Project (https://universe.roboflow.com/raven-cv-ivnon/raven-loader/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true)
RavenLoaderTEST=PPEsDataset(csv_file="C:/Users/vgarc/Desktop/TFG/DataSets/RAVEN - Loader.v1i.tensorflow/test/_annotations.csv",
                            root_dir="C:/Users/vgarc/Desktop/TFG/DataSets/RAVEN - Loader.v1i.tensorflow/test",
                            transform=transformResize)


# 8. check_ss Computer Vision Project (https://universe.roboflow.com/data2-1tbnu/check_ss) 
check_ssTEST=PPEsDataset(csv_file="C:/Users/vgarc/Desktop/TFG/DataSets/check_ss.v1i.tensorflow/test/_annotations.csv",
                        root_dir="C:/Users/vgarc/Desktop/TFG/DataSets/check_ss.v1i.tensorflow/test",
                        transform=transformResize)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
# TRAIN sets
# 1. PPE Dataset for Workplace Safety Computer Vision Project (https://universe.roboflow.com/siabar/ppe-dataset-for-workplace-safety)
WorkplaceSafetyTRAIN=PPEsDataset(csv_file="C:/Users/vgarc/Desktop/TFG/DataSets/PPE Dataset for Workplace Safety/train/_annotations.csv",
                                root_dir="C:/Users/vgarc/Desktop/TFG/DataSets/PPE Dataset for Workplace Safety/train/",
                                transform=transformResize)



# 2. Worker-Safety Computer Vision Project (https://universe.roboflow.com/computer-vision/worker-safety). 
WorkerSafetyTRAIN=PPEsDataset(csv_file="C:/Users/vgarc/Desktop/TFG/DataSets/Worker-Safety.v1-workersafety.tensorflow/train/_annotations.csv",
                             root_dir="C:/Users/vgarc/Desktop/TFG/DataSets/Worker-Safety.v1-workersafety.tensorflow/train/",
                             transform=transformResize)


# 3. PPE Detection Computer Vision Project (https://universe.roboflow.com/pram/ppe-detection-z3v2w)
PPEDetectionTRAIN=PPEsDataset(csv_file="C:/Users/vgarc/Desktop/TFG/DataSets/PPE Detection.v2-ppedetpramv2.tensorflow/train/_annotations.csv",
                             root_dir="C:/Users/vgarc/Desktop/TFG/DataSets/PPE Detection.v2-ppedetpramv2.tensorflow/train",
                             transform=transformResize) 


# 4. TallerYOLO Computer Vision Project (https://universe.roboflow.com/gonzalo-8ifbf/talleryolo-mc28j)
TallerYOLOTRAIN=PPEsDataset(csv_file="C:/Users/vgarc/Desktop/TFG/DataSets/TallerYOLO.v3i.tensorflow/train/_annotations.csv",
                           root_dir="C:/Users/vgarc/Desktop/TFG/DataSets/TallerYOLO.v3i.tensorflow/train",
                           transform=transformResize)


# 5. PPE2 Computer Vision Project (https://universe.roboflow.com/bangga/ppe-2-ynh14)
PPE2TRAIN=PPEsDataset(csv_file="C:/Users/vgarc/Desktop/TFG/DataSets/PPE 2.v2i.tensorflow/train/_annotations.csv",
                     root_dir="C:/Users/vgarc/Desktop/TFG/DataSets/PPE 2.v2i.tensorflow/train",
                     transform=transformResize)


# 6. Heavy_Equipment Computer Vision Project (https://universe.roboflow.com/kfu-ye4kz/heavy_equipment-ifaqm)
Heavy_EquipmentTRAIN=PPEsDataset(csv_file="C:/Users/vgarc/Desktop/TFG/DataSets/Heavy_Equipment.v2i.tensorflow/train/_annotations.csv",
                                root_dir="C:/Users/vgarc/Desktop/TFG/DataSets/Heavy_Equipment.v2i.tensorflow/train",
                                transform=transformResize)


# 7. RAVEN - Loader Computer Vision Project (https://universe.roboflow.com/raven-cv-ivnon/raven-loader/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true)
RavenLoaderTRAIN=PPEsDataset(csv_file="C:/Users/vgarc/Desktop/TFG/DataSets/RAVEN - Loader.v1i.tensorflow/train/_annotations.csv",
                            root_dir="C:/Users/vgarc/Desktop/TFG/DataSets/RAVEN - Loader.v1i.tensorflow/train",
                            transform=transformResize)


# 8. check_ss Computer Vision Project (https://universe.roboflow.com/data2-1tbnu/check_ss) 
check_ssTRAIN=PPEsDataset(csv_file="C:/Users/vgarc/Desktop/TFG/DataSets/check_ss.v1i.tensorflow/train/_annotations.csv",
                          root_dir="C:/Users/vgarc/Desktop/TFG/DataSets/check_ss.v1i.tensorflow/train",
                          transform=transformResize)


# 9. gogglessss Computer Vision Project (https://universe.roboflow.com/safetynew/gogglessss).
goglesssTRAIN=PPEsDataset(csv_file="C:/Users/vgarc/Desktop/TFG/DataSets/gogglessss.v1i.tensorflow/train/_annotations.csv",
                          root_dir="C:/Users/vgarc/Desktop/TFG/DataSets/gogglessss.v1i.tensorflow/train",
                          transform=transformResize) 

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# VALIDATION sets
# 1. PPE Dataset for Workplace Safety Computer Vision Project (https://universe.roboflow.com/siabar/ppe-dataset-for-workplace-safety)
WorkplaceSafetyVALID=PPEsDataset(csv_file="C:/Users/vgarc/Desktop/TFG/DataSets/PPE Dataset for Workplace Safety/valid/_annotations.csv",
                                root_dir="C:/Users/vgarc/Desktop/TFG/DataSets/PPE Dataset for Workplace Safety/valid/",
                                transform=transformResize)

# 2. Worker-Safety Computer Vision Project (https://universe.roboflow.com/computer-vision/worker-safety). 
WorkerSafetyVALID=PPEsDataset(csv_file="C:/Users/vgarc/Desktop/TFG/DataSets/Worker-Safety.v1-workersafety.tensorflow/valid/_annotations.csv",
                             root_dir="C:/Users/vgarc/Desktop/TFG/DataSets/Worker-Safety.v1-workersafety.tensorflow/valid/",
                             transform=transformResize)

# 3. PPE Detection Computer Vision Project (https://universe.roboflow.com/pram/ppe-detection-z3v2w)
PPEDetectionVALID=PPEsDataset(csv_file="C:/Users/vgarc/Desktop/TFG/DataSets/PPE Detection.v2-ppedetpramv2.tensorflow/valid/_annotations.csv",
                             root_dir="C:/Users/vgarc/Desktop/TFG/DataSets/PPE Detection.v2-ppedetpramv2.tensorflow/valid",
                             transform=transformResize) 

# 4. TallerYOLO Computer Vision Project (https://universe.roboflow.com/gonzalo-8ifbf/talleryolo-mc28j)
TallerYOLOVALID=PPEsDataset(csv_file="C:/Users/vgarc/Desktop/TFG/DataSets/TallerYOLO.v3i.tensorflow/valid/_annotations.csv",
                           root_dir="C:/Users/vgarc/Desktop/TFG/DataSets/TallerYOLO.v3i.tensorflow/valid",
                           transform=transformResize)

# 5. PPE2 Computer Vision Project (https://universe.roboflow.com/bangga/ppe-2-ynh14)
PPE2VALID=PPEsDataset(csv_file="C:/Users/vgarc/Desktop/TFG/DataSets/PPE 2.v2i.tensorflow/valid/_annotations.csv",
                     root_dir="C:/Users/vgarc/Desktop/TFG/DataSets/PPE 2.v2i.tensorflow/valid",
                     transform=transformResize)

# 6. Heavy_Equipment Computer Vision Project (https://universe.roboflow.com/kfu-ye4kz/heavy_equipment-ifaqm)
Heavy_EquipmentVALID=PPEsDataset(csv_file="C:/Users/vgarc/Desktop/TFG/DataSets/Heavy_Equipment.v2i.tensorflow/valid/_annotations.csv",
                                root_dir="C:/Users/vgarc/Desktop/TFG/DataSets/Heavy_Equipment.v2i.tensorflow/valid",
                                transform=transformResize)

# 7. RAVEN - Loader Computer Vision Project (https://universe.roboflow.com/raven-cv-ivnon/raven-loader/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true)
RavenLoaderVALID=PPEsDataset(csv_file="C:/Users/vgarc/Desktop/TFG/DataSets/RAVEN - Loader.v1i.tensorflow/valid/_annotations.csv",
                            root_dir="C:/Users/vgarc/Desktop/TFG/DataSets/RAVEN - Loader.v1i.tensorflow/valid",
                            transform=transformResize)

# 8. check_ss Computer Vision Project (https://universe.roboflow.com/data2-1tbnu/check_ss) 
check_ssVALID=PPEsDataset(csv_file="C:/Users/vgarc/Desktop/TFG/DataSets/check_ss.v1i.tensorflow/valid/_annotations.csv",
                          root_dir="C:/Users/vgarc/Desktop/TFG/DataSets/check_ss.v1i.tensorflow/valid",
                          transform=transformResize)

# 9. gogglessss Computer Vision Project (https://universe.roboflow.com/safetynew/gogglessss).
goglesssVALID=PPEsDataset(csv_file="C:/Users/vgarc/Desktop/TFG/DataSets/gogglessss.v1i.tensorflow/valid/_annotations.csv",
                          root_dir="C:/Users/vgarc/Desktop/TFG/DataSets/gogglessss.v1i.tensorflow/valid",
                          transform=transformResize)

#------------------------------------------------------------------------------------------------------------------------------------------------
print("C:/Users/vgarc/Desktop/TFG/DataSets/background \n")
print(backgroundpics.__countCategory__())

print("C:/Users/vgarc/Desktop/TFG/DataSets/PPE Dataset for Workplace Safety \n")
print(WorkplaceSafetyTEST.__countCategory__())
print(WorkplaceSafetyTRAIN.__countCategory__())
print(WorkplaceSafetyVALID.__countCategory__())

print("C:/Users/vgarc/Desktop/TFG/DataSets/Worker-Safety.v1-workersafety.tensorflow \n")
print(WorkerSafetyTEST.__countCategory__())
print(WorkerSafetyTRAIN.__countCategory__())
print(WorkerSafetyVALID.__countCategory__())

print("C:/Users/vgarc/Desktop/TFG/DataSets/PPE Detection.v2-ppedetpramv2.tensorflow \n")
print(PPEDetectionTEST.__countCategory__())
print(PPEDetectionTRAIN.__countCategory__())
print(PPEDetectionVALID.__countCategory__())

print("C:/Users/vgarc/Desktop/TFG/DataSets/TallerYOLO.v3i.tensorflow \n")
print(TallerYOLOTEST.__countCategory__())
print(TallerYOLOTRAIN.__countCategory__())
print(TallerYOLOVALID.__countCategory__())

print("C:/Users/vgarc/Desktop/TFG/DataSets/PPE 2.v2i.tensorflow \n")
print(PPE2TEST.__countCategory__())
print(PPE2TRAIN.__countCategory__())
print(PPE2VALID.__countCategory__())

print("C:/Users/vgarc/Desktop/TFG/DataSets/Heavy_Equipment.v2i.tensorflow \n")
print(Heavy_EquipmentTEST.__countCategory__())
print(Heavy_EquipmentTRAIN.__countCategory__())
print(Heavy_EquipmentVALID.__countCategory__())

print("C:/Users/vgarc/Desktop/TFG/DataSets/RAVEN - Loader.v1i.tensorflow \n")
print(RavenLoaderTEST.__countCategory__())
print(RavenLoaderTRAIN.__countCategory__())
print(RavenLoaderVALID.__countCategory__())

print("C:/Users/vgarc/Desktop/TFG/DataSets/check_ss.v1i.tensorflow \n")
print(check_ssTEST.__countCategory__())
print(check_ssTRAIN.__countCategory__())
print(check_ssVALID.__countCategory__())

print("C:/Users/vgarc/Desktop/TFG/DataSets/gogglessss.v1i.tensorflow \n")
print(goglesssTRAIN.__countCategory__())
print(goglesssVALID.__countCategory__())

#Visualizator(TallerYOLOTRAIN)