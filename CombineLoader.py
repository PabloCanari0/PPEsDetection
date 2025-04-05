import os
import glob
from CustomData import * # Includes all available datasets defined as a type of class PPEsDataset and a dataset image visualizator
# This script allows data frame combination (concatenation) and uses a Data Loader to load in parallel
# as well as batching and shuffling the chosen data
# Dataset list from which to choose (each number on the list includes train, test and validation sets for the same dataset
# add the suffix TEST, TRAIN, VALID to access one of the specific sets for each dataset):
#   1. WorkPlaceSafety
#   2. WorkerSafety
#   3. PPEDetection
#   4. TallerYOLO
#   5. PPE2
#   6. Heavy_Equipment
#   7. RavenLoader
#   8. check_ss
#   9. goglesss* (this one only TRAIN and VALID sets!!!)
#------------------------------------------------------------------------------------------------------------------------------------------
# FUNCTIONS
def normalize_class(name): # Function for class normalization
    categories={ #Dictionary for class normalization, all classes need to have the same format (e.g different formats: helmet, Helmet)
    "background" : "background",
    "boots": ["Boots","Safety_Boots","Safety_shoes","Shoes"],
    "ear_protection":["Ear-protection","Headphones","Ear Protectors"],
    "glasses":["Glass","goggles","Glasses"],
    "gloves":["Glove","Safety_Gloves","Gloves"],
    "helmet":["Helmet","helmet","Hardhat"],
    "mask":["Mask","Safety_Mask"],
    "human":["Person","person"],
    "vest":["Vest","vest","Safety_Vest"],
    "dump_truck":["Dumb_truck"],
    "wheel_loader":["Loader"],
    "excavator":["Excavator"],
    "road_roller":["Roller","roller"],
    "bulldozer":["Bull_dozer"]
}
    for standard, variations in categories.items(): # Reads an element from the dictionary
        if name in variations:
            return standard # Returns it's corresponding normalized name
    return name  

def CombineDatasets(dsets): # Dataset combination (CSV concatenation, class normalization, image gathering)
    # Create dir that will contain all the images from the combined datasets.If a new combination of datasets wants to be used, the DIR must be DELETED beforehand
    os.mkdir("C:/Users/vgarc/Desktop/TFG/DataSets/FinalDataset") 
    print("C:/Users/vgarc/Desktop/TFG/DataSets/FinalDataset directory created!")
    dst_dir="C:/Users/vgarc/Desktop/TFG/DataSets/FinalDataset"
    FinalDataFrame = pd.concat([ds.PPE_frame for ds in dsets], ignore_index=True) # Combined dataframe CSV file

    # Class normalization in the final CSV file and dataframe
    FinalDataFrame['class']=FinalDataFrame['class'].apply(normalize_class) 
    FinalDataFrame.to_csv("C:/Users/vgarc/Desktop/TFG/DataSets/FinalDataset/final_dataset_normalized.csv", index=False)
    print("Combined dataframe final_dataset_normalized.csv created")

    # Put all the images from the different datasets together in one same DIR
    for ds in dsets: # For every  dataset 
        for file in os.listdir(ds.root_dir): # For all the files in the dataset
            if file.endswith(".jpg"): # If it is a jpg file
                file_name=os.path.join(ds.root_dir,file) # Full path to image
                shutil.copy(file_name,dst_dir) # Copy it to final dataset directory
    
    # Turn the final combined dataset into the PPEsDataset class defined in CustomDat.py, in order to access class parameters and fucntions
    FinalDataset=PPEsDataset(csv_file="C:/Users/vgarc/Desktop/TFG/DataSets/FinalDataset/final_dataset_normalized.csv",
                          root_dir="C:/Users/vgarc/Desktop/TFG/DataSets/FinalDataset",
                          transform=transformResize)

    for key,values in FinalDataset.__countCategory__().items(): # Show category count in Final Dataset
        print(f"{key}: {values}")
           
#------------------------------------------------------------------------------------------------------------------------------------------------
# MAIN
dsets=[PPEDetectionTRAIN, PPE2TRAIN, TallerYOLOTRAIN, goglesssTRAIN, Heavy_EquipmentTRAIN, RavenLoaderTRAIN, check_ssTRAIN, backgroundpics] # Chosen datasets
CombineDatasets(dsets)