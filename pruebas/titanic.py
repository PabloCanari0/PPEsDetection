import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Titanic survivor predictor 
# 1. First step, get and clean the data
df=pd.read_csv("C:/Users/vgarc/Desktop/TFG/titanic/train.csv") # Read the data spreadsheet
df.isna().sum() # Locate which columns have Nan values, we will replace them with the mode of each column
df.mode().iloc[0] # Calculates the mode of each column, if there's a draw, get the first value