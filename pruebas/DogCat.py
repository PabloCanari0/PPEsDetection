#!/usr/bin/env python
import torch
import fastai
from fastai.imports import *
import matplotlib.pyplot as plt

path = untar_data(URLs.PETS)/'images'  # Descarga y descomprime las imágenes

def isCat(x): return x[0].isupper()  # Archivos de gatos empiezan con mayúscula

dls = ImageDataLoaders.from_name_func(
    path,  # ← Se debe usar `path`, no `'.'`
    get_image_files(path),
    valid_pct=0.2,
    seed=42,
    label_func=isCat,
    item_tfms=Resize(192),
    num_workers=0  
)
dls.show_batch()  # Muestra un batch de imágenes con sus etiquetas
plt.show()
#learn = vision_learner(dls, arch=resnet18, metrics=error_rate)  # Modelo resnet18
#learn.fine_tune(3)  # Entrenamiento con 3 épocas
#learner.export('model.pkl')  # Guarda el modelo entrenado
#print("Model saved at : ",learn.path) #Ruta del modelo entrenado

