import os

# Script para generar annotations para  dataset de background. No tendrá bounding boxes, 
# porque un fondo vacío no contiene objetos en sí
path="C:/Users/vgarc/Desktop/TFG/DataSets/background" # Ruta a donde están las imágenes
imgs=os.listdir(path) #Recoge todas las imágenes
f=open("C:/Users/vgarc/Desktop/TFG/DataSets/background/_annotations.csv","w") # Abrir archivo CSV de annotations
i=0
for I in imgs: # Recorre todas las imágenes
    if i==0: #i=0
        i=1
        f.write(",".join(["filename","width","height","class","xmin","ymin","xmax","ymax"]) + "\n")

    else: # i=1
        f.write(f"{I},640,640,background,0,0,0,0\n") # Escribir etiquetas para cada imagen




