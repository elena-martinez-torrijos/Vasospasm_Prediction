#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 21:20:11 2024

@author: elenamartineztorrijos
"""
import cv2
import pydicom
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model

# Cargar el modelo
model = load_model('/Users/elenamartineztorrijos/Desktop/TFG/unet_model.h5')

# Ruta a la imagen DICOM
dcm_path = '/Users/elenamartineztorrijos/Desktop/TFG/HSA AI Carlos III/HSA 129/DICOM/ST00001/SE00002/IM00029'
ds = pydicom.dcmread(dcm_path)
img_dcm = ds.pixel_array


# Aplicar la escala de grises almacenada en el archivo DICOM si está disponible
if 'RescaleIntercept' in ds and 'RescaleSlope' in ds:
    img_dcm = img_dcm * ds.RescaleSlope + ds.RescaleIntercept


# Extraer Window Center y Window Width si están disponibles
if 'WindowCenter' in ds and 'WindowWidth' in ds:
    WL = ds.WindowCenter
    WW = ds.WindowWidth
    print(f'Window Level (Center): {WL}, Window Width: {WW}')
else:
    print('Window Level and Width are not defined in this DICOM.')


lower_limit = WL - WW / 2
upper_limit = WL + WW / 2
img_windowed = np.clip(img_dcm, lower_limit, upper_limit)

# Escala la imagen para el rango de visualización completo [0, 255]
img_windowed_display = ((img_windowed - lower_limit) / (upper_limit - lower_limit) * 255).astype('uint8')

# Si la imagen no es en color, convertirla a RGB
if img_windowed_display.ndim == 2:
    img_windowed_display = cv2.cvtColor(img_windowed_display, cv2.COLOR_GRAY2RGB)

# Redimensionar la imagen al tamaño de entrada del modelo
image_resized = cv2.resize(img_windowed_display, (256, 256)) / 255.0  # Normalizar los valores de los píxeles
image_expanded = np.expand_dims(image_resized, axis=0)  # Añadir una dimensión de lote

# Realizar la predicción
predicted_mask = model.predict(image_expanded)[0]

# Umbralizar la máscara de salida para obtener una segmentación binaria
threshold = 0.5
predicted_mask = (predicted_mask > threshold).astype(np.float32)

# Visualizar la imagen original y la máscara predicha
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(image_resized)  # Usar cmap='gray' si es una imagen de grises
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(predicted_mask.squeeze(), cmap='gray')
plt.title('Predicted Mask')
plt.axis('off')

plt.show()
