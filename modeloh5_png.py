#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 20:14:14 2024

@author: elenamartineztorrijos
"""

from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Cargar el modelo
model = load_model('/Users/elenamartineztorrijos/Desktop/unet_model.h5')

# Preparar la imagen (asegúrate de usar el mismo preprocesamiento que usaste durante el entrenamiento)
image_path = '/Users/elenamartineztorrijos/Desktop/dataset/images/090.png'
image = cv2.imread(image_path, cv2.IMREAD_COLOR)
image_resized = cv2.resize(image, (256, 256)) / 255.0  # Normalizar los valores de los píxeles si fue parte del preprocesamiento
image_expanded = np.expand_dims(image_resized, axis=0)  # Añadir una dimensión de lote

# Realizar la predicción
predicted_mask = model.predict(image_expanded)[0]

# Umbralizar la máscara de salida para obtener una segmentación binaria
threshold = 0.5
predicted_mask = (predicted_mask > threshold).astype(np.float32)

if image_resized.dtype == np.float64:
    image_resized = image_resized.astype(np.float32)

# Si los valores están en el rango 0-1, escalarlos al rango 0-255 y convertir a uint8
if image_resized.max() <= 1.0:
    image_resized = (255 * image_resized).astype(np.uint8)

image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)

# Visualizar la imagen original y la máscara predicha
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(predicted_mask.squeeze(), cmap='gray')
plt.title('Predicted Mask')
plt.axis('off')

plt.show()
