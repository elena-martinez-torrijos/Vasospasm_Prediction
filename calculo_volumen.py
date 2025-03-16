#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 10:05:47 2024

@author: elenamartineztorrijos
"""
import pydicom
import numpy as np
import os
from glob import glob
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam as LegacyAdam
from tensorflow.keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt

# Aplicar windowing y normalizar imágenes
def apply_window(image, ds):
    if 'WindowCenter' in ds and 'WindowWidth' in ds:
        window_center = ds.WindowCenter
        window_width = ds.WindowWidth
        # En caso de que estos sean multivaluados, tomar el primer valor
        window_center = window_center[0] if isinstance(window_center, pydicom.multival.MultiValue) else window_center
        window_width = window_width[0] if isinstance(window_width, pydicom.multival.MultiValue) else window_width
    else:
        print('Window Level and Width are not defined in this DICOM.')
        return image # Devuelve la imagen sin windowing si no hay metadatos definidos
    
    lower_limit = window_center - window_width / 2
    upper_limit = window_center + window_width / 2
    image = np.clip(image, lower_limit, upper_limit)
    return ((image - lower_limit) / (upper_limit - lower_limit) * 255).astype('uint8')

# Directorio donde se encuentran tus archivos DICOM
dicom_dir = '/Users/elenamartineztorrijos/Desktop/TFG/HSA AI Carlos III/HSA 129/DICOM/ST00001/SE00002/'

# Lista todos los archivos DICOM
dicom_files = sorted(glob(os.path.join(dicom_dir, '*')))
dicom_files = [f for f in dicom_files if os.path.isfile(f)]
if not dicom_files:
    raise ValueError("No DICOM files found in the specified directory.")

# Crear un array 3D numpy para almacenar las imágenes procesadas
image_volume = []
for f in dicom_files:
    ds = pydicom.dcmread(f)
    img_dcm = ds.pixel_array
    img_windowed = apply_window(img_dcm, ds)
    image_volume.append(img_windowed)
image_volume = np.stack(image_volume, axis=-1)

total_slices = image_volume.shape[2]
percent_lower = 0  # Percentage of slices to exclude at the start
percent_upper = 0 # Percentage of slices to exclude at the end

# Calculating the actual slice indices to use
start_slice = int(total_slices * (percent_lower / 100.0))
end_slice = int(total_slices * (1 - percent_upper / 100.0))

selected_slices = image_volume[:, :, start_slice:end_slice]

# Define your custom loss function here
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    return bce(y_true, y_pred) + (1 - dice_coef(y_true, y_pred))

def iou(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    sum_ = K.sum(K.square(y_true), axis=[1,2,3]) + K.sum(K.square(y_pred), axis=[1,2,3])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)

# Load the trained model, using the legacy optimizer for M1/M2 Macs
model_path = '/Users/elenamartineztorrijos/Desktop/TFG/Seg_Model_definitivo/segmentation_model_v1.h5'

model = load_model(
    model_path,
    custom_objects={
        "Adam": LegacyAdam,
        "bce_dice_loss": bce_dice_loss,
        "iou": iou,
        "dice_coef": dice_coef
    }
)
# Preparar el volumen para segmentación
ref_image = pydicom.dcmread(dicom_files[0])
segmented_volume = np.zeros((ref_image.Rows, ref_image.Columns, selected_slices.shape[2]), dtype=np.float32)

# Procesar cada slice del volumen
for i in range(selected_slices.shape[2]):
    img_normalized = cv2.cvtColor(selected_slices[:, :, i], cv2.COLOR_GRAY2RGB) / 255.0
    img_resized = cv2.resize(img_normalized, (256, 256))
    img_expanded = np.expand_dims(img_resized, axis=0)
    predicted_mask = model.predict(img_expanded)[0]
    threshold = 0.5
    predicted_mask = (predicted_mask > threshold).astype(np.float32)
    segmented_volume[:, :, i] = cv2.resize(predicted_mask.squeeze(), (ref_image.Rows, ref_image.Columns))

# Mostrar cada décima slice
num_to_display = 10  # Cambia este número para mostrar más o menos slices
fig, axs = plt.subplots(2, num_to_display, figsize=(20, 5))

for n in range(num_to_display):
    slice_idx = n * (selected_slices.shape[2] // num_to_display)
    img = selected_slices[:, :, slice_idx]
    mask = segmented_volume[:, :, slice_idx]
    axs[0, n].imshow(img, cmap='gray')
    axs[0, n].axis('off')
    axs[0, n].set_title(f'Slice {slice_idx}')
    axs[1, n].imshow(mask, cmap='gray')
    axs[1, n].axis('off')
    axs[1, n].set_title(f'Mask {slice_idx}')

plt.tight_layout()
plt.show()

# Obtener los metadatos del archivo DICOM
slice_thickness = float(ref_image.SliceThickness)
pixel_spacing = ref_image.PixelSpacing
voxel_volume = slice_thickness * pixel_spacing[0] * pixel_spacing[1]  # Volumen por vóxel en mm³

# Contar el número de píxeles segmentados (valor 1 en la máscara binaria)
num_segmented_pixels = np.sum(segmented_volume)

# Calcular el volumen total de la hemorragia
hemorrhage_volume_mm3 = num_segmented_pixels * voxel_volume
hemorrhage_volume_ml = hemorrhage_volume_mm3 / 1000  # Convertir a mililitros (1 ml = 1 cm³ = 1000 mm³)

print(f'Total volume approximation: {hemorrhage_volume_mm3} mm³ or {hemorrhage_volume_ml} ml.')
