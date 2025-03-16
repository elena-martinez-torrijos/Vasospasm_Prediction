#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 19:08:13 2024

@author: elenamartineztorrijos
"""
import pydicom
import numpy as np
import os
from glob import glob
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam as LegacyAdam
from tensorflow.keras import backend as K
import tensorflow as tf

# Constants for the image size and channels
HEIGHT = 256
WIDTH = 256
CHANNELS = 3  # Set to 3 for RGB images
SHAPE = (HEIGHT, WIDTH, CHANNELS)

# DICOM WINDOWING AND DATA NORMALIZATION
def correct_dcm(dcm):
    x = dcm.pixel_array + 1000
    px_mode = 4096
    x[x >= px_mode] = x[x >= px_mode] - px_mode
    dcm.PixelData = x.tobytes()
    dcm.RescaleIntercept = -1000

def window_image(img, window_center, window_width):
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    windowed_img = np.clip(img, img_min, img_max)
    return windowed_img

def normalize_image(image):
    image = cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
    image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return image

def apply_window_level(dcm, window_center, window_width):
    img = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
    windowed_img = window_image(img, window_center, window_width)
    return windowed_img

# Model prediction
def predict_mask(model, image):
    image_expanded = np.expand_dims(image, axis=0)  # Add batch dimension
    predicted_mask = model.predict(image_expanded)[0]  # Predict and take the first batch element
    return predicted_mask

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

# Directorio donde se encuentran tus archivos DICOM
dicom_dir = '/Users/elenamartineztorrijos/Desktop/TFG/HSA AI Carlos III/HSA 129/DICOM/ST00001/SE00002/'

# Lista todos los archivos DICOM
dicom_files = sorted(glob(os.path.join(dicom_dir, '*')))
dicom_files = [f for f in dicom_files if os.path.isfile(f)]
if not dicom_files:
    raise ValueError("No DICOM files found in the specified directory.")

# Preparar el volumen para segmentación
ref_image = pydicom.dcmread(dicom_files[0])
segmented_volume = np.zeros((ref_image.Rows, ref_image.Columns, len(dicom_files)), dtype=np.float32)

# Aplicar la selección de rebanadas
total_slices = len(dicom_files)
percent_lower = 33  # Porcentaje de rebanadas a excluir al principio
percent_upper = 26  # Porcentaje de rebanadas a excluir al final
start_slice = int(total_slices * (percent_lower / 100.0))
end_slice = int(total_slices * (1 - percent_upper / 100.0))
dicom_files = dicom_files[start_slice:end_slice]

# Procesar cada slice del volumen
for i, f in enumerate(dicom_files):
    ds = pydicom.dcmread(f)
    img_dcm = ds.pixel_array
    img_windowed = apply_window_level(ds, window_center=40, window_width=80)
    img_normalized = normalize_image(img_windowed)
    img_rgb = cv2.cvtColor(img_normalized, cv2.COLOR_GRAY2RGB)
    img_resized = cv2.resize(img_rgb, (256, 256))
    predicted_mask = predict_mask(model, img_resized)
    threshold = 0.5
    predicted_mask = (predicted_mask > threshold).astype(np.float32)
    segmented_volume[:, :, i] = cv2.resize(predicted_mask.squeeze(), (ref_image.Rows, ref_image.Columns))

# Mostrar cada décima slice
num_to_display = 10  # Cambia este número para mostrar más o menos slices
fig, axs = plt.subplots(2, num_to_display, figsize=(20, 5))

for n in range(num_to_display):
    slice_idx = n * (len(dicom_files) // num_to_display)
    img = apply_window_level(pydicom.dcmread(dicom_files[slice_idx]), window_center=40, window_width=80)
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
