#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 13:53:03 2024

@author: elenamartineztorrijos
"""
import pandas as pd
import os
from glob import glob
import pydicom
import numpy as np
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

def draw_grid(image):
    # Suponer que el fondo es negro o casi negro
    threshold_background = 5
    
    # Dibujar una cuadrícula sobre toda la imagen
    grid_color = (200, 200, 200) # Gris claro
    image_with_grid = image.copy()
    
    # Tamaño de la imagen
    h, w = image.shape
    
    # Dibujar líneas verticales y horizontales para crear la cuadrícula
    thickness = 3
    for i in range(1, 5):
        cv2.line(image_with_grid, (i * w // 5, 0), (i * w // 5, h), grid_color, thickness)
        cv2.line(image_with_grid, (0, i * h // 5), (w, i * h // 5), grid_color, thickness)

    # Ahora borrar la cuadrícula donde la imagen es negra (el fondo)
    mask_background = image < threshold_background
    image_with_grid[mask_background] = image[mask_background]
    
    return image_with_grid

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

# Ruta al archivo de Excel
excel_file = '/Users/elenamartineztorrijos/Desktop/TFG/Data List HSA.xlsx'

# Cargar el archivo de Excel en un DataFrame de pandas
df = pd.read_excel(excel_file)

# Definir el directorio base donde se encuentran tus archivos DICOM
base_dicom_dir = '/Users/elenamartineztorrijos/Desktop/TFG/HSA AI Carlos III/'

# Iterar a través de cada fila del DataFrame
for index, row in df.iterrows():
    patient_id = row['Patient']
    series_info = row['Series']
    
    if pd.isnull(series_info) or series_info == '-':  # Si no hay información de la serie
        print(f"No series information for patient {patient_id}.")
        continue  # Salta al siguiente paciente

    # Asumiendo que la columna 'Series' contiene cadenas de texto
    series_parts = str(series_info).split('+')
    series_volumes = []

    for series_part in series_parts:
        series_id = series_part.strip()  # Eliminar espacios en blanco
        
        # Construir la ruta completa para cada serie de imágenes
        series_path = os.path.join(base_dicom_dir, patient_id, 'DICOM', 'ST00001', series_id)
        
        # Leer archivos DICOM
        dicom_files = sorted(glob(os.path.join(series_path, '*')))
        dicom_files = [f for f in dicom_files if os.path.isfile(f)]

        if not dicom_files:
            print(f"No DICOM files found for series {series_id} in patient {patient_id}.")
            continue  # Salta a la siguiente parte de la serie
        
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

        series_volumes.append(segmented_volume)

    # Ahora fuera del bucle, manejar la concatenación de los volúmenes si es necesario
    if series_volumes:
        total_volume = np.concatenate(series_volumes, axis=2) if len(series_volumes) > 1 else series_volumes[0]

        # Visualizar el número total de slices
        total_slices_after_concat = total_volume.shape[2]
        print(f'Número total de slices para el paciente {patient_id}: {total_slices_after_concat}')


        # Obtener los metadatos del archivo DICOM
        slice_thickness = float(ref_image.SliceThickness)
        pixel_spacing = ref_image.PixelSpacing
        voxel_volume = slice_thickness * pixel_spacing[0] * pixel_spacing[1]  # Volumen por vóxel en mm³

        # Contar el número de píxeles segmentados (valor 1 en la máscara binaria)
        num_segmented_pixels = np.sum(total_volume)

        # Calcular el volumen total de la hemorragia
        hemorrhage_volume_mm3 = num_segmented_pixels * voxel_volume
        hemorrhage_volume_ml = hemorrhage_volume_mm3 / 1000  # Convertir a mililitros (1 ml = 1 cm³ = 1000 mm³)

        print(f'Total volume approximation for patient {patient_id}: {hemorrhage_volume_mm3} mm³ or {hemorrhage_volume_ml} ml.')

        # Calcular el porcentaje de hemorragia en cada sección de la cuadrícula
        grid_size = 5
        hemorrhage_percentages = np.zeros((grid_size, grid_size))

        for i in range(grid_size):
            for j in range(grid_size):
                start_row = i * ref_image.Rows // grid_size
                end_row = (i + 1) * ref_image.Rows // grid_size
                start_col = j * ref_image.Columns // grid_size
                end_col = (j + 1) * ref_image.Columns // grid_size

                section_mask = segmented_volume[start_row:end_row, start_col:end_col, :]
                hemorrhage_pixels_section = np.sum(section_mask)

                if num_segmented_pixels > 0:  # Prevenir división por cero
                    hemorrhage_percentages[i, j] = (hemorrhage_pixels_section / num_segmented_pixels) * 100
                else:
                    hemorrhage_percentages[i, j] = 0

        # Imprimir los porcentajes de hemorragia por sección
        for i in range(grid_size):
            for j in range(grid_size):
                print(f'Section ({i}, {j}): {hemorrhage_percentages[i, j]:.2f}% of total hemorrhage')

        