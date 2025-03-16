#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 12:48:08 2024

@author: elenamartineztorrijos
"""
import pandas as pd
import os
from glob import glob
import pydicom
import numpy as np
import cv2
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
    threshold_background = 5
    grid_color = (200, 200, 200)  # Light gray
    image_with_grid = image.copy()
    h, w = image.shape
    thickness = 3
    for i in range(1, 5):
        cv2.line(image_with_grid, (i * w // 5, 0), (i * w // 5, h), grid_color, thickness)
        cv2.line(image_with_grid, (0, i * h // 5), (w, i * h // 5), grid_color, thickness)
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
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    sum_ = K.sum(K.square(y_true), axis=[1, 2, 3]) + K.sum(K.square(y_pred), axis=[1, 2, 3])
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

# Path to the Excel file
excel_file = '/Users/elenamartineztorrijos/Desktop/TFG/Data List HSA.xlsx'

# Load the Excel file into a pandas DataFrame
df = pd.read_excel(excel_file)

# Define the base directory where your DICOM files are located
base_dicom_dir = '/Users/elenamartineztorrijos/Desktop/TFG/HSA AI Carlos III/'

# Create a DataFrame to store the results
results = []

def calculate_density(image, segmented_image):
    # Ensure the segmented_image is 2D for indexing
    segmented_image_2d = np.squeeze(segmented_image, axis=-1) if segmented_image.ndim == 3 else segmented_image
    # Resize segmented_image to match image dimensions
    segmented_image_resized = cv2.resize(segmented_image_2d, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    # Densidad (intensidad de píxel en HU) de la hemorragia
    hemorrhage_pixels = image[segmented_image_resized == 1]
    if hemorrhage_pixels.size == 0:
        return 0  
    average_density = np.mean(hemorrhage_pixels) 
    return average_density

# Iterate over all patients and DICOM series
for index, row in df.iterrows():
    patient_id = row['Patient']
    series_info = row['Series']
    
    if pd.isnull(series_info) or series_info == '-':  
        print(f"No series information for patient {patient_id}.")
        continue  

    series_parts = str(series_info).split('+')
    density_sum = 0
    density_count = 0
    
    for series_part in series_parts:
        series_id = series_part.strip()
        series_path = os.path.join(base_dicom_dir, patient_id, 'DICOM', 'ST00001', series_id)
        dicom_files = sorted(glob(os.path.join(series_path, '*')))
        dicom_files = [f for f in dicom_files if os.path.isfile(f)]
        
        print(f"Checking series path: {series_path}")

        if not dicom_files:
            print(f"No DICOM files found for series {series_id} in patient {patient_id}.")
            continue  
        
        ref_image = pydicom.dcmread(dicom_files[0])
        total_slices = len(dicom_files)
        percent_lower = 35  
        percent_upper = 20 
        start_slice = int(total_slices * (percent_lower / 100.0))
        end_slice = int(total_slices * (1 - percent_upper / 100.0))
        dicom_files = dicom_files[start_slice:end_slice]

        for f in dicom_files:
            ds = pydicom.dcmread(f)
            img_dcm = ds.pixel_array
            img_windowed = apply_window_level(ds, window_center=40, window_width=80)
            img_normalized = normalize_image(img_windowed)
            img_rgb = cv2.cvtColor(img_normalized, cv2.COLOR_GRAY2RGB)
            predicted_mask = predict_mask(model, img_rgb)
            threshold = 0.5
            mask = (predicted_mask > threshold).astype(np.float32)

            if np.any(mask):
                density_sum += calculate_density(img_dcm, mask)
                density_count += 1

    # Calculate average density for the patient
    if density_count > 0:
        average_density = density_sum / density_count
    else:
        average_density = 0

    # Adding the results to the list
    results.append({
        'Patient_ID': patient_id,
        'Average_Density_HU': average_density
    })

# Creating a DataFrame from the results list
result_df = pd.DataFrame(results)

# Saving the results to an Excel file
result_excel_file = '/Users/elenamartineztorrijos/Desktop/TFG/Average_Density_results2.xlsx'
result_df.to_excel(result_excel_file, index=False)
print(f"Results saved to {result_excel_file}")
