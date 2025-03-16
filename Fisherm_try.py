#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 12:33:50 2024

@author: elenamartineztorrijos
"""
import pydicom
import numpy as np
import os
from glob import glob
import cv2
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

def apply_window(image, ds):
    if 'WindowCenter' in ds and 'WindowWidth' in ds:
        window_center = ds.WindowCenter
        window_width = ds.WindowWidth
        window_center = window_center[0] if isinstance(window_center, pydicom.multival.MultiValue) else window_center
        window_width = window_width[0] if isinstance(window_width, pydicom.multival.MultiValue) else window_width
    else:
        print('Window Level and Width are not defined in this DICOM.')
        return image

    lower_limit = window_center - window_width / 2
    upper_limit = window_center + window_width / 2
    image = np.clip(image, lower_limit, upper_limit)
    return ((image - lower_limit) / (upper_limit - lower_limit) * 255).astype('uint8')

dicom_dir = '/Users/elenamartineztorrijos/Desktop/TFG/HSA AI Carlos III/HSA 129/DICOM/ST00001/SE00002/'
dicom_files = sorted(glob(os.path.join(dicom_dir, '*')))
dicom_files = [f for f in dicom_files if os.path.isfile(f)]
if not dicom_files:
    raise ValueError("No DICOM files found in the specified directory.")

image_volume = []
for f in dicom_files:
    ds = pydicom.dcmread(f)
    img_dcm = ds.pixel_array
    img_windowed = apply_window(img_dcm, ds)
    image_volume.append(img_windowed)
image_volume = np.stack(image_volume, axis=-1)

total_slices = image_volume.shape[2]
percent_lower = 33  # Percentage of slices to exclude at the start
percent_upper = 26  # Percentage of slices to exclude at the end

# Calculating the actual slice indices to use
start_slice = int(total_slices * (percent_lower / 100.0))
end_slice = int(total_slices * (1 - percent_upper / 100.0))

selected_slices = image_volume[:, :, start_slice:end_slice]

model_path = '/Users/elenamartineztorrijos/Desktop/TFG/unet_model.h5'
model = load_model(model_path)

segmented_volume = np.zeros((selected_slices.shape[0], selected_slices.shape[1], selected_slices.shape[2]), dtype=np.float32)

for i in range(selected_slices.shape[2]):
    img_normalized = cv2.cvtColor(selected_slices[:, :, i], cv2.COLOR_GRAY2RGB) / 255.0
    img_resized = cv2.resize(img_normalized, (256, 256))
    img_expanded = np.expand_dims(img_resized, axis=0)
    predicted_mask = model.predict(img_expanded)[0]
    threshold = 0.5
    predicted_mask = (predicted_mask > threshold).astype(np.float32)
    segmented_volume[:, :, i] = cv2.resize(predicted_mask.squeeze(), (selected_slices.shape[0], selected_slices.shape[1]))

slice_thickness = float(ds.SliceThickness)
pixel_spacing = ds.PixelSpacing
voxel_volume = slice_thickness * pixel_spacing[0] * pixel_spacing[1]
num_segmented_pixels = np.sum(segmented_volume)
hemorrhage_volume_mm3 = num_segmented_pixels * voxel_volume
hemorrhage_volume_ml = hemorrhage_volume_mm3 / 1000

print(f'Total volume approximation: {hemorrhage_volume_mm3} mm³ or {hemorrhage_volume_ml} ml.')

def classify_fisher_scale(volume_ml, segmented_vol):
    if volume_ml == 0:
        return "Grade 0: No blood detected"
    
    central_bleeding = np.sum(segmented_vol[segmented_vol.shape[0]//4:3*segmented_vol.shape[0]//4, segmented_vol.shape[1]//4:3*segmented_vol.shape[1]//4, :] > 0)
    
    if central_bleeding == 0:
        return "Grade 1: Thin, diffuse deposition of subarachnoid blood, less than 1 mm thick"
    
    if central_bleeding > 0 and volume_ml < 1:
        return "Grade 2: Cisternal blood without intraventricular blood, and layers 1 mm or greater"
    
    if central_bleeding > 0 and volume_ml >= 1:
        return "Grade 3: Cisternal blood with intraventricular blood, regardless of thickness"
    
    return "Grade 4: Diffuse or no subarachnoid blood, but with intracerebral or intraventricular clots"

fisher_grade = classify_fisher_scale(hemorrhage_volume_ml, segmented_volume)
print(f'Assigned Fisher Scale: {fisher_grade}')


num_to_display = 10  # Change this number to display more or fewer slices
fig, axs = plt.subplots(2, num_to_display, figsize=(20, 5))

# Adjusting the slice selection mechanism to avoid out-of-bounds error
slice_indices = np.linspace(0, selected_slices.shape[2] - 1, num_to_display, dtype=int)

for n, slice_idx in enumerate(slice_indices):
    img = selected_slices[:, :, slice_idx]
    mask = segmented_volume[:, :, slice_idx]
    axs[0, n].imshow(img, cmap='gray')
    axs[0, n].axis('off')
    axs[0, n].set_title(f'Slice {start_slice + slice_idx}')
    axs[1, n].imshow(mask, cmap='gray')
    axs[1, n].axis('off')
    axs[1, n].set_title(f'Mask {start_slice + slice_idx}')

plt.tight_layout()
plt.show()


