#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 19:35:33 2024

@author: elenamartineztorrijos
"""
import numpy as np
import pandas as pd
import pydicom
import cv2
import os
from glob import glob
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from sklearn.model_selection import train_test_split

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

# Path to the Excel file
excel_file = '/Users/elenamartineztorrijos/Desktop/TFG/Data List HSA.xlsx'

# Load the Excel file into a pandas DataFrame
df = pd.read_excel(excel_file)

# Define the base directory where your DICOM files are located
base_dicom_dir = '/Users/elenamartineztorrijos/Desktop/TFG/HSA AI Carlos III/'

# Create an empty list to store the results for each patient
results = []

# Iterating through each row of the DataFrame
for index, row in df.iterrows():
    patient_id = row['Patient']
    series_info = row['Series']
    
    if pd.isnull(series_info) or series_info == '-':  
        print(f"No series information for patient {patient_id}.")
        continue  

    series_parts = str(series_info).split('+')
    series_volumes = []

    for series_part in series_parts:
        series_id = series_part.strip()
        series_path = os.path.join(base_dicom_dir, patient_id, 'DICOM', 'ST00001', series_id)
        dicom_files = sorted(glob(os.path.join(series_path, '*')))
        dicom_files = [f for f in dicom_files if os.path.isfile(f)]

        if not dicom_files:
            print(f"No DICOM files found for series {series_id} in patient {patient_id}.")
            continue  
        
        ref_image = pydicom.dcmread(dicom_files[0])
        segmented_volume = np.zeros((ref_image.Rows, ref_image.Columns, len(dicom_files)), dtype=np.float32)
        total_slices = len(dicom_files)
        percent_lower = 33  
        percent_upper = 26  
        start_slice = int(total_slices * (percent_lower / 100.0))
        end_slice = int(total_slices * (1 - percent_upper / 100.0))
        dicom_files = dicom_files[start_slice:end_slice]

        for i, f in enumerate(dicom_files):
            ds = pydicom.dcmread(f)
            img_dcm = ds.pixel_array
            img_windowed = apply_window_level(ds, window_center=40, window_width=80)
            img_normalized = normalize_image(img_windowed)
            img_rgb = cv2.cvtColor(img_normalized, cv2.COLOR_GRAY2RGB)
            img_resized = cv2.resize(img_rgb, (256, 256))

        series_volumes.append(segmented_volume)

    if series_volumes:
        total_volume = np.concatenate(series_volumes, axis=2) if len(series_volumes) > 1 else series_volumes[0]
        total_slices_after_concat = total_volume.shape[2]
        print(f'Número total de slices para el paciente {patient_id}: {total_slices_after_concat}')



# Path to the Excel files
ivh_fisher_file = '/Users/elenamartineztorrijos/Desktop/TFG/Data List HSA.xlsx'
hemorrhage_volume_file = '/Users/elenamartineztorrijos/Desktop/TFG/hemorrhage_results.xlsx'

# Load the Excel files into pandas DataFrames
ivh_fisher_df = pd.read_excel(ivh_fisher_file)
hemorrhage_volume_df = pd.read_excel(hemorrhage_volume_file)

# Combine the DataFrames using 'Patient' as the key
combined_df = pd.merge(ivh_fisher_df, hemorrhage_volume_df, on='Patient')

# Verify the combined DataFrame
print(combined_df.head())

# Check unique values and distribution of the target variable
print(combined_df['Fisher Modified Scale (ICU initial)'].value_counts())

# Determine if the target variable is binary or multiclass
#num_classes = len(combined_df['Fisher Modified Scale (ICU initial)'].unique())

# Prepare input features and target variable
features = combined_df[['IVH', 'Fisher Modified Scale (ICU initial)', 'Total Volume (ml)', 
                        'Section (0, 0)', 'Section (0, 1)', 'Section (0, 2)', 'Section (0, 3)', 'Section (0, 4)',
                        'Section (1, 0)', 'Section (1, 1)', 'Section (1, 2)', 'Section (1, 3)', 'Section (1, 4)',
                        'Section (2, 0)', 'Section (2, 1)', 'Section (2, 2)', 'Section (2, 3)', 'Section (2, 4)',
                        'Section (3, 0)', 'Section (3, 1)', 'Section (3, 2)', 'Section (3, 3)', 'Section (3, 4)',
                        'Section (4, 0)', 'Section (4, 1)', 'Section (4, 2)', 'Section (4, 3)', 'Section (4, 4)']]

target = combined_df['Fisher Modified Scale (ICU initial)']

non_numeric_columns = features.select_dtypes(exclude=['float64', 'int64']).columns
print(non_numeric_columns)

features = features.dropna(subset=['Fisher Modified Scale (ICU initial)'])

# Mapear valores de 'Fisher Modified Scale (ICU initial)' a números
fisher_mapping = {'1': 1, '2': 2, '3': 3, '4': 4}
features.loc[:, 'Fisher Modified Scale (ICU initial)'] = features['Fisher Modified Scale (ICU initial)'].map(fisher_mapping)

print(features['Fisher Modified Scale (ICU initial)'].unique())


# Convert to numpy arrays
X = features.values
y = target.values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.dtype)
print(X_test.dtype)

X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)


num_classes = 4 

# Define the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=SHAPE))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
#print(classification_report(y_test, y_pred_classes))