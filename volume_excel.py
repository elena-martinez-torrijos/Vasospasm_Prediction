#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 12:25:50 2024

@author: elenamartineztorrijos
"""
import pandas as pd
import os
from glob import glob
import pydicom
import numpy as np

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
        
        # Leer el primer archivo para obtener las dimensiones
        ref_image = pydicom.dcmread(dicom_files[0])
        num_slices = len(dicom_files)
        image_volume = np.zeros((ref_image.Rows, ref_image.Columns, num_slices), dtype=ref_image.pixel_array.dtype)

        for i, dicom_file in enumerate(dicom_files):
            dicom_data = pydicom.dcmread(dicom_file)
            image_volume[:, :, i] = dicom_data.pixel_array
        
        series_volumes.append(image_volume)

    # Fuera del bucle manejar la concatenación de los volúmenes si es necesario
    if series_volumes:
        if len(series_volumes) > 1:
            # Concatenar volúmenes
            total_volume = np.concatenate(series_volumes, axis=2)
        else:
            # Solo hay un volumen
            total_volume = series_volumes[0]

        print(f"Volumen 3D creado con éxito para el paciente {patient_id}.")
    