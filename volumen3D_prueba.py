#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 23:00:05 2024

@author: elenamartineztorrijos
"""
import pydicom
import numpy as np
import os
from glob import glob

# Directorio donde se encuentran tus archivos DICOM
dicom_dir = '/Users/elenamartineztorrijos/Desktop/TFG/HSA AI Carlos III/HSA 80/DICOM/ST00001/SE00004'

# Lista todos los archivos en el directorio, asumiendo que todos son archivos DICOM
dicom_files = sorted(glob(os.path.join(dicom_dir, '*')))

# Filtra para asegurarse de que son archivos y no directorios
dicom_files = [f for f in dicom_files if os.path.isfile(f)]
if not dicom_files:
    raise ValueError("No DICOM files found in the specified directory.")

# Leer el primer archivo para obtener las dimensiones
ref_image = pydicom.dcmread(dicom_files[0])

# Asignar el número de slices basado en la cantidad de archivos DICOM
num_slices = len(dicom_files)

# Crear un array 3D numpy para almacenar las imágenes
image_volume = np.zeros((ref_image.Rows, ref_image.Columns, num_slices), dtype=ref_image.pixel_array.dtype)

# Loop para leer cada archivo DICOM y almacenar las imágenes en el array numpy
for i, dicom_file in enumerate(dicom_files):
    dicom_data = pydicom.dcmread(dicom_file)
    image_volume[:, :, i] = dicom_data.pixel_array

print("Volumen 3D creado con éxito.")

