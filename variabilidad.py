#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 15:03:02 2024

@author: elenamartineztorrijos
"""
import pandas as pd
import numpy as np

# Cargar el archivo Excel en un DataFrame de pandas
archivo_excel = "/Users/elenamartineztorrijos/Desktop/neuroradiologist_classification.xlsx"
df = pd.read_excel(archivo_excel)

# Seleccionar las columnas relevantes
columnas_seleccionadas = ['Clase (neuroradiólogo 1)', 'Clase (neuroradiólogo 2)', 'Predicted Class']
df_seleccionado = df[columnas_seleccionadas]

# Crear la matriz de confusión
def confusion_matrix_manual(y_true, y_pred, num_classes):
    matrix = np.zeros((num_classes, num_classes))
    for t, p in zip(y_true, y_pred):
        matrix[t][p] += 1
    return matrix

# Número de clases (asumiendo que las clases son 0, 1, 2, ..., num_classes-1)
num_classes = len(df_seleccionado['Clase (neuroradiólogo 1)'].unique())

# Crear la matriz de confusión
conf_matrix_1_2 = confusion_matrix_manual(df_seleccionado.iloc[:, 0], df_seleccionado.iloc[:, 1], num_classes)
conf_matrix_1_pred = confusion_matrix_manual(df_seleccionado.iloc[:, 0], df_seleccionado.iloc[:, 2], num_classes)
conf_matrix_2_pred = confusion_matrix_manual(df_seleccionado.iloc[:, 1], df_seleccionado.iloc[:, 2], num_classes)

# Crear la matriz de ponderación
weight_matrix = np.zeros((num_classes, num_classes))
for i in range(num_classes):
    for j in range(num_classes):
        weight_matrix[i][j] = abs(i - j)

# Función para calcular el Kappa ponderado manualmente
def weighted_kappa_manual(conf_matrix, weight_matrix):
    total = np.sum(conf_matrix)
    expected_matrix = np.outer(np.sum(conf_matrix, axis=1), np.sum(conf_matrix, axis=0)) / total
    weighted_matrix = weight_matrix / np.max(weight_matrix)
    observed_score = np.sum(conf_matrix * (1 - weighted_matrix))
    expected_score = np.sum(expected_matrix * (1 - weighted_matrix))
    kappa = (observed_score - expected_score) / (total - expected_score)
    return kappa

# Calcular el Kappa ponderado
weighted_kappa_1_2 = weighted_kappa_manual(conf_matrix_1_2, weight_matrix)
weighted_kappa_1_pred = weighted_kappa_manual(conf_matrix_1_pred, weight_matrix)
weighted_kappa_2_pred = weighted_kappa_manual(conf_matrix_2_pred, weight_matrix)

# Resultados
print("Kappa ponderado entre Clase (neuroradiólogo 1) y Clase (neuroradiólogo 2):", weighted_kappa_1_2)
print("Kappa ponderado entre Clase (neuroradiólogo 1) y Predicted Class:", weighted_kappa_1_pred)
print("Kappa ponderado entre Clase (neuroradiólogo 2) y Predicted Class:", weighted_kappa_2_pred)
