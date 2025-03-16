#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 22:01:10 2024

@author: elenamartineztorrijos
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm

def prepare_data():
    # Cargar datos
    df_hsa = pd.read_excel('/Users/elenamartineztorrijos/Desktop/TFG/Data List HSA.xlsx', usecols=['Fisher Modified Scale  (neuroradiologist 1)'])
    df_hemorrhage = pd.read_excel('/Users/elenamartineztorrijos/Desktop/TFG/Results_model_seg.xlsx', usecols=['Total Volume (ml)'])
    
    # Encontrar y filtrar por pacientes comunes
    df_combined = pd.concat([df_hsa, df_hemorrhage], axis=1)
    
    # Convertir Fisher Scale a numérico
    df_combined['Fisher Modified Scale  (neuroradiologist 1)'] = pd.to_numeric(df_combined['Fisher Modified Scale  (neuroradiologist 1)'], errors='coerce')

    # Llenar los valores NaN con la media de las columnas respectivas
    df_combined.fillna(df_combined.mean(), inplace=True)

    return df_combined


def plot_regression(df):
    x = df['Total Volume (ml)']
    y = df['Fisher Modified Scale  (neuroradiologist 1)'].astype(int)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    y_pred = intercept + slope * x

    plt.figure(figsize=(10, 5))
    plt.scatter(x, y, color='blue', label='Datos reales')
    plt.plot(x, y_pred, color='red', label=f'Línea de regresión y={intercept:.2f}+{slope:.2f}*x\nR^2={r_value**2:.2f}')
    plt.title('Relación entre Volumen Total y Escala de Fisher del Neuroradiólogo')
    plt.xlabel('Total Volume (ml)')
    plt.ylabel('Fisher Modified Scale (neuroradiologist 1)')
    plt.legend()
    plt.show()

df_combined = prepare_data()
plot_regression(df_combined)
