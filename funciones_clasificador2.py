#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 18:07:44 2024

@author: elenamartineztorrijos
"""

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

def prepare_data():
    '''
    Returns
    -------
    df_combined
    X
    Y
    '''
    
    # Load necessary columns directly from the specified Excel files
    #df_hsa = pd.read_excel('/Users/elenamartineztorrijos/Desktop/TFG/Data List HSA.xlsx', usecols=['Patient', 'IVH', 'Bleed', 'Supraselar cistern bleed', 'Fisher Modified Scale  (neuroradiologist 1)'], sheet_name='Hoja3')
    #df_hsa = pd.read_excel('/Users/elenamartineztorrijos/Desktop/TFG/Data List HSA.xlsx', usecols=['Patient','Fisher Modified Scale  (neuroradiologist 1)','AnyVasospasm','Edad','Sexo'], sheet_name='Hoja4')
    df_hsa = pd.read_excel('/Users/elenamartineztorrijos/Desktop/TFG/Data List HSA.xlsx', usecols=['Patient','IVH', 'Fisher Modified Scale  (neuroradiologist 1)','AnyVasospasm','Edad'], sheet_name='Hoja4')
    
    #df_hemorrhage = pd.read_excel('/Users/elenamartineztorrijos/Desktop/TFG/Results_model_seg.xlsx', usecols=['Patient', 'Total Volume (ml)', 'Section (0, 2)', 'Section (1, 2)',  'Section (2, 2)', 'Section (3, 2)', 'Section (4, 2)'])
    df_hemorrhage = pd.read_excel('/Users/elenamartineztorrijos/Desktop/TFG/Results_model_seg.xlsx', usecols=['Patient', 'Total Volume (ml)'])
    
    # Load the Max_Slice_Thickness data
    #df_thickness = pd.read_excel('/Users/elenamartineztorrijos/Desktop/TFG/Max_Slice_Thickness_results.xlsx', usecols=['Patient', 'Max_Slice_Thickness'])
    df_thickness = pd.read_excel('/Users/elenamartineztorrijos/Desktop/TFG/Max_Slice_Thickness_results.xlsx', usecols=['Patient'])

    # Load the Average_Density data
    #df_density = pd.read_excel('/Users/elenamartineztorrijos/Desktop/neuroradiologist_classification.xlsx', usecols=['Patient', 'Predicted Class'])

    # Find common patients across all datasets
    common_patients = set(df_hsa['Patient']) & set(df_hemorrhage['Patient'])

    # Filter data for common patients
    df_hsa_common = df_hsa[df_hsa['Patient'].isin(common_patients)]
    df_hemorrhage_common = df_hemorrhage[df_hemorrhage['Patient'].isin(common_patients)]
    df_thickness_common = df_thickness[df_thickness['Patient'].isin(common_patients)]
    #df_density_common = df_density[df_density['Patient'].isin(common_patients)]

    # Combine data by merging on 'Patient'
    df_combined = pd.merge(df_hsa_common, df_hemorrhage_common, on='Patient')
    df_combined = pd.merge(df_combined, df_thickness_common, on='Patient')
    #df_combined = pd.merge(df_combined, df_density_common, on='Patient')
    # Find common patients across all datasets
    common_patients = set(df_hsa['Patient']) & set(df_hemorrhage['Patient'])# & set(df_thickness['Patient'])

    # Filter data for common patients
    #df_hsa_common = df_hsa[df_hsa['Patient'].isin(common_patients)]
    #df_hemorrhage_common = df_hemorrhage[df_hemorrhage['Patient'].isin(common_patients)]
    #df_thickness_common = df_thickness[df_thickness['Patient'].isin(common_patients)]

    # Combine data by merging on 'Patient'
    #df_combined = pd.merge(df_hsa_common, df_hemorrhage_common, on='Patient')
    #df_combined = pd.merge(df_combined, df_thickness_common, on='Patient')

    # Prepare X and Y datasets
    X = df_combined.drop(['Patient', 'AnyVasospasm'], axis=1)
    Y = df_combined['AnyVasospasm'].astype(int)
    
    return df_combined, X, Y

df_combined, X, Y = prepare_data()
print(df_combined)

def get_plot_balanced_classes(data, balanced_data):
        
     
     # Datos antes del balanceo
     plt.figure(figsize=(12, 6))
    
     plt.subplot(1, 2, 1)  # Primer subgráfico para el histograma antes del balanceo
     plt.hist(data, bins=range(min(data), max(data) + 2), align='left', color='blue', alpha=0.7)
     plt.title('Class Distribution Before SMOTE')
     plt.xlabel('Class Label')
     plt.ylabel('Frequency')
     plt.xticks(range(min(data), max(data) + 1))  # Asegura que se muestren todas las etiquetas de clase
    
     # Datos después del balanceo
     plt.subplot(1, 2, 2)  # Segundo subgráfico para el histograma después del balanceo
     plt.hist(balanced_data, bins=range(min(balanced_data), max(balanced_data) + 2), align='left', color='green', alpha=0.7)
     plt.title('Class Distribution After SMOTE')
     plt.xlabel('Class Label')
     plt.ylabel('Frequency')
     plt.xticks(range(min(balanced_data), max(balanced_data) + 1))  # Asegura que se muestren todas las etiquetas de clase
    
     plt.tight_layout()  # Ajustar automáticamente los parámetros de subplot
     plt.show()  # Mostrar los gráficos



def apply_model(df, model, scaler, random_search):
    '''
    Function for applying the model to the complete dataset

    Parameters
    ----------
    df : Dataset with all the patients for applying the model
    model : model to apply
    scaler
    random_search

    Returns
    -------
    df : df with the predictions for all the patients
    accuracy_ajustada 

    '''
    
    X_total = df.drop(['Patient', 'Fisher Modified Scale  (neuroradiologist 1)'], axis=1)

    
    # Escalar todo el conjunto de datos
    X_total_scaled = scaler.transform(X_total)  # Usando el scaler ya ajustado previamente

    # SMOTE no es necesario aplicarlo aquí ya que solo queremos predecir, no entrenar
    # Predicción sobre todo el conjunto de datos
    Y_total_pred = random_search.predict(X_total_scaled)


    # Añadir las predicciones al dataframe
    df['Predicted Fisher Scale'] = Y_total_pred #+ 1  # sumamos 1 para volver al escalamiento original

    # Ajustar las predicciones de la escala de Fisher basadas en la presencia de IVH
    #df['Predicted Fisher Scale'] = df.apply(lambda row: row['Predicted Fisher Scale'] if row['IVH'] == 0 else (2 if row['Predicted Fisher Scale'] in [2, 4] else 4), axis=1)
    

    # Ahora df_combined contiene una nueva columna con las predicciones de la escala modificada de Fisher
    
    def calculate_score(row):
        if row['Fisher Modified Scale  (neuroradiologist 1)'] == 0:
            return 1 - (abs(row['Fisher Modified Scale  (neuroradiologist 1)'] - row['Predicted Fisher Scale']) / (row['Fisher Modified Scale  (neuroradiologist 1)'] + 1))

        else:
            return 1 - (abs(row['Fisher Modified Scale  (neuroradiologist 1)'] - row['Predicted Fisher Scale']) / row['Fisher Modified Scale  (neuroradiologist 1)'])

    
    df['score'] = df.apply(calculate_score, axis=1)
    accuracy_ajustada = df['score'].mean()

    # Calcular diferencias y proporcionar porcentaje de diferencia
    df['Dif 1-2'] = np.abs(df['Fisher Modified Scale  (neuroradiologist 1)'] - df['Predicted Fisher Scale'])

    # Calcular porcentajes de diferencias
    diff_percent_1_2 = 100 * np.mean(df['Dif 1-2'] > 0)

    print(f"Porcentaje de diferencias Neuroradiólogo 1 vs Predicted Fisher Scale: {diff_percent_1_2}%")

    # Contar las diferencias
    print(f"Número de diferencias Neuroradiólogo 1 vs Predicted Fisher Scale: {np.sum(df['Dif 1-2'] > 0)}")

    return df,  accuracy_ajustada
    

def get_line_plot(df):
    '''
    Function for obtaining a plot comparing the predicted Fisher vs the True Fisher value
    
    df -> Dataframe to plot containing the real Fisher value and the predicted one
    
    Returns
    -------
    None.

    '''
    
    patients = df['Patient']  
    real_scale = df['Fisher Modified Scale  (neuroradiologist 1)']
    predicted_scale = df['Predicted Fisher Scale']

    plt.figure(figsize=(20, 6))  # Configura el tamaño del gráfico
    plt.plot(patients, real_scale, label='Real Fisher Scale', marker='o')  # Línea con marcadores para escala real
    plt.plot(patients, predicted_scale, label='Predicted Fisher Scale', marker='x')  # Línea con marcadores para escala predicha

    plt.xlabel('Patient ID')  # Etiqueta del eje x
    plt.ylabel('Fisher Scale (0 to 5)')  # Etiqueta del eje y
    plt.title('Comparison of Real and Predicted Fisher Modified Scale by Patient')  # Título del gráfico
    plt.xticks(rotation=90)  # Rotar las etiquetas del eje x para mejor visualización
    plt.ylim(0, 5)  # Limitar los valores del eje y de 0 a 5
    plt.legend()  # Añadir leyenda
    plt.grid(True)  # Añadir rejilla para facilitar la lectura

    plt.tight_layout()  # Ajustar automáticamente los parámetros del subplot para que el gráfico se ajuste al área de visualización
    plt.show()  # Mostrar el gráfico
    
    
