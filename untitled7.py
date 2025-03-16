#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 18:50:47 2024

@author: elenamartineztorrijos
"""
import pandas as pd

# Carga los datos de Excel
data = pd.read_excel('/Users/elenamartineztorrijos/Desktop/neuroradiologist_classification.xlsx', usecols=['Clase (neuroradiólogo 1)', 'AnyVasospasm'],nrows=140)


# Separar los datos para cada clase
clase_0 = data[data["Clase (neuroradiólogo 1)"] == 0]
clase_1 = data[data["Clase (neuroradiólogo 1)"] == 1]
clase_2 = data[data["Clase (neuroradiólogo 1)"] == 2]

# Define la clase real para cada grupo
clase_real_0 = clase_0["AnyVasospasm"]
clase_real_1 = clase_1["AnyVasospasm"]
clase_real_2 = clase_2["AnyVasospasm"]

# Define la clase predicha para cada grupo
clase_predicha_0 = clase_0["Clase (neuroradiólogo 1)"]
clase_predicha_1 = clase_1["Clase (neuroradiólogo 1)"]
clase_predicha_2 = clase_2["Clase (neuroradiólogo 1)"]

# Calcula los verdaderos positivos (VP), verdaderos negativos (VN), falsos positivos (FP) y falsos negativos (FN) para cada clase
VP_1 = sum((clase_real_1 == 1) & (clase_predicha_1 == 1))
VN_1 = sum((clase_real_0 == 0) & (clase_predicha_1 == 0))
FP_1 = sum((clase_real_0 == 0) & (clase_predicha_1 == 1))
FN_1 = sum((clase_real_1 == 1) & (clase_predicha_1 == 0))

VP_2 = sum((clase_real_2 == 1) & (clase_predicha_2 == 1))
VN_2 = sum((clase_real_0 == 0) & (clase_predicha_2 == 0))
FP_2 = sum((clase_real_0 == 0) & (clase_predicha_2 == 1))
FN_2 = sum((clase_real_2 == 1) & (clase_predicha_2 == 0))

# Calcula la sensibilidad y la especificidad para cada clase
sensibilidad_1 = VP_1 / (VP_1 + FN_1)
especificidad_1 = VN_1 / (VN_1 + FP_1)

sensibilidad_2 = VP_2 / (VP_2 + FN_2)
especificidad_2 = VN_2 / (VN_2 + FP_2)

print("Sensibilidad clase 1:", sensibilidad_1)
print("Especificidad clase 1:", especificidad_1)
print("Sensibilidad clase 2:", sensibilidad_2)
print("Especificidad clase 2:", especificidad_2)
