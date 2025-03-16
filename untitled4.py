#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 17:37:46 2024

@author: elenamartineztorrijos
"""

import pandas as pd
import re

# Función para extraer el número después de "HSA" en cada identificador
def extract_number(patient):
    match = re.search(r'\d+', patient)
    if match:
        return int(match.group())
    else:
        return 0

# Carga la tabla desde el archivo Excel
df = pd.read_excel("/Users/elenamartineztorrijos/Desktop/TFG/patient_predictions_lr.xlsx")

# Ordena la tabla por la columna "Patient"
df['Number'] = df['Patient'].apply(extract_number)
df_sorted = df.sort_values(by='Number')

# Elimina la columna temporal 'Number'
df_sorted.drop('Number', axis=1, inplace=True)

# Guarda el DataFrame ordenado en un nuevo archivo Excel
df_sorted.to_excel("/Users/elenamartineztorrijos/Desktop/archivo_ordenado_FIN.xlsx", index=False)
