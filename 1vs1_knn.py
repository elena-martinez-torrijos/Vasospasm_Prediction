#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 13:21:01 2024

@author: elenamartineztorrijos
"""

######################## DATA ########################

import pandas as pd

# Cargar columnas específicas del archivo Data List HSA.xlsx
# Supongamos que quieres cargar las columnas 'Patient' y 'Data1'
df_hsa = pd.read_excel('/Users/elenamartineztorrijos/Desktop/TFG/Data List HSA.xlsx', usecols=['Patient', 'IVH', 'Fisher Modified Scale (ICU initial)'])

# Cargar columnas específicas del archivo hemorrhage_results.xlsx
# Supongamos que quieres cargar las columnas 'Patient' y 'Result'
df_hemorrhage = pd.read_excel('/Users/elenamartineztorrijos/Desktop/TFG/hemorrhage_results.xlsx')

# Identificar los pacientes comunes entre los dos conjuntos de datos
common_patients = set(df_hsa['Patient']).intersection(set(df_hemorrhage['Patient']))

# Filtrar los datos para incluir solo los pacientes comunes
df_hsa_common = df_hsa[df_hsa['Patient'].isin(common_patients)]
df_hemorrhage_common = df_hemorrhage[df_hemorrhage['Patient'].isin(common_patients)]

# Combinar los datos en un solo DataFrame
df_combined = pd.merge(df_hsa_common, df_hemorrhage_common, on='Patient', how='inner')

print(df_combined)

######################## MODEL ########################

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import LeaveOneOut

# Preparación de los datos
X = df_combined.drop(['Patient', 'Fisher Modified Scale (ICU initial)'], axis=1)
Y = df_combined['Fisher Modified Scale (ICU initial)'].astype(int)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Ajustes de SMOTE
print("Distribución de clases antes de SMOTE:", Y_train.value_counts())
smote = SMOTE(k_neighbors=1)  # Uso del mínimo de vecinos posibles
X_train_balanced, Y_train_balanced = smote.fit_resample(X_train_scaled, Y_train)
print("Distribución de clases después de SMOTE:", pd.Series(Y_train_balanced).value_counts())

# Configuración del modelo
knn = KNeighborsClassifier()
param_grid = {'n_neighbors': range(1, 5)}
grid_search = GridSearchCV(knn, param_grid, cv=3)  # Uso de un número de pliegues reducido
ovo_classifier = OneVsOneClassifier(grid_search)
ovo_classifier.fit(X_train_balanced, Y_train_balanced)

# Evaluación del modelo
Y_pred = ovo_classifier.predict(X_test_scaled)
print(confusion_matrix(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))


# Imprime el informe de clasificación ajustando el manejo de divisiones por cero
print(classification_report(Y_test, Y_pred, zero_division=1))
