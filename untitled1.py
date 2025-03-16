#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 09:48:33 2024

@author: elenamartineztorrijos
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
import numpy as np
from funciones_clasificador import get_line_plot, apply_model, get_plot_balanced_classes, prepare_data

df_combined, X, Y = prepare_data()

# Eliminar filas donde 'IVH' es NA
df_combined = df_combined.dropna(subset=['IVH'])

# Contar muestras y categorías
print("Número total de muestras:", len(Y))
print("Número de categorías:", len(set(Y)))

print(df_combined.columns)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Asumiendo que df_combined es tu DataFrame y que 'Y_binary' es tu vector de respuesta binario
# Asume que 'Fisher Modified Scale (neuroradiologist 1)' es la columna de respuesta
X = df_combined.drop(['Patient','Fisher Modified Scale  (neuroradiologist 1)'], axis=1)

# Binarización de etiquetas
Y_binary = [1 if y > 2 else 0 for y in df_combined['Fisher Modified Scale  (neuroradiologist 1)']]

# Verificar las etiquetas binarias
print("Etiquetas binarias (primeras 10):", Y_binary[:10])


# Ajustar las etiquetas de clase
#Y_adjusted = Y #- 1  # Esto restará 1 a cada etiqueta en Y

# División de datos ajustada con las etiquetas modificadas
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y_binary, test_size=0.3, random_state=None, stratify=Y_binary)

# Verificar los tamaños de las etiquetas de entrenamiento y prueba
print("Etiquetas de entrenamiento:", Y_train[:10])
print("Etiquetas de prueba:", Y_test[:10])


# Asegúrate de que los datos Y_train y Y_test sean np.array para operaciones de sum y comparación correctas
Y_train_np = np.array(Y_train)
Y_test_np = np.array(Y_test)

# Imprimir la proporción de clases en los conjuntos original, de entrenamiento y de prueba
print("Distribución original:", {0: np.sum(np.array(Y_binary) == 0), 1: np.sum(np.array(Y_binary) == 1)})
print("Distribución en entrenamiento:", {0: np.sum(Y_train == 0), 1: np.sum(Y_train == 1)})
print("Distribución en prueba:", {0: np.sum(Y_test == 0), 1: np.sum(Y_test == 1)})

# Escalar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Verificar que no haya valores no numéricos en X_train o X_test
print("Verificación de tipos de datos en X_train:", X_train.dtypes)

# Convertir de nuevo a DataFrame para mantener nombres de características
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, Y_train)

# Predicciones de probabilidad
Y_prob_log_reg = log_reg.predict_proba(X_test_scaled)[:, 1]

# Modelo K-Nearest Neighbors
#knn = KNeighborsClassifier()
#knn.fit(X_train, Y_train)
# Revisar las primeras 10 etiquetas binarias de Y_test
print("Primeras 10 etiquetas de prueba (Y_test):", Y_test[:10])




from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Calcular los valores para la curva ROC
fpr, tpr, thresholds = roc_curve(Y_test, Y_prob_log_reg)
roc_auc = auc(fpr, tpr)

# Graficar la curva ROC
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

from sklearn.metrics import precision_recall_curve, average_precision_score

precision, recall, _ = precision_recall_curve(Y_test, Y_prob_log_reg)
average_precision = average_precision_score(Y_test, Y_prob_log_reg)

# Graficar la curva Precision-Recall
plt.figure()
plt.step(recall, precision, where='post', label='Precision-Recall curve (AP = %0.2f)' % average_precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall curve')
plt.legend(loc="upper right")
plt.show()
