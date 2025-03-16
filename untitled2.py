#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 10:24:01 2024

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

# Binarización de etiquetas
Y_binary = [1 if y > 3 else 0 for y in df_combined['Fisher Modified Scale  (neuroradiologist 1)']]

# Verificar las etiquetas binarias
print("Etiquetas binarias (primeras 10):", Y_binary[:10])


# Ajustar las etiquetas de clase
#Y_adjusted = Y #- 1  # Esto restará 1 a cada etiqueta en Y

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# Asumimos que df_combined y Y_binary ya están definidos correctamente
X = df_combined.drop(['Patient', 'Fisher Modified Scale  (neuroradiologist 1)'], axis=1)
Y = np.array(Y_binary)  # Asegurándonos de que Y es un numpy array

# Dividir los datos
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)

# Escalar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenar el modelo K-NN
knn = KNeighborsClassifier(n_neighbors=5)  # Puedes ajustar el número de vecinos
knn.fit(X_train_scaled, Y_train)

# Obtener las probabilidades de las predicciones para el conjunto de prueba
Y_prob_knn = knn.predict_proba(X_test_scaled)[:, 1]

# Modelo K-Nearest Neighbors
#knn = KNeighborsClassifier()
#knn.fit(X_train, Y_train)
# Revisar las primeras 10 etiquetas binarias de Y_test
print("Primeras 10 etiquetas de prueba (Y_test):", Y_test[:10])




from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Calcular la curva ROC y AUC
fpr_knn, tpr_knn, _ = roc_curve(Y_test, Y_prob_knn)
roc_auc_knn = auc(fpr_knn, tpr_knn)

# Graficar la curva ROC
plt.figure()
plt.plot(fpr_knn, tpr_knn, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_knn)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for K-NN')
plt.legend(loc="lower right")
plt.show()

from sklearn.metrics import precision_recall_curve, average_precision_score
# Calcular Precision-Recall y su puntaje medio
precision_knn, recall_knn, _ = precision_recall_curve(Y_test, Y_prob_knn)
average_precision_knn = average_precision_score(Y_test, Y_prob_knn)

# Graficar la curva Precision-Recall
plt.figure()
plt.step(recall_knn, precision_knn, where='post', color='blue', label='Precision-Recall curve (AP = %0.2f)' % average_precision_knn)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall curve for K-NN')
plt.legend(loc="upper right")
plt.show()
