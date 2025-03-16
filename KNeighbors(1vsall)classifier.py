#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 14:00:57 2024

@author: elenamartineztorrijos
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier

from funciones_clasificador import get_line_plot, apply_model, get_plot_balanced_classes, prepare_data

df_combined, X, Y = prepare_data()

# Ajustar las etiquetas de clase
Y_adjusted = Y #- 1  # Esto restará 1 a cada etiqueta en Y

# División de datos ajustada con las etiquetas modificadas
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y_adjusted, test_size=0.2, random_state=42, stratify=Y_adjusted)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SMOTE para equilibrar clases
smote = SMOTE(k_neighbors=1)

X_train_balanced, Y_train_balanced = smote.fit_resample(X_train_scaled, Y_train)

get_plot_balanced_classes(Y_train, Y_train_balanced)

knn = KNeighborsClassifier()
param_grid = {
    'n_neighbors': range(1, 20),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, verbose=1, scoring='accuracy')
ovr_knn = OneVsRestClassifier(grid_search)
ovr_knn.fit(X_train_balanced, Y_train_balanced)

# Mostrar el mejor número de vecinos para cada clasificador
for i, estimator in enumerate(ovr_knn.estimators_):
    print(f"El número de vecinos para el clasificador {i} es {estimator.best_estimator_.get_params()['n_neighbors']}")

# Predicciones y evaluación
Y_pred = ovr_knn.predict(X_test_scaled)

# Ajustar predicciones basadas en la columna IVH
Y_pred_adjusted = [4 if (ivh == 1) else pred for ivh, pred in zip(df_combined.loc[X_test.index, 'IVH'], Y_pred)]


print("Matriz de confusión:")
print(confusion_matrix(Y_test, Y_pred_adjusted))
print("Informe de clasificación:")
print(classification_report(Y_test, Y_pred_adjusted, zero_division=0))


df_combined, accuracy_ajustada = apply_model(df_combined, grid_search, scaler, ovr_knn)
print(df_combined[['Patient', 'Fisher Modified Scale  (neuroradiologist 1)', 'Predicted Fisher Scale','score']])
print("Adjusted accuracy:", accuracy_ajustada)

# Obtain the comparison plots
get_line_plot(df_combined)

from sklearn.metrics import cohen_kappa_score
kappa = cohen_kappa_score(Y_test, Y_pred_adjusted)
print("Cohen's Kappa:", kappa)

from sklearn.metrics import f1_score
f1 = f1_score(Y_test, Y_pred_adjusted, average='weighted')
print("F1-Score:", f1)

from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(Y_test, Y_pred_adjusted)
print("Confusion Matrix:\n", conf_matrix)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(Y_test, Y_pred_adjusted)
print("Accuracy:", accuracy)
