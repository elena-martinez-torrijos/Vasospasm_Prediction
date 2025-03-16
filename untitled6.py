#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 13:25:48 2024

@author: elenamartineztorrijos
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
import warnings
from funciones_clasificador2 import prepare_data

# Suprimir advertencias de métricas indefinidas
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Preparación de datos
df_combined, X, Y = prepare_data()

# Y ya está definido como 0 y 1, no necesita pd.cut
X = df_combined.drop(['Patient', 'AnyVasospasm'], axis=1)

# Configuración de validación cruzada estratificada
n_splits = 10

# Listas para almacenar las métricas
confusion_matrices = []
classification_reports = []
roc_aucs = []
precision_recall_aucs = []
feature_importances = []
fpr_list = []
tpr_list = []
precision_list = []
recall_list = []
sensitivities = []
specificities = []

# Función para promediar los reportes de clasificación
def average_dicts(dicts):
    keys = set(k for d in dicts if isinstance(d, dict) for k in d.keys())
    average_dict = {}
    for key in keys:
        if isinstance(dicts[0].get(key, {}), dict):
            average_dict[key] = average_dicts([d.get(key, {k: 0 for k in dicts[0].get(key, {})}) if isinstance(d, dict) else {} for d in dicts])
        else:
            average_dict[key] = np.mean([d.get(key, 0) if isinstance(d, dict) else 0 for d in dicts])
    return average_dict

# Función para calcular la desviación estándar de los reportes de clasificación
def std_dicts(dicts):
    keys = set(k for d in dicts if isinstance(d, dict) for k in dicts[0].keys())
    std_dict = {}
    for key in keys:
        if isinstance(dicts[0].get(key, {}), dict):
            std_dict[key] = std_dicts([d.get(key, {k: 0 for k in dicts[0].get(key, {})}) if isinstance(d, dict) else {} for d in dicts])
        else:
            std_dict[key] = np.std([d.get(key, 0) if isinstance(d, dict) else 0 for d in dicts])
    return std_dict

# Iterar sobre el proceso de validación cruzada
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Definición de hiperparámetros para LogisticRegression
param_grid = {
    'penalty': ['l1', 'l2', 'elasticnet', 'none'],
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'saga', 'lbfgs'],
    'max_iter': [100, 200, 300]
}

# Bucle para la validación cruzada
for train_index, test_index in skf.split(X, Y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
    
    # Ajustar el número de vecinos en SMOTE basado en la clase con menos muestras
    min_samples = Y_train.value_counts().min()
    smote_neighbors = max(min_samples - 1, 1)
    smote = SMOTE(random_state=42, k_neighbors=smote_neighbors)
    X_train_smote, Y_train_smote = smote.fit_resample(X_train, Y_train)
    
    # Escalar los datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_smote)
    X_test_scaled = scaler.transform(X_test)
    
    # GridSearchCV para LogisticRegression
    lr_grid = GridSearchCV(LogisticRegression(random_state=42), param_grid, cv=3, n_jobs=-1, scoring='f1_macro')
    lr_grid.fit(X_train_scaled, Y_train_smote)
    logistic_regression = lr_grid.best_estimator_
    
    # Predicciones y evaluación
    Y_pred = logistic_regression.predict(X_test_scaled)
    
    # Crear una matriz de confusión de tamaño fijo (2x2) y rellenar con ceros si es necesario
    cm = confusion_matrix(Y_test, Y_pred, labels=[0, 1])
    fixed_cm = np.zeros((2, 2))
    fixed_cm[:cm.shape[0], :cm.shape[1]] = cm
    confusion_matrices.append(fixed_cm)
    
    classification_reports.append(classification_report(Y_test, Y_pred, output_dict=True))
    
    # Obtener probabilidades para la clase positiva
    Y_prob = logistic_regression.predict_proba(X_test_scaled)[:, 1]
    
    # Calcular ROC AUC
    fpr, tpr, thresholds = roc_curve(Y_test, Y_prob)
    roc_auc = auc(fpr, tpr)
    roc_aucs.append(roc_auc)
    fpr_list.append(fpr)
    tpr_list.append(tpr)
    
    # Calcular sensibilidad y especificidad para el umbral óptimo
    optimal_idx = np.argmax(tpr - fpr)
    sensitivity = tpr[optimal_idx]
    specificity = 1 - fpr[optimal_idx]
    sensitivities.append(sensitivity)
    specificities.append(specificity)

    # Calcular Precision-Recall AUC
    precision, recall, _ = precision_recall_curve(Y_test, Y_prob)
    average_precision = average_precision_score(Y_test, Y_prob)
    precision_recall_aucs.append(average_precision)
    precision_list.append(precision)
    recall_list.append(recall)

    # Importancia de características usando Permutation Importance
    result = permutation_importance(logistic_regression, X_test_scaled, Y_test, n_repeats=10, random_state=42, n_jobs=-1)
    feature_importances.append(result.importances_mean)

# Promediar y calcular la desviación estándar de los reportes de clasificación
mean_classification_report = average_dicts(classification_reports)
std_classification_report = std_dicts(classification_reports)

# Promediar y calcular la desviación estándar de las ROC AUCs
mean_roc_auc = np.mean(roc_aucs)
std_roc_auc = np.std(roc_aucs)

# Promediar y calcular la desviación estándar de las Precision-Recall AUCs
mean_precision_recall_auc = np.mean(precision_recall_aucs)
std_precision_recall_auc = np.std(precision_recall_aucs)

# Promediar y calcular la desviación estándar de las matrices de confusión
mean_confusion_matrix = np.mean(confusion_matrices, axis=0)
std_confusion_matrix = np.std(confusion_matrices, axis=0)

# Promediar y calcular la desviación estándar de la importancia de características
mean_feature_importances = np.mean(feature_importances, axis=0)
std_feature_importances = np.std(feature_importances, axis=0)

# Promediar y calcular la desviación estándar de sensibilidad y especificidad
mean_sensitivity = np.mean(sensitivities)
std_sensitivity = np.std(sensitivities)
mean_specificity = np.mean(specificities)
std_specificity = np.std(specificities)

# Mostrar resultados
print("Classification Report (mean ± std):\n", mean_classification_report, "\n±\n", std_classification_report)
print("ROC AUC (mean ± std):\n", mean_roc_auc, "\n±\n", std_roc_auc)
print("Precision-Recall AUC (mean ± std):\n", mean_precision_recall_auc, "\n±\n", std_precision_recall_auc)
print("Confusion Matrix (mean ± std):\n", mean_confusion_matrix, "\n±\n", std_confusion_matrix)
print(f"Sensitivity (mean ± std): {mean_sensitivity:.4f} ± {std_sensitivity:.4f}")
print(f"Specificity (mean ± std): {mean_specificity:.4f} ± {std_specificity:.4f}")

# Mostrar importancia de características
print("Feature Importances (mean ± std):\n")
for feature_name, mean, std in zip(X.columns, mean_feature_importances, std_feature_importances):
    print(f"{feature_name}: {mean:.4f} ± {std:.4f}")

# Función para graficar la curva ROC promedio con bandas de confianza
def plot_mean_roc_curve(fpr_list, tpr_list, mean_roc_auc):
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    for fpr, tpr in zip(fpr_list, tpr_list):
        tprs_interp = np.interp(mean_fpr, fpr, tpr)
        tprs_interp[0] = 0.0
        tprs.append(tprs_interp)
    
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(tprs, axis=0)
    
    plt.figure(figsize=(10, 7))
    plt.plot(mean_fpr, mean_tpr, label=f'Mean ROC curve (AUC = {mean_roc_auc:.2f})')
    plt.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, alpha=0.2)
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Mean Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

# Llamar a la función de graficación de ROC promedio
plot_mean_roc_curve(fpr_list, tpr_list, mean_roc_auc)

# Función para graficar la curva de precisión-recall promedio con bandas de confianza
def plot_mean_precision_recall_curve(precision_list, recall_list, mean_precision_recall_auc):
    mean_recall = np.linspace(0, 1, 100)
    precisions = []
    for precision, recall in zip(precision_list, recall_list):
        precision_interp = np.interp(mean_recall, recall[::-1], precision[::-1])
        precisions.append(precision_interp)
    
    mean_precision = np.mean(precisions, axis=0)
    std_precision = np.std(precisions, axis=0)
    
    plt.figure(figsize=(10, 7))
    plt.step(mean_recall, mean_precision, where='post', label=f'Precision-Recall curve (AP = {mean_precision_recall_auc:.2f})')
    plt.fill_between(mean_recall, mean_precision - std_precision, mean_precision + std_precision, alpha=0.2)
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve')
    plt.legend(loc="upper right")
    plt.show()

# Llamar a la función de graficación de Precision-Recall promedio
plot_mean_precision_recall_curve(precision_list, recall_list, mean_precision_recall_auc)
