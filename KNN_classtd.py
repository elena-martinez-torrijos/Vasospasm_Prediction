#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 10:00:46 2024

@author: elenamartineztorrijos
"""
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.exceptions import UndefinedMetricWarning
from funciones_clasificador import prepare_data
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import warnings

# Suprimir advertencias de métricas indefinidas
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Preparación de datos
df_combined, X, Y = prepare_data()

# Definición de categorías para la respuesta
Y = pd.cut(df_combined['Fisher Modified Scale  (neuroradiologist 1)'],
           bins=[-float('inf'), 1, 3, float('inf')], labels=[0, 1, 2], right=False)

# Eliminamos columnas no necesarias
X = df_combined.drop(['Patient', 'Fisher Modified Scale  (neuroradiologist 1)'], axis=1)

# Configuración de validación cruzada estratificada
n_splits = 10

# Listas para almacenar las métricas
confusion_matrices = []
classification_reports = []
roc_aucs = []
precision_recall_aucs = []
fpr_list = []
tpr_list = []
feature_importances = []

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

# Definición de hiperparámetros para KNeighborsClassifier
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'p': [1, 2]
}

# Bucle para la validación cruzada
for train_index, test_index in skf.split(X, Y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
    
    # Ajustar el número de vecinos en SMOTE basado en la clase con menos muestras
    min_samples = Y_train.value_counts().min()
    if min_samples > 1:
        smote_neighbors = min(min_samples - 1, 5)
        smote = SMOTE(random_state=42, k_neighbors=smote_neighbors)
    else:
        smote = RandomOverSampler(random_state=42)

    X_train_resampled, Y_train_resampled = smote.fit_resample(X_train, Y_train)
    
    # Escalar los datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)
    
    # GridSearchCV para KNeighborsClassifier
    knn_grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=3, n_jobs=-1, scoring='f1_macro')
    knn_grid.fit(X_train_scaled, Y_train_resampled)
    knn = knn_grid.best_estimator_
    
    # Predicciones y evaluación
    Y_pred = knn.predict(X_test_scaled)
    
    # Crear una matriz de confusión de tamaño fijo (3x3) y rellenar con ceros si es necesario
    cm = confusion_matrix(Y_test, Y_pred, labels=[0, 1, 2])
    fixed_cm = np.zeros((3, 3))
    fixed_cm[:cm.shape[0], :cm.shape[1]] = cm
    confusion_matrices.append(fixed_cm)
    
    classification_reports.append(classification_report(Y_test, Y_pred, output_dict=True))
    
    # Obtener probabilidades para cada clase
    Y_prob = knn.predict_proba(X_test_scaled)
    
    # Calcular ROC AUC para cada clase y almacenar fpr y tpr
    roc_auc = {}
    fpr_fold = {}
    tpr_fold = {}
    for i, class_label in enumerate(knn.classes_):
        fpr, tpr, _ = roc_curve(Y_test == class_label, Y_prob[:, i])
        roc_auc[class_label] = auc(fpr, tpr)
        fpr_fold[class_label] = fpr
        tpr_fold[class_label] = tpr
    roc_aucs.append(roc_auc)
    fpr_list.append(fpr_fold)
    tpr_list.append(tpr_fold)

    # Calcular Precision-Recall AUC para cada clase
    precision_recall_auc = {}
    for i, class_label in enumerate(knn.classes_):
        precision, recall, _ = precision_recall_curve(Y_test == class_label, Y_prob[:, i])
        average_precision = average_precision_score(Y_test == class_label, Y_prob[:, i])
        precision_recall_auc[class_label] = average_precision
    precision_recall_aucs.append(precision_recall_auc)

    # Importancia de características usando Permutation Importance
    result = permutation_importance(knn, X_test_scaled, Y_test, n_repeats=10, random_state=42, n_jobs=-1)
    feature_importances.append(result.importances_mean)

# Promediar y calcular la desviación estándar de los reportes de clasificación
mean_classification_report = average_dicts(classification_reports)
std_classification_report = std_dicts(classification_reports)

# Promediar y calcular la desviación estándar de las ROC AUCs
mean_roc_auc = pd.DataFrame(roc_aucs).mean(axis=0)
std_roc_auc = pd.DataFrame(roc_aucs).std(axis=0)

# Promediar y calcular la desviación estándar de las Precision-Recall AUCs
mean_precision_recall_auc = pd.DataFrame(precision_recall_aucs).mean(axis=0)
std_precision_recall_auc = pd.DataFrame(precision_recall_aucs).std(axis=0)

# Promediar y calcular la desviación estándar de las matrices de confusión
mean_confusion_matrix = np.mean(confusion_matrices, axis=0)
std_confusion_matrix = np.std(confusion_matrices, axis=0)

# Promediar y calcular la desviación estándar de la importancia de características
mean_feature_importances = np.mean(feature_importances, axis=0)
std_feature_importances = np.std(feature_importances, axis=0)

# Mostrar resultados
print("Classification Report (mean ± std):\n", mean_classification_report, "\n±\n", std_classification_report)
print("ROC AUC (mean ± std):\n", mean_roc_auc, "\n±\n", std_roc_auc)
print("Precision-Recall AUC (mean ± std):\n", mean_precision_recall_auc, "\n±\n", std_precision_recall_auc)
print("Confusion Matrix (mean ± std):\n", mean_confusion_matrix, "\n±\n", std_confusion_matrix)

# Mostrar importancia de características
print("Feature Importances (mean ± std):\n")
for feature_name, mean, std in zip(X.columns, mean_feature_importances, std_feature_importances):
    print(f"{feature_name}: {mean:.4f} ± {std:.4f}")

# Función para promediar las curvas ROC
def mean_roc_curve(fpr_list, tpr_list, classes):
    mean_fpr = np.linspace(0, 1, 100)
    tprs = {cls: [] for cls in classes}
    for fold in range(len(fpr_list)):
        for cls in classes:
            fpr = fpr_list[fold][cls]
            tpr = tpr_list[fold][cls]
            tprs_interp = np.interp(mean_fpr, fpr, tpr)
            tprs_interp[0] = 0.0
            tprs[cls].append(tprs_interp)
    
    mean_tpr = {cls: np.mean(tprs[cls], axis=0) for cls in classes}
    std_tpr = {cls: np.std(tprs[cls], axis=0) for cls in classes}
    for cls in classes:
        mean_tpr[cls][-1] = 1.0
    return mean_fpr, mean_tpr, std_tpr

# Función para graficar la curva ROC promedio con bandas de confianza
def plot_mean_roc_curve(fpr_list, tpr_list, classes, mean_roc_auc):
    mean_fpr, mean_tpr, std_tpr = mean_roc_curve(fpr_list, tpr_list, classes)
    
    plt.figure(figsize=(10, 7))
    for cls in classes:
        plt.plot(mean_fpr, mean_tpr[cls], label=f'Mean ROC curve (Class {cls}) (AUC = {mean_roc_auc[cls]:.2f})')
        plt.fill_between(mean_fpr, mean_tpr[cls] - std_tpr[cls], mean_tpr[cls] + std_tpr[cls], alpha=0.2)
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Mean Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

# Función para calcular y graficar la curva ROC para cada clase
def plot_roc_curves(fpr_list, tpr_list, classes):
    plt.figure(figsize=(10, 7))
    for i, class_label in enumerate(classes):
        for fold, (fpr, tpr) in enumerate(zip(fpr_list, tpr_list)):
            plt.plot(fpr[class_label], tpr[class_label], alpha=0.3, lw=1)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

# Función para calcular y graficar la curva de precisión-recall para cada clase
def plot_precision_recall_curve(Y_test, Y_prob, classes):
    plt.figure(figsize=(10, 7))
    for i in range(len(classes)):
        precision, recall, _ = precision_recall_curve(Y_test == classes[i], Y_prob[:, i])
        average_precision = average_precision_score(Y_test == classes[i], Y_prob[:, i])
        plt.step(recall, precision, where='post', label='Precision-Recall curve of class {0} (AP = {1:0.2f})'.format(classes[i], average_precision))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve')
    plt.legend(loc="upper right")
    plt.show()

# Llamar a la función de graficación de ROC
plot_roc_curves(fpr_list, tpr_list, knn.classes_)

# Llamar a la función de graficación de Precision-Recall
plot_precision_recall_curve(Y_test, Y_prob, knn.classes_)

# Llamar a la función de graficación de ROC promedio
plot_mean_roc_curve(fpr_list, tpr_list, knn.classes_, mean_roc_auc)
