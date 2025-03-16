#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 09:58:27 2024

@author: elenamartineztorrijos
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.exceptions import UndefinedMetricWarning
from funciones_clasificador import prepare_data
from imblearn.over_sampling import ADASYN, RandomOverSampler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import shap
import warnings
import seaborn as sns

# Suprimir advertencias de métricas indefinidas
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Preparación de datos
df_combined, X, Y = prepare_data()

# Definición de categorías para la respuesta
Y = df_combined['Fisher Modified Scale  (neuroradiologist 1)']

# Eliminamos columnas no necesarias
X = df_combined.drop(['Patient', 'Fisher Modified Scale  (neuroradiologist 1)'], axis=1)

# Función para graficar la distribución de clases
def plot_class_distribution(Y, title):
    sns.countplot(x=Y)
    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.show()

# Graficar la distribución antes de aplicar ADASYN
plot_class_distribution(Y, 'Class Distribution Before ADASYN')

# Configuración de validación cruzada estratificada
n_splits = 10

# Listas para almacenar las métricas
confusion_matrices = []
classification_reports = []
roc_aucs = []
precision_recall_aucs = []
fpr_list = []
tpr_list = []
precision_list = []
recall_list = []
all_predictions = []

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

# Bucle para la validación cruzada
for fold, (train_index, test_index) in enumerate(skf.split(X, Y)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

    # Ajustar el número de vecinos en ADASYN basado en la clase con menos muestras
    min_samples = Y_train.value_counts().min()
    if (min_samples > 1):
        adasyn_neighbors = min(min_samples - 1, 5)
        adasyn = ADASYN(random_state=42, n_neighbors=adasyn_neighbors)
    else:
        adasyn = RandomOverSampler(random_state=42)

    X_train_resampled, Y_train_resampled = adasyn.fit_resample(X_train, Y_train)

    # Graficar la distribución después de aplicar ADASYN en la primera iteración
    if (fold == 0):
        plot_class_distribution(Y_train_resampled, 'Class Distribution After ADASYN')

    # Escalar los datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)

    # Hiperparametros para RandomForest
    rf_params = {
        'n_estimators': [100, 200, 300, 400, 500],
        'max_features': ['sqrt', 'log2', 0.5, 0.7, 0.9],
        'max_depth': [None, 4, 5, 6, 7, 8, 10, 12],
        'criterion': ['gini', 'entropy'],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # GridSearchCV para RandomForest
    rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=3, n_jobs=-1, scoring='f1_macro')
    rf_grid.fit(X_train_scaled, Y_train_resampled)
    random_forest = rf_grid.best_estimator_

    # Predicciones y evaluación
    Y_pred = random_forest.predict(X_test_scaled)
    all_predictions.append((Y_test, Y_pred))

    # Crear una matriz de confusión de tamaño fijo (3x3) y rellenar con ceros si es necesario
    cm = confusion_matrix(Y_test, Y_pred, labels=[0, 1, 2])
    fixed_cm = np.zeros((3, 3))
    fixed_cm[:cm.shape[0], :cm.shape[1]] = cm
    confusion_matrices.append(fixed_cm)

    classification_reports.append(classification_report(Y_test, Y_pred, output_dict=True))

    # Obtener probabilidades para cada clase
    Y_prob = random_forest.predict_proba(X_test_scaled)

    # Calcular ROC AUC para cada clase y almacenar fpr y tpr
    roc_auc = {}
    fpr_fold = {}
    tpr_fold = {}
    for i, class_label in enumerate(random_forest.classes_):
        fpr, tpr, _ = roc_curve(Y_test == class_label, Y_prob[:, i])
        roc_auc[class_label] = auc(fpr, tpr)
        fpr_fold[class_label] = fpr
        tpr_fold[class_label] = tpr
    roc_aucs.append(roc_auc)
    fpr_list.append(fpr_fold)
    tpr_list.append(tpr_fold)

    precision_recall_auc = {}
    precision_fold = {}
    recall_fold = {}
    for i, class_label in enumerate(random_forest.classes_):
        precision, recall, _ = precision_recall_curve(Y_test == class_label, Y_prob[:, i])
        average_precision = average_precision_score(Y_test == class_label, Y_prob[:, i])
        precision_recall_auc[class_label] = average_precision
        precision_fold[class_label] = precision
        recall_fold[class_label] = recall
    precision_recall_aucs.append(precision_recall_auc)
    precision_list.append(precision_fold)
    recall_list.append(recall_fold)

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

# Mostrar resultados
print("Classification Report (mean ± std):\n", mean_classification_report, "\n±\n", std_classification_report)
print("ROC AUC (mean ± std):\n", mean_roc_auc, "\n±\n", std_roc_auc)
print("Precision-Recall AUC (mean ± std):\n", mean_precision_recall_auc, "\n±\n", std_precision_recall_auc)
print("Confusion Matrix (mean ± std):\n", mean_confusion_matrix, "\n±\n", std_confusion_matrix)

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

# Función para promediar las curvas Precision-Recall
def mean_precision_recall_curve(precision_list, recall_list, classes):
    mean_recall = np.linspace(0, 1, 100)
    precisions = {cls: [] for cls in classes}
    for fold in range(len(precision_list)):
        for cls in classes:
            precision = precision_list[fold][cls]
            recall = recall_list[fold][cls]
            precisions_interp = np.interp(mean_recall, recall[::-1], precision[::-1])
            precisions[cls].append(precisions_interp)
    
    mean_precision = {cls: np.mean(precisions[cls], axis=0) for cls in classes}
    std_precision = {cls: np.std(precisions[cls], axis=0) for cls in classes}
    return mean_recall, mean_precision, std_precision

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

# Función para graficar la curva Precision-Recall promedio con bandas de confianza
def plot_mean_precision_recall_curve(precision_list, recall_list, classes, mean_precision_recall_auc):
    mean_recall, mean_precision, std_precision = mean_precision_recall_curve(precision_list, recall_list, classes)
    
    plt.figure(figsize=(10, 7))
    for cls in classes:
        plt.plot(mean_recall, mean_precision[cls], label=f'Mean Precision-Recall curve (Class {cls}) (AP = {mean_precision_recall_auc[cls]:.2f})')
        plt.fill_between(mean_recall, mean_precision[cls] - std_precision[cls], mean_precision[cls] + std_precision[cls], alpha=0.2)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Mean Precision-Recall Curve')
    plt.legend(loc="lower right")
    plt.show()

# Función para calcular y graficar las curvas ROC para cada clase
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
def plot_precision_recall_curves(precision_list, recall_list, classes):
    plt.figure(figsize=(10, 7))
    for i, class_label in enumerate(classes):
        for fold, (precision, recall) in enumerate(zip(precision_list, recall_list)):
            plt.plot(recall[class_label], precision[class_label], alpha=0.3, lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower right")
    plt.show()

# Llamar a la función de graficación de ROC
plot_roc_curves(fpr_list, tpr_list, random_forest.classes_)

# Llamar a la función de graficación de Precision-Recall
plot_precision_recall_curves(precision_list, recall_list, random_forest.classes_)

# Llamar a la función de graficación de ROC promedio
plot_mean_roc_curve(fpr_list, tpr_list, random_forest.classes_, mean_roc_auc)

# Llamar a la función de graficación de Precision-Recall promedio
plot_mean_precision_recall_curve(precision_list, recall_list, random_forest.classes_, mean_precision_recall_auc)

# Interpretability using SHAP values
explainer = shap.TreeExplainer(random_forest)
shap_values = explainer.shap_values(X_test_scaled)

# Mostrar importancia de características individuales
feature_importances = random_forest.feature_importances_
for feature_name, importance in zip(X.columns, feature_importances):
    print(f"{feature_name}: {importance:.4f}")

# Plot SHAP summary plot for individual features
shap.summary_plot(shap_values, X_test_scaled, feature_names=X.columns)

# Almacenar predicciones en un archivo Excel
all_predictions_df = pd.DataFrame()
for fold, (y_test, y_pred) in enumerate(all_predictions):
    fold_df = pd.DataFrame({'Fold': fold, 'True': y_test, 'Predicted': y_pred})
    all_predictions_df = pd.concat([all_predictions_df, fold_df], ignore_index=True)

all_predictions_df.to_excel('/Users/elenamartineztorrijos/Desktop/TFG/predicciones.xlsx', index=False)
