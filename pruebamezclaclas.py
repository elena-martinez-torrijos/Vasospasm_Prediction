#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 16:50:43 2024

@author: elenamartineztorrijos
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.exceptions import UndefinedMetricWarning
from funciones_clasificador import prepare_data
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import shap
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

# Generación de nuevas características polinómicas
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)

# Configuración de validación cruzada estratificada
n_splits = 10
test_size = 0.3  # Tamaño de prueba único para simplificar

# Lista para almacenar las métricas
confusion_matrices = []
classification_reports = []
roc_aucs = []
precision_recall_aucs = []
logistic_weights = []
fpr_list = []
tpr_list = []

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

# Pipelines para diferentes modelos
pipelines = {
    'RandomForest': Pipeline([
        ('poly', poly),
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(random_state=42))
    ]),
    'LogisticRegression': Pipeline([
        ('poly', poly),
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, random_state=42, multi_class='ovr'))
    ]),
    'GradientBoosting': Pipeline([
        ('poly', poly),
        ('scaler', StandardScaler()),
        ('clf', GradientBoostingClassifier(random_state=42))
    ]),
    'XGBoost': Pipeline([
        ('poly', poly),
        ('scaler', StandardScaler()),
        ('clf', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42))
    ])
}

# Hiperparametros para GridSearchCV
param_grids = {
    'RandomForest': {
        'clf__n_estimators': [100, 200],
        'clf__max_features': ['sqrt', 'log2', 0.5, 0.7],
        'clf__max_depth': [4, 5, 6, 7, 8],
        'clf__criterion': ['gini', 'entropy']
    },
    'LogisticRegression': {
        'clf__C': [0.1, 1, 10, 100],
        'clf__solver': ['lbfgs', 'liblinear'],
        'clf__class_weight': ['balanced', None]
    },
    'GradientBoosting': {
        'clf__n_estimators': [100, 200],
        'clf__learning_rate': [0.01, 0.1, 0.2],
        'clf__max_depth': [3, 4, 5]
    },
    'XGBoost': {
        'clf__n_estimators': [100, 200],
        'clf__learning_rate': [0.01, 0.1, 0.2],
        'clf__max_depth': [3, 4, 5],
        'clf__subsample': [0.8, 0.9, 1.0]
    }
}

# Iterar sobre diferentes modelos
for model_name, pipeline in pipelines.items():
    print(f"Evaluating model: {model_name}")
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=42)
    
    for fold, (train_index, test_index) in enumerate(sss.split(X, Y)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

        # Ajustar el número de vecinos en ADASYN basado en la clase con menos muestras
        min_samples = Y_train.value_counts().min()
        if min_samples > 1:
            adasyn_neighbors = min(min_samples - 1, 5)
            adasyn = ADASYN(random_state=42, n_neighbors=adasyn_neighbors)
        else:
            adasyn = RandomOverSampler(random_state=42)

        X_train_resampled, Y_train_resampled = adasyn.fit_resample(X_train, Y_train)

        # Búsqueda de hiperparámetros
        grid = GridSearchCV(pipeline, param_grids[model_name], cv=3, n_jobs=-1)
        grid.fit(X_train_resampled, Y_train_resampled)
        best_model = grid.best_estimator_

        # Predicciones y evaluación
        Y_pred = best_model.predict(X_test)
        cm = confusion_matrix(Y_test, Y_pred, labels=[0, 1, 2])
        fixed_cm = np.zeros((3, 3))
        fixed_cm[:cm.shape[0], :cm.shape[1]] = cm
        confusion_matrices.append(fixed_cm)

        classification_reports.append(classification_report(Y_test, Y_pred, output_dict=True))

        Y_prob = best_model.predict_proba(X_test)
        roc_auc = {}
        fpr_fold = {}
        tpr_fold = {}
        for i, class_label in enumerate(best_model.classes_):
            fpr, tpr, _ = roc_curve(Y_test == class_label, Y_prob[:, i])
            roc_auc[class_label] = auc(fpr, tpr)
            fpr_fold[class_label] = fpr
            tpr_fold[class_label] = tpr
        roc_aucs.append(roc_auc)
        fpr_list.append(fpr_fold)
        tpr_list.append(tpr_fold)

        precision_recall_auc = {}
        for i, class_label in enumerate(best_model.classes_):
            precision, recall, _ = precision_recall_curve(Y_test == class_label, Y_prob[:, i])
            average_precision = average_precision_score(Y_test == class_label, Y_prob[:, i])
            precision_recall_auc[class_label] = average_precision
        precision_recall_aucs.append(precision_recall_auc)

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

# Función para calcular y graficar las curvas ROC para cada clase
def plot_roc_curves(fpr_list, tpr_list, classes):
    plt.figure(figsize=(10, 7))
    for i, class_label in enumerate(classes):
        for fold, (fpr, tpr) in enumerate(zip(fpr_list, tpr_list)):
            plt.plot(fpr[class_label], tpr[class_label], alpha=0.3, lw=1, label=f'Fold {fold} (Class {class_label})')
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
plot_roc_curves(fpr_list, tpr_list, best_model.classes_)

# Llamar a la función de graficación de Precision-Recall
plot_precision_recall_curve(Y_test, Y_prob, best_model.classes_)

# Llamar a la función de graficación de ROC promedio
plot_mean_roc_curve(fpr_list, tpr_list, best_model.classes_, mean_roc_auc)

# Interpretability using SHAP values
explainer = shap.Explainer(best_model.named_steps['clf'], X_test)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test, feature_names=poly.get_feature_names_out(X.columns))
