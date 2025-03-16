#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 14:00:46 2024

@author: elenamartineztorrijos
"""
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from funciones_clasificador import get_line_plot, apply_model, get_plot_balanced_classes, prepare_data

df_combined, X, Y = prepare_data()
    
# Ajustar las etiquetas de clase
Y_adjusted = Y  # No resta 1 a cada etiqueta en Y

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


params = {
    'max_depth': [3, 4, 5, 6, 7],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [50, 100, 200, 300],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0]
}

model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
random_search = RandomizedSearchCV(model, param_distributions=params, n_iter=100, cv=5, verbose=1, random_state=42, n_jobs=-1)
random_search.fit(X_train_balanced, Y_train_balanced)

Y_pred = random_search.predict(X_test_scaled)

# Ajustar predicciones basadas en la columna IVH
Y_pred_adjusted = [4 if (ivh == 1) else pred for ivh, pred in zip(df_combined.loc[X_test.index, 'IVH'], Y_pred)]


print("Matriz de confusión:")
print(confusion_matrix(Y_test, Y_pred_adjusted))
print("Informe de clasificación:")
print(classification_report(Y_test, Y_pred_adjusted, zero_division=1))


df_combined, accuracy_ajustada = apply_model(df_combined, model, scaler, random_search)
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
