#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 09:48:33 2024

@author: elenamartineztorrijos
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

def prepare_data():
    '''
    Returns
    -------
    df_combined
    X
    Y
    '''

    # Load necessary columns directly from the specified Excel files
    df_hsa = pd.read_excel('/Users/elenamartineztorrijos/Desktop/TFG/Data List HSA.xlsx', usecols=['Patient', 'IVH', 'Bleed', 'Supraselar cistern bleed', 'Fisher Modified Scale  (neuroradiologist 1)'], sheet_name='Hoja3')
    
    df_hemorrhage = pd.read_excel('/Users/elenamartineztorrijos/Desktop/TFG/Results_model_seg.xlsx', usecols=['Patient', 'Section (0, 1)', 'Section (0, 2)', 'Section (0, 3)', 'Section (1, 1)', 'Section (1, 2)', 'Section (1, 3)', 'Section (1, 4)', 'Section (2, 0)', 'Section (2, 1)', 'Section (2, 2)', 'Section (2, 3)', 'Section (2, 4)', 'Section (3, 0)', 'Section (3, 1)', 'Section (3, 2)', 'Section (3, 3)', 'Section (4, 1)', 'Section (4, 2)', 'Section (4, 3)', 'Hemorrhage in Column 2'])
    
    df_thickness = pd.read_excel('/Users/elenamartineztorrijos/Desktop/TFG/Max_Slice_Thickness_results.xlsx', usecols=['Patient', 'Max_Slice_Thickness'])

    df_density = pd.read_excel('/Users/elenamartineztorrijos/Desktop/TFG/Average_Density_results.xlsx', usecols=['Patient', 'Average_Density_HU'])

    # Find common patients across all datasets
    common_patients = set(df_hsa['Patient']) & set(df_hemorrhage['Patient']) & set(df_thickness['Patient']) & set(df_density['Patient'])

    # Filter data for common patients
    df_hsa_common = df_hsa[df_hsa['Patient'].isin(common_patients)]
    df_hemorrhage_common = df_hemorrhage[df_hemorrhage['Patient'].isin(common_patients)]
    df_thickness_common = df_thickness[df_thickness['Patient'].isin(common_patients)]
    df_density_common = df_density[df_density['Patient'].isin(common_patients)]

    # Combine data by merging on 'Patient'
    df_combined = pd.merge(df_hsa_common, df_hemorrhage_common, on='Patient')
    df_combined = pd.merge(df_combined, df_thickness_common, on='Patient')
    df_combined = pd.merge(df_combined, df_density_common, on='Patient')

    # Prepare X and Y datasets
    X = df_combined.drop(['Patient', 'Fisher Modified Scale  (neuroradiologist 1)'], axis=1)
    Y = df_combined['Fisher Modified Scale  (neuroradiologist 1)'].astype(int)
    
    return df_combined, X, Y

# Prepare data
df_combined, X, Y = prepare_data()

# Count samples and categories
print("Número total de muestras:", len(Y))
print("Número de categorías:", len(set(Y)))
print(df_combined.columns)

# Assuming 'df_combined' is your DataFrame and 'Y_binary' is your binary response vector
X = df_combined.drop(['Patient', 'Fisher Modified Scale  (neuroradiologist 1)'], axis=1)
Y_binary = [1 if y > 3 else 0 for y in df_combined['Fisher Modified Scale  (neuroradiologist 1)']]

# Verify binary labels
print("Etiquetas binarias (primeras 10):", Y_binary[:10])

# Initialize lists to store ROC AUC scores, curves, and logistic regression weights
roc_aucs = []
roc_curves = []
pr_curves = []
log_reg_weights = []

# Perform the training and evaluation 10 times
for i in range(10):
    # Split data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y_binary, test_size=0.3, random_state=i, stratify=Y_binary)
    
    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train logistic regression model
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train_scaled, Y_train)
    
    # Store logistic regression weights
    log_reg_weights.append(log_reg.coef_[0])
    
    # Predict probabilities
    Y_prob_log_reg = log_reg.predict_proba(X_test_scaled)[:, 1]

    # Create DataFrame for predictions and true labels
    predictions_df = pd.DataFrame({'True_Labels': Y_test, 'Predicted_Probabilities': Y_prob_log_reg})

    # Save predictions to Excel
    predictions_df.to_excel('/Users/elenamartineztorrijos/Desktop/TFG/Predictions.xlsx', index=False)

# Calculate ROC AUC
roc_auc = roc_auc_score(Y_test, Y_prob_log_reg)
roc_aucs.append(roc_auc)

# Calculate ROC curve
fpr, tpr, _ = roc_curve(Y_test, Y_prob_log_reg)
roc_curves.append((fpr, tpr))

# Calculate Precision-Recall curve
precision, recall, _ = precision_recall_curve(Y_test, Y_prob_log_reg)
pr_curves.append((precision, recall))

# Calculate mean and standard deviation of ROC AUC
mean_roc_auc = np.mean(roc_aucs)
std_roc_auc = np.std(roc_aucs)

# Calculate mean and standard deviation of logistic regression weights
mean_weights = np.mean(log_reg_weights, axis=0)
std_weights = np.std(log_reg_weights, axis=0)

# Display weights
print("Pesos del regresor logístico (media ± desviación estándar):")
for feature, mean, std in zip(X.columns, mean_weights, std_weights):
    print(f"{feature}: {mean:.4f} ± {std:.4f}")

# Plot ROC curve with shaded area for standard deviation
mean_fpr = np.linspace(0, 1, 100)
mean_tpr = np.mean([np.interp(mean_fpr, fpr, tpr) for fpr, tpr in roc_curves], axis=0)
std_tpr = np.std([np.interp(mean_fpr, fpr, tpr) for fpr, tpr in roc_curves], axis=0)

plt.figure()
plt.plot(mean_fpr, mean_tpr, color='darkorange', lw=2, label=f'ROC curve (mean AUC = {mean_roc_auc:.2f})')
plt.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, color='darkorange', alpha=0.2)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Plot Precision-Recall curve with shaded area for standard deviation
mean_recall = np.linspace(0, 1, 100)
mean_precision = np.mean([np.interp(mean_recall, recall[::-1], precision[::-1]) for precision, recall in pr_curves], axis=0)
std_precision = np.std([np.interp(mean_recall, recall[::-1], precision[::-1]) for precision, recall in pr_curves], axis=0)

plt.figure()
plt.step(mean_recall, mean_precision, color='darkorange', lw=2, label=f'Precision-Recall curve (AP = {average_precision_score(Y_test, Y_prob_log_reg):.2f})')
plt.fill_between(mean_recall, mean_precision - std_precision, mean_precision + std_precision, color='darkorange', alpha=0.2, step='post')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall curve')
plt.legend(loc="upper right")
plt.show()
