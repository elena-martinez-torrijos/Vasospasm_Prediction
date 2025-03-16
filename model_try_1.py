#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 12:52:52 2024

@author: elenamartineztorrijos
"""
import pydicom
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam as LegacyAdam
from tensorflow.keras import backend as K
import tensorflow as tf

# Constants for the image size and channels
HEIGHT = 256
WIDTH = 256
CHANNELS = 3  # Set to 3 for RGB images
SHAPE = (HEIGHT, WIDTH, CHANNELS)

# DICOM WINDOWING AND DATA NORMALIZATION
def correct_dcm(dcm):
    x = dcm.pixel_array + 1000
    px_mode = 4096
    x[x >= px_mode] = x[x >= px_mode] - px_mode
    dcm.PixelData = x.tobytes()
    dcm.RescaleIntercept = -1000

def window_image(img, window_center, window_width):
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    windowed_img = np.clip(img, img_min, img_max)
    return windowed_img

def normalize_image(image):
    image = cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
    image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return image

def apply_window_level(dcm_path, window_center, window_width):
    dcm = pydicom.dcmread(dcm_path)
    img = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
    windowed_img = window_image(img, window_center, window_width)
    normalized_img = normalize_image(windowed_img)
    # Replicate the grayscale image across 3 channels if necessary
    if normalized_img.ndim == 2 or (normalized_img.ndim == 3 and normalized_img.shape[-1] == 1):
        normalized_img = np.stack((normalized_img,)*3, axis=-1)
    return normalized_img

# Model prediction
def predict_mask(model, image):
    image_expanded = np.expand_dims(image, axis=0)  # Add batch dimension
    predicted_mask = model.predict(image_expanded)[0]  # Predict and take the first batch element
    return predicted_mask

# Define your custom loss function here
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    return bce(y_true, y_pred) + (1 - dice_coef(y_true, y_pred))

def iou(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    sum_ = K.sum(K.square(y_true), axis=[1,2,3]) + K.sum(K.square(y_pred), axis=[1,2,3])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)

# Load the trained model, using the legacy optimizer for M1/M2 Macs
model_path = '/Users/elenamartineztorrijos/Desktop/TFG/Seg_Model_definitivo/segmentation_model_v1.h5'

model = load_model(
    model_path,
    custom_objects={
        "Adam": LegacyAdam,
        "bce_dice_loss": bce_dice_loss,
        "iou": iou,
        "dice_coef": dice_coef
    }
)

# DICOM path
dcm_path = '/Users/elenamartineztorrijos/Desktop/TFG/HSA AI Carlos III/HSA 1/DICOM/ST00001/SE00003/IM00007'

# Window settings (adjust these values as necessary)
window_center = 40  # Example window level for brain windowing
window_width = 80   # Example window width for brain windowing

# Apply window level and normalization
windowed_img = apply_window_level(dcm_path, window_center, window_width)

# Predict mask
predicted_mask = predict_mask(model, windowed_img)

# Threshold the predicted mask
threshold = 0.5
predicted_mask_thresholded = (predicted_mask > threshold).astype(np.float32)

# Display images
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(windowed_img, cmap='gray')
plt.title('Windowed Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(predicted_mask_thresholded, cmap='gray')
plt.title('Predicted Mask')
plt.axis('off')

plt.show()
