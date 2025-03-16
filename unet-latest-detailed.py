import numpy as np
import pandas as pd
import os
import cv2
from glob import glob
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Input, Activation, BatchNormalization, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt
import seaborn as sns
import random
random.seed(3)

# Load images
image_file = sorted(glob('/Users/elenamartineztorrijos/Desktop/TFG/dataset/dataset/images/*.png'))
images = np.array([cv2.resize(cv2.imread(path), (256, 256)) for path in image_file])
mask_file = sorted(glob('/Users/elenamartineztorrijos/Desktop/TFG/dataset/dataset/masks/*.png'))
masks = np.array([cv2.resize(cv2.imread(path), (256, 256)) for path in mask_file])
print(images.shape, masks.shape)

# Prepare the data
def diagnosis(mask_path):
    value = np.max(cv2.imread(mask_path))
    return '1' if value > 0 else '0'

df = pd.DataFrame({"image_path": image_file, "mask_path": mask_file, "diagnosis":[diagnosis(path) for path in mask_file]})
print(len(df))
df.head()

# Display data
def show_data(df, positive=True):
    fig = plt.figure(figsize=(8,8))
    grid = ImageGrid(fig, 111, nrows_ncols=(3, 1), axes_pad=0.5)
    for ax, idx in zip(grid, range(len(df))):
        img = cv2.imread(df.iloc[idx]['image_path'])
        mask = cv2.imread(df.iloc[idx]['mask_path'], 0)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.imshow(mask, cmap='jet', alpha=0.5)  # Overlay mask
        ax.axis('off')
    plt.show()

show_data(df[df['diagnosis']=='1'].sample(3))

# Split data
np.random.seed(42)
perm = np.random.permutation(len(df))
train_idx, val_idx = perm[:int(len(df)*0.8)], perm[int(len(df)*0.8):]
train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

# Data generator
def train_generator(df, image_size=(256,256), batch_size=4, augment=True):
    image_datagen = ImageDataGenerator(rescale=1./255, rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.1, zoom_range=0.1, horizontal_flip=True, fill_mode="nearest") if augment else ImageDataGenerator(rescale=1./255)
    mask_datagen = ImageDataGenerator(rescale=1./255, rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.1, zoom_range=0.1, horizontal_flip=True, fill_mode="nearest") if augment else ImageDataGenerator(rescale=1./255)

    image_generator = image_datagen.flow_from_dataframe(df, x_col="image_path", class_mode=None, color_mode="rgb", target_size=image_size, batch_size=batch_size, seed=42)
    mask_generator = mask_datagen.flow_from_dataframe(df, x_col="mask_path", class_mode=None, color_mode="grayscale", target_size=image_size, batch_size=batch_size, seed=42)

    while True:
        images = image_generator.next()
        masks = mask_generator.next()
        masks[masks > 0.5] = 1
        masks[masks <= 0.5] = 0
        yield images, masks

# U-Net model
def unet(input_size=(256,256,3)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    up1 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv3), conv2], axis=-1)
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(up1)
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4)
    up2 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv4), conv1], axis=-1)
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(up2)
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv5)
    output = Conv2D(1, (1, 1), activation='sigmoid')(conv5)
    model = Model(inputs=[inputs], outputs=[output])
    return model

# Compile model
model = unet()
model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# Train model
train_gen = train_generator(train_df, batch_size=4)  # Reduced batch size
val_gen = train_generator(val_df, batch_size=4, augment=False)  # Consistent batch size for validation
results = model.fit(train_gen, steps_per_epoch=50, epochs=10, validation_data=val_gen, validation_steps=10)  # Reduced number of epochs

model_save_path = '/Users/elenamartineztorrijos/Desktop/unet_model.h5'  # Cambia esto por la ruta real
model.save(model_save_path)

# Plot results
plt.figure(figsize=(8, 4))
plt.subplot(121)
plt.plot(results.history['loss'], label='train_loss')
plt.plot(results.history['val_loss'], label='val_loss')
plt.title('Loss')
plt.legend()
plt.subplot(122)
plt.plot(results.history['accuracy'], label='train_acc')
plt.plot(results.history['val_accuracy'], label='val_acc')
plt.title('Accuracy')
plt.legend()
plt.show()
