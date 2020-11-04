# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 19:02:19 2020

@author: AA
"""

"""Using CT scans to build a classifier to predict presence of viral pneumonia"""
#importing the Libraries
import tensorflow as tf
import keras
import keras.layers
import numpy as np
import zipfile
import os
import sys


#downloading and extracting data 

#unzip the data to the specified directory
"cT-0 is the normal CT scans"
"CT-23 is the abnormal CT scans with pneumonia"

#downloading the normal data
url = "https://github.com/hasibzunair/3D-image-classification-tutorial/releases/download/v0.2/CT-0.zip"
filename = os.path.join(os.getcwd(), "CT-0.zip")
keras.utils.get_file(filename, url)

#downloading the abnormal data
url = "https://github.com/hasibzunair/3D-image-classification-tutorial/releases/download/v0.2/CT-23.zip"
filename2 = os.path.join(os.getcwd(), "CT-23.zip")
keras.utils.get_file(filename2, url)


#directory to store the data
os.chdir('Desktop')
if os.path.isdir('CT scans') is False:
    os.mkdir('CT scans')


with zipfile.ZipFile("CT-0.zip", "r") as normal:
    normal.extractall("Desktop/CT scans/")
    
with zipfile.ZipFile("CT-23.zip", "r") as abnormal:
    abnormal.extractall("Desktop/CT scans/")

os.listdir('Desktop/CT scans')
n = len(os.listdir('Desktop/CT scans/CT-0'))
abn = len(os.listdir('Desktop/CT scans/CT-0'))
print('CT scans with normal lung tissue {}'.format(n))
print('CT scans with abnormal lung tissue {}'.format(abn))



#loading and preprocessing data
import nibabel as nib
from scipy import ndimage


#list of normal scans
normal_scans = []
for root, dirs, files in os.walk("Desktop/CT scans/CT-0"):      
    for img in files:
        normal_scans.append(os.path.join(root, img))

assert len(normal_scans) == 100


#list of abnormal scans
abnormal_scans = []
for root, dirs, files in os.walk("Desktop/CT scans/CT-23"):      
    for img in files:
        abnormal_scans.append(os.path.join(root, img))
        
assert len(abnormal_scans) == 100


#loading nifti files
loaded = []
for i in normal_scans:
    i = nib.load(i)
    i = i.get_fdata()
    loaded.append(i)

loaded_abnormal = []
for i in abnormal_scans:
    i = nib.load(i)
    i = i.get_fdata()
    loaded_abnormal.append(i)



#resizing the 3D volume
def resize_volume(img):
    # Set the desired depth
    desired_depth = 64
    desired_width = 128
    desired_height = 128
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate
    img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img



import random
t = resize_volume(random.choice(loaded))
assert t.shape == (128,128,64)



#resizing the scans
loaded_volume = []
for i in loaded:
    i = resize_volume(i)
    loaded_volume.append(i)


abnormal_volume = []
for i in loaded_abnormal:
    i = resize_volume(i)
    abnormal_volume.append(i)



"Build the training and validation datasets"
#assign labels to the scans
"assigning 0 to normal scan, assigning 1 to abnormal scan"
abnormal_labels = np.array([1 for _ in range(len(abnormal_volume))])
normal_labels = np.array([0 for _ in range(len(loaded_volume))])

#splitting to  tarining and validation sets
x_train = np.concatenate((loaded_volume[:80], abnormal_volume[:80]), axis =0)
y_train = np.concatenate((abnormal_labels[:80], normal_labels[:80]), axis=0)
x_val = np.concatenate((loaded_volume[80:], abnormal_volume[80:]), axis =0)
y_val = np.concatenate((abnormal_labels[80:], normal_labels[80:]), axis=0)

print('number of training scans {} and number of validation scans {}'.format(x_train.shape[0], x_val.shape[0]))

x_train.shape
x_val.shape
y_train.shape
y_val.shape

assert len(y_train) == len(x_train)
assert len(y_val) == len(x_val)



"expand fourth dimensions for training the 3D model"
x_train = np.expand_dims(x_train,axis = -1)
x_val = np.expand_dims(x_val, axis =-1)


"Build the 3D CNN Model"
def get_model(width=128, height=128, depth=64):
    
    inputs = keras.Input((width, height, depth,1))

    x = keras.layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = keras.layers.MaxPool3D(pool_size=2)(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = keras.layers.MaxPool3D(pool_size=2)(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = keras.layers.MaxPool3D(pool_size=2)(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    x = keras.layers.MaxPool3D(pool_size=2)(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.GlobalAveragePooling3D()(x)
    x = keras.layers.Dense(units=512, activation="relu")(x)
    x = keras.layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="3DCNN")
    return model

# Build model.
model = get_model(width=128, height=128, depth=64)
model.summary()




"train the model on the data"
model.compile(loss="binary_crossentropy",optimizer='Adam',metrics=["acc"])

filepath = os.path.join('Desktop','best')
checkpoint_cb = keras.callbacks.ModelCheckpoint(filepath,save_weights_only=True, monitor="val_loss", save_best_only=True, mode ='min')
early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_acc", patience=3)

epochs = 20
history = model.fit(x_train,y_train, validation_data = (x_val,y_val), epochs = epochs,verbose =2, callbacks=[checkpoint_cb, early_stopping_cb])


"or we also can save and load model without checkpoint"
# The model weights (that are considered the best) are loaded into the model.
model.save_weights("Desktop/best/kerasweights.h5")
model.load_weights(filepath)




"visualise the performance"
epoch = range(len(history.history['loss']))
import matplotlib.pyplot as plt
plt.plot(epoch, history.history['loss'], label ='train')
plt.plot(epoch,history.history['val_loss'], label ='validation')
plt.legend()
plt.show()


"make predictions"
from keras.preprocessing import image
img = os.path.join('Desktop/CT scans/CT-23', 'study_0939.nii.gz')
img = nib.load(img)
img = img.get_fdata()
img = resize_volume(img)
img = np.expand_dims(img, axis=0)
img = np.expand_dims(img, axis =-1)
prediction = model.predict(img)


img2 = x_val[0]
prediction2 = model.predict(np.expand_dims(img2, axis =0))














































