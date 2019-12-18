from __future__ import absolute_import, division, print_function, unicode_literals
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
keras = tf.keras
from PIL import Image
import os
import sys
from keras import backend as K
from keras.models import load_model

#tf.debugging.set_log_device_placement(True)
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

label_dict = {"bed": np.array([1, 0, 0, 0, 0, 0, 0]),
              "bird": np.array([0, 1, 0, 0, 0, 0, 0]),
              "happy": np.array([0, 0, 1, 0, 0, 0, 0]),
              "left": np.array([0, 0, 0, 1, 0, 0, 0]),
              "up": np.array([0, 0, 0, 0, 1, 0, 0]),
              "silence": np.array([0, 0, 0, 0, 0, 1, 0]),
              "other": np.array([0, 0, 0, 0, 0, 0, 1])
              }

folders = next(os.walk(r"C:\Users\Harry\Desktop\Data"))[1]

X = []
Y = []

for folder in folders:
    y = label_dict[folder]
    c = 0
    for file in os.listdir(r"C:\Users\Harry\Desktop\Data" + fr"\{folder}"):
        with open(r"C:\Users\Harry\Desktop\Data" + fr"\{folder}" + fr"\{file}", 'rb') as f:
            image = Image.open(f)
            #image = image.resize((224, 224))
            image = np.asarray(image)
            image = image[:,:,:3]
            image = (image/127.5) - 1
            X.append(image)
            Y.append(np.array([y]).T)
            image = None
        c += 1
        if c >= 16:  ########
            break

X = np.array(X)
Y = np.array(Y)

BATCH_SIZE = 1
EPOCHS = 2

print(K.tensorflow_backend._get_available_gpus())

base_model = keras.applications.InceptionV3(include_top=False, input_shape=(480, 640, 3), weights="imagenet")

feature_batch = base_model(np.zeros(shape=(BATCH_SIZE, 480, 640, 3)))
base_model.trainable = False

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)

prediction_layer = keras.layers.Dense(7, activation="softmax")
prediction_batch = prediction_layer(feature_batch_average)

model = tf.keras.Sequential([
  base_model,
  global_average_layer,
  prediction_layer
])

base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
              loss='binary_crossentropy',
              metrics=['accuracy'])

print(model.summary())

history = model.fit(x=X, y=Y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.15)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

model.save("modelboii.h5")

image = Image.open(r"C:\Users\Harry\Desktop\Data\happy\2d82a556_nohash_0.png")
image = image.resize((224, 224))
image = np.asarray(image)
image = image[:,:,:3]
image = (image/127.5) - 1
image = np.array([image])

result = model.predict(image)

print(result)

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()