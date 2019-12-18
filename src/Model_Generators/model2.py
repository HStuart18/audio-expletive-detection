import tensorflow as tf
import numpy as np
from PIL import Image

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

image = Image.open(r"C:\Users\Harry\Desktop\Data\happy\2d82a556_nohash_0.png")
image = np.asarray(image)
image = image[:,:,:3]
image = (image/127.5) - 1
image = np.array([image])

model = tf.keras.models.load_model(r"C:\Users\Harry\source\repos\HStuart18\audio-expletive-detection\modelboii.h5")

result = model.predict(image)

print(result)