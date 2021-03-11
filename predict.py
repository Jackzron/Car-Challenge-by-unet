from keras_applications.densenet import preprocess_input
import matplotlib.pyplot as plt
from keras_preprocessing import image
import numpy as np
from tensorflow_core.python.keras.models import load_model
import matplotlib.pyplot as pyplot

img_path = r"E:\train_data\carChallenge\train_hq\00087a6bd4dc_01.jpg"
img = image.load_img(img_path, target_size=(512, 512, 3))
x = image.img_to_array(img)
# print(x.shape)
x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)
model = load_model('modelWithWeight.h5')
preds = model.predict(x)
# print(preds)
print(preds.shape)
preds = preds.reshape(512, 512)
print(preds.shape)
preds = np.multiply(preds, 255.0)
print(preds.shape)
# preds = np.squeeze(preds)
# pyplot.imshow(preds)
# pyplot.show()
# print(preds)

import matplotlib

matplotlib.image.imsave('out.jpg', preds)