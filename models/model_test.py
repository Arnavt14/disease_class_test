import numpy as np
import cv2
import tensorflow as tf


def load_model():
  model_v = tf.keras.models.load_model(r"C:\\Users\\tyagi\Downloads\Disease_Classification\models\edema_classifier.h5")
  return model_v

model_v = load_model()

def pred():
    img = cv2.imread(r"C:\\Users\\tyagi\HCL PROJ\edema\train\Edema\00006906_025.png")
    resize = tf.image.resize(img, (256, 256))
    yhat = model_v.predict(np.expand_dims(resize / 255, 0))

    if yhat > 0.5:
        print(f'Normal')
    else:
        print(f'Edema')

pred()


