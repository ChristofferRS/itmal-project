import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
from sys import exit
from tools.prepro import *
from tools.model import *
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import models
from tensorflow.keras import layers


if __name__=="__main__":
    AUTOTUNE = tf.data.AUTOTUNE
    pprint("Loading Data")
    train,test,val=load_data("data/pump/*/*/*.wav")

    pprint("Reloading Model")
    model=build_model(train)
    print(model.summary())

    test_audio = []
    test_labels = []

    for audio, label in test:
      test_audio.append(audio.numpy())
      test_labels.append(label.numpy())

    test_audio = np.array(test_audio)
    test_labels = np.array(test_labels)
    y_pred = np.argmax(model.predict(test_audio), axis=1)
    y_true = test_labels

    test_acc = sum(y_pred == y_true) / len(y_true)
    print(f'Test set accuracy: {test_acc:.0%}')
