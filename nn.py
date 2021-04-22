import os
from tools.tools import *
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import models
from tensorflow.keras import layers




# Prepare the data
dat=tf.data.Dataset.list_files("data/pump/*/*/*.wav")
AUTOTUNE = tf.data.AUTOTUNE
waveform_ds = dat.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
waveform_ds = waveform_ds.map(get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)


def is_test(x, y):
    return x % 4 == 0

def is_train(x, y):
    return not is_test(x, y)

recover = lambda x,y: y
test_dataset = waveform_ds.enumerate().filter(is_test).map(recover)
train_dataset = waveform_ds.enumerate().filter(is_train).map(recover)

batch_size = 64
print(f"Using batch size: {batch_size}")
train_dataset = train_dataset.batch(batch_size)
test_dataset = test_dataset.batch(batch_size)

for spectrogram, _ in train_dataset.take(1):
  input_shape = spectrogram.shape
print('Input shape:', input_shape)
num_labels = 2

norm_layer = preprocessing.Normalization()
norm_layer.adapt(train_dataset.take(5).map(lambda x, _: x))

model = models.Sequential([
    #layers.Input(shape=input_shape),
    #preprocessing.Resizing(32, 32), 
    #norm_layer,
    layers.Conv2D(32, 3, activation='relu',input_shape=input_shape),
    #layers.Conv2D(64, 3, activation='relu'),
    #layers.MaxPooling2D(),
    #layers.Dropout(0.25),
    #layers.Flatten(),
    #layers.Dense(128, activation='relu'),
    #layers.Dropout(0.5),
    #layers.Dense(num_labels),
])

model.summary()


model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

EPOCHS = 10
history = model.fit(
    train_dataset, 
    validation_data=test_dataset,  
    epochs=EPOCHS,
    callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
)


#
##https://www.tensorflow.org/tutorials/audio/simple_audio
#optim=tf.keras.optimizers.SGD(learning_rate=0.1)
#model.compile(optimizer=optim,loss="MSE")
#
#model.fit(X_train,Y_train,epochs=100)
#print()
#print("Model Eval:")
#model.evaluate(X_test,Y_test,verbose=1)
