import os
from tools.tools import *
import tensorflow as tf
import matplotlib.pyplot as plt

# Prepare the data
dat=tf.data.Dataset.list_files("data/pump/*/*/*.wav")
AUTOTUNE = tf.data.AUTOTUNE
waveform_ds = dat.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
waveform_ds = waveform_ds.map(get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)
for sp,w_id in waveform_ds.take(1):
    fig, axes = plt.subplots(1, figsize=(12, 8))
    print(sp.shape)
    plot_spectrogram(sp.numpy(), axes)
    axes.set_title('Spectrogram')
    plt.show()


#https://www.tensorflow.org/tutorials/audio/simple_audio
optim=tf.keras.optimizers.SGD(learning_rate=0.1)
model.compile(optimizer=optim,loss="MSE")

model.fit(X_train,Y_train,epochs=100)
print()
print("Model Eval:")
model.evaluate(X_test,Y_test,verbose=1)
