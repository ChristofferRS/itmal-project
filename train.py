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

#model.evaluate(X_test,Y_test,verbose=1)
if __name__=="__main__":
    pprint("Loading Data")
    train,test,val_ds=load_data("data/pump/*/*/*.wav")
    pprint("Building Model")
    model=build_model(train)
    print(model.summary())

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    EPOCHS = 20
    stopper=tf.keras.callbacks.EarlyStopping(patience=5)
    history = model.fit(train, validation_data=val_ds, epochs=EPOCHS, callbacks=stopper)
    model.save('saved_model')
    #metrics = history.history
    #hist_dat=np.array([[history.epoch],[metrics['loss']],[metrics['loss']]])
    #np.savetxt("train_data.csv",hist_dat,delimiter=",")

