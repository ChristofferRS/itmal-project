import tensorflow as tf
from random import random
import numpy as np
from sklearn.model_selection import train_test_split

def gen_data(N,p):
    x=np.array([[random()/2 for _ in range(2)] for i in range(N)])
    y=np.array([i[0] + i[1] for i in x])
    return train_test_split(x,y,test_size=p)



X_train,X_test,Y_train,Y_test=gen_data(200,0.33)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(5,input_dim=2,activation="sigmoid"),
    tf.keras.layers.Dense(1,activation="sigmoid")
    ])

optim=tf.keras.optimizers.SGD(learning_rate=0.1)
model.compile(optimizer=optim,loss="MSE")

model.fit(X_train,Y_train,epochs=100)
print()
print("Model Eval:")
model.evaluate(X_test,Y_test,verbose=1)




