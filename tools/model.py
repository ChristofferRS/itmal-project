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

AUTOTUNE = tf.data.AUTOTUNE
def preprocess_dataset(files):
    """
    Process the data. Expand the axis, gets the id (normal/abnormal)
    and then return the dataset

    """
    files_ds = tf.data.Dataset.from_tensor_slices(files)
    output_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
    output_ds = output_ds.map(get_spectrogram_and_label_id,  num_parallel_calls=AUTOTUNE)
    output_ds = output_ds.map(lambda x,y: (tf.expand_dims(x,axis=-1),y),  num_parallel_calls=AUTOTUNE)
    return output_ds

def load_data(glob):
    """
    Load the data in the data dir downloaded using `make`
    Data is loaded into a tensorflow Data object for improved peformance after
    a train test split has been made

    :param glob: The glob for all the files to load
    :type glob: str

    """
    filenames = tf.io.gfile.glob(glob)
    filenames = tf.random.shuffle(filenames)

# Train Test Split
    train_files = filenames[:3000]
    test_files = filenames[-800:]


    train_ds = preprocess_dataset(train_files)
    test_ds = preprocess_dataset(test_files)

    batch_size = 60
    train_ds = train_ds.batch(batch_size)

    return train_ds,test_ds

def build_model(train_ds):
    """
    Build the ML model. Sets up the desired layers and 
    compiles the tf.keres model

    :param train_ds: The training dataset used for nomalisation and determining the input_shape

    """
    

    # Get the input shape for the model
    for spectrogram, _ in train_ds.take(1):
        input_shape = spectrogram.shape[1:]
    print(f'Input shape: {input_shape}')

    # Normalisation Layer
    norm_layer = preprocessing.Normalization()
    norm_layer.adapt(train_ds.take(30).map(lambda x, _: x))
    
    #Model layout
    model = models.Sequential([
        layers.InputLayer(input_shape=input_shape),
        #preprocessing.Resizing(32, 32), 
        norm_layer,
        layers.Conv2D(60, 3, activation='relu'),
        layers.Conv2D(60, 3, activation='relu'),
        layers.Flatten(),
        layers.Dense(60, activation='relu'),
        #layers.Dropout(0.5),
        layers.Dense(1),
    ])
    return model



def pprint(text):
    """
    Pretty print

    :param text: Text to print
    :type text: str

    """
    print(text.center(20," ").center(100,"="))
