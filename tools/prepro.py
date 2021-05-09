import tensorflow as tf
import tensorflow_io as tfio
import os
from librosa import load
import numpy as np

def get_waveform_and_label(file_path):
    """
    Get the waveform and the label for the auido file ie. normal or abnormal

    :param file_path: The path to the file to get the label and waveform for
    :type file_path: str

    :returns: tensorflow Data object
    """
    label = get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    return waveform, label

def get_label(file_path):
    """
    Get the label for the audio file `-2` since it is to steps pu that
    the label apears

    :param file_path: The filepath of the file to get the label for
    :type file_path: str

    """
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[-2]

def decode_audio(audio_binary):
    """
    Function for decoding the audio. Have to use tf_io_nighly to be able to decode
    our wav. This is due to the difference between wav_ext and wav_pcm

    :param audio_binary: the binnary of the audio to decode
    :type audio_binary: A Tensor of type string.

    :returns: Decoded audio.

    """
    audio = tfio.audio.decode_wav(audio_binary,dtype=tf.int16)[0:160000,0]
    return audio

def get_spectrogram(waveform):
    """
    Get the spectrogram for the waveform. Calculates the 
    Short-time Fourier Transform using `tf.signal.stft`

    :pram waveform: The decoded audio waveform

    :returns: Tensor of float32

    """
    waveform = tf.cast(waveform, tf.float32)
    spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=100)
    spectrogram = tf.abs(spectrogram)
    return spectrogram[:,0:20]

def plot_spectrogram(spectrogram, ax):
    """
    Plots the spectogram

    """
    log_spec = np.log(spectrogram.T)
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
    Y = range(height)
    ax.pcolormesh(X, Y, log_spec)

def get_spectrogram_and_label_id(audio, label):
    """
    Get both the spectogram and the label_id (0 or 1 for normal or abnormal).

    :param auido: The decoded audio
    :param label: The label for the audio
    
    """
    spectrogram = get_spectrogram(audio)
    label_id = tf.argmax(label == ["abnormal","normal"])
    return spectrogram, label_id

if __name__=="__main__":
    import matplotlib.pyplot as plt
    print("Testing on file:")
    print("data/pump/id_00/abnormal/00000003.wav")
    print("============")
    wav,lab=get_waveform_and_label("data/pump/id_00/abnormal/00000008.wav")
    spectrogram = get_spectrogram(wav)
    print(spectrogram.shape)
    fig, axes = plt.subplots(2, figsize=(12, 8))
    timescale = np.arange(wav.shape[0])
    plot_spectrogram(spectrogram.numpy(), axes[1])
    axes[1].set_title('Abnormal')
    wav,lab=get_waveform_and_label("data/pump/id_02/normal/00000008.wav")
    spectrogram = get_spectrogram(wav)
    plot_spectrogram(spectrogram.numpy(), axes[0])
    axes[0].set_title('Normal')
    plt.show()





