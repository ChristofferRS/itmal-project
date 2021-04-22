import tensorflow as tf
import tensorflow_io as tfio
import os
from librosa import load
import numpy as np

def get_waveform_and_label(file_path):
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
    audio = tfio.audio.decode_wav(audio_binary,dtype=tf.int16)[0:1000,0]
    return audio

def get_spectrogram(waveform):
    waveform = tf.cast(waveform, tf.float32)
    spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    return spectrogram

def plot_spectrogram(spectrogram, ax):
    log_spec = np.log(spectrogram.T)
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
    Y = range(height)
    ax.pcolormesh(X, Y, log_spec)

def get_spectrogram_and_label_id(audio, label):
    spectrogram = get_spectrogram(audio)
    label_id = (label == "normal")
    return spectrogram, label_id

if __name__=="__main__":
    import matplotlib.pyplot as plt
    print("Testing on file:")
    print("data/pump/id_00/abnormal/00000003.wav")
    print("============")
    wav,lab=get_waveform_and_label("data/pump/id_00/abnormal/00000003.wav")
    spectrogram = get_spectrogram(wav)
    fig, axes = plt.subplots(2, figsize=(12, 8))
    timescale = np.arange(wav.shape[0])
    axes[0].plot(timescale, wav)
    axes[0].set_title('Waveform')
    axes[0].set_xlim([0, 16000])
    plot_spectrogram(spectrogram.numpy(), axes[1])
    axes[1].set_title('Spectrogram')
    plt.show()





