#Lib for data manipulation ie. loading and stuff.
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
import numpy as np
from librosa.feature import mfcc
from librosa import load

class sound:
    def __init__(self,path):
        self.path=path
        self.data,self.sample_rate=self.__load()
    def __load(self):
        return load(self.path)
    def draw(self):
        plt.subplot(221)
        plt.specgram(self.data,Fs=self.sample_rate)
        plt.subplot(222)
        plt.plot(self.freq,self.magnitude)
        plt.subplot(223)
        plt.specgram(self.mfcc)
        plt.show()
    def calc_fft(self):
        fft = np.fft.fft(self.data)
        self.magnitude=np.abs(fft)
        self.freq = np.linspace(0,self.sample_rate,len(self.magnitude))
        self.freq=self.freq[:int(len(self.magnitude)/2)]
        self.magnitude=self.magnitude[:int(len(self.magnitude)/2)]


    def calc_mfcc(self):
        self.mfcc=mfcc(self.data,n_fft=2048,hop_length=512,n_mfcc=13)

# For testing the class
if __name__=="__main__":
    S=sound('data/pump/id_00/abnormal/00000000.wav')
    S.calc_fft()
    S.calc_mfcc()
    S=sound('data/pump/id_00/normal/00000000.wav')
    S.calc_fft()
    S.calc_mfcc()
    

