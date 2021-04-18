import matplotlib.pyplot as plt
from scipy.io.wavfile import read
import numpy as np
import os, random

def show_random(directory,N):
    """
    Plots random spectograms from the given directory

    @type directory: path as string
    @param directory: directory for the pum id ie. pump/id_02
    @type N: number
    @param directory: Number of columns to plot
    """

    fil_norm=random.choices(os.listdir(directory+"/normal/"),k=N) #change dir name to whatever
    fil_abnorm=random.choices(os.listdir(directory+"/abnormal/"),k=N) #change dir name to whatever

    fig,subs=plt.subplots(2,N)
    for idx,(nfil,afil) in enumerate(zip(fil_norm,fil_abnorm)):
            rate,ab = read(f"{directory}/abnormal/{afil}")
            rate,norm = read(f"{directory}/normal/{nfil}")
            subs[0][idx].specgram(ab[:,0],Fs=rate)
            subs[0][idx].title.set_text("Abnormal")
            subs[1][idx].specgram(norm[:,0],Fs=rate)
            subs[1][idx].title.set_text("Normal")
    plt.show()
