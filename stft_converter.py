import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

def generate_spectrogram(signal, fs=1000, save_path='spectrogram.png'):
    f, t, Sxx = spectrogram(signal, fs)
    plt.figure()
    plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()