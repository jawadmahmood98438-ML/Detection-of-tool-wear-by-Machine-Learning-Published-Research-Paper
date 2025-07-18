import numpy as np
import matplotlib.pyplot as plt
import pywt

def generate_scalogram(signal, wavelet='morl', save_path='scalogram.png'):
    scales = np.arange(1, 128)
    coefficients, _ = pywt.cwt(signal, scales, wavelet)
    power = np.abs(coefficients) ** 2
    plt.figure()
    plt.imshow(power, extent=[0, 1, 1, 128], cmap='viridis', aspect='auto',
               vmax=abs(power).max(), vmin=-abs(power).max())
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()