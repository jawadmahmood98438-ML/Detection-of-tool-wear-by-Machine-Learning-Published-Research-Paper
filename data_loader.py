import os
import numpy as np

def load_force_signal(file_path):
    return np.loadtxt(file_path, delimiter=',')

def load_all_signals(directory):
    data = []
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            data.append(load_force_signal(os.path.join(directory, filename)))
    return data