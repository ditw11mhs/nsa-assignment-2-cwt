import numpy as np
import pandas as pd
from numba import jit
from scipy import signal


def data_loader(file_path: str, col_name: str = ["Time", "Data"]):
    output = pd.read_csv(file_path, delimiter="\t", header=None, names=col_name)
    return output


@jit
def complex_morlet_wavelet(t):
    f_o = 0.849
    w_o = 2 * np.pi * f_o

    w_re = np.power(np.pi, (-1 / 4)) * np.exp((-(t**2)) / 2) * np.cos(w_o * t)
    w_im = np.power(np.pi, (-1 / 4)) * np.exp((-(t**2)) / 2) * np.sin(w_o * t)
    return w_re, w_im


@jit
def compute_t_matrix(wavelet_function, t, scale):
    t_matrix = t / scale
    return wavelet_function(t_matrix)


def compute_cwt(
    input_signal,
    wavelet_function,
    scale_resolution,
    scale_limit_up,
    scale_limit_down=1e-12,
):
    data = np.array(input_signal["Data"]).reshape(1, -1)
    n_data = len(input_signal["Data"])

    # Prepare variable to make a 2d matrix of t
    dt = np.mean(input_signal["Time"])
    t = np.arange(-n_data, n_data, dt).reshape(1, -1)
    scale = np.arange(scale_limit_down, scale_limit_up, scale_resolution).reshape(-1, 1)
    scale = np.flip(scale, axis=0)  # <- This makes an decreasing scale
    print("Preparation Done!")
    
    # Create a 2d matrix of wavelet using 2d matrix of t as an input
    w_re_matrix, w_im_matrix = compute_t_matrix(wavelet_function, t, scale)
    print("Wavelet Matrix Done!")
    
    # Calculate CWT using the correlation of data against each row of the wavelet matrix
    cwt_re = signal.correlate(w_re_matrix, data, "valid")
    print("CWT Real Done!")
    cwt_im = signal.correlate(w_im_matrix, data, "valid")
    print("CWT Imaginary Done!")
    cwt = np.sqrt(np.square(cwt_re) + np.square(cwt_im)) / np.sqrt(scale)
    print("Last CWT Done! \n")
    return cwt
