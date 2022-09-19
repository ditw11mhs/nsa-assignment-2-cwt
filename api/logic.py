import numpy as np
import pandas as pd
from numba import jit, njit
from scipy import signal


def data_loader(file_path: str, col_name: str = ["Time", "Data"]):
    output = pd.read_csv(file_path, delimiter="\t", header=None, names=col_name)
    return output


def get_wavelet_function(function_name: str):
    wave_func_dict = {"Complex Morlet": complex_morlet_wavelet}

    return wave_func_dict[function_name]


@njit
def complex_morlet_wavelet(t):
    # f_o = 0.849
    # w_o = 2 * np.pi * f_o
    # power_pi = np.power(np.pi,(-0.25))
    # Precomputation to further optimize calucaltion

    w_o = 5.33442
    power_pi = 0.75113
    pre_compute = power_pi * np.exp((-np.square(t)) * 0.5)
    w_re = pre_compute * np.cos(w_o * t)
    w_im = pre_compute * np.sin(w_o * t)
    return w_re, w_im


@njit
def compute_t_matrix(wavelet_function, t, scale):
    t_matrix = t / scale
    return wavelet_function(t_matrix)


@jit(forceobj=True)
def numba_correlate(input_1, input_2, mode):
    return signal.correlate(input_1, input_2, mode)


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
    cwt_re = numba_correlate(w_re_matrix, data, "valid")
    print("CWT Real Done!")
    cwt_im = numba_correlate(w_im_matrix, data, "valid")
    print("CWT Imaginary Done!")
    cwt = np.sqrt(np.square(cwt_re) + np.square(cwt_im)) / np.sqrt(scale)
    print("Last CWT Done! \n")
    return cwt
