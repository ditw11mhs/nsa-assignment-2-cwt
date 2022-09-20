import numpy as np
import pandas as pd
import streamlit as st
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

    w_o = np.float32(5.33442)
    power_pi = np.float32(0.75113)
    pre_compute_pi_e = power_pi * np.exp((-np.square(t)) * 0.5).astype("float32")
    pre_compute_wo_t = w_o * t

    cos_matrix = np.cos(pre_compute_wo_t).astype("float32")
    sin_matrix = np.sin(pre_compute_wo_t).astype("float32")
    cos_sin_matrix = np.stack((cos_matrix, sin_matrix), axis=2)

    w_matrix = np.expand_dims(pre_compute_pi_e, axis=2) * cos_sin_matrix
    return w_matrix


@njit
def compute_t_matrix(t, scale):
    t_matrix = t / scale
    return t_matrix


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
    data = np.array(input_signal["Data"]).reshape(1, -1, 1).astype("float16")
    n_data = len(input_signal["Data"])

    with st.spinner("Preparing Variables"):  # Prepare variable to make a 2d matrix of t
        dt = np.mean(input_signal["Time"])
        t = np.arange(-n_data, n_data, dt).reshape(1, -1).astype("float32")
        scale = np.arange(scale_limit_down, scale_limit_up, scale_resolution).reshape(
            -1, 1
        )
        scale = np.flip(scale, axis=0).astype(
            "float32"
        )  # <- This makes an decreasing scale
    st.success("Preparation Done!")

    # Create a 2d matrix of wavelet using 2d matrix of t as an input
    with st.spinner("Calculating Wavelet Matrix"):
        t_matrix = compute_t_matrix(t, scale)
        w_matrix = wavelet_function(t_matrix)
    st.success("Wavelet Matrix Calculation Done!")

    # Calculate CWT using the correlation of data against each row of the wavelet matrix
    with st.spinner("Calculating CWT Matrix"):
        cwt_matrix = numba_correlate(w_matrix, data, "valid")
    st.success("CWT Matrix Calculation Done!")

    with st.spinner("Calculating Magnitude"):
        cwt = np.sqrt(np.sum(np.square(cwt_matrix), axis=2)) / np.sqrt(scale)
    st.success("Last CWT Done!")
    return cwt[::-1, ::-1]
