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
    sin_matrix = -np.sin(pre_compute_wo_t).astype("float32")
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
    time = np.array(input_signal["Time"])
    n_data = len(input_signal["Data"])

    with st.spinner("Preparing Variables"):  # Prepare variable to make a 2d matrix of t
        dt = np.mean(time)
        t_dt = np.arange(-n_data, n_data, dt).reshape(1, -1).astype("float32")
        # t = time.copy()
        # t_reverse_negative = t[::-1] * -1
        # t_calc = (
        #     np.concatenate((t_reverse_negative[:-1], t))
        #     .reshape(1, -1)
        #     .astype("float32")
        # )
        scale = np.arange(scale_limit_down, scale_limit_up, scale_resolution).reshape(
            -1, 1
        )
        scale = np.flip(scale, axis=0).astype(
            "float32"
        )  # <- This makes a decreasing scale
    st.success("Preparation Done!")

    # Create a 2d matrix of wavelet using 2d matrix of t as an input
    with st.spinner("Calculating Wavelet Matrix"):
        t_matrix = compute_t_matrix(t_dt, scale)
        w_matrix = wavelet_function(t_matrix)
    st.success("Wavelet Matrix Calculation Done!")

    # Calculate CWT using the correlation of data against each row of the wavelet matrix
    with st.spinner("Calculating CWT Matrix"):
        cwt_matrix = numba_correlate(w_matrix, data, "valid")
    st.success("CWT Matrix Calculation Done!")

    with st.spinner("Calculating Magnitude"):
        cwt = np.sqrt(np.sum(np.square(cwt_matrix), axis=2)) / np.sqrt(scale)
        padding = cwt.shape[1] - n_data
        cwt = cwt[:, :-padding]
    st.success("Last CWT Done!")

    frequency_scale = 0.849 / scale
    print(frequency_scale)
    return (
        cwt[::-1, ::-1],
        frequency_scale.reshape(
            -1,
        ),
        input_signal["Time"],
    )


def compute_cwt_threshold(
    input_signal,
    wavelet_function,
    scale_resolution,
    scale_limit_up,
    scale_limit_down,
    c_con,
    t_start,
    t_end,
):
    f = 1000
    n_start = int(t_start * f)
    n_end = int(t_end * f)
    print(n_start)
    data = np.array(input_signal["Data"]).reshape(1, -1, 1).astype("float16")
    print(data.shape)
    time = np.array(input_signal["Time"])
    n_data = len(input_signal["Data"])

    with st.spinner("Preparing Variables"):  # Prepare variable to make a 2d matrix of t
        dt = np.mean(time)
        t_dt = np.arange(-n_data, n_data, dt).reshape(1, -1).astype("float32")
        scale = np.arange(scale_limit_down, scale_limit_up, scale_resolution).reshape(
            -1, 1
        )
        scale = np.flip(scale, axis=0).astype(
            "float32"
        )  # <- This makes a decreasing scale
    st.success("Preparation Done!")

    # Create a 2d matrix of wavelet using 2d matrix of t as an input
    with st.spinner("Calculating Wavelet Matrix"):
        t_matrix = compute_t_matrix(t_dt, scale)
        w_matrix = wavelet_function(t_matrix)
    st.success("Wavelet Matrix Calculation Done!")

    # Calculate CWT using the correlation of data against each row of the wavelet matrix
    with st.spinner("Calculating CWT Matrix"):
        cwt_matrix = numba_correlate(w_matrix, data, "valid")
    st.success("CWT Matrix Calculation Done!")

    with st.spinner("Calculating Magnitude"):
        cwt = np.sqrt(np.sum(np.square(cwt_matrix), axis=2)) / np.sqrt(scale)
        padding = cwt.shape[1] - n_data
        cwt = cwt[:, :-padding]
    st.success("Last CWT Done!")

    threshold_mask = cwt > (np.min(cwt) + c_con * np.max(cwt))
    thresholded_cwt = cwt * threshold_mask
    frequency_scale = 10000 * 0.849 / scale
    return (
        thresholded_cwt[::-1, n_end:n_start:-1],
        frequency_scale.reshape(
            -1,
        ),
        time[n_start:n_end],
    )


def compute_shannon_envelope(x_df, window_size, c_env):
    # Standardize input dimension
    x_input = np.array(x_df["Data"])

    # Normalized input signal with its absolute maximum value
    x_norm = x_input / np.max(np.absolute(x_input))

    # Precompute Shannon energy
    x_norm_sq = np.square(x_norm)
    shannon_nrg = -x_norm_sq * np.log(x_norm_sq)

    # Apply moving average filter to make an average Shannon energy
    avg_shannon = (
        np.convolve(
            shannon_nrg,
            np.ones(
                window_size,
            ),
            "full",
        )[: -(window_size - 1)]
        / window_size
    )

    # Compute Mean and Standard Deviation of average Shannon energy
    mean_avg_shannon = np.mean(avg_shannon)
    sdev_avg_shannon = np.sqrt(
        np.sum(np.square(avg_shannon - mean_avg_shannon)) / window_size
    )

    # Calculate Shannon Envelope
    shannon_env = (avg_shannon - mean_avg_shannon) / sdev_avg_shannon

    # Threshold Shannon Envelope
    threshold_mask = shannon_env > (np.min(shannon_env) + c_env * np.max(shannon_env))
    shannon_thres = shannon_env * threshold_mask

    output_df = pd.DataFrame(
        data={
            "Data": x_input,
            "Threshold": threshold_mask * np.max(shannon_env),
            "Shannon Envelope (thresholded)": shannon_thres,
            "Shannon Envelope": shannon_env,
            "Time": x_df["Time"],
        }
    )
    return output_df
