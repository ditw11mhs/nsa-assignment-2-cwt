import plotly.express as px
import streamlit as st
from plotly import graph_objects as go

from api import logic, utils


def main():
    st.header("App")

    file = st.file_uploader("Data", ["txt"])
    if file:
        output = logic.data_loader(file)
        st.plotly_chart(
            px.line(output, x="Time", y="Data", title="Data from " + file.name)
        )
        with st.expander("CWT"):
            with st.form("cwt_state"):
                wavelet_function = st.selectbox("Wavelet Function", ["Complex Morlet"])
                c1, c2, c3 = st.columns(3)
                scale_limit_down = c1.number_input(
                    "Scale Lower Limit",
                    min_value=0.0,
                    max_value=100.0,
                    value=1e-12,
                    format="%f",
                )
                scale_limit_up = c2.number_input(
                    "Scale Upper Limit",
                    min_value=0.0,
                    max_value=100.0,
                    value=50.0,
                    format="%f",
                )
                scale_resolution = c3.number_input(
                    "Scale Resolution",
                    min_value=0.0,
                    max_value=10.0,
                    value=0.01,
                    format="%f",
                )
                if st.form_submit_button("CWT"):
                    cwt, scale, time = logic.compute_cwt(
                        output,
                        logic.get_wavelet_function(wavelet_function),
                        scale_resolution=scale_resolution,
                        scale_limit_up=scale_limit_up,
                        scale_limit_down=scale_limit_down,
                    )
                    fig_heatmap = go.Figure(data=go.Contour(x=time, z=cwt))
                    st.plotly_chart(fig_heatmap)
                    fig_3d = go.Figure(data=go.Surface(x=time, z=cwt))
                    fig_3d.update_traces(
                        contours_z=dict(show=True, usecolormap=True, project_z=True)
                    )
                    st.plotly_chart(fig_3d)

        with st.expander("Shannon Envelope"):
            with st.form("Shannon Envelope"):
                window_size = st.number_input("Window Size", min_value=1, value=20)
                c_env = st.number_input(
                    "c env", min_value=0.01, max_value=0.5, format="%f"
                )
                if st.form_submit_button("Shannon Envelope"):
                    shannon_envelope_df = logic.compute_shannon_envelope(
                        output, window_size, c_env
                    )
                    st.line_chart(
                        data=shannon_envelope_df,
                        x="Time",
                    )
        with st.expander("CWT Threshold"):
            with st.form("cwt_state_threshold"):
                wavelet_function = st.selectbox("Wavelet Function", ["Complex Morlet"])
                c1, c2, c3 = st.columns(3)
                c4, c5, c6 = st.columns(3)
                scale_limit_down = c1.number_input(
                    "Scale Lower Limit",
                    min_value=0.0,
                    max_value=100.0,
                    value=1e-12,
                    format="%f",
                )
                scale_limit_up = c2.number_input(
                    "Scale Upper Limit",
                    min_value=0.0,
                    max_value=100.0,
                    value=50.0,
                    format="%f",
                )
                scale_resolution = c3.number_input(
                    "Scale Resolution",
                    min_value=0.0,
                    max_value=10.0,
                    value=0.01,
                    format="%f",
                )
                time_start = c4.number_input("Time Start", value=0.01, format="%f")
                time_end = c5.number_input("Time End", value=2.00, format="%f")
                c_con = c6.number_input(
                    "c con",
                    min_value=0.01,
                    max_value=0.99,
                    value=0.01,
                    format="%f",
                )

                if st.form_submit_button("CWT"):
                    cwt, scale, time = logic.compute_cwt_threshold(
                        output,
                        logic.get_wavelet_function(wavelet_function),
                        scale_resolution=scale_resolution,
                        scale_limit_up=scale_limit_up,
                        scale_limit_down=scale_limit_down,
                        c_con=c_con,
                        t_start=time_start,
                        t_end=time_end,
                    )
                    fig_heatmap = go.Figure(data=go.Contour(x=time, z=cwt))
                    st.plotly_chart(fig_heatmap)
                    # fig_3d = go.Figure(data=go.Surface(x=time, z=cwt))
                    # fig_3d.update_traces(
                    #     contours_z=dict(show=True, usecolormap=True, project_z=True)
                    # )
                    # st.plotly_chart(fig_3d)


if __name__ == "__main__":
    main()
