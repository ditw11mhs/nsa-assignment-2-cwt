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

        with st.form("cwt_state"):
            wavelet_function = st.selectbox("Wavelet Function", ["Complex Morlet"])
            c1, c2, c3 = st.columns(3)
            scale_limit_down = c1.number_input(
                "Scale Lower Limit",
                min_value=0.0,
                max_value=100.0,
                value=0.00001,
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



if __name__ == "__main__":
    main()
