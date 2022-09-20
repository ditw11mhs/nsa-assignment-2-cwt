import plotly.express as px
import streamlit as st
from bokeh.plotting import figure, show
from plotly import graph_objects as go

from api import logic, utils


def main():
    st.header("App")

    file = st.file_uploader("Data", ["txt"])
    if file:
        output = logic.data_loader(file)
        # st.plotly_chart(
        #     px.line(output, x="Time", y="Data", title="Data from " + file.name)
        # )
        p = figure(
            title="Data from " + file.name,
            x_axis_label="Time (s)",
            y_axis_label="Data",
        )
        p.line(output["Time"], output["Data"], line_width=2)
        st.bokeh_chart(p, use_container_width=True)

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
                "Scale Upper Limit", min_value=0.0, max_value=100.0, value=50.0,format = "%f"
            )
            scale_resolution = c3.number_input(
                "Scale Resolution", min_value=0.0, max_value=10.0, value=0.01,format="%f"
            )
            if st.form_submit_button("CWT"):
                cwt = logic.compute_cwt(
                    output,
                    logic.get_wavelet_function(wavelet_function),
                    scale_resolution=scale_resolution,
                    scale_limit_up=scale_limit_up,
                    scale_limit_down=scale_limit_down,
                )
                p = figure(tooltips=[("x", "$x"), ("y", "$y"), ("value", "@image")])
                p.x_range.range_padding = p.y_range.range_padding = 0

                # must give a vector of image data for image parameter
                p.image(
                    image=[cwt],
                    x=0,
                    y=0,
                    dw=10,
                    dh=10,
                    palette="Spectral11",
                    level="image",
                )
                p.grid.grid_line_width = 0.5
                st.bokeh_chart(p, use_container_width=True)


if __name__ == "__main__":
    main()
