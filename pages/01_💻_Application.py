import os

import plotly.express as px
import streamlit as st
from plotly import graph_objects as go

from api import logic, utils


def main():
    st.header("App")

    output = logic.data_loader(os.path.join("data", "pcg_1.txt"))
    st.dataframe(output)

    import numpy as np

    inp = np.arange(-100, 100, 0.01)
    w_re, w_im = logic.complex_morlet_wavelet(inp)
    st.line_chart(w_re)

    dummy = np.arange(100)
    cwt = logic.compute_cwt(output, logic.complex_morlet_wavelet, 0.01, 100)
    # fig = go.Figure(data=go.Heatmap(z=cwt, connectgaps=True, zsmooth="best"))
    # fig = go.Figure(data=go.Contour(z=cwt))
    # fig = px.imshow(cwt)
    # st.plotly_chart(fig)
    print(cwt)


if __name__ == "__main__":
    main()
