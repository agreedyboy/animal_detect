import streamlit as st
import numpy as np
import plotly.express as px

df = px.data.gapminder()

fig = px.scatter(
    df.query("year==2007"),
    x="gdpPercap",
    y="lifeExp",
    size="pop",
    color="continent",
    hover_name="country",
    log_x=True,
    size_max=60
)

tab1 , tab2 = st.tabs(["Streamlit" , "Plotly"])

with tab1:
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

with tab2:
    st.plotly_chart(fig , theme=None , use_container_width=True)
