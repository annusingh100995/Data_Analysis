import streamlit as st
import pandas as pd 
import numpy as np 
import plotly_express as px

# cache is used to store the df so that data is not loaded
# everytime a value changes
df = st.cache(pd.read_csv)("football_data.csv")

# selecting the  clubs of the players
clubs = st.sidebar.multiselect('Show Players for clubs?', df['Club'].unique())

# selecting the nationalities of the players
nationalities = st.sidebar.multiselect('Show Players of Nationality', df['Nationality'].unique())

# creating a new df with the selected players
new_df = df[(df['Club'].isin(clubs)) & (df['Nationality'].isin(nationalities))]

st.write(new_df)

fig = px.scatter(new_df, x ='Overall', y='Age', color='Name')
# plotting 
st.plotly_chart(fig)
