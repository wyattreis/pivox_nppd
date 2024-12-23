# -------------------------------------------------------------------------------
# Name          Pivox Data Dashboard - Freeman Tower Snow Depth
# Description:  Streamlit dashboard to visualize the pivox snow depth data at
#               the Freeman Tower outside of Boise, Idaho.
# Author:       Wyatt Reis
#               US Army Corps of Engineers
#               Cold Regions Research and Engineering Laboratory (CRREL)
#               Wyatt.K.Reis@usace.army.mil
# Created:      October 2024
# Updated:      October 2024
# -------------------------------------------------------------------------------
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from io import BytesIO

# -------------------------------------------------------------------------------
# Import and plot Pivox data from Github Repo
pivoxData_url = 'https://raw.githubusercontent.com/wyattreis/pivox-pipeline/main/scan_snowdepth_df.csv'
pivoxData = pd.read_csv(pivoxData_url).drop(columns=['SD_mean', 'SD_std'])
pivoxData['datetime'] = pd.to_datetime(pivoxData['datetime'])

moresData = pd.read_csv(r'C:/Users/RDCRLWKR/Documents/FileCloud/My Files/Active Projects/Snow Working Group/Pivox/Technical/Scripts/pivox-pipeline/mores_daily_2024.csv')
moresData['date'] = pd.to_datetime(moresData['date'])
moresDataFiltered = moresData[(moresData['date'] >= '2023-11-01') & (moresData['date'] <= '2024-06-01')]

# Snow depth timeseries
fig_depth = px.line(pivoxData, x='datetime', y='tif_elev_mean', title='Mean Snow Depth',
              labels={'datetime': '', 'tif_elev_mean': 'Mean Snow Depth (m)'},
              markers=True)

fig_depth.update_layout(
    title_font=dict(size=24, family='Arial'),  # Increase title font size
    xaxis=dict(
        title_font=dict(size=18,  color='black'),  # Set x-axis title color
        tickfont=dict(size=16,  color='black')     # Set x-axis tick labels color
    ),
    yaxis=dict(
        title_font=dict(size=18,  color='black'),  # Set y-axis title color
        tickfont=dict(size=16,  color='black')    # Set y-axis tick labels color
    )   
)

# Snow depth standard deviation timeseries
fig_std = px.line(pivoxData, x='datetime', y='tif_elev_std', title='Snow Depth Standard Deviation',
              labels={'datetime': '', 'tif_elev_std': 'Standard Deviation (m)'},
              markers=True)

fig_std.update_layout(
    title_font=dict(size=24, family='Arial'),  # Increase title font size
    xaxis=dict(
        title_font=dict(size=18,  color='black'),  # Set x-axis title color
        tickfont=dict(size=16,  color='black')     # Set x-axis tick labels color
    ),
    yaxis=dict(
        title_font=dict(size=18,  color='black'),  # Set y-axis title color
        tickfont=dict(size=16,  color='black')    # Set y-axis tick labels color
    )  
)

# Load the data from .npy file
data_dict = np.load(r'C:/Users/RDCRLWKR/Documents/FileCloud/My Files/Active Projects/Snow Working Group/Pivox/Technical/Data/elevation_data_all.npy', allow_pickle=True).item()

# url = "https://raw.githubusercontent.com/wyattreis/pivox-pipeline/main/elevation_data_all.npy"

# # Fetch and load the .npy file
# response = requests.get(url)
# response.raise_for_status()  # Check for HTTP errors
# data_dict = np.load(BytesIO(response.content), allow_pickle=True).item()

filtered_dict = {key: values for key, values in data_dict.items() if key.endswith('_2000') or key.endswith('_2001')}

fig = go.Figure()
# Loop through the dictionary and add a trace for each entry
for category, values in filtered_dict.items():
    fig.add_trace(go.Histogram(
        x=values,
        name=category,
        marker=dict(
            color='lightblue',      # Set the bar color to light blue
            line=dict(
                color='black',      # Set the outline color to black
                width=1             # Set the outline width
            )
        ),
        nbinsx=100,
        visible=False     # Set opacity for better visibility
    ))

fig.data[0].visible = True

# Create buttons for each trace
buttons = []
for i, category in enumerate(filtered_dict.keys()):
    date_str = category[:8]  # Get the date part
    year, month, day = date_str[:4], date_str[4:6], date_str[6:8]
    formatted_date = f"{month}/{day}/{year}"  # Format date as m/d/year

    buttons.append(dict(
        label=formatted_date,
        method='update',
        args=[{'visible': [j == i for j in range(len(filtered_dict))]},  # Toggle visibility
               {'title': f'<b>Snow Depth Histogram for: {formatted_date}</b>'}]  # Update title
    ))

# Add buttons to the layout
fig.update_layout(
    title=f'Snow Depth Histogram for: {list(filtered_dict.keys())[0][:8]}',
    xaxis_title='Snow Depth (m)',
    yaxis_title='Points',
    title_font=dict(size=24, family='Arial'),
    xaxis=dict(
        title_font=dict(size=18,  color='black'),  # Set x-axis title color
        tickfont=dict(size=16,  color='black')     # Set x-axis tick labels color
    ),
    yaxis=dict(
        title_font=dict(size=18,  color='black'),  # Set y-axis title color
        tickfont=dict(size=16,  color='black')    # Set y-axis tick labels color
    ),
    barmode='overlay',
    updatemenus=[{
        'buttons': buttons,
        'direction': 'down',
        'showactive': True,
        'x': 0.0,
        'xanchor': 'left',
        'y': 1.0,
        'yanchor': 'top',
        'font': dict(color='black', size=14)
    }],
    height=400
)

# -------------------------------------------------------------------------------
# Create the figure
fig_comb = go.Figure()
# Add the first line (Mean Snow Depth)
fig_comb.add_trace(go.Scatter(
    x=pivoxData['datetime'],
    y=pivoxData['tif_elev_mean'],
    name='Pivox Mean Snow Depth (m)',
    line=dict(color='#004949'),
    mode='lines+markers',
    yaxis='y'
))

# Add the second line (Snow Depth Standard Deviation)
fig_comb.add_trace(go.Scatter(
    x=pivoxData['datetime'],
    y=pivoxData['tif_elev_std']*100,
    name='Standard Deviation (cm)',
    line=dict(color='#009292'),
    mode='lines+markers',
    yaxis='y2'
))

# Add the Mores Creek Daily snow depth
fig_comb.add_trace(go.Scatter(
    x=moresDataFiltered['date'],
    y=moresDataFiltered['sd']/39.37,
    name='Mores Creek SNOTEL Snow Depth (m)',
    line=dict(color='#DB6D00'),
    mode='lines+markers',
    yaxis='y'
))

# Update the layout to add the second y-axis
fig_comb.update_layout(
    title='<b>Snow Depth Timeseries</b>',
    title_font=dict(size=24, family='Arial'),
    xaxis=dict(
        title='',
        title_font=dict(size=18, color='black'),
        tickfont=dict(size=16, color='black')
    ),
    yaxis=dict(
        title='Snow Depth (m)',
        title_font=dict(size=18, color='#004949'),
        tickfont=dict(size=16, color='black')
    ),
    yaxis2=dict(
        title='Standard Deviation (cm)',
        title_font=dict(size=18, color='#009292'),
        tickfont=dict(size=16, color='black'),
        overlaying='y',  # Overlay this axis on top of the first y-axis
        side='right'     # Display the second y-axis on the right
    ),
    legend=dict(
        x=0.02,
        y=1.1,
        bgcolor='white',
        bordercolor='black',
        borderwidth=2,
        font=dict(size=16)
    ),
    height=550
)

# -------------------------------------------------------------------------------
st.set_page_config(layout="wide")

left_co, cent_co, right_co = st.columns([0.25, 0.5, 0.25])
with cent_co:
    st.title('Freeman Tower Pivox Data Dashboard', anchor=False)

col1, col2, col3 = st.columns([0.55, 0.2, 0.25])
with col1:
    st.plotly_chart(fig_comb)

with col2:
    st.markdown(
        "<h2 style='text-align: left; font-family: Arial; font-size: 24px;'>Pivox Data Table</h2>", 
        unsafe_allow_html=True
    )
    st.dataframe(pivoxData, height=400)

with col3:
    st.markdown(
        "<h2 style='text-align: left; font-family: Arial; font-size: 24px;'>Optical Image: 7/2/2024</h2>", 
        unsafe_allow_html=True
    )
    st.image(r'C:\Users\RDCRLWKR\Documents\FileCloud\My Files\Active Projects\Snow Working Group\Pivox\Technical\Data\raw pyvox\20240702-2000-34.PIVOX1.jpeg')

st.plotly_chart(fig)

