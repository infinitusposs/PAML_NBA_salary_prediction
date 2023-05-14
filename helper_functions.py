import numpy as np                      # pip install numpy
import pandas as pd                     # pip install pandas
import streamlit as st                  # pip install streamlit
from sklearn.metrics import recall_score, precision_score, accuracy_score
import plotly.express as px

# All pages


def fetch_dataset():
    """
    This function renders the file uploader that fetches the dataset either from local machine

    Input:
        - page: the string represents which page the uploader will be rendered on
    Output: None
    """
    # Check stored data
    df = None
    data = None
    if 'data' in st.session_state:
        df = st.session_state['data']
    else:
        data = st.file_uploader(
            'Upload a Dataset', type=['csv', 'txt'])

        if (data):
            df = pd.read_csv(data)
    if df is not None:
        st.session_state['data'] = df
    return df

def user_input_features(df, chart_type, x=None, y=None):
    """
    This function renders the feature selection sidebar

    Input:
        - df: pandas dataframe containing dataset
        - chart_type: the type of selected chart
        - x: features
        - y: targets
    Output:
        - dictionary of sidebar filters on features
    """
    side_bar_data = []

    select_columns = []
    if (x is not None):
        select_columns.append(x)
    if (y is not None):
        select_columns.append(y)
    if (x is None and y is None):
        select_columns = list(df.select_dtypes(include='number').columns)

    for idx, feature in enumerate(select_columns):
        try:
            f = st.sidebar.slider(
                str(feature),
                float(df[str(feature)].min()),
                float(df[str(feature)].max()),
                (float(df[str(feature)].min()), float(df[str(feature)].max())),
                key=chart_type+str(idx)
            )
        except Exception as e:
            print(e)
        side_bar_data.append(f)
    return side_bar_data

def visualize_features(df, chart_select, numeric_columns):
    df_copy = df.copy()
    df_copy['salary (million)'] = df_copy['salary'] / 1000000.0
    df_copy['inflationAdjSalary (million)'] = df_copy['salary'] / 1000000.0
    try:
        if chart_select == "Scatterplot":
            x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
            y_values = st.sidebar.selectbox('Y axis', options=numeric_columns)
            if x_values in ['salary', 'inflationAdjSalary']:
                x_values = x_values + " (million)"
            if y_values in ['salary', 'inflationAdjSalary']:
                y_values = y_values + " (million)"
            side_bar_data = user_input_features(df_copy, chart_select, x_values, y_values)
            plot = px.scatter(
                data_frame=df_copy,
                x=x_values,
                y=y_values,
                labels=dict(x=x_values, y=y_values),
                range_x=[side_bar_data[0][0], side_bar_data[0][1]],
                range_y=[side_bar_data[1][0], side_bar_data[1][1]],
                title=chart_select + " of y=" + y_values + " versus x=" + x_values
            )
            st.write(plot)
        elif chart_select == "Histogram":
            x_values = st.sidebar.selectbox('Feature', options=numeric_columns)
            if x_values in ['salary', 'inflationAdjSalary']:
                x_values = x_values + " (million)"
            side_bar_data = user_input_features(df_copy, chart_select, x_values)
            plot = px.histogram(
                df_copy,
                x=x_values,
                range_x=[side_bar_data[0][0], side_bar_data[0][1]],
                title = chart_select + " of " + x_values

            )
            st.write(plot)
        elif chart_select == "Lineplot":
            x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
            y_values = st.sidebar.selectbox('Y axis', options=numeric_columns)
            if x_values in ['salary', 'inflationAdjSalary']:
                x_values = x_values + " (million)"
            if y_values in ['salary', 'inflationAdjSalary']:
                y_values = y_values + " (million)"
            side_bar_data = user_input_features(df_copy, chart_select, x_values, y_values)
            plot = px.line(
                data_frame=df_copy,
                x=x_values,
                y=y_values,
                labels=dict(x=x_values, y=y_values),
                range_x=[side_bar_data[0][0], side_bar_data[0][1]],
                range_y=[side_bar_data[1][0], side_bar_data[1][1]],
                title=chart_select + " of y=" + y_values + " versus x=" + x_values
            )
            st.write(plot)
        elif chart_select == "Boxplot":
            x_values = st.sidebar.selectbox('Feature', options=numeric_columns)
            if x_values in ['salary', 'inflationAdjSalary']:
                x_values = x_values + " (million)"
            side_bar_data = user_input_features(df_copy, chart_select, x_values)
            plot = px.box(
                df_copy[df_copy[x_values] <= side_bar_data[0][1]],
                y=x_values,
                range_y=[side_bar_data[0][0], side_bar_data[0][1]],
                title=chart_select + " of " + x_values
            )
            st.write(plot)


    except Exception as e:
        print(e)
