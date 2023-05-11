import streamlit as st
import pandas as pd
#############################################

st.markdown("# Practical Applications of Machine Learning (PAML)")

#############################################

st.markdown("### Final Project - NBA Salary Prediction")

#############################################

st.title('Deploy Application')

#############################################
def salary_predict(df):
    model = st.session_state['deploy_model']
    salary = model.predict(df)
    return salary

#############################################

df = None
if 'data' in st.session_state:
    df = st.session_state['data']
else:
    st.write(
        '### The NBA Salary Prediction Application is under construction. Coming to you soon.')

# Deploy App
if df is not None:
    st.markdown('### NBA Salary Prediction')

    st.markdown('#### This app can predict NBA Player salary for next season based on their performance. '
                'All salaries are adjusted for inflation to reflect the value in 2023.')

    st.markdown('#### 1. Choose how you want to enter player stats')
    input_select = st.radio(label="Choose one of the following to enter data",
             options=["Search stats by name and year", "Enter manually"])

    scaler = st.session_state.scaler
    features = st.session_state.features

    if input_select == "Search stats by name and year":
        st.markdown("#### 2. Select Player name and season end year")
        st.markdown("Example: If you want to predict the salary for season 2010-2011, you will choose **2010** for ***season end year***")
        hidden_cols = ["seasonStartYear", "salary", "inflationAdjSalary"]
        col1, col2 = st.columns(2)
        with(col1):
            player_select = st.selectbox(
                label='Select player name',
                options=df['Player'].unique(),
                key=2
            )
        with(col2):
            year_select = st.selectbox(
                label='Select season end year',
                options=df[df['Player'] == player_select]['seasonStartYear']
            )
        if player_select and year_select and st.button(label="Search"):
            st.markdown("The following data is the stats of " + player_select + " in " + str(year_select-1) + "-" + str(year_select) + " season:")
            display_df = df[(df['Player'] == player_select) & (df['seasonStartYear'] == year_select)]
            st.dataframe(display_df.loc[:, ~display_df.columns.isin(hidden_cols)])

            st.markdown("#### 3. Predict " + player_select + "'s salary in " + str(year_select) + "-" + str(year_select+1) + " season")

            salary_pred = salary_predict(scaler.transform(display_df[features]))
            st.markdown("The **predicted** salary in " + str(year_select) + "-" + str(year_select+1) + " season is $" + format(round(salary_pred[0][0]), ",") + ".")
            st.markdown("The **actual** salary in " + str(year_select) + "-" + str(year_select+1) + " season is $" + format(display_df['inflationAdjSalary'].values[0], ",") + ".")

    if input_select == "Enter manually":
        st.markdown("#### 2. Please provide the following data")
        st.markdown("Please refer to [NBA glossary](https://www.basketball-reference.com/about/glossary.html) about the definition of each field")
        input_df = pd.DataFrame([], columns=features, index=["input"])
        cols = st.columns(3)
        for i, col in enumerate(features):
            with(cols[i % 3]):
                input_df.loc["input", col] = st.text_input(col)
        st.dataframe(input_df)
        predict_button = st.button("Predict")
        if predict_button and (input_df == "").sum().sum() > 0:
            st.markdown("###### Please fill all data")
        elif predict_button:
            salary_pred = salary_predict(scaler.transform(input_df[features]))
            st.markdown("The **predicted** salary for the next season is $" + format(round(salary_pred[0][0]), ",") + ".")


