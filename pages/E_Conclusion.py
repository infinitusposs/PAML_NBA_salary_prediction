import streamlit as st
import pandas as pd
#############################################

st.markdown("# Practical Applications of Machine Learning (PAML)")

#############################################

st.markdown("### Final Project - NBA Salary Prediction")

#############################################

st.title('Conclusion & Limitations')

eval_df = st.session_state.eval_df if 'eval_df' in st.session_state else None

if eval_df is not None:

    st.markdown('##### 1. The relationship between salaries and performance stats is not linear.')
    st.markdown('A non-linear model such as Random Forest or Deep Learning may give us better results.')
    st.dataframe(eval_df)

    st.markdown('##### 2. Average yearly salary for the next contract rather than for the next season')
    st.markdown('Since NBA players usually sign contracts for multiple years and the salaries during contract period do not change'
                'according to their performance (ignore bonuses and incentives, not included in the datasets), we should predict the salary'
                'for the next contract based on its performance during the current contract.')
    st.markdown('###### Example:')
    st.write("One of the famour low-ball contract: Stephen Curry")
    st.write("")
    df = st.session_state.data
    df_curry = df[df['Player'] == 'Stephen Curry']
    df_curry['avg_PTS'] = round(df_curry['PTS'] / df_curry['G'], 2)
    df_curry['avg_REB'] = round(df_curry['TRB'] / df_curry['G'], 2)
    df_curry['avg_AST'] = round(df_curry['AST'] / df_curry['G'], 2)
    df_curry['Season'] = (df_curry['seasonStartYear'] - 1).astype('string') + '/' + df_curry['seasonStartYear'].astype('string').apply(lambda x: x[-2:])
    df_curry['Salary_Season'] = (df_curry['seasonStartYear']).astype('string') + '/' + (df_curry['seasonStartYear'] + 1).astype('string').apply(lambda x: x[-2:])
    df_curry['Sign_newContract'] = [False, False, True, False, False, False, False, True]
    features = ['Season', 'avg_PTS', 'avg_REB', 'avg_AST', 'Sign_newContract',
                'Salary_Season', 'inflationAdjSalary']
    df_curry = df_curry[features]
    st.write(df_curry)