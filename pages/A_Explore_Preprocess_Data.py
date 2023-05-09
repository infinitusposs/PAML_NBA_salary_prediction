import streamlit as st                  # pip install streamlit
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from helper_functions import fetch_dataset

#############################################

st.markdown("# Practical Applications of Machine Learning (PAML)")

#############################################

st.markdown("### Final Project - NBA Salary Prediction")

#############################################

st.markdown('# Explore & Preprocess Dataset')

#############################################
df = None
df = fetch_dataset()

if df is not None:
    # Display original dataframe
    st.markdown('View initial data with missing values or invalid inputs')
    st.markdown('You have uploaded the dataset.')
    # df.drop(df.columns[0], axis=1, inplace=True)
    st.dataframe(df)

    # Inspect the dataset
    st.markdown('### Inspect and visualize some interesting features')

    # Deal with missing values
    st.markdown('### Handle missing values')
    
    select_features_miss = st.multiselect(
            'Select specify features for checking missing values or Check the box to select all features',
            options = df.columns
    )
    all_features = st.checkbox("Select all")

    def drop_missing_values(df, col):
        df.dropna(subset=[col], inplace=True)
        return df
    
    if all_features:
        for column in df.columns:
            num = df[column].isnull().sum()
            if num > 0:
                st.markdown('Drop '+str(num) +' missing values under '+column)
                drop_missing_values(df, column)
        st.markdown('Finish Handling missing values')
    else:
        for column in select_features_miss:
            num = df[column].isnull().sum()
            if num > 0:
                st.markdown('Drop '+str(num) +' missing values under '+column)
                drop_missing_values(df, column)
        st.markdown('Finish Handling missing values')

    # Handle Text and Categorical Attributes
    st.markdown('### Handling Non-numerical Features')
    non_numeric_columns = df.select_dtypes(exclude='number').columns.tolist()
    select_features_cater = st.multiselect(
            'Select specify features to encoding into numerical features',
            options = non_numeric_columns
    )
    for feature in select_features_cater:
        df[feature] = df[feature].astype('category')
        name = feature+'_code'
        df[name] = df[feature].cat.codes

    # Correlation Heatmap
    st.markdown('### Correlation Heatmap')
    df_numeric = df.select_dtypes(include='number')
    corr = df_numeric.corr()
    inflationAdjSalary_corr = corr['inflationAdjSalary'].sort_values(ascending=False)
    number = st.number_input('Display the top features having the correlation with inflationAdjSalary',
                             min_value=2, max_value=15, value=11, step=1)
    st.dataframe(inflationAdjSalary_corr[:number])

    fig, ax = plt.subplots(figsize=(12,10))
    features = inflationAdjSalary_corr.index[:number]
    df_select = df[features]
    corr = df_select.corr()

    # Generate a heatmap of the correlation matrix
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    plt.title('Correlation Heatmap')
    st.write(fig)

    # Some feature selections/engineerings here
    st.markdown('### Select Relevant/Useful Features')

    col1, col2 = st.columns(2)

    with(col1):
            # Choose yourself
        select_features = st.multiselect(
                'Select specify features for further analysis',
                options = df.select_dtypes(include='number').columns.tolist()
        )
    with(col2):
        # Choose from the top corrlation features
        number = st.number_input('Select the top corrlation features for further analysis',
                                min_value=2, max_value=15, value=11, step=1)
        features_final = inflationAdjSalary_corr.index[:number]

    if select_features:
        df = df[select_features]
    else:
        df = df[features_final]
    
    # Remove outliers
    st.markdown('### Remove outliers')

    def remove_outliers(df, column_name):
        """
        Remove outliers from a given column in a pandas DataFrame.

        Parameters:
        df (pandas.DataFrame): the input DataFrame
        column_name (str): the name of the column to remove outliers from

        Returns:
        pandas.DataFrame: the DataFrame with outliers removed
        """
        # Calculate the first and third quartiles
        q1 = df[column_name].quantile(0.25)
        q3 = df[column_name].quantile(0.75)

        # Calculate the interquartile range
        iqr = q3 - q1

        # Define the outlier range
        outlier_range = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)

        # Remove outliers from the specified column
        df = df[(df[column_name] >= outlier_range[0]) & (df[column_name] <= outlier_range[1])]

        return df
    
    select_features_out = st.multiselect(
            'Select specify features for removing outliers or Check the box to select all features',
            options = df.columns
    )
    all_features_out = st.checkbox("Select all features")

    if all_features_out:
        for column in df.columns:
            df = remove_outliers(df, column)
    else:
        for column in select_features_out:
            df = remove_outliers(df, column)

    # Normalize your data if needed
    st.markdown('### Normalize data')

    MinMaxScalerName, StandardScalerName = st.columns(2)
    X = df.loc[:, ~df.columns.isin(['inflationAdjSalary'])]
    with(MinMaxScalerName):
        if st.button('MinMax Scaler'):
            scaler = MinMaxScaler()
            X = scaler.fit_transform(X)
            st.markdown('Finish Normalize')
    with(StandardScalerName):
        if st.button('Standard Scaler'):
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            st.markdown('Finish Normalize')

    st.markdown('### You have preprocessed the dataset.')
    st.dataframe(df)

    st.write('Continue to Train Model')
