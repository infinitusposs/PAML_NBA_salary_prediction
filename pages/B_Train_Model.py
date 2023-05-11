import numpy as np                      # pip install numpy
import pandas as pd                     # pip install pandas
import matplotlib.pyplot as plt         # pip install matplotlib
import streamlit as st                  # pip install streamlit
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import Lasso
import random
import plotly.express as px
random.seed(10)

#############################################

st.markdown("# Practical Applications of Machine Learning (PAML)")

#############################################

st.markdown("### Final Project - NBA Salary Prediction")

#############################################

st.title('Train Model')

#############################################
def split_dataset(X, y, number, random_state=42):
    """
    This function splits the dataset into the train data and the test data

    Input:
        - X: training features
        - y: training targets
        - number: the ratio of test samples
    Output:
        - X_train: training features
        - X_val: test/validation features
        - y_train: training targets
        - y_val: test/validation targets
    """
    X_train = []
    X_val = []
    y_train = []
    y_val = []
    try:
        # Split data into train/test split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=number / 100.0, random_state=random_state)
        # Add code here

        train_percentage = len(X_train) / (len(X_train) + len(X_val)) * 100
        test_percentage = len(X_val) / (len(X_train) + len(X_val)) * 100

        # Print dataset split result
        st.markdown(
            'The training dataset contains {0:.2f} observations ({1:.2f}%) and the test dataset contains {2:.2f} observations ({3:.2f}%).'.format(
                len(X_train),
                train_percentage,
                len(X_val),
                test_percentage))
        # Save state of train and test splits in st.session_state
        st.session_state['X_train'] = X_train
        st.session_state['X_val'] = X_val
        st.session_state['y_train'] = y_train
        st.session_state['y_val'] = y_val
    except:
        print('Exception thrown; testing test size to 0')
    return X_train, X_val, y_train, y_val

def train_multiple_regression(model_label, X_train, y_train):
    multi_reg_model=None

    # Train model. Handle errors with try/except statement
    try:
        multi_reg_model = LinearRegression().fit(X_train, y_train)
        st.session_state[model_label] = multi_reg_model
    except Exception as e:
        print(e)
    return multi_reg_model


def train_ridge_regression(model_label, X_train, y_train, ridge_params, ridge_cv_fold):
    ridge_cv = None

    # Train model. Handle errors with try/except statement
    # Add code here
    try:
        ridge_cv = Pipeline([('ridgeCV', GridSearchCV(estimator=Ridge(),
                                                      param_grid=ridge_params, cv=ridge_cv_fold))])
        ridge_cv.fit(X_train, y_train)
        st.session_state[model_label] = ridge_cv
    except Exception as e:
        print(e)

    return ridge_cv


# Checkpoint 8
def train_lasso_regression(model_label, X_train, y_train, lasso_params, lasso_cv_fold):
    lasso_cv = None

    # Train model. Handle errors with try/except statement
    # Add code here
    try:
        lasso_cv = Pipeline([('lassoCV', GridSearchCV(estimator=Lasso(),
                                                      param_grid=lasso_params, cv=lasso_cv_fold))])
        lasso_cv.fit(X_train, y_train)
        st.session_state[model_label] = lasso_cv
    except Exception as e:
        print(e)

    return lasso_cv

def train_polynomial_regression(model_label, X_train, y_train, poly_degree, poly_include_bias):
    poly_reg_model=None

    # Train model. Handle errors with try/except statement
    # Add code here
    try:
        poly_reg_model = Pipeline([('poly', PolynomialFeatures(degree=poly_degree, include_bias=poly_include_bias)),
                                   ('polyReg', LinearRegression())])
        poly_reg_model.fit(X_train, y_train)
        st.session_state[model_label] = poly_reg_model
    except Exception as e:
        print(e)
    return poly_reg_model


def inspect_coefficients(models_trained, inspect_labels):
    """
    This function gets the coefficients of the trained models

    Input:
        - models: all trained models
        - inspect_models: the models to be inspected on
    Output:
        - out_dict: a dicionary contains the coefficients of the selected models, with the following keys:
            - 'Multiple Linear Regression'
            - 'Polynomial Regression'
            - 'Ridge Regression'
            - 'Lasso Regression'
    """

    # Add code here
    for label in inspect_labels:
        model_name = models_trained[label]

        if model_name == 'Multiple Linear Regression':
            model = st.session_state[label]
            st.write(label + " coefficients are " + str(model.coef_.tolist()))

        if model_name == 'Ridge Regression':
            model = st.session_state[label].named_steps['ridgeCV']
            st.write(label + " coefficients are " + str(model.best_estimator_.coef_))

            cv_results = st.session_state[label].named_steps['ridgeCV'].cv_results_
            st.write(label + ": Cross Validation results are")
            st.write(pd.DataFrame(cv_results))

        if model_name == 'Polynomial Regression':
            model = st.session_state[label].named_steps['polyReg']
            st.write(model_name + " coefficients are " + str(model.coef_.tolist()))

        if model_name == 'Lasso Regression':
            model = st.session_state[label].named_steps['lassoCV']
            st.write(label + " coefficients are " + str(model.best_estimator_.coef_))

            cv_results = st.session_state[label].named_steps['lassoCV'].cv_results_
            st.write(label + ": Cross Validation results are")
            st.write(pd.DataFrame(cv_results))

#############################################
df = st.session_state['processed_data'] if "processed_data" in st.session_state else None


if df is not None:
    # Display dataframe as table
    st.dataframe(df)
    # --------------------------------------------------------------------------
    # Select variable to predict
    X = df.loc[:, ~df.columns.isin(['inflationAdjSalary'])]
    Y = df.loc[:, df.columns.isin(['inflationAdjSalary'])]
    st.markdown('### Variable to predict')
    st.markdown("InflationAdjSalary")
    st.dataframe(Y)
    # --------------------------------------------------------------------------
    # Select input features
    st.markdown('### Input features')
    cols = st.columns(3)
    for i, col in enumerate(df.columns[1:]):
        with(cols[i%3]):
            st.markdown(str(i+1) + " - " + col)
    st.dataframe(X)

    # --------------------------------------------------------------------------
    # Split dataset
    st.markdown('### Split dataset into Train/Validation/Test sets')
    st.markdown(
        '#### Enter the percentage of validation/test data to use for training the model')
    number = st.number_input(
        label='Enter size of test set (X%)', min_value=0, max_value=100, value=30, step=1)
    # Compute the percentage of test and training data
    X_train, X_val, y_train, y_val = split_dataset(X, Y, number)
    # --------------------------------------------------------------------------
    # Train models
    st.markdown('### Train models')
    models_dict = {} if 'models_dict' not in st.session_state else st.session_state.models_dict
    models_trained = {} if 'models_trained' not in st.session_state else st.session_state.models_trained
    model_options = ['Multiple Linear Regression', 'Ridge Regression', 'Lasso Regression', 'Polynomial Regression']

    # Collect ML Models of interests
    st.markdown("#### 1. Choose the models that you want to add")
    col1, col2 = st.columns(2)
    with(col1):
        model_select = st.selectbox(
            label='Select regression model for prediction',
            options=model_options,
        )
    with(col2):
        model_label = st.text_input(label="Label your model")
    add_button = st.button(label="Add model", key="add_model" + str(len(models_dict)))
    if add_button and model_label in models_trained:
        st.write("Label " + "<" + model_label + "> has already exist and associated model has been trained. Please use another label." )
    elif add_button and model_select and model_label:
        models_dict[model_label] = model_select
        st.session_state.models_dict= models_dict
    elif add_button:
        st.write("Please select a model and label it")

    if len(models_dict):
        st.markdown("#### 2. Set parameters for your models")

    for label, model in models_dict.items():
        st.markdown('##### ' + label + " (" + model + ")")
        # Multiple Linear Regression
        if (model_options[0] in model):
            if st.button('Train Multiple Linear Regression Model', key="lr-"+label):
                train_multiple_regression(label, X_train, y_train)
                models_trained[label] = model
                st.session_state.models_trained = models_trained

            if label not in models_trained:
                st.write(label + ' is untrained')
            else:
                st.write(label + ' trained')

        # Ridge Regression
        if model_options[1] in model:
            ridge_cv_fold = st.number_input(
                label='Enter the number of folds of the cross validation',
                min_value=1,
                max_value=10,
                value=3,
                step=1,
                key='ridge_cv_fold_numberinput-' + label
            )
            st.write('You set the number of folds to: {}'.format(ridge_cv_fold))

            solvers = ['auto', 'svd', 'cholesky', 'lsqr',
                       'sparse_cg', 'sag', 'saga', 'lbfgs']
            ridge_solvers = st.multiselect(
                label='Select solvers for ridge regression',
                options=solvers,
                default=solvers[0],
                key='ridge_reg_solver_multiselect-' + label
            )
            st.write('You select the following solver(s): {}'.format(ridge_solvers))

            ridge_alphas = st.text_input(
                label='Input a list of alpha values, separated by comma',
                value='1.0,0.5',
                key='ridge_alphas_textinput-' + label
            )
            st.write('You select the following alpha value(s): {}'.format(ridge_alphas))

            ridge_params = {
                'solver': ridge_solvers,
                'alpha': [float(val) for val in ridge_alphas.split(',')]
            }

            if st.button('Train Ridge Regression Model', key="rr-" + label):
                train_ridge_regression(
                    label, X_train, y_train, ridge_params, ridge_cv_fold)
                models_trained[label] = model
                st.session_state.models_trained = models_trained

            if label not in models_trained:
                st.write(label + ' is untrained')
            else:
                st.write(label + ' trained')

        # Lasso Regression
        if model_options[2] in model:
            lasso_cv_fold = st.number_input(
                label='Enter the number of folds of the cross validation',
                min_value=1,
                max_value=10,
                value=3,
                step=1,
                key='lasso_cv_fold_numberinput-' + label
            )
            st.write('You set the number of folds to: {}'.format(lasso_cv_fold))

            lasso_tol = st.text_input(
                label='Input a list of tolerance values, separated by comma',
                value='0.001,0.0001',
                key='lasso_tol_textinput-' + label
            )
            st.write('You select the following tolerance value(s): {}'.format(lasso_tol))

            lasso_alphas = st.text_input(
                label='Input a list of alpha values, separated by comma',
                value='1.0,0.5',
                key='lasso_alphas_textinput-' + label
            )
            st.write('You select the following alpha value(s): {}'.format(lasso_alphas))

            lasso_params = {
                'tol': [float(val) for val in lasso_tol.split(',')],
                'alpha': [float(val) for val in lasso_alphas.split(',')]
            }

            if st.button('Train Lasso Regression Model', key="lasso-"+label):
                train_lasso_regression(
                    label, X_train, y_train, lasso_params, lasso_cv_fold)
                models_trained[label] = model
                st.session_state.models_trained = models_trained

            if label not in models_trained:
                st.write(label + ' is untrained')
            else:
                st.write(label + ' trained')

        # Polynomial Regression
        if model_options[3] in model:
            poly_degree = st.number_input(
                label='Enter the degree of polynomial',
                min_value=0,
                max_value=10,
                value=2,
                step=1,
                key='poly_degree_numberinput-'+label
            )
            st.write('You set the polynomial degree to: {}'.format(poly_degree))

            poly_include_bias = st.checkbox('include bias')
            st.write('You set include_bias to: {}'.format(poly_include_bias))

            if st.button('Train Polynomial Regression Model'):
                train_polynomial_regression(
                    label, X_train, y_train, poly_degree, poly_include_bias)
                models_trained[label] = model
                st.session_state.models_trained = models_trained

            if label not in models_trained:
                st.write(label + ' is untrained')
            else:
                st.write(label + ' trained')
    # --------------------------------------------------------------------------
    # Inspect coefficients
    st.markdown('### Inspect model coefficients')

    inspect_labels = st.multiselect(
        label='Select models to inspect coefficients',
        options=list(models_trained),
        key='inspect_multiselect'
    )
    inspect_coefficients(models_trained, inspect_labels)


    st.write('Continue to Test Model')
