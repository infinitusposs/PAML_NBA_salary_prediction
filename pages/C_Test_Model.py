import streamlit as st                  # pip install streamlit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
from helper_functions import fetch_dataset
random.seed(10)

#############################################

st.markdown("# Practical Applications of Machine Learning (PAML)")

#############################################

st.markdown("### Final Project - NBA Salary Prediction")

#############################################

st.title('Test Model')

#############################################
def mae(y_true, y_pred):
    mae_score = mean_absolute_error(y_true, y_pred)
    return mae_score

def rmse(y_true, y_pred):
    rmse_score = mean_squared_error(y_true, y_pred, squared=False)
    return rmse_score

def r2(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    return r2

METRICS_MAP = {
    'mean_absolute_error': mae,
    'root_mean_squared_error': rmse,
    'r2_score': r2
}

def evaluate(labels, models, metrics, X_train, y_train, X_val, y_val):
    df_index = []
    data = []
    for i, model in enumerate(models):
        metric_train = []
        metric_val = []
        df_index.append((labels[i], 'Training'))
        df_index.append((labels[i], 'Validation'))
        for m in metrics:
            y_pred = model.predict(X_train)
            metric_train.append(METRICS_MAP[m](y_train, y_pred))
            y_pred = model.predict(X_val)
            metric_val.append(METRICS_MAP[m](y_val, y_pred))
        data.append(metric_train)
        data.append(metric_val)
    multi_index = pd.MultiIndex.from_tuples(df_index, names=['model_label', 'dataset'])
    df = pd.DataFrame(data, index=multi_index, columns=metrics)

    return df

def plot_learning_curve(X_train, X_val, y_train, y_val, trained_model, metrics, model_name):
    """
    This function plots the learning curve. Note that the learning curve is calculated using
    increasing sizes of the training samples
    Input:
        - X_train: training features
        - X_val: validation/test features
        - y_train: training targets
        - y_val: validation/test targets
        - trained_model: the trained model to be calculated learning curve on
        - metrics: a list of metrics to be computed
        - model_name: the name of the model being checked
    Output:
        - fig: the plotted figure
        - df: a dataframe containing the train and validation errors, with the following keys:
            - df[metric_fn.__name__ + " Training Set"] = train_errors
            - df[metric_fn.__name__ + " Validation Set"] = val_errors
    """
    fig = make_subplots(rows=len(metrics), cols=1,
                        shared_xaxes=True, vertical_spacing=0.1)
    df = pd.DataFrame()

    # Add code here
    train_errors = {}
    val_errors = {}
    batch_size = 500
    train_sizes = []
    for i in range(batch_size, len(X_train)+1, batch_size):
        train_sizes.append(i)
        trained_model.fit(X_train[:i], y_train[:i])
        y_train_pred = trained_model.predict(X_train[:i])
        y_val_pred = trained_model.predict(X_val)

        for m in metrics:
            train_error = METRICS_MAP[m](y_train[:i], y_train_pred)
            val_error = METRICS_MAP[m](y_val, y_val_pred)
            if m not in train_errors:
                train_errors[m] = [train_error]
            else:
                train_errors[m].append(train_error)
            if m not in val_errors:
                val_errors[m] = [val_error]
            else:
                val_errors[m].append(val_error)

    for i, m in enumerate(metrics):
        df[METRICS_MAP[m].__name__ + " Training Set"] = train_errors[m]
        df[METRICS_MAP[m].__name__ + " Validation Set"] = val_errors[m]
        fig.add_trace(go.Scatter(x=list(train_sizes), y=train_errors[m], name="Train", mode="lines"), row=i+1, col=1)
        fig.add_trace(go.Scatter(x=list(train_sizes), y=val_errors[m], name="Val", mode="lines"), row=i+1, col=1)

        fig.update_yaxes(title_text=METRICS_MAP[m].__name__.upper(), row=i+1, col=1)
    fig.update_layout(height=600, width=800, title_text="Model: " + model_name + " (" + st.session_state.models_trained[model_name] + ")")
    fig.update_xaxes(title_text="Training Set Size", row=len(metrics), col=1)
    return fig, df



#############################################
df = st.session_state['processed_data'] if "processed_data" in st.session_state else None
trained_models = list(st.session_state.models_trained)if "models_trained" in st.session_state else None

if df is not None and trained_models is not None:
    st.markdown("### 1. Choose models")
    X_train = st.session_state.X_train
    y_train = st.session_state.y_train
    X_val = st.session_state.X_val
    y_val = st.session_state.y_val
    metric_options = ['mean_absolute_error', 'root_mean_squared_error', 'r2_score']



    # Select a trained classification model for evaluation
    labels_select = st.multiselect(
        label='Select trained models for evaluation',
        options=trained_models
    )

    if (labels_select):
        st.write(
            'You selected the following models for evaluation: {}'.format(labels_select))
        st.markdown('### 2. Choose metrics')

        metric_select = []
        cols = st.columns(len(metric_options))
        for i, col in enumerate(cols):
            with(col):
                if st.checkbox(label=metric_options[i]):
                    metric_select.append(metric_options[i])


        models = []
        if metric_select:
            st.markdown('### 3. Evaluation Result')
            st.markdown('#### Evaluation Table')
            for label in labels_select:
                models.append(st.session_state[label])
            eval_df = evaluate(labels_select, models, metric_select, X_train, y_train, X_val, y_val)
            st.session_state['eval_df'] = eval_df
            st.dataframe(eval_df)

            st.markdown('#### Learning curves')
            for i in range(len(models)):
                fig, df = plot_learning_curve(X_train, X_val, y_train, y_val, models[i], metric_select, labels_select[i])
                st.plotly_chart(fig)



    # --------------------------------------------------------------------------
    # Select a model to deploy from the trained models
    st.markdown("### Choose your Deployment Model")
    model_select = st.selectbox(
        label='Select the model you want to deploy',
        options=trained_models,
    )

    if (model_select):
        st.write('You selected the model: {}'.format(model_select))
        st.session_state['deploy_model'] = st.session_state[model_select]

    st.write('Continue to Deploy Model')
