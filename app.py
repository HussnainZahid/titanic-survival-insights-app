import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet

from utils.data_utils import (
    load_data, clean_data, engineer_features,
    apply_filters, perform_stats, detect_outliers,
    train_ml_model, predict_survival
)

st.set_page_config(
    page_title="Titanic Dashboard",
    layout="wide"
)

@st.cache_data
def get_data():
    df = load_data()
    df = clean_data(df)
    df = engineer_features(df)
    return df

df = get_data()
model, le_sex, model_acc = train_ml_model(df)

st.sidebar.header("Filters")
filters = {
    "Pclass": st.sidebar.multiselect("Pclass", df['Pclass'].unique(), df['Pclass'].unique()),
    "Sex": st.sidebar.multiselect("Sex", df['Sex'].unique(), df['Sex'].unique()),
    "AgeMin": st.sidebar.slider("Age Min", 0, int(df['Age'].max()), 0),
    "AgeMax": st.sidebar.slider("Age Max", 0, int(df['Age'].max()), int(df['Age'].max())),
    "FareMin": st.sidebar.slider("Fare Min", 0.0, float(df['Fare'].max()), 0.0),
    "FareMax": st.sidebar.slider("Fare Max", 0.0, float(df['Fare'].max()), float(df['Fare'].max())),
    "Embarked": st.sidebar.multiselect("Embarked", df['Embarked'].unique(), df['Embarked'].unique()),
    "Title": st.sidebar.multiselect("Title", df['Title'].unique(), df['Title'].unique())
}

filtered = apply_filters(df, filters)

st.title("🛳️ Titanic Data Dashboard")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview","Summary","Visuals","Advanced","Insights"]
)

with tab1:
    st.metric("Survival Rate", f"{filtered['Survived'].mean()*100:.1f}%")
    st.metric("Model Accuracy", f"{model_acc*100:.1f}%")
    st.dataframe(filtered)

with tab2:
    st.write(filtered.describe())
    st.write(filtered.describe(include=['object']))

with tab3:
    fig = px.histogram(filtered, x='Age', color='Survived')
    st.plotly_chart(fig)

with tab4:
    st.write("Outliers Age:", len(detect_outliers(filtered,'Age')))
    corr = filtered.select_dtypes(np.number).corr()
    st.dataframe(corr)

with tab5:
    st.write(perform_stats(filtered))
    pclass = st.selectbox("Pclass", [1,2,3])
    sex = st.selectbox("Sex", ['male','female'])
    age = st.slider("Age",0,int(df['Age'].max()),30)
    fare = st.slider("Fare",0.0,float(df['Fare'].max()),32.0)
    fs = st.slider("Family Size",1,11,1)
    if st.button("Predict"):
        prob = predict_survival(model, le_sex, pclass, sex, age, fare, fs)
        st.success(f"Survival chance: {prob*100:.1f}%")
