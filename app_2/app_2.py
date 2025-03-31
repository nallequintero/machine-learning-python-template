import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from pickle import load

model = load(open('models/depression_rforest_model.pkl', 'rb'))
class_dic = {'0': 'no', '1': 'si'}

st.markdown('<style>description{color:white;}</style>', unsafe_allow_html=True)

st.markdown("<description> A student depression dataset typically contains data aimed at analyzing," +
"understanding, and predicting depression levels among students." +
"It may include features such as demographic information (age, gender),"+
"academic performance (grades, attendance), lifestyle habits (sleep patterns, exercise, social activities)," +
"mental health history, and responses to standardized depression scales. </description>", unsafe_allow_html=True)

st.title("Depression - Model prediction")

gender = st.selectbox("Choose your gender", ["Female", "Male"])
age = st.selectbox("Choose your age:", np.arange(18, 58, 1))
cgpa = st.slider("Choose your cgpa: ", min_value=0.01, max_value=10.0, value=5.0, step=0.01)
have_you_ever_had_suicidal_thoughts_ = st.selectbox("Have you ever had suicidal thoughts?", ["Yes", "No"])
workstudy_hours = st.slider("Choose your workstudy hours: ", min_value=0, max_value=12, value=5, step=1)
family_history_of_mental_illness = st.selectbox("Do you have a family history of mental illness?", ["Yes", "No"])

if st.button("Realizar predicción"):
    try:
        # Mapeo de variables categóricas
        gender_mapping = {'Female': 'Female', 'Male': 'Male'}
        suicidal_mapping = {'Yes': 'Yes', 'No': 'No'}
        family_history_mapping = {'Yes': 'Yes', 'No': 'No'}

        gender_encoded = gender_mapping[gender]
        suicidal_encoded = suicidal_mapping[have_you_ever_had_suicidal_thoughts_]
        family_history_encoded = family_history_mapping[family_history_of_mental_illness]

        # Crea los datos sin procesar en formato DataFrame
        raw_data = pd.DataFrame(
            [[gender_encoded, age, cgpa, suicidal_encoded, workstudy_hours, family_history_encoded]],
            columns=['gender', 'age', 'cgpa', 'have_you_ever_had_suicidal_thoughts_', 'workstudy_hours', 'family_history_of_mental_illness']
        )

        # Aplica el preprocesador
        #processed_data = proccesor.transform(raw_data)  # Asegúrate de que el preprocesador sea compatible con los datos
        #st.write("Datos procesados:", processed_data)  # Depuración opcional

        # Realiza la predicción
        prediction = str(model.predict(raw_data)[0])
        pred_class = class_dic[prediction]

        # Muestra el resultado
        st.success(f"La predicción es: {pred_class}")

    except Exception as e:
        st.error(f"Error durante la predicción: {e}")
