from flask import Flask, render_template, request
from pickle import load
import pandas as pd

app = Flask(__name__, template_folder='templates')

# Carga el modelo y el preprocesador
with open('models/depression_rforest_model.pkl', 'rb') as file:
    model = load(file)

with open('models/proccesor.pkl', 'rb') as file:
    proccesor = load(file)

class_dic = {'0': 'no', '1': 'si'}

@app.route('/', methods=['GET', 'POST'])
def index():
    pred_class = None
    if request.method == 'POST':
        try:
            # Recoge los datos del formulario
            gender = request.form['gender']
            age = float(request.form['age'])
            cgpa = float(request.form['cgpa'])
            have_you_ever_had_suicidal_thoughts_ = request.form['have_you_ever_had_suicidal_thoughts_']
            workstudy_hours = float(request.form['workstudy_hours'])
            family_history_of_mental_illness = request.form['family_history_of_mental_illness']

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

            print(f"Datos originales: {raw_data}")  # Depuración

            # Aplica el preprocesador
            #processed_data = proccesor.transform(raw_data)
            #print(f"Datos procesados (como NumPy): {processed_data}")  # Depuración

            # Realiza la predicción
            #prediction = str(model.predict(processed_data)[0])
            prediction = str(model.predict(raw_data)[0])
            pred_class = class_dic[prediction]

        except Exception as e:
            print(f"Error durante la predicción: {e}")
            pred_class = "Error en la predicción"

    return render_template("entry.html", prediction=pred_class)