from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging


app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

#Cargar el moddelo de entrenamiento
model = joblib.load('model.pkl')
app.logger.debug("Model loaded successfully.")

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos del formulario
        abdomen = float(request.form['abdomen'])
        antena = float(request.form['antena'])

        # Convertir los datos a un DataFrame
        data_df = pd.DataFrame([[abdomen, antena]], columns=['abdomen', 'antena'])
        app.logger.debug(f"DataFrame creado: {data_df}")

        # Realizar la predicción
        prediction = model.predict(data_df)
        app.logger.debug(f"Predicción: {prediction[0]}")

        # Devolver la predicción como JSON
        return jsonify({'categoria': prediction[0]})
    except Exception as e:
        app.logger.error(f"Error en la predicción: {str(e)}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)