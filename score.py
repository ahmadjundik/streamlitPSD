import joblib
import json
from azureml.core.model import Model
import numpy as np
import os

def init():
    global model
    # Mendapatkan path dari model yang diregistrasi di Azure ML
    model_path = Model.get_model_path('Users/khairurrijal/project-directory/modelxgb.sav')
    # Memuat model menggunakan joblib
    model = joblib.load(model_path)

def run(raw_data):
    try:
        # Mengkonversi data JSON menjadi numpy array
        data = np.array(json.loads(raw_data)['data'])
        # Melakukan prediksi menggunakan model yang telah dimuat
        predictions = model.predict(data)
        # Mengembalikan hasil prediksi sebagai JSON
        return json.dumps(predictions.tolist())
    except Exception as e:
        error = str(e)
        return json.dumps({"error": error})


