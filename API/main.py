from flask import Flask, request, jsonify
import pickle
from xgboost import XGBClassifier
import traceback
import pandas as pd
import numpy as np

# Your API definition
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if model:
        try:
            json_ = request.json
            print(json_)
            query = pd.get_dummies(pd.DataFrame(json_))
            query = query.reindex(columns=model_columns, fill_value=0)

            prediction = list(model.predict(query))

            return jsonify({'prediction': str(prediction)})

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Aún no hay un modelo')
        return ('Aún no hay un modelo')

if __name__ == '__main__':
    with open('modelxgb_con_pesos.pickle', 'rb') as f:
        model = pickle.load(f)
    print ('Modelo cargado')
    with open('columns_modelxgb_con_pesos.pickle', 'rb') as f:
        model_columns = pickle.load(f) 
    print ('Columnas del modelo cargadas')

    app.run(debug=True,port=4002)