import os
import json
import pickle

import pandas as pd


def init():
    global model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'automl_best_model.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

def run(data):
    try:
        data = json.loads(data)
        data = data["data"]
        data = pd.DataFrame(data)
        result = model.predict(data)
        # You can return any data type, as long as it is JSON serializable.
        return json.dumps({"result": result.tolist()})
    except Exception as e:
        return json.dumps({"error": str(e)})