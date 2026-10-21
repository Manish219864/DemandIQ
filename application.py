from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

## Route for home page
@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        try:
            data = CustomData(
                store_nbr=int(request.form.get('store_nbr', 44)),
                family=request.form.get('family', 'GROCERY I'),
                date=request.form.get('date', '2017-08-16'),
                onpromotion=int(request.form.get('onpromotion', 0)),
                oil_price=float(request.form.get('oil_price', 50.0)),
                type=request.form.get('type', 'A'),
                city=request.form.get('city', 'Quito'),
                state=request.form.get('state', 'Pichincha'),
                cluster=int(request.form.get('cluster', 5))
            )
            pred_df = data.get_data_as_dataframe()
            print("Received Input DataFrame:\n", pred_df)

            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)
            print("Model Prediction:", results[0])

            formatted_result = f"{round(float(results[0]), 2):,}"
            return render_template('index.html', results=formatted_result)

        except Exception as e:
            print(f"Error during prediction: {e}")
            return render_template('index.html', results=f"Error: {e}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
