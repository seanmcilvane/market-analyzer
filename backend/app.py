from flask import Flask, request, jsonify
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
import mlflow.pyfunc

app = Flask(__name__)

# Load the model
spark = SparkSession.builder.appName("Model Serving App").getOrCreate()
# model = PipelineModel.load("model")
model = mlflow.pyfunc.load_model("model")

@app.route('/', methods=['GET'])
def get():
    return "woo"

@app.route('/predict', methods=['POST'])
def predict():
    # Expect JSON input containing the new data for prediction
    data = request.json
    df = spark.createDataFrame([data])  # Convert JSON to DataFrame
    prediction = model.transform(df)
    # Convert prediction to JSON or a simple response
    return jsonify(prediction.select('prediction').collect())

if __name__ == '__main__':
    # app.run(host='172.17.0.1')
    app.run(host='0.0.0.0', port=8080)