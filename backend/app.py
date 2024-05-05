from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.types import StructType, StructField, FloatType
import typing
from typing import List
import mlflow.pyfunc
import numpy as np
import pandas as pd

app = FastAPI()

# Load the model
spark = SparkSession.builder.appName("Model Serving App").getOrCreate()
# model = PipelineModel.load("model")
model = mlflow.pyfunc.load_model("model")

class PredictionInput(BaseModel):
    data: List[List[float]]


@app.get("/")
async def get():
    return "woo"

@app.post("/predict/")
async def predict(input: PredictionInput):
    # # Expect JSON input containing the new data for prediction
    # data = request.json
    try:
        schema = StructType([
            StructField("sepal_length", FloatType(), True),
            StructField("sepal_width", FloatType(), True),
            StructField("petal_length", FloatType(), True),
            StructField("petal_width", FloatType(), True)
        ])
        #data = np.ndarray([5.1, 3.5, 1.4, 0.2])
        # data_df = spark.createDataFrame([tuple(input.data)], schema=schema)
        data_df = pd.DataFrame([input.data], columns=["sepal_length", "sepal_width", "petal_length", "petal_width"])
        predictions = model.predict(data_df)
        return {"prediction": predictions}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)