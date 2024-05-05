from pyspark.sql import SparkSession
import requests
from io import StringIO
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import mlflow
import mlflow.spark


def download_data(url):
    response = requests.get(url)
    if response.status_code == 200:
        with open("iris.data", "w") as f:
            f.write(response.text)
    else:
        print("Failed to download the dataset.")
        return False
    return True

# URL of the Iris dataset
data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# Download the data
if not download_data(data_url):
    raise Exception("Data download failed!")

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("Iris Logistic Regression with MLflow") \
    .getOrCreate()

# Load the dataset from local file system
dataset = spark.read.csv("iris.data", inferSchema=True, header=False)
dataset = dataset.withColumnRenamed("_c0", "sepal_length") \
                 .withColumnRenamed("_c1", "sepal_width") \
                 .withColumnRenamed("_c2", "petal_length") \
                 .withColumnRenamed("_c3", "petal_width") \
                 .withColumnRenamed("_c4", "species")

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Continue with the rest of your pipeline setup and MLflow as before


# Index labels, adding metadata to the label column.
labelIndexer = StringIndexer(inputCol="species", outputCol="indexedLabel").fit(dataset)

# Automatically identify categorical features, and index them.
assembler = VectorAssembler(inputCols=["sepal_length", "sepal_width", "petal_length", "petal_width"], 
                            outputCol="features")

# Train a LogisticRegression model.
lr = LogisticRegression(labelCol="indexedLabel", featuresCol="features")

# Chain indexers and logistic regression in a Pipeline
pipeline = Pipeline(stages=[labelIndexer, assembler, lr])

# Start MLflow experiment
# mlflow.set_experiment(experiment_id="2")
with mlflow.start_run():

    # Train model
    model = pipeline.fit(dataset)

    # Make predictions
    predictions = model.transform(dataset)

    # Evaluate the model
    evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Test set accuracy = " + str(accuracy))

    # Log parameters, metrics, and model to MLflow
    mlflow.log_param("regParam", lr.getOrDefault('regParam'))
    mlflow.log_metric("accuracy", accuracy)
    mlflow.spark.log_model(model, "model")
    # model.write().overwrite.save("model")
    # mlflow.spark.log_model(model, "model")

    # End MLflow experiment
    mlflow.end_run()

# Stop Spark session
spark.stop()