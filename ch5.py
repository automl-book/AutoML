from __future__ import print_function
from pyspark.context import SparkContext
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import LogisticRegression
from pyspark.sql.session import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import LogisticRegression
from  pyspark.ml.classification import LogisticRegressionModel
from flask import jsonify

from flask import Flask, request
import json

app=Flask(__name__)
sc = SparkContext()
spark = SparkSession(sc)
@app.route ("/trainModel", methods=['POST'])
def logisticModel():
    try:

        json_data = request.get_json(force=True)
        data_path = json_data["data_path"]
        model_path= json_data["model_path"]
        data_path = data_path.replace('\\', '/')
        model_path = model_path.replace('\\', '/')

        df = spark.read.options(inferSchema='True', delimiter=',', header='True').csv(data_path)
        cols = df.columns

        stages = []

        label_stringIdx = StringIndexer(inputCol='payment', outputCol='label')
        stages += [label_stringIdx]
        numericCols = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5",
                       "PAY_6", "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
                       "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]
        assemblerInputs = numericCols
        assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
        stages += [assembler]

        pipeline = Pipeline(stages=stages)
        pipelineModel = pipeline.fit(df)
        df = pipelineModel.transform(df)

        selectedCols = ['label', 'features'] + cols
        df = df.select(selectedCols)

        train, test = df.randomSplit([0.7, 0.3], seed=2018)

        lr = LogisticRegression(featuresCol='features', labelCol='label', maxIter=10)
        lrModel = lr.fit(train)
        lrModel.write().overwrite().save(model_path)

        return "Model Created Successfully"
    except Exception as e:
        return {"Error": str(e)}


@app.route ("/predict", methods=['POST'])
def predict():
    try:
        # sc = SparkContext()
        # spark = SparkSession(sc)
        json_data = request.get_json(force=True)
        model_path = json_data["model_path"]
        data_path = json_data["data_path"]

        data_path = data_path.replace('\\', '/')
        model_path = model_path.replace('\\', '/')

        numericCols = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5",
                       "PAY_6", "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
                       "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]

        df = spark.read.options(inferSchema='True', delimiter=',', header='True').csv(data_path)

        stages = []
        label_stringIdx = StringIndexer(inputCol='payment', outputCol='label')
        stages += [label_stringIdx]

        assemblerInputs = numericCols
        assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
        stages += [assembler]

        pipeline = Pipeline(stages=stages)
        pipelineModel = pipeline.fit(df)
        df = pipelineModel.transform(df)
        new_model = LogisticRegressionModel.load(model_path)
        prediction = new_model.transform(df)
        df_pred = prediction.toPandas()

        df_pred = df_pred[["ID","prediction"]]
        list_pred = df_pred.to_dict(orient='records')
        result = {"results": list_pred}


        return jsonify(result)
    except Exception as e:
        return {"Error": str(e)}




if __name__ == "__main__" :
    app.run(port="5000")

