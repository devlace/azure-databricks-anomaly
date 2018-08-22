// Databricks notebook source
// MAGIC %md
// MAGIC ## Setup

// COMMAND ----------

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature._
import org.apache.spark.sql.functions._
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator, BinaryClassificationEvaluator}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}

// Model Directory
val modelDir = "/mnt/blob_storage/models"
val randomSeed = 123

// COMMAND ----------

// MAGIC %md
// MAGIC ## Load and transform data

// COMMAND ----------

// Read data
spark.catalog.refreshTable("kdd") // need to refresh to invalidate cache
val df = spark.read.table("kdd")

// Clean data
val cleanDf = df
  .withColumn("is_anomaly", when(col("label") === "normal.", 0).otherwise(1))
  .na.drop()

// Clean up labels for anomaly
display(cleanDf)

val columns = cleanDf.columns.toSet
val features = columns -- Set("id", "label", "is_anomaly")
val categoricalFeatures = Set("protocol_type", "service", "flag")
val continuousFeatures = features -- categoricalFeatures

// Split
val Array(training, test) = cleanDf.randomSplit(Array(0.8, 0.2), seed = 123)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Define Feature Estimators and Transformers

// COMMAND ----------

// Label indexer
val labelIndexer = new StringIndexer()
  .setInputCol("label")
  .setOutputCol("label_index")
val labelIndexerModel = labelIndexer.fit(cleanDf)

// Categorical Feature Indexers
val indexers = categoricalFeatures.map({ colName =>
  new StringIndexer().setInputCol(colName).setOutputCol(colName + "_index").setHandleInvalid("keep")
}).toArray

// Encoders
val encoder = new OneHotEncoderEstimator()
  .setInputCols(categoricalFeatures.map(colName => colName + "_index").toArray)
  .setOutputCols(categoricalFeatures.map(colName => colName + "_encoded").toArray)

// Vector Assembler
var selectedFeatures = continuousFeatures ++ categoricalFeatures.map(colName => colName + "_encoded") 
val assembler = new VectorAssembler()
  .setInputCols(selectedFeatures.toArray)
  .setOutputCol("features")

// Standard Scalar
val standardScalar = new StandardScaler()
  .setInputCol("features")
  .setOutputCol("norm_features")
  .setWithMean(true)
  .setWithStd(true)

// Convert indexed labels back to original labels.
val labelConverter = new IndexToString()
  .setInputCol("prediction")
  .setOutputCol("predicted_label")
  .setLabels(labelIndexerModel.labels)                                                         

// COMMAND ----------

// MAGIC %md
// MAGIC ## Build Data Transformation pipeline
// MAGIC ![Data Transform Pipeline](files/images/TransformPipeline.PNG)

// COMMAND ----------

// Transform pipeline
val transformPipeline = new Pipeline().setStages(indexers ++ Array(labelIndexer, encoder, assembler, standardScalar))
val transformedDf = transformPipeline
  .fit(cleanDf)
  .transform(cleanDf)

// Split data
val Array(transformedTraining, transformedTest) = transformedDf.randomSplit(Array(0.8, 0.2), seed = randomSeed)

display(transformedDf.select("label_index", "norm_features"))

// COMMAND ----------

// MAGIC %md
// MAGIC ## GBT Binary classification
// MAGIC ![GBT Model](files/images/GBTModel.PNG)

// COMMAND ----------

// Train a GBT model.
val gbt = new GBTClassifier()
  .setLabelCol("is_anomaly")
  .setFeaturesCol("norm_features")
  .setMaxIter(10)
  .setFeatureSubsetStrategy("auto")

// Fit pipeline
val gbtModel = gbt.fit(transformedTraining)

// Make predictions.
val gbtPredictions = gbtModel.transform(transformedTest)
gbtPredictions.select("prediction", "label", "features").show(10)

val gbtEvaluator = new BinaryClassificationEvaluator()
  .setMetricName("areaUnderROC")
  .setLabelCol("is_anomaly")
  .setRawPredictionCol("rawPrediction")
val gbtAccuracy = gbtEvaluator.evaluate(gbtPredictions)
println(s"Test Error = ${(1.0 - gbtAccuracy)}")

// COMMAND ----------

// MAGIC %md
// MAGIC ## Random Forest Multiclassification - End to end pipeline
// MAGIC ![RandomForest Model](files/images/RandomForestPipeline.PNG)

// COMMAND ----------

// Train a RandomForest model.
val rf = new RandomForestClassifier()
  .setLabelCol("label_index")
  .setFeaturesCol("norm_features")
  .setNumTrees(10)

// Chain indexers and Random Forest in a Pipeline.
val rfPipeline = new Pipeline().setStages(indexers ++ Array(labelIndexer, encoder, assembler, standardScalar, rf, labelConverter))

// Fit pipeline
val rfPipelineModel = rfPipeline.fit(training)

// Make predictions.
val rfPredictions = rfPipelineModel.transform(test)
rfPredictions.select("predicted_label", "label", "features").show(10)

// Evaluate
val rfEvaluator = new MulticlassClassificationEvaluator()
  .setMetricName("accuracy")
  .setLabelCol("label_index")
  .setPredictionCol("prediction")
val rfAccuracy = rfEvaluator.evaluate(rfPredictions)
println(s"Test Error = ${(1.0 - rfAccuracy)}")


// COMMAND ----------

// MAGIC %md
// MAGIC ## Logistic Regression with CrossValidation
// MAGIC ![Logistic Regression w/ CrossValidation](files/images/LogRegCVPipeline.PNG)

// COMMAND ----------

// Train a Logistic Regres model.
val lr = new LogisticRegression()
  .setMaxIter(10)
  .setLabelCol("label_index")
  .setFeaturesCol("norm_features")

// Define ParamGrid
val lrParamGrid = new ParamGridBuilder()
  .addGrid(lr.regParam, Array(0.1, 0.01))
  .addGrid(lr.elasticNetParam, Array(0.1, 0.5, 0.8))
  .build()

// Define evaluator
val lrEvaluator = new MulticlassClassificationEvaluator()
  .setMetricName("accuracy")
  .setLabelCol("label_index")
  .setPredictionCol("prediction")

// CrossValidation model
val lrCv = new CrossValidator()
  .setEstimator(lr)
  .setEvaluator(lrEvaluator)
  .setEstimatorParamMaps(lrParamGrid)
  .setNumFolds(3)

// Chain indexers and Random Forest in a Pipeline.
val lrCvPipeline = new Pipeline().setStages(Array(lrCv, labelConverter))

// Fit model
val lrCvPipelineModel = lrCvPipeline.fit(transformedTraining)

// Make predictions with test
val lrCvPredictions = lrCvPipelineModel.transform(transformedTest)
lrCvPredictions.select("predicted_label", "label", "features").show(10)

// Evaluate
val lrCvAccuracy = lrEvaluator.evaluate(lrCvPredictions)
println(s"Test Error = ${(1.0 - lrCvAccuracy)}")


// COMMAND ----------

// MAGIC %md
// MAGIC ## Save models

// COMMAND ----------

gbtModel.write.overwrite().save(s"$modelDir/GBT")
rfPipelineModel.write.overwrite().save(s"$modelDir/RandomForestPipeline")
lrCvPipelineModel.write.overwrite().save(s"$modelDir/LogRegPipeline")