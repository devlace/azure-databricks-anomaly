// Databricks notebook source
val df = spark.read.table("kdd")
val Array(training, test) = df.randomSplit(Array(0.8, 0.2), seed = 123)

// COMMAND ----------

// Transform data

// Remove nulls

// Clean up labels for anomaly
df.withColumn()

// COMMAND ----------

import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.OneHotEncoderEstimator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.Pipeline

val columns = df.columns.toSet
val features = columns -- Set("label")
val categoricalFeatures = Set("protocol_type", "service", "flag")
val continuousFeatures = features -- categoricalFeatures

// Indexers
val indexers = categoricalFeatures.map({ colName =>
  new StringIndexer().setInputCol(colName).setOutputCol(colName + "Index")
}).toArray

// Encoders
val encoder = new OneHotEncoderEstimator()
  .setInputCols(categoricalFeatures.map(colName => colName + "Index").toArray)
  .setOutputCols(categoricalFeatures.map(colName => colName + "Encoded").toArray)

// Label Indexer
val labelIndexer = new StringIndexer()
  .setInputCol("label")
  .setOutputCol("labelIndex")

// Vector Assembler
var selectedFeatures = continuousFeatures ++ categoricalFeatures.map(colName => colName + "Encoded") 
val assembler = new VectorAssembler()
  .setInputCols(selectedFeatures.toArray)
  .setOutputCol("features")

// Pipeline
val transformPipeline = new Pipeline().setStages(indexers ++ Array(encoder, labelIndexer, assembler))

// Transform training
val transformedTrainingData = transformPipeline.fit(training).transform(training)
display(transformedTrainingData)

// COMMAND ----------

import org.apache.spark.ml.iforest.IForest

val iForest = new IForest()
  .setNumTrees(100)
  .setMaxSamples(256)
  .setContamination(0.35)
  .setBootstrap(false)
  .setMaxDepth(100)
  .setSeed(123456L)
  .setFeaturesCol("features")
  .setLabelCol("labelIndex")

val pipeline = new Pipeline().setStages(indexers ++ Array(encoder, labelIndexer, assembler, iForest))
val model = pipeline.fit(training)
val predictions = model.transform(training)

display(predictions)

// COMMAND ----------

import ml.combust.bundle.BundleFile
import ml.combust.mleap.spark.SparkSupport._
import org.apache.spark.ml.bundle.SparkBundleContext
import resource._

implicit val context = SparkBundleContext().withDataset(predictions)
//save our pipeline to a zip file
//MLeap can save a file to any supported java.nio.FileSystem
(for(modelFile <- managed(BundleFile("jar:file:/tmp/test-pipeline-json.zip"))) yield {
  model.writeBundle.save(modelFile)(context)
}).tried.get

// COMMAND ----------

val transformedTraining = model.transform(training)
display(transformedTraining)

// COMMAND ----------

// MAGIC %md
// MAGIC ### Sample iForest

// COMMAND ----------

val startTime = System.currentTimeMillis()

// Wisconsin Breast Cancer Dataset
val dataset = spark.read.option("inferSchema", "true")
.csv("data/anomaly-detection/breastw.csv")

// Index label values: 2 -> 0, 4 -> 1
val indexer = new StringIndexer()
.setInputCol("_c10")
.setOutputCol("label")

val assembler = new VectorAssembler()
assembler.setInputCols(Array("_c1", "_c2", "_c3", "_c4", "_c5", "_c6", "_c7", "_c8", "_c9"))
assembler.setOutputCol("features")

val iForest = new IForest()
.setNumTrees(100)
.setMaxSamples(256)
.setContamination(0.35)
.setBootstrap(false)
.setMaxDepth(100)
.setSeed(123456L)

val pipeline = new Pipeline().setStages(Array(indexer, assembler, iForest))
val model = pipeline.fit(dataset)
val predictions = model.transform(dataset)

val binaryMetrics = new BinaryClassificationMetrics(
predictions.select("prediction", "label").rdd.map {
case Row(label: Double, ground: Double) => (label, ground)
}
)

val endTime = System.currentTimeMillis()
println(s"Training and predicting time: ${(endTime - startTime) / 1000} seconds.")
println(s"The model's auc: ${binaryMetrics.areaUnderROC()}")
