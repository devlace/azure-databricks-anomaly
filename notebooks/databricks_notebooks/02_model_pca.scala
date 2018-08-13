// Databricks notebook source
import org.apache.spark.sql.functions.{when, col}

val df = spark.read.table("kdd")
display(df)

// Transform data
val transformed_df = df.withColumnRenamed("label", "original_label")
  .withColumn("label_name", when(col("original_label") === "normal.", "normal").otherwise("anomaly"))

// Remove nulls

// Clean up labels for anomaly
display(transformed_df)

// COMMAND ----------

import org.apache.spark.ml.feature.{StringIndexer, OneHotEncoderEstimator, VectorAssembler, PCA, StandardScaler}
import org.apache.spark.ml.Pipeline

val columns = df.columns.toSet
val features = columns -- Set("id", "label", "original_label")
val categoricalFeatures = Set("protocol_type", "service", "flag")
val continuousFeatures = features -- categoricalFeatures

// Split data
val Array(training, test) = transformed_df.randomSplit(Array(0.8, 0.2), seed = 123)

// Indexers
val indexers = categoricalFeatures.map({ colName =>
  new StringIndexer().setInputCol(colName).setOutputCol(colName + "_index").setHandleInvalid("keep")
}).toArray

// Encoders
val encoder = new OneHotEncoderEstimator()
  .setInputCols(categoricalFeatures.map(colName => colName + "_index").toArray)
  .setOutputCols(categoricalFeatures.map(colName => colName + "_encoded").toArray)

// Label Indexer
val labelIndexer = new StringIndexer()
  .setInputCol("label_name")
  .setOutputCol("label")

// Vector Assembler
var selectedFeatures = continuousFeatures ++ categoricalFeatures.map(colName => colName + "_encoded") 
val assembler = new VectorAssembler()
  .setInputCols(selectedFeatures.toArray)
  .setOutputCol("features")

val standardScalar = new StandardScaler()
  .setInputCol("features")
  .setOutputCol("normalized_features")
  .setWithMean(true)
  .setWithStd(true)

// Pipeline
val transformPipeline = new Pipeline()
  .setStages(indexers ++ Array(encoder, labelIndexer, assembler, standardScalar))

// Transform training
val transformedTraining = transformPipeline
  .fit(training)
  .transform(training)
  .select("normalized_features", "label")
  .cache()

display(transformedTraining)

// COMMAND ----------

// Fit PCA model

val pca = new PCA()
  .setInputCol("normalized_features")
  .setOutputCol("pca_features")
  .setK(3)
  .fit(transformedTraining)

val pcaResult = pca
  .transform(transformedTraining)
  .select("normalized_features", "pca_features", "label")
  .cache()

display(pcaResult)

// COMMAND ----------

pca.pc.numCols

// COMMAND ----------

import breeze.linalg.{DenseVector, sum}
import breeze.numerics.pow
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.functions._

val reconstruction = udf((v: Vector) => { 
  // Reconstruct vector using Principal components
  pca.pc.multiply(v) 
})
val results = pcaResult
  .withColumn("reconstruction", reconstruction(pcaResult("pca_features")))

val anomalyScore = udf((v:Vector, x:Vector) => {
  // Calculate error (sum of squared differences)
  val vB = DenseVector(v.toArray)
  val xB = DenseVector(x.toArray)
  val diff = vB - xB
  sum(pow(diff, 2))
})
val resultsScore = results
  .withColumn("anomalyScore", anomalyScore(results("normalized_features"), results("reconstruction")))
  .select("label", "anomalyScore")

display(resultsScore)

// import breeze.linalg.DenseVector
// import org.apache.spark.ml.linalg.Vector
// import org.apache.spark.sql.functions._

// val reconstruction = udf[Vector, (Vector, Vector)] { (v,x) =>
//   val pcInv = inv(new breeze.linalg.DenseMatrix(3, 122, pca.pc.toArray))
//   val vB = DenseVector(v.toArray)
//   val recontructedVector = vB * pcInv
//   val dRecon = new org.apache.spark.ml.linalg.DenseVector(recon.toArray)
//   dRecon
// }

// val results = pcaResult
//   .withColumn("reconstruction", reconstruction(pcaResult("pca_features"), pcaResult("normalized_features")))

// COMMAND ----------

display(results)

// COMMAND ----------

import breeze.linalg.DenseVector
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.functions._

val distance = udf[Double, Vector] { v =>
  val vB = DenseVector(v.toArray)
  vB.t * vB
}

val results = transformedTrainingData
  .withColumn("distance", distance(transformedTrainingData("pca_features")))

display(results)

// COMMAND ----------

display(
  results.groupBy("label")
    .agg(
      avg("distance").as("avg"),
      
    )
  )

// COMMAND ----------

val Row(threshold: Double) = results.filter(col("label") === 0).groupBy().avg("distance").first
val upperThreshold = threshold * 1.05
val lowerThreshold = threshold * 0.95



// COMMAND ----------


val resultsPred = results.withColumn("predicted", 
  when(col("distance") >= lowerThreshold && col("distance") <= upperThreshold, "normal").otherwise("anomaly"))

display(resultsPred)

// COMMAND ----------



// COMMAND ----------

import breeze.linalg.{DenseVector, inv}
import org.apache.spark.ml.stat.Correalation
import org.apache.spark.ml.linalg.{Matrix, Vector}
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.functions._

def withMahalanobois(df: DataFrame, inputCol: String): DataFrame = {
  val Row(coeff1: Matrix) = Correlation.corr(df, inputCol).head

  val invCovariance = inv(new breeze.linalg.DenseMatrix(2, 2, coeff1.toArray))

  val mahalanobois = udf[Double, Vector] { v =>
    val vB = DenseVector(v.toArray)
    vB.t * invCovariance * vB
  }

  df.withColumn("mahalanobois", mahalanobois(df(inputCol)))
}

val mahalanobois = udf[Double, Vector] { v =>
    val vB = DenseVector(v.toArray)
    vB.t * invCovariance * vB
  }

val withMahalanobois: DataFrame = withMahalanobois(transformedTrainingData, "pca_features")
display(withMahalanobois)



// COMMAND ----------

val coeff1 = Correlation.corr(transformedTrainingData, "pca_features").head

// COMMAND ----------

val Row(coeff1: Matrix) = Correlation.corr(transformedTrainingData, "pca_features").head

// COMMAND ----------

val invCovariance = inv(new breeze.linalg.DenseMatrix(2, 2, coeff1.toArray))

// COMMAND ----------

