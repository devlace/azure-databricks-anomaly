// Databricks notebook source


// COMMAND ----------

package org.apache.spark.ml.feature

import org.apache.hadoop.fs.Path

import org.apache.spark.ml._
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.util._
// import org.apache.spark.ml.feature.{PCA, PCAModel}
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{StructField, StructType, DoubleType}

import breeze.linalg.{DenseVector, sum}
import breeze.numerics.pow

/**
 * Params for [[PCAAnomaly]] and [[PCAAnomalyModel]].
 */
trait PCAAnomalyParams extends Params with HasInputCol with HasOutputCol {
  final val outputPCACol = new Param[String](this, "outputPCACol", "The output column with PCA features")
  final val outputAbsScoreCol = new Param[String](this, "outputAbsScoreCol", "The output column with non-normalized Anomaly Scores")
  final val labelCol = new Param[String](this, "labelCol", "Label column")
  setDefault(outputPCACol, "pca_features")
  setDefault(outputAbsScoreCol, "nonnorm_anomaly_score")
  setDefault(labelCol, "label")
  
  final val k: IntParam = new IntParam(this, "k", "the number of principal components (> 0)",
    ParamValidators.gt(0))
  
  /** Validates and transforms the input schema. */
  protected def validateAndTransformSchema(schema: StructType): StructType = {
    //SchemaUtils.checkColumnType(schema, $(inputCol), new VectorUDT)
    require(!schema.fieldNames.contains($(outputCol)), s"Output column ${$(outputCol)} already exists.")
    val outputFields = schema.fields :+ 
      StructField($(outputPCACol), new VectorUDT, false) :+ 
      StructField($(outputCol), DoubleType, false)
    StructType(outputFields)
  }
}

/**
 * PCA trains a model to project vectors to a lower dimensional space of the top `PCA!.k`
 * principal components.
 */
class PCAAnomaly (override val uid: String)
  extends Estimator[PCAAnomalyModel] with PCAAnomalyParams with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("pca_anomaly"))

  def setInputCol(value: String): this.type = set(inputCol, value)
  def setOutputCol(value: String): this.type = set(outputCol, value)
  def setLabelCol(value: String): this.type = set(labelCol, value)
  def setOutputPCACol(value: String): this.type = set(outputPCACol, value)
  def setOutputAbsScoreCol(value: String): this.type = set(outputAbsScoreCol, value)
  def setK(value: Int): this.type = set(k, value)

  /**
   * Computes a [[PCAAnomalyModel]] that contains the principal components of the input vectors.
   */
  override def fit(dataset: Dataset[_]): PCAAnomalyModel = {
    transformSchema(dataset.schema, logging = true)
    
    // remove anomalies
    val cleanDataset = dataset.filter(col($(labelCol)) === 0)
    
    // Fit regular PCA model
    val pcaModel = new PCA()
      .setInputCol($(inputCol))
      .setOutputCol($(outputPCACol))
      .setK($(k))
      .fit(cleanDataset)
    
    copyValues(new PCAAnomalyModel(uid, pcaModel).setParent(this))
  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  override def copy(extra: ParamMap): PCAAnomaly = defaultCopy(extra)
}

object PCAAnomaly extends DefaultParamsReadable[PCAAnomaly] {
  override def load(path: String): PCAAnomaly = super.load(path)
}

/**
 * Model fitted by [[PCAAnomaly]]. Uses PCA to detect anomalies
 *
 * @param pcaModel A PCA model
 */
class PCAAnomalyModel (
  override val uid: String, 
  val pcaModel: PCAModel)
  extends Model[PCAAnomalyModel] with PCAAnomalyParams with MLWritable {

  import PCAAnomalyModel._

  def setInputCol(value: String): this.type = set(inputCol, value)
  def setOutputCol(value: String): this.type = set(outputCol, value)
  def setLabelCol(value: String): this.type = set(labelCol, value)
  def setOutputPCACol(value: String): this.type = set(outputPCACol, value)
  def setOutputAbsScoreCol(value: String): this.type = set(outputAbsScoreCol, value)
  def setK(value: Int): this.type = set(k, value)
    
  /**
   * Transform a vector by computed Principal Components.
   *
   * @note Vectors to be transformed must be the same length as the source vectors given
   * to `PCAAnomaly.fit()`.
   */
  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)
    
    val pcaResults = pcaModel.transform(dataset)
    
    val anomalyScoreUdf = udf((originalFeatures:Vector, pcaFeatures:Vector) => {
      // Reconstruct vector using Principal components
      val reconstructedFeatures = pcaModel.pc.multiply(pcaFeatures) 
      
      // Calculate error (sum of squared differences)
      val originalFeaturesB = DenseVector(originalFeatures.toArray)
      val reconstructedFeaturesB = DenseVector(reconstructedFeatures.toArray)
      val diff = originalFeaturesB - reconstructedFeaturesB
      val error = sum(pow(diff, 2))
      error
    })
    val anomalyScore = pcaResults.withColumn($(outputAbsScoreCol), anomalyScoreUdf(col($(inputCol)), col($(outputPCACol))))
    
    // Normalize
    val Row(maxVal: Double) = anomalyScore.select(max($(outputAbsScoreCol))).head
    val Row(minVal: Double) = anomalyScore.select(min($(outputAbsScoreCol))).head
    val nomarlizeAnomalyScore = anomalyScore
      .withColumn($(outputCol), (col($(outputAbsScoreCol)) - minVal) / (maxVal - minVal))
    
    nomarlizeAnomalyScore
  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  override def copy(extra: ParamMap): PCAAnomalyModel = {
    val copied = new PCAAnomalyModel(uid, pcaModel)
    copyValues(copied, extra).setParent(parent)
  }

  override def write: MLWriter = new PCAAnomalyModelWriter(this)
}

object PCAAnomalyModel extends MLReadable[PCAAnomalyModel] {

  private[PCAAnomalyModel] class PCAAnomalyModelWriter(instance: PCAAnomalyModel) extends MLWriter {
    override protected def saveImpl(path: String): Unit = {
      DefaultParamsWriter.saveMetadata(instance, path, sc)
      val pcaPath = new Path(path, "pca").toString
      instance.pcaModel.save(pcaPath)
    }
  }

  private class PCAAnomalyModelReader extends MLReader[PCAAnomalyModel] {

    private val className = classOf[PCAAnomalyModel].getName

    /**
     * Loads a [[PCAAnomalyModel]] from data located at the input path.
     *
     * @param path path to serialized model data
     * @return a [[PCAAnomalyModel]]
     */
    override def load(path: String): PCAAnomalyModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)
      val pcaPath = new Path(path, "pca").toString
      val pcaModel = PCAModel.load(pcaPath)
      val model = new PCAAnomalyModel(metadata.uid, pcaModel)
      DefaultParamsReader.getAndSetParams(model, metadata)
      model
    }
  }

  override def read: MLReader[PCAAnomalyModel] = new PCAAnomalyModelReader

  override def load(path: String): PCAAnomalyModel = super.load(path)
}



// COMMAND ----------

import org.apache.spark.ml.feature.{StringIndexer, OneHotEncoderEstimator, VectorAssembler, PCA, StandardScaler, MinMaxScaler, PCAAnomaly}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.functions._
import breeze.linalg.{DenseVector, sum}
import breeze.numerics.pow

// Read data
spark.catalog.refreshTable("kdd") // need to refresh to invalidate cache
val df = spark.read.table("kdd")

// Transform data
val transformed_df = df
  .withColumnRenamed("label", "original_label")
  .withColumn("rainbow", when(col("original_label") === "normal.", 0).otherwise(1))
  .na.drop()

// Clean up labels for anomaly
display(transformed_df)

val columns = transformed_df.columns.toSet
val features = columns -- Set("id", "rainbow", "original_label")
val categoricalFeatures = Set("protocol_type", "service", "flag")
val continuousFeatures = features -- categoricalFeatures
|
// Split data
val Array(training, test) = transformed_df.randomSplit(Array(0.8, 0.2), seed = 123)


// COMMAND ----------

// Indexers
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

// PCA Anomaly model
val pcaAnom = new PCAAnomaly()
  .setInputCol("norm_features")
  .setOutputPCACol("pca_features")  
  .setOutputCol("anomaly_score")
  .setLabelCol("rainbow")
  .setK(3)

// COMMAND ----------

// Pipeline
val mainPipeline = new Pipeline()
  .setStages(indexers ++ 
     Array(encoder, assembler, standardScalar, pcaAnom)) //pcaAnom

// Fit pipeline
val mainPipelineModel = mainPipeline.fit(training)

// Save pipeline
mainPipelineModel
  .write
  .overwrite
  .save("mnt/blob_storage/models/PCAAnomalyModel")

// COMMAND ----------

// MAGIC %md
// MAGIC ## Use Model to predict anomalies

// COMMAND ----------

// MAGIC %md
// MAGIC #### Using training data

// COMMAND ----------

import org.apache.spark.ml.{Pipeline, PipelineModel}

val model = PipelineModel.load("/mnt/blob_storage/models/PCAAnomalyModel")

val transformedTraining = model.transform(training)
 // .select("original_label", "rainbow", "anomaly_score")

display(transformedTraining.groupBy("original_label").agg(avg("anomaly_score")))
//display(transformedTraining)

// COMMAND ----------

display(transformedTraining.groupBy("rainbow").agg(avg("anomaly_score")))

// COMMAND ----------

// MAGIC %md
// MAGIC #### Using test data

// COMMAND ----------

val transformedTest = mainPipelineModel.transform(test)
  .select("rainbow", "anomaly_score")
  .cache()

display(transformedTest)

// COMMAND ----------

// display(transformedTest
//         .filter(col("label") === 0)
//         .withColumn("log_anom_score", ))

// COMMAND ----------

// MAGIC %md
// MAGIC ## Evaluate Model using Test data

// COMMAND ----------

import  org.apache.spark.ml.evaluation.BinaryClassificationEvaluator

val evaluator = new BinaryClassificationEvaluator()
  .setMetricName("areaUnderROC")
  .setLabelCol("rainbow")
  .setRawPredictionCol("anomaly_score")

evaluator.evaluate(transformedTraining)
