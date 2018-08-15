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
import org.apache.spark.rdd.RDD
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{StructField, StructType, DoubleType}
import org.apache.spark.mllib.linalg.VectorUDT

import breeze.linalg.{DenseVector, sum}
import breeze.numerics.pow


/**
 * Params for [[PCAAnomaly]] and [[PCAAnomalyModel]].
 */
trait PCAAnomalyParams extends Params with HasInputCol with HasOutputCol {
  //final val inputCol= new Param[String](this, "inputCol", "The input column")
  //final val outputCol = new Param[String](this, "outputCol", "The output column")
  final val outputPCACol = new Param[String](this, "outputPCACol", "The output column with PCA features")
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
  def setOutputPCACol(value: String): this.type = set(outputPCACol, value)
  def setK(value: Int): this.type = set(k, value)

  /**
   * Computes a [[PCAAnomalyModel]] that contains the principal components of the input vectors.
   */
  override def fit(dataset: Dataset[_]): PCAAnomalyModel = {
    transformSchema(dataset.schema, logging = true)
    
    // Fit regular PCA model
    val pcaModel = new PCA()
      .setInputCol($(inputCol))
      .setOutputCol($(outputPCACol))
      .setK($(k))
      .fit(dataset)
    
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

  /** @group setParam */
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)

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
    val result = pcaResults.withColumn($(outputCol), anomalyScoreUdf(col($(inputCol)), col($(outputPCACol)))).cache()
    result
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
      new PCAAnomalyModel(metadata.uid, pcaModel)
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
val df = spark.read.table("kdd")

// Transform data
val transformed_df = df.withColumnRenamed("label", "original_label")
  .withColumn("label_name", when(col("original_label") === "normal.", "normal").otherwise("anomaly"))

// Drop nulls
// Lace TODO

// Clean up labels for anomaly
display(transformed_df)

val columns = df.columns.toSet
val features = columns -- Set("id", "label", "original_label")
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

// Label Indexer
val labelIndexer = new StringIndexer()
  .setInputCol("label_name")
  .setOutputCol("label")

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
  .setK(3)

// Vectorize Anomaly Score
val anomalyAssembler = new VectorAssembler()
  .setInputCols(Array("anomaly_score"))
  .setOutputCol("anomaly_score_vec")

// Normalize anomaly score
val anomalyScoreScalar = new MinMaxScaler()
  .setInputCol("anomaly_score_vec")
  .setOutputCol("norm_anomaly_score_vec")

// Pipeline
val mainPipeline = new Pipeline()
  .setStages(indexers ++ 
     Array(encoder, labelIndexer, assembler, standardScalar, 
       pcaAnom, anomalyAssembler, anomalyScoreScalar)) 

// Fit pipeline
val mainPipelineModel = mainPipeline.fit(training)

// Save pipeline
mainPipelineModel.save("mnt/blob_storage/models/MainPCAAnomalyModel")

// COMMAND ----------

val vecToDoubleUdf = udf((v: Vector) => { v.toArray(0) })

val transformedTraining = mainPipelineModel.transform(training)
  .withColumn("norm_anomaly_score", vecToDoubleUdf(col("norm_anomaly_score_vec")))
  .select("label", "norm_anomaly_score")
  .cache()

display(transformedTraining)

// COMMAND ----------

val transformedTest = mainPipelineModel.transform(test)
  .withColumn("norm_anomaly_score", vecToDoubleUdf(col("norm_anomaly_score_vec")))
  .select("label", "norm_anomaly_score")
  .cache()

display(transformedTest)