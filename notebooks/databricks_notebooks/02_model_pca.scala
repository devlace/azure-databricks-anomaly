// Databricks notebook source
import org.apache.spark.ml.feature.{StringIndexer, OneHotEncoderEstimator, VectorAssembler, PCA, StandardScaler, MinMaxScaler}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.functions._
import breeze.linalg.{DenseVector, sum}
import breeze.numerics.pow

// COMMAND ----------

// MAGIC %md
// MAGIC ## Read in data and perform data cleaning

// COMMAND ----------

// Read data
val df = spark.read.table("kdd")

// Transform data
val transformed_df = df.withColumnRenamed("label", "original_label")
  .withColumn("label_name", when(col("original_label") === "normal.", "normal").otherwise("anomaly"))

// Drop nulls
// Lace TODO

// Clean up labels for anomaly
display(transformed_df)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Build data transformation ML pipeline

// COMMAND ----------

val columns = df.columns.toSet
val features = columns -- Set("id", "label", "original_label")
val categoricalFeatures = Set("protocol_type", "service", "flag")
val continuousFeatures = features -- categoricalFeatures
|
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
  .setOutputCol("norm_features")
  .setWithMean(true)
  .setWithStd(true)

// Pipeline
val transformPipeline = new Pipeline()
  .setStages(indexers ++ Array(encoder, labelIndexer, assembler, standardScalar))

// Transform training
val transformedTraining = transformPipeline
  .fit(training)
  .transform(training)
  .select("norm_features", "label")
  .cache()

display(transformedTraining)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Perform Principal Component Analysis

// COMMAND ----------

// Fit PCA model
val pca = new PCA()
  .setInputCol("norm_features")
  .setOutputCol("pca_features")
  .setK(3)
  .fit(transformedTraining)

val pcaResult = pca
  .transform(transformedTraining)
  .select("label", "pca_features", "norm_features")
  .cache()

display(pcaResult)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Reconstruct features and calculate Anomaly Score
// MAGIC Reconstruct the features using the Principal Components and the feature vectors. Then, calculate the normalized error, in this case the sum of squared differences from the original feature vector and the reconstructed features from the principal components. This becomes the Anomaly Score.

// COMMAND ----------

val reconstructionUdf = udf((v: Vector) => { 
  // Reconstruct vector using Principal components
  pca.pc.multiply(v) 
})
val anomalyScoreUdf = udf((v:Vector, x:Vector) => {
  // Calculate error (sum of squared differences)
  val vB = DenseVector(v.toArray)
  val xB = DenseVector(x.toArray)
  val diff = vB - xB
  val error = sum(pow(diff, 2))
  error
})
val anomalyScore = pcaResult
  .withColumn("reconstruction", reconstructionUdf(col("pca_features")))
  .withColumn("anomaly_score", anomalyScoreUdf(col("norm_features"), col("reconstruction")))

// Vectorize Anomaly Score
val anomalyAssembler = new VectorAssembler()
  .setInputCols(Array("anomaly_score"))
  .setOutputCol("anomaly_score_vec")

// Normalize anomaly score
val anomalyScoreScalar = new MinMaxScaler()
  .setInputCol("anomaly_score_vec")
  .setOutputCol("norm_anomaly_score_vec")

// Pipeline
val postTransformPipeline = new Pipeline()
  .setStages(Array(anomalyAssembler, anomalyScoreScalar))

val postTransformPipelineModel = postTransformPipeline
  .fit(anomalyScore)

val vecToDoubleUdf = udf((v: Vector) => { v.toArray(0) })
val predictions = postTransformPipelineModel
  .transform(anomalyScore)
  .withColumn("norm_anomaly_score", vecToDoubleUdf(col("norm_anomaly_score_vec")))
  .select("label", "norm_anomaly_score")
  .cache()

display(predictions)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Evaluate Model

// COMMAND ----------

import  org.apache.spark.ml.evaluation.BinaryClassificationEvaluator

val evaluator = new BinaryClassificationEvaluator()
  .setMetricName("areaUnderROC")
  .setLabelCol("label")
  .setRawPredictionCol("norm_anomaly_score")

var auc = evaluator.evaluate(predictions)

// COMMAND ----------

// import breeze.linalg.DenseVector
// import org.apache.spark.ml.linalg.Vector
// import org.apache.spark.sql.functions._

// val distance = udf[Double, Vector] { v =>
//   val vB = DenseVector(v.toArray)
//   vB.t * vB
// }

// val results = transformedTrainingData
//   .withColumn("distance", distance(transformedTrainingData("pca_features")))

// display(results)

// COMMAND ----------

// MAGIC %md
// MAGIC # Custom Transformer and Estimator

// COMMAND ----------

import org.apache.spark.ml._
import org.apache.spark.ml.linalg._
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.util._
import org.apache.spark.ml.feature.{PCA, PCAModel}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{StructField, StructType}
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
    val outputFields = schema.fields :+ StructField($(outputCol), new VectorUDT, false)
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
 * @param pc A principal components Matrix. Each column is one principal component.
 * @param explainedVariance A vector of proportions of variance explained by
 *                          each principal component.
 */
class PCAAnomalyModel (
    override val uid: String,
    val pcaModel: PCAModel)
  extends Model[PCAAnomalyModel] with PCAAnomalyParams with MLWritable {

  //import PCAAnomalyModel._

  /** @group setParam */
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  /**
   * Transform a vector by computed Principal Components.
   *
   * @note Vectors to be transformed must be the same length as the source vectors given
   * to `PCA.fit()`.
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
    pcaResults.withColumn($(outputCol), anomalyScoreUdf(col($(inputCol)), col($(outputPCACol))))
  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  override def copy(extra: ParamMap): PCAAnomalyModel = {
    val copied = new PCAAnomalyModel(uid, pcaModel)
    copyValues(copied, extra).setParent(parent)
  }

  override def write: MLWriter = ??? //new PCAModelWriter(this)
}

// object PCAModel extends MLReadable[PCAModel] {

//   private[PCAModel] class PCAModelWriter(instance: PCAModel) extends MLWriter {

//     private case class Data(pc: DenseMatrix, explainedVariance: DenseVector)

//     override protected def saveImpl(path: String): Unit = {
//       DefaultParamsWriter.saveMetadata(instance, path, sc)
//       val data = Data(instance.pc, instance.explainedVariance)
//       val dataPath = new Path(path, "data").toString
//       sparkSession.createDataFrame(Seq(data)).repartition(1).write.parquet(dataPath)
//     }
//   }

//   private class PCAModelReader extends MLReader[PCAModel] {

//     private val className = classOf[PCAAnomalyModel].getName

//     /**
//      * Loads a [[PCAAnomalyModel]] from data located at the input path. Note that the model includes an
//      * `explainedVariance` member that is not recorded by Spark 1.6 and earlier. A model
//      * can be loaded from such older data but will have an empty vector for
//      * `explainedVariance`.
//      *
//      * @param path path to serialized model data
//      * @return a [[PCAModel]]
//      */
//     override def load(path: String): PCAAnomalyModel = {
//       val metadata = DefaultParamsReader.loadMetadata(path, sc, className)

//       val dataPath = new Path(path, "data").toString
//       val model = if (majorVersion(metadata.sparkVersion) >= 2) {
//         val Row(pc: DenseMatrix, explainedVariance: DenseVector) =
//           sparkSession.read.parquet(dataPath)
//             .select("pc", "explainedVariance")
//             .head()
//         new PCAModel(metadata.uid, pc, explainedVariance)
//       } else {
//         // pc field is the old matrix format in Spark <= 1.6
//         // explainedVariance field is not present in Spark <= 1.6
//         val Row(pc: OldDenseMatrix) = sparkSession.read.parquet(dataPath).select("pc").head()
//         new PCAModel(metadata.uid, pc.asML,
//           Vectors.dense(Array.empty[Double]).asInstanceOf[DenseVector])
//       }
//       metadata.getAndSetParams(model)
//       model
//     }
//   }

//   override def read: MLReader[PCAAnomalyModel] = new PCAAnomalyModelReader

//   override def load(path: String): PCAAnomalyModel = super.load(path)
// }

// COMMAND ----------

// Fit PCA model
val pcaAnomaly = new PCAAnomaly()
  .setInputCol("norm_features")
  .setOutputPCACol("pca_features")  
  .setOutputCol("anomaly_score")
  .setK(3)
  .fit(transformedTraining)

val pcaResult = pcaAnomaly
  .transform(transformedTraining)
  .select("label", "anomaly_score", "pca_features", "norm_features")
  .cache()

display(pcaResult)

// COMMAND ----------

import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{col, explode, udf}
import org.apache.spark.sql.types.{DataTypes, StructType}

/**
 * An example demonstrating how to write a custom Transformer in a 3rd-party application.
 * This example intentionally avoids using any private Spark APIs.
 *
 * @param uid  All types inheriting from `Identifiable` require a `uid`.
 *             This includes Transformers, Estimators, and Models.
 */
class PCAAnomaly(override val uid: String) extends Transformer {

  // Transformer Params
  // Defining a Param requires 3 elements:
  //  - Param definition
  //  - Param getter method
  //  - Param setter method
  // (The getter and setter are technically not required, but they are nice standards to follow.)

  /**
   * Param for input column name.
   */
  final val inputCol: Param[String] = new Param[String](this, "inputCol", "input column name")
  final def getInputCol: String = $(inputCol)
  final def setInputCol(value: String): PCAAnomaly = set(inputCol, value)

  /**
   * Param for output column name.
   */
  final val outputCol: Param[String] = new Param[String](this, "outputCol", "output column name")
  final def getOutputCol: String = $(outputCol)
  final def setOutputCol(value: String): PCAAnomaly = set(outputCol, value)

  // (Optional) You can set defaults for Param values if you like.
  setDefault(inputCol -> "myInputCol", outputCol -> "myOutputCol")

  // Transformer requires 3 methods:
  //  - transform
  //  - transformSchema
  //  - copy

  // Our flatMap will split strings by commas.
  private val myFlatMapFunction: String => Seq[String] = { input: String =>
    input.split(",")
  }

  /**
   * This method implements the main transformation.
   * Its required semantics are fully defined by the method API: take a Dataset or DataFrame,
   * and return a DataFrame.
   *
   * Most Transformers are 1-to-1 row mappings which add one or more new columns and do not
   * remove any columns.  However, this restriction is not required.  This example does a flatMap,
   * so we could either (a) drop other columns or (b) keep other columns, making copies of values
   * in each row as it expands to multiple rows in the flatMap.  We do (a) for simplicity.
   */
  override def transform(dataset: Dataset[_]): DataFrame = {
    val flatMapUdf = udf(myFlatMapFunction)
    dataset.select(explode(flatMapUdf(col($(inputCol)))).as($(outputCol)))
  }

  /**
   * Check transform validity and derive the output schema from the input schema.
   *
   * We check validity for interactions between parameters during `transformSchema` and
   * raise an exception if any parameter value is invalid. Parameter value checks which
   * do not depend on other parameters are handled by `Param.validate()`.
   *
   * Typical implementation should first conduct verification on schema change and parameter
   * validity, including complex parameter interaction checks.
   */
  override def transformSchema(schema: StructType): StructType = {
    // Validate input type.
    // Input type validation is technically optional, but it is a good practice since it catches
    // schema errors early on.
    val actualDataType = schema($(inputCol)).dataType
    require(actualDataType.equals(DataTypes.StringType),
      s"Column ${$(inputCol)} must be StringType but was actually $actualDataType.")

    // Compute output type.
    // This is important to do correctly when plugging this Transformer into a Pipeline,
    // where downstream Pipeline stages may expect use this Transformer's output as their input.
    DataTypes.createStructType(
      Array(
        DataTypes.createStructField($(outputCol), DataTypes.StringType, false)
      )
    )
  }

  /**
   * Creates a copy of this instance.
   * Requirements:
   *  - The copy must have the same UID.
   *  - The copy must have the same Params, with some possibly overwritten by the `extra`
   *    argument.
   *  - This should do a deep copy of any data members which are mutable.  That said,
   *    Transformers should generally be immutable (except for Params), so the `defaultCopy`
   *    method often suffices.
   * @param extra  Param values which will overwrite Params in the copy.
   */
  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)
}