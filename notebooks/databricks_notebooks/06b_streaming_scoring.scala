// Databricks notebook source
import org.apache.spark.eventhubs.{ ConnectionStringBuilder, EventHubsConf, EventPosition }
import org.apache.spark.sql.functions.{ explode, split }
import org.apache.spark.sql.streaming.Trigger.ProcessingTime
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.{Vector, Vectors}

// COMMAND ----------

// MAGIC %md
// MAGIC ## Setup
// MAGIC Retrieve secrets, setup EventHub connection, load save Anomaly Model

// COMMAND ----------

// Retrieve storage credentials
val ehNamespace = dbutils.secrets.get(scope = "storage_scope", key = "eventhub_namespace")
val ehData = dbutils.secrets.get(scope = "storage_scope", key = "eventhub_data_name")
val ehDataListenKey = dbutils.secrets.get(scope = "storage_scope", key = "eventhub_data_listen_key")
val ehAnom = dbutils.secrets.get(scope = "storage_scope", key = "eventhub_anom_name")
val ehAnomSendKey = dbutils.secrets.get(scope = "storage_scope", key = "eventhub_anom_send_key")

// Set storage mount path
val storage_mount_path = "/mnt/blob_storage"

// Set data path
val data_path = "/mnt/blob_storage/data/for_streaming"

// Load model
val model = PipelineModel.load(s"$storage_mount_path/models/RandomForestPipeline")

// Setup EH connection
val dataEhConnectionString = ConnectionStringBuilder()
  .setNamespaceName(ehNamespace)
  .setEventHubName(ehData)
  .setSasKeyName("listen")
  .setSasKey(ehDataListenKey)
  .build
val dataEhConf = EventHubsConf(dataEhConnectionString)
  .setStartingPosition(EventPosition.fromEndOfStream)

val anomEhConnectionString = ConnectionStringBuilder()
  .setNamespaceName(ehNamespace)
  .setEventHubName(ehAnom)
  .setSasKeyName("send")
  .setSasKey(ehAnomSendKey)
  .build
val anomEhConf = EventHubsConf(anomEhConnectionString)
  .setStartingPosition(EventPosition.fromEndOfStream)


// COMMAND ----------

// MAGIC %md
// MAGIC ## Read message from EventHubs

// COMMAND ----------

// Read stream
val incomingStream = spark
  .readStream
  .format("eventhubs")
  .options(dataEhConf.toMap)
  .load()

// Event Hub message format is JSON and contains "body" field
// Body is binary, so we cast it to string to see the actual content of the message
val messages =
  incomingStream
  .withColumn("Offset", $"offset".cast(LongType))
  .withColumn("Time (readable)", $"enqueuedTime".cast(TimestampType))
  .withColumn("Timestamp", $"enqueuedTime".cast(LongType))
  .withColumn("Body", $"body".cast(StringType))
  .withWatermark("Time (readable)", "10 minutes")
  .select("Offset", "Time (readable)", "Timestamp", "Body")

messages.printSchema

// COMMAND ----------

// MAGIC %md
// MAGIC ## Transform and enrich message through joining with static data

// COMMAND ----------

var messageTransformed = 
  messages
  .select(
    get_json_object($"Body", "$.id").cast(StringType).alias("id"),
    get_json_object($"Body", "$.duration").cast(FloatType).alias("duration"),
    get_json_object($"Body", "$.protocol_type").cast(StringType).alias("protocol_type"), 
    get_json_object($"Body", "$.service").cast(StringType).alias("service"), 
    get_json_object($"Body", "$.src_bytes").cast(FloatType).alias("src_bytes"), 
    get_json_object($"Body", "$.dst_bytes").cast(FloatType).alias("dst_bytes"), 
    get_json_object($"Body", "$.flag").cast(StringType).alias("flag"),
    get_json_object($"Body", "$.land").cast(ShortType).alias("land"),
    get_json_object($"Body", "$.wrong_fragment").cast(FloatType).alias("wrong_fragment"),
    get_json_object($"Body", "$.urgent").cast(FloatType).alias("urgent"),
    $"Timestamp")

// Join with static table
val kdd_unlabeled = spark.read.table("kdd_unlabeled")
val messageAll = messageTransformed
  .join(kdd_unlabeled, messageTransformed("id") === kdd_unlabeled("id"), "left_outer")
  .drop(kdd_unlabeled("id"))
  .drop(kdd_unlabeled("duration"))
  .drop(kdd_unlabeled("protocol_type"))
  .drop(kdd_unlabeled("service"))
  .drop(kdd_unlabeled("src_bytes"))
  .drop(kdd_unlabeled("dst_bytes"))
  .drop(kdd_unlabeled("flag"))
  .drop(kdd_unlabeled("land"))
  .drop(kdd_unlabeled("wrong_fragment"))
  .drop(kdd_unlabeled("urgent"))

messageAll.printSchema

// COMMAND ----------

// MAGIC %md
// MAGIC ## Use model to identify Anomalies in data stream

// COMMAND ----------

// Make predictions
val anomalies = model.transform(messageAll).filter("prediction == 1")

// COMMAND ----------

// MAGIC %md
// MAGIC ## Output anomalies

// COMMAND ----------

// Output to console
var query = anomalies
  .select("id", "probability", "prediction") //filter for easy viewing
  .writeStream
  .outputMode("append")
  .format("console")
  .option("truncate", false)
  .start()
query.awaitTermination()

// COMMAND ----------

// Wrap in body tag
val anomalies_wrapper = anomalies.select(to_json(
  struct(
    $"id",
    $"norm_anomaly_score")).alias("body"))

val query =
  anomalies_wrapper
    .writeStream
    .format("eventhubs")
    .outputMode("update")
    .options(anomEhConf.toMap)
    .trigger(ProcessingTime("25 seconds"))
    .option("checkpointLocation", s"$data_path/checkpoints/anomalies/")
    .start()

// COMMAND ----------

println(query.lastProgress)