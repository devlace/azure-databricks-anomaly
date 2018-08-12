// Databricks notebook source
// Retrieve storage credentials
val eh_namespace = dbutils.secrets.get(scope = "storage_scope", key = "eventhub_namespace")
val eh_name = dbutils.secrets.get(scope = "storage_scope", key = "eventhub")
val eh_listen_key = dbutils.secrets.get(scope = "storage_scope", key = "eventhub_listen_key")

// Set storage mount path
val storage_mount_path = "/mnt/blob_storage"

import org.apache.spark.eventhubs.{ ConnectionStringBuilder, EventHubsConf, EventPosition }
import org.apache.spark.sql.functions.{ explode, split }
import org.apache.spark.sql.streaming.Trigger.ProcessingTime

val connectionString = ConnectionStringBuilder()
  .setNamespaceName(eh_namespace)
  .setEventHubName(eh_name)
  .setSasKeyName("listen")
  .setSasKey(eh_listen_key)
  .build

val eventHubsConf = EventHubsConf(connectionString)
  .setStartingPosition(EventPosition.fromEndOfStream)

// COMMAND ----------

import org.apache.spark.eventhubs._
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._

val incomingStream = spark
  .readStream
  .format("eventhubs")
  .options(eventHubsConf.toMap)
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

var messageTransformed = 
  messages
  .select(get_json_object($"Body", "$.duration").cast(FloatType).alias("duration"),
          get_json_object($"Body", "$.protocol_type").cast(StringType).alias("protocol_type"), 
          get_json_object($"Body", "$.service").cast(StringType).alias("service"), 
          get_json_object($"Body", "$.src_bytes").cast(FloatType).alias("src_bytes"), 
          get_json_object($"Body", "$.dst_bytes").cast(FloatType).alias("dst_bytes"), 
          get_json_object($"Body", "$.flag").cast(StringType).alias("flag"),
          get_json_object($"Body", "$.land").cast(ShortType).alias("land"),
          get_json_object($"Body", "$.wrong_fragment").cast(FloatType).alias("wrong_fragment"),
          get_json_object($"Body", "$.urgent").cast(FloatType).alias("urgent"),
          $"Timestamp")

// COMMAND ----------

// Output
var query = messageTransformed
  .writeStream
  .outputMode("append")
  .format("console")
  .option("truncate", false)
  .start()
query.awaitTermination()