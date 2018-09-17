// Databricks notebook source
import org.apache.spark.eventhubs.{ ConnectionStringBuilder, EventHubsConf, EventPosition }
import org.apache.spark.sql.functions.{ explode, split, to_json, struct }
import org.apache.spark.sql.streaming.Trigger.ProcessingTime

// Retrieve storage credentials
val ehNamespace = dbutils.secrets.get(scope = "storage_scope", key = "eventhub_namespace")
val ehData = dbutils.secrets.get(scope = "storage_scope", key = "eventhub_data_name")
val ehDataSendKey = dbutils.secrets.get(scope = "storage_scope", key = "eventhub_data_send_key")

// Set data path
val data_path = "/mnt/blob_storage/data/for_streaming"

val connectionString = ConnectionStringBuilder()
  .setNamespaceName(ehNamespace)
  .setEventHubName(ehData)
  .setSasKeyName("send")
  .setSasKey(ehDataSendKey)
  .build

val eventHubsConf = EventHubsConf(connectionString)
  .setStartingPosition(EventPosition.fromEndOfStream)

// COMMAND ----------

val kdd_schema = spark.read.table("kdd_unlabeled").schema
val kdd_unlabeled_df = spark
  .readStream
  .schema(kdd_schema)
  .csv(s"$data_path/kddcup.testdata.unlabeled/")

val kdd_unlabeled_df_json = kdd_unlabeled_df.select(to_json(
  struct(
    $"id",
    $"duration", 
    $"protocol_type", 
    $"service", 
    $"src_bytes", 
    $"dst_bytes", 
    $"flag",
    $"land",
    $"wrong_fragment",
    $"urgent")).alias("body"))

// COMMAND ----------

// // Output to console
// var query = kdd_unlabeled_df_json
//   .writeStream
//   .outputMode("append")
//   .format("console")
//   .option("truncate", false)
//   .start()
// query.awaitTermination()

// COMMAND ----------

val query =
  kdd_unlabeled_df_json
    .writeStream
    .format("eventhubs")
    .outputMode("update")
    .options(eventHubsConf.toMap)
    .trigger(ProcessingTime("10 seconds"))
    .option("checkpointLocation", s"$data_path/checkpoints/kdd_unlabeled_gen/")
    .start()