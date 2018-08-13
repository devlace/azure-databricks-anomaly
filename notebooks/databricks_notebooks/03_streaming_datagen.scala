// Databricks notebook source
import org.apache.spark.eventhubs.{ ConnectionStringBuilder, EventHubsConf, EventPosition }
import org.apache.spark.sql.functions.{ explode, split, to_json, struct }
import org.apache.spark.sql.streaming.Trigger.ProcessingTime

// Retrieve storage credentials
val eh_namespace = dbutils.secrets.get(scope = "storage_scope", key = "eventhub_namespace")
val eh_name = dbutils.secrets.get(scope = "storage_scope", key = "eventhub")
val eh_send_key = dbutils.secrets.get(scope = "storage_scope", key = "eventhub_send_key")

// Set data path
val data_path = "/mnt/blob_storage/data/for_streaming"

val connectionString = ConnectionStringBuilder()
  .setNamespaceName(eh_namespace)
  .setEventHubName(eh_name)
  .setSasKeyName("send")
  .setSasKey(eh_send_key)
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

val query =
  kdd_unlabeled_df_json
    .writeStream
    .format("eventhubs")
    .outputMode("update")
    .options(eventHubsConf.toMap)
    .trigger(ProcessingTime("25 seconds"))
    .option("checkpointLocation", s"$data_path/checkpoints/kdd_unlabeled_gen/")
    .start()