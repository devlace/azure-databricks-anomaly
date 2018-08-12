// Databricks notebook source
// Retrieve storage credentials
val eh_namespace = dbutils.secrets.get(scope = "storage_scope", key = "eventhub_namespace")
val eh_name = dbutils.secrets.get(scope = "storage_scope", key = "eventhub")
val eh_key = dbutils.secrets.get(scope = "storage_scope", key = "eventhub_key")

// COMMAND ----------

import org.apache.spark.eventhubs.{ ConnectionStringBuilder, EventHubsConf, EventPosition }
import org.apache.spark.sql.functions.{ explode, split }

val connectionString = ConnectionStringBuilder()
  .setNamespaceName(eh_namespace)
  .setEventHubName(eh_name)
  .setSasKeyName("KEY_NAME")
  .setSasKey("KEY")
  .build

// To connect to an Event Hub, EntityPath is required as part of the connection string.
// Here, we assume that the connection string from the Azure portal does not have the EntityPath part.
val connectionString = ConnectionStringBuilder("{EVENT HUB CONNECTION STRING FROM AZURE PORTAL}")
  .setEventHubName("{EVENT HUB NAME}")
  .build
val eventHubsConf = EventHubsConf(connectionString)
  .setStartingPosition(EventPosition.fromEndOfStream)
  
val eventhubs = spark.readStream
  .format("eventhubs")
  .options(eventHubsConf.toMap)
  .load()