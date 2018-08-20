# Databricks notebook source
from pyspark.sql import functions as F

# COMMAND ----------

df = spark.read.table("kdd")

# COMMAND ----------

display(df)

# COMMAND ----------

df.printSchema()

# COMMAND ----------

df.count()

# COMMAND ----------

# Summary on continuous features
cols = df.columns
noncont_features = ['id', 'protocol_type', 'service', 'flag', 'label']
cont_features = [x for x in cols if x not in noncont_features]

summary_df = df.select(cont_features).summary().cache()
display(summary_df)

# COMMAND ----------

# Normal vs Anomalies
transformed_df = (df\
  .withColumn("label", F.when(df.label == "normal.", 0).otherwise(1))\
  .groupBy("label")
  .agg(F.count("id")))

display(transformed_df)

# COMMAND ----------

# Count by label
transformed_df = (df\
  .groupBy("label")\
  .agg(F.count("label").alias("num_requests"))\
  .orderBy("num_requests", ascending=False))

display(transformed_df)

# COMMAND ----------

