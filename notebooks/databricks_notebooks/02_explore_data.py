# Databricks notebook source
df = spark.read.table("kdd")

# COMMAND ----------

df.head()

# COMMAND ----------

display(df)

# COMMAND ----------

# Fraud vs non fraud

# COMMAND ----------

# Check for nulls
dir(df)

# COMMAND ----------

dir(df.stat)

# COMMAND ----------

summary_df = df.summary()
display(summary_df)

# COMMAND ----------

describe_df = df.describe()
display(describe_df)

# COMMAND ----------

df.count()

# COMMAND ----------



# COMMAND ----------

df.stat.crosstab("name", "item")

# COMMAND ----------

df.columns

# COMMAND ----------

categoricalFeatures = ['protocol_type', 'service', 'flag']

display(df.select(categoricalFeatures).describe())

# COMMAND ----------



# COMMAND ----------

from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoderEstimator

# Indexers
categoricalFeatures = ['protocol_type', 'service', 'flag']
indexers = [StringIndexer(inputCol=column, outputCol=column + '_index', handleInvalid='keep') for column in categoricalFeatures]

# encoders
encoders = [OneHotEncoderEstimator(inputCols=[column + '_index'], outputCols=[column + '_encoded']) for column in categoricalFeatures]

# Label Indexer
labelIndexer = [StringIndexer(inputCol='label', outputCol='label_index')]

# COMMAND ----------

from pyspark.ml import Pipeline

pipeline = Pipeline(stages=indexers + encoders + labelIndexer)
transformedDF = pipeline.fit(df).transform(df)
display(transformedDF)

# COMMAND ----------

selected_features = [
 'duration',
 'protocol_type_encoded',
 'service_encoded',
 'flag_encoded',
 'src_bytes',
 'dst_bytes',
 'land',
 'wrong_fragment',
 'urgent',
 'hot',
 'num_failed_logins',
 'logged_in',
 'num_compromised',
 'root_shell',
 'su_attempted',
 'num_root',
 'num_file_creations',
 'num_shells',
 'num_access_files',
 'num_outbound_cmds',
 'is_host_login',
 'is_guest_login',
 'count',
 'srv_count',
 'serror_rate',
 'srv_serror_rate',
 'rerror_rate',
 'srv_rerror_rate',
 'same_srv_rate',
 'diff_srv_rate',
 'srv_diff_host_rate',
 'dst_host_count',
 'dst_host_srv_count',
 'dst_host_same_srv_rate',
 'dst_host_diff_srv_rate',
 'dst_host_same_src_port_rate',
 'dst_host_srv_diff_host_rate',
 'dst_host_serror_rate',
 'dst_host_srv_serror_rate',
 'dst_host_rerror_rate',
 'dst_host_srv_rerror_rate',
]

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(inputCols=selected_features, outputCol='features')
transformedDF = assembler.transform(transformedDF)
display(transformedDF)

# COMMAND ----------

from pyspark.ml.classification import RandomForestClassifier

rf = RandomForestClassifier(labelCol='label_index', 
                            featuresCol='features',
                            maxDepth=5)

# COMMAND ----------

# Final Pipeline
pipeline = Pipeline(stages=indexers + encoders + labelIndexer + [assembler, rf])

# COMMAND ----------

model = pipeline.fit(train_df)