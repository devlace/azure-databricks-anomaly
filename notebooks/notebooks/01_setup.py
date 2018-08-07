# Databricks notebook source
# MAGIC %md
# MAGIC # Mount blob storage

# COMMAND ----------

# Retrieve storage credentials
storage_account = dbutils.secrets.get(scope = "storage_scope", key = "storage_account")
storage_key = dbutils.secrets.get(scope = "storage_scope", key = "storage_key")

# Set mount path
storage_mount_path = "/mnt/blob_storage"

# Unmount if existing
for mp in dbutils.fs.mounts():
  if mp.mountPoint == storage_mount_path:
    dbutils.fs.unmount(storage_mount_path)

# Refresh mounts
dbutils.fs.refreshMounts()

# COMMAND ----------

# Mount
dbutils.fs.mount(
  source = "wasbs://databricks@" + storage_account + ".blob.core.windows.net",
  mount_point = storage_mount_path,
  extra_configs = {"fs.azure.account.key." + storage_account + ".blob.core.windows.net": storage_key})

# Refresh mounts
dbutils.fs.refreshMounts()

# COMMAND ----------

# MAGIC %md
# MAGIC # Download Data

# COMMAND ----------

import os
import gzip
import shutil
from urllib.request import urlretrieve

def download_and_uncompress_gz(data_url, out_file):
  tmp_loc = '/tmp/data.gz'
  # Download
  urlretrieve(data_url, tmp_loc)
  # Uncompress
  with gzip.open(tmp_loc, 'rb') as f_in:
    with open(out_file, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
  # Cleanup
  os.remove(tmp_loc)
  

# Note that Azure Databricks configures each cluster node with a FUSE mount that allows processes running on cluster nodes to read and write to the underlying
# distributed storage layer with local file APIs
# See here: https://docs.azuredatabricks.net/user-guide/dbfs-databricks-file-system.html#access-dbfs-using-local-file-apis
download_and_uncompress_gz(data_url='https://archive.ics.uci.edu/ml/machine-learning-databases/kddcup99-mld/kddcup.data.gz',
                          out_file='/dbfs' + storage_mount_path + '/kddcup.data.csv')

download_and_uncompress_gz(data_url='http://kdd.ics.uci.edu/databases/kddcup99/kddcup.testdata.unlabeled.gz',
                          out_file='/dbfs' + storage_mount_path + '/kddcup.testdata.unlabeled.csv')


# COMMAND ----------

# MAGIC %md
# MAGIC # Create tables

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create KDD Table

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS kdd_temp;
# MAGIC CREATE TABLE kdd_temp
# MAGIC (
# MAGIC   duration FLOAT,
# MAGIC   protocol_type STRING,
# MAGIC   service STRING,
# MAGIC   flag STRING,
# MAGIC   src_bytes FLOAT,
# MAGIC   dst_bytes FLOAT,
# MAGIC   land SHORT,
# MAGIC   wrong_fragment FLOAT,
# MAGIC   urgent FLOAT,
# MAGIC   hot FLOAT,
# MAGIC   num_failed_logins FLOAT,
# MAGIC   logged_in SHORT,
# MAGIC   num_compromised FLOAT,
# MAGIC   root_shell FLOAT,
# MAGIC   su_attempted FLOAT,
# MAGIC   num_root FLOAT,
# MAGIC   num_file_creations FLOAT,
# MAGIC   num_shells FLOAT,
# MAGIC   num_access_files FLOAT,
# MAGIC   num_outbound_cmds FLOAT,
# MAGIC   is_host_login SHORT,
# MAGIC   is_guest_login SHORT,
# MAGIC   count FLOAT,
# MAGIC   srv_count FLOAT,
# MAGIC   serror_rate FLOAT,
# MAGIC   srv_serror_rate FLOAT,
# MAGIC   rerror_rate FLOAT,
# MAGIC   srv_rerror_rate FLOAT,
# MAGIC   same_srv_rate FLOAT,
# MAGIC   diff_srv_rate FLOAT,
# MAGIC   srv_diff_host_rate FLOAT,
# MAGIC   dst_host_count FLOAT,
# MAGIC   dst_host_srv_count FLOAT,
# MAGIC   dst_host_same_srv_rate FLOAT,
# MAGIC   dst_host_diff_srv_rate FLOAT,
# MAGIC   dst_host_same_src_port_rate FLOAT,
# MAGIC   dst_host_srv_diff_host_rate FLOAT,
# MAGIC   dst_host_serror_rate FLOAT,
# MAGIC   dst_host_srv_serror_rate FLOAT,
# MAGIC   dst_host_rerror_rate FLOAT,
# MAGIC   dst_host_srv_rerror_rate FLOAT,
# MAGIC   label STRING
# MAGIC )
# MAGIC USING CSV
# MAGIC LOCATION '/mnt/blob_storage/kddcup.data.csv'
# MAGIC OPTIONS ("header"="false");
# MAGIC 
# MAGIC -- LACE: TODO, convert to databricks delta
# MAGIC DROP TABLE IF EXISTS kdd;
# MAGIC CREATE TABLE kdd 
# MAGIC USING org.apache.spark.sql.parquet
# MAGIC AS SELECT * FROM kdd_temp;
# MAGIC 
# MAGIC -- Drop temporary table
# MAGIC DROP TABLE kdd_temp;

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Unlablled KDD table

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS kdd_unlabeled_temp;
# MAGIC CREATE TABLE kdd_unlabeled_temp
# MAGIC (
# MAGIC   duration FLOAT,
# MAGIC   protocol_type STRING,
# MAGIC   service STRING,
# MAGIC   flag STRING,
# MAGIC   src_bytes FLOAT,
# MAGIC   dst_bytes FLOAT,
# MAGIC   land SHORT,
# MAGIC   wrong_fragment FLOAT,
# MAGIC   urgent FLOAT,
# MAGIC   hot FLOAT,
# MAGIC   num_failed_logins FLOAT,
# MAGIC   logged_in SHORT,
# MAGIC   num_compromised FLOAT,
# MAGIC   root_shell FLOAT,
# MAGIC   su_attempted FLOAT,
# MAGIC   num_root FLOAT,
# MAGIC   num_file_creations FLOAT,
# MAGIC   num_shells FLOAT,
# MAGIC   num_access_files FLOAT,
# MAGIC   num_outbound_cmds FLOAT,
# MAGIC   is_host_login SHORT,
# MAGIC   is_guest_login SHORT,
# MAGIC   count FLOAT,
# MAGIC   srv_count FLOAT,
# MAGIC   serror_rate FLOAT,
# MAGIC   srv_serror_rate FLOAT,
# MAGIC   rerror_rate FLOAT,
# MAGIC   srv_rerror_rate FLOAT,
# MAGIC   same_srv_rate FLOAT,
# MAGIC   diff_srv_rate FLOAT,
# MAGIC   srv_diff_host_rate FLOAT,
# MAGIC   dst_host_count FLOAT,
# MAGIC   dst_host_srv_count FLOAT,
# MAGIC   dst_host_same_srv_rate FLOAT,
# MAGIC   dst_host_diff_srv_rate FLOAT,
# MAGIC   dst_host_same_src_port_rate FLOAT,
# MAGIC   dst_host_srv_diff_host_rate FLOAT,
# MAGIC   dst_host_serror_rate FLOAT,
# MAGIC   dst_host_srv_serror_rate FLOAT,
# MAGIC   dst_host_rerror_rate FLOAT,
# MAGIC   dst_host_srv_rerror_rate FLOAT
# MAGIC )
# MAGIC USING CSV
# MAGIC LOCATION '/mnt/blob_storage/kddcup.testdata.unlabeled.csv'
# MAGIC OPTIONS ("header"="false");
# MAGIC 
# MAGIC -- LACE: TODO, convert to databricks delta
# MAGIC DROP TABLE IF EXISTS kdd_unlabeled;
# MAGIC CREATE TABLE kdd_unlabeled 
# MAGIC USING org.apache.spark.sql.parquet
# MAGIC AS SELECT * FROM kdd_unlabeled_temp;
# MAGIC 
# MAGIC -- Drop temporary table
# MAGIC DROP TABLE kdd_unlabeled_temp;
