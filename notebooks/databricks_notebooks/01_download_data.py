# Databricks notebook source
# MAGIC %md
# MAGIC # Mount blob storage

# COMMAND ----------

# Set mount path
storage_mount_path = "/mnt/blob_storage"

# Unmount if existing
for mp in dbutils.fs.mounts():
  if mp.mountPoint == storage_mount_path:
    dbutils.fs.unmount(storage_mount_path)

# Refresh mounts
dbutils.fs.refreshMounts()

# COMMAND ----------

# Retrieve storage credentials
storage_account = dbutils.secrets.get(scope = "storage_scope", key = "storage_account")
storage_key = dbutils.secrets.get(scope = "storage_scope", key = "storage_key")

# Try to print out:
storage_key

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
  
  # Create dir if not exist
  dir_path = os.path.dirname(out_file)
  if not os.path.exists(dir_path):
    os.makedirs(dir_path)
    
  # Uncompress
  with gzip.open(tmp_loc, 'rb') as f_in:
    with open(out_file, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
        
  # Cleanup
  os.remove(tmp_loc)
  

# Note that Azure Databricks configures each cluster node with a FUSE mount that allows processes running on cluster nodes to read and write to the underlying
# distributed storage layer with local file APIs
# See here: https://docs.azuredatabricks.net/user-guide/dbfs-databricks-file-system.html#access-dbfs-using-local-file-apis
# 'https://archive.ics.uci.edu/ml/machine-learning-databases/kddcup99-mld/kddcup.data.gz'
download_and_uncompress_gz(data_url='https://lacedemodata.blob.core.windows.net/data/kddcup.data.gz',
                           out_file='/dbfs' + storage_mount_path + '/data/raw/kddcup.data.csv')

# 'http://kdd.ics.uci.edu/databases/kddcup99/kddcup.testdata.unlabeled.gz'
download_and_uncompress_gz(data_url='https://lacedemodata.blob.core.windows.net/data/kddcup.testdata.unlabeled.gz',
                           out_file='/dbfs' + storage_mount_path + '/data/raw/kddcup.testdata.unlabeled.csv')