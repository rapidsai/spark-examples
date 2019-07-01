Get Started with XGBoost4J-Spark on Databricks
======================================================
This is a getting started guide to XGBoost4J-Spark on Databricks. At the end of this guide, the reader will be able to run a sample Apache Spark application that runs on NVIDIA GPUs on Databricks.

Prerequisites
-------------
* Apache Spark 2.4+ running in DataBricks Runtime 5.3 or 5.4 ML with GPU hardware. Make sure it matches requirements below. Use nodes with 1 GPU each.
* Hardware Requirements
  * NVIDIA Pascal™ GPU architecture or better
  * Multi-node clusters with homogenous GPU configuration
* Software Requirements
  * Ubuntu 16.04/CentOS
  * NVIDIA driver 410.48+
  * CUDA V10.0/9.2
  * NCCL 2.4.7

* The number of GPUs per node dictates the number of Spark executors that can run in that node. Each executor should only be allowed to run 1 task at any given time. 

Get Application Jar and Dataset
-------------------------------
* Jars: Please download the Rapids XGBoost4j Spark jars.  There are 3 jars.  
Download the required jars into a local directory. Databricks ML Runtime supports cuda 9.2 so download the correct jars. If the below commands don't work then you can go to Maven Central and search for the 3 jars: ai.rapids.cudf version 0.8-Beta ,ai.rapids.xgboost4j - version 0.90-Beta, and ai.rapids.xgboost4j-spark 0.90-Beta.  

```
$ wget https://search.maven.org/remotecontent?filepath=ai/rapids/cudf/0.8-Beta/cudf-0.8-Beta.jar
$ wget https://search.maven.org/remotecontent?filepath=ai/rapids/xgboost4j/0.90-Beta/xgboost4j-0.90-Beta.jar
$ wget https://search.maven.org/remotecontent?filepath=ai/rapids/xgboost4j-spark/0.90-Beta/xgboost4j-spark-0.90-Beta.jar
``` 

* Download Dataset: https://rapidsai-data.s3.us-east-2.amazonaws.com/spark/mortgage.zip

```
$ wget  https://rapidsai-data.s3.us-east-2.amazonaws.com/spark/mortgage.zip
$ unzip mortgage.zip
$ find . -type f -print|sort
./data/mortgage/csv/test/mortgage_eval_merged.csv
./data/mortgage/csv/train/mortgage_train_merged.csv
``` 

* Copy the data into Databricks DBFS:

```
Go to "Import and Explore Data" 
Create New Table - default location is /FileStore/tables
Select mortgage_eval_merged.csv and mortgage_train_merged.csv to upload.
```

* Upload the XGBoost-4j Spark jars

```
Go to "Import Library"
Select "Upload" and then "Jar"
Select the 3 XGBoost Spark jars and upload them. Save the file names and locations for the step below.
```

* Add a Startup init script  
In a terminal create an init script that has commands to copy the jars you imported above over the existing Databricks provided XGBoost jars. Note you will have to change the jar names below to the ones you uploaded.  If you didn't save the names go to the Databricks file browser and look in /FileStore/jars/ If you are using a runtime other then Databricks 5.3 or 5.4 you will have to check the versions of the databricks provided jars in /databricks/jars and update the script accordingly.  

```
$ cat /dbfs/databricks/scripts/init.sh
sudo cp /dbfs/FileStore/jars/2958ae11_357c_4f5d_b9ee_3212e3c0ec5c-xgboost4j_0_90_SNAPSHOT_cuda92-567a4.jar /databricks/jars/spark--maven-trees--ml--xgboost--ml.dmlc--xgboost4j--ml.dmlc__xgboost4j__0.81.jar
sudo cp /dbfs/FileStore/jars/492544bd_53eb_42b7_a9f3_ba682c991839-cudf_0_9_SNAPSHOT-64b01.jar /databricks/jars/
sudo cp /dbfs/FileStore/jars/7b5344c1_38b1_4f9c_a951_f1bce67eb20b-xgboost4j_spark_0_90_SNAPSHOT_cuda92_tom9-c8c97.jar /databricks/jars/spark--maven-trees--ml--xgboost--ml.dmlc--xgboost4j-spark--ml.dmlc__xgboost4j-spark__0.81.jar

Upload the init.sh script into /databricks/scripts/init.sh.  See https://docs.databricks.com/user-guide/clusters/init-scripts.html for more details about configuring cluster-scoped init script.
```

Start A Databricks Cluster
--------------------------
1. Create a Databricks cluster that meets the above prerequisites. Make sure to use one of the ML Databricks runtimes.
2. Disable autoscaling.
3. Choose the number of workers that matches the number of GPUs you want to use.
4. Select a worker type that has a GPU for the worker.
5. Update the cluster Configuration "Advanced Options" to include an "Init Scripts". Add your init.sh script you uploaded above: "dbfs:/databricks/scripts/init.sh"
6. Optionally add other configurations.

Import the GPU Mortgage Example Notebook
---------------------------
1. See https://docs.databricks.com/user-guide/notebooks/notebook-manage.html on how to import a notebook.
2. Import the notebook: https://github.com/rapidsai/spark-examples/tree/master/notebook/databricks/xgboost_notebook_mortgage-gpu_scala.scala

The example notebook comes with the following configuration, you can adjust this according to your setup.
```
val commParamMap = Map(
    "eta" -> 0.1,
    "gamma" -> 0.1,
    "missing" -> 0.0,
    "max_depth" -> 10,
    "max_leaves" -> 256,
    "grow_policy" -> "depthwise",
    "min_child_weight" -> 30,
    "lambda" -> 1,
    "scale_pos_weight" -> 2,
    "subsample" -> 1,
    "nthread" -> 1,
    "num_round" -> 100,
    "num_workers" -> 1,
    "tree_method" -> "gpu_hist")
```

3. Run all the cells in the notebook. 

4. View the results
In the cell 9 (Training), 10 (Transforming) and 11 (Accuracty of Evaluation) you will see the output.

```
--------------
==> Benchmark: Elapsed time for [train]: 26.776s
--------------

--------------
==> Benchmark: Elapsed time for [transform]: 0.073s
--------------

------Accuracy of Evaluation------
0.9875489273547466
```

<sup>*</sup> The timings in this Getting Started guide are only illustrative. Please see our [release announcement](https://medium.com/rapids-ai/nvidia-gpus-and-apache-spark-one-step-closer-2d99e37ac8fd) for official benchmarks.

