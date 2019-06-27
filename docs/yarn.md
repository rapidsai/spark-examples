Get Started with XGBoost4J-Spark on Apache Hadoop YARN
======================================================
This is a getting started guide to XGBoost4J-Spark on Apache Hadoop YARN. At the end of this guide, the reader will be able to run a sample Apache Spark application that runs on NVIDIA GPUs.

Prerequisites
-------------
* Apache Spark 2.3+ running on YARN.
* Hardware Requirements
  * NVIDIA Pascal™ GPU architecture or better
  * Multi-node clusters with homogenous GPU configuration
* Software Requirements
  * Ubuntu 16.04/CentOS
  * NVIDIA driver 410.48+
  * CUDA V10.0/9.2
  * NCCL 2.4.7
* `EXCLUSIVE_PROCESS` must be set for all GPUs in each NodeManager. This can be accomplished using the `nvidia-smi` utility:

  ```
  nvidia-smi -i [gpu index] -c EXCLUSIVE_PROCESS
  ```
  
  For example:
  
  ```
  nvidia-smi -i 0 -c EXCLUSIVE_PROCESS
  ```
  
  Sets `EXCLUSIVE_PROCESS` for GPU _0_.
* The number of GPUs per NodeManager dictates the number of Spark executors that can run in that NodeManager. Additionally, cores per Spark executor and cores per Spark task must match, such that each executor can run 1 task at any given time. For example: if each NodeManager has 4 GPUs, there should be 4 executors running on each NodeManager, and each executor should run 1 task (for a total of 4 tasks running on 4 GPUs). In order to achieve this, you may need to adjust `spark.task.cpus` and `spark.executor.cores` to match (both set to 1 by default). Additionally, we recommend adjusting `executor-memory` to divide host memory evenly amongst the number of GPUs in each NodeManager, such that Spark will schedule as many executors as there are GPUs in each NodeManager.
* The `SPARK_HOME` environment variable is assumed to point to the cluster's Apache Spark installation.

Get Application Jar and Dataset
-------------------------------
1. Jar: Please build the sample_xgboost_apps jar with dependencies as specified in the [README](https://github.com/rapidsai/spark-examples)
2. Dataset: https://rapidsai-data.s3.us-east-2.amazonaws.com/spark/mortgage.zip

First place the required jar and dataset in a local directory. In this example the jar is in the `xgboost4j_spark/jars` directory, and the `mortgage.zip` dataset was unzipped in the `xgboost4j_spark/data` directory. 

```
[xgboost4j_spark]$ find . -type f -print|sort
./data/mortgage/csv/test/mortgage_eval_merged.csv
./data/mortgage/csv/train/mortgage_train_merged.csv
./jars/sample_xgboost_apps-0.1.4-jar-with-dependencies.jar
``` 

Create a directory in HDFS, and copy:

```
[xgboost4j_spark]$ hadoop fs -mkdir /tmp/xgboost4j_spark
[xgboost4j_spark]$ hadoop fs -copyFromLocal * /tmp/xgboost4j_spark
```

Verify that the jar and dataset are in HDFS:

```
[xgboost4j_spark]$ hadoop fs -find /tmp/xgboost4j_spark -print|grep "\."|sort
/tmp/xgboost4j_spark/data/mortgage/csv/test/mortgage_eval_merged.csv
/tmp/xgboost4j_spark/data/mortgage/csv/train/mortgage_train_merged.csv
/tmp/xgboost4j_spark/jars/sample_xgboost_apps-0.1.4-jar-with-dependencies.jar
```

Launch GPU Mortgage Example
---------------------------
Variables required to run spark-submit command:

```
# location where data was downloaded 
export DATA_PATH=hdfs:/tmp/xgboost4j_spark/data

# location for the required jar
export JARS_PATH=hdfs:/tmp/xgboost4j_spark/jars

# spark deploy mode (see Apache Spark documentation for more information) 
export SPARK_DEPLOY_MODE=cluster

# run a single executor for this example to limit the number of spark tasks and
# partitions to 1 as currently this number must match the number of input files
export SPARK_NUM_EXECUTORS=1

# spark driver memory
export SPARK_DRIVER_MEMORY=4g

# spark executor memory
export SPARK_EXECUTOR_MEMORY=8g

# example class to use
export EXAMPLE_CLASS=ai.rapids.spark.examples.mortgage.GPUMain

# XGBoost4J example jar
export JAR_EXAMPLE=${JARS_PATH}/sample_xgboost_apps-0.1.4-jar-with-dependencies.jar

# tree construction algorithm
export TREE_METHOD=gpu_hist
```

Run spark-submit:

```
${SPARK_HOME}/bin/spark-submit                                                  \
 --master yarn                                                                  \
 --deploy-mode ${SPARK_DEPLOY_MODE}                                             \
 --num-executors ${SPARK_NUM_EXECUTORS}                                         \
 --driver-memory ${SPARK_DRIVER_MEMORY}                                         \
 --executor-memory ${SPARK_EXECUTOR_MEMORY}                                     \
 --class ${EXAMPLE_CLASS}                                                       \
 ${JAR_EXAMPLE}                                                                 \
 -trainDataPath=${DATA_PATH}/mortgage/csv/train/mortgage_train_merged.csv       \
 -evalDataPath=${DATA_PATH}/mortgage/csv/test/mortgage_eval_merged.csv          \
 -format=csv                                                                    \
 -numWorkers=${SPARK_NUM_EXECUTORS}                                             \
 -treeMethod=${TREE_METHOD}                                                     \
 -numRound=100                                                                  \
 -maxDepth=8                                                                    
```

In the `stdout` driver log, you should see timings<sup>*</sup> (in seconds), and the RMSE accuracy metric:

```
--------------
==> Benchmark: Elapsed time for [train]: 29.642s
--------------

--------------
==> Benchmark: Elapsed time for [transform]: 21.272s
--------------

------Accuracy of Evaluation------
0.9874184013493451
```

Launch CPU Mortgage Example
---------------------------
If you are running this example after running the GPU example above, please set these variables, to set both training and testing to run on the CPU exclusively:

```
# example class to use
export EXAMPLE_CLASS=ai.rapids.spark.examples.mortgage.CPUMain

# tree construction algorithm
export TREE_METHOD=hist
```

This is the full variable listing, if you are running the CPU example from scratch:

```
# location where data was downloaded 
export DATA_PATH=hdfs:/tmp/xgboost4j_spark/data

# location where required jar was downloaded
export JARS_PATH=hdfs:/tmp/xgboost4j_spark/jars

# spark deploy mode (see Apache Spark documentation for more information) 
export SPARK_DEPLOY_MODE=cluster

# run a single executor for this example to limit the number of spark tasks and
# partitions to 1 as currently this number must match the number of input files
export SPARK_NUM_EXECUTORS=1

# spark driver memory
export SPARK_DRIVER_MEMORY=4g

# spark executor memory
export SPARK_EXECUTOR_MEMORY=8g

# example class to use
export EXAMPLE_CLASS=ai.rapids.spark.examples.mortgage.CPUMain

# XGBoost4J example jar
export JAR_EXAMPLE=${JARS_PATH}/sample_xgboost_apps-0.1.4-jar-with-dependencies.jar

# tree construction algorithm
export TREE_METHOD=hist
```

This is the same command as for the GPU example, repeated for convenience:

```
${SPARK_HOME}/bin/spark-submit                                                  \
 --master yarn                                                                  \
 --deploy-mode ${SPARK_DEPLOY_MODE}                                             \
 --num-executors ${SPARK_NUM_EXECUTORS}                                         \
 --driver-memory ${SPARK_DRIVER_MEMORY}                                         \
 --executor-memory ${SPARK_EXECUTOR_MEMORY}                                     \
 --class ${EXAMPLE_CLASS}                                                       \
 ${JAR_EXAMPLE}                                                                 \
 -trainDataPath=${DATA_PATH}/mortgage/csv/train/mortgage_train_merged.csv       \
 -evalDataPath=${DATA_PATH}/mortgage/csv/test/mortgage_eval_merged.csv          \
 -format=csv                                                                    \
 -numWorkers=${SPARK_NUM_EXECUTORS}                                             \
 -treeMethod=${TREE_METHOD}                                                     \
 -numRound=100                                                                  \
 -maxDepth=8                                                                    
```

In the `stdout` driver log, you should see timings<sup>*</sup> (in seconds), and the RMSE accuracy metric:

```
--------------
==> Benchmark: Elapsed time for [train]: 286.398s
--------------

--------------
==> Benchmark: Elapsed time for [transform]: 49.836s
--------------

------Accuracy of Evaluation------
0.9873709530950067 
```

<sup>*</sup> The timings in this Getting Started guide are only illustrative. Please see our [release announcement](https://medium.com/rapids-ai/nvidia-gpus-and-apache-spark-one-step-closer-2d99e37ac8fd) for official benchmarks.

