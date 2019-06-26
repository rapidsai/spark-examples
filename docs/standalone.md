Get Started with XGBoost4J-Spark on an Apache Spark Standalone Cluster
======================================================================
This is a getting started guide to XGBoost4J-Spark on an Apache Spark Standalone Cluster. At the end of this guide, the reader will be able to run a sample Apache Spark application that runs on NVIDIA GPUs.

Prerequisites
-------------
* Apache Spark 2.3+ Standalone Cluster
* Hardware Requirements
  * NVIDIA Pascal™ GPU architecture or better
  * Multi-node clusters with homogenous GPU configuration
* Software Requirements
  * Ubuntu 16.04/CentOS
  * NVIDIA driver 410.48+
  * CUDA V10.0/9.2
  * NCCL 2.4.7
* `EXCLUSIVE_PROCESS` must be set for all GPUs in each host. This can be accomplished using the `nvidia-smi` utility:

  ```
  nvidia-smi -i [gpu index] -c EXCLUSIVE_PROCESS
  ```
  
  For example:
  
  ```
  nvidia-smi -i 0 -c EXCLUSIVE_PROCESS
  ```
  
  Sets `EXCLUSIVE_PROCESS` for GPU _0_.
* The number of GPUs in each host dictates the number of Spark executors that can run there. Additionally, cores per Spark executor and cores per Spark task must match, such that each executor can run 1 task at any given time. For example: if each host has 4 GPUs, there should be 4 executors running on each host, and each executor should run 1 task (for a total of 4 tasks running on 4 GPUs). In Spark Standalone mode, the default configuration is for an executor to take up all the cores assigned to each Spark Worker. In this example, we will limit the number of cores to 1, to match our dataset. Please see https://spark.apache.org/docs/latest/spark-standalone.html for more documentation regarding Standalone configuration.
* The `SPARK_HOME` environment variable is assumed to point to the cluster's Apache Spark installation.

Get Application Jar and Dataset
-------------------------------
1. Jar: Please build the sample_xgboost_apps jar with dependencies as specified in the [README](https://github.com/rapidsai/spark-examples)
2. Dataset: https://rapidsai-data.s3.us-east-2.amazonaws.com/spark/mortgage.zip

Place the required jar and dataset in a local directory. In this example the jar is in the `xgboost4j_spark/jars` directory, and the `mortgage.zip` dataset was unzipped in the `xgboost4j_spark/data` directory. 

```
[xgboost4j_spark]$ find . -type f -print|sort
./data/mortgage/csv/test/mortgage_eval_merged.csv
./data/mortgage/csv/train/mortgage_train_merged.csv
./jars/sample_xgboost_apps-0.1.4-jar-with-dependencies.jar
``` 

Launch a Standalone Spark Cluster
---------------------------------
1. Start the Spark Master process:

```
${SPARK_HOME}/sbin/start-master.sh
```

Note the hostname or ip address of the Master host, so that it can be given to each Worker process, in this example the Master and Worker will run on the same host.

2. Start a Spark slave process:

```
export SPARK_MASTER=spark://`hostname -f`:7077
export SPARK_CORES_PER_WORKER=1

${SPARK_HOME}/sbin/start-slave.sh ${SPARK_MASTER} -c ${SPARK_CORES_PER_WORKER} 
```

Note that in this example the Master and Worker processes are both running on the same host. This is not a requirement, as long as all hosts that are used to run the Spark app have access to the dataset.

Launch GPU Mortgage Example
---------------------------
Variables required to run spark-submit command:

```
# this is the same master host we defined while launching the cluster
export SPARK_MASTER=spark://`hostname -f`:7077

# location where data was downloaded 
export DATA_PATH=./xgboost4j_spark/data

# location for the required jar
export JARS_PATH=./xgboost4j_spark/jars

# Currently the number of tasks and executors must match the number of input files.
# For this example, we will set these such that we have 1 executor, with 1 core per executor

## take up the the whole worker
export SPARK_CORES_PER_EXECUTOR=${SPARK_CORES_PER_WORKER}

## run 1 executor
export SPARK_NUM_EXECUTORS=1

## cores/executor * num_executors, which in this case is also 1, limits
## the number of cores given to the application
export TOTAL_CORES=$((SPARK_CORES_PER_EXECUTOR * SPARK_NUM_EXECUTORS))

# spark driver memory
export SPARK_DRIVER_MEMORY=4g

# spark executor memory
export SPARK_EXECUTOR_MEMORY=8g

# example class to use
export EXAMPLE_CLASS=ai.rapids.spark.examples.mortgage.GPUMain

# XGBoost4J example jar (holds example classes):
export JAR_EXAMPLE=${JARS_PATH}/sample_xgboost_apps-0.1.4-jar-with-dependencies.jar

# tree construction algorithm
export TREE_METHOD=gpu_hist
```

Run spark-submit:

```
${SPARK_HOME}/bin/spark-submit                                                  \
 --master ${SPARK_MASTER}                                                       \
 --driver-memory ${SPARK_DRIVER_MEMORY}                                         \
 --executor-memory ${SPARK_EXECUTOR_MEMORY}                                     \
 --conf spark.cores.max=${TOTAL_CORES}                                          \
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
==> Benchmark: Elapsed time for [train]: 28.005s
--------------

--------------
==> Benchmark: Elapsed time for [transform]: 21.287s
--------------

------Accuracy of Evaluation------
0.9874260661801348
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
# this is the same master host we defined while launching the cluster
export SPARK_MASTER=spark://`hostname -f`:7077

# location where data was downloaded 
export DATA_PATH=./xgboost4j_spark/data

# location where the required jar was downloaded
export JARS_PATH=./xgboost4j_spark/jars

# Currently the number of tasks and executors must match the number of input files.
# For this example, we will set these such that we have 1 executor, with 1 core per executor

## take up the the whole worker
export SPARK_CORES_PER_EXECUTOR=${SPARK_CORES_PER_WORKER}

## run 1 executor
export SPARK_NUM_EXECUTORS=1

## cores/executor * num_executors, which in this case is also 1, limits
## the number of cores given to the application
export TOTAL_CORES=$((SPARK_CORES_PER_EXECUTOR * SPARK_NUM_EXECUTORS))

# spark driver memory
export SPARK_DRIVER_MEMORY=4g

# spark executor memory
export SPARK_EXECUTOR_MEMORY=8g

# example class to use
export EXAMPLE_CLASS=ai.rapids.spark.examples.mortgage.CPUMain

# XGBoost4J example jar (holds example classes):
export JAR_EXAMPLE=${JARS_PATH}/sample_xgboost_apps-0.1.4-jar-with-dependencies.jar

# tree construction algorithm
export TREE_METHOD=hist
```

This is the same command as for the GPU example, repeated for convenience:

```
${SPARK_HOME}/bin/spark-submit                                                  \
 --master ${SPARK_MASTER}                                                       \
 --driver-memory ${SPARK_DRIVER_MEMORY}                                         \
 --executor-memory ${SPARK_EXECUTOR_MEMORY}                                     \
 --conf spark.cores.max=${TOTAL_CORES}                                          \
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
==> Benchmark: Elapsed time for [train]: 306.936s
--------------

--------------
==> Benchmark: Elapsed time for [transform]: 52.867s
--------------

------Accuracy of Evaluation------
0.9873709530950067
```

<sup>*</sup> The timings in this Getting Started guide are only illustrative. 
