Get Started with XGBoost4J-Spark on Apache Hadoop YARN
======================================================
This is a getting started guide to XGBoost4J-Spark on Apache Hadoop YARN supporting GPU scheduling. At the end of this guide, the reader will be able to run a sample Apache Spark application that runs on NVIDIA GPUs.

Prerequisites
-------------
* Apache Spark 3.0+ running on YARN supporting GPU scheduling. (e.g.: Spark 3.0-preview2, Hadoop-Yarn 3.1.0)
* Hardware Requirements
  * NVIDIA Pascal™ GPU architecture or better
  * Multi-node clusters with homogenous GPU configuration
* Software Requirements
  * Ubuntu 16.04/CentOS7
  * CUDA V10.1/10.0   （CUDA 9.2 is no longer supported）
  * NVIDIA driver compatible with your CUDA
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
* The number of GPUs per NodeManager dictates the number of Spark executors that can run in that NodeManager. Additionally, cores per Spark executor and cores per Spark task must match, such that each executor can run 1 task at any given time. For example: if each NodeManager has 4 GPUs, there should be 4 or less executors running on each NodeManager, and each executor should run 1 task (e.g.: A total of 4 tasks running on 4 GPUs). In order to achieve this, you may need to adjust `spark.task.cpus` and `spark.executor.cores` to match (both set to 1 by default). Additionally, we recommend adjusting `executor-memory` to divide host memory evenly amongst the number of GPUs in each NodeManager, such that Spark will schedule as many executors as there are GPUs in each NodeManager.
* The `SPARK_HOME` environment variable is assumed to point to the cluster's Apache Spark installation.
* Enable GPU scheduling and isolation in Hadoop Yarn on each host. Please refe to [here](https://hadoop.apache.org/docs/r3.1.0/hadoop-yarn/hadoop-yarn-site/UsingGpus.html) for more details.

Get Jars and Dataset
-------------------------------
1. Application Jar: Please build the sample_xgboost_apps jar with dependencies as specified in the [guide](/getting-started-guides/building-sample-apps/scala.md)
2. Rapids Plugin Jar: You can download it from [here](TBD)
3. Dataset: https://rapidsai-data.s3.us-east-2.amazonaws.com/spark/mortgage.zip

First place the required jar and dataset in a local directory. In this example the jar is in the `xgboost4j_spark/jars` directory, and the `mortgage.zip` dataset was unzipped in the `xgboost4j_spark/data` directory. 

```
[xgboost4j_spark]$ find . -type f -print|sort
./data/mortgage/csv/test/mortgage_eval_merged.csv
./data/mortgage/csv/train/mortgage_train_merged.csv
./jars/rapids-4-spark-1.0-preview2.jar
./jars/sample_xgboost_apps-0.2.2-jar-with-dependencies.jar
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
/tmp/xgboost4j_spark/jars/rapids-4-spark-1.0-preview2.jar
/tmp/xgboost4j_spark/jars/sample_xgboost_apps-0.2.2-jar-with-dependencies.jar
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
export JAR_EXAMPLE=${JARS_PATH}/sample_xgboost_apps-0.2.2-jar-with-dependencies.jar

# Rapids plugin jar, working as the sql plugin on Spark3.0
export JAR_RAPIDS=${JARS_PATH}/rapids-4-spark-1.0-preview2.jar

# tree construction algorithm
export TREE_METHOD=gpu_hist
```

Run spark-submit:

```
${SPARK_HOME}/bin/spark-submit                                                  \
 --conf spark.sql.extensions=ai.rapids.spark.Plugin \
 --conf spark.rapids.memory.gpu.pooling.enabled=false \
 --conf spark.executor.resource.gpu.amount=1 \
 --conf spark.task.resource.gpu.amount=1 \
 --conf spark.executor.resource.gpu.discoveryScript=./getGpusResources.sh \
 --files $SPARK_HOME/examples/src/main/scripts/getGpusResources.sh \
 --jars ${JAR_RAPIDS}                                           \
 --master yarn                                                                  \
 --deploy-mode ${SPARK_DEPLOY_MODE}                                             \
 --num-executors ${SPARK_NUM_EXECUTORS}                                         \
 --driver-memory ${SPARK_DRIVER_MEMORY}                                         \
 --executor-memory ${SPARK_EXECUTOR_MEMORY}                                     \
 --class ${EXAMPLE_CLASS}                                                       \
 ${JAR_EXAMPLE}                                                                 \
 -dataPath=train::${DATA_PATH}/mortgage/csv/train/mortgage_train_merged.csv       \
 -dataPath=trans::${DATA_PATH}/mortgage/csv/test/mortgage_eval_merged.csv          \
 -format=csv                                                                    \
 -numWorkers=${SPARK_NUM_EXECUTORS}                                             \
 -treeMethod=${TREE_METHOD}                                                     \
 -numRound=100                                                                  \
 -maxDepth=8                                                                    
```

In the `stdout` driver log, you should see timings<sup>*</sup> (in seconds), and the accuracy metric:

```
--------------
==> Benchmark: Elapsed time for [Mortgage GPU train csv stub Unknown Unknown Unknown]: 29.642s
--------------

--------------
==> Benchmark: Elapsed time for [Mortgage GPU transform csv stub Unknown Unknown Unknown]: 21.272s
--------------

--------------
==> Benchmark: Accuracy for [Mortgage GPU Accuracy csv stub Unknown Unknown Unknown]: 0.9874184013493451
--------------
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
export JAR_EXAMPLE=${JARS_PATH}/sample_xgboost_apps-0.2.2-jar-with-dependencies.jar

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
 -dataPath=train::${DATA_PATH}/mortgage/csv/train/mortgage_train_merged.csv       \
 -dataPath=trans::${DATA_PATH}/mortgage/csv/test/mortgage_eval_merged.csv          \
 -format=csv                                                                    \
 -numWorkers=${SPARK_NUM_EXECUTORS}                                             \
 -treeMethod=${TREE_METHOD}                                                     \
 -numRound=100                                                                  \
 -maxDepth=8                                                                    
```

In the `stdout` driver log, you should see timings<sup>*</sup> (in seconds), and the accuracy metric:

```
--------------
==> Benchmark: Elapsed time for [Mortgage CPU train csv stub Unknown Unknown Unknown]: 286.398s
--------------

--------------
==> Benchmark: Elapsed time for [Mortgage CPU transform csv stub Unknown Unknown Unknown]: 49.836s
--------------

--------------
==> Benchmark: Accuracy for [Mortgage CPU Accuracy csv stub Unknown Unknown Unknown]: 0.9873709530950067
--------------
```

<sup>*</sup> The timings in this Getting Started guide are only illustrative. Please see our [release announcement](https://medium.com/rapids-ai/nvidia-gpus-and-apache-spark-one-step-closer-2d99e37ac8fd) for official benchmarks.

