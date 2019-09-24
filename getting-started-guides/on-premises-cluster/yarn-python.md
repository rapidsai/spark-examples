Get Started with XGBoost4J-Spark on Apache Hadoop YARN
======================================================
This is a getting started guide to XGBoost4J-Spark on Apache Hadoop YARN. At the end of this guide, the reader will be able to run a sample Apache Spark Python application that runs on NVIDIA GPUs.

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
  * Python 2.7/3.4/3.5/3.6/3.7
  * NumPy
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
1. *samples.zip* and *main.py*: Please build the files as specified in the [guide](/getting-started-guides/building-sample-apps/python.md)
2. Jars: Please download the following jars:
    * [*cudf-0.9-cuda10.jar*](https://search.maven.org/remotecontent?filepath=ai/rapids/cudf/0.9/cudf-0.9-cuda10.jar) (For CUDA 9.2, please download [*cudf-0.9.jar*](https://search.maven.org/remotecontent?filepath=ai/rapids/cudf/0.9/cudf-0.9.jar) instead, and replace *cudf-0.9-cuda10.jar* with *cudf-0.9.jar* throughout this whole guide)
    * [*xgboost4j_2.11-1.0.0-Beta.jar*](https://search.maven.org/remotecontent?filepath=ai/rapids/xgboost4j_2.11/1.0.0-Beta/xgboost4j_2.11-1.0.0-Beta.jar)
    * [*xgboost4j-spark_2.11-1.0.0-Beta.jar*](https://search.maven.org/remotecontent?filepath=ai/rapids/xgboost4j-spark_2.11/1.0.0-Beta/xgboost4j-spark_2.11-1.0.0-Beta.jar)
3. Dataset: https://rapidsai-data.s3.us-east-2.amazonaws.com/spark/mortgage.zip

Place dataset and other files in a local directory. In this example the dataset was unzipped in the `xgboost4j_spark_python/data` directory, and all other files in the `xgboost4j_spark_python/libs` directory.

```
[xgboost4j_spark_python]$ find . -type f | sort
./data/mortgage/csv/test/mortgage_eval_merged.csv
./data/mortgage/csv/train/mortgage_train_merged.csv
./libs/cudf-0.9-cuda10.jar
./libs/main.py
./libs/samples.zip
./libs/xgboost4j_2.11-1.0.0-Beta.jar
./libs/xgboost4j-spark_2.11-1.0.0-Beta.jar
```

Create a directory in HDFS, and copy:

```
[xgboost4j_spark_python]$ hadoop fs -mkdir /tmp/xgboost4j_spark_python
[xgboost4j_spark_python]$ hadoop fs -copyFromLocal * /tmp/xgboost4j_spark_python
```

Verify that the jar and dataset are in HDFS:

```
[xgboost4j_spark_python]$ hadoop fs -find /tmp/xgboost4j_spark_python | grep "\." | sort
/tmp/xgboost4j_spark_python/data/mortgage/csv/test/mortgage_eval_merged.csv
/tmp/xgboost4j_spark_python/data/mortgage/csv/train/mortgage_train_merged.csv
/tmp/xgboost4j_spark_python/libs/cudf-0.9-cuda10.jar
/tmp/xgboost4j_spark_python/libs/main.py
/tmp/xgboost4j_spark_python/libs/samples.zip
/tmp/xgboost4j_spark_python/libs/xgboost4j_2.11-1.0.0-Beta.jar
/tmp/xgboost4j_spark_python/libs/xgboost4j-spark_2.11-1.0.0-Beta.jar
```

Launch GPU Mortgage Example
---------------------------
Variables required to run spark-submit command:

```
# location where data was downloaded
export DATA_PATH=hdfs:/tmp/xgboost4j_spark_python/data

# location for the required libraries
export LIBS_PATH=hdfs:/tmp/xgboost4j_spark_python/libs

# spark deploy mode (see Apache Spark documentation for more information)
export SPARK_DEPLOY_MODE=cluster

# run a single executor for this example to limit the number of spark tasks and
# partitions to 1 as currently this number must match the number of input files
export SPARK_NUM_EXECUTORS=1

# spark driver memory
export SPARK_DRIVER_MEMORY=4g

# spark executor memory
export SPARK_EXECUTOR_MEMORY=8g

# python entrypoint
export SPARK_PYTHON_ENTRYPOINT=${LIBS_PATH}/main.py

# example class to use
export EXAMPLE_CLASS=ai.rapids.spark.examples.mortgage.gpu_main

# additional jars for XGBoost4J example
export SPARK_JARS=${LIBS_PATH}/cudf-0.9-cuda10.jar,${LIBS_PATH}/xgboost4j_2.11-1.0.0-Beta.jar,${LIBS_PATH}/xgboost4j-spark_2.11-1.0.0-Beta.jar

# additional Python files for XGBoost4J example
export SPARK_PY_FILES=${LIBS_PATH}/xgboost4j-spark_2.11-1.0.0-Beta.jar,${LIBS_PATH}/samples.zip

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
 --jars ${SPARK_JARS}                                                           \
 --py-files ${SPARK_PY_FILES}                                                   \
 ${SPARK_PYTHON_ENTRYPOINT}                                                     \
 --mainClass=${EXAMPLE_CLASS}                                                   \
 --trainDataPath=${DATA_PATH}/mortgage/csv/train/mortgage_train_merged.csv      \
 --evalDataPath=${DATA_PATH}/mortgage/csv/test/mortgage_eval_merged.csv         \
 --format=csv                                                                   \
 --numWorkers=${SPARK_NUM_EXECUTORS}                                            \
 --treeMethod=${TREE_METHOD}                                                    \
 --numRound=100                                                                 \
 --maxDepth=8
```

In the `stdout` driver log, you should see timings<sup>*</sup> (in seconds), and the RMSE accuracy metric:

```
----------------------------------------------------------------------------------------------------
Training takes 10.75 seconds

----------------------------------------------------------------------------------------------------
Transformation takes 4.38 seconds

----------------------------------------------------------------------------------------------------
Accuracy is 0.997544753891
```

Launch CPU Mortgage Example
---------------------------
If you are running this example after running the GPU example above, please set these variables, to set both training and testing to run on the CPU exclusively:

```
# example class to use
export EXAMPLE_CLASS=ai.rapids.spark.examples.mortgage.cpu_main

# tree construction algorithm
export TREE_METHOD=hist
```

This is the full variable listing, if you are running the CPU example from scratch:

```
# location where data was downloaded
export DATA_PATH=hdfs:/tmp/xgboost4j_spark_python/data

# location for the required libraries
export LIBS_PATH=hdfs:/tmp/xgboost4j_spark_python/libs

# spark deploy mode (see Apache Spark documentation for more information)
export SPARK_DEPLOY_MODE=cluster

# run a single executor for this example to limit the number of spark tasks and
# partitions to 1 as currently this number must match the number of input files
export SPARK_NUM_EXECUTORS=1

# spark driver memory
export SPARK_DRIVER_MEMORY=4g

# spark executor memory
export SPARK_EXECUTOR_MEMORY=8g

# python entrypoint
export SPARK_PYTHON_ENTRYPOINT=${LIBS_PATH}/main.py

# example class to use
export EXAMPLE_CLASS=ai.rapids.spark.examples.mortgage.cpu_main

# additional jars for XGBoost4J example
export SPARK_JARS=${LIBS_PATH}/cudf-0.9-cuda10.jar,${LIBS_PATH}/xgboost4j_2.11-1.0.0-Beta.jar,${LIBS_PATH}/xgboost4j-spark_2.11-1.0.0-Beta.jar

# additional Python files for XGBoost4J example
export SPARK_PY_FILES=${LIBS_PATH}/xgboost4j-spark_2.11-1.0.0-Beta.jar,${LIBS_PATH}/samples.zip

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
 --jars ${SPARK_JARS}                                                           \
 --py-files ${SPARK_PY_FILES}                                                   \
 ${SPARK_PYTHON_ENTRYPOINT}                                                     \
 --mainClass=${EXAMPLE_CLASS}                                                   \
 --trainDataPath=${DATA_PATH}/mortgage/csv/train/mortgage_train_merged.csv      \
 --evalDataPath=${DATA_PATH}/mortgage/csv/test/mortgage_eval_merged.csv         \
 --format=csv                                                                   \
 --numWorkers=${SPARK_NUM_EXECUTORS}                                            \
 --treeMethod=${TREE_METHOD}                                                    \
 --numRound=100                                                                 \
 --maxDepth=8
```

In the `stdout` driver log, you should see timings<sup>*</sup> (in seconds), and the RMSE accuracy metric:

```
----------------------------------------------------------------------------------------------------
Training takes 10.76 seconds

----------------------------------------------------------------------------------------------------
Transformation takes 1.25 seconds

----------------------------------------------------------------------------------------------------
Accuracy is 0.998526852335
```

<sup>*</sup> The timings in this Getting Started guide are only illustrative. Please see our [release announcement](https://medium.com/rapids-ai/nvidia-gpus-and-apache-spark-one-step-closer-2d99e37ac8fd) for official benchmarks.

