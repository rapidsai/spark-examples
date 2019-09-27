Get Started with XGBoost4J-Spark on Databricks
======================================================
This is a getting started guide to XGBoost4J-Spark on Databricks. At the end of this guide, the reader will be able to run a sample Apache Spark application that runs on NVIDIA GPUs on Databricks.

Prerequisites
-------------
* Apache Spark 2.4+ running in DataBricks Runtime 5.3 ML with GPU, 5.4 ML with GPU, or 5.5 ML with GPU.  Make sure it matches the hardware and software requirements below.
* Hardware Requirements
  * NVIDIA Pascal™ GPU architecture or better
  * Multi-node clusters with homogenous GPU configuration
* Software Requirements
  * Ubuntu 16.04/CentOS
  * NVIDIA driver 410.48+
  * CUDA V10.0/9.2
  * NCCL 2.4.7

The number of GPUs per node dictates the number of Spark executors that can run in that node. Each executor should only be allowed to run 1 task at any given time.

Start A Databricks Cluster
--------------------------
Create a Databricks cluster (`Clusters` -> `+ Create Cluster`) that meets the above prerequisites.
1. Make sure to use one of the 5.3 ML with GPU, 5.4 ML with GPU, or 5.5 LTS ML with GPU Databricks runtimes.  
2. Use nodes with 1 GPU each such as p3.xlarge or Standard\_NC6s\_v3. We currently don't support nodes with multiple GPUs.  p2 (AWS) and NC12/24 (Azure) nodes do not meet the architecture requirements for the XGBoost worker (although they can be used for the driver node).  
3. Under Autopilot Options, disable autoscaling.
4. Choose the number of workers that matches the number of GPUs you want to use.
5. Select a worker type that has 1 GPU for the worker like p3.xlarge or NC6s_v3, for example.


* After you start a Databricks cluster, use the initialization notebooks -- [5.3 & 5.4 notebook](/getting-started-guides/csp/databricks/init-notebook-for-rapids-spark-xgboost-on-databricks-gpu-5.3-5.4.ipynb
) or [5.5 notebook](/getting-started-guides/csp/databricks/init-notebook-for-rapids-spark-xgboost-on-databricks-gpu-5.5.ipynb
) to setup execution.

The initialization notebooks will perform the following steps:
Downloading the CUDA and Rapids XGBoost4j Spark jars
Creating a new directory for initialization script in Databricks file system (DBFS)
Creating an initialization script inside the new directory to copy jars inside Databricks jar directory
Download and decompress the Sample Mortgage Notebook dataset 

After executing the steps in the initialization notebook, please follow the 1. Cluster initialization script and 2. Install the xgboost4j_spark jar in the cluster to ensure it is ready for XGBoost training.

Add cluster initialization script
---------------------------
1. See [Initialization scripts](https://docs.databricks.com/user-guide/clusters/init-scripts.html) for how to configure cluster initialization scripts.
2. Edit your cluster, adding an initialization script from dbfs:/databricks/init_scripts/init.sh in the "Advanced Options" under "Init Scripts" tab
3. Reboot the cluster


Install the xgboost4j_spark jar in the cluster
---------------------------
1. See [Libraries](https://docs.databricks.com/user-guide/libraries.html) for how to install jars from DBFS
2. Go to "Libraries" tab under your cluster and install dbfs:/FileStore/jars/xgboost4j-spark_2.11-1.0.0-Beta.jar in your cluster by selecting the "DBFS" option for installing jars

These steps will ensure you have a GPU Cluster ready for importing XGBoost notebooks or create your own XGBoost Application for training.


Import the GPU Mortgage Example Notebook
---------------------------
1. See [Managing Notebooks](https://docs.databricks.com/user-guide/notebooks/notebook-manage.html) on how to import a notebook.
2. Import the example notebook: [XGBoost4j-Spark mortgage notebook](/examples/notebooks/python/mortgage-gpu.ipynb)
3. Inside the mortgage example notebook, update the data paths from 
"/data/datasets/mortgage-small/train" to "dbfs:/FileStore/tables/mortgage/csv/train/mortgage_train_merged.csv"
"/data/datasets/mortgage-small/eval" to "dbfs:/FileStore/tables/mortgage/csv/test/mortgage_eval_merged.csv"

The example notebook comes with the following configuration, you can adjust this according to your setup.
See supported configuration options here: [xgboost parameters](/examples/app-parameters/supported_xgboost_parameters_python.md)
```
params = { 
    'eta': 0.1,
    'gamma': 0.1,
    'missing': 0.0,
    'treeMethod': 'gpu_hist',
    'maxDepth': 10, 
    'maxLeaves': 256,
    'growPolicy': 'depthwise',
    'minChildWeight': 30.0,
    'lambda_': 1.0,
    'scalePosWeight': 2.0,
    'subsample': 1.0,
    'nthread': 1,
    'numRound': 100,
    'numWorkers': 1,
}

```

4. Run all the cells in the notebook.

5. View the results
In the cell 5 (Training), 7 (Transforming) and 8 (Accuracy of Evaluation) you will see the output.

```
--------------
==> Benchmark: 
Training takes 6.48 seconds
--------------

--------------
==> Benchmark: Transformation takes 3.2 seconds

--------------

------Accuracy of Evaluation------
Accuracy is 0.9980699597729774

```

<sup>*</sup> The timings in this Getting Started guide are only illustrative. Please see our [release announcement](https://medium.com/rapids-ai/nvidia-gpus-and-apache-spark-one-step-closer-2d99e37ac8fd) for official benchmarks.



