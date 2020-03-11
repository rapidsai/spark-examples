Get Started with XGBoost4J-Spark with Apache Toree Jupyter Notebook
===================================================================
This is a getting started guide to XGBoost4J-Spark using an [Apache Toree](https://toree.apache.org/) Jupyter notebook. At the end of this guide, the reader will be able to run a sample notebook that runs on NVIDIA GPUs.

Before you begin, please ensure that you have setup a [Spark Standalone Cluster](/getting-started-guides/on-prem-cluster/standalone-scala.md).

It is assumed that the `SPARK_MASTER` and `SPARK_HOME` environment variables are defined and point to the master spark URL (e.g. `spark://localhost:7077`), and the home directory for Apache Spark respectively.

1. Make sure you have jupyter notebook installed:
  Install Toree:
  ```
  pip install toree
  ```

2. Install kernel configured for our example:
  ```
  export SPARK_EXAMPLES=[full path to spark-examples repo]
  export SPARK_JARS=${SPARK_EXAMPLES}/sample_xgboost_apps-0.1.5-jar-with-dependencies.jar

  jupyter toree install                                                             \
  --spark_home=${SPARK_HOME}                                                        \
  --user                                                                            \
  --kernel_name="XGBoost4j-Spark"                                                   \
  --spark_opts='--master ${SPARK_MASTER} --jars ${SPARK_JARS}'  
  ```

2. Launch the notebook:
  ```
  jupyter notebook
  ```

Then you start your notebook and open [`mortgage-gpu.ipynb`](/examples/notebooks/scala/mortgage-gpu.ipynb) to explore.

Please ensure that the *XGBoost4j-Spark* kernel is running.
