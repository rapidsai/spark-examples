Get Started with XGBoost4J-Spark with Apache Toree Jupyter Notebook
===================================================================
This is a getting started guide to XGBoost4J-Spark using an [Apache Toree](https://toree.apache.org/) Jupyter notebook. At the end of this guide, the reader will be able to run a sample notebook that runs on NVIDIA GPUs.

Before you begin, please ensure that you have setup a [Spark Standalone Cluster](/getting-started-guides/on-prem-cluster/standalone-scala.md).

It is assumed that the `SPARK_MASTER` and `SPARK_HOME` environment variables are defined and point to the master spark URL (e.g. `spark://localhost:7077`), and the home directory for Apache Spark respectively.

1. Make sure you have jupyter notebook installed first.
2. Download the 'toree' built against scala2.12 from [here](TBD) to local, and install it.
  ```
  # Install Toree
  pip install <local_path_toree>/toree-pip-0.5-SNAPSHOT.tar.gz
  ```

3. Install a new kernel configured for our example and with gpu enabled:
  ```
  export SPARK_EXAMPLES=[full path to spark-examples repo]
  export RAPIDS_JAR=[full path to rapids plugin jar]
  export SPARK_JARS=${SPARK_EXAMPLES}/sample_xgboost_apps-0.2.2-jar-with-dependencies.jar,${RAPIDS_JAR}

  jupyter toree install                                \
  --spark_home=${SPARK_HOME}                             \
  --user                                          \
  --toree_opts='--nosparkcontext'                         \
  --kernel_name="XGBoost4j-Spark"                         \
  --spark_opts='--master ${SPARK_MASTER} --jars ${SPARK_JARS}       \
    --conf spark.sql.extensions=ai.rapids.spark.Plugin \
    --conf spark.rapids.memory.gpu.pooling.enabled=false \
    --conf spark.executor.resource.gpu.amount=1 \
    --conf spark.task.resource.gpu.amount=1 \
    --conf spark.executor.resource.gpu.discoveryScript=./getGpusResources.sh \
    --files $SPARK_HOME/examples/src/main/scripts/getGpusResources.sh'
  ```

4. Launch the notebook:
  ```
  jupyter notebook
  ```

Then you start your notebook and open [`mortgage-gpu.ipynb`](/examples/notebooks/scala/mortgage-gpu.ipynb) to explore.

Please ensure that the *XGBoost4j-Spark* kernel is running.
