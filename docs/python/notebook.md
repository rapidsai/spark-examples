Get Started with XGBoost4J-Spark with Jupyter Notebook
===================================================================
This is a getting started guide to XGBoost4J-Spark using an [Jupyter notebook](https://jupyter.org/). At the end of this guide, the reader will be able to run a sample notebook that runs on NVIDIA GPUs.

Before you begin, please ensure that you have setup a [Spark Standalone Cluster](/docs/python/standalone.md).

It is assumed that the `SPARK_MASTER` and `SPARK_HOME` environment variables are defined and point to the master spark URL (e.g. `spark://localhost:7077`), and the home directory for Apache Spark respectively.

1. Make sure you have [Jupyter notebook installed](https://jupyter.org/install.html). If you install it with conda, please makes sure your Python version is consistent.

2. Make sure you have `SPARK_JARS` and `SPARK_PY_FILES` set properly. You may need to update these env variables because the working directory will be changed:
  ```
  export LIBS_PATH=[full path to xgboost4j_spark/libs]
  export SPARK_JARS=${LIBS_PATH}/cudf-0.9-cuda10.jar,${LIBS_PATH}/xgboost4j_2.11-1.0.0-Beta.jar,${LIBS_PATH}/xgboost4j-spark_2.11-1.0.0-Beta.jar
  export SPARK_PY_FILES=${LIBS_PATH}/xgboost4j-spark_2.11-1.0.0-Beta.jar,${LIBS_PATH}/samples.zip
  ```

3. Go to the project root directory and launch the notebook:
  ```
  PYSPARK_DRIVER_PYTHON=jupyter \
  PYSPARK_DRIVER_PYTHON_OPTS=notebook \
  pyspark \
  --master ${SPARK_MASTER} \
  --jars ${SPARK_JARS} \
  --py-files ${SPARK_PY_FILES}
  ```

Then you start your notebook and open [`mortgage-gpu.ipynb`](/notebook/python/mortgage-gpu.ipynb) to explore.
