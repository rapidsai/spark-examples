This repo provides examples about how to use GPU powered XGBoost-Spark to train XGBoost model and use it to do predictions. 

Then content mainly contains：

  * [Prepare Example Jar](#Prepare-Example-Jar)
  * [Prepare Dataset](#Prepare-Dataset)
  * [Run Example App](#Run-Example-App)


# Prepare Example Jar
We use [maven](https://maven.apache.org/) to build jar package.

Our example relies on [cuDF](https://github.com/rapidsai/cudf) and [XGBoost](https://github.com/rapidsai/xgboost)


## Example App Jars
You should build the jar from current repo.

Suppose $EXAMPLE_HOME points to the directory where you place example repo.

```bash
cd $EXAMPLE_HOME/spark-examples
mvn package 
```
The command above will build a jar package with default cuda version `9.2`. If your cuda version is 10.0, you should do:

```bash
mvn package -DxgbClassifier=cuda10
```

Then you will find both `sample_xgboost_apps-0.1.4.jar` and `sample_xgboost_apps-0.1.4-jar-with-dependencies.jar` in your `target` folder.

`sample_xgboost_apps-0.1.4-jar-with-dependencies.jar` contains `cudf`, `xgboost4j` and `xgboost4j-spark` dependency jars while `sample_xgboost_apps-0.1.4.jar` doesn't. When we use `spark-submit` to run our example apps, we need to add `--jars` to attach all dependency jars if we submit `sample_xgboost_app-0.1.4.jar`. Details are in [spark-submit](#spark-submit). On the other hand, if we submit `sample_xgboost_apps-0.1.4-jar-with-dependencies.jar`, we don't have to add `--jar` parameter.

# Prepare Dataset

We have 3 example apps, you can choose to download transformed trainable dataset directly or download raw data, and run transformation jobs on your own.

## Download Trainable Dataset

You can get a small size transformed dataset for each example in `datasets` folder in this repo: 

1. [Mortgage Dataset(csv)](https://github.com/rapidsai/spark-examples/blob/master/datasets/mortgage-small.tar.gz?raw=true)
2. [Taxi Dataset(csv)](https://github.com/rapidsai/spark-examples/blob/master/datasets/taxi-small.tar.gz?raw=true)
3. [Agaricus(csv)](https://github.com/rapidsai/spark-examples/blob/master/datasets/agaricus.tar.gz?raw=true)


Let's take the Mortgage app for example, we need to extract the dataset from tar.gz file and put it in `/data/mortgage`:

```bash
tar -xvzf mortgage-small.tar.gz -C /data/mortgage
```

## Run an ETL job on your own
Or you could download raw dataset and run ETL(data transformation) jobs since the raw data is not trainable.

### For Mortgage
1. download raw data: https://rapidsai.github.io/demos/datasets/mortgage-data
2. install [jupyter notebook with Toree](#jupyternotebook)
3. run [Mortgage ETL job](https://github.com/rapidsai/spark-examples/xgboost/notebook/ETL/MortgageETL.ipynb)

### For Taxi
1. download raw data:
```bash
wget https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_20{09..16}-{01..12}.csv
```
2. install cudatoolkit and numba:
```bash
conda install numba
conda install cudatoolkit
```
we use `conda` to install pacakges here, you can also use `pip`.

1. run [Taxi ETL job](https://github.com/rapidsai/spark-examples/xgboost/notebook/ETL/Taxi_ETL.ipynb)


# Run Example App

We have two ways to demonstrate our apps: Run our apps with `spark-submit` in a console or set up a jupyter notebook with [`Toree`](https://toree.apache.org/)

## spark-submit
Here we only use a small part of the Mortgage dataset to demo, e.g. dataset of 2000Q1:

Run GPU version:
```bash
spark-submit --class ai.rapids.spark.examples.mortgage.GPUMain --master spark://$HOSTNAME:7077 \
 --executor-memory 32G \
 --jars /data/spark/libs/cudf-0.8-SNAPSHOT-cuda10.jar,/data/spark/libs/xgboost4j-0.90-SNAPSHOT.jar,/data/spark/libs/xgboost4j-spark-0.90-SNAPSHOT.jar \
/data/spark/libs/sample_xgboost_apps-0.1.3.jar \
-format=csv \
-num_round=100 \
-trainDataPath=/data/mortgage/csv/2009Q1/train/ \
-evalDataPath=/data/mortgage/csv/2009Q1/test \
-modelPath=/tmp/models/mortgage \
-overwrite=true
```

then you will see its logs like:

```console
19/06/13 17:00:39 WARN Utils: Your hostname, pc resolves to a loopback address: 127.0.1.1; using 10.19.183.124 instead (on interface eno1)
19/06/13 17:00:39 WARN Utils: Set SPARK_LOCAL_IP if you need to bind to another address
19/06/13 17:00:39 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
19/06/13 17:00:40 WARN Utils: Service 'SparkUI' could not bind on port 4040. Attempting port 4041.

------ Training ------
19/06/13 17:00:46 WARN XGBoostSpark: Missing weight column!
Tracker started, with env={DMLC_NUM_SERVER=0, DMLC_TRACKER_URI=10.19.183.124, DMLC_TRACKER_PORT=9091, DMLC_NUM_WORKER=1}
Elapsed time [train]: 23.251s

------ Transforming ------
Elapsed time [transform]: 0.914s
...
...
```

Run CPU version:

```bash
spark-submit --class ai.rapids.spark.examples.mortgage.CPUMain --master spark://$HOSTNAME:7077 \
 --executor-memory 32G \
 --conf spark.task.cpus=2 \
 --jars  /data/spark/libs/cudf-0.8-SNAPSHOT-cuda10.jar,/data/spark/libs/xgboost4j-0.90-SNAPSHOT.jar,/data/spark/libs/xgboost4j-spark-0.90-SNAPSHOT.jar \
/data/spark/libs/sample_xgboost_apps-0.1.3.jar \
-format=csv \
-num_round=100 \
-trainDataPath=/data/mortgage/csv/2009Q1/train/ \
-evalDataPath=/data/mortgage/csv/2009Q1/test \
-modelPath=/tmp/models/mortgage \
-overwrite=true \
-num_workers=6 \
-nthreads=2

```

logs like:

```console
19/06/13 17:39:43 WARN Utils: Your hostname, pc resolves to a loopback address: 127.0.1.1; using 10.19.183.124 instead (on interface eno1)
19/06/13 17:39:43 WARN Utils: Set SPARK_LOCAL_IP if you need to bind to another address
19/06/13 17:39:43 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
19/06/13 17:39:43 WARN Utils: Service 'SparkUI' could not bind on port 4040. Attempting port 4041.

------ Training ------
19/06/13 17:39:45 WARN Utils: Truncated the string representation of a plan since it was too large. This behavior can be adjusted by setting 'spark.debug.maxToStringFields' in SparkEnv.conf.
Tracker started, with env={DMLC_NUM_SERVER=0, DMLC_TRACKER_URI=10.19.183.124, DMLC_TRACKER_PORT=9091, DMLC_NUM_WORKER=6}
Elapsed time [train]: 152.021s

------ Transforming ------
Elapsed time [transform]: 0.209s
...
...
```

You can try other apps by modifying the `--class` and `--trainDataPath`/ `--evalDataPath` parameters


### supported parameters
1. all [xgboost parameters](https://xgboost.readthedocs.io/en/latest/parameter.html) are supported
2. -format=csv/parquet : The format of the data for training/transforming, now supports 'csv' and 'parquet' only. Required.
3. -mode=all/train/transform. To control the behavior of the sample app, default is 'all' if not specified.
   * all: Do both training and transforming, will save model to 'modelPath' if specified
   * train: Do training only, will save model to 'modelPath' if specified.
   * transform: Do transforming only, 'modelPath' is required to locate the model data to be loaded.
4. -trainDataPath=path : Path to your training data file(s), required when mode is NOT 'transform'.
5. -trainEvalDataPath=path : Path to your data file(s) for training with evaluation. Optional.
6. -evalDataPath=path : Path to your test(evaluation) data file(s), required when mode is NOT 'train'.
7. -modelPath=path : Path to save model after training, or where to load model for transforming only. Required only when mode is 'transform'.
8. -overwrite=true/false : Whether to overwrite the current model data under 'modelPath'. Default is false. You may need to set to true to avoid IOException when saving the model to a path already exists.
9. -hasHeader=true/false : Indicate if your csv file has header.


## <a name="jupyternotebook"></a> jupyter notebook

Make sure you have jupyter notebook installed.

Install Toree:
```bash
pip install toree
```

install scala kernel with Toree:
```bash
jupyter toree install --spark_home=$SPARK_HOME --spark_opts='--master=spark://<YOUR_IP:PORT> \
--executor-memory 32G \
--jars /data/spark/libs/cudf-0.8-SNAPSHOT-cuda10.jar,/data/spark/libs/xgboost4j-0.90-SNAPSHOT.jar,/data/spark/libs/xgboost4j-spark-0.90-SNAPSHOT.jar' \
--user \
--kernel_name=<YOUR_KERNEL_NAME>

```

Then you start your notebook and open [`mortgage-gpu.ipynb`](https://github.com/rapidsai/spark-examples/xgboost/notebook/mortgage-gpu.ipynb) to explore.
