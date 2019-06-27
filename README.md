RAPIDS.ai XGBoost-Spark Apache Spark Examples
=============================================

This repo provides example applications that demonstrate the RAPIDS.ai GPU-accelerated XGBoost-Spark project.

Table of contents:

  * [Build Examples Jar](#Build-Examples-Jar)
  * [Getting Started Guides](#Getting-Started-Guides)
  * [Example Applications](#Example-Applications)
  * [Preparing Datasets](#Preparing-Datasets)

# Build Examples Jar
We use [maven](https://maven.apache.org/) to build jar package.

Our example relies on [cuDF](https://github.com/rapidsai/cudf) and [XGBoost](https://github.com/rapidsai/xgboost/tree/rapids-spark)

1. Clone this repo.
2. Use maven to build the code:

   ```
   cd spark-examples
   mvn package 
   ```
   The command above will build a jar package with default cuda version `9.2`. If your cuda version is 10.0, you should do:

   ```
   mvn package -DxgbClassifier=cuda10
   ```

   Then you will find both `sample_xgboost_apps-0.1.4.jar` and `sample_xgboost_apps-0.1.4-jar-with-dependencies.jar` in your `target` folder. To make it simple, we'll choose the "jar-with-dependencies" (assembly jar) for all the examples in these pages. The plain jar can be used, but other dependencies need to be added manually: `cudf`, `xgboost4j`, and `xgboost4j-spark`.

# Getting Started Guides
Follow one of our Getting Started guides to run the sample mortgage dataset:

- [Standalone](docs/standalone.md)
- [YARN](docs/yarn.md)
- [Kubernetes](docs/kubernetes.md)
- [Apache Toree Notebook](docs/toree.md)

## Supported Parameters
For reference, the following parameters are supported:

1. All [xgboost parameters](https://xgboost.readthedocs.io/en/latest/parameter.html) are supported.
2. `-format=[csv|parquet]`: The format of the data for training/transforming, now supports 'csv' and 'parquet' only. *Required*.
3. `-mode=[all|train|transform]`. To control the behavior of the sample app, default is 'all' if not specified.
   * all: Do both training and transforming, will save model to 'modelPath' if specified
   * train: Do training only, will save model to 'modelPath' if specified.
   * transform: Do transforming only, 'modelPath' is required to locate the model data to be loaded.
4. `-trainDataPath=[path]`: Path to your training data file(s), required when mode is NOT 'transform'.
5. `-trainEvalDataPath=[path]`: Path to your data file(s) for training with evaluation. Optional.
6. `-evalDataPath=[path]`: Path to your test(evaluation) data file(s), required when mode is NOT 'train'.
7. `-modelPath=[path]`: Path to save model after training, or where to load model for transforming only. Required only when mode is 'transform'.
8. `-overwrite=[true|false]`: Whether to overwrite the current model data under 'modelPath'. Default is false. You may need to set to true to avoid IOException when saving the model to a path already exists.
9. `-hasHeader=[true|false]`: Indicate if your csv file has header.

# Example Applications
There are three example apps included in this repo.

- [Mortgage](/src/main/scala/ai/rapids/spark/examples/mortgage) 
- [Taxi](/src/main/scala/ai/rapids/spark/examples/taxi)
- [Agaricus](/src/main/scala/ai/rapids/spark/examples/agaricus)

Each example requires a different dataset. If you would like to produce your own dataset, the code used to prepare each sample is included in this repo (please see [Preparing Datasets](docs/preparing_datasets.md)).

## Download Trainable Dataset

You can get a small size transformed dataset for each example class in `datasets` folder in this repo: 

1. [Mortgage Dataset(csv)](https://github.com/rapidsai/spark-examples/blob/master/datasets/mortgage-small.tar.gz?raw=true)
2. [Mortgage Dataset(csv) (1 GB uncompressed)](https://rapidsai-data.s3.us-east-2.amazonaws.com/spark/mortgage.zip)
3. [Taxi Dataset(csv)](https://github.com/rapidsai/spark-examples/blob/master/datasets/taxi-small.tar.gz?raw=true)
4. [Agaricus(csv)](https://github.com/rapidsai/spark-examples/blob/master/datasets/agaricus.tar.gz?raw=true)

Please note that the data above is only provided for convenience to get started with each example. In order to test for performance, please prepare a larger dataset by following [Preparing Datasets](docs/preparing_datasets.md).
