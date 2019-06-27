RAPIDS.ai XGBoost-Spark Apache Spark Examples
=============================================

This repo provides example applications that demonstrate the RAPIDS.ai GPU-accelerated XGBoost-Spark project.

# Example Applications
There are three example apps included in this repo: [Mortgage](/src/main/scala/ai/rapids/spark/examples/mortgage), [Taxi](/src/main/scala/ai/rapids/spark/examples/taxi), and [Agaricus](/src/main/scala/ai/rapids/spark/examples/agaricus).

Each example requires a different dataset. If you would like to produce your own dataset, the code used to prepare each sample is included in this repo (please see [Preparing Datasets](docs/preparing_datasets.md)).

# Build Examples Jar
Our example relies on [cuDF](https://github.com/rapidsai/cudf) and [XGBoost](https://github.com/rapidsai/xgboost/tree/rapids-spark)

Follow these steps to build the jar:

```
git clone https://github.com/rapidsai/spark-examples.git
cd spark-examples
mvn package -DxgbClassifier=cuda10 # omit xgbClassifier for cuda 9.2
```

Then you will find both `sample_xgboost_apps-0.1.4.jar` and `sample_xgboost_apps-0.1.4-jar-with-dependencies.jar` in your `target` folder. Use the "jar-with-dependencies" (assembly jar) for all the examples in these pages. The plain jar can be used, but other dependencies need to be added manually: `cudf`, `xgboost4j`, and `xgboost4j-spark`.

# Getting Started Guides

You can get a small size transformed dataset for each example class in `datasets` folder in this repo: 

1. [Mortgage Dataset(csv)](/datasets/mortgage-small.tar.gz?raw=true)
2. [Mortgage Dataset(csv) (1 GB uncompressed)](https://rapidsai-data.s3.us-east-2.amazonaws.com/spark/mortgage.zip)
3. [Taxi Dataset(csv)](/datasets/taxi-small.tar.gz?raw=true)
4. [Agaricus(csv)](/datasets/agaricus.tar.gz?raw=true)

Please note that the data above is only provided for convenience to get started with each example. In order to test for performance, please prepare a larger dataset by following [Preparing Datasets](docs/preparing_datasets.md).

Try one of the Getting Started guides below. Please note that they target the Mortage dataset as written, but with a few changes to `EXAMPLE_CLASS`, `trainDataPath`, and `evalDataPath`, they can be easily adapted to the Taxi or Agaricus datasets.

- [Standalone](docs/standalone.md)
- [YARN](docs/yarn.md)
- [Kubernetes](docs/kubernetes.md)
- [Apache Toree Notebook](docs/toree.md)

These examples use default parameters for demo purposes. For a full list please see [Supported XGBoost Parameters](/docs/supported_xgboost_parameters.md).

# Contact Us

Please see the [RAPIDS](https://rapids.ai/community.html) website for contact information.

# License

This content is licensed under the [Apache License 2.0](/LICENSE)
