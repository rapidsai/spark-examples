This repo provides example applications that demonstrate the RAPIDS.ai GPU-accelerated XGBoost-Spark project.

There are three example apps included in this repo: [Mortgage](/src/main/scala/ai/rapids/spark/examples/mortgage), [Taxi](/src/main/scala/ai/rapids/spark/examples/taxi), and [Agaricus](/src/main/scala/ai/rapids/spark/examples/agaricus).

### Build Examples Jar
Our example relies on [cuDF](https://github.com/rapidsai/cudf) and [XGBoost](https://github.com/rapidsai/xgboost/tree/rapids-spark)

Follow these steps to build the jar:

```
git clone https://github.com/rapidsai/spark-examples.git
cd spark-examples
mvn package -DxgbClassifier=cuda10 # omit xgbClassifier for cuda 9.2
```

### Getting Started Guides

Try one of the Getting Started guides below. Please note that they target the Mortage dataset as written, but with a few changes to `EXAMPLE_CLASS`, `trainDataPath`, and `evalDataPath`, they can be easily adapted to the Taxi or Agaricus datasets.

You can get a small size datasets for each example in the [datasets](/datasets) folder. These datasets are only provided for convenience. In order to test for performance, please prepare a larger dataset by following [Preparing Datasets](docs/preparing_datasets.md). We also provide a larger dataset: [Morgage Dataset (1 GB uncompressed)](https://rapidsai-data.s3.us-east-2.amazonaws.com/spark/mortgage.zip), which is used in the guides below.

- [Standalone](docs/standalone.md)
- [YARN](docs/yarn.md)
- [Kubernetes](docs/kubernetes.md)
- [Apache Toree Notebook](docs/toree.md)

These examples use default parameters for demo purposes. For a full list please see [Supported XGBoost Parameters](/docs/supported_xgboost_parameters.md).

### Contact Us

Please see the [RAPIDS](https://rapids.ai/community.html) website for contact information.

### License

This content is licensed under the [Apache License 2.0](/LICENSE)
