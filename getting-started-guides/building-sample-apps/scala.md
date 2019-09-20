# Build XGBoost Scala Examples

Our examples rely on [cuDF](https://github.com/rapidsai/cudf) and [XGBoost](https://github.com/rapidsai/xgboost/tree/rapids-spark)

##### Build Process

Follow these steps to build the Scala jars:

```
git clone https://github.com/rapidsai/spark-examples.git
cd spark-examples/examples/apps/scala
mvn package -Dcuda.classifier=cuda10 # omit cuda.classifier for cuda 9.2
```

##### Generated Jars

The build process generates two jars:

+ *sample_xgboost_apps-0.1.4.jar* : only classes for the examples are included, so it should be submitted to spark together with other dependent jars
+ *sample_xgboost_apps-0.1.4-jar-with-dependencies.jar*: both classes for the examples and the classes from dependent jars are included

##### Build Options

Classifiers:

+ *cuda.classifier*: omit this classifier for CUDA 9.2 building, and set *cuda10* for CUDA 10.0 building
