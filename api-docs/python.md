# Python API for XGBoost-Spark

This doc focuses on GPU related Python API interfaces. 7 new classes are introduced:

- [CrossValidator](#crossvalidator)
- [GpuDataset](#gpudataset)
- [GpuDataReader](#gpudatareader)
- [XGBoostClassifier](#xgboostclassifier)
- [XGBoostClassificationModel](#xgboostclassificationmodel)
- [XGBoostRegressor](#xgboostregressor)
- [XGBoostRegressionModel](#xgboostregressionmodel)

### CrossValidator

The full name is `ml.dmlc.xgboost4j.scala.spark.rapids.CrossValidator`, and it is a wrapper around [Scala CrossValidator](scala.md#crossvalidator).

##### Constructors

+ CrossValidator()

##### Methods

*Note: Only GPU related methods are listed below.*

+ fit(dataset): This method triggers the corss validation for hyperparameter tuninng.
    + dataset: a [GpuDataset](#gpudataset) used for cross validation
    + returns the best [Model](https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.ml.Model)[\_] for the given hyperparameters.
    + Note: For CPU version, you can still call `fit` by passing a [Dataset](https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.sql.Dataset).

### GpuDataset

The full name is `ml.dmlc.xgboost4j.scala.spark.rapids.GpuDataset`. A GpuDataset is an object that is produced by [GpuDataReader](#gpudatareader)s and consumed by [XGBoostClassifier](#xgboostclassifier)s and [XGBoostRegressor](#xgboostregressor)s. No constructors or methods are exposed for this class.

### GpuDataReader

The full name is `ml.dmlc.xgboost4j.scala.spark.rapids.GpuDataReader`. A GpuDataReader sets options and builds [GpuDataset](#gpudataset) from data sources. The data loading is a lazy operation. It occurs when the data is processed later.

##### Constructors

+ GpuDataReader(spark_session)
    + spark_session: a [SparkSession](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html?highlight=sparksession#pyspark.sql.SparkSession) for data loading

##### Methods

+ format(source): This method sets data format. Valid values include *csv*, *parquet* and *orc*.
    + source: a String represents the data format to set
    + returns the data reader itself
+ schema(schema): This method sets data schema.
    + schema: data schema either in [StructType](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html?highlight=sparksession#pyspark.sql.types.StructType) format or a DDL-formatted String (e.g., *a INT, b STRING, c DOUBLE*)
    + returns the data reader itself
+ option(key, value): This method sets an option.
    + key: a String represents the option key
    + value: the option value, valid types include *Boolean*, *Integer*, *Float* and *String*
    + returns the data reader itself
+ options(options). This method sets options.
    + options: an option Dictionary[String, String]
    + returns the data reader itself
+ load(\*paths): This method builds a [GpuDataset](#gpudataset).
    + paths: the data source paths, might be empty, one path, or a list of paths
    + returns a [GpuDataset](#gpudataset) as the result
+ csv(\*paths): This method builds a [GpuDataset](#gpudataset).
    + paths: the CSV data paths, might be one path or a list of paths
    + returns a [GpuDataset](#gpudataset) as the result
+ parquet(\*paths): This method builds a [GpuDataset](#gpudataset).
    + paths: the Parquet data paths, might be one path or a list of paths
    + returns a [GpuDataset](#gpudataset) as the result
+ orc(\*paths):. This method builds a [GpuDataset](#gpudataset).
    + paths: the ORC data paths, might be one path or a list of paths
    + returns a [GpuDataset](#gpudataset) as the result

##### Options

+ Common options
    + asFloats: A Boolean flag indicates whether cast all numeric values to floats. Default is True.
    + maxRowsPerChunk: An Integer specifies the max rows per chunk. Default is 2147483647 (2^31-1).
+ Options for CSV
    + comment: A single character used for skipping lines beginning with this character. Default is empty string. By default, it is disabled.
    + header: A Boolean flag indicates whether the first line should be used as names of columns. Default is False.
    + nullValue: The string representation of a null(None) value. Default is empty string.
    + quote: A single character used for escaping quoted values where the separator can be part of the value. Default is `"`.
    + sep: A single character as a separator between adjacent values. Default is `,`.

### XGBoostClassifier

The full name is `ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier`. It is a wrapper around [Scala XGBoostClassifier](scala.md#xgboostclassifier).

#####  Constructors

+ XGBoostClassifier(\*\*params)
    + all [standard xgboost parameters](https://xgboost.readthedocs.io/en/latest/parameter.html) are supported, but please note a few differences:
        + only camelCase is supported when specifying parameter names, e.g., *maxDepth*
        + parameter *lambda* is renamed to *lambda_*, because *lambda* is a keyword in Python

##### Methods

*Note: Only GPU related methods are listed below.*

+ setFeaturesCols(features_cols). This method sets the feature columns for training.
    + features_cols: a list of feature column names in String format to set
    + returns the classifier itself
+ setEvalSets(eval_sets): This method sets eval sets for training.
    + eval_sets: eval sets of type Dictionary[String, [GpuDataset](#gpudataset)] for training (For CPU training, the type is Dictionary[String, [DataFrame](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame)])
    + returns the classifier itself
+ fit(dataset): This method triggers the training.
    + dataset: a [GpuDataset](#gpudataset) to train
    + returns the training result as a [XGBoostClassificationModel](#xgboostclassificationmodel)
    + Note: For CPU training, you can still call fit to train a [DataFrame](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame)

### XGBoostClassificationModel

The full name is `ml.dmlc.xgboost4j.scala.spark.XGBoostClassificationModel`. It is a wrapper around [Scala XGBoostClassificationModel](scala.md#xgboostclassificationmodel).

##### Methods

*Note: Only GPU related methods are listed below.*

+ transform(dataset:): This method  predicts results based on the model.
    + dataset: a [GpuDataset](#gpudataset) to predicate
    + returns a [DataFrame](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame) with the prediction

### XGBoostRegressor

The full name is `ml.dmlc.xgboost4j.scala.spark.XGBoostRegressor`. It is a wrapper around [Scala XGBoostRegressor](scala.md#xgboostregressor).

#####  Constructors

+ XGBoostRegressor(\*\*params)
    + all [standard xgboost parameters](https://xgboost.readthedocs.io/en/latest/parameter.html) are supported, but please note a few differences:
        + only camelCase is supported when specifying parameter names, e.g., *maxDepth*
        + parameter *lambda* is renamed to *lambda_*, because *lambda* is a keyword in Python

##### Methods

*Note: Only GPU related methods are listed below.*

+ setFeaturesCols(features_cols). This method sets the feature columns for training.
    + features_cols: a list of feature column names in String format to set
    + returns the regressor itself
+ setEvalSets(eval_sets): This method sets eval sets for training.
    + eval_sets: eval sets of type Dictionary[String, [GpuDataset](#gpudataset)] for training (For CPU training, the type is Dictionary[String, [DataFrame](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame)])
    + returns the regressor itself
+ fit(dataset): This method triggers the training.
    + dataset: a [GpuDataset](#gpudataset) to train
    + returns the training result as a [XGBoostRegressionModel](#xgboostregressionmodel)
    + Note: For CPU training, you can still call fit to train a [DataFrame](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame)

### XGBoostRegressionModel

The full name is `ml.dmlc.xgboost4j.scala.spark.XGBoostRegressionModel`. It is a wrapper around [Scala XGBoostRegressionModel](scala.md#xgboostregressionmodel).

##### Methods

*Note: Only GPU related methods are listed below.*

+ transform(dataset:): This method  predicts results based on the model.
    + dataset: a [GpuDataset](#gpudataset) to predicate
    + returns a [DataFrame](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame) with the prediction
