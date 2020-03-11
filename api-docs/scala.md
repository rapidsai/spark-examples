# Scala API for XGBoost-Spark

This doc focuses on GPU related Scala API interfaces. 7 new classes are introduced:

- [CrossValidator](#crossvalidator)
- [GpuDataset](#gpudataset)
- [GpuDataReader](#gpudatareader)
- [XGBoostClassifier](#xgboostclassifier)
- [XGBoostClassificationModel](#xgboostclassificationmodel)
- [XGBoostRegressor](#xgboostregressor)
- [XGBoostRegressionModel](#xgboostregressionmodel)

### CrossValidator

The full name is `ml.dmlc.xgboost4j.scala.spark.rapids.CrossValidator`, extending from the Spark's [CrossValidator](https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.ml.tuning.CrossValidator).

##### Constructors

+ CrossValidator()

##### Methods

*Note: Only GPU related methods are listed below.*

+ fit(dataset: [GpuDataset](#gpudataset)): [Model](https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.ml.Model)[\_]. This method triggers the corss validation for hyperparameter tuninng.
    + dataset: a [GpuDataset](#gpudataset) used for cross validation
    + returns the best [Model](https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.ml.Model)[\_] for the given hyperparameters. Please note this model returned here is actually a [XGBoostClassificationModel](#xgboostclassificationmodel) for [XGBoostClassifier](#xgboostclassifier), or a  [XGBoostRegressionModel](#xgboostregressionmodel) for [XGBoostRegressor](#xgboostregressor). You need to cast it to the right model for calling the GPU version `transform`(dataset: [GpuDataset](#gpudataset)).
    + Note: For CPU version, you can still call `fit`(dataset: [Dataset](https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.sql.Dataset)[\_])


### GpuDataset

The full name is `ml.dmlc.xgboost4j.scala.spark.rapids.GpuDataset`. A GpuDataset is an object that is produced by [GpuDataReader](#gpudatareader)s and consumed by [XGBoostClassifier](#xgboostclassifier)s and [XGBoostRegressor](#xgboostregressor)s. No constructors or methods are exposed for this class.

### GpuDataReader

The full name is `ml.dmlc.xgboost4j.scala.spark.rapids.GpuDataReader`. A GpuDataReader sets options and builds [GpuDataset](#gpudataset) from data sources. The data loading is a lazy operation. It occurs when the data is processed later.

##### Constructors

+ GpuDataReader(sparkSession: [SparkSession](https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.sql.SparkSession))
    + sparkSession: spark session for data loading

##### Methods

+ format(source: String): [GpuDataReader](#gpudatareader). This method sets data format. Valid values include *csv*, *parquet* and *orc*.
    + source: data format to set
    + returns the data reader itself
+ schema(schema: [StructType](https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.sql.types.StructType)): [GpuDataReader](#gpudatareader). This method sets data schema.
    + schema: data schema in [StructType](https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.sql.types.StructType) format
    + returns the data reader itself
+ schema(schemaString: String): [GpuDataReader](#gpudatareader). This method sets data schema.
    + schemaString: data schema in DDL-formatted String, e.g., *a INT, b STRING, c DOUBLE*
    + returns the data reader itself
+ option(key: String, value: String): [GpuDataReader](#gpudatareader). This method sets an option.
    + key: the option key
    + value: the option value in string format
    + returns the data reader itself
+ option(key: String, value: Boolean): [GpuDataReader](#gpudatareader). This method sets an option.
    + key: the option key
    + value: the Boolean option value
    + returns the data reader itself
+ option(key: String, value: Long): [GpuDataReader](#gpudatareader). This method sets an option.
    + key: the option key
    + value: the Long option value
    + returns the data reader itself
+ option(key: String, value: Double): [GpuDataReader](#gpudatareader). This method sets an option.
    + key: the option key
    + value: the Double option value
    + returns the data reader itself
+ options(options: scala.collection.Map[String, String]): [GpuDataReader](#gpudatareader). This method sets options.
    + options: the options Map to set
    + returns the data reader itself
+ options(options: java.util.Map[String, String]): [GpuDataReader](#gpudatareader). This method sets options. It is designed for Java  compatibility.
    + options: the options Map to set
    + returns the data reader itself
+ load(): [GpuDataset](#gpudataset). This method builds a [GpuDataset](#gpudataset).
    + returns a [GpuDataset](#gpudataset) as the result
+ load(path: String): [GpuDataset](#gpudataset). This method builds a [GpuDataset](#gpudataset).
    + path: the data source path
    + returns a [GpuDataset](#gpudataset) as the result
+ load(paths: String\*): [GpuDataset](#gpudataset). This method builds a [GpuDataset](#gpudataset).
    + paths: the data source paths
    + returns a [GpuDataset](#gpudataset) as the result
+ csv(path: String): [GpuDataset](#gpudataset). This method builds a [GpuDataset](#gpudataset).
    + path: the CSV data path
    + returns a [GpuDataset](#gpudataset) as the result
+ csv(paths: String\*): [GpuDataset](#gpudataset). This method builds a [GpuDataset](#gpudataset).
    + paths: the CSV data paths
    + returns a [GpuDataset](#gpudataset) as the result
+ parquet(path: String): [GpuDataset](#gpudataset). This method builds a [GpuDataset](#gpudataset).
    + path: the Parquet data path
    + returns a [GpuDataset](#gpudataset) as the result
+ parquet(paths: String\*): [GpuDataset](#gpudataset). This method builds a [GpuDataset](#gpudataset).
    + paths: the Parquet data paths
    + returns a [GpuDataset](#gpudataset) as the result
+ orc(path: String): [GpuDataset](#gpudataset). This method builds a [GpuDataset](#gpudataset).
    + path: the ORC data path
    + returns a [GpuDataset](#gpudataset) as the result
+ orc(paths: String\*): [GpuDataset](#gpudataset). This method builds a [GpuDataset](#gpudataset).
    + paths: the ORC data paths
    + returns a [GpuDataset](#gpudataset) as the result

##### Options

+ Common options
    + asFloats: A Boolean flag indicates whether cast all numeric values to floats. Default is true.
    + maxRowsPerChunk: An Int specifies the max rows per chunk. Default is Int.MaxValue.
+ Options for CSV
    + comment: A single character used for skipping lines beginning with this character. Default is empty string. By default, it is disabled.
    + header: A Boolean flag indicates whether the first line should be used as names of columns. Default is false.
    + nullValue: The string representation of a null value. Default is empty string.
    + quote: A single character used for escaping quoted values where the separator can be part of the value. Default is `"`.
    + sep: A single character as a separator between adjacent values. Default is `,`.

### XGBoostClassifier

The full name is `ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier`. It extends [ProbabilisticClassifier](https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.ml.classification.ProbabilisticClassifier)[[Vector](https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.ml.linalg.Vector), [XGBoostClassifier](#xgboostclassifier), [XGBoostClassificationModel](#xgboostclassificationmodel)].

#####  Constructors

+ XGBoostClassifier(xgboostParams: Map[String, Any])
    + all [standard xgboost parameters](https://xgboost.readthedocs.io/en/latest/parameter.html) are supported
    + eval_sets: Map[String, [GpuDataset](#gpudataset)]. This parameter sets the eval sets for training. (For CPU training, the type of parameter eval_sets is Map[String, [DataFrame](https://spark.apache.org/docs/latest/api/scala/org/apache/spark/sql/package.html#DataFrame=org.apache.spark.sql.Dataset[org.apache.spark.sql.Row])])

##### Methods

*Note: Only GPU related methods are listed below.*

+ setFeaturesCols(value: Seq[String]): [XGBoostClassifier](#xgboostclassifier). This method sets the feature columns for training.
    + value: a sequence of feature column names to set
    + returns the classifier itself
+ setEvalSets(evalSets: Map[String, [GpuDataset](#gpudataset)]): [XGBoostClassifier](#xgboostclassifier). This method sets eval sets for training.
    + evalSets: eval sets for training (For CPU training, the type is Map[String, [DataFrame](https://spark.apache.org/docs/latest/api/scala/org/apache/spark/sql/package.html#DataFrame=org.apache.spark.sql.Dataset[org.apache.spark.sql.Row])])
    + returns the classifier itself
+ fit(dataset: [GpuDataset](#gpudataset)):  [XGBoostClassificationModel](#xgboostclassificationmodel). This method triggers the training.
    + dataset: a [GpuDataset](#gpudataset) to train
    + returns the training result as a [XGBoostClassificationModel](#xgboostclassificationmodel)
    + Note: For CPU training, you can still call fit(dataset: [Dataset](https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.sql.Dataset)[\_])

### XGBoostClassificationModel

The full name is `ml.dmlc.xgboost4j.scala.spark.XGBoostClassificationModel`. It extends [ProbabilisticClassificationModel](https://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.ml.classification.ProbabilisticClassificationModel)[[Vector](https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.ml.linalg.Vector), [XGBoostClassificationModel](#xgboostclassificationmodel)].

##### Methods

*Note: Only GPU related methods are listed below.*

+ transform(dataset: [GpuDataset](#gpudataset)): [DataFrame](https://spark.apache.org/docs/latest/api/scala/org/apache/spark/sql/package.html#DataFrame=org.apache.spark.sql.Dataset[org.apache.spark.sql.Row]). This method  predicts results based on the model.
    + dataset: a [GpuDataset](#gpudataset) to predicate
    + returns a [DataFrame](https://spark.apache.org/docs/latest/api/scala/org/apache/spark/sql/package.html#DataFrame=org.apache.spark.sql.Dataset[org.apache.spark.sql.Row]) with the prediction

### XGBoostRegressor

The full name is `ml.dmlc.xgboost4j.scala.spark.XGBoostRegressor`. It extends [Predictor](https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.ml.Predictor)[[Vector](https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.ml.linalg.Vector), [XGBoostRegressor](#xgboostregressor), [XGBoostRegressionModel](#xgboostregressionmodel)].

#####  Constructors

+ XGBoostRegressor(xgboostParams: Map[String, Any])
    + all [standard xgboost parameters](https://xgboost.readthedocs.io/en/latest/parameter.html) are supported
    + eval_sets: Map[String, [GpuDataset](#gpudataset)]. This parameter sets the eval sets for training. (For CPU training, the type of parameter eval_sets is Map[String, [DataFrame](https://spark.apache.org/docs/latest/api/scala/org/apache/spark/sql/package.html#DataFrame=org.apache.spark.sql.Dataset[org.apache.spark.sql.Row])])

##### Methods

*Note: Only GPU related methods are listed below.*

+ setFeaturesCols(value: Seq[String]): [XGBoostRegressor](#xgboostregressor). This method sets the feature columns for training.
    + value: a sequence of feature column names to set
    + returns the regressor itself
+ setEvalSets(evalSets: Map[String, [GpuDataset](#gpudataset)]): [XGBoostRegressor](#xgboostregressor). This method sets eval sets for training.
    + evalSets: eval sets for training (For CPU training, the type is Map[String, [DataFrame](https://spark.apache.org/docs/latest/api/scala/org/apache/spark/sql/package.html#DataFrame=org.apache.spark.sql.Dataset[org.apache.spark.sql.Row])])
    + returns the regressor itself
+ fit(dataset: [GpuDataset](#gpudataset)):  [XGBoostRegressionModel](#xgboostregressionmodel). This method triggers the training.
    + dataset: a [GpuDataset](#gpudataset) to train
    + returns the training result as a [XGBoostRegressionModel](#xgboostregressionmodel)
    + Note: For CPU training, you can still call fit(dataset: [Dataset](https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.sql.Dataset)[\_])

### XGBoostRegressionModel

The full name is `ml.dmlc.xgboost4j.scala.spark.XGBoostRegressionModel`. It extends [PredictionModel](https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.ml.PredictionModel)[[Vector](https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.ml.linalg.Vector), [XGBoostRegressionModel](#xgboostregressionmodel)].

##### Methods

*Note: Only GPU related methods are listed below.*

+ transform(dataset: [GpuDataset](#gpudataset)): [DataFrame](https://spark.apache.org/docs/latest/api/scala/org/apache/spark/sql/package.html#DataFrame=org.apache.spark.sql.Dataset[org.apache.spark.sql.Row]). This method  predicts results based on the model.
    + dataset: a [GpuDataset](#gpudataset) to predicate
    + returns a [DataFrame](https://spark.apache.org/docs/latest/api/scala/org/apache/spark/sql/package.html#DataFrame=org.apache.spark.sql.Dataset[org.apache.spark.sql.Row]) with the prediction
