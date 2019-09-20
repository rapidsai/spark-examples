// Databricks notebook source
// MAGIC %md # Introduction to XGBoost Spark with GPU
// MAGIC 
// MAGIC Mortgage is an example of xgboost classifier. In this notebook, we will show you how to load data, train the xgboost model and use this model to predict if someone is "deliquency". Comparing to original XGBoost Spark codes, there're only two API differences.
// MAGIC 
// MAGIC 
// MAGIC ## Load libraries
// MAGIC First we load some common libraries that both GPU version and CPU version xgboost will use:

// COMMAND ----------

import ml.dmlc.xgboost4j.scala.spark.{XGBoostClassifier, XGBoostClassificationModel}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.types.{DoubleType, IntegerType, StructField, StructType}

// COMMAND ----------

// MAGIC %md what is new to xgboost-spark users is only `rapids.GpuDataReader`

// COMMAND ----------

import ml.dmlc.xgboost4j.scala.spark.rapids.{GpuDataReader, GpuDataset}

// COMMAND ----------

// MAGIC %md Some libraries needed for CPU version are not needed in GPU version any more. The extra libraries needed for CPU are like below:
// MAGIC 
// MAGIC ```scala
// MAGIC import org.apache.spark.ml.feature.VectorAssembler
// MAGIC import org.apache.spark.sql.DataFrame
// MAGIC import org.apache.spark.sql.functions._
// MAGIC import org.apache.spark.sql.types.FloatType
// MAGIC ```

// COMMAND ----------

// MAGIC %md ## Set your dataset path

// COMMAND ----------

// Set the paths of datasets for training and prediction
// You need to update them to your real paths!
val trainPath = "/FileStore/tables/mortgage_train_merged.csv"
val evalPath  = "/FileStore/tables/mortgage_eval_merged.csv"

// COMMAND ----------

// MAGIC %md ## Set the schema of the dataset
// MAGIC for mortgage example, the data has 27 columns: 26 features and 1 label. "deinquency_12" is set as label column. The schema will be used to help load data in the future. We also defined some key parameters used in xgboost training process. We also set some basic xgboost parameters here.

// COMMAND ----------

val labelColName = "delinquency_12"
val schema = StructType(List(
  StructField("orig_channel", DoubleType),
  StructField("first_home_buyer", DoubleType),
  StructField("loan_purpose", DoubleType),
  StructField("property_type", DoubleType),
  StructField("occupancy_status", DoubleType),
  StructField("property_state", DoubleType),
  StructField("product_type", DoubleType),
  StructField("relocation_mortgage_indicator", DoubleType),
  StructField("seller_name", DoubleType),
  StructField("mod_flag", DoubleType),
  StructField("orig_interest_rate", DoubleType),
  StructField("orig_upb", IntegerType),
  StructField("orig_loan_term", IntegerType),
  StructField("orig_ltv", DoubleType),
  StructField("orig_cltv", DoubleType),
  StructField("num_borrowers", DoubleType),
  StructField("dti", DoubleType),
  StructField("borrower_credit_score", DoubleType),
  StructField("num_units", IntegerType),
  StructField("zip", IntegerType),
  StructField("mortgage_insurance_percent", DoubleType),
  StructField("current_loan_delinquency_status", IntegerType),
  StructField("current_actual_upb", DoubleType),
  StructField("interest_rate", DoubleType),
  StructField("loan_age", DoubleType),
  StructField("msa", DoubleType),
  StructField("non_interest_bearing_upb", DoubleType),
  StructField(labelColName, IntegerType)))

val commParamMap = Map(
  "eta" -> 0.1,
  "gamma" -> 0.1,
  "missing" -> 0.0,
  "max_depth" -> 10,
  "max_leaves" -> 256,
  "grow_policy" -> "depthwise",
  "min_child_weight" -> 30,
  "lambda" -> 1,
  "scale_pos_weight" -> 2,
  "subsample" -> 1,
  "nthread" -> 1,
  "num_round" -> 100)

// COMMAND ----------

// MAGIC %md ## Create a new spark session and load data
// MAGIC we must create a new spark session to continue all spark operations. It will also be used to initilize the `GpuDataReader` which is a data reader powered by GPU.
// MAGIC 
// MAGIC Here's the first API difference, we now use GpuDataReader to load dataset. Similar to original Spark data loading API, GpuDataReader also uses chaining call of "option", "schema","csv". For CPU verions data reader, the code is like below:
// MAGIC 
// MAGIC ```scala
// MAGIC val dataReader = spark.read
// MAGIC ```
// MAGIC 
// MAGIC NOTE: in this notebook, we have uploaded dependency jars when installing toree kernel. If we don't upload them at installation time, we can also upload in notebook by [%AddJar magic](https://toree.incubator.apache.org/docs/current/user/faq/). However, there's one restriction for `%AddJar`: the jar uploaded can only be available when `AddJar` is called after a new spark session is created. We must use it as below:
// MAGIC 
// MAGIC ```scala
// MAGIC import org.apache.spark.sql.SparkSession
// MAGIC val spark = SparkSession.builder().appName("mortgage-GPU").getOrCreate
// MAGIC %AddJar file:/data/libs/cudf-0.9-cuda10.jar
// MAGIC %AddJar file:/data/libs/xgboost4j_2.11-1.0.0-Beta_on_Rapids.jar
// MAGIC %AddJar file:/data/libs/xgboost4j-spark_2.11-1.0.0-Beta_on_Rapids.jar
// MAGIC // ...
// MAGIC ```

// COMMAND ----------

val spark = SparkSession.builder().appName("mortgage-gpu").getOrCreate

// COMMAND ----------

// MAGIC %md Here's the first API difference, we now use `GpuDataReader` to load dataset. Similar to original Spark data loading API, `GpuDataReader` also uses chaining call of "option", "schema","csv". For `CPU` verion data reader, the code is like below:
// MAGIC 
// MAGIC ```scala
// MAGIC val dataReader = spark.read
// MAGIC ```
// MAGIC 
// MAGIC `featureNames` is used to tell xgboost which columns are `feature` and while column is `label`

// COMMAND ----------

val reader = new GpuDataReader(spark).option("header", true).schema(schema)
val featureNames = schema.filter(_.name != labelColName).map(_.name)

// COMMAND ----------

// MAGIC %md Now we can use `dataReader` to read data directly. However, in CPU version, we have to use `VectorAssembler` to assemble all feature columns into one column. The reason will be explained later. the CPU version code is as below:
// MAGIC 
// MAGIC ```scala
// MAGIC object Vectorize {
// MAGIC   def apply(df: DataFrame, featureNames: Seq[String], labelName: String): DataFrame = {
// MAGIC     val toFloat = df.schema.map(f => col(f.name).cast(FloatType))
// MAGIC     new VectorAssembler()
// MAGIC       .setInputCols(featureNames.toArray)
// MAGIC       .setOutputCol("features")
// MAGIC       .transform(df.select(toFloat:_*))
// MAGIC       .select(col("features"), col(labelName))
// MAGIC   }
// MAGIC }
// MAGIC 
// MAGIC val trainSet = reader.csv(trainPath)
// MAGIC val evalSet = reader.csv(evalPath)
// MAGIC trainSet = Vectorize(trainSet, featureCols, labelName)
// MAGIC evalSet = Vectorize(evalSet, featureCols, labelName)
// MAGIC 
// MAGIC ```
// MAGIC 
// MAGIC While with GpuDataReader, `VectorAssembler` is not needed any more. We can simply read data by:

// COMMAND ----------

val trainSet = reader.csv(trainPath)
val evalSet = reader.csv(evalPath)

// COMMAND ----------

// MAGIC %md ## Add XGBoost parameters for GPU version
// MAGIC 
// MAGIC Another modification is `num_workers` should be set to the number of machines with GPU in Spark cluster, while it can be set to the number of your CPU cores in CPU version. CPU version parameters:
// MAGIC 
// MAGIC ```scala
// MAGIC // difference in parameters
// MAGIC "tree_method" -> "hist", 
// MAGIC "num_workers" -> 12
// MAGIC ```

// COMMAND ----------

val xgbParamFinal = commParamMap ++ Map("tree_method" -> "gpu_hist", "num_workers" -> 1)

// COMMAND ----------

val xgbClassifier = new XGBoostClassifier(xgbParamFinal)
      .setLabelCol(labelColName)
      // === diff ===
      .setFeaturesCols(featureNames)

// COMMAND ----------

// MAGIC %md ## Benchmark and train
// MAGIC The benchmark object is for calculating training time. We will use it to compare with xgboost in CPU version.

// COMMAND ----------

object Benchmark {
  def time[R](phase: String)(block: => R): (R, Float) = {
    val t0 = System.currentTimeMillis
    val result = block // call-by-name
    val t1 = System.currentTimeMillis
    println("Elapsed time [" + phase + "]: " + ((t1 - t0).toFloat / 1000) + "s")
    (result, (t1 - t0).toFloat / 1000)
  }
}

// Start training
println("\n------ Training ------")
val (xgbClassificationModel, _) = Benchmark.time("train") {
  xgbClassifier.fit(trainSet)
}


// COMMAND ----------

// MAGIC %md ## Transformation and evaluation
// MAGIC We use `evalSet` to evaluate our model and use some key columns to show our predictions. Finally we use `MulticlassClassificationEvaluator` to calculate an overall accuracy of our predictions.

// COMMAND ----------

println("\n------ Transforming ------")
val (results, _) = Benchmark.time("transform") {
  val ret = xgbClassificationModel.transform(evalSet).cache()
  ret.foreachPartition(_ => ())
  ret
}
results.select("orig_channel", labelColName,"rawPrediction","probability","prediction").show(10)

println("\n------Accuracy of Evaluation------")
val evaluator = new MulticlassClassificationEvaluator().setLabelCol(labelColName)
val accuracy = evaluator.evaluate(results)
println(accuracy)

// COMMAND ----------

// MAGIC %md ## Save the model to disk and load model
// MAGIC We save the model to disk and then load it to memory. We can use the loaded model to do a new prediction.

// COMMAND ----------

xgbClassificationModel.write.overwrite.save("/data/model/mortgage")

val modelFromDisk = XGBoostClassificationModel.load("/data/model/mortgage")

val (results2, _) = Benchmark.time("transform2") {
  modelFromDisk.transform(evalSet)
}
results2.show(10)
