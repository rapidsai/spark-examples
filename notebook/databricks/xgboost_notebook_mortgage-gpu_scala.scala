// Databricks notebook source
import ml.dmlc.xgboost4j.scala.spark.{XGBoostClassifier, XGBoostClassificationModel}
import ml.dmlc.xgboost4j.scala.spark.rapids.GpuDataReader
import ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.types.{DoubleType, IntegerType, StructField, StructType}


// COMMAND ----------

val trainPath = "/FileStore/tables/mortgage_train_merged.csv"
val evalPath  = "/FileStore/tables/mortgage_eval_merged.csv"

// COMMAND ----------

sc.listJars.foreach(println)

// COMMAND ----------

val spark = SparkSession.builder.appName("mortgage-gpu").getOrCreate

// COMMAND ----------

val dataReader = new GpuDataReader(spark)

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
    "num_round" -> 100,
    "num_workers" -> 1,
    "tree_method" -> "gpu_hist")


// COMMAND ----------

var (trainSet, evalSet) = {
  dataReader.option("header", true).schema(schema)
  (dataReader.csv(trainPath), dataReader.csv(evalPath))}

val featureNames = schema.filter(_.name != labelColName).map(_.name)


// COMMAND ----------


object Benchmark {
  def time[R](phase: String)(block: => R): (R, Float) = {
    val t0 = System.currentTimeMillis
    val result = block // call-by-name
    val t1 = System.currentTimeMillis
    println("==> Benchmark: Elapsed time for [" + phase + "]: " + ((t1 - t0).toFloat / 1000) + "s")
    (result, (t1 - t0).toFloat / 1000)
  }
}

// COMMAND ----------

val modelPath = "/FileStore/models/mortgage"
val xgbClassifier = new XGBoostClassifier(commParamMap).setLabelCol(labelColName).setFeaturesCols(featureNames)
println("\n------ Training ------")

val (model, _) = Benchmark.time("train") {
        xgbClassifier.fit(trainSet)
}
// Save model if modelPath exists
model.write.overwrite().save(modelPath)
val xgbClassificationModel = model

// COMMAND ----------

println("\n------ Transforming ------")
val (results, _) = Benchmark.time("transform") {
  xgbClassificationModel.transform(evalSet)
}
results.show(10)


// COMMAND ----------

println("\n------Accuracy of Evaluation------")
val evaluator = new MulticlassClassificationEvaluator().setLabelCol(labelColName)
val accuracy = evaluator.evaluate(results)
println(accuracy)
