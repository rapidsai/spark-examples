package ai.rapids.spark.examples.taxi

import ai.rapids.spark.examples.utility.{Benchmark, Vectorize, XGBoostArgs}
import ml.dmlc.xgboost4j.scala.spark.{XGBoostRegressionModel, XGBoostRegressor}
import ml.dmlc.xgboost4j.scala.spark.rapids.GpuDataReader
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.sql.SparkSession

// Only 3 differences between CPU and GPU. Please refer to '=== diff ==='
object GPUMain extends Taxi {

  def main(args: Array[String]): Unit = {
    val xgboostArgs = XGBoostArgs.parse(args)

    // build spark session
    val objName = this.getClass.getSimpleName.stripSuffix("$")
    val spark = SparkSession.builder()
      .appName(s"Taxi-$objName-${xgboostArgs.format}")
      .getOrCreate()

    // === diff ===
    // build data reader
    val dataReader = new GpuDataReader(spark)

    // load datasets, the order is (train, train-eval, eval)
    var datasets = xgboostArgs.dataPaths.map(_.map{
      path =>
        xgboostArgs.format match {
          case "csv" => dataReader.option("header", xgboostArgs.hasHeader).schema(schema).csv(path)
          case "parquet" => dataReader.parquet(path)
          case _ => throw new IllegalArgumentException("Unsupported data file format!")
        }
    })

    val featureNames = schema.filter(_.name != labelColName).map(_.name)

    // === diff ===
    // No need to vectorize data since GPU support multiple feature columns via API 'setFeaturesCols'

    val xgbRegressionModel = if (xgboostArgs.isToTrain) {
      // build XGBoost XGBoostRegressor
      val xgbParamFinal = xgboostArgs.xgboostParams(commParamMap +
        // Add train-eval dataset if specified
        ("eval_sets" -> datasets(1).map(ds => Map("test" -> ds)).getOrElse(Map.empty))
      )
      val xgbRegressor = new XGBoostRegressor(xgbParamFinal)
        .setLabelCol(labelColName)
        // === diff ===
        .setFeaturesCols(featureNames)

      println("\n------ Training ------")
      val (model, _) = Benchmark.time("train") {
        xgbRegressor.fit(datasets(0).get)
      }
      // Save model if modelPath exists
      xgboostArgs.modelPath.foreach(path =>
        if(xgboostArgs.isOverwrite) model.write.overwrite().save(path) else model.save(path))
      model
    } else {
      XGBoostRegressionModel.load(xgboostArgs.modelPath.get)
    }

    if (xgboostArgs.isToTransform) {
      println("\n------ Transforming ------")
      var (prediction, _) = Benchmark.time("transform") {
        val ret = xgbRegressionModel.transform(datasets(2).get).cache()
        ret.foreachPartition(_ => ())
        ret
      }
      prediction = if (xgboostArgs.isShowFeatures) {
        prediction
      } else {
        prediction.select(labelColName, "prediction")
      }
      prediction.show(xgboostArgs.numRows)

      println("\n------Accuracy of Evaluation------")
      val evaluator = new RegressionEvaluator().setLabelCol(labelColName)
      val (rmse, _) = Benchmark.time("evaluation") {
        evaluator.evaluate(prediction)
      }
      println(s"RMSE == $rmse")
    }

    spark.close()
  }
}
