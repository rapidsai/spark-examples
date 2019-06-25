package ai.rapids.spark.examples.agaricus

import ai.rapids.spark.examples.utility.{Benchmark, SparkSetup, Vectorize, XGBoostArgs}
import ml.dmlc.xgboost4j.scala.spark.rapids.GpuDataReader
import ml.dmlc.xgboost4j.scala.spark.{XGBoostClassificationModel, XGBoostClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.types.{FloatType, StructField, StructType}

// Only 3 differences between CPU and GPU. Please refer to '=== diff ==='
object CPUMain {
  def main(args: Array[String]): Unit = {

    val labelName = "label"
    def featureNames(length: Int): List[String] =
      0.until(length).map(i => s"feature_$i").toList.+:(labelName)

    def schema(length: Int): StructType =
      StructType(featureNames(length).map(n => StructField(n, FloatType)))

    val xgboostArgs = XGBoostArgs.parse(args)
    val dataSchema = schema(126)

    // build spark session
    val objName = this.getClass.getSimpleName.stripSuffix("$")
    val spark = SparkSetup(args, "AgaricusAppFor$objName")
    spark.sparkContext.setLogLevel("WARN")

    // === diff ===
    // build data reader
    val dataReader = spark.read

    // load datasets, the order is (train, train-eval, eval)
    var datasets = xgboostArgs.dataPaths.map(_.map{
      path =>
        xgboostArgs.format match {
          case "csv" => dataReader.option("header", xgboostArgs.hasHeader).schema(dataSchema).csv(path)
          case "parquet" => dataReader.parquet(path)
          case _ => throw new IllegalArgumentException("Unsupported data file format!")
        }
    })

    val featureCols = dataSchema.filter(_.name != labelName).map(_.name)

    // === diff ===
    datasets = datasets.map(_.map(ds => Vectorize(ds, featureCols, labelName)))

    val xgbClassificationModel = if (xgboostArgs.isToTrain) {
      // build XGBoost classifier
      val paramMap = xgboostArgs.xgboostParams(Map(
        "eta" -> 0.1,
        "missing" -> 0.0,
        "max_depth" -> 2,
        "eval_sets" -> datasets(1).map(ds => Map("test" -> ds)).getOrElse(Map.empty)
      ))
      val xgbClassifier = new XGBoostClassifier(paramMap)
        .setLabelCol(labelName)
        // === diff ===
        .setFeaturesCol("features")

      println("\n------ Training ------")
      val (model, _) = Benchmark.time("train") {
        xgbClassifier.fit(datasets(0).get)
      }
      // Save model if modelPath exists
      xgboostArgs.modelPath.foreach(path =>
        if(xgboostArgs.isOverwrite) model.write.overwrite().save(path) else model.save(path))
      model
    } else {
      XGBoostClassificationModel.load(xgboostArgs.modelPath.get)
    }

    if (xgboostArgs.isToTransform) {
      // start transform
      println("\n------ Transforming ------")
      var (results, _) = Benchmark.time("transform") {
        val ret = xgbClassificationModel.transform(datasets(2).get).cache()
        ret.foreachPartition(_ => ())
        ret
      }
      results = if (xgboostArgs.isShowFeatures) {
        results
      } else {
        results.select(labelName, "rawPrediction", "probability", "prediction")
      }
      results.show(xgboostArgs.numRows)

      println("\n------Accuracy of Evaluation------")
      val evaluator = new MulticlassClassificationEvaluator().setLabelCol(labelName)
      val accuracy = evaluator.evaluate(results)

      println(s"accuracy == $accuracy")
    }

    spark.close()
  }
}
