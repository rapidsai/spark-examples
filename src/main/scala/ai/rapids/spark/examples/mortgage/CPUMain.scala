package ai.rapids.spark.examples.mortgage

import ai.rapids.spark.examples.utility.{Benchmark, Vectorize, XGBoostArgs}
import ml.dmlc.xgboost4j.scala.spark.{XGBoostClassificationModel, XGBoostClassifier}
import ml.dmlc.xgboost4j.scala.spark.rapids.GpuDataReader
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

// Only 3 differences between CPU and GPU. Please refer to '=== diff ==='
object CPUMain extends Mortgage {

  def main(args: Array[String]): Unit = {
    val xgboostArgs = XGBoostArgs.parse(args)

    // build spark session
    val objName = this.getClass.getSimpleName.stripSuffix("$")
    val spark = SparkSession.builder()
      .appName(s"Mortgage-$objName-${xgboostArgs.format}")
      .getOrCreate()

    // === diff ===
    // build data reader
    val dataReader = spark.read

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
    datasets = datasets.map(_.map(ds => Vectorize(ds, featureNames, labelColName)))

    val xgbClassificationModel = if (xgboostArgs.isToTrain) {
      // build XGBoost classifier
      val xgbParamFinal = xgboostArgs.xgboostParams(commParamMap +
        // Add train-eval dataset if specified
        ("eval_sets" -> datasets(1).map(ds => Map("test" -> ds)).getOrElse(Map.empty))
      )
      val xgbClassifier = new XGBoostClassifier(xgbParamFinal)
        .setLabelCol(labelColName)
        // === diff ===
        .setFeaturesCol("features")

      // Start training
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
      println("\n------ Transforming ------")
      var (results, _) = Benchmark.time("transform") {
        val ret = xgbClassificationModel.transform(datasets(2).get).cache()
        // Trigger the transformation
        ret.foreachPartition(_ => ())
        ret
      }
      results = if (xgboostArgs.isShowFeatures) {
        results
      } else {
        results.select(labelColName, "rawPrediction", "probability", "prediction")
      }
      results.show(xgboostArgs.numRows)

      println("\n------Accuracy of Evaluation------")
      val evaluator = new MulticlassClassificationEvaluator().setLabelCol(labelColName)
      val accuracy = evaluator.evaluate(results)
      println(accuracy)
    }

    spark.close()
  }
}