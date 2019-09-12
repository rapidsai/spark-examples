/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ai.rapids.spark.examples.mq2008

import ai.rapids.spark.examples.utility.{Benchmark, Vectorize, XGBoostArgs}
import ml.dmlc.xgboost4j.scala.spark.{XGBoostRegressionModel, XGBoostRegressor}
import ml.dmlc.xgboost4j.scala.spark.rapids.GpuDataReader
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.sql.SparkSession

// Only 3 differences between CPU and GPU. Please refer to '=== diff ==='
object CPUMain extends MQ2008 {

  def main(args: Array[String]): Unit = {
    val xgboostArgs = XGBoostArgs.parse(args)

    def logTitle(phase: String = ""): String = {
      val processor = this.getClass.getSimpleName.stripSuffix("$").substring(0, 3)
      val format = xgboostArgs.format
      s"MQ2008 $processor $phase $format"
    }
    // build spark session
    val spark = SparkSession.builder()
      .appName(logTitle().split(" ").filter(_.nonEmpty).mkString("-"))
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

    // === diff ===
    datasets = datasets.map(_.map(ds => Vectorize(featureNames, ds, labelColName, groupColName)))

    val xgbRegressionModel = if (xgboostArgs.isToTrain) {
      // build XGBoost XGBoostRegressor
      val xgbParamFinal = xgboostArgs.xgboostParams(paramMap +
        // Add train-eval dataset if specified
        ("eval_sets" -> datasets(1).map(ds => Map("test" -> ds)).getOrElse(Map.empty))
      )
      val xgbRegressor = new XGBoostRegressor(xgbParamFinal)
        .setLabelCol(labelColName)
        .setGroupCol(groupColName)
        // === diff ===
        .setFeaturesCol("features")

      println("\n------ Training ------")
      val (model, _) = Benchmark.time(logTitle("train")) {
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
      var (prediction, _) = Benchmark.time(logTitle("transform")) {
        val ret = xgbRegressionModel.transform(datasets(2).get).cache()
        ret.foreachPartition(_ => ())
        ret
      }
      prediction = if (xgboostArgs.isShowFeatures) {
        prediction
      } else {
        prediction.select(labelColName, groupColName, "prediction")
      }
      prediction.show(xgboostArgs.numRows)

      println("\n------Accuracy of Evaluation------")
      val evaluator = new RegressionEvaluator().setLabelCol(labelColName)
      val (rmse, _) = Benchmark.time(logTitle("evaluation")) {
        evaluator.evaluate(prediction)
      }
      println(s"RMSE == $rmse")
    }

    spark.close()
  }
}
