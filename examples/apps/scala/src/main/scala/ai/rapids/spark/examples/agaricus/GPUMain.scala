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
package ai.rapids.spark.examples.agaricus

import ai.rapids.spark.examples.utility.{Benchmark, SparkSetup, Vectorize, XGBoostArgs}
import ml.dmlc.xgboost4j.scala.spark.rapids.GpuDataReader
import ml.dmlc.xgboost4j.scala.spark.{XGBoostClassificationModel, XGBoostClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.types.{FloatType, StructField, StructType}

// Only 3 differences between CPU and GPU. Please refer to '=== diff ==='
object GPUMain {
  def main(args: Array[String]): Unit = {

    val labelName = "label"
    def featureNames(length: Int): List[String] =
      0.until(length).map(i => s"feature_$i").toList.+:(labelName)

    def schema(length: Int): StructType =
      StructType(featureNames(length).map(n => StructField(n, FloatType)))

    val dataSchema = schema(126)
    val xgboostArgs = XGBoostArgs.parse(args)
    val processor = this.getClass.getSimpleName.stripSuffix("$").substring(0, 3)
    val appInfo = Seq("Agaricus", processor, xgboostArgs.format)

    // build spark session
    val spark = SparkSetup(args, appInfo.mkString("-"))
    val benchmark = Benchmark(appInfo(0), appInfo(1), appInfo(2))
    // === diff ===
    // build data reader
    val dataReader = new GpuDataReader(spark)
      .option("asFloats", xgboostArgs.asFloats).option("maxRowsPerChunk", xgboostArgs.maxRowsPerChunk)

    // load datasets, the order is (train, train-eval, eval)
    var datasets = xgboostArgs.dataPaths.map(_.map{
      path =>
        xgboostArgs.format match {
          case "csv" => dataReader.option("header", xgboostArgs.hasHeader).schema(dataSchema).csv(path)
          case "parquet" => dataReader.parquet(path)
          case "orc" => dataReader.orc(path)
          case _ => throw new IllegalArgumentException("Unsupported data file format!")
        }
    })

    val featureCols = dataSchema.filter(_.name != labelName).map(_.name)

    // === diff ===
    // No need to vectorize data since GPU support multiple feature columns via API 'setFeaturesCols'

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
        .setFeaturesCols(featureCols)

      println("\n------ Training ------")
      val (model, _) = benchmark.time("train") {
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
      var (results, _) = benchmark.time("transform") {
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
      evaluator.evaluate(results) match {
        case accuracy if !accuracy.isNaN =>
          benchmark.value(accuracy, "Accuracy", "Accuracy for")
        // Throw an exception when NaN ?
      }
    }

    spark.close()
  }
}
