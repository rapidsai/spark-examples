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
package ai.rapids.spark.examples.mortgage

import ai.rapids.spark.examples.utility.{Benchmark, Vectorize, XGBoostArgs}
import ml.dmlc.xgboost4j.scala.spark.{XGBoostClassificationModel, XGBoostClassifier}
import ml.dmlc.xgboost4j.scala.spark.rapids.GpuDataReader
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

// Only 3 differences between CPU and GPU. Please refer to '=== diff ==='
object GPUMain extends Mortgage {

  def main(args: Array[String]): Unit = {
    val xgboostArgs = XGBoostArgs.parse(args)

    // build spark session
    val objName = this.getClass.getSimpleName.stripSuffix("$")
    val spark = SparkSession.builder()
      .appName(s"Mortgage-$objName-${xgboostArgs.format}")
      .getOrCreate()

    // === diff ===
    // build data reader
    val dataReader = new GpuDataReader(spark)

    // load datasets, the order is (train, train-eval, eval)
    var datasets = xgboostArgs.dataPaths.map(_.map{
      path =>
        xgboostArgs.format match {
          case "csv" => dataReader
            .option("header", xgboostArgs.hasHeader)
            .option("asFloats", xgboostArgs.asFloats)
            .option("maxRowsPerChunk", xgboostArgs.maxRowsPerChunk)
            .schema(schema)
            .csv(path)
          case "parquet" => dataReader
            .option("asFloats", xgboostArgs.asFloats)
            .option("maxRowsPerChunk", xgboostArgs.maxRowsPerChunk)
            .parquet(path)
          case "orc" => dataReader
            .option("asFloats", xgboostArgs.asFloats)
            .option("maxRowsPerChunk", xgboostArgs.maxRowsPerChunk)
            .orc(path)
          case _ => throw new IllegalArgumentException("Unsupported data file format!")
        }
    })

    val featureNames = schema.filter(_.name != labelColName).map(_.name)

    // === diff ===
    // No need to vectorize data since GPU support multiple feature columns via API 'setFeaturesCols'

    val xgbClassificationModel = if (xgboostArgs.isToTrain) {
      // build XGBoost classifier
      val xgbParamFinal = xgboostArgs.xgboostParams(commParamMap +
        // Add train-eval dataset if specified
        ("eval_sets" -> datasets(1).map(ds => Map("test" -> ds)).getOrElse(Map.empty))
      )
      val xgbClassifier = new XGBoostClassifier(xgbParamFinal)
        .setLabelCol(labelColName)
        // === diff ===
        .setFeaturesCols(featureNames)

      // Start training
      println("\n------ Training ------")
      val (model, _) = Benchmark.time(s"Mortgage GPU train ${xgboostArgs.format}") {
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
      var (results, _) = Benchmark.time(s"Mortgage GPU transform ${xgboostArgs.format}") {
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
      println("Accuracy: " + accuracy)
    }

    spark.close()
  }
}
