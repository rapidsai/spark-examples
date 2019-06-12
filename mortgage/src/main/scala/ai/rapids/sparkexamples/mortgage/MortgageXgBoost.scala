/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package ai.rapids.sparkexamples.mortgage

import ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{FeatureHasher, StringIndexer, VectorAssembler}
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.DoubleType

object MortgageXgBoost {
  val catCols = List(
    "orig_channel",
    "first_home_buyer",
    "loan_purpose",
    "property_type",
    "occupancy_status",
    "property_state",
    "product_type",
    "relocation_mortgage_indicator",
    "seller_name",
    "mod_flag"
  )

  val numericCols = List(
    "orig_interest_rate",
    "orig_upb",
    "orig_loan_term",
    "orig_ltv",
    "orig_cltv",
    "num_borrowers",
    "dti",
    "borrower_credit_score",
    "num_units",
    "zip",
    "mortgage_insurance_percent",
    "current_loan_delinquency_status",
    "current_actual_upb",
    "interest_rate",
    "loan_age",
    "msa",
    "non_interest_bearing_upb",
    "delinquency_12"
  )

  val allCols: List[String] = catCols ++ numericCols

  def transformUnassembled(df: DataFrame): (DataFrame, DataFrame) = {
    val featureDF = df.select(catCols.map(c => (md5(col(c)) % 100).alias(c)) ++ numericCols.map(c => col(c)): _*)
      .withColumn("delinquency_12", when(col("delinquency_12") > 0, 1.0).otherwise(0.0))
      .na.fill(0.0f)
    val Array(dtrain, dtest) = featureDF.randomSplit(Array(0.99, 0.01))
    (dtrain, dtest)
  }

  def transform(df: DataFrame): (DataFrame, DataFrame) = {
    val featureDF = df.select(catCols.map(c => (md5(col(c)) % 100).alias(c)) ++ numericCols.map(c => col(c)): _*)
      .withColumn("delinquency_12", when(col("delinquency_12") > 0, 1.0).otherwise(0.0))
      .na.fill(0.0f)
    val dmatrix = getDataFrameMatrix(featureDF)
    val Array(dtrain, dtest) = dmatrix.randomSplit(Array(0.99, 0.01))
    (dtrain, dtest)
  }

  def getDataFrameMatrix(df: DataFrame): DataFrame = {
    val colsWithoutLabel = allCols.toArray.filter(_ != "delinquency_12")

    val assembler = new VectorAssembler()
      .setInputCols(colsWithoutLabel)
      .setOutputCol("features")

    assembler
      .transform(df)
      .drop(colsWithoutLabel: _*)
  }

  def runXGB(trainDF: DataFrame,
             testDF: DataFrame,
             numRound: Int,
             nWorkers: Int,
             nThreads: Int = 1,
             treeMethod: String = "hist",
             maxDepth: Int = 8,
             growPolicy: String = "depthwise",
             useExternalMemory: Boolean = false): (Double, String, String) = {

    // define parameters
    val paramMap = Map(
      "eta" -> 0.1,
      "gamma" -> 0.1,
      "tree_method" -> treeMethod,
      "max_depth" -> maxDepth,
      "max_leaves" -> 256,
      "grow_policy" -> growPolicy,
      "min_child_weight" -> 30,
      "lambda" -> 1,
      "scale_pos_weight" -> 2,
      "subsample" -> 1,
      "num_workers" -> nWorkers,
      "nthread" -> nThreads,
      "num_round" -> numRound,
      "use_external_memory" -> useExternalMemory,
      "missing" -> 0.0f,
      "verbosity" -> 3
    )

    // train the model
    val trainT0 = System.nanoTime()

    val model = new XGBoostClassifier(paramMap)
      .setFeaturesCol("features")
      .setLabelCol("delinquency_12")
      .fit(trainDF)

    val trainT1 = System.nanoTime()

    // run prediction
    val testT0 = System.nanoTime()
    val predictions = model.setInferBatchSize(10 << 20).transform(testDF)
    val testT1 = System.nanoTime()

    // evaluate
    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("delinquency_12")
      .setRawPredictionCol("prediction")
      .setMetricName("areaUnderROC")

    // cast prediction column to DoubleType (NOTE: must do this for AUC evaluation to work)
    val predictions2 = predictions
      .withColumn("predictionDouble",
        predictions.col("prediction").cast(DoubleType))
      .drop("prediction")
      .withColumnRenamed("predictionDouble", "prediction")

    // calculate auc
    val auc = evaluator.evaluate(predictions2)
    val timeToTrain = trainT1 - trainT0
    val timeToTest = testT1 - testT0

    (auc, timeToTrain + "ns", timeToTest + "ns")
  }
}
