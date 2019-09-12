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

import org.apache.spark.sql.types.{FloatType, IntegerType, StructField, StructType}

private[mq2008] trait MQ2008 {

  lazy val paramMap = Map(
    "objective" -> "rank:pairwise",
    "eta" -> 0.1,
    "min_child_weight" -> 0.1,
    "gamma" -> 1.0,
    "max_depth" -> 6,
    "num_round" -> 50,
    "missing" -> 0.0
  )

  // Define column names and schema
  val labelColName = "label"
  val groupColName = "group"
  def featureNames: Seq[String] = (0 until 46).map(i => s"feature_$i")

  def schema: StructType = StructType(
    Seq(
      StructField(labelColName, FloatType),
      StructField(groupColName, IntegerType)
    ) ++ featureNames.map(name =>
      StructField(name, FloatType)
    )
  )
}