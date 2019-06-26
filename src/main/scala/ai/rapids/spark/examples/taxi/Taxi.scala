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
package ai.rapids.spark.examples.taxi

import org.apache.spark.sql.types.{FloatType, IntegerType, StructField, StructType}

private[taxi] trait Taxi {
  val labelColName = "fare_amount"

  lazy val commParamMap = Map(
    "learning_rate" -> 0.05,
    "max_depth" -> 8,
    "subsample" -> 0.8,
    "gamma" -> 1
  )

  lazy val schema =
    StructType(Array(
      StructField("vendor_id", FloatType),
      StructField("passenger_count", FloatType),
      StructField("trip_distance", FloatType),
      StructField("pickup_longitude", FloatType),
      StructField("pickup_latitude", FloatType),
      StructField("rate_code", FloatType),
      StructField("store_and_fwd", FloatType),
      StructField("dropoff_longitude", FloatType),
      StructField("dropoff_latitude", FloatType),
      StructField(labelColName, FloatType),
      StructField("hour", FloatType),
      StructField("year", IntegerType),
      StructField("month", IntegerType),
      StructField("day", FloatType),
      StructField("day_of_week", FloatType),
      StructField("is_weekend", FloatType)
    ))
}
