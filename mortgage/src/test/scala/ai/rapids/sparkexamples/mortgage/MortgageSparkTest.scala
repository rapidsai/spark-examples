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

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.scalatest.{FlatSpec, Matchers}

class MortgageSparkTest extends FlatSpec with Matchers {
  it should "extract mortgage data" in {
    val session = SparkSession.builder
      .master("local[6]")
      .appName("UnitTest")
      .getOrCreate()

    session.sparkContext.setLogLevel("warn")

    val df = Run.csv(
      session,
      "mortgage/src/test/resources/Performance_2007Q3.txt_0",
      "mortgage/src/test/resources/Acquisition_2007Q3.txt"
    ).sort(col("loan_id"), col("monthly_reporting_period"))
    assert(df.count() === 10000)

    val (train, eval) = MortgageXgBoost.transform(df)
    println(MortgageXgBoost.runXGB(train, eval, 10, 6))
  }
}
