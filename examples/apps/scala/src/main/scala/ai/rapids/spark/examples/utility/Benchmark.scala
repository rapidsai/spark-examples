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
package ai.rapids.spark.examples.utility

import ml.dmlc.xgboost4j.java.EnvironmentDetector
import sys.process._

object Benchmark {
  def time[R](phase: String)(block: => R): (R, Float) = {

    val env = getRunEnvironment()
    val t0 = System.currentTimeMillis
    val result = block // call-by-name
    val t1 = System.currentTimeMillis
    println("\n--------------")
    println("==> Benchmark: Elapsed time for [" + phase + " " + env + "]: " + ((t1 - t0).toFloat / 1000) + "s")
    println("--------------\n")
    (result, (t1 - t0).toFloat / 1000)
  }

  def getRunEnvironment(): String = {
    val cudaVersion = EnvironmentDetector.getCudaVersion().orElse("9.2.0")
    val cuda = if (cudaVersion.startsWith("9.")) {
      "cuda9"
    } else {
      "cuda10"
    }
    try {
      val os = "apt -v".!
      "benchmark " + cuda + " ubuntu16"
    } catch {
      case x : java.io.IOException => {
        "benchmark " + cuda + " centos7"
      }
    }
  }
}
