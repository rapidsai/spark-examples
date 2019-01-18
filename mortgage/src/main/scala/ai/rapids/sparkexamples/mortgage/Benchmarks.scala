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

case class ETLArgs(perfPath: String, acqPath: String, output: String)

case class BenchmarkArgs(input: String, bench: String, workers: Int, samples: Int, rounds: Int, threads: Int,
                         predictor: String, treeMethod: String, maxDepth: Int, growPolicy: String)

object Benchmark {
  def etlArgs(input: Array[String]): ETLArgs =
    ETLArgs(input(0), input(1), input(2))

  def args(input: Array[String]): BenchmarkArgs =
    BenchmarkArgs(input(0), input(1), input(2).toInt, input(3).toInt, input(4).toInt, input(5).toInt, input(6),
      input(7), input(8).toInt, input(9))

  def session: SparkSession = {
    val builder = SparkSession.builder.appName("MortgageJob")

    val master = System.getenv("SPARK_MASTER")
    if (master != null) {
      builder.master(master)
    }

    val spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("warn")

    spark.sqlContext.clearCache()

    spark
  }

  def time[R](block: => R): (R, Float) = {
    val t0 = System.currentTimeMillis
    val result = block // call-by-name
    val t1 = System.currentTimeMillis
    println("Elapsed time: " + ((t1 - t0).toFloat / 1000) + "s")
    (result, (t1 - t0).toFloat / 1000)
  }

}

object ETL {
  def main(args: Array[String]): Unit = {
    val jobArgs = Benchmark.etlArgs(args)

    val session = Benchmark.session

    val dfPerf = CreatePerformanceDelinquency.prepare(ReadPerformanceCsv(session, jobArgs.perfPath))
    val dfAcq = ReadAcquisitionCsv(session, jobArgs.acqPath)
    val df = CleanAcquisitionPrime(session, dfPerf, dfAcq)
    val (train, eval) = MortgageXgBoost.transform(df)
    train
      .write
      .mode("overwrite")
      .parquet(jobArgs.output + "/train")
    eval
      .write
      .mode("overwrite")
      .parquet(jobArgs.output + "/eval")
  }
}

object MLBenchmark {
  def main(args: Array[String]): Unit = {
    val jobArgs = Benchmark.args(args)

    val session = Benchmark.session
    import session.implicits._

    val dftrain = session.sqlContext.read.parquet(jobArgs.input + "/train").repartition(jobArgs.workers).cache()
    val dfEval = session.sqlContext.read.parquet(jobArgs.input + "/eval").cache()

    dftrain.count()
    dfEval.count()

    val timings = 0.until(jobArgs.samples).map { _ =>
      val ((acc, tTrain, tTest), time) = Benchmark.time {
        MortgageXgBoost.runXGB(dftrain, dfEval, jobArgs.rounds, jobArgs.workers, jobArgs.threads,
          jobArgs.predictor, jobArgs.treeMethod, jobArgs.maxDepth, jobArgs.growPolicy)
      }
      s"$acc, $tTrain, $tTest, $time"
    }

    timings.toDF("time").coalesce(1).write.mode("overwrite").text(jobArgs.bench + "/ml.csv")

    dftrain.unpersist(true)
    dfEval.unpersist(true)
  }
}
