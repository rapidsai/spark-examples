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
import org.apache.spark.storage.StorageLevel

case class ETLArgs(perfPath: String, acqPath: String, output: String)

case class BenchmarkArgs(input: String, bench: String, workers: Int, samples: Int, rounds: Int, threads: Int,
                         treeMethod: String, maxDepth: Int, growPolicy: String, useExternalMemory: Boolean)

case class FullBenchmarkArgs(
  perfPath: String,
  acqPath: String,
  output: String,
  bench: String,
  workers: Int,
  samples: Int,
  rounds: Int,
  threads: Int,
  treeMethod: String,
  maxDepth: Int,
  growPolicy: String,
  useExternalMemory: Boolean
)

object Benchmark {
  def etlArgs(input: Array[String]): ETLArgs =
    ETLArgs(input(0), input(1), input(2))

  def args(input: Array[String]): BenchmarkArgs =
    BenchmarkArgs(input(0), input(1), input(2).toInt, input(3).toInt, input(4).toInt, input(5).toInt, input(6),
      input(7).toInt, input(8), input(9).toBoolean)

  def argsFull(input: Array[String]): FullBenchmarkArgs =
    FullBenchmarkArgs(input(0), input(1), input(2), input(3), input(2).toInt, input(3).toInt, input(4).toInt,
      input(5).toInt, input(6), input(7).toInt, input(8), input(9).toBoolean)

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
    import session.implicits._
    val storageLevel = StorageLevel.MEMORY_ONLY

    // Load CSV
    val ((pCsv, aCsv), csvTime) = Benchmark.time {
      val perfCsv = ReadPerformanceCsv(session, jobArgs.perfPath).persist(storageLevel)
      val acqCsv = ReadAcquisitionCsv(session, jobArgs.acqPath).persist(storageLevel)

      // Force Execution
      perfCsv.count()
      acqCsv.count()

      (perfCsv, acqCsv)
    }


    // Clean/Transform
    val (cleanDF, transformTime) = Benchmark.time {
      val perf = CreatePerformanceDelinquency.prepare(pCsv)
      val cdf = CleanAcquisitionPrime(session, perf, aCsv).persist(storageLevel)
      cdf.count()
      cdf
    }

    pCsv.unpersist(true)
    aCsv.unpersist(true)

    val ((dfTrain, dfEval), vectorTime) = Benchmark.time {
      val (t, e) = MortgageXgBoost.transform(cleanDF)
      t.persist(storageLevel)
      e.persist(storageLevel)
      t.count()
      (t, e)
    }
    dfEval.count()

    val (_, saveTime) = Benchmark.time{
      dfTrain
        .write
        .mode("overwrite")
        .parquet(jobArgs.output + "/train")

      dfEval
        .write
        .mode("overwrite")
        .parquet(jobArgs.output + "/eval")
    }

    cleanDF.unpersist(true)
    dfTrain.unpersist(true)
    dfEval.unpersist(true)

    val timings = List(
      s"vectorization_time, $vectorTime",
      s"transform_time, $transformTime",
      s"load_time, $csvTime",
      s"save_time, $saveTime"
    )

    timings
      .toDF("time")
      .coalesce(1)
      .write
      .mode("overwrite")
      .text(jobArgs.output + "/timing/etl.csv")
  }
}

object ConvertToLibSVM {
  def main(args: Array[String]): Unit = {
    val jobArgs = Benchmark.etlArgs(args)
    val session = Benchmark.session

    val df = Run.csv(session, jobArgs.perfPath, jobArgs.acqPath)
    val (dfTrain, dfEval) = MortgageXgBoost.transform(df)

    dfTrain
      .write
      .mode("overwrite")
      .format("libsvm")
      .save(jobArgs.output + "/train")

    dfEval
      .write
      .mode("overwrite")
      .format("libsvm")
      .save(jobArgs.output + "/eval")
  }
}

object MLBenchmark {
  def main(args: Array[String]): Unit = {
    val jobArgs = Benchmark.args(args)

    val session = Benchmark.session
    import session.implicits._

    val dftrain = session.sqlContext.read.parquet(jobArgs.input + "/train").repartition(jobArgs.workers).cache()
    val dfEval = session.sqlContext.read.parquet(jobArgs.input + "/eval").repartition(jobArgs.workers).cache()

    dftrain.count()
    dfEval.count()

    val timings = 0.until(jobArgs.samples).map { _ =>
      val ((acc, tTrain, tTest), time) = Benchmark.time {
        MortgageXgBoost.runXGB(dftrain, dfEval, jobArgs.rounds, jobArgs.workers, jobArgs.threads,
          jobArgs.treeMethod, jobArgs.maxDepth, jobArgs.growPolicy, jobArgs.useExternalMemory)
      }
      s"$acc, $tTrain, $tTest, $time"
    }

    timings.toDF("time").coalesce(1).write.mode("overwrite").text(jobArgs.bench + "/ml.csv")

    dftrain.unpersist(true)
    dfEval.unpersist(true)
  }
}

object FullBenchmark {
  def main(args: Array[String]): Unit = {
    val jobArgs = Benchmark.argsFull(args)

    val session = Benchmark.session
    import session.implicits._

    val storageLevel = StorageLevel.MEMORY_ONLY

    // Load CSV
    val ((pCsv, aCsv), csvTime) = Benchmark.time {
      val perfCsv = ReadPerformanceCsv(session, jobArgs.perfPath).persist(storageLevel)
      val acqCsv = ReadAcquisitionCsv(session, jobArgs.acqPath).persist(storageLevel)

      // Force Execution
      perfCsv.count()
      acqCsv.count()

      (perfCsv, acqCsv)
    }


    // Clean/Transform
    val (cleanDF, transformTime) = Benchmark.time {
      val perf = CreatePerformanceDelinquency.prepare(pCsv)
      val cdf = CleanAcquisitionPrime(session, perf, aCsv).persist(storageLevel)
      cdf.count()
      cdf
    }

    pCsv.unpersist(true)
    aCsv.unpersist(true)

    val ((dfTrain, dfEval), vectorTime) = Benchmark.time {
      val (t, e) = MortgageXgBoost.transform(cleanDF)
      t.persist(storageLevel)
      e.persist(storageLevel)
      t.count()
      (t, e)
    }
    dfEval.count()

    cleanDF.unpersist(true)

    val ((acc, tTrain, tTest), time) = Benchmark.time {
      MortgageXgBoost.runXGB(
        dfTrain,
        dfEval,
        jobArgs.rounds,
        jobArgs.workers,
        jobArgs.threads,
        jobArgs.treeMethod,
        jobArgs.maxDepth,
        jobArgs.growPolicy,
        jobArgs.useExternalMemory
      )
    }

    dfTrain.unpersist(true)
    dfEval.unpersist(true)

    val timings = List(
      s"accuracy, $acc",
      s"train_time, $tTrain",
      s"test_time, $tTest",
      s"xgboost_time, $time",
      s"vectorization_time, $vectorTime",
      s"transform_time, $transformTime",
      s"load_time, $csvTime"
    )

    timings
      .toDF("time")
      .coalesce(1)
      .write
      .mode("overwrite")
      .text(jobArgs.bench + "/full.csv")
  }
}
