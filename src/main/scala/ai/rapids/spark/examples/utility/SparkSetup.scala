package ai.rapids.spark.examples.utility

import org.apache.spark.sql.SparkSession

object SparkSetup {
  def apply(args: Array[String], appName: String) = {
    val builder = SparkSession.builder()
    val masterBuilder = Option(System.getenv("SPARK_MASTER")).map{master =>
      builder.master(master)
    }.getOrElse(builder)

    masterBuilder.appName(appName).getOrCreate()
  }

  def apply(args: Array[String]): SparkSession = SparkSetup(args, "default")

}
