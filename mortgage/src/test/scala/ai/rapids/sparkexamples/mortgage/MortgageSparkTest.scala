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

    df.count()

    assert(df.count() === 10000)

    val cols = df.schema.fields.map(_.name).sorted
    val sortedDf = df.select(cols.map(n => col(n)):_*)

    val (train, eval) = MortgageXgBoost.transform(df)

    println(MortgageXgBoost.runXGB(train, eval, 10, 6))
  }
}
